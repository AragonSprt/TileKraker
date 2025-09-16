import os
import time
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2
import mss
import chess
from stockfish import Stockfish

# ---------- User config ----------
STOCKFISH_PATH = r"D:\Cyril\OneDrive\Documents\Development\Python\TileKraker\bin\stockfish\stockfish-windows-x86-64-avx2.exe"
TEMPLATES_DIR = "templates"
SCAN_INTERVAL = 1.0  # seconds between scans
STOCKFISH_DEPTH = 12
MATCH_THRESHOLD = 0.60  # template matching threshold (0 to 1)
WARP_SIZE = 480  # size in px for warped board (square)
# ---------------------------------

# Init engine
stockfish = Stockfish(path=STOCKFISH_PATH, depth=STOCKFISH_DEPTH)

def load_templates(templates_dir):
    # Load piece templates as grayscale numpy arrays
    req = ["wP","wN","wB","wR","wQ","wK","bP","bN","bB","bR","bQ","bK"]
    pieces = {}
    for name in req:
        p = os.path.join(templates_dir, f"{name}.png")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing template: {p}")
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read template {p}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pieces[name] = gray
    return pieces

# ---------- Board auto-detection ----------
def order_points_clockwise(pts):
    # Order the four corners (Top Left, Top Right, Bottom Left, Bottom Right)
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect

def find_board_quad(img_bgr):
    # Find largest 4-point contour approximating the board, returns 4 points or None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_quad = None

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                # check aspect ratio ~ square
                x,y,w,h = cv2.boundingRect(approx)
                ar = float(w)/float(h) if h>0 else 0
                if 0.6 < ar < 1.6 and area > 10000:  # tune area threshold if needed
                    max_area = area
                    best_quad = approx.reshape(4,2)
    if best_quad is None:
        return None
    return order_points_clockwise(best_quad)

def warp_board(img_bgr, quad, size=WARP_SIZE):
    # Transform squares to even size
    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(quad.astype("float32"), dst)
    warped = cv2.warpPerspective(img_bgr, M, (size, size))
    return warped

# ---------- Piece detection ----------
def detect_pieces_on_board_img(img_bgr, templates, match_threshold=MATCH_THRESHOLD, assume_white_bottom=True):
    # Given a square board image, detect pieces via template matching.
    # Returns a 8x8 grid of piece chars (or None) where grid[rank][file], rank 0 = top row (board's top)

    h, w = img_bgr.shape[:2]
    sq_h = h // 8
    sq_w = w // 8
    if sq_h == 0 or sq_w == 0:
        return None

    # Pre-resize templates to square size
    templates_resized = {}
    for name, tpl in templates.items():
        tpl_r = cv2.resize(tpl, (sq_w, sq_h), interpolation=cv2.INTER_AREA)
        templates_resized[name] = tpl_r

    grid = [[None for _ in range(8)] for _ in range(8)]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    for r in range(8):
        for f in range(8):
            y1, x1 = r*sq_h, f*sq_w
            crop = img_gray[y1:y1+sq_h, x1:x1+sq_w]
            best_score = -1.0
            best_name = None
            for name, tpl in templates_resized.items():
                # matchTemplate yields single value for same-size template
                res = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED)
                score = float(res[0][0])
                if score > best_score:
                    best_score = score
                    best_name = name
            if best_score >= match_threshold:
                # store piece symbol; best_name like 'wP'
                piece_symbol = best_name[1]  # 'P','N'
                color = best_name[0]  # 'w' or 'b'
                char = piece_symbol.upper() if color == 'w' else piece_symbol.lower()
                grid[r][f] = char
            else:
                grid[r][f] = None
    return grid

def fen_from_grid(grid, side_to_move='w'):
    # Build FEN string from grid where grid[0] is top rank (board top = rank 8)
    ranks = []
    for r in range(8):
        empty_count = 0
        fen_rank = ""
        for f in range(8):
            c = grid[r][f]
            if c is None:
                empty_count += 1
            else:
                if empty_count:
                    fen_rank += str(empty_count)
                    empty_count = 0
                fen_rank += c
        if empty_count:
            fen_rank += str(empty_count)
        ranks.append(fen_rank)
    # ranks is top->bottom FEN wants rank8/7/... so that's correct.
    fen_pos = "/".join(ranks)
    # minimal extra fields: side to move, castling -, enpassant -, halfmove 0, fullmove 1
    fen = f"{fen_pos} {side_to_move} - - 0 1"
    return fen

# ---------- Gradient buttons ----------
def make_gradient_image(w, h, from_color, to_color, radius=8):
    # Return a PIL image with vertical gradient and rounded corners for buttons
    base = Image.new("RGBA", (w, h))
    draw = ImageDraw.Draw(base)
    # vertical gradient
    for i in range(h):
        t = i / (h-1)
        r = int(from_color[0] * (1-t) + to_color[0] * t)
        g = int(from_color[1] * (1-t) + to_color[1] * t)
        b = int(from_color[2] * (1-t) + to_color[2] * t)
        draw.line([(0,i),(w,i)], fill=(r,g,b,255))
    # rounded corners mask
    mask = Image.new("L", (w,h), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([(0,0),(w-1,h-1)], radius=radius, fill=255)
    base.putalpha(mask)
    return base

# ---------- Main App ----------
class AutoLiveCoachApp:
    def __init__(self, root):
        self.root = root
        root.title("TileKraker - Your Live Chess Coach")
        self.templates = None
        try:
            self.templates = load_templates(TEMPLATES_DIR)
        except Exception as e:
            messagebox.showwarning("Templates", f"Could not load templates from {TEMPLATES_DIR}:\n{e}")

        self.scan_interval = SCAN_INTERVAL
        self.assume_white_bottom = True
        self.scanning = False
        self._stop_event = threading.Event()
        self._thread = None
        self.last_img = None
        self.last_best = None
        self.button_images = {}

        # Top controls (with gradient buttons)
        top = tk.Frame(root)
        top.pack(fill="x", pady=6, padx=6)

        self.btn_start = self._make_gradient_button(top, "Start Scanning", (255,180,0), (255,110,0), command=self.toggle_scanning)
        self.btn_start.pack(side="left", padx=6)

        self.btn_reload = self._make_gradient_button(top, "Reload Templates", (100,200,255), (10,130,255), command=self.reload_templates)
        self.btn_reload.pack(side="left", padx=6)

        self.btn_one_shot = self._make_gradient_button(top, "One-shot Detect", (180,255,180), (60,200,60), command=self.one_shot_detect)
        self.btn_one_shot.pack(side="left", padx=6)

        self.flip_var = tk.IntVar(value=1)
        cb = tk.Checkbutton(top, text="White at bottom", variable=self.flip_var, command=self.toggle_orientation)
        cb.pack(side="left", padx=8)

        tk.Label(top, text="Interval (s):").pack(side="left", padx=(10,2))
        self.interval_entry = tk.Entry(top, width=4)
        self.interval_entry.insert(0, str(SCAN_INTERVAL))
        self.interval_entry.pack(side="left")

        # Canvas to show the board capture
        self.canvas = tk.Canvas(root, width=WARP_SIZE, height=WARP_SIZE, bg="black")
        self.canvas.pack(padx=8, pady=8)
        self.canvas_img_id = None
        self.tkimg = None

        # Status bar
        status = tk.Frame(root)
        status.pack(fill="x", padx=6, pady=(0,6))
        self.status_label = tk.Label(status, text="Idle.")
        self.status_label.pack(side="left", fill="x", expand=True)
        self.best_label = tk.Label(status, text="Best: -", font=("Arial", 14, "bold"))
        self.best_label.pack(side="right")

        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _make_gradient_button(self, parent, text, c1, c2, command=None):
        img = make_gradient_image(120, 34, c1, c2)
        draw = ImageDraw.Draw(img)
        # text centered
        w, h = img.size
        draw.text((w//2, h//2), text, anchor="mm", fill=(255,255,255,255))
        tkimg = ImageTk.PhotoImage(img)
        self.button_images[text] = tkimg  # keep reference
        btn = tk.Button(parent, image=tkimg, borderwidth=0, relief="raised", command=command)
        return btn

    def toggle_orientation(self):
        self.assume_white_bottom = bool(self.flip_var.get())

    def reload_templates(self):
        try:
            self.templates = load_templates(TEMPLATES_DIR)
            messagebox.showinfo("Templates", "Reloaded templates successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reload templates:\n{e}")

    def one_shot_detect(self):
        # capture screen once and do full pipeline
        sct = mss.mss()
        monitor = sct.monitors[0]
        raw = sct.grab(monitor)
        screen = Image.frombytes("RGB", raw.size, raw.rgb)
        img_bgr = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
        quad = find_board_quad(img_bgr)
        if quad is None:
            self.status_label.config(text="Board quad not found.")
            return
        warped = warp_board(img_bgr, quad, size=WARP_SIZE)
        grid = detect_pieces_on_board_img(warped, self.templates, match_threshold=MATCH_THRESHOLD, assume_white_bottom=self.assume_white_bottom)
        if grid is None:
            self.status_label.config(text="Piece detection failed.")
            return
        fen = fen_from_grid(grid, side_to_move='w')
        try:
            stockfish.set_fen_position(fen)
            best = stockfish.get_best_move()
        except Exception as e:
            best = None
            print("Stockfish error:", e)
        self.last_img = warped
        self.last_best = best
        self.update_canvas_and_labels()

    def toggle_scanning(self):
        if self.scanning:
            self.stop_scanning()
        else:
            self.start_scanning()

    def start_scanning(self):
        try:
            interval = float(self.interval_entry.get())
        except:
            interval = SCAN_INTERVAL
        self.scan_interval = max(0.2, interval)
        if self.templates is None:
            messagebox.showerror("Templates missing", f"Put templates in {TEMPLATES_DIR} and press Reload Templates.")
            return
        self.scanning = True
        self.btn_start.config(relief="sunken")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.scan_loop, daemon=True)
        self._thread.start()
        self.status_label.config(text="Scanning...")

    def stop_scanning(self):
        self.scanning = False
        self.btn_start.config(relief="raised")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self.status_label.config(text="Stopped.")

    def scan_loop(self):
        sct = mss.mss()
        while not self._stop_event.is_set():
            monitor = sct.monitors[0]
            raw = sct.grab(monitor)
            screen = Image.frombytes("RGB", raw.size, raw.rgb)
            img_bgr = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
            quad = find_board_quad(img_bgr)
            if quad is None:
                self.root.after(0, lambda: self.status_label.config(text="Board not found — move/resize window."))
                time.sleep(self.scan_interval)
                continue
            warped = warp_board(img_bgr, quad, size=WARP_SIZE)
            grid = detect_pieces_on_board_img(warped, self.templates, match_threshold=MATCH_THRESHOLD, assume_white_bottom=self.assume_white_bottom)
            if grid is None:
                self.root.after(0, lambda: self.status_label.config(text="Piece detection failed."))
                time.sleep(self.scan_interval)
                continue
            fen = fen_from_grid(grid, side_to_move='w')
            try:
                stockfish.set_fen_position(fen)
                best = stockfish.get_best_move()
            except Exception as e:
                best = None
                print("Stockfish error", e)
            self.last_img = warped
            self.last_best = best
            self.root.after(0, self.update_canvas_and_labels)
            time.sleep(self.scan_interval)

    def update_canvas_and_labels(self):
        if self.last_img is None:
            return
        img = self.last_img.copy()
        # overlay highlight for best move
        if self.last_best:
            try:
                bf, tf = self.last_best[:2], self.last_best[2:4]
                # convert algebraic 'e2' to row/col on warped image
                from_sq = chess.parse_square(self.last_best[:2])
                to_sq = chess.parse_square(self.last_best[2:4])
                def sq_to_rowcol(sq):
                    f = chess.square_file(sq)
                    r = chess.square_rank(sq)
                    if self.assume_white_bottom:
                        row = 7 - r
                        col = f
                    else:
                        row = r
                        col = 7 - f
                    return row, col
                fr, fc = sq_to_rowcol(from_sq)
                tr, tc = sq_to_rowcol(to_sq)
                h, w = img.shape[:2]
                sqh, sqw = h//8, w//8
                # highlight from (semi-transparent)
                overlay = img.copy()
                cv2.rectangle(overlay, (fc*sqw, fr*sqh), ((fc+1)*sqw, (fr+1)*sqh), (0,255,255), -1)
                cv2.rectangle(overlay, (tc*sqw, tr*sqh), ((tc+1)*sqw, (tr+1)*sqh), (0,128,255), -1)
                img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
            except Exception as e:
                print("Highlight exception:", e)
        # convert to Tk image and display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        # resize to canvas while keeping aspect ratio
        cw = self.canvas.winfo_width() or WARP_SIZE
        ch = self.canvas.winfo_height() or WARP_SIZE
        pil = pil.resize((cw, ch), Image.ANTIALIAS)
        self.tkimg = ImageTk.PhotoImage(pil)
        if self.canvas_img_id is None:
            self.canvas_img_id = self.canvas.create_image(0,0,anchor="nw", image=self.tkimg)
        else:
            self.canvas.itemconfig(self.canvas_img_id, image=self.tkimg)

        # update labels
        if self.last_best:
            self.best_label.config(text=f"Best: {self.last_best}")
            self.status_label.config(text="Board detected and analyzed.")
        else:
            self.best_label.config(text="Best: -")
            self.status_label.config(text="Analyzed but no best move (engine error?)")

    def on_close(self):
        self.stop_scanning()
        self.root.destroy()

# ---------- Run ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = AutoLiveCoachApp(root)
    root.mainloop()
