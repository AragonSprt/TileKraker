"""
Created by Aragon on 07/16/25.

Template capture helper for the Live Chess Coach.
- Auto-detects board on screen, warps it, and shows it.
- Click on the center of each piece to capture templates in order:
  wP,wN,wB,wR,wQ,wK,bP,bN,bB,bR,bQ,bK

Outputs saved as ./templates/<name>.png
"""

import os
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import mss
import numpy as np
import cv2

# ---------- Config ----------
OUTPUT_DIR = "templates"
WARP_SIZE = 480   # warped board size (square)
DEFAULT_TEMPLATE_SAVE_SIZE = 64  # saved PNG size (64x64)
TEMPLATE_ORDER = ["wP","wN","wB","wR","wQ","wK","bP","bN","bB","bR","bQ","bK"]
BUTTON_FONT = None  # default; Pillow will fallback to system font
# ----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- board detection helpers ----------
def order_points_clockwise(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_board_quad(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            x,y,w,h = cv2.boundingRect(approx)
            ar = float(w)/h if h>0 else 0
            if area > max_area and area > 10000 and 0.6 < ar < 1.6:
                max_area = area
                best = approx.reshape(4,2)
    if best is None:
        return None
    return order_points_clockwise(best)

def warp_board(img_bgr, quad, size=WARP_SIZE):
    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(quad.astype("float32"), dst)
    warped = cv2.warpPerspective(img_bgr, M, (size, size))
    return warped

# ---------- Gradient rounded-button generator ----------
def make_rounded_gradient(w, h, color1, color2, radius=12, shadow=True, text=None, font_size=14, text_color=(255,255,255)):
    """
    Returns a PIL Image with rounded gradient + optional shadow and centered text.
    color1, color2 = (r,g,b)
    """
    base = Image.new("RGBA", (w, h), (0,0,0,0))

    # optional shadow layer
    if shadow:
        shadow_img = Image.new("RGBA", (w, h), (0,0,0,0))
        sd = ImageDraw.Draw(shadow_img)
        offset = (2, 3)
        rect = (offset[0], offset[1], w - 1, h - 1)
        sd.rounded_rectangle(rect, radius=radius, fill=(0,0,0,80))
        base = Image.alpha_composite(shadow_img, base)

    # gradient rectangle
    grad = Image.new("RGBA", (w, h), (0,0,0,0))
    gdraw = ImageDraw.Draw(grad)
    for i in range(h):
        t = i / (h - 1) if h > 1 else 0
        r = int(color1[0] * (1 - t) + color2[0] * t)
        g = int(color1[1] * (1 - t) + color2[1] * t)
        b = int(color1[2] * (1 - t) + color2[2] * t)
        gdraw.line([(0, i), (w, i)], fill=(r, g, b, 255))

    # rounded mask
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([(0,0),(w-1,h-1)], radius=radius, fill=255)
    grad.putalpha(mask)
    base = Image.alpha_composite(base, grad)

    # subtle top highlight for glossy look
    highlight = Image.new("RGBA", (w, h), (255,255,255,0))
    hd = ImageDraw.Draw(highlight)
    hd.rounded_rectangle([(2,2),(w-3,h//2)], radius=max(2, radius//2), fill=(255,255,255,30))
    base = Image.alpha_composite(base, highlight)

    # draw text (robust sizing across Pillow versions)
    if text:
        td = ImageDraw.Draw(base)
        try:
            # try to load truetype if provided, otherwise default
            font = ImageFont.truetype(BUTTON_FONT, font_size) if BUTTON_FONT else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Compute text size in a way that works across Pillow versions
        try:
            # preferred: ImageFont.getsize (works for most font objects)
            tw, th = font.getsize(text)
        except Exception:
            try:
                # fallback: ImageDraw.textbbox (newer Pillow)
                bbox = td.textbbox((0,0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                # last resort: try textsize (older). wrap in try to avoid AttributeError.
                try:
                    tw, th = td.textsize(text, font=font)
                except Exception:
                    # give up and use approximate values
                    tw, th = len(text) * (font_size // 2), font_size

        # center the text
        x = (w - tw) // 2
        y = (h - th) // 2
        td.text((x, y), text, font=font, fill=text_color + (255,))

    return base

# ---------- Capture app (UI uses fancy buttons) ----------
class TemplateCaptureApp:
    def __init__(self, root):
        self.root = root
        root.title("TileKraker — Capture Templates")
        self.warp_size = WARP_SIZE
        self.template_save_size = DEFAULT_TEMPLATE_SAVE_SIZE
        self.templates_captured = {}
        self.current_index = 0

        # keep references to generated images so Tk doesn't GC them
        self._button_images = {}

        # Top controls area
        top = tk.Frame(root)
        top.pack(fill="x", padx=10, pady=10)

        self.btn_screenshot = self._create_pretty_button(top, "Grab Screen", (90,200,255), (10,120,255), command=self.grab_screen)
        self.btn_screenshot.pack(side="left", padx=6)

        self.btn_load = self._create_pretty_button(top, "Load Image", (200,240,180), (90,180,60), command=self.load_image)
        self.btn_load.pack(side="left", padx=6)

        self.btn_one_shot = self._create_pretty_button(top, "Finish & Save", (255,170,100), (255,90,30), command=self.finish)
        self.btn_one_shot.pack(side="right", padx=6)

        # crop size entry
        mid = tk.Frame(root)
        mid.pack(fill="x", padx=10)
        tk.Label(mid, text="Crop size (px):").pack(side="left")
        self.size_entry = tk.Entry(mid, width=5)
        self.size_entry.insert(0, str(int(self.warp_size//8 * 0.9)))
        self.size_entry.pack(side="left", padx=(6,0))

        # info
        self.info_label = tk.Label(root, text="Press 'Grab Screen' or 'Load Image' to begin.", anchor="w")
        self.info_label.pack(fill="x", padx=10, pady=(6,0))

        # canvas for warped board
        self.canvas = tk.Canvas(root, width=self.warp_size, height=self.warp_size, bg="#222")
        self.canvas.pack(padx=10, pady=10)
        self.canvas_image_id = None
        self.tkimg = None
        self.display_scale = 1.0

        # capture controls and thumbnails
        ctrl = tk.Frame(root)
        ctrl.pack(fill="x", padx=10, pady=(0,10))
        self.target_label = tk.Label(ctrl, text="Next: " + TEMPLATE_ORDER[self.current_index])
        self.target_label.pack(side="left")
        self.btn_skip = tk.Button(ctrl, text="Skip", command=self.skip)
        self.btn_skip.pack(side="left", padx=6)
        self.btn_undo = tk.Button(ctrl, text="Undo", command=self.undo)
        self.btn_undo.pack(side="left", padx=6)

        self.thumb_frame = tk.Frame(root)
        self.thumb_frame.pack(fill="x", padx=10, pady=(0,10))

        # internals
        self.warped_img_bgr = None
        self.canvas.bind("<Button-1>", self.on_click)

    def _create_pretty_button(self, parent, text, c1, c2, width=120, height=36, command=None):
        # normal, hover, pressed images
        normal = make_rounded_gradient(width, height, c1, c2, radius=12, shadow=True, text=text, font_size=12)
        hover = make_rounded_gradient(width, height, tuple(min(255,int(c1[i]*0.9)) for i in range(3)), tuple(min(255,int(c2[i]*0.9)) for i in range(3)), radius=12, shadow=True, text=text, font_size=12)
        pressed = make_rounded_gradient(width, height, tuple(max(0,int(c1[i]*1.1)) if c1[i]*1.1<=255 else 255 for i in range(3)), tuple(max(0,int(c2[i]*1.1)) if c2[i]*1.1<=255 else 255 for i in range(3)), radius=12, shadow=True, text=text, font_size=12)
        tk_n = ImageTk.PhotoImage(normal)
        tk_h = ImageTk.PhotoImage(hover)
        tk_p = ImageTk.PhotoImage(pressed)
        # keep references
        self._button_images[text + "_n"] = tk_n
        self._button_images[text + "_h"] = tk_h
        self._button_images[text + "_p"] = tk_p

        btn = tk.Label(parent, image=tk_n, bd=0, cursor="hand2")
        if command:
            btn.bind("<Button-1>", lambda e: (btn.config(image=tk_p), self.root.after(130, lambda: (command(), btn.config(image=tk_n)))))
        btn.bind("<Enter>", lambda e: btn.config(image=tk_h))
        btn.bind("<Leave>", lambda e: btn.config(image=tk_n))
        return btn

    def grab_screen(self):
        self.info_label.config(text="Grabbing full screen and attempting to detect the board...")
        sct = mss.mss()
        monitor = sct.monitors[0]
        raw = sct.grab(monitor)
        screen = Image.frombytes("RGB", raw.size, raw.rgb)
        img_bgr = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
        quad = find_board_quad(img_bgr)
        if quad is None:
            self.info_label.config(text="Board not found automatically. Try Load Image or ensure board is visible.")
            messagebox.showwarning("Board not found", "Could not detect board automatically. You can load a screenshot manually.")
            return
        warped = warp_board(img_bgr, quad, size=self.warp_size)
        self.set_warped_image(warped)
        self.info_label.config(text="Board detected. Click centers of pieces to capture templates in order shown.")
        sq = self.warp_size // 8
        default_crop = int(sq * 0.9)
        self.size_entry.delete(0, tk.END)
        self.size_entry.insert(0, str(default_crop))

    def load_image(self):
        path = filedialog.askopenfilename(title="Load board screenshot/image", filetypes=[("Image files","*.png;*.jpg;*.jpeg;*.bmp"), ("All files","*.*")])
        if not path:
            return
        pil = Image.open(path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        quad = find_board_quad(img_bgr)
        if quad is not None:
            warped = warp_board(img_bgr, quad, size=self.warp_size)
            self.set_warped_image(warped)
            self.info_label.config(text="Board auto-detected from image. Click to capture templates.")
        else:
            h,w = img_bgr.shape[:2]
            side = min(h,w)
            cy, cx = h//2, w//2
            half = side//2
            crop = img_bgr[cy-half:cy-half+side, cx-half:cx-half+side]
            warped = cv2.resize(crop, (self.warp_size, self.warp_size))
            self.set_warped_image(warped)
            self.info_label.config(text="Loaded image (no quad). Click to capture templates.")
        sq = self.warp_size // 8
        default_crop = int(sq * 0.9)
        self.size_entry.delete(0, tk.END)
        self.size_entry.insert(0, str(default_crop))

    def set_warped_image(self, warped_bgr):
        self.warped_img_bgr = warped_bgr.copy()
        img_rgb = cv2.cvtColor(self.warped_img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        cw = self.canvas.winfo_width() or self.warp_size
        ch = self.canvas.winfo_height() or self.warp_size
        scale = min(cw/self.warp_size, ch/self.warp_size)
        self.display_scale = scale
        display_size = int(self.warp_size * scale)
        pil_resized = pil.resize((display_size, display_size), Image.ANTIALIAS)
        self.tkimg = ImageTk.PhotoImage(pil_resized)
        self.canvas.delete("all")
        self.canvas_image_id = self.canvas.create_image((cw-display_size)//2, (ch-display_size)//2, anchor="nw", image=self.tkimg)
        offset_x = (cw-display_size)//2
        offset_y = (ch-display_size)//2
        ds = display_size
        for i in range(1,8):
            x = offset_x + int(i * display_size / 8)
            self.canvas.create_line(x, offset_y, x, offset_y+ds, fill="#ffffff80")
            y = offset_y + int(i * display_size / 8)
            self.canvas.create_line(offset_x, y, offset_x+ds, y, fill="#ffffff80")

    def on_click(self, event):
        if self.warped_img_bgr is None:
            return
        if self.current_index >= len(TEMPLATE_ORDER):
            messagebox.showinfo("Done", "All templates captured. Press Finish & Save or Undo.")
            return
        cw = self.canvas.winfo_width() or self.warp_size
        ch = self.canvas.winfo_height() or self.warp_size
        display_size = int(self.warp_size * self.display_scale)
        offset_x = (cw-display_size)//2
        offset_y = (ch-display_size)//2
        x_disp = event.x - offset_x
        y_disp = event.y - offset_y
        if x_disp < 0 or y_disp < 0 or x_disp >= display_size or y_disp >= display_size:
            return
        x_orig = int(x_disp / self.display_scale)
        y_orig = int(y_disp / self.display_scale)
        try:
            crop_s = int(self.size_entry.get())
        except:
            crop_s = int(self.warp_size // 8 * 0.9)
        half = crop_s // 2
        x1 = max(0, x_orig - half)
        y1 = max(0, y_orig - half)
        x2 = min(self.warp_size, x_orig + half)
        y2 = min(self.warp_size, y_orig + half)
        crop = self.warped_img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            messagebox.showerror("Crop error", "Could not crop at this location.")
            return
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb).convert("RGBA")
        pil_save = pil_crop.resize((self.template_save_size, self.template_save_size), Image.ANTIALIAS)
        name = TEMPLATE_ORDER[self.current_index]
        out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
        pil_save.save(out_path)
        self.templates_captured[name] = pil_save
        self.current_index += 1
        self.update_thumbnails()
        if self.current_index < len(TEMPLATE_ORDER):
            self.target_label.config(text="Next: " + TEMPLATE_ORDER[self.current_index])
            self.info_label.config(text=f"Captured {name}. Click the center of {TEMPLATE_ORDER[self.current_index]}.")
        else:
            self.target_label.config(text="All captured")
            self.info_label.config(text="All templates captured. Press Finish & Save or Undo to redo last.")

    def update_thumbnails(self):
        for widget in self.thumb_frame.winfo_children():
            widget.destroy()
        for name in TEMPLATE_ORDER:
            frame = tk.Frame(self.thumb_frame, width=56, height=56)
            frame.pack(side="left", padx=4)
            if name in self.templates_captured:
                im = self.templates_captured[name]
                tkim = ImageTk.PhotoImage(im)
                lbl = tk.Label(frame, image=tkim)
                lbl.image = tkim
                lbl.pack()
                tk.Label(frame, text=name, font=("Arial",8)).pack()
            else:
                lbl = tk.Label(frame, text="—", width=6, height=3, bg="#eee")
                lbl.pack()
                tk.Label(frame, text=name, font=("Arial",8)).pack()

    def skip(self):
        if self.current_index < len(TEMPLATE_ORDER):
            self.current_index += 1
            if self.current_index < len(TEMPLATE_ORDER):
                self.target_label.config(text="Next: " + TEMPLATE_ORDER[self.current_index])
            else:
                self.target_label.config(text="All captured")
            self.info_label.config(text="Skipped one. You can undo if that was a mistake.")
            self.update_thumbnails()

    def undo(self):
        if self.current_index == 0:
            return
        self.current_index -= 1
        name = TEMPLATE_ORDER[self.current_index]
        path = os.path.join(OUTPUT_DIR, f"{name}.png")
        if os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass
        if name in self.templates_captured:
            del self.templates_captured[name]
        self.target_label.config(text="Next: " + TEMPLATE_ORDER[self.current_index])
        self.info_label.config(text=f"Undid {name}. Click to capture again.")
        self.update_thumbnails()

    def finish(self):
        saved = []
        for name in TEMPLATE_ORDER:
            p = os.path.join(OUTPUT_DIR, f"{name}.png")
            if os.path.exists(p):
                saved.append(name)
        messagebox.showinfo("Saved", f"Saved templates: {saved}\nFiles in folder: {os.path.abspath(OUTPUT_DIR)}")
        self.info_label.config(text="Finished. You can re-run to capture different templates.")

if __name__ == "__main__":
    root = tk.Tk()
    app = TemplateCaptureApp(root)
    root.mainloop()
