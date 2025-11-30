#!/usr/bin/env python3
"""
virtual_bg_fixed_preview.py
SAFE + WORKING VERSION
Includes:
 - Live Camera Preview
 - Background Replacement (Image/GIF/PNG Alpha)
 - Scrollable Thumbnails
 - Add Background (file dialog)
 - Screenshot Save
 - CPU-friendly segmentation
"""

import os, glob, time, threading, queue, json, shutil
from datetime import datetime
import cv2
import numpy as np
import PySimpleGUI as sg
import mediapipe as mp
from PIL import Image, ImageSequence

# ---------------- Configs ----------------
FRAME_W, FRAME_H = 1280, 720
PREVIEW_W, PREVIEW_H = 960, 540
ROOT_BG = "backgrounds"
OUT_DIR = "outputs"
CONFIG_FILE = "config_stable.json"
THUMB_W, THUMB_H = 160, 90

os.makedirs(ROOT_BG, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Helpers ----------------
def now_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_config():
    cfg = {"mode": "None", "last_idx": None}
    if os.path.exists(CONFIG_FILE):
        try:
            cfg.update(json.load(open(CONFIG_FILE)))
        except:
            pass
    return cfg

def save_config(cfg):
    json.dump(cfg, open(CONFIG_FILE, "w"), indent=2)

config = load_config()

# ------------- Load backgrounds ---------------
def load_backgrounds():
    bg_list = []
    for p in sorted(glob.glob(f"{ROOT_BG}/*.*")):
        name = os.path.basename(p)
        ext = os.path.splitext(name)[1].lower()
        try:
            if ext == ".png":
                pil = Image.open(p).convert("RGBA")
                arr = np.array(pil)
                bgr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGBA2BGR)
                alpha = arr[..., 3]/255.0
                bgr = cv2.resize(bgr, (FRAME_W, FRAME_H))
                alpha = cv2.resize(alpha, (FRAME_W, FRAME_H))
                bg_list.append((name, "img_alpha", (bgr, alpha)))

            elif ext == ".gif":
                frames = []
                pil = Image.open(p)
                for f in ImageSequence.Iterator(pil):
                    fr = f.convert("RGB").resize((FRAME_W, FRAME_H))
                    frames.append(cv2.cvtColor(np.array(fr), cv2.COLOR_RGB2BGR))
                bg_list.append((name, "gif", frames))

            else:
                img = cv2.imread(p)
                if img is None: continue
                img = cv2.resize(img, (FRAME_W, FRAME_H))
                bg_list.append((name, "img", img))

        except Exception as e:
            print("Error:", e)

    return bg_list

bg_list = load_backgrounds()

# -------------- Segmentation ---------------
mp_selfie = mp.solutions.selfie_segmentation
seg = mp_selfie.SelfieSegmentation(model_selection=1)

cap = cv2.VideoCapture(0)
cap.set(3, FRAME_W)
cap.set(4, FRAME_H)

frame_q = queue.Queue(maxsize=2)
mask_q  = queue.Queue(maxsize=2)
out_q   = queue.Queue(maxsize=2)

running = True

# ---------------- Threads ------------------
def capture_loop():
    while running:
        ok, f = cap.read()
        if not ok: continue
        f = cv2.resize(f, (FRAME_W, FRAME_H))
        if not frame_q.full():
            frame_q.put(f)

def seg_loop():
    while running:
        try:
            f = frame_q.get(timeout=0.2)
        except:
            continue

        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        m = seg.process(rgb).segmentation_mask.astype(np.float32)
        if not mask_q.full():
            mask_q.put((f, m))

def apply_mask(frame, mask, bg):
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    return (frame*mask3 + bg*(1-mask3)).astype(np.uint8)

t1 = threading.Thread(target=capture_loop, daemon=True)
t2 = threading.Thread(target=seg_loop, daemon=True)
t1.start()
t2.start()

# ---------------- GUI ---------------------
sg.theme("DarkBlue")

def make_thumb(bg_entry):
    name, typ, data = bg_entry
    if typ == "img":
        img = data
    elif typ == "img_alpha":
        img = data[0]
    else:
        img = data[0][0]

    t = cv2.resize(img, (THUMB_W, THUMB_H))
    ok, buf = cv2.imencode(".png", t)
    return buf.tobytes()

thumb_rows = []
row=[]
for i, entry in enumerate(bg_list):
    row.append(sg.Button(image_data=make_thumb(entry), key=f"TH_{i}", pad=(2,2)))
    if len(row)==4:
        thumb_rows.append(row)
        row=[]
if row:
    thumb_rows.append(row)

layout = [
    [sg.Image(size=(PREVIEW_W, PREVIEW_H), key="preview")],
    [sg.Text("Mode:"), sg.Combo(["None","Image"], default_value=config["mode"], key="mode")],
    [sg.Button("Add Background"), sg.Button("Reload"), sg.Button("Save Screenshot")],
    [sg.Text("", key="status", size=(40,2))],
    [sg.Frame("Backgrounds", [[sg.Column(thumb_rows, scrollable=True, size=(700,300), key="thumb_col")]])],
    [sg.Exit()]
]

window = sg.Window("Virtual BG - FIXED PREVIEW", layout, finalize=True)

selected_idx = config.get("last_idx", None)

last_out = None
gif_index = 0

# --------------- Main Loop -----------------
try:
    while True:
        event, values = window.read(timeout=10)
        if event in (sg.WIN_CLOSED, "Exit"):
            break

        if event == "Add Background":
            files = sg.popup_get_file("Select images:", multiple_files=True)
            if files:
                for f in files.split(";"):
                    shutil.copy2(f, ROOT_BG)
                bg_list = load_backgrounds()
                sg.popup("Backgrounds added. Click Reload.")
            continue

        if event == "Reload":
            bg_list = load_backgrounds()
            # rebuild thumbnails
            rows=[]
            row=[]
            for i,e in enumerate(bg_list):
                row.append(sg.Button(image_data=make_thumb(e), key=f"TH_{i}"))
                if len(row)==4:
                    rows.append(row)
                    row=[]
            if row: rows.append(row)
            window["thumb_col"].update(rows)
            window["status"].update("Reloaded backgrounds")
            continue

        if event == "Save Screenshot":
            if last_out is not None:
                path = f"{OUT_DIR}/shot_{now_ts()}.png"
                cv2.imwrite(path, last_out)
                window["status"].update(f"Saved: {path}")
            continue

        if isinstance(event,str) and event.startswith("TH_"):
            selected_idx = int(event.split("_")[1])
            window["status"].update(f"Selected: {bg_list[selected_idx][0]}")
            continue

        # process frame
        try:
            frame, mask = mask_q.get_nowait()
        except queue.Empty:
            continue

        out = frame.copy()
        mode = values["mode"]

        if mode=="Image" and selected_idx is not None:
            name, typ, data = bg_list[selected_idx]
            if typ == "img":
                bg = data
            elif typ == "img_alpha":
                bg = data[0]
            else:
                seq = data
                bg = seq[gif_index % len(seq)]
                gif_index += 1

            out = apply_mask(frame, mask, bg)

        # update preview
        prev = cv2.resize(out, (PREVIEW_W, PREVIEW_H))
        ok, buf = cv2.imencode(".png", prev)
        if ok:
            window["preview"].update(data=buf.tobytes())

        last_out = out

finally:
    running = False
    time.sleep(0.2)
    cap.release()
    window.close()
    save_config({"mode": values.get("mode","None"), "last_idx": selected_idx})
