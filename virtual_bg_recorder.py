"""
virtual_bg_recorder.py — FIXED VERSION
Google Meet–style background change + blur + solid color + video bg + recording
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import glob
import time
from datetime import datetime

# ---------------------- SETTINGS -------------------------
CAM_INDEX = 0
FRAME_W = 1280
FRAME_H = 720
FPS = 20

BG_IMG_DIR = "backgrounds"
BG_VIDEO_DIR = "bg_videos"
OUT_DIR = "outputs"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(BG_IMG_DIR, exist_ok=True)
os.makedirs(BG_VIDEO_DIR, exist_ok=True)

# ---------------------- LOAD BACKGROUNDS -------------------
bg_images_paths = sorted(glob.glob(os.path.join(BG_IMG_DIR, "*.*")))
bg_images = []

for p in bg_images_paths:
    img = cv2.imread(p)
    if img is not None:
        bg_images.append(cv2.resize(img, (FRAME_W, FRAME_H)))

bg_video_paths = sorted(glob.glob(os.path.join(BG_VIDEO_DIR, "*.*")))
bg_cap = None

print("Loaded Image Backgrounds:", len(bg_images))
print("Loaded Video Backgrounds:", len(bg_video_paths))

# ---------------------- MEDIAPIPE SEGMENTATION -------------
mp_selfie = mp.solutions.selfie_segmentation
seg = mp_selfie.SelfieSegmentation(model_selection=1)


# ---------------------- UTILITY FUNCTIONS -------------------
def apply_virtual_bg(frame, mask, bg):
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    f = (frame * mask3).astype(np.uint8)
    b = (bg * (1 - mask3)).astype(np.uint8)
    return f + b


def blur_background(frame, mask, level=1):
    k = 25 + level * 20
    bg = cv2.GaussianBlur(frame, (k, k), 0)
    return apply_virtual_bg(frame, mask, bg)


def color_background(frame, mask, color):
    bg = np.full_like(frame, color, dtype=np.uint8)
    return apply_virtual_bg(frame, mask, bg)


# ---------------------- MAIN APP ---------------------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    use_bg_image = True
    use_bg_video = False
    blur_mode = 0
    color_idx = 0
    no_replace = False
    mirror = True
    show_grid = False
    bg_idx = 0

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 0), (255, 255, 255)]

    if bg_video_paths:
        bg_cap = cv2.VideoCapture(bg_video_paths[0])
    else:
        bg_cap = None

    print("\nCONTROLS:")
    print("1–9 = Change background image")
    print("b = Blur background")
    print("c = Solid color background")
    print("v = Video background")
    print("n = No background replacement")
    print("m = Mirror ON/OFF")
    print("g = Grid ON/OFF")
    print("q = Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera Error")
            break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        if mirror:
            frame = cv2.flip(frame, 1)

        # segmentation
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = seg.process(rgb).segmentation_mask
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = np.clip(mask, 0, 1)

        final = frame.copy()

        if not no_replace:

            # 1 — Background Image
            if use_bg_image and len(bg_images) > 0:
                bg = bg_images[bg_idx % len(bg_images)]
                final = apply_virtual_bg(frame, mask, bg)

            # 2 — Blur
            if blur_mode > 0:
                final = blur_background(frame, mask, blur_mode)

            # 3 — Solid Color
            if not use_bg_image and not use_bg_video and blur_mode == 0:
                final = color_background(frame, mask, colors[color_idx])

            # 4 — Video Background
            if use_bg_video and bg_cap:
                retb, bgf = bg_cap.read()
                if not retb:
                    bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    retb, bgf = bg_cap.read()

                if retb:
                    bgf = cv2.resize(bgf, (FRAME_W, FRAME_H))
                    final = apply_virtual_bg(frame, mask, bgf)

        # Grid overlay
        if show_grid:
            h, w = final.shape[:2]
            cv2.line(final, (w//3, 0), (w//3, h), (255, 255, 255), 1)
            cv2.line(final, (2*w//3, 0), (2*w//3, h), (255, 255, 255), 1)
            cv2.line(final, (0, h//3), (w, h//3), (255, 255, 255), 1)
            cv2.line(final, (0, 2*h//3), (w, 2*h//3), (255, 255, 255), 1)

        cv2.imshow("Virtual Background", final)
        key = cv2.waitKey(1) & 0xFF

        # ---------------- KEY CONTROLS ----------------
        if key == ord('q'):
            break

        if ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(bg_images):
                bg_idx = idx
                use_bg_image = True
                use_bg_video = False
                no_replace = False
                print(f"Switched to image #{idx+1}")

        if key == ord('b'):
            blur_mode = (blur_mode + 1) % 4
            use_bg_image = False
            use_bg_video = False
            no_replace = False
            print("Blur mode:", blur_mode)

        if key == ord('c'):
            color_idx = (color_idx + 1) % len(colors)
            use_bg_image = False
            use_bg_video = False
            no_replace = False
            print("Solid color background:", colors[color_idx])

        if key == ord('v'):
            if bg_video_paths:
                use_bg_video = not use_bg_video
                use_bg_image = False
                no_replace = False
                print("Video background:", use_bg_video)
            else:
                print("No video found in bg_videos/")

        if key == ord('n'):
            no_replace = True
            print("Background replacement OFF")

        if key == ord('m'):
            mirror = not mirror
            print("Mirror:", mirror)

        if key == ord('g'):
            show_grid = not show_grid
            print("Grid:", show_grid)

    cap.release()
    if bg_cap:
        bg_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

