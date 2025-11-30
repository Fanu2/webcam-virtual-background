"""
virtual_bg_recorder.py

A self-contained script to preview webcam with virtual-background effects
(similar to Google Meet) and record video (optionally with audio).

Dependencies:
  pip install opencv-python mediapipe numpy sounddevice scipy

Optional: ffmpeg in PATH for audio+video muxing (recommended on most systems).
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
import glob
import threading
import queue
import sounddevice as sd
import scipy.io.wavfile as wavfile
import subprocess
from datetime import datetime

# --------- Configuration ----------
CAM_INDEX = 0
OUT_DIR = "outputs"
BG_IMG_DIR = "backgrounds"
BG_VIDEO_DIR = "bg_videos"
FPS = 20  # output recording fps
FRAME_W = 1280
FRAME_H = 720
AUDIO_SR = 44100  # audio sample rate
# ---------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(BG_IMG_DIR, exist_ok=True)
os.makedirs(BG_VIDEO_DIR, exist_ok=True)

# Load background images
bg_images_paths = sorted(glob.glob(os.path.join(BG_IMG_DIR, "*.*")))
bg_images = []
for p in bg_images_paths:
    img = cv2.imread(p)
    if img is not None:
        bg_images.append(cv2.resize(img, (FRAME_W, FRAME_H)))
# Load background videos (as capture objects)
bg_video_paths = sorted(glob.glob(os.path.join(BG_VIDEO_DIR, "*.*")))

# Mediapipe Selfie Segmentation
mp_selfie = mp.solutions.selfie_segmentation
segmentation_model = mp_selfie.SelfieSegmentation(model_selection=1)

# Audio recorder (simple threaded)
audio_queue = queue.Queue()
record_audio_flag = threading.Event()

def audio_record_worker(filename_wav):
    """Record audio to filename_wav using sounddevice in a separate thread."""
    try:
        # Using a callback to avoid blocking
        def callback(indata, frames, time_info, status):
            if record_audio_flag.is_set():
                audio_queue.put(indata.copy())
        # stream
        with sd.InputStream(channels=1, samplerate=AUDIO_SR, callback=callback):
            frames = []
            while record_audio_flag.is_set() or not audio_queue.empty():
                try:
                    piece = audio_queue.get(timeout=0.5)
                    frames.append(piece)
                except queue.Empty:
                    continue
            if frames:
                arr = np.concatenate(frames, axis=0)
                wavfile.write(filename_wav, AUDIO_SR, arr)
    except Exception as e:
        print("Audio recording error:", e)

# Utility: apply background replacement using segmentation mask
def apply_virtual_bg(frame, mask, bg_frame):
    # mask is float [0..1], same size as frame, where 1 -> person
    mask_3 = np.repeat(mask[:, :, None], 3, axis=2)
    fg = (frame.astype(np.float32) * mask_3)
    bg = (bg_frame.astype(np.float32) * (1.0 - mask_3))
    out = cv2.add(fg, bg).astype(np.uint8)
    return out

# Utility: blur background
def blur_background(frame, mask, ksize=35):
    bg = cv2.blur(frame, (ksize, ksize))
    return apply_virtual_bg(frame, mask, bg)

# Utility: solid color background
def color_background(frame, mask, color=(0,255,0)):
    bg = np.full_like(frame, color, dtype=np.uint8)
    return apply_virtual_bg(frame, mask, bg)

# main
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    # state variables
    bg_idx = 0
    use_bg_image = len(bg_images) > 0
    use_bg_video = False
    bg_video_idx = 0
    blur_mode = 0  # 0 off, 1 low, 2 med, 3 high
    mirror = True
    show_grid = False
    color_idx = 0
    colors = [(0,255,0), (255,0,0), (0,0,0), (255,255,255)]
    no_replace = False

    # Prepare video writer
    recording = False
    video_writer = None
    audio_thread = None
    audio_fname = None
    wav_fname = None

    # video bg capture object
    bg_cap = None
    if len(bg_video_paths) > 0:
        bg_cap = cv2.VideoCapture(bg_video_paths[bg_video_idx])

    print("Controls: 1-9: bg image | b: blur | c: color | v: bg video | n: none | m: mirror")
    print("g: grid | s: snapshot | r: start/stop recording | q: quit")

    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not available")
            break
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        if mirror:
            frame = cv2.flip(frame, 1)

        # run segmentation
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = segmentation_model.process(rgb)
        mask = None
        if results.segmentation_mask is not None:
            mask = results.segmentation_mask
            # refine mask: threshold and gaussian
            mask = cv2.GaussianBlur(mask, (11,11), 0)
            mask = cv2.normalize(mask, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
        else:
            mask = np.ones((FRAME_H, FRAME_W), dtype=np.float32)

        # choose background frame
        final = frame.copy()
        if no_replace:
            final = frame.copy()
        else:
            if use_bg_image and not use_bg_video:
                bg_frame = bg_images[bg_idx % len(bg_images)]
                if blur_mode:
                    final = blur_background(frame, mask, ksize=15 + blur_mode*10)
                else:
                    final = apply_virtual_bg(frame, mask, bg_frame)
            elif use_bg_video and bg_cap is not None:
                # read bg frame
                retbg, bgf = bg_cap.read()
                if not retbg:
                    bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    retbg, bgf = bg_cap.read()
                if retbg:
                    bgf = cv2.resize(bgf, (FRAME_W, FRAME_H))
                    if blur_mode:
                        final = blur_background(frame, mask, ksize=15 + blur_mode*10)
                    else:
                        final = apply_virtual_bg(frame, mask, bgf)
                else:
                    final = frame.copy()
            elif color_idx is not None and not use_bg_image and not use_bg_video:
                final = color_background(frame, mask, colors[color_idx % len(colors)])
            else:
                final = frame.copy()

        # overlay grid
        if show_grid:
            # rule of thirds
            h, w = final.shape[:2]
            cv2.line(final, (w//3, 0), (w//3, h), (255,255,255), 1)
            cv2.line(final, (2*w//3, 0), (2*w//3, h), (255,255,255), 1)
            cv2.line(final, (0, h//3), (w, h//3), (255,255,255), 1)
            cv2.line(final, (0, 2*h//3), (w, 2*h//3), (255,255,255), 1)

        # show FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time else 0.0
        prev_time = curr_time
        cv2.putText(final, f"FPS: {int(fps)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # show recording indicator
        if recording:
            cv2.circle(final, (FRAME_W-30, 30), 10, (0,0,255), -1)
            cv2.putText(final, "REC", (FRAME_W-90, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow("VirtualBG Recorder", final)

        # write frame if recording
        if recording and video_writer is not None:
            video_writer.write(final)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # ESC
            if recording:
                print("Stopping recording before exit...")
                recording = False
                # stop audio thread
                record_audio_flag.clear()
                if audio_thread:
                    audio_thread.join(timeout=2)
                if video_writer:
                    video_writer.release()
                # mux if audio exists
                if wav_fname and os.path.exists(wav_fname):
                    outmp4 = video_fname.replace(".mp4", "_with_audio.mp4")
                    ffmpeg_mux(video_fname, wav_fname, outmp4)
                    print("Saved with audio:", outmp4)
            break

        # numeric keys: switch bg images
        if key >= ord('1') and key <= ord('9'):
            idx = key - ord('1')
            if idx < len(bg_images):
                bg_idx = idx
                use_bg_image = True
                use_bg_video = False
                no_replace = False
                print("Switched to background image", bg_idx)

        if key == ord('b'):
            blur_mode = (blur_mode + 1) % 4
            print("Blur mode:", blur_mode)

        if key == ord('c'):
            # switch to color backgrounds
            color_idx = (color_idx + 1) % len(colors)
            use_bg_image = False
            use_bg_video = False
            no_replace = False
            print("Solid color background:", colors[color_idx])

        if key == ord('v'):
            if len(bg_video_paths) == 0:
                print("No bg videos found in", BG_VIDEO_DIR)
            else:
                use_bg_video = not use_bg_video
                if use_bg_video:
                    if bg_cap is None:
                        bg_cap = cv2.VideoCapture(bg_video_paths[bg_video_idx])
                    print("Using background video")
                else:
                    print("Stopped background video")

        if key == ord('n'):
            no_replace = True
            print("Background replacement OFF (raw webcam)")

        if key == ord('m'):
            mirror = not mirror
            print("Mirror:", mirror)

        if key == ord('g'):
            show_grid = not show_grid
            print("Grid:", show_grid)

        if key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap = os.path.join(OUT_DIR, f"snapshot_{timestamp}.png")
            cv2.imwrite(snap, final)
            print("Saved snapshot:", snap)

        if key == ord('r'):
            # toggle recording
            if not recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_fname = os.path.join(OUT_DIR, f"recording_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_fname, fourcc, FPS, (FRAME_W, FRAME_H))
                recording = True
                print("Started recording:", video_fname)
                # try to start audio thread
                wav_fname = os.path.join(OUT_DIR, f"audio_{timestamp}.wav")
                try:
                    record_audio_flag.set()
                    audio_thread = threading.Thread(target=audio_record_worker, args=(wav_fname,), daemon=True)
                    audio_thread.start()
                    print("Audio thread started")
                except Exception as e:
                    print("Could not start audio:", e)
                    wav_fname = None
            else:
                # stop
                recording = False
                record_audio_flag.clear()
                if audio_thread:
                    audio_thread.join(timeout=2)
                if video_writer:
                    video_writer.release()
                    print("Stopped recording:", video_fname)
                # if audio exists, mux
                if wav_fname and os.path.exists(wav_fname):
                    outmp4 = video_fname.replace(".mp4", "_with_audio.mp4")
                    try:
                        ffmpeg_mux(video_fname, wav_fname, outmp4)
                        print("Saved with audio:", outmp4)
                    except Exception as e:
                        print("ffmpeg mux failed:", e)
                        print("Video saved at:", video_fname)

        # small delay to reduce CPU
        time.sleep(0.001)

    cap.release()
    if bg_cap:
        bg_cap.release()
    cv2.destroyAllWindows()

def ffmpeg_mux(video_file, wav_file, out_file):
    """Use ffmpeg to mux audio + video into an MP4. Requires ffmpeg in PATH."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_file,
        "-i", wav_file,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        out_file
    ]
    print("Running ffmpeg:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()

