"""
virtual_bg_all_in_one.py
All-in-one virtual background app (GUI + virtual cam + recording + streaming + advanced matting)
- Falls back to MediaPipe segmentation if advanced models are not present
- Uses pyvirtualcam for virtual webcam output
- Uses ffmpeg for muxing & RTMP streaming
"""

import os, glob, time, threading, queue, subprocess, math, sys, traceback
from datetime import datetime
import cv2
import numpy as np
import PySimpleGUI as sg
import mediapipe as mp
import sounddevice as sd
import scipy.io.wavfile as wavfile
import requests
import noisereduce as nr  # optional, installed earlier
try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
    HAVE_PYVIRT = True
except Exception:
    HAVE_PYVIRT = False

# ---------- SETTINGS ----------
FRAME_W, FRAME_H = 1280, 720
FPS = 20
BG_DIR = "backgrounds"
BG_VID_DIR = "bg_videos"
OUT_DIR = "outputs"
MODEL_DIR = "models"
AUDIO_SR = 44100
AUTO_SNAPSHOT_SEC = 0  # 0 disables auto snapshots by default
# ------------------------------

os.makedirs(BG_DIR, exist_ok=True)
os.makedirs(BG_VID_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- Load background images and videos ----------
def load_backgrounds():
    imgs = []
    paths = sorted(glob.glob(os.path.join(BG_DIR, "*.*")))
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.resize(img, (FRAME_W, FRAME_H))
        imgs.append((os.path.basename(p), img))
    vids = sorted(glob.glob(os.path.join(BG_VID_DIR, "*.*")))
    return imgs, vids

bg_images, bg_videos = load_backgrounds()

# ---------- Segmentation backend selection ----------
mp_selfie = mp.solutions.selfie_segmentation
seg_model = mp_selfie.SelfieSegmentation(model_selection=1)

# Optional advanced matting (if model exists) - placeholder loader
ADVANCED_MATTING_AVAILABLE = False
# Example: check for modnet file
modnet_path = os.path.join(MODEL_DIR, "modnet.pth")
if os.path.exists(modnet_path):
    ADVANCED_MATTING_AVAILABLE = True
    # In real full build we'd load the MODNet model here (torch load)
    print("Advanced matting model found (MODNet). Will attempt to use it if GPU/torch available.")
else:
    print("No advanced matting models found. Using MediaPipe segmentation (CPU) as fallback.")

# ---------- Threaded Capture / Processing Pipeline ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

frame_queue = queue.Queue(maxsize=4)
result_queue = queue.Queue(maxsize=4)
running = threading.Event()
running.set()

def capture_loop():
    while running.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        frame_queue.put(frame)
    # release handled elsewhere

def processing_loop(state):
    """Grab frame from frame_queue, compute mask and postprocess"""
    # state is a dict holding UI options etc.
    temporal_mask = None
    alpha_smoothing = 0.6  # temporal smoothing factor
    while running.is_set():
        try:
            frame = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        # mirror if required
        if state['mirror']:
            frame = cv2.flip(frame, 1)

        # segmentation: choose advanced matting if available and requested
        mask = None
        if state['use_advanced_matte'] and ADVANCED_MATTING_AVAILABLE:
            # Placeholder: call MODNet or RVM inference (not implemented here)
            # For now fallback to mediapipe
            pass

        # fallback: mediapipe selfie segmentation
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = seg_model.process(rgb)
        if res and res.segmentation_mask is not None:
            mask = res.segmentation_mask
            # refine mask
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            # normalize & clip
            mask = np.clip(mask, 0.0, 1.0)
        else:
            mask = np.ones((FRAME_H, FRAME_W), dtype=np.float32)

        # Temporal smoothing to reduce flicker
        if temporal_mask is None:
            temporal_mask = mask
        else:
            temporal_mask = (alpha_smoothing * temporal_mask) + ((1 - alpha_smoothing) * mask)
            temporal_mask = np.clip(temporal_mask, 0.0, 1.0)

        # skin smoothing / beautify (simple bilateral filter on face area)
        proc_frame = frame.copy()
        if state['beautify']:
            # simple face detector to find ROI (use mp.face_mesh or Haar)
            gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
            # use Haar cascades shipped with OpenCV if present
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = cascade.detectMultiScale(gray, 1.1, 6)
            for (x,y,w,h) in faces:
                roi = proc_frame[y:y+h, x:x+w]
                roi = cv2.bilateralFilter(roi, d=9, sigmaColor=75, sigmaSpace=75)
                proc_frame[y:y+h, x:x+w] = roi

        # store results
        result_queue.put((proc_frame, temporal_mask))
    # end

# Start threads
state = {
    'mirror': True,
    'use_advanced_matte': False,  # UI toggle
    'beautify': False
}
capture_thread = threading.Thread(target=capture_loop, daemon=True)
proc_thread = threading.Thread(target=processing_loop, args=(state,), daemon=True)
capture_thread.start()
proc_thread.start()

# ---------- Audio recording helper ----------
audio_queue = queue.Queue()
record_audio_flag = threading.Event()
audio_frames = []

def audio_callback(indata, frames, time_info, status):
    if record_audio_flag.is_set():
        audio_queue.put(indata.copy())

def start_audio_recording():
    record_audio_flag.set()
    audio_frames.clear()
    # start input stream
    sd.default.samplerate = AUDIO_SR
    sd.default.channels = 1
    audio_stream = sd.InputStream(callback=audio_callback)
    audio_stream.start()
    return audio_stream

def stop_audio_recording(stream, wav_out_path):
    record_audio_flag.clear()
    # gather queued frames
    frames = []
    while not audio_queue.empty():
        frames.append(audio_queue.get())
    if frames:
        arr = np.concatenate(frames, axis=0)
        wavfile.write(wav_out_path, AUDIO_SR, arr)
    try:
        stream.stop()
        stream.close()
    except Exception:
        pass

# ---------- Video compose & background apply utilities ----------
def apply_virtual_bg(frame, mask, bg):
    # mask: HxW float [0..1], 1 where person present
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    fg = (frame.astype(np.float32) * mask3)
    bg = (bg.astype(np.float32) * (1.0 - mask3))
    out = (fg + bg).astype(np.uint8)
    return out

def blur_background(frame, mask, strength=1):
    k = 21 + strength * 20
    bg = cv2.GaussianBlur(frame, (k | 1, k | 1), 0)
    return apply_virtual_bg(frame, mask, bg)

def color_background(frame, mask, color):
    bg = np.full_like(frame, color, dtype=np.uint8)
    return apply_virtual_bg(frame, mask, bg)

def adjust_brightness_contrast(img, brightness=0, contrast=1.0):
    # brightness: -100..100 (int), contrast: 0.5..2.0 (float)
    out = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    return out

# ---------- GUI Layout ----------
sg.theme("DarkBlue3")

# build thumbnail list
def build_thumbnail_grid(bg_images):
    thumbs = []
    row = []
    for name, img in bg_images:
        thumb = cv2.resize(img, (120, int(120*FRAME_H/FRAME_W)))
        # convert to PNG bytes for PySimpleGUI
        is_success, buffer = cv2.imencode(".png", thumb)
        if not is_success:
            continue
        b = buffer.tobytes()
        row.append(sg.Button('', image_data=b, key=f"TH_{name}", tooltip=name, pad=(2,2)))
        if len(row) == 6:
            thumbs.append(row)
            row = []
    if row:
        thumbs.append(row)
    return thumbs

thumb_grid = build_thumbnail_grid(bg_images)

controls_col = [
    [sg.Text("Mode:"), sg.Combo(['None','Image','Blur','Solid Color','Video'], default_value='None', key='mode')],
    [sg.Button("Start Virtual Cam", key='start_vcam'), sg.Button("Stop Virtual Cam", key='stop_vcam', disabled=True)],
    [sg.Button("Start Recording", key='start_rec'), sg.Button("Stop Recording", key='stop_rec', disabled=True)],
    [sg.Button("Start Stream", key='start_stream'), sg.Button("Stop Stream", key='stop_stream', disabled=True)],
    [sg.Button("Snapshot", key='snapshot'), sg.Checkbox('Auto snapshot every (sec)', key='auto_snap_en', default=False), sg.Input('0', size=(6,1), key='auto_snap_interval')],
    [sg.Text("Blur Strength"), sg.Slider(range=(0,3), orientation='h', size=(20,15), default_value=0, key='blur_slider')],
    [sg.Text("Brightness"), sg.Slider(range=(-80,80), orientation='h', size=(20,15), default_value=0, key='bright_slider')],
    [sg.Text("Contrast x"), sg.Slider(range=(50,200), orientation='h', size=(20,15), default_value=100, key='contrast_slider')],
    [sg.Checkbox('Beautify', key='beautify'), sg.Checkbox('Use Advanced Matte (if available)', key='adv_matte')],
    [sg.Text("AI Background Gen (optional)"), sg.Input('', key='ai_prompt', size=(30,1)), sg.Button("Generate", key='ai_gen')],
    [sg.Input('', key='rtmp_url', size=(40,1)), sg.Text("RTMP URL"),],
    [sg.Text('', key='status', size=(60,3))]
]

layout = [
    [sg.Column([[sg.Image(filename='', key='preview', size=(FRAME_W//2, FRAME_H//2))]], pad=(0,0)),
     sg.Column(controls_col)],
]

# add thumbnail grid below if exists
if thumb_grid:
    for r in thumb_grid:
        layout.append(r)

layout.append([sg.Button('Reload Backgrounds', key='reload_bg'), sg.Button('Exit')])

window = sg.Window('Virtual Background — All-in-One', layout, finalize=True, resizable=True)

# ---------- State variables ----------
vcam = None
vcam_active = False
recording = False
record_proc = None
ffmpeg_proc = None
audio_stream = None
wav_temp = None
video_temp = None
streaming = False
auto_snapshot_thread = None

# virtual camera start function
def start_virtual_cam():
    global vcam, vcam_active
    if not HAVE_PYVIRT:
        window['status'].update('pyvirtualcam not installed - virtual cam disabled')
        return False
    try:
        vcam = pyvirtualcam.Camera(width=FRAME_W, height=FRAME_H, fps=FPS, fmt=PixelFormat.BGR)
        vcam_active = True
        window['status'].update('Virtual camera started')
        return True
    except Exception as e:
        window['status'].update(f'Virtual cam start failed: {e}')
        return False

def stop_virtual_cam():
    global vcam, vcam_active
    try:
        if vcam:
            vcam.close()
            vcam = None
        vcam_active = False
        window['status'].update('Virtual camera stopped')
    except Exception as e:
        window['status'].update(f'Could not stop vcam: {e}')

# recording start/stop
def start_recording():
    global recording, audio_stream, wav_temp, video_temp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_temp = os.path.join(OUT_DIR, f"video_{timestamp}.mp4")
    wav_temp = os.path.join(OUT_DIR, f"audio_{timestamp}.wav")
    # open ffmpeg writer (we will write raw frames with cv2 VideoWriter; we'll mux audio later)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(video_temp, fourcc, FPS, (FRAME_W, FRAME_H))
    # start audio
    try:
        audio_stream = start_audio_recording()
    except Exception as e:
        window['status'].update(f"Audio start failed: {e}")
        audio_stream = None
    recording = True
    window['status'].update("Recording started")
    return vw, audio_stream

def stop_recording(vw, audio_stream):
    global recording, wav_temp, video_temp
    recording = False
    try:
        if audio_stream:
            stop_audio_recording(audio_stream, wav_temp)
    except Exception:
        pass
    try:
        vw.release()
    except Exception:
        pass
    # mux audio and video using ffmpeg if audio exists
    final_out = video_temp.replace(".mp4", "_final.mp4")
    if wav_temp and os.path.exists(wav_temp):
        cmd = ["ffmpeg", "-y", "-i", video_temp, "-i", wav_temp, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", final_out]
        subprocess.run(cmd)
        window['status'].update(f"Saved: {final_out}")
    else:
        window['status'].update(f"Saved video: {video_temp}")

# streaming start stop (RTMP) using ffmpeg
def start_streaming(rtmp_url):
    global ffmpeg_proc, streaming
    # create ffmpeg process expecting raw video frames over pipe or reuse file-based approach
    # Here we'll use a simple approach: write to local pipe via ffmpeg stdin (rawvideo)
    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{FRAME_W}x{FRAME_H}',
        '-r', str(FPS),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-b:v', '2500k',
        '-maxrate', '2500k',
        '-bufsize', '5000k',
        '-g', '50',
        '-f', 'flv',
        rtmp_url
    ]
    ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    streaming = True
    window['status'].update("Streaming started")

def stop_streaming():
    global ffmpeg_proc, streaming
    if ffmpeg_proc:
        try:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.terminate()
            ffmpeg_proc = None
        except Exception:
            pass
    streaming = False
    window['status'].update("Streaming stopped")

# auto snapshot thread
def auto_snapshot_runner(interval):
    while running.is_set():
        time.sleep(interval)
        # trigger a snapshot event by writing to window
        window.write_event_value('AUTO_SNAPSHOT', None)

# ---------- Main UI loop ----------
current_vw = None  # VideoWriter for recording
audio_handle = None
last_snapshot_time = 0
auto_snap_thread = None

try:
    while True:
        event, values = window.read(timeout=1)
        # handle UI events
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == 'reload_bg':
            bg_images, bg_videos = load_backgrounds()
            # rebuild thumbnails not implemented live to keep code shorter
            window['status'].update(f"Reloaded backgrounds: {len(bg_images)}")
        if event == 'start_vcam':
            if start_virtual_cam():
                window['start_vcam'].update(disabled=True)
                window['stop_vcam'].update(disabled=False)
        if event == 'stop_vcam':
            stop_virtual_cam()
            window['start_vcam'].update(disabled=False)
            window['stop_vcam'].update(disabled=True)
        if event == 'start_rec':
            if not recording:
                current_vw, audio_handle = start_recording()
                window['start_rec'].update(disabled=True)
                window['stop_rec'].update(disabled=False)
        if event == 'stop_rec':
            if recording:
                stop_recording(current_vw, audio_handle)
                window['start_rec'].update(disabled=False)
                window['stop_rec'].update(disabled=True)
        if event == 'start_stream':
            url = values.get('rtmp_url') or ''
            if url.strip():
                start_streaming(url.strip())
                window['start_stream'].update(disabled=True)
                window['stop_stream'].update(disabled=False)
            else:
                window['status'].update("Please enter RTMP URL before starting stream")
        if event == 'stop_stream':
            stop_streaming()
            window['start_stream'].update(disabled=False)
            window['stop_stream'].update(disabled=True)
        if event == 'ai_gen':
            # Basic placeholder: call external API if configured (not implemented here)
            window['status'].update("AI generation requires external API — configure in script")
        if event == 'snapshot':
            window.write_event_value('DO_SNAPSHOT', None)
        if event == 'auto_snap_en':
            pass
        if event == 'AUTO_SNAPSHOT':
            window.write_event_value('DO_SNAPSHOT', None)

        # fetch processed result if available
        try:
            proc_frame, mask = result_queue.get_nowait()
        except queue.Empty:
            proc_frame = None
            mask = None

        if proc_frame is not None:
            # apply background based on selected mode in UI
            mode = values.get('mode', 'None')
            blur_strength = int(values.get('blur_slider', 0))
            bright = int(values.get('bright_slider', 0))
            contrast = float(values.get('contrast_slider', 100)) / 100.0
            beautify_flag = values.get('beautify', False)
            use_adv = values.get('adv_matte', False)
            # update state for threads if necessary
            state['beautify'] = beautify_flag
            state['use_advanced_matte'] = use_adv and ADVANCED_MATTING_AVAILABLE

            frame_display = proc_frame.copy()
            # apply brightness/contrast
            if bright != 0 or contrast != 1.0:
                frame_display = adjust_brightness_contrast(frame_display, brightness=bright, contrast=contrast)

            out = frame_display.copy()
            if mode == 'Image' and bg_images:
                sel_idx = 0
                # choose first image for now (you can expand UI to pick)
                _, bg_img = bg_images[sel_idx]
                out = apply_virtual_bg(frame_display, mask, bg_img)
            elif mode == 'Blur':
                out = blur_background(frame_display, mask, strength=blur_strength)
            elif mode == 'Solid Color':
                out = color_background(frame_display, mask, (0,255,0))
            elif mode == 'Video' and bg_videos:
                # basic video bg: play first video file via VideoCapture
                bg_cap = cv2.VideoCapture(bg_videos[0])
                retb, bgf = bg_cap.read()
                if retb:
                    bgf = cv2.resize(bgf, (FRAME_W, FRAME_H))
                    out = apply_virtual_bg(frame_display, mask, bgf)
                bg_cap.release()
            else:
                out = frame_display

            # show grid / overlays if needed (not full UI)
            # send to preview in GUI
            imgbytes = cv2.imencode('.png', out)[1].tobytes()
            window['preview'].update(data=imgbytes)

            # virtual camera output
            if vcam_active and vcam is not None:
                try:
                    vcam.send(out)
                    vcam.sleep_until_next_frame()
                except Exception as e:
                    window['status'].update(f"Virtual cam error: {e}")

            # recording: write to video writer
            if recording and current_vw is not None:
                try:
                    current_vw.write(out)
                except Exception as e:
                    window['status'].update(f"Recording write error: {e}")

            # streaming: feed raw frames to ffmpeg stdin
            if streaming and ffmpeg_proc:
                try:
                    # write raw bgr24 bytes
                    ffmpeg_proc.stdin.write(out.tobytes())
                except Exception as e:
                    window['status'].update(f"Stream pipe error: {e}")

        # snapshot handling
        if event == 'DO_SNAPSHOT':
            # save last displayed frame to outputs
            if proc_frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join(OUT_DIR, f"snapshot_{timestamp}.png")
                cv2.imwrite(snap_path, out)
                window['status'].update(f"Saved snapshot: {snap_path}")

        # small sleep to avoid UI freeze
        time.sleep(0.001)

except Exception as e:
    traceback.print_exc()
finally:
    running.clear()
    time.sleep(0.3)
    try:
        cap.release()
    except:
        pass
    try:
        if vcam_active and vcam:
            vcam.close()
    except:
        pass
    window.close()
    print("Exiting app.")

