#!/usr/bin/env python3
"""
virtualbg_obs_style.py
OBS-style minimal GUI + all enhancements (CPU-friendly).
Place next to folders: backgrounds/, bg_videos/, outputs/, models/
"""
import os, glob, time, json, threading, queue, subprocess, shutil, traceback
from datetime import datetime
import cv2
import numpy as np
import PySimpleGUI as sg
from PIL import Image, ImageSequence
import mediapipe as mp

# Optional libs
try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
    HAVE_PYVIRT = True
except Exception:
    HAVE_PYVIRT = False

try:
    import sounddevice as sd
    from scipy.io import wavfile
    HAVE_SOUND = True
except Exception:
    HAVE_SOUND = False

# ---------------- Settings ----------------
FRAME_W, FRAME_H = 1280, 720      # processing resolution
PREVIEW_W, PREVIEW_H = 960, 540   # display preview resolution
FPS = 20
ROOT_BG = "backgrounds"
BG_VID_DIR = "bg_videos"
OUT_DIR = "outputs"
CONFIG_FILE = "virtualbg_config.json"
THUMB_W, THUMB_H = 160, 90
# audio settings
AUDIO_SR = 44100

os.makedirs(ROOT_BG, exist_ok=True)
os.makedirs(BG_VID_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Utilities ----------------
def now_ts(): return datetime.now().strftime("%Y%m%d_%H%M%S")
def clamp01(a): return np.clip(a, 0.0, 1.0)

# ---------------- Config ----------------
default_cfg = {
    "last_category": "All",
    "last_idx": None,
    "mode": "None",
    "slideshow": False,
    "slide_interval": 5,
    "autozoom": False,
    "beautify": False,
}
if os.path.exists(CONFIG_FILE):
    try:
        cfg = {**default_cfg, **json.load(open(CONFIG_FILE))}
    except:
        cfg = default_cfg.copy()
else:
    cfg = default_cfg.copy()

def save_cfg():
    try:
        json.dump(cfg, open(CONFIG_FILE,"w"), indent=2)
    except: pass

# ---------------- Background loading ----------------
def list_categories():
    cats = ["All"]
    for entry in sorted(os.listdir(ROOT_BG)):
        p = os.path.join(ROOT_BG, entry)
        if os.path.isdir(p):
            cats.append(entry)
    return cats

def load_bg_list(category="All"):
    items = []
    if category == "All":
        # search ROOT_BG and subfolders
        for root, _, files in os.walk(ROOT_BG):
            for f in sorted(files):
                if f.startswith('.'): continue
                p = os.path.join(root, f)
                items.append(p)
    else:
        p = os.path.join(ROOT_BG, category)
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.startswith('.'): continue
                items.append(os.path.join(p,f))
    # build structured list
    result = []
    for p in items:
        name = os.path.relpath(p, ROOT_BG)
        ext = os.path.splitext(p)[1].lower()
        try:
            if ext == ".png":
                pil = Image.open(p).convert("RGBA")
                arr = np.array(pil)
                bgr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGBA2BGR)
                alpha = arr[..., 3].astype(np.float32)/255.0
                bgr = cv2.resize(bgr, (FRAME_W, FRAME_H))
                alpha = cv2.resize(alpha, (FRAME_W, FRAME_H))
                result.append((name, "img_alpha", (bgr, alpha)))
            elif ext == ".gif":
                frames = []
                for f in ImageSequence.Iterator(Image.open(p)):
                    fr = f.convert("RGB").resize((FRAME_W, FRAME_H), Image.LANCZOS)
                    frames.append(cv2.cvtColor(np.array(fr), cv2.COLOR_RGB2BGR))
                if frames:
                    result.append((name, "gif", frames))
            elif ext in (".jpg",".jpeg",".bmp",".webp"):
                img = cv2.imread(p)
                if img is None: continue
                img = cv2.resize(img, (FRAME_W, FRAME_H))
                result.append((name, "img", img))
        except Exception as e:
            print("bg load err", p, e)
    return result

# ---------------- Mediapipe models ----------------
mp_selfie = mp.solutions.selfie_segmentation
seg = mp_selfie.SelfieSegmentation(model_selection=1)
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# ---------------- Threaded pipeline ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

capture_q = queue.Queue(maxsize=4)
seg_q = queue.Queue(maxsize=4)
out_q = queue.Queue(maxsize=4)
running = threading.Event()
running.set()

def capture_loop():
    while running.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01); continue
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        if not capture_q.full():
            capture_q.put(frame)

def seg_loop():
    while running.is_set():
        try:
            frame = capture_q.get(timeout=0.5)
        except queue.Empty:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = seg.process(rgb)
        if res and res.segmentation_mask is not None:
            mask = res.segmentation_mask.astype(np.float32)
        else:
            mask = np.ones((FRAME_H, FRAME_W), dtype=np.float32)
        if not seg_q.full():
            seg_q.put((frame, mask))

# simple mask cleaner
def clean_mask(mask, strength=1):
    m = (mask*255).astype(np.uint8)
    k = 3 + strength*2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = cv2.GaussianBlur(m, (7 + strength*4, 7 + strength*4), 0)
    return (m.astype(np.float32)/255.0)

def apply_bg(frame, mask, bg):
    mask3 = np.repeat(mask[:,:,None],3,axis=2)
    fg = frame.astype(np.float32) * mask3
    bgf = bg.astype(np.float32) * (1.0 - mask3)
    return np.clip(fg + bgf,0,255).astype(np.uint8)

def apply_alpha_bg(frame, mask, bg, alpha):
    # combine segmentation mask with bg alpha: where alpha high, background shows through more
    combined = clamp01(mask * (1.0 - alpha))
    return apply_bg(frame, combined, bg)

def hybrid_blur(frame, mask, strength=1):
    kern = 5 + strength*4
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern,kern))
    inner = cv2.erode((mask*255).astype(np.uint8), kernel, iterations=1).astype(np.float32)/255.0
    inner_bg = cv2.GaussianBlur(frame, (15,15), 0)
    outer_bg = cv2.blur(frame, (25+strength*20, 25+strength*20))
    mask3_in = np.repeat(inner[:,:,None],3,axis=2)
    mask3_out = 1.0 - mask3_in
    composed = (frame.astype(np.float32) * mask3_in + inner_bg.astype(np.float32)*(mask3_out*0.5) + outer_bg.astype(np.float32)*(mask3_out*0.5)).astype(np.uint8)
    feather = cv2.GaussianBlur((mask*255).astype(np.uint8), (9,9),0).astype(np.float32)/255.0
    return apply_bg(frame, feather, composed)

def beautify_face(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fd = face_detector.process(rgb)
    if not fd.detections:
        return frame
    out = frame.copy()
    for d in fd.detections:
        bb = d.location_data.relative_bounding_box
        x = int(bb.xmin * FRAME_W); y = int(bb.ymin * FRAME_H)
        w = int(bb.width * FRAME_W); h = int(bb.height * FRAME_H)
        x1 = max(0, x-10); y1 = max(0, y-10); x2 = min(FRAME_W, x+w+10); y2 = min(FRAME_H, y+h+10)
        roi = out[y1:y2, x1:x2]
        if roi.size==0: continue
        sm = cv2.bilateralFilter(roi, d=9, sigmaColor=75, sigmaSpace=75)
        out[y1:y2, x1:x2] = sm
    return out

def detect_face_center(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fd = face_detector.process(rgb)
    if not fd.detections:
        return None
    d = fd.detections[0]
    bb = d.location_data.relative_bounding_box
    x = int((bb.xmin + bb.width/2.0)*FRAME_W)
    y = int((bb.ymin + bb.height/2.0)*FRAME_H)
    return (x,y)

def auto_zoom_crop(frame, center, scale=1.8):
    cx,cy = center
    w = int(FRAME_W/scale); h = int(FRAME_H/scale)
    x1 = max(0, cx - w//2); y1 = max(0, cy - h//2)
    x2 = min(FRAME_W, x1 + w); y2 = min(FRAME_H, y1 + h)
    crop = frame[y1:y2, x1:x2]
    if crop.size==0: return frame
    return cv2.resize(crop, (FRAME_W, FRAME_H))

# ---------------- post thread ----------------
def postproc_loop(state):
    last_time = time.time()
    gif_counters = {}
    while running.is_set():
        try:
            frame, mask = seg_q.get(timeout=0.5)
        except queue.Empty:
            continue
        mask_clean = clean_mask(mask, strength=1)
        out = frame.copy()
        mode = state['mode']
        sel = state.get('selected_idx', None)
        bg_list = state.get('bg_list', [])
        # slideshow tick
        if state.get('slideshow', False) and len(bg_list)>0:
            if time.time() - state.get('slide_last', 0) >= state.get('slide_interval',5):
                state['slide_last'] = time.time()
                state['selected_idx'] = ((state.get('selected_idx') or 0) + 1) % len(bg_list)

        if mode == 'Image' and sel is not None and 0<=sel < len(bg_list):
            name, typ, data = bg_list[sel]
            if typ == 'img':
                out = apply_bg(frame, mask_clean, data)
            elif typ == 'img_alpha':
                out = apply_alpha_bg(frame, mask_clean, data[0], data[1])
            elif typ == 'gif':
                # rotate simple based on time
                idx = int(time.time()*10) % len(data)
                out = apply_bg(frame, mask_clean, data[idx])
        elif mode == 'Blur':
            out = hybrid_blur(frame, mask_clean, strength=state.get('blur_strength',1))
        elif mode == 'Color':
            out = apply_bg(frame, mask_clean, np.full_like(frame, (0,255,0)))
        else:
            out = frame.copy()

        # auto-zoom
        if state.get('autozoom', False):
            fc = detect_face_center(out)
            if fc:
                out = auto_zoom_crop(out, fc, scale=1.8)

        # beautify
        if state.get('beautify', False):
            out = beautify_face(out)

        # FPS overlay
        now = time.time()
        fps = 1.0/(now-last_time) if last_time else 0.0
        last_time = now
        cv2.putText(out, f"{int(fps)} FPS", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

        if not out_q.full():
            out_q.put(out)

# ---------------- Start threads ----------------
state = {
    'mode': cfg.get('mode','None'),
    'selected_idx': cfg.get('last_idx', None),
    'bg_list': load_bg_list(cfg.get('last_category','All')),
    'slideshow': cfg.get('slideshow', False),
    'slide_interval': cfg.get('slide_interval', 5),
    'slide_last': time.time(),
    'blur_strength': 1,
    'autozoom': cfg.get('autozoom', False),
    'beautify': cfg.get('beautify', False),
}

t_cap = threading.Thread(target=capture_loop, daemon=True)
t_seg = threading.Thread(target=seg_loop, daemon=True)
t_post = threading.Thread(target=postproc_loop, args=(state,), daemon=True)
t_cap.start(); t_seg.start(); t_post.start()

# ---------------- Recording (video+audio) ----------------
recording = False
video_writer = None
audio_stream = None
wav_path = None
video_path = None
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if recording:
        audio_q.put(indata.copy())

def start_audio_recording(path_wav):
    try:
        sd.default.samplerate = AUDIO_SR
        sd.default.channels = 1
        stream = sd.InputStream(callback=audio_callback)
        stream.start()
        return stream
    except Exception as e:
        print("audio start failed", e)
        return None

def stop_audio_recording(stream, out_wav):
    try:
        # drain queue
        frames = []
        while not audio_q.empty():
            frames.append(audio_q.get())
        if frames:
            arr = np.concatenate(frames, axis=0)
            wavfile.write(out_wav, AUDIO_SR, arr)
    except Exception as e:
        print("audio stop failed", e)
    try:
        if stream:
            stream.stop(); stream.close()
    except: pass

# ---------------- GUI (OBS-style) ----------------
sg.theme("DarkBlue3")
categories = list_categories()
if cfg.get('last_category') not in categories:
    cfg['last_category'] = 'All'
selected_category = cfg.get('last_category','All')
bg_list = state['bg_list']

def make_thumb_bytes(entry):
    nm, typ, data = entry
    if typ=='img':
        img = data
    elif typ=='img_alpha':
        img = data[0]
    elif typ=='gif':
        img = data[0][0]
    else:
        img = np.zeros((FRAME_H,FRAME_W,3), dtype=np.uint8)
    th = cv2.resize(img, (THUMB_W, THUMB_H))
    ok, buf = cv2.imencode(".png", th)
    return buf.tobytes() if ok else None

def build_thumb_rows(bg_list):
    rows=[]; row=[]
    for i,entry in enumerate(bg_list):
        b = make_thumb_bytes(entry)
        key = f"TH_{i}"
        row.append(sg.Button(image_data=b, key=key, pad=(2,2)))
        if len(row)==4:
            rows.append(row); row=[]
    if row: rows.append(row)
    return rows

thumb_rows = build_thumb_rows(bg_list)

controls = [
    [sg.Text("VirtualBG Camera", font=("Helvetica",14))],
    [sg.Text("Category:"), sg.Combo(categories, default_value=selected_category, key="category", enable_events=True)],
    [sg.Button("Add Background"), sg.Button("Reload"), sg.Button("Save Screenshot")],
    [sg.Text("Mode:"), sg.Combo(["None","Image","Blur","Color"], default_value=state['mode'], key="mode", enable_events=True)],
    [sg.Checkbox("Slideshow", default=cfg.get('slideshow',False), key="slideshow"), sg.Text("Interval(s)"), sg.Input(str(cfg.get('slide_interval',5)), size=(5,1), key="slide_interval")],
    [sg.Checkbox("Auto-Zoom", default=cfg.get('autozoom',False), key="autozoom"), sg.Checkbox("Beautify", default=cfg.get('beautify',False), key="beautify")],
    [sg.Button("Start Virtual Camera"), sg.Button("Stop Virtual Camera", disabled=not HAVE_PYVIRT), sg.Button("Start Recording"), sg.Button("Stop Recording", disabled=True)],
    [sg.Text("", key="status", size=(60,2))]
]

layout = [
    [sg.Column([[sg.Image(size=(PREVIEW_W, PREVIEW_H), key="preview")]], element_justification='center'),
     sg.Column(controls)],
    [sg.Frame("Backgrounds", [[sg.Column(thumb_rows, scrollable=True, size=(700,300), key="thumb_col")]])],
    [sg.Exit()]
]

window = sg.Window("VirtualBG - OBS Style", layout, finalize=True, resizable=True)
selected_idx = state.get('selected_idx', None)
last_out = None
vcam = None
vcam_active = False

# highlight helper
def highlight(idx):
    for i in range(len(state['bg_list'])):
        k = f"TH_{i}"
        if k in window.AllKeysDict:
            if i==idx:
                window[k].update(button_color=('white','darkblue'))
            else:
                window[k].update(button_color=(None,None))

if selected_idx is not None:
    highlight(selected_idx)

# ---------------- Main UI loop ----------------
try:
    while True:
        event, values = window.read(timeout=50)
        if event in (sg.WIN_CLOSED, 'Exit'):
            break

        # Add Background
        if event == "Add Background":
            files = sg.popup_get_file("Select images (multiple allowed)", multiple_files=True, file_types=(("Images","imgs","*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.webp"),))
            if files:
                files = files.split(";")
                dest_dir = ROOT_BG if values.get("category","All")=="All" else os.path.join(ROOT_BG, values.get("category"))
                os.makedirs(dest_dir, exist_ok=True)
                for f in files:
                    try:
                        base = os.path.basename(f)
                        dst = os.path.join(dest_dir, base)
                        if os.path.exists(dst):
                            name,ext = os.path.splitext(base)
                            dst = os.path.join(dest_dir, f"{name}_{now_ts()}{ext}")
                        shutil.copy2(f, dst)
                    except Exception as e:
                        print("copy err", e)
                # reload
                state['bg_list'] = load_bg_list(values.get("category","All"))
                window['thumb_col'].update(build_thumb_rows(state['bg_list']))
                window['status'].update("Added and reloaded backgrounds")
            continue

        # Reload backgrounds or category change
        if event == "Reload" or event == "category":
            selected_category = values.get("category","All")
            state['bg_list'] = load_bg_list(selected_category)
            window['thumb_col'].update(build_thumb_rows(state['bg_list']))
            window['status'].update(f"Loaded {len(state['bg_list'])} backgrounds")
            cfg['last_category'] = selected_category
            save_cfg()
            continue

        # Save screenshot
        if event == "Save Screenshot":
            if last_out is not None:
                path = os.path.join(OUT_DIR, f"shot_{now_ts()}.png")
                cv2.imwrite(path, last_out)
                window['status'].update(f"Saved {path}")
            continue

        # Thumbnail click
        if isinstance(event, str) and event.startswith("TH_"):
            idx = int(event.split("_",1)[1])
            if 0 <= idx < len(state['bg_list']):
                state['selected_idx'] = idx
                highlight(idx)
                window['mode'].update('Image')
                window['status'].update(f"Selected: {state['bg_list'][idx][0]}")
                cfg['last_idx'] = idx; save_cfg()
            continue

        # Start Virtual Camera
        if event == "Start Virtual Camera":
            if not HAVE_PYVIRT:
                window['status'].update("pyvirtualcam not installed; virtual cam unavailable")
                continue
            try:
                vcam = pyvirtualcam.Camera(width=FRAME_W, height=FRAME_H, fps=FPS, fmt=PixelFormat.BGR)
                vcam_active = True
                window['status'].update("VirtualBG Camera started")
                window['Start Virtual Camera'].update(disabled=True)
                window['Stop Virtual Camera'].update(disabled=False)
            except Exception as e:
                window['status'].update(f"Virtual cam error: {e}")
            continue

        if event == "Stop Virtual Camera":
            try:
                if vcam:
                    vcam.close()
                vcam_active = False
                vcam = None
                window['status'].update("VirtualBG Camera stopped")
                window['Start Virtual Camera'].update(disabled=False)
                window['Stop Virtual Camera'].update(disabled=True)
            except Exception as e:
                window['status'].update(f"Stop vcam err: {e}")
            continue

        # Start Recording
        if event == "Start Recording":
            if recording:
                continue
            ts = now_ts()
            video_path = os.path.join(OUT_DIR, f"record_{ts}.mp4")
            wav_path  = os.path.join(OUT_DIR, f"audio_{ts}.wav")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (FRAME_W, FRAME_H))
            if HAVE_SOUND:
                audio_stream = start_audio_recording(wav_path)
            else:
                audio_stream = None
            recording = True
            window['Start Recording'].update(disabled=True)
            window['Stop Recording'].update(disabled=False)
            window['status'].update("Recording started")
            continue

        if event == "Stop Recording":
            if not recording:
                continue
            recording = False
            # stop audio and write wav
            if HAVE_SOUND:
                stop_audio_recording(audio_stream, wav_path)
            # close writer
            try:
                video_writer.release()
            except:
                pass
            # mux if wav present
            if HAVE_SOUND and os.path.exists(wav_path):
                final = video_path.replace(".mp4","_final.mp4")
                cmd = ["ffmpeg","-y","-i", video_path, "-i", wav_path, "-c:v","copy","-c:a","aac","-b:a","192k", final]
                try:
                    subprocess.run(cmd, check=True)
                    window['status'].update(f"Saved {final}")
                except Exception as e:
                    window['status'].update(f"Saved video only: {video_path} (ffmpeg mux failed)")
            else:
                window['status'].update(f"Saved video: {video_path}")
            window['Start Recording'].update(disabled=False)
            window['Stop Recording'].update(disabled=True)
            continue

        # UI toggles
        if event == "mode":
            state['mode'] = values.get('mode','None')
        if event == "slideshow":
            state['slideshow'] = values.get('slideshow', False)
            cfg['slideshow'] = state['slideshow']; save_cfg()
        if event == "slide_interval":
            try:
                state['slide_interval'] = int(values.get('slide_interval',5)); cfg['slide_interval']=state['slide_interval']; save_cfg()
            except: pass
        if event == "autozoom":
            state['autozoom'] = values.get('autozoom', False); cfg['autozoom']=state['autozoom']; save_cfg()
        if event == "beautify":
            state['beautify'] = values.get('beautify', False); cfg['beautify']=state['beautify']; save_cfg()

        # get processed frame
        try:
            out = out_q.get_nowait()
            last_out = out
        except queue.Empty:
            out = last_out

        if out is not None:
            # preview
            prev = cv2.resize(out, (PREVIEW_W, PREVIEW_H))
            ok, buf = cv2.imencode(".png", prev)
            if ok:
                window['preview'].update(data=buf.tobytes())

            # send to virtual cam
            if vcam_active and vcam:
                try:
                    vcam.send(out)
                    vcam.sleep_until_next_frame()
                except Exception:
                    pass

            # write recording frame
            if recording and 'video_writer' in locals() and video_writer is not None:
                try:
                    video_writer.write(out)
                except Exception:
                    pass

except Exception as e:
    traceback.print_exc()
finally:
    running.clear()
    time.sleep(0.2)
    try:
        cap.release()
    except: pass
    try:
        if vcam_active and vcam:
            vcam.close()
    except: pass
    # finalize recording if mid-record
    if 'video_writer' in locals() and video_writer is not None:
        try: video_writer.release()
        except: pass
    save_cfg()
    window.close()
    print("Exited cleanly.")

