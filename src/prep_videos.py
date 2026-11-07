# src/prep_videos.py — trim raw clips to 12 s, cap at 30 fps, resize to 1280 px long side
import glob
import os

import cv2

RAW, OUT = "data/videos/raw", "data/videos/test"
MAX_SECONDS, LONG_SIDE, MAX_FPS = 12, 1280, 30
os.makedirs(OUT, exist_ok=True)

for path in sorted(glob.glob(f"{RAW}/*.mp4")):
    name = os.path.basename(path)
    cap = cv2.VideoCapture(path)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps_in / MAX_FPS))            # 60 fps -> keep every 2nd frame
    fps_out = fps_in / step
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = LONG_SIDE / max(w, h)
    size = (round(w * s) // 2 * 2, round(h * s) // 2 * 2)

    writer = cv2.VideoWriter(f"{OUT}/{name}", cv2.VideoWriter_fourcc(*"mp4v"), fps_out, size)
    n_in, n_out, max_out = 0, 0, int(fps_out * MAX_SECONDS)
    while n_out < max_out:
        ok, frame = cap.read()
        if not ok:
            break
        if n_in % step == 0:
            writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
            n_out += 1
        n_in += 1
    cap.release(); writer.release()
    print(f"{name}: {w}x{h} @ {fps_in:.0f} fps -> {size[0]}x{size[1]} @ {fps_out:.0f} fps, "
          f"{n_out} frames ({n_out/fps_out:.1f}s)")