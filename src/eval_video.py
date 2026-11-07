# src/eval_video.py — video leak-rate evaluation on CVAT "MOT 1.1" labels
import gc
import glob
import io
import json
import os
import time
import zipfile
from collections import defaultdict

import cv2
import numpy as np

from anonymize import expand
from detect import FaceDetector
from track import IoUTracker, iou_matrix

MODEL = "models/face_detection_yunet_2026may.onnx"
PAD, COVER_THR = 0.25, 0.9
MIN_FACE_H = 40                                   # labelling protocol: every face >= 40 px tall is labelled
EVAL_FRAMES = {"A": 300, "C": 300}                # labelled frame range per clip (0..N-1)
MODES = {
    "fast":     dict(scales=[640], thr=0.5),
    "balanced": dict(scales=[1024], thr=0.3),
    "max":      dict(scales=[640, 1600], thr=0.3),
}


def load_mot(zip_path):
    """CVAT MOT 1.1 -> {frame_idx (0-based): [(track_id, x1, y1, x2, y2), ...]}"""
    with zipfile.ZipFile(zip_path) as z:
        name = [n for n in z.namelist() if n.endswith("gt.txt")][0]
        text = z.read(name).decode().strip()
    gt = defaultdict(list)
    if not text:
        return gt
    rows = np.loadtxt(io.StringIO(text), delimiter=",", ndmin=2)
    for r in rows:
        if len(r) > 6 and r[6] == 0:              # ignored box
            continue
        f, tid, x, y, w, h = int(r[0]) - 1, int(r[1]), r[2], r[3], r[4], r[5]
        if h < MIN_FACE_H:
            continue
        gt[f].append((tid, x, y, x + w, y + h))
    return gt


def blur_mask(boxes, H, W):
    """Exactly the ellipses anonymize.py blurs."""
    m = np.zeros((H, W), np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = expand(b, PAD, W, H)
        w, h = x2 - x1, y2 - y1
        if w < 4 or h < 4:
            continue
        cv2.ellipse(m, (x1 + w // 2, y1 + h // 2), (w // 2, h // 2), 0, 0, 360, 1, -1)
    return m


def face_coverage(G, m):
    """Fraction of each face's ellipse (inscribed in its GT box) that is blurred. None = face off-frame."""
    H, W = m.shape
    out = []
    for x1, y1, x2, y2 in G.astype(int):
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        w, h = x2 - x1, y2 - y1
        if w < 2 or h < 2:
            out.append(None)
            continue
        e = np.zeros((h, w), np.uint8)
        cv2.ellipse(e, (w // 2, h // 2), (max(1, w // 2), max(1, h // 2)), 0, 0, 360, 1, -1)
        out.append(float((e & m[y1:y2, x1:x2]).sum() / max(int(e.sum()), 1)))
    return out


def longest_run(frames):
    best = run = 0
    prev = None
    for f in sorted(frames):
        run = run + 1 if prev is not None and f == prev + 1 else 1
        best, prev = max(best, run), f
    return best


def run(video, gt, mode, use_track, n_frames):
    cfg = MODES[mode]
    det = FaceDetector(MODEL, score_thr=cfg["thr"], scales=cfg["scales"])
    tracker = IoUTracker()
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    f, times = 0, []
    total = leaks = fp = 0
    leaked_frames, seen_tracks = defaultdict(list), set()
    while f < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]

        t0 = time.perf_counter()
        d = det.detect(frame)
        if use_track:
            boxes = np.array([b for b, _, _ in tracker.update(d)], np.float32).reshape(-1, 4)
        else:
            boxes = d[:, :4].copy(); boxes[:, 2:] += boxes[:, :2]
        times.append(time.perf_counter() - t0)

        g = gt.get(f, [])
        G = np.array([x[1:] for x in g], np.float32).reshape(-1, 4)
        cov = face_coverage(G, blur_mask(boxes, H, W))
        for (tid, *_), c in zip(g, cov):
            if c is None:
                continue
            total += 1
            seen_tracks.add(tid)
            if c < COVER_THR:
                leaks += 1
                leaked_frames[tid].append(f)

        # false blurs: only boxes in the labelled size range that overlap no labelled face
        if len(boxes):
            big = (boxes[:, 3] - boxes[:, 1]) >= 0.75 * MIN_FACE_H
            near = iou_matrix(boxes, G).max(1) if len(G) else np.zeros(len(boxes))
            fp += int((big & (near < 0.1)).sum())
        f += 1
    cap.release()
    del det, tracker

    longest = max((longest_run(v) for v in leaked_frames.values()), default=0)
    return dict(mode=mode, track=use_track, frames=f, fps=fps, face_frames=total, leak_frames=leaks,
                leak_rate=leaks / max(total, 1), tracks=len(seen_tracks), tracks_leaked=len(leaked_frames),
                longest_leak_frames=longest, longest_leak_s=longest / fps,
                fp_per_frame=fp / max(f, 1), median_ms=float(np.median(times) * 1000))


if __name__ == "__main__":
    zips = sorted(glob.glob("data/videos/labels/*.zip"))
    all_res = {}
    hdr = f"{'clip':<5}{'mode':<10}{'track':<7}{'leak %':>8}{'tracks leaked':>15}{'longest leak':>14}{'FP/frame':>10}{'ms':>7}"
    print(hdr); print("-" * len(hdr))
    for zp in zips:
        clip = os.path.splitext(os.path.basename(zp))[0]
        gt = load_mot(zp)
        n_frames = EVAL_FRAMES.get(clip, max(gt) + 1 if gt else 0)
        res = []
        for mode in MODES:
            for use_track in (False, True):
                r = run(f"data/videos/test/{clip}.mp4", gt, mode, use_track, n_frames)
                gc.collect()
                res.append(r)
                print(f"{clip:<5}{mode:<10}{'on' if use_track else 'off':<7}{r['leak_rate']*100:>7.2f}%"
                      f"{r['tracks_leaked']:>9}/{r['tracks']:<5}{r['longest_leak_s']:>12.2f}s"
                      f"{r['fp_per_frame']:>10.2f}{r['median_ms']:>7.1f}")
        all_res[clip] = res

    print("\nALL CLIPS (pooled)")
    for mode in MODES:
        for use_track in (False, True):
            rs = [r for res in all_res.values() for r in res if r["mode"] == mode and r["track"] == use_track]
            lf, ff = sum(r["leak_frames"] for r in rs), sum(r["face_frames"] for r in rs)
            print(f"{mode:<10}{'on' if use_track else 'off':<7} leak {lf/max(ff,1)*100:6.2f}%  "
                  f"({lf}/{ff} face-frames) | longest leak {max(r['longest_leak_s'] for r in rs):.2f}s")

    with open("results/video_eval.json", "w") as fh:
        json.dump(dict(protocol=dict(min_face_h=MIN_FACE_H, eval_frames=EVAL_FRAMES, pad=PAD,
                                     cover_thr=COVER_THR), clips=all_res), fh, indent=1)
    print("\nSaved results/video_eval.json")