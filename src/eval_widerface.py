# src/eval_widerface.py — WIDER FACE val: recall by face size, privacy coverage, false positives, speed
import argparse
import json
import time

import cv2
import numpy as np
from tqdm import tqdm

from detect import FaceDetector
from track import iou_matrix

ROOT = "data/widerface"
MODEL = "models/face_detection_yunet_2026may.onnx"
PAD = 0.25            # same padding as anonymize.py
COVER_THR = 0.9       # face counts as protected if >= 90% of its area is inside one blur box
BUCKETS = [(0, 16, "<16px"), (16, 32, "16-32"), (32, 64, "32-64"), (64, 128, "64-128"), (128, 1e9, ">128")]


def load_gt(path):
    with open(path) as f:
        lines = [l.strip() for l in f]
    gt, i = [], 0
    while i < len(lines):
        name, n = lines[i], int(lines[i + 1])
        i += 2
        rows = [list(map(int, lines[i + k].split()[:10])) for k in range(max(n, 1))]
        i += max(n, 1)
        boxes = np.array([r[:4] for r in rows[:n]], np.float32).reshape(-1, 4)
        invalid = np.array([r[7] for r in rows[:n]], bool)
        gt.append((name, boxes, invalid))
    return gt


def coverage(g, p):
    """Fraction of each GT box area covered by its best single predicted box."""
    if len(g) == 0:
        return np.zeros(0)
    if len(p) == 0:
        return np.zeros(len(g))
    x1 = np.maximum(g[:, None, 0], p[None, :, 0]); y1 = np.maximum(g[:, None, 1], p[None, :, 1])
    x2 = np.minimum(g[:, None, 2], p[None, :, 2]); y2 = np.minimum(g[:, None, 3], p[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return (inter / (area_g[:, None] + 1e-7)).max(1)


def run_config(gt, scales, thr):
    label = "+".join(str(s) for s in scales)
    det = FaceDetector(MODEL, score_thr=thr, scales=scales)
    stats = {lab: [0, 0, 0] for _, _, lab in BUCKETS}          # total, detected (IoU>=0.5), covered
    fp, times = 0, []

    for name, boxes, invalid in tqdm(gt, desc=f"{label} thr {thr}", leave=False):
        img = cv2.imread(f"{ROOT}/WIDER_val/images/{name}")
        t0 = time.perf_counter()
        d = det.detect(img)
        times.append(time.perf_counter() - t0)

        pred = d[:, :4].copy(); pred[:, 2:] += pred[:, :2]
        g = boxes.copy(); g[:, 2:] += g[:, :2]
        valid = (~invalid) & (boxes[:, 2] > 0) & (boxes[:, 3] > 0)

        # one-to-one greedy matching at IoU >= 0.5
        ious = iou_matrix(g, pred)
        mg, mp = np.zeros(len(g), bool), np.zeros(len(pred), bool)
        for flat in np.argsort(-ious, axis=None):
            gi, pi = np.unravel_index(flat, ious.shape)
            if ious[gi, pi] < 0.5:
                break
            if mg[gi] or mp[pi]:
                continue
            mg[gi] = mp[pi] = True

        # privacy coverage with padded boxes (as actually blurred)
        pw, ph = pred[:, 2] - pred[:, 0], pred[:, 3] - pred[:, 1]
        padded = np.stack([pred[:, 0] - PAD * pw, pred[:, 1] - PAD * ph,
                           pred[:, 2] + PAD * pw, pred[:, 3] + PAD * ph], 1)
        cov = coverage(g, padded)

        # false positives: unmatched predictions that don't overlap any labelled face
        if len(pred):
            near_gt = (ious >= 0.3).any(0) if len(g) else np.zeros(len(pred), bool)
            fp += int((~mp & ~near_gt).sum())

        for k in np.where(valid)[0]:
            h = boxes[k, 3]
            for lo, hi, lab in BUCKETS:
                if lo <= h < hi:
                    s = stats[lab]
                    s[0] += 1; s[1] += int(mg[k]); s[2] += int(cov[k] >= COVER_THR)

    tot = sum(s[0] for s in stats.values())
    return dict(
        config=label, scales=scales, score_thr=thr, images=len(gt), faces=tot,
        recall={lab: s[1] / max(s[0], 1) for lab, s in stats.items()},
        coverage={lab: s[2] / max(s[0], 1) for lab, s in stats.items()},
        faces_per_bucket={lab: s[0] for lab, s in stats.items()},
        recall_all=sum(s[1] for s in stats.values()) / tot,
        coverage_all=sum(s[2] for s in stats.values()) / tot,
        fp_per_image=fp / len(gt),
        median_ms=float(np.median(times) * 1000),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="evaluate only first N images (quick test)")
    ap.add_argument("--out", default="results/widerface_eval_multiscale.json")
    args = ap.parse_args()

    gt = load_gt(f"{ROOT}/wider_face_split/wider_face_val_bbx_gt.txt")
    if args.limit:
        gt = gt[: args.limit]

    configs = [([1024], 0.3), ([1600], 0.3), ([640, 1600], 0.3)]
    results = [run_config(gt, sc, thr) for sc, thr in configs]

    labs = [lab for _, _, lab in BUCKETS]
    print(f"\nImages: {len(gt)} | valid faces: {results[0]['faces']}")
    print("Recall (IoU >= 0.5) by face height:")
    print(f"{'config':<18}" + "".join(f"{l:>8}" for l in labs) + f"{'recall':>8}{'cover':>8}{'FP/img':>8}{'ms':>7}")
    for r in results:
        print(f"{r['config']:>9} thr {r['score_thr']:<4}" +
              "".join(f"{r['recall'][l]:>8.3f}" for l in labs) +
              f"{r['recall_all']:>8.3f}{r['coverage_all']:>8.3f}{r['fp_per_image']:>8.2f}{r['median_ms']:>7.1f}")
    print("\nCoverage (>= 90% of face inside blur box) by face height:")
    for r in results:
        print(f"{r['config']:>9} thr {r['score_thr']:<4}" + "".join(f"{r['coverage'][l]:>8.3f}" for l in labs))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nSaved {args.out}")