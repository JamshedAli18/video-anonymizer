# src/track.py — lightweight IoU tracker with constant-velocity prediction (fills detector misses)
import numpy as np


def iou_matrix(a, b):
    """a: (N,4), b: (M,4) as x1,y1,x2,y2 -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


class Track:
    def __init__(self, box, tid):
        self.box, self.vel, self.id = box.astype(np.float32), np.zeros(4, np.float32), tid
        self.missed = 0


class IoUTracker:
    def __init__(self, iou_thr=0.3, max_age=10):
        self.iou_thr, self.max_age = iou_thr, max_age
        self.tracks, self.next_id = [], 0

    def update(self, dets_xywh):
        """dets: (N,5) x,y,w,h,score. Returns list of (box_xyxy, track_id, is_predicted)."""
        dets = dets_xywh[:, :4].copy()
        dets[:, 2:] += dets[:, :2]                                   # -> x1,y1,x2,y2

        preds = np.array([t.box + t.vel for t in self.tracks], np.float32).reshape(-1, 4)
        ious = iou_matrix(preds, dets)
        matched_t, matched_d = set(), set()
        for flat in np.argsort(-ious, axis=None):                    # greedy: best IoU first
            ti, di = np.unravel_index(flat, ious.shape)
            if ious[ti, di] < self.iou_thr:
                break
            if ti in matched_t or di in matched_d:
                continue
            t = self.tracks[ti]
            t.vel = 0.5 * t.vel + 0.5 * (dets[di] - t.box)
            t.box, t.missed = dets[di], 0
            matched_t.add(ti); matched_d.add(di)

        for ti, t in enumerate(self.tracks):                         # coast unmatched tracks
            if ti not in matched_t:
                t.box, t.missed = preds[ti], t.missed + 1
        for di in range(len(dets)):                                  # start new tracks
            if di not in matched_d:
                self.tracks.append(Track(dets[di], self.next_id)); self.next_id += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]
        return [(t.box.copy(), t.id, t.missed > 0) for t in self.tracks]