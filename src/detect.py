# src/detect.py — YuNet face detector wrapper with optional multi-scale inference
import cv2
import numpy as np


class FaceDetector:
    """Returns faces as an (N, 5) array: x, y, w, h, score — in original frame coordinates.

    scales: list of target longest-side sizes. One value = single scale.
            Several values (e.g. [640, 1600]) = run at each size and merge with NMS.
            Sizes larger than the frame upscale it (helps tiny faces).
    One detector instance per scale keeps its own fixed input size (no reallocation every frame).
    """

    def __init__(self, model_path, score_thr=0.5, nms_thr=0.3, top_k=5000,
                 max_side=640, scales=None, merge_iou=0.4):
        self.model_path, self.nms_thr, self.top_k = model_path, nms_thr, top_k
        self.scales = scales or [max_side]
        self.score_thr, self.merge_iou = score_thr, merge_iou
        self.dets = {}                                   # scale -> [detector, input_size]

    def _get(self, side, size):
        entry = self.dets.get(side)
        if entry is None:
            det = cv2.FaceDetectorYN.create(self.model_path, "", size, self.score_thr, self.nms_thr, self.top_k)
            self.dets[side] = [det, size]
            return det
        if entry[1] != size:
            entry[0].setInputSize(size)
            entry[1] = size
        return entry[0]

    def _detect_once(self, frame, side):
        h, w = frame.shape[:2]
        scale = side / max(h, w)
        if abs(scale - 1) < 1e-3:
            img, scale = frame, 1.0
        else:
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            img = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=interp)

        # Pad to multiples of 32 (required by the dynamic-shape model); boxes are unaffected
        ih, iw = img.shape[:2]
        ph, pw = (32 - ih % 32) % 32, (32 - iw % 32) % 32
        if ph or pw:
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)

        det = self._get(side, (img.shape[1], img.shape[0]))
        _, faces = det.detect(img)
        if faces is None:
            return np.zeros((0, 5), np.float32)
        boxes = faces[:, :4] / scale
        scores = faces[:, 14:15]
        return np.hstack([boxes, scores]).astype(np.float32)

    def detect(self, frame):
        d = np.vstack([self._detect_once(frame, s) for s in self.scales])
        if len(self.scales) > 1 and len(d) > 1:
            keep = cv2.dnn.NMSBoxes(d[:, :4].tolist(), d[:, 4].tolist(), self.score_thr, self.merge_iou)
            d = d[np.array(keep).flatten()] if len(keep) else d[:0]
        return d