# Privacy-Preserving Video Anonymisation

Real-time face anonymisation for video on a laptop CPU — detect, track and blur every face, with a
privacy-focused evaluation that measures **leaks** (faces left visible), not just detection accuracy.

![Before / after](web_assets/images/A_060_before.jpg)
![Before / after](web_assets/images/A_060_after.jpg)

## How it works

```
 video frame ─► YuNet face detector ─► IoU tracker ─► elliptical blur ─► anonymised frame
               (1 or 2 scales + NMS)  (keeps blurring   (25% padding,
                                       a lost face for   blur at 1/8 scale)
                                       up to 10 frames)
```

- **Detector:** [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (OpenCV Zoo, MIT), 0.23 MB ONNX model.
- **Multi-scale mode:** runs at 640 px and 1600 px and merges results — upscaling recovers tiny faces, the small scale keeps large faces inside YuNet's ~10–300 px range.
- **Tracker:** greedy IoU matching with constant-velocity prediction; covers frames where the detector misses a face.
- **Anonymisation:** blur / pixelate / solid, elliptical mask, 25% padding.

## Modes

| Mode | Setting | Use |
|---|---|---|
| Fast | 640 px, score ≥ 0.5 | Live video, real time |
| Balanced | 1024 px, score ≥ 0.3 | Near real time |
| Max privacy | 640 + 1600 px, score ≥ 0.3 | Recorded video, crowds, small faces |

## Results

All timings on a laptop CPU (Intel Core i5-1235U, 8 GB RAM, no GPU).

### WIDER FACE validation — stress test on hard images
3,226 images, 39,112 valid faces (39% are smaller than 16 px).
**Coverage** = face counts as protected if ≥ 90% of it lies inside the padded blur box.

| Mode | Recall (IoU ≥ 0.5) | Coverage | Coverage, faces < 16 px | False positives / image | ms / image |
|---|---|---|---|---|---|
| Fast (640, 0.5) | 0.438 | 0.490 | — | 0.83 | 22.6 |
| Balanced (1024, 0.3) | 0.657 | 0.752 | 0.570 | 4.82 | 52.9 |
| 1600 only, 0.3 (ablation) | 0.738 | 0.814 | 0.701 | 5.34 | 136.9 |
| **Max (640 + 1600, 0.3)** | **0.744** | **0.829** | **0.714** | 8.10 | 183.2 |

- Upscaling to 1600 px raises coverage of tiny faces from 0.57 to 0.70, but lowers coverage of faces > 128 px
  (0.953 → 0.923) because they exceed the detector's range; adding the 640 px pass fixes this (0.968).

### Video leak test — own labelled clips
2 Pexels clips (720p, 300 frames each, 6 people, 1,800 face-frames), every face ≥ 40 px labelled frame-by-frame in CVAT.
A face-frame **leaks** if < 90% of the face ellipse is blurred.

| Mode | Leak rate | Longest leak | False blurs / frame | ms / frame (detect + track) |
|---|---|---|---|---|
| Fast | 0.00% | 0 s | 0.00 | 13–24 |
| Balanced | 0.00% | 0 s | 0.03–0.15 | 33–35 |
| Max | 0.00% | 0 s | 0.06–0.69 | 109–157 |

**Takeaway:** for clear footage with faces ≥ 40 px, the fast real-time mode is sufficient. The heavier modes only
pay off for small faces and crowds, as the WIDER FACE stress test shows.

## Limitations

- Small faces (< 16 px) remain the main weakness: even max mode leaves ~29% of them uncovered on WIDER FACE.
- The video test set is small (2 clips) and was labelled by a single annotator; the detector never missed a face on
  these clips, so the benefit of tracking is not yet measured.
- False-positive counts on WIDER FACE are approximate (some detections may be real but unlabelled faces).
- Faces only — licence plates are not handled yet.

## Run it

```bash
pip install -r requirements.txt
python src/pipeline.py --source 0 --show                          # webcam
python src/pipeline.py --source input.mp4 --output out.mp4 --method blur
```

Reproduce the evaluation:

```bash
python src/eval_widerface.py        # needs WIDER FACE val in data/widerface/
python src/eval_video.py            # needs clips in data/videos/test/ and CVAT MOT 1.1 labels in data/videos/labels/
```

## Data & licences

- **Code:** MIT (see `LICENSE`).
- **YuNet model:** MIT, © Shiqi Yu — Wu, Peng & Yu, *YuNet: A Tiny Millisecond-level Face Detector*, Machine Intelligence Research, 2023.
- **WIDER FACE:** CC BY-NC-ND 4.0 — used for evaluation numbers only; no images redistributed.
  Yang et al., *WIDER FACE: A Face Detection Benchmark*, CVPR 2016.
- **Demo clips:** Pexels License — sources listed in `web_assets/data/credits.json`.

## Author

[JamshedAli18](https://github.com/JamshedAli18)