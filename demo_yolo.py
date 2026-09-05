"""
ACIES × YOLO — Adaptive Object Detection with Data Collection

Runs YOLO with ACIES on webcam/video/images and outputs:
- Real-time overlay with metrics
- JSON log of every frame (resolution, cost, detections, abstention)
- Summary report at the end

Usage:
    python demo_yolo.py                    # webcam
    python demo_yolo.py --source video.mp4 # video file
    python demo_yolo.py --source image.jpg # single image
    python demo_yolo.py --frames 200       # limit to 200 frames
"""

import sys
import os
import json
import time
import argparse
import random
from datetime import datetime

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_VIDEOIO_PRIORITY_BACKEND"] = "1"

try:
    import cv2
except ImportError:
    print("pip install opencv-python")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("pip install ultralytics")
    sys.exit(1)

from acies import APCController, APCConfig, HardwareProfile
from acies.multiclass import MultiClassBelief


COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]

RESOLUTIONS = {
    "64p": 64, "128p": 128, "224p": 224,
    "320p": 320, "512p": 512, "1024p": 1024,
}


class ACIESYOLO:
    """ACIES + YOLO with full data collection."""

    def __init__(self, model_size="n", confidence_threshold=0.85, output_dir="output"):
        self.yolo = YOLO(f"yolov8{model_size}.pt")

        self.config = APCConfig(
            confidence_threshold=confidence_threshold,
            max_cost_per_image=500,
            degradation_patience=3,
            abstention_confidence=0.55,
            max_steps=5,
            hardware=HardwareProfile.desktop_gpu(),
        )
        self.apc = APCController(self.config)

        # Multi-class image naming
        self.classifier = MultiClassBelief(
            class_names=[
                "person", "vehicle", "animal", "object", "indoor",
                "outdoor", "food", "furniture", "electronics", "other"
            ],
            prior_alpha=1.0,
        )
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Data collection
        self.frame_log = []
        self.frame_count = 0
        self.fps_history = []

    def process_frame(self, frame):
        """Process one frame and collect data."""
        start = time.time()
        self.frame_count += 1

        # Clarity function
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean() / 255.0
        contrast = gray.std() / 128.0
        edges = cv2.Canny(gray, 50, 150).mean() / 255.0

        def clarity_fn(action):
            base = min(brightness * 1.2, 1.0) * min(contrast * 1.5, 1.0)
            base *= (1.0 - edges * 0.3)
            if 'crop' in action.name:
                clarity = base * 0.85
            elif action.name == '1024p':
                clarity = base * 1.05
            else:
                clarity = base
            return min(max(clarity + random.uniform(-0.03, 0.03), 0.1), 0.99)

        # Run ACIES
        result = self.apc.run(true_class=1, clarity_fn=clarity_fn)

        # Determine resolution
        if result.abstained or result.degraded:
            chosen_res = 320
            action_name = "fallback"
        else:
            action_name = result.steps[-1].action.name
            chosen_res = RESOLUTIONS.get(action_name, 320)

        # Run YOLO
        h, w = frame.shape[:2]
        scale = chosen_res / max(w, h)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        detections = self.yolo(small, verbose=False)[0]
        boxes = detections.boxes

        # Classify image based on YOLO detections
        detected_classes = [COCO_CLASSES[int(c.cls[0])] for c in boxes]
        image_category = self._categorize_image(detected_classes, brightness, contrast)
        self.classifier.update(image_category, clarity=result.final_belief if not result.abstained else 0.3)
        prediction = self.classifier.prediction
        top3 = self.classifier.top_k(3)

        elapsed = time.time() - start
        self.fps_history.append(1.0 / max(elapsed, 0.001))

        # Collect frame data
        frame_data = {
            "frame": self.frame_count,
            "timestamp": datetime.now().isoformat(),
            "resolution_chosen": chosen_res,
            "action": action_name,
            "acies_cost": round(result.total_cost, 1),
            "acies_steps": result.n_steps,
            "acies_confidence": round(result.final_belief, 3),
            "abstained": result.abstained,
            "degraded": result.degraded,
            "budget_exceeded": result.cost_budget_exceeded,
            "avg_clarity": round(result.avg_clarity, 3),
            "n_detections": len(boxes),
            "detection_classes": detected_classes,
            "detection_confs": [round(float(c.conf[0]), 3) for c in boxes],
            "image_name": prediction,
            "image_name_confidence": round(self.classifier.confidence, 3),
            "image_name_top3": [(n, round(p, 3)) for n, p in top3],
            "frame_brightness": round(brightness, 3),
            "frame_contrast": round(contrast, 3),
            "frame_edges": round(edges, 3),
            "process_time_ms": round(elapsed * 1000, 1),
        }
        self.frame_log.append(frame_data)

        # Scale boxes for drawing
        scale_x = w / small.shape[1] if chosen_res != w else 1.0
        scale_y = h / small.shape[0] if chosen_res != w else 1.0

        return {
            "frame": frame, "boxes": boxes,
            "scale_x": scale_x, "scale_y": scale_y,
            "result": result, "chosen_res": chosen_res,
            "data": frame_data,
        }

    def _categorize_image(self, detected_classes, brightness, contrast):
        """Map YOLO detections + image stats to a scene category."""
        if not detected_classes:
            if brightness < 0.3:
                return "indoor"
            elif brightness > 0.7 and contrast > 0.3:
                return "outdoor"
            return "other"

        # Priority mapping
        priority = {
            "person": "person", "car": "vehicle", "truck": "vehicle",
            "bus": "vehicle", "motorcycle": "vehicle", "bicycle": "vehicle",
            "cat": "animal", "dog": "animal", "bird": "animal",
            "horse": "animal", "sheep": "animal", "cow": "animal",
            "chair": "furniture", "couch": "furniture", "bed": "furniture",
            "dining table": "furniture", "potted plant": "furniture",
            "tv": "electronics", "laptop": "electronics", "cell phone": "electronics",
            "keyboard": "electronics", "mouse": "electronics", "remote": "electronics",
            "bottle": "food", "cup": "food", "bowl": "food",
            "apple": "food", "banana": "food", "sandwich": "food",
            "pizza": "food", "cake": "food",
        }

        for cls in detected_classes:
            if cls in priority:
                return priority[cls]

        return "object"

    def draw(self, info):
        """Draw overlay with metrics."""
        frame = info["frame"].copy()
        h, w = frame.shape[:2]
        d = info["data"]

        # Draw detections
        for box in info["boxes"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = int(x1 * info["scale_x"]), int(y1 * info["scale_y"])
            x2, y2 = int(x2 * info["scale_x"]), int(y2 * info["scale_y"])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{COCO_CLASSES[cls]} {conf:.2f}"
            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Status bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        if d["abstained"]:
            status_color, status = (0, 0, 255), "ABSTAIN"
        elif d["degraded"]:
            status_color, status = (0, 165, 255), "DEGRADED"
        else:
            status_color, status = (0, 255, 0), "ACTIVE"

        avg_fps = sum(self.fps_history[-30:]) / min(len(self.fps_history), 30)

        # Image name line
        image_name = d["image_name"]
        name_conf = d["image_name_confidence"]
        top3 = d["image_name_top3"]
        top3_str = " | ".join([f"{n}({p:.0%})" for n, p in top3])

        lines = [
            f"IMAGE: {image_name.upper()} ({name_conf:.0%})  |  {status}  |  {d['resolution_chosen']}p",
            f"Top3: {top3_str}",
            f"Cost: {d['acies_cost']:.0f}/500  Steps: {d['acies_steps']}  "
            f"Conf: {d['acies_confidence']:.2f}  FPS: {avg_fps:.1f}",
            f"Objects: {d['n_detections']}  |  Brightness: {d['frame_brightness']:.2f}  "
            f"Contrast: {d['frame_contrast']:.2f}  Edges: {d['frame_edges']:.2f}",
        ]

        cv2.putText(frame, f"ACIES x YOLO", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 55 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return frame

    def save_report(self):
        """Save full report to JSON."""
        def make_serializable(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        report = {
            "session": {
                "timestamp": datetime.now().isoformat(),
                "total_frames": self.frame_count,
                "config": {
                    "confidence_threshold": self.config.confidence_threshold,
                    "max_cost_per_image": self.config.max_cost_per_image,
                    "degradation_patience": self.config.degradation_patience,
                },
            },
            "summary": make_serializable(self.apc.summary()),
            "frames": make_serializable(self.frame_log),
            "resolution_distribution": {},
            "detection_stats": {},
        }

        # Resolution distribution
        for f in self.frame_log:
            res = str(f["resolution_chosen"])
            report["resolution_distribution"][res] = \
                report["resolution_distribution"].get(res, 0) + 1

        # Detection stats
        all_classes = []
        for f in self.frame_log:
            all_classes.extend(f["detection_classes"])
        for cls in set(all_classes):
            report["detection_stats"][cls] = all_classes.count(cls)

        path = os.path.join(self.output_dir, "report.json")
        with open(path, "w") as fp:
            json.dump(report, fp, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("SESSION REPORT")
        print("=" * 60)
        print(f"  Frames: {self.frame_count}")
        print(f"  Avg FPS: {sum(self.fps_history)/len(self.fps_history):.1f}")

        s = self.apc.summary()
        print(f"  Avg ACIES cost: {s['avg_cost']:.1f}")
        print(f"  Abstained: {sum(1 for f in self.frame_log if f['abstained'])}")
        print(f"  Degraded: {sum(1 for f in self.frame_log if f['degraded'])}")
        print(f"  Budget exceeded: {sum(1 for f in self.frame_log if f['budget_exceeded'])}")
        print(f"  Total detections: {sum(f['n_detections'] for f in self.frame_log)}")

        print(f"\n  Resolution distribution:")
        for res, count in sorted(report["resolution_distribution"].items()):
            print(f"    {res}p: {count} frames ({count/self.frame_count*100:.0f}%)")

        if report["detection_stats"]:
            print(f"\n  Detected objects:")
            for cls, count in sorted(report["detection_stats"].items(), key=lambda x: -x[1]):
                print(f"    {cls}: {count}")

        # Image naming stats
        name_stats = {}
        for f in self.frame_log:
            name = f["image_name"]
            name_stats[name] = name_stats.get(name, 0) + 1
        if name_stats:
            print(f"\n  Image categories (named by ACIES):")
            for name, count in sorted(name_stats.items(), key=lambda x: -x[1]):
                print(f"    {name}: {count} frames ({count/self.frame_count*100:.0f}%)")

        print(f"\n  Full report saved to: {path}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--conf", type=float, default=0.85)
    parser.add_argument("--frames", type=int, default=0, help="Max frames (0=unlimited)")
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    demo = ACIESYOLO(model_size=args.model, confidence_threshold=args.conf)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open {source}")
        return

    print("ACIES x YOLO — Press 'q' to quit")
    print(f"Config: thr={args.conf}, model=yolov8{args.model}")
    print("-" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            info = demo.process_frame(frame)
            display = demo.draw(info)
            cv2.imshow("ACIES x YOLO", display)

            if args.frames > 0 and demo.frame_count >= args.frames:
                print(f"\nReached {args.frames} frames limit.")
                break

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        demo.save_report()


if __name__ == "__main__":
    main()
