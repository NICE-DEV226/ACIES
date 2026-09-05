"""
ACIES × YOLO — Adaptive Object Detection

ACIES decides the resolution for each frame.
YOLO runs on the chosen resolution.
Result: same accuracy, lower cost.

Usage:
    python demo_yolo.py                    # webcam
    python demo_yolo.py --source video.mp4 # video file
    python demo_yolo.py --source image.jpg # single image
"""

import sys
import time
import argparse
import random

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


# YOLO class names
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

# Resolution presets (width)
RESOLUTIONS = {
    "64p": 64,
    "128p": 128,
    "224p": 224,
    "320p": 320,
    "512p": 512,
    "1024p": 1024,
}


class ACIESYOLO:
    """ACIES + YOLO adaptive object detection."""

    def __init__(self, model_size="n", confidence_threshold=0.85):
        print(f"Loading YOLOv8-{model_size}...")
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

        self.frame_count = 0
        self.total_detections = 0
        self.fps_history = []
        self.cost_history = []

    def process_frame(self, frame):
        """Process one frame with ACIES + YOLO."""
        start = time.time()
        self.frame_count += 1

        # ACIES decides resolution
        clarity_fn = self._make_clarity_fn(frame)
        result = self.apc.run(true_class=1, clarity_fn=clarity_fn)

        # Determine resolution from ACIES decision
        if result.abstained or result.degraded:
            # Fallback: use lowest resolution
            chosen_res = 320
            yolo_confidence = 0.0
        else:
            # Use last action's resolution
            last_action = result.steps[-1].action.name
            chosen_res = RESOLUTIONS.get(last_action, 320)
            yolo_confidence = result.final_belief

        # Run YOLO on chosen resolution
        h, w = frame.shape[:2]
        if chosen_res != w:
            scale = chosen_res / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            small_frame = cv2.resize(frame, (new_w, new_h))
        else:
            small_frame = frame

        detections = self.yolo(small_frame, verbose=False)[0]
        boxes = detections.boxes
        n_detections = len(boxes)
        self.total_detections += n_detections

        # Scale boxes back to original size
        if chosen_res != w:
            scale_x = w / small_frame.shape[1]
            scale_y = h / small_frame.shape[0]
        else:
            scale_x, scale_y = 1.0, 1.0

        elapsed = time.time() - start
        self.fps_history.append(1.0 / max(elapsed, 0.001))
        self.cost_history.append(result.total_cost)

        return {
            "frame": frame,
            "small_frame": small_frame,
            "boxes": boxes,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "result": result,
            "chosen_res": chosen_res,
            "yolo_confidence": yolo_confidence,
            "n_detections": n_detections,
            "elapsed": elapsed,
        }

    def _make_clarity_fn(self, frame):
        """Simulate clarity based on frame content."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean() / 255.0
        contrast = gray.std() / 128.0
        edges = cv2.Canny(gray, 50, 150).mean() / 255.0

        def clarity_fn(action):
            # More edges = harder = lower clarity
            base = min(brightness * 1.2, 1.0) * min(contrast * 1.5, 1.0)
            base *= (1.0 - edges * 0.3)  # Penalize complex scenes

            if 'crop' in action.name:
                clarity = base * 0.85
            elif action.name == '1024p':
                clarity = base * 1.05
            else:
                clarity = base

            return min(max(clarity + random.uniform(-0.03, 0.03), 0.1), 0.99)

        return clarity_fn

    def draw_results(self, info):
        """Draw detections and ACIES info on frame."""
        frame = info["frame"].copy()
        h, w = frame.shape[:2]

        # Draw YOLO detections
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

        # ACIES overlay
        result = info["result"]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Status
        if result.abstained:
            status_color = (0, 0, 255)
            status = "ABSTAIN"
        elif result.degraded:
            status_color = (0, 165, 255)
            status = "DEGRADED"
        else:
            status_color = (0, 255, 0)
            status = "ACTIVE"

        avg_fps = sum(self.fps_history[-30:]) / min(len(self.fps_history), 30)
        avg_cost = sum(self.cost_history[-30:]) / min(len(self.cost_history), 30)

        lines = [
            f"Status: {status}  Resolution: {info['chosen_res']}p  "
            f"Detections: {info['n_detections']}",
            f"ACIES cost: {result.total_cost:.0f}  Steps: {result.n_steps}  "
            f"Confidence: {result.final_belief:.2f}",
            f"FPS: {avg_fps:.1f}  Avg cost: {avg_cost:.0f}  "
            f"Frame: {self.frame_count}",
        ]

        cv2.putText(frame, f"ACIES × YOLO", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 55 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def print_summary(self):
        print("\n" + "=" * 50)
        print("SESSION SUMMARY")
        print("=" * 50)
        summary = self.apc.summary()
        print(f"  Frames processed: {self.frame_count}")
        print(f"  Total detections: {self.total_detections}")
        print(f"  Avg cost: {summary['avg_cost']:.1f}")
        print(f"  Avg accuracy: {summary['avg_accuracy']:.2%}")
        print(f"  Avg FPS: {sum(self.fps_history)/len(self.fps_history):.1f}")
        print(f"  Safety violations: {summary['safety']['n_violations']}")
        print(f"  Budget exceeded: {sum(1 for r in self.apc._run_history if r.cost_budget_exceeded)}")


def main():
    parser = argparse.ArgumentParser(description="ACIES × YOLO Adaptive Detection")
    parser.add_argument("--source", default="0", help="Camera id, video path, or image")
    parser.add_argument("--model", default="n", choices=["n", "s", "m", "l", "x"],
                        help="YOLO model size")
    parser.add_argument("--conf", type=float, default=0.85, help="ACIES confidence threshold")
    args = parser.parse_args()

    # Parse source
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    demo = ACIESYOLO(model_size=args.model, confidence_threshold=args.conf)

    if isinstance(source, int):
        # Webcam
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Cannot open camera {source}")
            return

        print("ACIES × YOLO — Press 'q' to quit")
        print("-" * 50)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            info = demo.process_frame(frame)
            display = demo.draw_results(info)
            cv2.imshow("ACIES × YOLO", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif source.endswith(('.jpg', '.jpeg', '.png')):
        # Single image
        frame = cv2.imread(source)
        if frame is None:
            print(f"Cannot read {source}")
            return

        info = demo.process_frame(frame)
        display = demo.draw_results(info)
        cv2.imwrite("result.jpg", display)
        print(f"Result saved to result.jpg")
        print(f"  Resolution chosen: {info['chosen_res']}p")
        print(f"  Detections: {info['n_detections']}")
        print(f"  ACIES cost: {info['result'].total_cost:.1f}")

    else:
        # Video file
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Cannot open {source}")
            return

        print(f"Processing {source}...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            info = demo.process_frame(frame)
            display = demo.draw_results(info)
            cv2.imshow("ACIES × YOLO", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    demo.print_summary()


if __name__ == "__main__":
    main()
