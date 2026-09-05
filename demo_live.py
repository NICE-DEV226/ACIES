"""
ACIES Live Demo — Webcam with adaptive perception.

Usage:
    python demo_live.py

Requirements:
    pip install opencv-python
    ACIES must be installed or in PYTHONPATH
"""

import sys
import os
import time
import random

# Fix Qt/Wayland issues on Linux — force X11
os.environ["QT_QPA_PLATFORM"] = "xcb"

try:
    import cv2
except ImportError:
    print("OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)

from acies import APCController, APCConfig


class LiveDemo:
    """Demo ACIES sur flux webcam temps réel."""

    def __init__(self):
        self.config = APCConfig(
            confidence_threshold=0.88,
            max_cost_per_image=400,
            degradation_patience=3,
            abstention_confidence=0.55,
            max_steps=5,
        )
        self.apc = APCController(self.config)
        self.cap = None
        self.fps_history = []
        self.frame_count = 0

    def start(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"Cannot open camera {camera_id}")
            return

        print("ACIES Live Demo — Press 'q' to quit")
        print(f"Config: thr={self.config.confidence_threshold}, "
              f"budget={self.config.max_cost_per_image}")
        print("-" * 50)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            start_time = time.time()

            # Simulate clarity based on frame analysis
            clarity_fn = self._make_clarity_fn(frame)

            # Run ACIES
            result = self.apc.run(true_class=1, clarity_fn=clarity_fn)

            elapsed = time.time() - start_time
            self.fps_history.append(1.0 / max(elapsed, 0.001))
            self.frame_count += 1

            # Draw overlay
            self._draw_overlay(frame, result, elapsed)

            cv2.imshow('ACIES Live Demo', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        self._print_summary()
        self.cap.release()
        cv2.destroyAllWindows()

    def _make_clarity_fn(self, frame):
        """Create clarity function based on frame analysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean() / 255.0
        contrast = gray.std() / 128.0

        def clarity_fn(action):
            # Simulate clarity based on frame properties
            base_clarity = min(brightness * 1.2, 1.0) * min(contrast * 1.5, 1.0)

            # Add action-dependent factor
            if 'crop' in action.name:
                clarity = base_clarity * 0.9
            elif action.name == '1024p':
                clarity = base_clarity * 1.1
            else:
                clarity = base_clarity

            return min(max(clarity + random.uniform(-0.05, 0.05), 0.1), 0.99)

        return clarity_fn

    def _draw_overlay(self, frame, result, elapsed):
        """Draw ACIES info on frame."""
        h, w = frame.shape[:2]

        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Decision
        if result.abstained:
            color = (0, 0, 255)  # Red
            decision_text = "ABSTAIN (I don't know)"
        elif result.degraded:
            color = (0, 165, 255)  # Orange
            decision_text = "DEGRADED (image too noisy)"
        else:
            color = (0, 255, 0)  # Green
            decision_text = f"Decision: {result.decision}"

        cv2.putText(frame, decision_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Metrics
        avg_fps = sum(self.fps_history[-30:]) / min(len(self.fps_history), 30)
        info_lines = [
            f"Cost: {result.total_cost:.0f}/{self.config.max_cost_per_image}  "
            f"Steps: {result.n_steps}  "
            f"Confidence: {result.final_belief:.2f}",
            f"FPS: {avg_fps:.1f}  "
            f"Budget exceeded: {result.cost_budget_exceeded}  "
            f"Clarity: {result.avg_clarity:.2f}",
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 55 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Action used
        if result.steps:
            last_action = result.steps[-1].action.name
            cv2.putText(frame, f"Action: {last_action}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _print_summary(self):
        print("\n" + "=" * 50)
        print("SESSION SUMMARY")
        print("=" * 50)
        summary = self.apc.summary()
        print(f"  Total frames: {self.frame_count}")
        print(f"  Avg cost: {summary['avg_cost']:.1f}")
        print(f"  Avg accuracy: {summary['avg_accuracy']:.2%}")
        print(f"  Avg latency: {summary['avg_latency_ms']:.1f}ms")
        print(f"  Safety violations: {summary['safety']['n_violations']}")
        print(f"  Emergency overrides: {summary['safety']['n_emergency']}")


if __name__ == "__main__":
    demo = LiveDemo()
    demo.start(camera_id=0)
