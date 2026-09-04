from acies import (
    APCController, APCConfig, APCResult,
    HardwareProfile, ActionType,
    BeliefState, ClarityLearner, SafetyLayer,
    Conviction, ChangePointDetector,
)

print("ACIES import OK")
print(f"Version: {__import__('acies').__version__}")
print(f"Actions: {[a.name for a in __import__('acies').build_standard_actions()]}")
print(f"Hardware profiles: jetson_orin, rpi5, desktop_gpu, edge_tpu")
