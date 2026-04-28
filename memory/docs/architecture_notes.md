# Architecture Notes

Canonical architecture is tracked in `ARCHITECTURE.md`.

Current high-level pipeline:

```text
Video -> Pose Estimation -> Kinematics -> BiomechanicalSample -> PINN -> Optimiser
```

Phase 10 personal fine-tuning is blocked until translational metrics from video
are validated. The current architecture-level issue is recovering scene-fixed
horizontal motion from panned single-camera footage, or moving future data
capture to two-camera DLT.
