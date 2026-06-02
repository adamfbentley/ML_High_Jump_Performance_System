# Architecture Notes

Canonical architecture is tracked in `ARCHITECTURE.md`.

Current high-level pipeline:

```text
Video -> Pose Estimation -> Kinematics -> BiomechanicalSample -> PINN -> Optimiser
```

Phase 10 personal fine-tuning remains blocked. The rescue path for historical
panned footage is closed: egomotion helps, but automatic scene anchors are
unreliable and vertical phone tilt corrupts gravity-mpp fits.

Stationary footage has been imported locally and five clips have run through
the direct anatomical production branch. The MediaPipe wrapper now preserves decoded
frame timing by retaining zero-visibility placeholders for missed detections
and using median decoded timestamp spacing for nominal fps. Two newer clips
pass the implemented report gates after adding windowed pose validity. One
newer clip still lacks a contact interval. Overlay review also shows that a
boolean contact flag can select an early approach stride after later tracking
drops out.

The egomotion, automatic scene-anchor, gravity-mpp, and hand-label branches are
historical panned-footage rescue infrastructure, not stationary admission
gates. Private fixed-camera pose overlays now exist for all five clips. The
takeoff-window check and explicit `stationary_camera` source are implemented,
but the next pass must add admitted-only sample caching, explicit fixed-camera
confirmation, and a stricter minimum launch threshold for the anchor review.
Collect a closer 60 fps session before personal fine-tuning. Two-camera DLT
remains the gold-standard progression. See
`memory/plans/stationary_footage_validation_plan.md`.
