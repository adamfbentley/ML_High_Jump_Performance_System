# Open Questions

Updated 2026-05-05 after the stationary-footage decision.

## Closed

- ~~How should scene-fixed horizontal displacement be recovered from panned
  single-camera footage?~~ Egomotion (`src/pose_estimation/egomotion.py`)
  recovers most of it; remaining gap isolated to mpp scale, not pan.
- ~~Is a crossbar/upright homography sufficient for historical footage?~~
  No — the Hough auto-detector locks onto wrong vertical edges on these
  clips; rejected on every labelled clip. Hand-labelled scene anchors are
  the only viable apparatus reference for existing footage.
- ~~Future filming policy: tripod-only is the cheapest unblocker for any
  clip where automated calibration fails. Should this be made the default
  for all new sessions, regardless of Phase 9d outcome?~~ Yes. Stationary
  footage is now the default requirement for training-grade physics and
  optimisation. Handheld footage remains useful for exploratory analysis,
  detector development, and relative technique review only.

## Also closed (2026-05-05)

- ~~Why does anatomical mpp underestimate vh by ~0.9-1.5 m/s even on the
  tripod control? Phase 9d apparatus-anchored mpp is the test.~~ Overtaken by
  the gravity-mpp idea (universal across sports, no apparatus needed). Then
  empirically resolved by testing gravity-mpp on Imogen's panned footage and
  observing the failure mode: vertical camera tilt during flight corrupts
  the parabola fit. The right answer is stationary capture, not deeper
  recalibration on handheld clips.
- ~~Can the labelled upright separation (4.02 m) close the residual mpp gap
  to <0.3 m/s on the tripod control?~~ Did not get tested in isolation
  because gravity-mpp was tried first and the panned-footage failure mode
  already decided the policy question.
- ~~For panned clips where the apparatus is briefly visible, is a single
  takeoff-frame mpp value enough, or do we need a per-frame mpp track?~~
  Moot. Panned phone footage is no longer in the training pipeline.

## Open

- Should the peak-CoM target range be revised after reviewing bar-relative
  CoM results with the BMS PhD student?
- What stationary capture protocol gives the best accuracy/time trade-off
  for Imogen: one side/oblique tripod camera, two-camera DLT, or a staged
  progression from one fixed camera to two?
- On the first stationary capture session, does gravity-mpp converge with
  anatomical mpp to within ≤0.3 m/s? If yes, both are working and Phase 10
  fine-tuning can proceed. If no, the residual is a real anatomical bias
  worth diagnosing before fine-tune.
- Is one tripod side-view sufficient for the run-up direction velocity, or
  do we need an end-of-runway camera to capture the approach-direction
  component that side-view projects away?
- Are there any panned clips worth retaining for *relative* analysis (stride
  rhythm, joint angles, body alignment) even though they cannot give
  training-grade translational metrics?
