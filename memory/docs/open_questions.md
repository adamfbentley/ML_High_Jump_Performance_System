# Open Questions

Updated 2026-06-13 after stationary foot-contact verification.

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
- ~~Should contact detection key on ankle height?~~ No. The ankle/malleolus
  can remain above the ground band during plantarflexed toe-off. Use the lowest
  heel/forefoot marker per foot, with ankle fallback only for reduced skeletons.
- ~~Should peak-CoM anchor review use global `argmax(com_pos[:,1])` even when
  it precedes the launch-velocity peak?~~ No. `_windowed_peak_com_frame`
  keeps the global CoM apex when physically consistent, but searches after the
  vy peak when early approach-stride Y contamination wins the global argmax.

## Also closed (2026-05-05)

- ~~Why does anatomical mpp underestimate vh by ~0.9-1.5 m/s even on the
  tripod control? Phase 9d apparatus-anchored mpp is the test.~~ Overtaken by
  the gravity-mpp idea (universal across sports, no apparatus needed). Then
  empirically resolved by testing gravity-mpp on Athlete A's panned footage and
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
- Do the available stationary sets' camera placements provide
  enough approach-direction coverage for training-grade takeoff velocity?
  The newer trio now passes the implemented report gates, but collect a closer
  60 fps session before personal fine-tuning to get sample size and a held-out
  subset.
- Is one tripod side-view sufficient for the run-up direction velocity, or
  do we need an end-of-runway camera to capture the approach-direction
  component that side-view projects away? The Athlete A_takeoff-stationary
  clips (camera pointed roughly along the runway, not side-on) proved this
  matters: vh collapsed to near zero because the run-up is in the depth
  direction, which monocular cannot recover.
- Are there any panned clips worth retaining for *relative* analysis (stride
  rhythm, joint angles, body alignment) even though they cannot give
  training-grade translational metrics?

## Closed (2026-06-03 prep hardening)

- ~~What launch-velocity floor should anchor review require?~~ Use
  `vy >= 2.0 m/s`; a merely positive value can admit a weak stride near apex.
- ~~How should admitted-only caching be enforced?~~ Save `.npz` only for
  `training_grade` clips, record every decision in ignored local
  `_admission_manifest.json`, and refuse legacy mixed caches at fine-tune load.
- ~~Where should fixed-camera confirmation live?~~ Require explicit CLI
  operator confirmation and publish it in both the report and local admission
  manifest before emitting the `stationary_camera` source.
