# Open Questions

Updated 2026-05-04 after the hand-label evaluation closed several earlier
threads.

## Closed

- ~~How should scene-fixed horizontal displacement be recovered from panned
  single-camera footage?~~ Egomotion (`src/pose_estimation/egomotion.py`)
  recovers most of it; remaining gap isolated to mpp scale, not pan.
- ~~Is a crossbar/upright homography sufficient for historical footage?~~
  No — the Hough auto-detector locks onto wrong vertical edges on these
  clips; rejected on every labelled clip. Hand-labelled scene anchors are
  the only viable apparatus reference for existing footage.

## Open

- **Why does anatomical mpp underestimate vh by ~0.9-1.5 m/s even on the
  tripod control?** Hypothesis: p95 of thigh/shank projection captures the
  closest-depth instance, but the takeoff-side leg is planted/tilted at
  toe-off and never presents a fully in-plane projection at the takeoff
  zone. Phase 9d (apparatus-anchored mpp at the takeoff frame) is the
  test.
- Can the labelled upright separation (4.02 m) close the residual mpp gap
  to <0.3 m/s on the tripod control? If yes, re-evaluate egomotion on the
  4 panned clips and consider Phase 10. If no, two-camera DLT or
  end-of-runway filming becomes the only path.
- Should the peak-CoM target range be revised after reviewing bar-relative
  CoM results with the BMS PhD student?
- For panned clips where the apparatus is briefly visible (typically 2-7
  hand-labelled frames per clip), is a single takeoff-frame mpp value
  enough, or do we need a per-frame mpp track interpolated from the
  labelled apparatus across the takeoff window?
- Future filming policy: tripod-only is the cheapest unblocker for any
  clip where automated calibration fails. Should this be made the default
  for all new sessions, regardless of Phase 9d outcome?
