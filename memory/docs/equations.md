# Equations

Core equations used across the project:

```text
g_vec = [0, -9.81, 0]
F_GRF = m * (a_CoM - g_vec)
h_max = h0 + vy^2 / (2 * 9.81)
takeoff_angle_deg = atan2(vy, sqrt(vx^2 + vz^2)) converted to degrees
```

Phase 9b report takeoff frame:

```text
takeoff_frame = final frame of final detected ankle-ground contact before flight
fallback = argmax(vy), only when no contact interval is detected
```

Current caution: takeoff angle is not training-grade until horizontal velocity
is scene-fixed. Historical panned footage is retained for relative technique
analysis only; stationary footage is required for Phase 10 inputs.

Historical Phase 9c scene homography:

```text
upright_separation_m = 4.02
scene_origin = midpoint(left_upright_base, right_upright_base)
scene_x_axis = right_upright_base - left_upright_base
scene_y_axis = vertical up

image_px = [u, v, 1]^T
scene_m = H_t * image_px
x_scene = scene_m[0] / scene_m[2]
y_scene = scene_m[1] / scene_m[2]

left_base_px  -> [-upright_separation_m / 2, 0]
right_base_px -> [ upright_separation_m / 2, 0]
left_top_px   -> [-upright_separation_m / 2, bar_height_m]
right_top_px  -> [ upright_separation_m / 2, bar_height_m]
```

This was a panned-footage rescue experiment. It remains implemented but is not
part of the stationary-camera admission path.

Historical Phase 9e gravity-mpp experiment:

```text
During true flight:
acceleration_px_s2 = magnitude(second_derivative(CoM_px, time))
gravity_mpp = 9.81 / acceleration_px_s2

quality gates:
downward_acceleration_fraction >= 0.65
horizontal_acceleration_fraction <= 0.45
flight_parabola_y_r_squared >= 0.75
```

`gravity_mpp` is correct on synthetic projectile data but was corrupted by
vertical camera tilt in handheld footage. It is retained as a historical
experiment and is not part of the stationary-camera admission path.

Current stationary-camera assumption:

```text
camera_pan = camera_tilt = camera_zoom = 0 during the attempt
image_frame_x is scene-fixed
horizontal_source = stationary_camera
```

Confirmed fixed-camera clips use the direct Phase 9a anatomical production path.
This removes camera-motion contamination, not fixed single-camera projection or
anatomical-scale limitations.
