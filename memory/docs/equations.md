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
is scene-fixed.

Phase 9c scene homography:

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

When a frame has fewer than four reliable apparatus anchors, Phase 9c falls
back to Phase 9a anatomical scale for that frame.
