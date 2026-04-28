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
