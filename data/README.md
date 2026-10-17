# Static IMU logs

Raw logs are gitignored -- they are large, device-specific, and not reproducible by
someone else running this repo. Record your own.

Expected files:

| File | Source | Used by |
|---|---|---|
| `bmi270_static.csv` | BMI270 breakout, raw registers, device on a bench, >= 8 h | `scripts/fit_allan.py` |
| `phone_static.csv` | Phone IMU, **uncalibrated** streams only, >= 8 h | `scripts/fit_allan.py` (fallback) |

Expected columns: `timestamp_s, gx, gy, gz, ax, ay, az` with gyro in rad/s and accel
in m/s^2.

## Recording conditions that actually matter

- **Stationary.** Allan variance measures the sensor; any motion measures the motion.
- **Thermally quiet.** Bias drifts with temperature, and a slow thermal ramp shows up
  as a false random-walk slope at long tau. Avoid a spot near a vent, a window, or a
  laptop exhaust. Log temperature alongside if the part exposes it.
- **Long.** The longest usable tau is roughly record_length / 10. Eight hours gets you
  to ~1000 s with any confidence, which is enough to see the random-walk upturn.
- **Uninterrupted.** Dropped samples corrupt the short-tau end. Check the timestamp
  delta histogram before trusting anything.

## Phone logs: verify before use

Phone IMU streams are often processed, and the dangerous case is online bias
correction -- it acts as a high-pass and erases the long-tau regime you recorded
overnight to see, while still producing a plausible-looking curve.

Use `TYPE_GYROSCOPE_UNCALIBRATED` / `TYPE_ACCELEROMETER_UNCALIBRATED` on Android, or
`CMMotionManager.gyroData` / `.accelerometerData` (never `deviceMotion`) on iOS. Then
run the four diagnostics in `scripts/fit_allan.py --diagnostics`:

1. PSD flatness -- a high-pass corner below ~0.01 Hz means the stream is being de-biased.
2. Running mean over hours -- raw bias random-walks; a de-biased stream is pinned near zero.
3. Calibrated minus uncalibrated (Android) -- the difference is the applied bias estimate.
4. Timestamp delta histogram -- jitter and dropped samples.
