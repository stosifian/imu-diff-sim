# Findings for the model session (from the validation check, 2026-10-15)

Context: `docs/validation_protocol.md`. These are requests against `noise.py` /
`params.py`. The validation session has not edited either file.

1. **Turn-on bias must be a searched parameter, not zero.**
   - `initial_bias()` returns zero. Its docstring says this is "the main place
     the model understates real-device error".
   - At `T_ep` = 20 s that understatement is 60–190× the random-walk drift
     (protocol §A1).
   - The docstring's worry is that a drawn bias would be nuisance noise for the
     search. That goes away if the bias is a **deterministic, signed, per-axis
     parameter** rather than a draw: add `gyroscope_turn_on_bias` and
     `accelerometer_turn_on_bias` (shape (3,)) to `ImuParams`.
   - Bound them with `bound · tanh(u)` like scale factor. The bound is the
     datasheet: gyro zero-rate offset ±0.5 dps and accel zero-g offset
     ±20 mg (DS000-08 Tables 2–3). Add ±TCO × ΔT on top (0.015 dps/K,
     0.25 mg/K) if the robot's thermal range is wider than the lab's.
   - `initial_bias(params)` returns them. Uniform DR samples the same box.
   - This also covers the "degenerate with thermal offset" argument in the
     docstring: one searched constant stands in for both.

2. **√2 in the gyro noise-density provenance. Now CONFIRMED from the
   datasheet itself (DS000-08 Tables 10–12; figure 12).**
   - The datasheet density is one-sided: rms/(density·√f₃dB) ≈ 1.0 at every
     ODR ≤ 1.6 kHz.
   - Datasheet-implied N: 8.64e-5 (performance mode) and 1.23e-4 (normal
     mode). The Luxonis nominal of 1.116e-4 matches normal mode, not
     performance mode. Name the mode in the provenance string, and make the
     sim's mode match it.
   - Accel: the datasheet implies 1.11e-3 vs Luxonis 1.405e-3.
   - Convention table and numeric proof: `validation/synth.py`, self-test G1.

3. **Tag every search-box edge with its source.**
   - DATASHEET where a min/max exists; LITERATURE (Suvorkin 2024) where the
     datasheet gives only a typical value or nothing.
   - The (0.5, 20) RW box currently has no source.
   - Plan to report every failure found with a plausibility tier T1–T4
     (protocol §A4).

4. **Check ∂L/∂K at nominal as soon as the sim runs.** If it's negligible, the
   policy can't see random walk inside an episode. Freeze K at nominal and
   drop it from the search, and say so in the report.

5. **Scale-factor bound** (figure 14). `SCALE_FACTOR_BOUND` = 2% matches the
   gyro no-CRT spec (±2%, Table 3). The accel spec is ±0.4% (Table 2), so a
   shared 2% bound makes the accel box 5× the datasheet. Split it into
   per-sensor bounds. Cross-axis (gyro 0.2%, accel 1%) is not modelled.

6. **Group delay** (figure 16). At 200 Hz the datasheet delay is 5.97 ms
   (gyro) / 5.4 ms (accel), about 30% of a 50 Hz policy step. The model has
   no delay and no LPF, so its per-sample white σ (N/√dt) also assumes a
   100 Hz noise bandwidth where the filtered part has 77 Hz.

7. **No action needed:** `imu_diff_sim.allan.overlapping_allan_deviation`
   agrees with the independent JAX implementation and with `allantools` to
   1e-9 relative (`tests/test_validation_selftest.py::test_C2_cross_check_against_main_allan_module`).
