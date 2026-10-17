"""BMI270 datasheet values used as spec overlays, each with its exact source.

Source: Bosch Sensortec, BMI270 Datasheet, document BST-BMI270-DS000-08,
Tables 2-3 (pp. 12-15) and Tables 7-13 (pp. 28-30). Values are transcribed
as printed. Interpretation is kept SEPARATE from transcription, and every
interpretive step is marked INTERPRETATION.

Datasheet p. 11: "Unless stated otherwise, the given values are over
lifetime, operating temperature and voltage ranges. Minimum/maximum values are
+-3 sigma." Most error specs below are printed as "+-x" in the TYP column, not
as min/max, so whether they are 1 sigma or 3 sigma is NOT stated. Plots show
the printed bound and do not convert it to a sigma.
"""

from __future__ import annotations

import numpy as np

DEG = np.pi / 180.0
G0 = 9.80665
DOC = "BST-BMI270-DS000-08"

# --- Gyroscope, Table 3 (pp. 13-15) ---------------------------------------
GYRO_ZERO_RATE_OFFSET_DPS = 0.5       # +-, TYP column, 25 C, soldered, lifetime
GYRO_TCO_DPS_PER_K = 0.015            # +-, zero-rate offset change over temperature
GYRO_SENS_ERR = 0.02                  # +-2 %, soldered, lifetime (no CRT)
GYRO_SENS_ERR_CRT = 0.004             # +-0.4 %, after CRT
GYRO_TCS_PER_K = 0.0002               # 0.02 %/K
GYRO_CROSS_AXIS = 0.002               # 0.2 %
GYRO_G_SENS_DPS_PER_G = 0.1           # MAX column
GYRO_ND_PERF_DPS = 0.007              # dps/sqrt(Hz), performance mode
GYRO_ND_NORMAL_DPS = 0.010            # dps/sqrt(Hz), normal mode
GYRO_LSB_DPS_2000 = 1.0 / 16.384      # Table 3 sensitivity at +-2000 dps

# Tables 10, 12, 13 (p. 30): per ODR, gyr_bwp = 0x02 (normal filter mode)
GYRO_ODR_HZ = np.array([25, 50, 100, 200, 400, 800, 1600, 3200, 6400])
GYRO_F3DB_HZ = np.array([11, 20, 39, 77, 152, 300, 557, 751, 712])
GYRO_RMS_PERF_MDPS = np.array([21.7, 30.7, 43.4, 61.4, 86.9, 123, 174, 302, 350])
GYRO_RMS_NORMAL_MDPS = np.array([31.0, 43.9, 62.0, 87.7, 124, 176, 248, 431, 500])
GYRO_GROUP_DELAY_MS = np.array([40, 20.5, 10.8, 5.97, 3.55, 2.34, 0.97, 0.82, 0.68])

# --- Accelerometer, Table 2 (pp. 12-13) -------------------------------------
ACC_ZERO_G_OFFSET_MG = 20.0           # +-, TYP column, 25 C, soldered, lifetime
ACC_TCO_MG_PER_K = 0.25               # +-
ACC_SENS_ERR = 0.004                  # +-0.4 %
ACC_TCS_PER_K = 0.00004               # 0.004 %/K
ACC_CROSS_AXIS = 0.01                 # 1 %
ACC_ND_MG = 0.16                      # mg/sqrt(Hz), normal mode, +-8 g

# Tables 7, 8, 9 (p. 28): per ODR, acc_bwp = 0x02
ACC_ODR_HZ = np.array([12.5, 25, 50, 100, 200, 400, 800, 1600])
ACC_F3DB_HZ = np.array([5.5, 11, 22, 44, 89, 178, 343, 740])
ACC_RMS_MG = np.array([0.38, 0.53, 0.75, 1.06, 1.51, 2.13, 2.96, 4.35])
ACC_GROUP_DELAY_MS = np.array([80, 40, 20.5, 10.5, 5.4, 2, 1.3, 0.6])


# --- INTERPRETATION: density convention -------------------------------------
# Table 12: RMS / sqrt(f_3dB) = 6.5-7.1 mdps/sqrt(Hz) at every ODR, which equals
# the printed 0.007. So Bosch's density is ONE-SIDED, with f_3dB as the noise
# bandwidth: rms^2 = ASD^2 * f_3dB. A two-sided reading would need a noise
# bandwidth of f_3dB/2, below the -3 dB point, which no low-pass filter has.
# The Allan/Kalibr N is two-sided (validation/synth.py), so N = ASD / sqrt(2).
# Using ENBW = f_3dB is itself an approximation. For a real filter ENBW >=
# f_3dB, so the true low-frequency ASD is <= the printed value and this N is
# an UPPER estimate.

def datasheet_N_gyro(mode: str = "perf") -> float:
    """Two-sided white-noise coefficient implied by the datasheet [rad/s/sqrt(Hz)]."""
    nd = GYRO_ND_PERF_DPS if mode == "perf" else GYRO_ND_NORMAL_DPS
    return nd * DEG / np.sqrt(2.0)


def datasheet_N_accel() -> float:
    """Two-sided white-noise coefficient implied by the datasheet [m/s^2/sqrt(Hz)]."""
    return ACC_ND_MG * 1e-3 * G0 / np.sqrt(2.0)


#: Luxonis depthai-core#886 (one unit, Kalibr/allan_variance_ros fit -> two-sided N, K).
LUXONIS = {
    "gyroscope_noise_density": 1.116e-4,
    "accelerometer_noise_density": 1.405e-3,
    "gyroscope_random_walk": 1.06e-5,
    "accelerometer_random_walk": 7.27e-5,
}
