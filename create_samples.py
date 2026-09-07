import numpy as np
import os

os.makedirs("static/samples", exist_ok=True)

# Generate a synthetic normal ECG (280 points)
t = np.linspace(0, 1, 280)
# P wave, QRS complex, T wave
p_wave = 0.1 * np.exp(-((t - 0.2) ** 2) / 0.002)
q_wave = -0.15 * np.exp(-((t - 0.38) ** 2) / 0.0005)
r_wave = 1.0 * np.exp(-((t - 0.4) ** 2) / 0.0008)
s_wave = -0.25 * np.exp(-((t - 0.42) ** 2) / 0.0005)
t_wave = 0.25 * np.exp(-((t - 0.65) ** 2) / 0.005)
baseline_noise = 0.02 * np.sin(2 * np.pi * 50 * t)

ecg_normal = p_wave + q_wave + r_wave + s_wave + t_wave + baseline_noise
# Repeat to make multi-beat signal (~1400 samples, 5 beats)
multi_beat_normal = np.tile(ecg_normal, 5)

# Save normal sample
np.savetxt("static/samples/sample_normal_ecg.csv", multi_beat_normal, fmt="%.6f", delimiter=",")

# Generate synthetic abnormal ECG (atrial fibrillation / arrhythmia - irregular R-R, ectopic beat)
ecg_abnormal = ecg_normal.copy()
# Add ectopic spike and elevated ST segment
ecg_abnormal += 0.4 * np.exp(-((t - 0.55) ** 2) / 0.003)
multi_beat_abnormal = np.tile(ecg_abnormal, 5) + 0.05 * np.random.randn(1400)

np.savetxt("static/samples/sample_abnormal_ecg.csv", multi_beat_abnormal, fmt="%.6f", delimiter=",")

print("Sample ECG files generated in static/samples/")
