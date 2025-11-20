import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.3, highcut=3, fs=25.0, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data, axis=0)

def remove_gravity(acc_data, fs=25.0, cutoff=0.2, order=2):
    # Real gravity removal = high-pass filter around 0.2–0.3 Hz
    b, a = butter(order, cutoff/(0.5*fs), btype='highpass')
    return filtfilt(b, a, acc_data, axis=0)

def smooth_signal(data, window_length=5, polyorder=2):
    # Savitzky-Golay filter for smoothing
    if window_length % 2 == 0:  # must be odd
        window_length += 1
    return savgol_filter(data, window_length=window_length, polyorder=polyorder, axis=0)

def preprocess_sensor_data(df, fs=25.0,
                           remove_gravity_flag=True,
                           apply_bandpass=True,
                           smooth=True):

    sensor_cols = [c for c in df.columns if any(s in c for s in ['acc_','gyr_','mag_'])]
    data = df[sensor_cols].values.copy()

    # Split sensors
    acc_idx = [i for i,c in enumerate(sensor_cols) if 'acc_' in c]
    gyr_idx = [i for i,c in enumerate(sensor_cols) if 'gyr_' in c]
    mag_idx = [i for i,c in enumerate(sensor_cols) if 'mag_' in c]

    acc = data[:, acc_idx]
    gyr = data[:, gyr_idx]
    mag = data[:, mag_idx]

    # 1. Gravity removal (HPF)
    if remove_gravity_flag:
        acc = remove_gravity(acc, fs=fs)

    # 2. Bandpass (but NOT for magnetometer)
    if apply_bandpass:
        acc = bandpass_filter(acc, lowcut=0.3, highcut=8, fs=fs)
        gyr = bandpass_filter(gyr, lowcut=0.5, highcut=8, fs=fs)
        # mag: no bandpass

    # 3. Smooth (only if necessary)
    if smooth:
        acc = smooth_signal(acc, window_length=7)
        gyr = smooth_signal(gyr, window_length=7)
        mag = smooth_signal(mag, window_length=9)  # gentler smoothing

    # Reassemble
    data[:, acc_idx] = acc
    data[:, gyr_idx] = gyr
    data[:, mag_idx] = mag

    df_filtered = pd.DataFrame(data, columns=sensor_cols, index=df.index)

    # Keep original metadata columns
    for col in df.columns:
        if col not in sensor_cols:
            df_filtered[col] = df[col]

    return df_filtered
