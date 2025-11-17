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

def remove_gravity(acc_data):
    # Simple high-pass filter to remove gravity (~0 Hz)
    return acc_data - np.mean(acc_data, axis=0)

def smooth_signal(data, window_length=5, polyorder=2):
    # Savitzky-Golay filter for smoothing
    if window_length % 2 == 0:  # must be odd
        window_length += 1
    return savgol_filter(data, window_length=window_length, polyorder=polyorder, axis=0)

def preprocess_sensor_data(df, fs=25.0, remove_gravity_flag=True, apply_bandpass=True, smooth=True):
    sensor_cols = [col for col in df.columns if any(sensor in col for sensor in ['acc_', 'gyr_', 'mag_'])]
    data = df[sensor_cols].values.copy()
    
    # Remove gravity if needed
    if remove_gravity_flag:
        acc_cols = [col for col in sensor_cols if 'acc_' in col]
        data[:, [sensor_cols.index(c) for c in acc_cols]] = remove_gravity(data[:, [sensor_cols.index(c) for c in acc_cols]])
    
    # Apply bandpass filter
    if apply_bandpass:
        data = bandpass_filter(data, fs=fs)
    
    # Smooth signals
    if smooth:
        data = smooth_signal(data)
    
    # Convert back to DataFrame
    df_filtered = pd.DataFrame(data, columns=sensor_cols, index=df.index)
    
    # Keep non-sensor columns
    non_sensor_cols = [col for col in df.columns if col not in sensor_cols]
    df_filtered[non_sensor_cols] = df[non_sensor_cols]
    
    return df_filtered
