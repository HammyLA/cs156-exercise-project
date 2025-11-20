import numpy as np

def create_windows(filtered_df, window_size=50, stride=25, idle_label=0):
    """
    Create sliding windows from IMU data per subject.
    Returns:
        X_segments: (num_windows, window_size, num_sensors)
        y_segments: (num_windows,)
        subject_windows: array of subject ids per window
    """
    
    sensor_cols = [col for col in filtered_df.columns if any(sensor in col for sensor in ['acc_', 'gyr_', 'mag_'])]

    X_segments_list = []
    y_segments_list = []
    subject_windows_list = []

    # Ensure 'exercise' is numeric
    if not np.issubdtype(filtered_df['exercise'].dtype, np.number):
        raise ValueError("'exercise' column must be numeric")

    for subject, df_sub in filtered_df.groupby('subject'):
        X = df_sub[sensor_cols].values
        y_labels = df_sub['exercise'].values

        n_samples, n_sensors = X.shape

        # Slide windows across full timeline (including idle)
        for start in range(0, n_samples - window_size + 1, stride):
            end = start + window_size
            X_window = X[start:end]
            
            # Use center frame label to avoid looking ahead
            center_idx = start + window_size // 2
            y_window = y_labels[center_idx]

            X_segments_list.append(X_window)
            y_segments_list.append(y_window)
            subject_windows_list.append(subject)

    X_segments = np.stack(X_segments_list, axis=0)
    y_segments = np.array(y_segments_list)
    subject_windows = np.array(subject_windows_list)

    return X_segments, y_segments, subject_windows
