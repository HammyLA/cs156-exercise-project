import numpy as np

# Sliding window segmentation
def segment_series(X, window_size=50, stride=25, y_labels=None):
    X_windows = []
    y_windows = []

    for start in range(0, len(X) - window_size + 1, stride):
        end = start + window_size
        X_windows.append(X[start:end])

        if y_labels is not None:
            # majority label in the window
            y_window = y_labels[start:end]
            counts = np.bincount((y_window).astype(int))  # shift -1,0,1 -> 0,1,2
            y_windows.append(np.argmax(counts) - 1)          # shift back
    return np.array(X_windows), np.array(y_windows)