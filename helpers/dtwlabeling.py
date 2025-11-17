import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# ----------------------
# Utility: resample a multivariate timeseries to target length
# ----------------------
def resample_multivariate(X, target_len):
    """
    X: (T, C) array
    returns: (target_len, C)
    """
    T, C = X.shape
    if T == target_len:
        return X.copy()
    xp = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_len)
    X_rs = np.vstack([np.interp(x_new, xp, X[:, c]) for c in range(C)]).T
    return X_rs

# ----------------------
# Simple multivariate DTW (classic DP). Works for short templates.
# ----------------------

def dtw_distance_multivariate(a, b):
    """
    a, b: (T, C) arrays
    returns scalar DTW distance (sum of Euclidean costs across features)
    """
    # Sum Euclidean distances over channels for each timestep pair
    distance, path = fastdtw(a, b, dist=euclidean)
    return distance


# ----------------------
# Create a template from N labeled reps
# ----------------------
def build_template_from_bounds(group_df, rep_bounds, sensor_cols, template_len=100, method='mean'):
    """
    rep_bounds: list of (start, end) indices (ints) relative to group_df
    method: 'mean' or 'median' to aggregate resampled reps
    returns: template (template_len, C)
    """
    reps = []
    for (s,e) in rep_bounds:
        X = group_df.iloc[s:e][sensor_cols].values
        if len(X) < 4:
            continue
        reps.append(resample_multivariate(X, template_len))
    if len(reps) == 0:
        raise ValueError("No valid reps to build template from.")
    reps = np.stack(reps, axis=0)  # (num_reps, template_len, C)
    if method == 'median':
        template = np.median(reps, axis=0)
    else:
        template = np.mean(reps, axis=0)
    return template

# ----------------------
# Slide-template matching (compute distance for every candidate window)
# ----------------------
def sliding_template_search(group_df, template, sensor_cols, min_len=None, max_len=None, step=1, resample_to_template=True):
    """
    Returns distances list aligned to start indices: distances[i] = distance for window starting at i.
    - If resample_to_template True: windows of varying length are resampled to template length before DTW.
    - min_len/max_len: if provided, only consider windows with length in [min_len, max_len]. 
      If None, we will use template length for fixed-window mode.
    """
    X = group_df[sensor_cols].values
    T = len(X)
    template_len = template.shape[0]
    distances = np.full(T, np.inf)
    # choose window lengths; if min_len/max_len None, use template_len and slide fixed window
    if min_len is None: min_len = template_len
    if max_len is None: max_len = template_len
    # consider windows of lengths between min_len and max_len (inclusive) - try just a few lengths to save time
    candidate_lengths = list(set([template_len] + [min_len, max_len])) if min_len != max_len else [template_len]

    for L in candidate_lengths:
        for start in range(0, T - L + 1, step):
            window = X[start:start+L]
            if resample_to_template:
                window_rs = resample_multivariate(window, template_len)
                dist = dtw_distance_multivariate(window_rs, template)
            else:
                # if same len, can do direct DTW
                window_rs = window
                dist = dtw_distance_multivariate(window_rs, template)
            # keep best (lowest) distance for this start
            if dist < distances[start]:
                distances[start] = dist
    return distances

import numpy as np

def detect_candidates_from_distances(distances, expected_count, neighborhood=10,
                                     method='percentile', percentile=15):
    """
    Detect candidate start indices for reps in a DTW distance array.
    
    distances: np.array, DTW distances aligned by start index
    expected_count: int, number of reps expected
    neighborhood: int, minimal separation between detected starts (prevents overlaps)
    method: 'percentile' (thresholding) or 'none'
    percentile: float, percentile cutoff if method='percentile'
    
    Returns: list of start indices (non-overlapping)
    """
    T = len(distances)
    valid_idx = np.where(np.isfinite(distances))[0]
    if len(valid_idx) == 0:
        return []

    # Local minima detection
    is_min = np.zeros_like(distances, dtype=bool)
    for i in valid_idx:
        left = max(0, i - neighborhood)
        right = min(T, i + neighborhood + 1)
        if distances[i] <= distances[left:right].min():
            is_min[i] = True

    candidate_idx = valid_idx[is_min[valid_idx]]

    if method == 'percentile':
        thresh = np.nanpercentile(distances[valid_idx], percentile)
        candidate_idx = candidate_idx[distances[candidate_idx] <= thresh]

    if len(candidate_idx) == 0:
        # fallback: top-k smallest distances
        candidate_idx = valid_idx[np.argsort(distances[valid_idx])[:expected_count*3]]

    # Sort by ascending distance
    candidate_sorted = candidate_idx[np.argsort(distances[candidate_idx])]
    selected = []
    taken = np.zeros_like(distances, dtype=bool)

    for idx in candidate_sorted:
        # enforce non-overlapping via neighborhood
        if not taken[idx]:
            selected.append(idx)
            lo = max(0, idx - neighborhood)
            hi = min(T, idx + neighborhood + 1)
            taken[lo:hi] = True
        if len(selected) >= expected_count:
            break

    # if fewer than expected_count, fill with remaining best distances (non-overlapping)
    if len(selected) < expected_count:
        remaining = np.setdiff1d(valid_idx, selected)
        remaining_sorted = remaining[np.argsort(distances[remaining])]
        for r in remaining_sorted:
            if not taken[r]:
                selected.append(r)
                lo = max(0, r - neighborhood)
                hi = min(T, r + neighborhood + 1)
                taken[lo:hi] = True
            if len(selected) >= expected_count:
                break

    return selected[:expected_count]


# ----------------------
# Convert start indices to rep bounds (start,end) using template_len
# ----------------------
def starts_to_bounds(starts, template_len, group_len, max_expand=10):
    bounds = []
    for s in starts:
        # basic bound: [s, s+template_len)
        e = min(group_len, s + template_len)
        bounds.append((int(s), int(e)))
    return bounds

# ----------------------
# Main wrapper per trial (subject-exercise)
# ----------------------
def pseudo_label_trial(group_df, rep_bounds_labeled, sensor_cols, expected_reps=9,
                       template_len=120, step=1, neighborhood=10, percentile=15):
    """
    Given one trial group_df and few labeled rep_bounds, returns:
      - final_bounds: list of (start,end) for both labeled and pseudo-labeled reps
      - mark which are original labels vs pseudo
    """
    # Build template
    template = build_template_from_bounds(group_df, rep_bounds_labeled, sensor_cols, template_len=template_len)
    # Search distances
    distances = sliding_template_search(group_df, template, sensor_cols, min_len=int(template_len*0.6),
                                        max_len=int(template_len*1.4), step=step, resample_to_template=True)
    # detect expected_reps start indices
    starts = detect_candidates_from_distances(distances, expected_count=expected_reps, neighborhood=neighborhood, percentile=percentile)
    # convert to bounds
    bounds = starts_to_bounds(starts, template_len, len(group_df))
    # Mark which match labeled bounds (overlap)
    combined = []
    labeled_set = []
    for b in bounds:
        combined.append({'start': b[0], 'end': b[1], 'source': 'pseudo'})
    # include original labeled reps and mark
    for (s,e) in rep_bounds_labeled:
        combined.append({'start': int(s), 'end': int(e), 'source': 'label'}) 
    # merge and remove duplicates/overlaps by preferring labeled bounds
    # naive merging: sort by start and collapse overlaps keeping labeled
    combined = sorted(combined, key=lambda x: x['start'])
    merged = []
    for item in combined:
        if not merged:
            merged.append(item)
            continue
        last = merged[-1]
        # if overlap
        if item['start'] < last['end']:
            # prefer labeled boundaries or the one with larger span
            if last['source'] == 'label':
                # keep last, maybe extend end
                last['end'] = max(last['end'], item['end'])
            elif item['source'] == 'label':
                # replace with item (labeled)
                merged[-1] = item
            else:
                # keep whichever is larger
                if (item['end'] - item['start']) > (last['end'] - last['start']):
                    merged[-1] = item
                else:
                    last['end'] = max(last['end'], item['end'])
        else:
            merged.append(item)
    # final trimming and sorting
    final_bounds = [(int(m['start']), int(m['end'])) for m in merged]
    sources = [m['source'] for m in merged]
    return final_bounds, sources, distances

# ----------------------
# Example: generate windows & labels with weights
# ----------------------
def windows_and_labels_from_bounds(group_df, bounds, sources, sensor_cols, label_value, window_size=50, stride=25):
    """
    For each bound, create windows inside it and label them with label_value.
    sources list indicates 'label' or 'pseudo' per bound; return sample weights:
      - labeled windows weight=1.0
      - pseudo windows weight=0.5 (you can change)
    """
    X_list = []
    y_list = []
    w_list = []
    for (b,src) in zip(bounds, sources):
        s,e = b
        if e - s < window_size:
            continue
        # for windows fully inside the rep
        for start in range(s, e - window_size + 1, stride):
            X_win = group_df.iloc[start:start+window_size][sensor_cols].values
            X_list.append(X_win)
            y_list.append(label_value)
            w_list.append(1.0 if src == 'label' else 0.5)
    if len(X_list) == 0:
        return np.empty((0,window_size,len(sensor_cols))), np.array([]), np.array([])
    return np.stack(X_list, axis=0), np.array(y_list), np.array(w_list)
