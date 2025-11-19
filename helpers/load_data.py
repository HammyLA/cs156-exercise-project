from functools import reduce
import pandas as pd
import os

def load_data(path):
    times = []
    data = []

    for subject in os.listdir(path):
        subject_path = os.path.join(path, subject)
        if not os.path.isdir(subject_path):
            continue  # skip non-folder items
        
        for exercise in os.listdir(subject_path):
            exercise_path = os.path.join(subject_path, exercise)
            if not os.path.isdir(exercise_path):
                continue

            # CASE A: .txt files directly inside exercise folder
            for f in os.listdir(exercise_path):
                file_path = os.path.join(exercise_path, f)
                if os.path.isfile(file_path) and f.endswith(".txt"):
                    times.append({
                        "subject": subject,
                        "exercise": exercise,
                        "file_name": f,
                        "path": file_path
                    })

            # CASE B: nested sensor folders inside exercise folder
            for sensor in os.listdir(exercise_path):
                sensor_path = os.path.join(exercise_path, sensor)
                if os.path.isdir(sensor_path):
                    for f in os.listdir(sensor_path):
                        file_path = os.path.join(sensor_path, f)
                        if f.endswith(".txt"):
                            data.append({
                                "subject": subject,
                                "exercise": exercise,
                                "sensor": sensor,
                                "file_name": f,
                                "path": file_path
                            })


    template_times = pd.DataFrame(times)
    df = pd.DataFrame(data)
    df = df.sort_values(by=['subject', 'exercise', 'sensor']).reset_index().drop("index", axis=1)
    return df, template_times

def load_and_merge_sensors(sensor_files, subject, exercise):
    """
    Load CSVs for a subject/exercise, add suffixes, merge on 'time index'.
    """
    files = sensor_files[
        (sensor_files['subject'] == subject) &
        (sensor_files['exercise'] == exercise)
    ]['path'].tolist()

    dfs = [pd.read_csv(f, sep=";", index_col='time index') for f in files]
    for i, df in enumerate(dfs):
        df = df.add_suffix(f"_u{i+1}")
        df = df.rename(columns={f"time_s{i+1}": "time index"})
        dfs[i] = df

    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on="time index", how="outer"),
        dfs
    )
    
    merged_df['subject'] = subject
    merged_df['exercise'] = exercise
    return merged_df

def collect_template_times(template_times, subject, exercise):
    files = template_times[
        (template_times['subject'] == subject) & 
        (template_times['exercise'] == exercise)
    ]['path'].tolist()
    dfs = [pd.read_csv(f, sep=";") for f in files]
    return dfs[0]