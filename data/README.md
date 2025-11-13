# Data Folder

This is where preprocessed data will be held for use in model training and evaluation

When you run `exploration.ipynb`, you should get `training_data.pkl` and `test_data.pkl`

*NOTE:* The sensors have a sampling rate of 25Hz making a time windoe of length 50 equivalent to 2 seconds

`training_data.pkl` contains:   
1. X_seg --> 2155 time windows of length 50 with data from 45 features (all 5 sensors)   
2. y_activity --> 2155 labels for each window (0 = idle/no execution) (1 = executing exercise)   
3. y_exercise --> 2155 labels for each window for what kind of exercise this window contains
4. y_execution --> 2155 time windows of length 50 with each timestep labeled as execution type (0 = idle, 1 = correct, 2 = too fast, 3 = not enough motion)