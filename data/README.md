# Data Folder

This is where preprocessed data will be held for use in model training and evaluation

When you run `exploration.ipynb`, you should get `training.pkl` and `test.pkl`

*NOTE:* The sensors have a sampling rate of 25Hz making a time windoe of length 50 equivalent to 2 seconds

`training.pkl` contains:   
1. `X_seg` --> 2155 time windows of length 50 with data from 45 features (all 5 sensors)   
2. `y_exercise` --> 2155 labels for each window for what kind of exercise this window contains  
3. `subject_windows` --> labels for what subject each window belongs to


`test.pkl` contains:    
1. `X_test` --> 8754 time windows of length 50 with data from 45 features
2. `y_test_exercise` --> 8754 labels for each window for what exercise it is
3. `subject_test_windows` --> labels for what subject each window belongs to