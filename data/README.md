# Data Folder

This is where preprocessed data will be held for use in model training and evaluation

When you run `exploration.ipynb`, you should get `training.pkl` and `test.pkl`

*NOTE:* The sensors have a sampling rate of 25Hz making a time windoe of length 50 equivalent to 2 seconds

`training.pkl` contains:   
1. `data` --> Full 55305 length 50 with data, subject, and exercises

`test.pkl` contains:    
1. `test_data` --> 220K+ time windows of length 50 with data from 45 features

`time.pkl` contains:
1. `times_df` --> the time windows for each execution type