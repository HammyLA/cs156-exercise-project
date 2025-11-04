# cs156-exercise-project

This is our final project for CS 156 - Introduction to Artificial Intelligence. Our project analyzes data taken from wearable sensors during physical therapy exercises to detect and evaluate exercises.

## Team 5

Members:   
- Ayman Rabia
- Tarif Khan
- Lawrence Cuenco
- Nathan Choi
- Anas Durrant

## Dataset

The dataset that we utilize can be found here:

URL: https://www.kaggle.com/datasets/rabieelkharoua/physical-therapy-exercises-dataset/data?select=Description.pdf

The dataset consists of 5 subjects performing 8 exercises each. Each subject is wearing 5 sensors on different parts of their body and depending on what kind of exercise they are doing. Each sensor has an accelerometer, gyroscope, magnetometer with a sampling rate of 25Hz. The sensor data is organized as a time series, in which there contains template times for each exercise to identify which sections of the data coincides to what execution type. Execution 1 is a correct execution, Execution 2 is performed to quick, and Execution 3 is performed without enough movement

Summary:
- 5 Subjects
- 8 Exercises
- 5 Sensors per subject
- 3 Execution Types
- Each sensor has a accelerometer, gyroscope, and magnetometer

## Setup

Follow these steps to set up the project locally:

### 1. Clone the repository
```bash
git clone https://github.com/HammyLA/cs156-exercise-project.git
cd your-repo-name
```

### 2. Set up a virtual environment and install dependencies
Create a virtual environment (recommended) and install the required packages:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Set up Kaggle API access
To download the dataset automatically from Kaggle:

1. Create a [Kaggle account](https://www.kaggle.com/).
2. Go to **Account Settings** → scroll down to **API** → click **Create New API Token**.  
   This will download a file called `kaggle.json`.
3. ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/ # Make sure the path to kaggle.json is correct
   ```




