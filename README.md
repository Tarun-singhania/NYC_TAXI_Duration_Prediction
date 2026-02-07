# NYC Taxi Trip Duration Prediction

This project builds a complete **Machine Learning and Deep Learning pipeline** to predict **NYC taxi trip duration** using real-world spatio-temporal data.  
It covers data preprocessing, feature engineering, anomaly detection, clustering, classical ML models, and a deep neural network.

---

## Project Objective

To accurately predict taxi trip duration by leveraging:
- Time-based patterns
- Geospatial distance calculations
- Traffic behavior
- Airport proximity
- Anomaly detection
- Non-linear machine learning models

The project compares multiple regression models and evaluates them using the **R² score**.

---

## Dataset

- NYC Taxi Trip Dataset  
- Files used:
  - `train.csv`
  - `val.csv`

### Target Variable
- `trip_duration` (log-transformed as `trip_duration_log`)

---

## Feature Engineering & Preprocessing

All preprocessing is handled inside the `preprocess_data()` function.

### Time-Based Features
- Day of week
- Hour of day
- Weekend indicator
- Rush hour indicator
- Traffic indicator (derived from speed heatmap)

### Geospatial Features
- Haversine distance between pickup and dropoff (km)
- Pickup and dropoff inside NYC boundary
- Airport proximity detection (JFK, LaGuardia, Newark)
- KMeans clustering on GPS coordinates
- Average trip duration per cluster-hour

### Traffic & Anomaly Detection
- Daily trip count anomalies using mean ± 1.5 standard deviation
- Binary anomaly flag for unusual days
- Speed estimation (km/hr)

### Data Cleaning
- Removed rare passenger counts (7, 8, 9)
- Filtered invalid trip durations
- Log transformation of trip duration
- Feature scaling using `StandardScaler`

---

## Models Implemented

### Linear Regression
- Used as a baseline model

### Decision Tree Regressor
- Controlled depth and minimum samples to reduce overfitting

### Random Forest Regressor
- 150 trees
- Feature subsampling
- Parallel training
- Best-performing tree-based model

### Neural Network (TensorFlow / Keras)

Architecture:
Input
→ Dense(256) + BatchNorm + Dropout(0.5)
→ Dense(128) + BatchNorm + Dropout(0.5)
→ Dense(64) + BatchNorm + Dropout(0.25)
→ Dense(1)


Training Details:
- Optimizer: Adam
- Loss: Mean Squared Error
- Batch size: 512
- Early stopping with patience = 15

---

## Results

Model performance evaluated on the **validation dataset** using **R² score**:

| Model                    | R² Score |
|--------------------------|----------|
| Linear Regression        | ~0.36    |
| Decision Tree Regressor  | ~0.64    |
| Random Forest Regressor  | **~0.67** |
| Neural Network           | ~0.66   |

The Random Forest Regressor achieved the **best overall performance**, capturing non-linear relationships and benefiting from extensive feature engineering.  
The Neural Network showed competitive performance with stable convergence after log transformation and regularization.

---

## Visualizations

- Heatmap of average traffic speed by hour and weekday
- Training vs validation loss curve for neural network

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn
