## Dataset Details

The dataset is created by augmenting rainfall, climate, and fertilizer data available for India. It includes the following key fields:

- **N (Nitrogen):** Ratio of Nitrogen content in soil
- **P (Phosphorous):** Ratio of Phosphorous content in soil
- **K (Potassium):** Ratio of Potassium content in soil
- **Temperature:** Soil temperature in degrees Celsius
- **Humidity:** Relative humidity in percentage
- **pH Value:** pH value of the soil
- **Rainfall:** Amount of rainfall in millimeters
  
## Model Performance

| Model                               | Accuracy | Precision | Recall  | F1 Score |
| ----------------------------------- | -------- | --------- | ------- | -------- |
| DecisionTreeClassifier              | 0.9795   | 0.9806    | 0.9795  |  0.9794  |
| BaggingClassifier with DT           | 0.9886   |  0.9889   | 0.9886  | 0.9886   |
| KNeighbourClassifier                | 0.9818   | 0.9830    | 0.9818  | 0.9818   |
| LGBMClassifier                      | 0.9932   | 0.9928    | 0.9932  | 0.9927   |
| RandomForestClassifier              | 0.9912   | 0.9921    | 0.9915  | 0.9913   |
| XGBoost                             | 0.9932   | 0.9935    | 0.9932  | 0.9931   |
| XGBoost with kfold                  | 0.9915   | 0.9920    | 0.9915  | 0.9914   |


