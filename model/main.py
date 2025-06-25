import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
import pickle

def create_model(data):
    X = data.drop('label', axis=1)
    y = data['label']

    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
     random_state=42, stratify=y)


    model = xgb.XGBClassifier(objective='multi:softmax', 
    num_class=22, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted') 
    recall = recall_score(y_test, y_pred, average='weighted')  
    f1 = f1_score(y_test, y_pred, average='weighted')  
    classification = classification_report(y_test, y_pred)

    

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Classification report: {classification}")

    return model, scaler


def get_data_clean():
    data = pd.read_csv("data/data.csv")
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])
    return data

def main():
    data = get_data_clean()
    model, scaler = create_model(data)

    with open("model/pkl", "wb") as f:
        pickle.dump(model, f)

    with open("model/pkl", "wb") as f:
        pickle.dump(scaler, f)
    


if __name__== '__main__':
    main()
