import streamlit as st
import pickle
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os


def get_clean_data():
    data = pd.read_csv("data/data.csv")
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])
    return data



def add_sidebar():
  st.sidebar.header("Nutrient & Climate Inputs")
  
  data = get_clean_data()
  
  slider_labels = [
         ("N (Nitrogen)", "N"),
        ("P (Phosphorus)", "P"),
        ("K (Potassium)", "K"),
        ("Temperature", "temperature"),
        ("Humidity", "humidity"),
        ("pH", "ph"),
        ("Rainfall", "rainfall"),
    ]

  input_dict = {}

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean())
    )
    
  return input_dict

def add_predictions(input_data):
  
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

   
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Crop Prediction")

    classifications = {'apple': 0, 'banana': 1, 'blackgram': 2, 'chickpea': 3, 'coconut': 4,
                        'coffee': 5, 'cotton': 6, 'grapes': 7, 'jute': 8, 'kidneybeans': 9,
                          'lentil': 10, 'maize': 11, 'mango': 12, 'mothbeans': 13, 'mungbean': 14,
                            'muskmelon': 15, 'orange': 16, 'papaya': 17, 'pigeonpeas': 18,
                              'pomegranate': 19, 'rice': 20, 'watermelon': 21}

    if prediction[0] == 0:
        st.write("The predicted crop is an apple.")

        
        image_path_apple_jpeg = "images/apple.jpeg"
        image_path_apple_jpg = "images/apple.jpg"

        
        if os.path.exists(image_path_apple_jpeg):
            st.image(image_path_apple_jpeg, caption="Predicted Apple Image")
        elif os.path.exists(image_path_apple_jpg):
            st.image(image_path_apple_jpg, caption="Predicted Apple Image")
        else:
            st.write("Image not found for apple.")
    else:
        predicted_crop_index = prediction[0]
        predicted_crop_name = list(classifications.keys())[list
        (classifications.values()).index(predicted_crop_index)]
        st.write(f"The predicted crop is {predicted_crop_name}.")

        image_path_jpeg = f"images/{predicted_crop_name.lower()}.jpeg"
        image_path_jpg = f"images/{predicted_crop_name.lower()}.jpg"

        if os.path.exists(image_path_jpeg):
            st.image(image_path_jpeg, caption=f"Predicted {predicted_crop_name} Image")
        elif os.path.exists(image_path_jpg):
            st.image(image_path_jpg, caption=f"Predicted {predicted_crop_name} Image")
        else:
            st.write(f"Image not found for {predicted_crop_name}.")


def main():
  st.set_page_config(
        page_title="Crop Recommendation System",
        page_icon="🌾", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
  with open("assets/style.css") as f:
     st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
  
  input_data = add_sidebar()

  
  with st.container():
        st.title("CultivateChoice Guide")
        st.markdown("Welcome to CultivateChoice Guide, your personalized advisor to optimal farming!  \nDiscover the perfect crops for your land, receive tailored recommendations, and cultivate success with ease.")

        add_predictions(input_data)


 
if __name__ == '__main__':
  main()
