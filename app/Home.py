import streamlit as st

def main():
    
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    st.title("Welcome to CultivateChoice Guide")

    st.markdown("This is a personalized advisor to optimal farming! Discover the perfect crops for your land, receive tailored recommendations, and cultivate success with ease.")
    
    st.image("images/Home.jpg", caption="Crop Recommendation", use_column_width=True)

    st.write("")  
    st.write("")  
    st.write("")  
    st.write("")  
    st.write("")  
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.header("Project Description")
    st.markdown(""" 
    CultivateChoice Guide is a web application designed to assist farmers in making informed decisions about crop selection based on various factors such as nutrient and climate inputs.

    The application utilizes machine learning models trained on a dataset containing information about different crops and their corresponding optimal conditions. By providing information about nutrient levels (N, P, K), temperature, humidity, pH, and rainfall, users can receive recommendations on suitable crops for cultivation.

    The project aims to empower farmers with data-driven insights to optimize agricultural productivity and enhance decision-making processes.
    """)

    st.header("Dataset")
    
  
    st.image("images/Dataset.png", caption="First Few rows of Crop Recommendation Dataset", use_column_width=True)
    
    st.markdown("""
    The dataset used for training the machine learning models contains information about various crops and their optimal conditions for growth. It includes features such as nutrient levels (N, P, K), temperature, humidity, pH, and rainfall, along with the corresponding crop labels.

    By analyzing this dataset, the models can learn patterns and relationships between different factors and crop types, enabling accurate predictions and recommendations for farmers.
    """)

    st.image("images/Models.png", caption="Accuracy Comparison of various models on Crop Recommendation Dataset", use_column_width=True)
    

if __name__ == '__main__':
    main()
