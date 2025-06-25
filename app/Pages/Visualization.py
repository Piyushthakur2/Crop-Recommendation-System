import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the graphs data
with open('model/graphs.pkl', 'rb') as f:
    graphs = pickle.load(f)
with open("assets/style.css") as f:
     st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

# Function to plot the graphs
def plot_graph(graph_name):
    graph_info = graphs[graph_name]
    st.image(graph_info['file_path'], caption=graph_info['title'], use_column_width=True)

    st.markdown("<br><br><br><br><br><br><br>", unsafe_allow_html=True)
 
    st.write(f"**Description**: {graph_info.get('desc', '')}")

# Sidebar
st.sidebar.title('Graphs')
graph_names = list(graphs.keys())
selected_graph = st.sidebar.selectbox('Select Graph', graph_names)

# Main content
st.title('Crop Recommendation Dataset Visualizations')
st.header(selected_graph.replace('_', ' ').capitalize())
plot_graph(selected_graph)
