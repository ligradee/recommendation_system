import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

icon_directory = r"C:\Users\ligra\Desktop\диплом\fashion-dataset\icon.png"
icon = Image.open(icon_directory)

PAGE_CONFIG = {"page_title":"Recommendation system", 
               "page_icon":icon, 
               "layout":"centered", 
               "initial_sidebar_state":"auto"}

st.set_page_config(**PAGE_CONFIG)


#загрузка векторов изображений
feature_list = np.array(pickle.load(open('description.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

#создание модели для обучение на основе ResNet
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

original_title = '<p style="font-family:Courier; color:Blue; font-size: 40px;">-RECOMMENDATION SYSTEM-</p>'
st.markdown(original_title, unsafe_allow_html=True)
#st.title('-Recommendation system-')

#загрузка файла для рекомендации
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def description_img(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


uploaded_files = st.file_uploader("Upload an image to search for similar: ", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    save_uploaded_file(uploaded_file)
    st.success('Image uploaded successfully')
    st.markdown('<p style="font-family:Courier; color:Black; font-size: 20px;">YOUR SEARCH</p>', unsafe_allow_html=True)
    #вывод изображения
    display_image = Image.open(uploaded_file)
    st.image(display_image, width=300)
    features = description_img(os.path.join("uploads",uploaded_file.name),model)
    #поиск рекомендаций
    st.markdown('<p style="font-family:Courier; color:Black; font-size: 20px;">SIMILAR PRODUCTS</p>', unsafe_allow_html=True)
    indices = recommend(features,feature_list)
    col1,col2,col3,col4,col5 = st.columns(5)
    
    with col1:
        st.image(filenames[indices[0][0]])
    with col2:
        st.image(filenames[indices[0][1]])
    with col3:
        st.image(filenames[indices[0][2]])
    with col4:
        st.image(filenames[indices[0][3]])
    with col5:
        st.image(filenames[indices[0][4]])
     
