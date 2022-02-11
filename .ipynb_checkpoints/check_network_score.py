import os
from PIL import Image
import numpy as np
import pandas as pd
import pickle
import tensorflow
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

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
    #обучение
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])
    
    print(type(indices))
    
    with open('indices.txt', 'w') as f1, open('distances.txt', 'w') as f2:
        for i in range(0, len(indices)):
            f1.write(str(indices[i]))
            f2.write(str(distances[i]))
    
    return indices

#удаление классов, которые встречаются слишком мало раз
def clean_tags(df):
    tags = df.type.value_counts().reset_index().rename(columns={'index':'type', 'type':'count'})
    #del_flags = del_flags.reset_index().rename(columns={'index':'type', 'type':'count'})
    del_tags = tags[tags['count'] < 2]
    buf  = df.type.reset_index()
    del_indexes = []

    for i in del_tags.type:
        array_buf = buf[buf['type'] == i]
        del_indexes = array_buf['index']
        df = df.drop(del_indexes)
    return df
    
    
    
def get_img_path(img_path, id):
    return img_path+'/'+str(id)+'.jpg'

#скачиваю датафрейм тегов
tags = pd.read_csv('C:/Users/ligra/Desktop/диплом/fashion-dataset/styles.csv', error_bad_lines=False)

missing_img = []
for idx, line in tags.iterrows():
    if not os.path.exists(os.path.join('C:/Users/ligra/Desktop/диплом/fashion-dataset/', 'images', str(line.id)+'.jpg')):
        print(os.path.join('C:/Users/ligra/Desktop/диплом/fashion-dataset/', 'images', str(line.id)+'.jpg'))
        missing_img.append(idx)
        
tags.drop(tags.index[missing_img], inplace=True)

tags = tags.rename(columns={'masterCategory':'gen_category',
                      'subCategory':'sub_category',
                      'articleType':'type',
                      'baseColour':'color',
                      'productDisplayName':'product_name'})


#загрузка векторов изображений
feature_list = np.array(pickle.load(open('description.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))


unique_tags_type = []        
    
for tag in tags.type.unique():
    unique_tags_type.append(str(tag).replace(' ', '_'))
     
le = preprocessing.LabelEncoder()
le.fit(unique_tags_type)

#создание модели для обучение на основе ResNet
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

    
#разделение выборки на обучающую и валидационную
tags = clean_tags(tags)
tags_train, tags_valid = train_test_split(tags, test_size=0.2, random_state=1, stratify=tags['type'])
#выделение тегов и индексов для каждой из выборок

#обучающие данные     
indexes_train = list(tags_train.id)
with open("indexes_train.txt", "w") as f:
    for index_train in indexes_train:
        f.write(str(index_train))
        f.write('\n')
    
type_tag_train = list(tags_train.type)
train_tags_preprocessing = []  
for tag in type_tag_train:
    train_tags_preprocessing.append(list(le.transform([str(tag).replace(' ', '_')]))[0])
    
#создание датасета с путями к изображениям обучающим
filenames_train = []
for id_train in indexes_train:
    filenames_train.append(get_img_path('C:/Users/ligra/Desktop/диплом/fashion-dataset/images',id_train))
    
description_list_train = []
for file in tqdm(filenames_train):
    description_list_train.append(description_img(file,model))
    
pickle.dump(description_list_train,open('description_train.pkl','wb'))
pickle.dump(filenames_train,open('filenames_train.pkl','wb'))
pickle.dump(train_tags_preprocessing,open('tags_train.pkl','wb'))

#валидационные данные
indexes_valid = list(tags_valid.id)
with open("indexes_valid.txt", "w") as f:
    for index_valid in indexes_valid:
        f.write(str(index_valid))
        f.write('\n')
        
type_tag_valid = list(tags_valid.type)
valid_tags_preprocessing = []  
for tag in type_tag_valid:
    valid_tags_preprocessing.append(list(le.transform([str(tag).replace(' ', '_')]))[0])
    
filenames_valid = []
for id_valid in indexes_valid:
    filenames_valid.append(get_img_path('C:/Users/ligra/Desktop/диплом/fashion-dataset/images',id_valid))
    
description_list_valid = []
for file in tqdm(filenames_valid):
    description_list_valid.append(description_img(file,model))
    
pickle.dump(description_list_valid,open('description_valid.pkl','wb'))
pickle.dump(filenames_valid,open('filenames_valid.pkl','wb'))
pickle.dump(valid_tags_preprocessing,open('tags_valid.pkl','wb'))

#подсчет скора

feature_list_train = np.array(pickle.load(open('description_train.pkl','rb')))
feature_list_valid = np.array(pickle.load(open('description_valid.pkl','rb')))
tags_train = pickle.load(open('tags_train.pkl','rb'))
tags_valid = pickle.load(open('tags_valid.pkl','rb'))

neighbors = KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list_train,tags_train)
y_pred = neighbors.predict(feature_list_valid)
ac = accuracy_score(tags_valid, y_pred)

print(ac)