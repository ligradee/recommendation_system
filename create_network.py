import tensorflow
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

#создание модели для обучение на основе ResNet
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#функция перевода изображений в вектора
def description_img(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    #расширение массива
    expanded_img_array = np.expand_dims(img_array, axis=0)
    #предварительная обработка
    preprocessed_img = preprocess_input(expanded_img_array)
    #генерация прогнозов для изображений
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

filenames = []
#создание датасета с путями к изображениям
for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

description_list = []
#создание датасета с изображениями в виде векторов
for file in tqdm(filenames):
    description_list.append(description_img(file,model))

pickle.dump(description_list,open('description.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))



