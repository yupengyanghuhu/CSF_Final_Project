#Importing libraries
#basics:
import pandas as pd
import numpy as np
import cv2 #Resizing images
import PIL.Image
from pandas.io.json import json_normalize, read_json #Load json file for the image names

#keras:
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

#sklearn:
from sklearn.preprocessing import LabelEncoder ## Encode labels
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold, train_test_split

#Setting options
pd.set_option('display.precision', 4)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

#Parameters:
IMAGE_SIZE = 128 #Cropping size to 128 * 128 pixels
RANDOM_SEED = 121
VALIDATION_SPLIT = 0.10

#Load data, annotation, data_info
IMG_JSON_PATH = '../AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json'
IMG_JPG_PATH = '../AgriculturalDisease_trainingset/images/' #Use as new path in the function 'change_image_paths'
IMG_LBL_PATH = '../label table.xlsx'

#Import image names and labels
def data_label_processing():
    
    #Importing label names
    image_name_df = pd.read_excel(IMG_LBL_PATH, sheet_name='Sheet1')
    image_name_df.columns = ['species', 'label_id', 'label']

    species = []
    for idx in image_name_df.index:
        if (pd.isnull(image_name_df.at[idx,'species'])):
            image_name_df.at[idx,'species'] = save_class
        else:
            save_class = image_name_df.at[idx,'species']
            species.append(save_class)
    imn_df = image_name_df.reset_index(drop=True)

    #Importing image names
    image_id_df = read_json(IMG_JSON_PATH)
    image_id_df.columns = ['label_id', 'image_name']
    imi_df = image_id_df.reset_index(drop=True)

    #Combining labels and image names
    images_df = pd.merge(imn_df, imi_df, how='inner', on='label_id')
    plant_df = images_df.sample(frac=1).reset_index(drop=True)

    return plant_df


#Defining img_resize
#Build a function to resize different leaf images...
def img_resize(imgpath, img_size):
    
    # resize the image to the specific size
    img = PIL.Image.open(imgpath)
    if (img.width > img.height):
        scale = float(img_size) / float(img.height)
        img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), img_size))).astype(np.float32)
    else:
        scale = float(img_size) / float(img.width)
        img = np.array(cv2.resize(np.array(img), (img_size, int(img.height * scale + 1)))).astype(np.float32)
        
    # crop the proper size and scale to [-1, 1]
    img = (img[
            (img.shape[0] - img_size) // 2:
            (img.shape[0] - img_size) // 2 + img_size,
            (img.shape[1] - img_size) // 2:
            (img.shape[1] - img_size) // 2 + img_size,
            :]-127)/128
    return img

#Change the original images names to new paths
def change_image_paths():
    
    img_data = []
    img_species = []
    img_label = []
    img_id = []

    cnt = 0
    for idx in plant_df.index:
        imgpath = IMG_JPG_PATH  + plant_df.at[idx,'image_name']
        cnt = cnt + 1
        if (cnt % 5000 == 0):
            print(str(cnt)+'th ---->', imgpath) #Add path before image name
        img_data.append(img_resize(imgpath, IMAGE_SIZE)) #get new image_datas_array
        img_species.append(plant_df.at[idx,'species']) #get species for each datas_array
        img_label.append(plant_df.at[idx,'label'])
        img_id.append(plant_df.at[idx,'label_id'])  
    
    return img_data, img_species, img_label, img_id
    
#Split original image data to train and test for future model use
def split_data():
    
    X_data = np.array(img_data)

    # Encode labels
    labelencoder = LabelEncoder()
    y_data_1d = labelencoder.fit_transform(img_label)
    y_data = to_categorical(y_data_1d)

    # Train/Test split for images
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)

    return [X_train, X_test, y_train, y_test]
