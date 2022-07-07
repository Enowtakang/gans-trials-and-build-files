"""
Define function to LOAD CUSTOM data

Below, we are loading the train and test
datasets for:
                bacterial spot diseased leaves

You can REUSE the code to load train and test
datasets for:
                healthy tomato leaves

"""
from os import listdir
from PIL import Image
import numpy as np


"""
- Define datasets paths (directory)
- Define Input and Resize Shapes
"""
train_disease_path = 'D:/DATASETS/tomato_data/train/bacterial_spot/'
train_disease = []
test_disease_path = 'D:/DATASETS/tomato_data/test/bacterial_spot/'
test_disease = []

input_shape = (256, 256, 3)
resize_shape = (256, 256)


"""
Define a function to load data from path
"""


def load_data(data_path, empty_data_list):
    directory_images = np.sort(listdir(data_path))

    for file in directory_images:

        img = Image.open(data_path + file)
        img = img.convert('RGB')
        img = img.resize(resize_shape)
        img = np.asarray(img)/255
        empty_data_list.append(img)

    empty_data_list = np.array(empty_data_list)

    return empty_data_list


"""
Try loading train_disease data
"""
train_X = load_data(train_disease_path, train_disease)
# print(train_X.shape)
test_X = load_data(test_disease_path, test_disease)
# print(test_X.shape)



