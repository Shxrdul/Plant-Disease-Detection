import cv2  # working with images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up the dataset
from tqdm import tqdm  # presents a progress bar

import tensorflow as tf  # Training model
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


# Returns class of image based on label
def label_img(img):
    word_label = img[0]

    if word_label == 'h':
        return [1, 0, 0, 0]
    elif word_label == 'b':
        return [0, 1, 0, 0]
    elif word_label == 'v':
        return [0, 0, 1, 0]
    elif word_label == 'l':
        return [0, 0, 0, 1]


# Creates an array of images in the form [image, label]
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read image in Grayscale (Preprocessing)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize image to default size
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train/train_data.npy', training_data)
    return training_data


# Creates an array of images in the form [image, label]
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        label = label_img(img)
        path = os.path.join(TEST_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read image in Grayscale (Preprocess)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize image to default size
        testing_data.append([np.array(img), np.array(label)])
    shuffle(testing_data)
    np.save('test/test_data.npy', testing_data)
    return testing_data


TRAIN_DIR = 'train/train'  # Directory with training dataset
TEST_DIR = 'test/test'  # Directory with Validation dataset
IMG_SIZE = 50  # Default image size
LR = 1e-3  # Learning rate
MODEL_NAME = 'plantDiseaseDetectionModel'

# Creating Dataaset
train_data = create_train_data()
test_data = process_test_data()

# If you have already created the dataset:
# train_data = np.load('train/train_data.npy')
# test_data = np.load('test/test_data.npy')

tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 4, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

# If you have already trained the Model:
# if os.path.exists('{}.meta'.format(MODEL_NAME)):
#     model.load(MODEL_NAME)
#     print('Model loaded Succescfully')

train = train_data[:]
test = test_data[:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

Y = [i[1] for i in train]
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)

print("Model Trained")
