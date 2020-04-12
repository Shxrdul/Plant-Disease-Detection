import cv2  # working with images
import numpy as np  # dealing with arrays
import os  # dealing with directories

import tensorflow as tf  # Training model
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


# Creates an array of image in the form [image, label]
def process_verify_data(filepath):
    verifying_data = []

    img_name = filepath.split('.')[0]
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    verifying_data = [np.array(img), img_name]

    np.save('classify/verify_data.npy', verifying_data)

    return verifying_data


def analysis(filepath):
    # Preprocessing
    verify_data = process_verify_data(filepath)

    str_label = "Cannot make a prediction."
    status = "Error"

    tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 4, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('Model loaded successfully.')
    else:
        print('Error: Create a model using cnnk.py first.')

    img_data, img_name = verify_data[0], verify_data[1]
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)

    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 0:
        str_label = 'Healthy'
    elif np.argmax(model_out) == 1:
        str_label = 'Bacterial'
    elif np.argmax(model_out) == 2:
        str_label = 'Viral'
    elif np.argmax(model_out) == 3:
        str_label = 'Lateblight'

    if str_label == 'Healthy':
        status = 'Healthy'
    else:
        status = 'Unhealthy'

    result = 'Status: ' + status + '.'

    if (str_label != 'Healthy'): result += '\nDisease: ' + str_label + '.'

    return result


IMG_SIZE = 50  # Default image size
LR = 1e-3  # Learning rate
MODEL_NAME = 'plantDiseaseDetectionModel'


def main():
    filepath = input('Enter Image File Name:\n')
    print(analysis('classify/' + filepath))


if __name__ == '__main__':
    main()
