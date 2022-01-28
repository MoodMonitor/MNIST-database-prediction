import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import cv2 as cv
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
import os
from time import time


def load_data_from_directory(directory):
    (x_train, y_train), = tf.keras.preprocessing.image_dataset_from_directory(
        f"{directory}",
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="grayscale",
        batch_size=10000,
        image_size=(28, 28),
        shuffle=True,
        validation_split=None,
        subset=None,
        seed=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False
    )
    return x_train, y_train


def create_mode_1D(model_name):
    """
    Create a model with with 1 dimension
    :param model_name: str, name of model to be saved
    :return: model object, callbacks
    """
    model = Sequential()
    model.add(Dense(units=512, activation='relu', input_shape=(784,)))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=10, verbose=1)
    mc = ModelCheckpoint(f"./{model_name}.h5", monitor="val_accuracy", verbose=1, save_best_only=True)
    cb = [es, mc]
    return model, cb


def prepare_MNIST_data_1D(white_scale=0):
    """
    Prepare MNIST dataset to be used in 1 dimension model
    :param white_scale: int, if set to 0 data have full grayscale range, of set to 1 data have only white and black
    pixels
    :return: 60000 images with shape (1, 784), 60000 vector labels,
    10000 images with shape (1, 784), 10000 vector labels
    """
    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    if white_scale == 1:
        x_train = np.where(x_train >= 1, 255, x_train)
        x_valid = np.where(x_valid >= 1, 255, x_valid)
    x_train = x_train.reshape(60000, 784)
    x_valid = x_valid.reshape(10000, 784)
    x_train = x_train / 255
    x_valid = x_valid / 255
    y_train = keras.utils.to_categorical(y_train)
    y_valid = keras.utils.to_categorical(y_valid)
    return x_train, y_train, x_valid, y_valid


def preprocess_directory_data_to_1D(directory, white_scale=0):
    """
    Preprocess data from given directory to be able to use in 1 dimension model
    :param directory: str, path to the images to be preprocessed
    :param white_scale: int, if set to 0 data have full grayscale range, of set to 1 data have only white and black
    pixel
    :return: images with shape (1, 784), vectored labels to images
    """
    x_train, y_train = load_data_from_directory(directory)
    if white_scale == 1:
        x_train = np.where(x_train >= 0.01, 255, x_train)
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_train = x_train / 255
    y_train = keras.utils.to_categorical(y_train)
    return x_train, y_train


def mnist_fit_model_1D(directory, model_name, white=0):
    """
    Fit one-dimensional model with MNIST training set and directory validation data
    :param directory: str, path to the directory validation data
    :param model_name: Model name
    :param white: int, if set to 0 data have full grayscale range, of set to 1 data have only white and black
    pixel
    :return: Model history
    """
    x_train, y_train, x_valid, y_valid = prepare_MNIST_data_1D(white)
    x_train2, y_train2 = preprocess_directory_data_to_1D(directory, white)
    model, cb = create_mode_1D(model_name)
    history = model.fit(x_train, y_train, epochs=100, verbose=1, validation_data=(x_train2, y_train2), callbacks=cb)
    return history


def preprocess_directory_data_to_2D(directory, white_scale=0):
    """
    Preprocess data from given directory to be able to use in 2 dimension model
    :param directory: str, path to the images to be preprocessed
    :param white_scale: int, if set to 0 data have full grayscale range, of set to 1 data have only white and black
    pixel
    :return: images with shape (28, 28, 1), vectored labels to images
    """
    x_train, y_train = load_data_from_directory(directory)
    if white_scale == 1:
        x_train = np.where(x_train >= 0.01, 255, x_train)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_train = x_train / 255
    y_train = keras.utils.to_categorical(y_train)
    return x_train, y_train


def create_mode_2D(model_name):
    """
    Create a model with with 2 dimension
    :param model_name: str, name of model to be saved
    :return: model object, callbacks
    """
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation="relu"))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(65, (3, 3), activation="relu"))
    model.add(MaxPool2D(2, 2))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])
    es = EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=10, verbose=1)
    mc = ModelCheckpoint(f"./{model_name}.h5", monitor="val_accuracy", verbose=1, save_best_only=True)
    cb = [es, mc]
    return model, cb


def create_mode_2D_save_every_epoch():
    """"
    Create a model with with 2 dimension, that will save each model
    :return: model object, callbacks
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation="relu"))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(65, (3, 3), activation="relu"))
    model.add(MaxPool2D(2, 2))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])
    es = EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=10, verbose=1)
    mc = ModelCheckpoint("Model" + "{epoch:02d}" + ".h5", monitor="val_accuracy", verbose=1,
                         save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch')
    cb = [es, mc]
    return model, cb


def prepare_MNIST_data_2D(white_scale=0):
    """
    Prepare MNIST dataset to be used in 2 dimension model
    :param white_scale: int, if set to 0 data have full grayscale range, of set to 1 data have only white and black
    pixels
    :return: 60000 images with shape (28, 28, 1), 60000 vector labels,
    10000 images with shape (28, 28, 1), 10000 vector labels
    """
    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    if white_scale == 1:
        x_train = np.where(x_train >= 1, 255, x_train)
        x_valid = np.where(x_valid >= 1, 255, x_valid)
    x_train = x_train.astype(np.float32) / 255
    x_valid = x_valid.astype(np.float32) / 255
    x_train = np.expand_dims(x_train, -1)
    x_valid = np.expand_dims(x_valid, -1)
    y_train = keras.utils.to_categorical(y_train)
    y_valid = keras.utils.to_categorical(y_valid)
    return x_train, y_train, x_valid, y_valid


def mnist_fit_model_2D(directory, model_name, white=0):
    """
    Fit two-dimensional model with MNIST training set and directory validation data
    :param directory: str, path to the directory validation data
    :param model_name: Model name
    :param white: int, if set to 0 data have full grayscale range, of set to 1 data have only white and black
    pixel
    :return: Model history
    """
    x_train, y_train, x_valid, y_valid = prepare_MNIST_data_2D(white)
    x_valid, y_valid = preprocess_directory_data_to_2D(directory, white)
    model, cb = create_mode_2D(model_name)
    history = model.fit(x_train, y_train, epochs=100, verbose=1, validation_data=(x_valid, y_valid), callbacks=cb)
    return history


def fit_model_1D(directory, model_name, white=0):
    """
    Fit one-dimensional model with directory data training set and MNIST validation data
    :param directory: str, path to the directory validation data
    :param model_name: Model name
    :param white: int, if set to 0 data have full grayscale range, of set to 1 data have only white and black
    pixel
    :return: Model history
    """
    x_train, y_train = preprocess_directory_data_to_1D(directory, white)
    x_train2, y_train2, x_valid, y_valid = prepare_MNIST_data_1D(white)
    model, cb = create_mode_1D(model_name)
    history = model.fit(x_train, y_train, epochs=100, verbose=1, validation_data=(x_train2, y_train2), callbacks=cb)
    return history


def fit_model_2D(directory, model_name, white=0):
    """
    Fit two-dimensional model with directory data training set and MNIST validation data
    :param directory: str, path to the directory validation data
    :param model_name: Model name
    :param white: int, if set to 0 data have full grayscale range, of set to 1 data have only white and black
    pixel
    :return: Model history
    """
    x_train, y_train = preprocess_directory_data_to_2D(directory, white)
    x_train2, y_train2, x_valid, y_valid = prepare_MNIST_data_2D(white)
    model, cb = create_mode_2D(model_name)
    history = model.fit(x_train, y_train, epochs=100, verbose=1, validation_data=(x_train2, y_train2), callbacks=cb)
    return history


def square_images(read_directory, save_directory):
    """
    Change shape of the images from rectangle to square
    :param read_directory: directory where images to be change are stored
    :param save_directory: directory where images will be saved
    :return: None
    """
    os.chdir(read_directory)
    path = os.getcwd()
    for num in range(10):
        os.chdir(f"./{num}")
        entries = os.scandir()
        for plik in entries:
            image = cv.imread(plik.name)
            sqaure_needed = int((image.shape[0] - image.shape[1]) / 2)
            if sqaure_needed > 0:
                border = cv.copyMakeBorder(image,
                                           top=0,
                                           bottom=0,
                                           left=sqaure_needed,
                                           right=sqaure_needed,
                                           borderType=cv.BORDER_CONSTANT,
                                           value=[0, 0, 0])
            else:
                sqaure_needed = int((image.shape[1] - image.shape[0]) / 2)
                border = cv.copyMakeBorder(image,
                                           top=sqaure_needed,
                                           bottom=sqaure_needed,
                                           left=0,
                                           right=0,
                                           borderType=cv.BORDER_CONSTANT,
                                           value=[0, 0, 0])
            cv.imwrite(fr"{save_directory}\{num}\{plik.name}", border)
        os.chdir(path)


def _border_image(read_directory, save_directory, change_value, x_value):
    """
    Make a padding with black pixels to images
    :param read_directory: directory where images to be change are stored
    :param save_directory: directory where images will be saved
    :param change_value: value of change on y axis
    :param x_value: value of change on x axis
    :return: None
    """
    os.chdir(read_directory)
    path = os.getcwd()
    for num in range(10):
        os.chdir(f"./{num}")
        entries = os.scandir()
        for plik in entries:
            image = cv.imread(plik.name)
            if x_value == 0:
                border = cv.copyMakeBorder(image, top=0, bottom=0, left=int(image.shape[0] / change_value),
                                           right=int(image.shape[0] / change_value), borderType=cv.BORDER_CONSTANT,
                                           value=[0, 0, 0])
            else:
                border = cv.copyMakeBorder(image, top=int(image.shape[0] / change_value),
                                           bottom=int(image.shape[0] / change_value),
                                           left=int(image.shape[0] / x_value) + int(image.shape[0] / change_value),
                                           right=int(image.shape[0] / x_value) + int(image.shape[0] / change_value),
                                           borderType=cv.BORDER_CONSTANT,
                                           value=[0, 0, 0])
            cv.imwrite(fr"{save_directory}\{num}\{plik.name}", border)
        os.chdir(path)


def _delete_files(directory):
    """
    Delete all files from directories named from 0 to 10
    :param directory: directory where folders are stored
    :return: None
    """
    os.chdir(directory)
    path = os.getcwd()
    for num in range(10):
        os.chdir(f"./{num}")
        entries = os.scandir()
        for plik in entries:
            os.remove(plik)
        os.chdir(path)


def _fit_the_best_padding(model_name, dimensions, data_directory, save_directory, x_value, white=0):
    """
    Find best padding to the model with given values
    :param model_name: name of used model
    :param dimensions: dimensions of used model, only supported 1 and 2
    :param data_directory: directory where images to be change are stored
    :param save_directory: directory where images will be saved
    :param x_value: value of change on x axis
    :param white: int, if set to 0 data have full grayscale range, of set to 1 data have only white and black
    pixel
    :return: best accuracy, change_value to have best padding
    """
    max_value = 0
    index_value = 0
    model = load_model(model_name)
    start_value = 1.50
    for number in range(1000):
        print("Attempt: ", number)
        _border_image(data_directory, save_directory, start_value + float(number / 100), x_value)
        if dimensions == 2:
            x_train, y_train = preprocess_directory_data_to_2D(save_directory, white)
        elif dimensions == 1:
            x_train, y_train = preprocess_directory_data_to_1D(save_directory, white)
        result = model.evaluate(x_train, y_train)
        if result[1] > max_value:
            max_value = result[1]
            index_value = number
        _delete_files(save_directory)
        if number - index_value > 100:
            break
    return max_value, start_value + float(index_value / 100)


def fit_padding_to_the_model(model_name, dimensions, data_directory, save_directory, white=0):
    """
    Brute force for find the best padding for model
    :param model_name: name of used model
    :param dimensions: dimensions of used model, only supported 1 and 2
    :param data_directory: directory where images to be change are stored
    :param save_directory: directory where images will be saved
    :param white: int, if set to 0 data have full grayscale range, of set to 1 data have only white and black
    pixel
    :return: the best accuracy, x axis value that need to be changed, y axis value that need to be changed
    """
    start_time = time()
    x_accuracy, x_value = _fit_the_best_padding(model_name, dimensions, data_directory, save_directory, 0, white)
    final_accuracy, y_value = _fit_the_best_padding(model_name, dimensions, data_directory, save_directory, x_value,
                                                    white)
    print("Padding time: ", (time() - start_time) / 60, "min")
    print("X accuracy: ", x_accuracy)
    return final_accuracy, x_value, y_value
