"""
Example of training the discriminator
model on real and random tomato bacteria
spot disease leaves
"""
from os import listdir
from PIL import Image
import numpy as np
from numpy import ones
from numpy import zeros
from numpy.random import rand
from numpy.random import randint
from keras.optimizers import adam_v2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU


"""
Define the stand-alone discriminator model
"""


def define_discriminator(in_shape=(256, 256, 3)):
    model = Sequential()

    """
    Normal convolutional layer
    """
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))

    """
    Down-sample convolutional layer
    """
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    """
    Second down-sample convolutional layer
    """
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    """
    Third down-sample convolutional layer
    """
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    """
    Build Classifier
    """
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    """
    Compile the model
    """
    opt = adam_v2.Adam(
        learning_rate=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


"""

Load and prepare the training images
(scale to [-1, 1]

"""


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
        img = np.asarray(img)
        empty_data_list.append(img)

    empty_data_list = np.array(empty_data_list)

    return empty_data_list


"""
Try loading train_disease data
"""
train_X = load_data(train_disease_path, train_disease)
# print(train_X.shape)

# test_X = load_data(test_disease_path, test_disease)
# print(test_X.shape)


def load_real_samples():
    """
    Convert from unsigned ints to floats
    """
    X = train_X.astype('float32')

    """
    Scale from [0, 255] to [-1, 1]
    """
    X = (X - 127.5) / 127.5

    return X


"""
Select real samples
"""


def generate_real_samples(dataset, n_samples):
    """
    Choose random instances
    """
    ix = randint(0, dataset.shape[0], n_samples)

    """
    Retrieve selected images
    """
    X = dataset[ix]

    """
    Generate 'real' class labels (1) 
    """
    y = ones((n_samples, 1))

    return X, y


"""
Generate n fake samples with class labels
"""


def generate_fake_samples(n_samples):
    """
    Generate uniform random numbers in [0, 1]
    """
    X = rand(256 * 256 * 3 * n_samples)

    """
    Update to have the range [-1, 1]
    """
    X = -1 + X * 2

    """
    Reshape into a batch of colored images
    """
    X = X.reshape((n_samples, 256, 256, 3))

    """
    Generate 'fake' class labels (0)
    """
    y = zeros((n_samples, 1))

    return X, y


"""
Train the discriminator model
"""


def train_discriminator(model,
                        dataset,
                        n_iter=20,
                        n_batch=128):

    half_batch = int(n_batch/2)

    """
    Manually enumerate epochs
    """
    for i in range(n_iter):
        """
        Get randomly selected 'real' samples
        """
        X_real, y_real = generate_real_samples(
            dataset, half_batch)

        """
        Update discriminator on real samples
        """
        _, real_acc = model.train_on_batch(
            X_real, y_real)

        """
        Generate fake samples
        """
        X_fake, y_fake = generate_fake_samples(
            half_batch)

        """
        Update discriminator on fake samples
        """
        _, fake_acc = model.train_on_batch(
            X_fake, y_fake)

        """
        Summarize performance
        """
        print('>%d real=%.0f%% fake=%.0f%%' % (
            i+1, real_acc*100, fake_acc*100))


"""
Define the discriminator model
"""
model = define_discriminator()

"""
Load image data
"""
dataset = load_real_samples()

"""
Fit the model
"""
train_discriminator(model, dataset)
