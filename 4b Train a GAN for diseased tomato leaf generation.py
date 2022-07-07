"""
DC-GAN for generating tomato leaves
with bacterial spot symptoms
"""
from os import listdir
from PIL import Image
import numpy as np
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import adam_v2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot as plt


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
Define the stand-alone generator model
"""


def define_generator(latent_dim):
    model = Sequential()

    """
    Foundation for 32 x 32 image
    """
    n_nodes = 256 * 32 * 32
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((32, 32, 256)))

    """
    Up-sample to 64 x 64
    """
    model.add(
        Conv2DTranspose(
            filters=128,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    """
    Up-sample to 128 x 128
    """
    model.add(
        Conv2DTranspose(
            filters=128,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    """
    Up-sample to 256 x 256
    """
    model.add(
        Conv2DTranspose(
            filters=128,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    """
    Output layer
    """
    model.add(
        Conv2D(
            filters=3,
            kernel_size=(3, 3),
            activation='tanh',
            padding='same'))

    return model


"""
Define the combined discriminator model
and generator model for UPDATING the generator
"""


def define_gan(g_model, d_model):
    """
    Make weights in the discriminator model
    not trainable
    """
    d_model.trainable = False

    """
    Connect them
    """
    model = Sequential()
    model.add(g_model)
    model.add(d_model)

    """
    Compile model
    """
    opt = adam_v2.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt)

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
Generate points in latent space 
as input for the generator
"""


def generate_latent_points(latent_dim, n_samples):
    """
    Generate points in latent space
    """
    x_input = randn(latent_dim * n_samples)

    """
    Reshape into a batch of inputs for the network
    """
    x_input = x_input.reshape(n_samples, latent_dim)

    return x_input


"""
Use the generator to generate 'n'
fake samples with class labels
"""


def generate_fake_samples(g_model,
                          latent_dim,
                          n_samples):
    """
    Generate points in latent space
    """
    x_input = generate_latent_points(latent_dim,
                                     n_samples)
    """
    Predict outputs
    """
    X = g_model.predict(x_input)

    """
    Create 'fake' class labels (0)
    """
    y = zeros((n_samples, 1))

    return X, y


"""
Create and save a plot of generated images
"""


def save_plot(examples, epoch, n=3):
    """
    Scale from [-1, 1] to [0, 1]
    """
    examples = (examples + 1) / 2.0

    """
    Plot images
    """
    for i in range(n * n):
        """
        Define subplot
        """
        plt.subplot(n, n, 1 + i)

        """
        Turn off axis
        """
        plt.axis('off')

        """
        Plot raw pixel data
        """
        plt.imshow(examples[i])

    """
    Save plot to file
    """
    filename = 'generated_plot_e%03d.png' % (epoch + 1)
    plt.savefig(filename)
    plt.close()


"""
- Evaluate the discriminator, 
- plot generated images, 
- save generator model
"""


def summarize_performance(epoch,
                          g_model,
                          d_model,
                          dataset,
                          latent_dim,
                          n_samples=150):
    """
    Prepare real samples
    """
    X_real, y_real = generate_real_samples(
        dataset, n_samples)

    """
    Evaluate discriminator on real samples
    """
    _, acc_real = d_model.evaluate(X_real,
                                   y_real,
                                   verbose=0)
    """
    Prepare fake samples
    """
    x_fake, y_fake = generate_fake_samples(
        g_model, latent_dim, n_samples)

    """
    Evaluate discriminator on fake samples
    """
    _, acc_fake = d_model.evaluate(x_fake,
                                   y_fake,
                                   verbose=0)
    """
    Summarize discriminator performance
    """
    print(
        '>Accuracy real: %.0f%%, fake: %.0f%%' % (
            acc_real*100, acc_fake*100))

    """
    Save plot
    """
    save_plot(x_fake, epoch)

    """
    Save generator model title file
    """
    filename = 'generator_model_%03d.h5' % (epoch+1)
    g_model.save(filename)


"""
Train the generator and discriminator
# 500 epochs
"""


def train(g_model,
          d_model,
          gan_model,
          dataset,
          latent_dim,
          n_epochs=500,
          n_batch=128):

    """
    Batch per epoch
    """
    batch_per_epoch = int(
        dataset.shape[0] / n_batch)

    """
    Half batch
    """
    half_batch = int(n_batch / 2)

    """
    Manually enumerate epochs
    """
    for i in range(n_epochs):
        """
        Enumerate batches over the training set
        """
        for j in range(batch_per_epoch):
            """
            Get randomly generated 'real' samples
            """
            X_real, y_real = generate_real_samples(
                dataset, half_batch)

            """
            Update discriminator model weights
            """
            d_loss_1, _ = d_model.train_on_batch(
                X_real, y_real)

            """
            Generate fake samples
            """
            X_fake, y_fake = generate_fake_samples(
                g_model,
                latent_dim,
                half_batch)

            """
            Update discriminator model weights
            """
            d_loss_2, _ = d_model.train_on_batch(
                X_fake, y_fake)

            """
            Prepare points in latent space 
            as input for the generator
            """
            X_gan = generate_latent_points(
                latent_dim, n_batch)

            """
            Create 'inverted labels' for 
            the fake samples
            """
            y_gan = ones((n_batch, 1))

            """
            Update the generator via the 
            discriminator's error 
            """
            g_loss = gan_model.train_on_batch(
                X_gan, y_gan)

            """
            Summarize loss on this batch
            """
            print(
                '>%d, %d/%d, d1=%.3f, d2=%.3f, g=%.3f' % (
                    i+1,
                    j+1,
                    batch_per_epoch,
                    d_loss_1,
                    d_loss_2,
                    g_loss))

        """
        Evaluate the model performance, sometimes
        """
        if (i + 1) % 10 == 0:
            summarize_performance(i,
                                  g_model,
                                  d_model,
                                  dataset,
                                  latent_dim)


"""
Specify the size of the latent space
"""
latent_dim = 100

"""
Create the discriminator
"""
d_model = define_discriminator()

"""
Create the generator
"""
g_model = define_generator(latent_dim)

"""
Create the GAN
"""
gan_model = define_gan(g_model, d_model)

"""
Load image data
"""
dataset = load_real_samples()

"""
Train model
"""
train(g_model, d_model, gan_model,
      dataset, latent_dim)
