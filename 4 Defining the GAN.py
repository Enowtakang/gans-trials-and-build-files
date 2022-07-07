"""
Create the three models
(generator,
discriminator,
combined generator and discriminator)
in the GAN
"""
from keras.optimizers import adam_v2
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.utils.vis_utils import plot_model


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
Define the size of the latent space
"""
latent_dim = 100

"""
Create the generator model
"""
g_model = define_generator(latent_dim)

"""
Create the discriminator model
"""
d_model = define_discriminator()

"""
Create the GAN
"""
gan_model = define_gan(g_model, d_model)

"""
Summarize the GAN model
"""
gan_model.summary()

"""
Plot the GAN model
"""
plot_model(gan_model,
           to_file='8.5 GAN Plot.png',
           show_shapes=True,
           show_layer_names=True)
