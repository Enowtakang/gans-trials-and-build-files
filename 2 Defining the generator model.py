"""
Defining the generator model
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model


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
Define the size of the latent space
"""
latent_dim = 100

"""
Define the generator model
"""
model = define_generator(latent_dim)

"""
Summarize the model
"""
model.summary()

"""
Plot the model
"""
plot_model(model,
           to_file='8.4 Generator Plot.png',
           show_shapes=True,
           show_layer_names=True)
