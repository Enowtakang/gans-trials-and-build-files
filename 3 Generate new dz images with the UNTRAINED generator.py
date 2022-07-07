"""
Defining and using the UNTRAINED
generator model
"""
from numpy import zeros
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from matplotlib import pyplot as plt


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
Specify size of the latent space
"""
latent_dim = 100

"""
Define the discriminator generator model
"""
model = define_generator(latent_dim)

"""
Generate samples
"""
n_samples = 9

X, _ = generate_fake_samples(model,
                             latent_dim,
                             n_samples)
"""
Scale pixel values from [-1, 1] to [0, 1]
"""
X = (X + 1) / 2.0

"""
Plot the generated samples
"""

for i in range(n_samples):
    """
    Define subplot
    """
    plt.subplot(3, 3, 1 + i)

    """
    Turn off axis labels
    """
    plt.axis('off')

    """
    Plot single image
    """
    plt.imshow(X[i])

"""
Show the figure
"""
plt.show()
