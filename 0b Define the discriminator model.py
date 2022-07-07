"""
Defining and Plotting the discriminator
model
"""
from keras.models import Sequential
from keras.optimizers import adam_v2
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
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
Define model
"""
model = define_discriminator()

"""
Summarize the model
"""
model.summary()

"""
Plot the model
"""
plot_model(model,
           to_file='8.3 Discriminator plot.png',
           show_shapes=True,
           show_layer_names=True)
