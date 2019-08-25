# Import required libraries
import tensorflow as tf

# File paths
PRETRAINED_WEIGHTS_PATH = "../pretrained_weights/vgg16.h5"

# Model parameters
input_height = 224
input_width = 224
input_channels = 3
num_classes = 12


def create():
    inp = tf.keras.layers.Input((input_height, input_width, input_channels))

    # Convolution block 1
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1a')(inp)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1b')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Convolution block 2
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2a')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2b')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Convolution block 3
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3a')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3b')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3c')(x)
    pool3a = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3a')(x)

    # Convolution block 4
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4a')(pool3a)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4b')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4c')(x)
    pool4a = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4a')(x)

    # Convolution block 5
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5a')(pool4a)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5b')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5c')(x)
    pool5 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    # Load pretrained weights for convolution blocks 1-5
    temp_model = tf.keras.models.Model(inp, pool5)
    temp_model.load_weights(PRETRAINED_WEIGHTS_PATH)

    # Convolution blocks 6-7
    x = tf.keras.layers.Conv2D(4096, (7, 7), activation='relu', padding='same', name="conv6")(pool5)
    x = tf.keras.layers.Dropout(0.5, name="dropout1")(x)
    x = tf.keras.layers.Conv2D(4096, (1, 1), activation='relu', padding='same', name="conv7")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout2")(x)

    # Addend 1
    pool3b = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='relu', padding='same', name="pool3b")(pool3a)

    # Addend 2
    pool4b = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='relu', padding='same', name="pool4b")(pool4a)
    pool4bx2 = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=(2, 2), strides=(2, 2), use_bias=False, name="pool4bx2")(pool4b)

    # Addend 3
    conv7x4 = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(4, 4), use_bias=False)(x)

    # Final output block
    outp = tf.keras.layers.Add(name="add")([pool3b, pool4bx2, conv7x4])
    outp = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=(8, 8), strides=(8, 8), use_bias=False, name="addx8")(outp)
    outp = tf.keras.layers.Activation('softmax', name="output")(outp)

    model = tf.keras.models.Model(inp, outp)
    model.summary()

    loss = 'categorical_crossentropy'
    optimizer = tf.keras.optimizers.SGD(lr=1e-2, decay=1.6e-3, momentum=0.9)
    metrics = ['accuracy']
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model
