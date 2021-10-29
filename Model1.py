import tensorflow as tf
import tensorflow.keras.layers.experimental.preprocessing as pre
import tensorflow.keras.layers as layers
def create_model(input_shape,dropout_rate,final_activation):
    img_inputs = tf.keras.Input(shape=input_shape)
    x = pre.Rescaling(1. / 255.)(img_inputs)
    x = pre.RandomFlip(input_shape=input_shape)(x)
    x = pre.RandomZoom(height_factor=(0, 0.1))(x)
    x = pre.RandomRotation(0.1)(x)
    x = pre.RandomTranslation(0.1, 0.1)(x)
    x = pre.RandomContrast(0.1)(x)
    #y = layers.MaxPooling2D(3, 3)(x)
    y = layers.Conv2D(3, (3, 3),padding="same", activation='relu')(x)
    y = tf.keras.layers.GaussianNoise(0.05)(y)
    prevmodel = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_tensor=y,
        input_shape=input_shape,
    )
    prevmodel.trainable = False
    x = layers.GlobalAveragePooling2D(name='avg_pool')(prevmodel.output)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation=final_activation)(x)
    model = tf.keras.Model(inputs=prevmodel.inputs, outputs=outputs)

    return model


    print(input_shape)
    print("gpu", tf.config.list_physical_devices('GPU'))
    print('start fit', flush=True)
    print(model.summary())