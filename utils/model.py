import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from keras import backend as K


def get_model():
    model = DenseNet121(include_top=False, input_shape=(1024, 1024, 3), classes=14)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(14, activation="sigmoid")(x)

    model = Model(inputs=model.input, outputs=predictions)

    return model
