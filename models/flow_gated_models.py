from keras.optimizers import Adam, SGD
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Dense, Flatten, Conv3D, MaxPooling3D, Dropout, Multiply, BatchNormalization)
from tensorflow.keras.layers import Lambda
from tensorflow.keras.regularizers import l2

class FlowGatedModels:
    def __init__(self, input_shape=(64, 224, 224, 5), learning_rate=0.005):
        self.input_shape = input_shape
        self.learning_rate = learning_rate

    def get_rgb(self, input_x):
        rgb = input_x[..., :3]
        return rgb

    def get_opt(self, input_x):
        opt = input_x[..., 3:5]
        return opt

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        # RGB Channel
        rgb = Lambda(self.get_rgb)(inputs)
        rgb = Conv3D(16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
        rgb = Conv3D(16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)

        rgb = Conv3D(16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
        rgb = Conv3D(16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)

        rgb = Conv3D(32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
        rgb = Conv3D(32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)

        rgb = Conv3D(32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
        rgb = Conv3D(32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)

        # Optical Flow Channel
        opt = Lambda(self.get_opt)(inputs)
        opt = Conv3D(16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
        opt = Conv3D(16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
        opt = MaxPooling3D(pool_size=(1, 2, 2))(opt)

        opt = Conv3D(16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
        opt = Conv3D(16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
        opt = MaxPooling3D(pool_size=(1, 2, 2))(opt)

        opt = Conv3D(32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
        opt = Conv3D(32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
        opt = MaxPooling3D(pool_size=(1, 2, 2))(opt)

        opt = Conv3D(32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='sigmoid', padding='same')(opt)
        opt = Conv3D(32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='sigmoid', padding='same')(opt)
        opt = MaxPooling3D(pool_size=(1, 2, 2))(opt)

        # Fusion and Pooling
        x = Multiply()([rgb, opt])
        x = MaxPooling3D(pool_size=(8, 1, 1))(x)

        # Merging Block
        x = Conv3D(64, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
        x = Conv3D(64, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)

        x = Conv3D(64, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
        x = Conv3D(64, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)

        x = Conv3D(128, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
        x = Conv3D(128, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
        x = MaxPooling3D(pool_size=(2, 3, 3))(x)

        # FC Layers
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)

        # Output Layer
        pred = Dense(2, activation='softmax')(x)

        # Build the model
        model = Model(inputs=inputs, outputs=pred)

        # Compile the model
        
        return model

