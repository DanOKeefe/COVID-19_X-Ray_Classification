import tensorflow as tf


def get_model():
    
    inputs = tf.keras.Input(shape=(200, 200, 1))
    conv_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    conv_2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(conv_1)
    maxpool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_2)
    drop_1 = tf.keras.layers.Dropout(0.25)(maxpool_1)
    
    conv_3 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(drop_1)
    conv_4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(conv_3)
    maxpool_2 = tf.keras.layers.MaxPooling2D(pool_size(2, 2))(conv_4)
    drop_2 = tf.keras.layers.Dropout(0.25)(maxpool_2)
    
    flatten = tf.keras.layers.Flatten()(drop_2)
    dense_1 = tf.keras.layers.Dense(512, activation='relu')(flatten)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense_1)
    
    
    outputs = tf.keras.layers.Dense(1)(hidden_2)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
