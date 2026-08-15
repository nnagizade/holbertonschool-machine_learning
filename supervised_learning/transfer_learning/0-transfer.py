#!/usr/bin/env python3
"""
Trains a convolutional neural network to classify the CIFAR 10
dataset using transfer learning with the ResNet50 Keras Application
"""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the data for the model

    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3) containing the
           CIFAR 10 data
        Y: numpy.ndarray of shape (m,) containing the CIFAR 10
           labels for X

    Returns:
        X_p, Y_p
            X_p is a numpy.ndarray containing the preprocessed X
            Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.resnet50.preprocess_input(X.astype('float32'))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    # to fix issue with saving keras applications
    K.learning_phase = K.backend.learning_phase

    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    input_tensor = K.Input(shape=(32, 32, 3))

    resize = K.layers.Lambda(
        lambda x: K.backend.resize_images(
            x, 7, 7, data_format='channels_last'
        )
    )(input_tensor)

    base_model = K.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    features = base_model(resize, training=False)
    feature_model = K.models.Model(input_tensor, features)

    # Hint 3: compute the frozen layers' output ONCE, then train
    # only the trainable head on these values to save time
    train_features = feature_model.predict(X_train_p, batch_size=128,
                                            verbose=1)
    test_features = feature_model.predict(X_test_p, batch_size=128,
                                           verbose=1)

    feat_input = K.Input(shape=train_features.shape[1:])
    x = K.layers.GlobalAveragePooling2D()(feat_input)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    head_output = K.layers.Dense(10, activation='softmax')(x)

    head_model = K.models.Model(feat_input, head_output)
    head_model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

    callback = K.callbacks.EarlyStopping(monitor='val_accuracy',
                                          patience=3,
                                          restore_best_weights=True)

    head_model.fit(train_features, Y_train_p,
                    validation_data=(test_features, Y_test_p),
                    batch_size=128,
                    epochs=20,
                    callbacks=[callback],
                    verbose=1)

    # Build the full end-to-end inference model (raw 32x32x3 input)
    # so it can be evaluated directly with preprocess_data output
    full_output = head_model(features)
    model = K.models.Model(input_tensor, full_output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.save('cifar10.h5')
