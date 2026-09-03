#!/usr/bin/env python3
"""Transfer learning on CIFAR 10 using ResNet50"""
import tensorflow as tf
from tensorflow import keras as K


def preprocess_data(X, Y):
    """
    Pre-processes data for the model
    X: numpy.ndarray of shape (m, 32, 32, 3) containing CIFAR 10 data
    Y: numpy.ndarray of shape (m,) containing CIFAR 10 labels
    Returns: X_p, Y_p
    """
    X_p = K.applications.resnet50.preprocess_input(X.astype('float32'))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    # Input placeholder at CIFAR resolution
    inputs = K.Input(shape=(32, 32, 3))

    # Resize up to 224x224 for ResNet50
    resized = K.layers.Lambda(
        lambda img: tf.image.resize(img, (224, 224))
    )(inputs)

    # Load frozen base model
    base_model = K.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    base_model.trainable = False

    # Build a model that only does resize + base (for extracting features)
    feature_extractor = K.Model(inputs, base_model(resized, training=False))
    feature_extractor.compile(optimizer='adam', loss='categorical_crossentropy')

    print("Extracting bottleneck features (train)...")
    train_features = feature_extractor.predict(X_train_p, batch_size=128, verbose=1)
    print("Extracting bottleneck features (test)...")
    test_features = feature_extractor.predict(X_test_p, batch_size=128, verbose=1)

    # Trainable head, trained on cached features
    head_input = K.Input(shape=train_features.shape[1:])
    x = K.layers.Dense(256, activation='relu')(head_input)
    x = K.layers.Dropout(0.5)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)
    head_model = K.Model(head_input, outputs)

    head_model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callback = K.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=3, restore_best_weights=True
    )

    head_model.fit(
        train_features, Y_train_p,
        validation_data=(test_features, Y_test_p),
        batch_size=128,
        epochs=20,
        callbacks=[callback],
        verbose=1
    )

    # Build the FULL end-to-end model (resize -> frozen base -> trained head)
    # so it saves as one usable model, matching 0-main.py's expected usage
    full_outputs = head_model(base_model(resized, training=False))
    full_model = K.Model(inputs, full_outputs)

    full_model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    full_model.save('cifar10.h5')
