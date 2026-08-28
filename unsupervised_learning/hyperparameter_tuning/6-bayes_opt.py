%%writefile 6-bayes_opt.py
#!/usr/bin/env python3
"""Optimizes a Keras CNN classifier on MNIST using GPyOpt Bayesian
optimization over 5 hyperparameters: learning rate, number of
filters, dropout rate, L2 regularization weight, and batch size.
"""
import GPyOpt
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as K


def load_data():
    """Loads and normalizes the MNIST dataset.
    Returns:
        X_train, Y_train, X_valid, Y_valid
    """
    (X_train, Y_train), (X_valid, Y_valid) = K.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    X_valid = X_valid.reshape(-1, 28, 28, 1).astype('float32') / 255
    Y_train = K.utils.to_categorical(Y_train, 10)
    Y_valid = K.utils.to_categorical(Y_valid, 10)
    return X_train, Y_train, X_valid, Y_valid


X_train, Y_train, X_valid, Y_valid = load_data()

iteration_count = 0
results = []


def build_model(learning_rate, filters, dropout_rate, l2_weight):
    """Builds and compiles a small CNN classifier.
    Args:
        learning_rate: learning rate for the Adam optimizer
        filters: number of filters in the convolutional layers
        dropout_rate: dropout rate applied after pooling
        l2_weight: L2 regularization weight applied to the dense
                   layer
    Returns:
        a compiled keras.Model
    """
    reg = K.regularizers.l2(l2_weight)
    inputs = K.Input(shape=(28, 28, 1))
    x = K.layers.Conv2D(filters, (3, 3), activation='relu',
                        padding='same')(inputs)
    x = K.layers.MaxPooling2D((2, 2))(x)
    x = K.layers.Dropout(dropout_rate)(x)
    x = K.layers.Conv2D(filters * 2, (3, 3), activation='relu',
                        padding='same')(x)
    x = K.layers.MaxPooling2D((2, 2))(x)
    x = K.layers.Dropout(dropout_rate)(x)
    x = K.layers.Flatten()(x)
    x = K.layers.Dense(128, activation='relu',
                       kernel_regularizer=reg)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs=inputs, outputs=outputs)
    optimizer = K.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def objective(x):
    """Objective function evaluated by GPyOpt. Trains a model for
    the given hyperparameter set and returns the negative
    validation accuracy (satisficing metric) to be minimized.
    Args:
        x: numpy.ndarray of shape (1, 5) containing
           [learning_rate, filters, dropout_rate, l2_weight,
           batch_size]
    Returns:
        numpy.ndarray of shape (1, 1) with the negative best
        validation accuracy achieved
    """
    global iteration_count
    iteration_count += 1

    learning_rate = float(x[:, 0])
    filters = int(x[:, 1])
    dropout_rate = float(x[:, 2])
    l2_weight = float(x[:, 3])
    batch_size = int(x[:, 4])

    model = build_model(learning_rate, filters, dropout_rate,
                        l2_weight)

    checkpoint_name = (
        'checkpoint_lr{:.5f}_filters{}_dropout{:.2f}_l2{:.5f}_'
        'batch{}.keras'.format(learning_rate, filters, dropout_rate,
                               l2_weight, batch_size)
    )

    checkpoint = K.callbacks.ModelCheckpoint(
        checkpoint_name, monitor='val_accuracy', save_best_only=True,
        mode='max')
    early_stop = K.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=3, mode='max',
        restore_best_weights=True)

    history = model.fit(
        X_train, Y_train, validation_data=(X_valid, Y_valid),
        epochs=15, batch_size=batch_size,
        callbacks=[checkpoint, early_stop], verbose=0)

    best_val_acc = max(history.history['val_accuracy'])

    print('Iteration {}: lr={:.5f}, filters={}, dropout={:.2f}, '
          'l2={:.5f}, batch_size={} -> val_accuracy={:.4f}'.format(
              iteration_count, learning_rate, filters, dropout_rate,
              l2_weight, batch_size, best_val_acc))

    results.append({
        'learning_rate': learning_rate,
        'filters': filters,
        'dropout_rate': dropout_rate,
        'l2_weight': l2_weight,
        'batch_size': batch_size,
        'val_accuracy': best_val_acc,
        'checkpoint': checkpoint_name,
    })

    return np.array([[-best_val_acc]])


domain = [
    {'name': 'learning_rate', 'type': 'continuous',
     'domain': (1e-4, 1e-2)},
    {'name': 'filters', 'type': 'discrete',
     'domain': (16, 32, 64)},
    {'name': 'dropout_rate', 'type': 'continuous',
     'domain': (0.0, 0.5)},
    {'name': 'l2_weight', 'type': 'continuous',
     'domain': (1e-5, 1e-2)},
    {'name': 'batch_size', 'type': 'discrete',
     'domain': (32, 64, 128)},
]


if __name__ == '__main__':
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=objective, domain=domain, model_type='GP',
        acquisition_type='EI', maximize=False)

    optimizer.run_optimization(max_iter=30)

    plt.figure()
    optimizer.plot_convergence()
    plt.savefig('convergence.png')
    plt.close()

    best_idx = int(np.argmax([r['val_accuracy'] for r in results]))
    best = results[best_idx]

    with open('bayes_opt.txt', 'w') as f:
        f.write('Bayesian Optimization Report\n')
        f.write('=============================\n\n')
        f.write('Total iterations run: {}\n\n'.format(len(results)))
        f.write('Best hyperparameters found:\n')
        f.write('  learning_rate: {:.6f}\n'.format(
            best['learning_rate']))
        f.write('  filters: {}\n'.format(best['filters']))
        f.write('  dropout_rate: {:.4f}\n'.format(
            best['dropout_rate']))
        f.write('  l2_weight: {:.6f}\n'.format(best['l2_weight']))
        f.write('  batch_size: {}\n'.format(best['batch_size']))
        f.write('  validation_accuracy: {:.4f}\n\n'.format(
            best['val_accuracy']))
        f.write('Best checkpoint file: {}\n\n'.format(
            best['checkpoint']))
        f.write('All iterations:\n')
        for i, r in enumerate(results, 1):
            f.write(
                '{}. lr={:.6f}, filters={}, dropout={:.4f}, '
                'l2={:.6f}, batch_size={}, val_accuracy={:.4f}\n'
                .format(i, r['learning_rate'], r['filters'],
                        r['dropout_rate'], r['l2_weight'],
                        r['batch_size'], r['val_accuracy']))

    print('Best hyperparameters:', best)
    print('Report saved to bayes_opt.txt')
    print('Convergence plot saved to convergence.png')
