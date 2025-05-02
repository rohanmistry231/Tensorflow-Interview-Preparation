import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import mnist, cifar10

# %% [1. Introduction to Model Architectures]
# TensorFlow supports various neural network architectures for different tasks.
# This file covers Feedforward Neural Networks (FNNs), Convolutional Neural Networks (CNNs),
# Recurrent Neural Networks (RNNs, LSTMs, GRUs), and Transfer Learning.

print("TensorFlow version:", tf.__version__)

# %% [2. Feedforward Neural Networks (FNNs)]
# FNNs are fully connected networks for tasks like regression or classification.
# Example: FNN for MNIST digit classification.
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
x_train_mnist = x_train_mnist.astype('float32') / 255.0
x_test_mnist = x_test_mnist.astype('float32') / 255.0
x_train_mnist = x_train_mnist.reshape(-1, 28 * 28)  # Flatten: (28, 28) -> (784,)
x_test_mnist = x_test_mnist.reshape(-1, 28 * 28)

fnn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
fnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\nFNN Model Summary:")
fnn_model.summary()
fnn_history = fnn_model.fit(x_train_mnist, y_train_mnist, epochs=5, batch_size=32, 
                            validation_data=(x_test_mnist, y_test_mnist), verbose=1)
print("FNN Test Accuracy:", fnn_history.history['val_accuracy'][-1].round(4))

# %% [3. Convolutional Neural Networks (CNNs)]
# CNNs are designed for image data, using convolutional and pooling layers.
# Example: CNN for MNIST digit classification.
x_train_mnist_2d = x_train_mnist.reshape(-1, 28, 28, 1)  # Reshape: (784,) -> (28, 28, 1)
x_test_mnist_2d = x_test_mnist.reshape(-1, 28, 28, 1)

cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\nCNN Model Summary:")
cnn_model.summary()
cnn_history = cnn_model.fit(x_train_mnist_2d, y_train_mnist, epochs=5, batch_size=32, 
                            validation_data=(x_test_mnist_2d, y_test_mnist), verbose=1)
print("CNN Test Accuracy:", cnn_history.history['val_accuracy'][-1].round(4))

# %% [4. Recurrent Neural Networks (RNNs, LSTMs, GRUs)]
# RNNs are designed for sequential data, with LSTMs and GRUs handling long-term dependencies.
# Example: Synthetic sequence classification (predict next value in a noisy sine wave).
np.random.seed(42)
t = np.linspace(0, 100, 1000)
x_seq = np.sin(0.1 * t) + np.random.normal(0, 0.1, 1000)
y_seq = (x_seq[1:] > x_seq[:-1]).astype(np.int32)  # 1 if next value increases, 0 otherwise
x_seq = x_seq[:-1]
sequence_length = 10
x_seq_data = np.array([x_seq[i:i+sequence_length] for i in range(len(x_seq) - sequence_length)])
y_seq_data = y_seq[sequence_length:]

x_seq_train, x_seq_test = x_seq_data[:800], x_seq_data[800:]
y_seq_train, y_seq_test = y_seq_data[:800], y_seq_data[800:]
x_seq_train = x_seq_train[..., np.newaxis]  # Shape: (800, 10, 1)
x_seq_test = x_seq_test[..., np.newaxis]   # Shape: (189, 10, 1)

# LSTM Model
lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=False, input_shape=(sequence_length, 1)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nLSTM Model Summary:")
lstm_model.summary()
lstm_history = lstm_model.fit(x_seq_train, y_seq_train, epochs=5, batch_size=16, 
                              validation_data=(x_seq_test, y_seq_test), verbose=1)
print("LSTM Test Accuracy:", lstm_history.history['val_accuracy'][-1].round(4))

# GRU Model
gru_model = tf.keras.Sequential([
    tf.keras.layers.GRU(32, return_sequences=False, input_shape=(sequence_length, 1)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nGRU Model Summary:")
gru_model.summary()
gru_history = gru_model.fit(x_seq_train, y_seq_train, epochs=5, batch_size=16, 
                            validation_data=(x_seq_test, y_seq_test), verbose=1)
print("GRU Test Accuracy:", gru_history.history['val_accuracy'][-1].round(4))

# %% [5. Transfer Learning with tf.keras.applications]
# Transfer learning uses pre-trained models for new tasks.
# Example: Fine-tune MobileNetV2 on CIFAR-10.
(x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = cifar10.load_data()
x_train_cifar = x_train_cifar.astype('float32') / 255.0
x_test_cifar = x_test_cifar.astype('float32') / 255.0
y_train_cifar = tf.keras.utils.to_categorical(y_train_cifar, 10)
y_test_cifar = tf.keras.utils.to_categorical(y_test_cifar, 10)

# Preprocess input for MobileNetV2 (resize to 96x96)
x_train_cifar_resized = tf.image.resize(x_train_cifar, [96, 96])
x_test_cifar_resized = tf.image.resize(x_test_cifar, [96, 96])

base_model = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model
transfer_model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("\nTransfer Learning Model Summary:")
transfer_model.summary()
transfer_history = transfer_model.fit(x_train_cifar_resized, y_train_cifar, epochs=5, batch_size=32, 
                                      validation_data=(x_test_cifar_resized, y_test_cifar), verbose=1)
print("Transfer Learning Test Accuracy:", transfer_history.history['val_accuracy'][-1].round(4))

# %% [6. Visualizing Training Progress]
# Plot validation accuracy for all models.
plt.figure()
plt.plot(fnn_history.history['val_accuracy'], label='FNN')
plt.plot(cnn_history.history['val_accuracy'], label='CNN')
plt.plot(lstm_history.history['val_accuracy'], label='LSTM')
plt.plot(gru_history.history['val_accuracy'], label='GRU')
plt.plot(transfer_history.history['val_accuracy'], label='Transfer (MobileNetV2)')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Model Architecture Comparison')
plt.legend()
plt.savefig('model_comparison.png')

# %% [7. Practical Application: Fine-Tuning Transfer Learning]
# Fine-tune MobileNetV2 by unfreezing some layers.
base_model.trainable = True
for layer in base_model.layers[:100]:  # Freeze first 100 layers
    layer.trainable = False
transfer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                       loss='categorical_crossentropy', metrics=['accuracy'])
fine_tune_history = transfer_model.fit(x_train_cifar_resized, y_train_cifar, epochs=3, batch_size=32, 
                                       validation_data=(x_test_cifar_resized, y_test_cifar), verbose=1)
print("\nFine-Tuned Transfer Learning Test Accuracy:", fine_tune_history.history['val_accuracy'][-1].round(4))

# %% [8. Interview Scenario: Model Selection]
# Discuss choosing architectures for specific tasks.
print("\nInterview Scenario: Model Selection")
print("FNN: Suitable for tabular data, simple classification/regression.")
print("CNN: Ideal for image data, leveraging spatial hierarchies.")
print("LSTM/GRU: Best for sequential/time-series data, handling long-term dependencies.")
print("Transfer Learning: Efficient for image tasks with limited data, using pre-trained models.")

# %% [9. Visualizing Predictions]
# Visualize CNN predictions on MNIST test set.
predictions = cnn_model.predict(x_test_mnist_2d[:5])
plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test_mnist_2d[i, :, :, 0], cmap='gray')
    plt.title(f"Pred: {np.argmax(predictions[i])}\nTrue: {y_test_mnist[i]}")
    plt.axis('off')
plt.savefig('cnn_predictions.png')

# %% [10. Comparing Model Parameters]
# Compare parameter counts for all models.
print("\nModel Parameter Counts:")
print("FNN Parameters:", fnn_model.count_params())
print("CNN Parameters:", cnn_model.count_params())
print("LSTM Parameters:", lstm_model.count_params())
print("GRU Parameters:", gru_model.count_params())
print("Transfer Learning Parameters:", transfer_model.count_params())