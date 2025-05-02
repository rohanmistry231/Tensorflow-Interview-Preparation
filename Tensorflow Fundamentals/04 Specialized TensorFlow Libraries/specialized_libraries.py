import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
try:
    import tensorflow_datasets as tfds
except ImportError:
    tfds = None
try:
    import tensorflow_hub as hub
except ImportError:
    hub = None
import os

# %% [1. Introduction to Specialized TensorFlow Libraries]
# TensorFlow offers specialized libraries for data, models, and deployment.
# Covers TensorFlow Datasets, TensorFlow Hub, Keras, TensorFlow Lite, and TensorFlow.js.

print("TensorFlow version:", tf.__version__)

# %% [2. TensorFlow Datasets]
# Load CIFAR-10 dataset using TensorFlow Datasets.
if tfds is None:
    print("\nTensorFlow Datasets not installed. Please install: `pip install tensorflow-datasets`")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
else:
    ds, info = tfds.load('cifar10', with_info=True, as_supervised=True)
    train_ds = ds['train']
    test_ds = ds['test']
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, 10)
        return image, label
    train_ds = train_ds.map(preprocess).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
    x_train, y_train = next(iter(train_ds.batch(50000)))
    x_test, y_test = next(iter(test_ds.batch(10000)))
    x_train, y_train = x_train.numpy(), y_train.numpy()
    x_test, y_test = x_test.numpy(), y_test.numpy()

print("\nCIFAR-10 Dataset (via TensorFlow Datasets):")
print("Train Shape:", x_train.shape, "Test Shape:", x_test.shape)

# Visualize dataset samples
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(f"Class: {np.argmax(y_train[i])}")
    plt.axis('off')
plt.savefig('cifar10_samples.png')

# %% [3. TensorFlow Hub]
# Use a pre-trained MobileNetV2 from TensorFlow Hub for transfer learning.
if hub is None:
    print("\nTensorFlow Hub not installed. Please install: `pip install tensorflow-hub`")
    base_model = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
else:
    hub_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
    base_model = hub.KerasLayer(hub_url, input_shape=(96, 96, 3), trainable=False)

# Resize images for MobileNetV2
x_train_resized = tf.image.resize(x_train, [96, 96]).numpy()
x_test_resized = tf.image.resize(x_test, [96, 96]).numpy()

# Build model with Hub layer
hub_model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])
hub_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("\nTensorFlow Hub Model Summary:")
hub_model.summary()
hub_history = hub_model.fit(x_train_resized, y_train, epochs=3, batch_size=32, 
                            validation_data=(x_test_resized, y_test), verbose=1)
print("TensorFlow Hub Test Accuracy:", hub_history.history['val_accuracy'][-1].round(4))

# %% [4. Keras]
# Build a CNN using Keras high-level API for rapid prototyping.
keras_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("\nKeras Model Summary:")
keras_model.summary()
keras_history = keras_model.fit(train_ds, epochs=3, validation_data=test_ds, verbose=1)
print("Keras Test Accuracy:", keras_history.history['val_accuracy'][-1].round(4))

# %% [5. TensorFlow Lite]
# Convert Keras model to TensorFlow Lite for edge deployment.
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()
tflite_path = "cifar10_cnn.tflite"
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print("\nTensorFlow Lite Model Saved to:", tflite_path)

# Evaluate TFLite model
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
correct = 0
total = 0
for x, y in test_ds.unbatch().take(100):
    x = x.numpy()[np.newaxis, ...]
    interpreter.set_tensor(input_index, x)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)
    if np.argmax(pred) == np.argmax(y):
        correct += 1
    total += 1
print("TensorFlow Lite Accuracy (Subset):", (correct / total).round(4))

# %% [6. TensorFlow.js]
# Provide instructions for converting Keras model to TensorFlow.js.
print("\nTensorFlow.js Conversion Instructions:")
print("1. Install: `pip install tensorflowjs`")
print(f"2. Convert: `tensorflowjs_converter --input_format=keras {tflite_path} tfjs_model`")
print("3. Use in browser: Load `tfjs_model/model.json` with TensorFlow.js")
print("Note: Requires tensorflowjs package and JavaScript environment.")

# %% [7. Visualizing Predictions]
# Visualize predictions from Keras model.
predictions = keras_model.predict(x_test[:5])
plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"Pred: {np.argmax(predictions[i])}\nTrue: {np.argmax(y_test[i])}")
    plt.axis('off')
plt.savefig('keras_predictions.png')

# %% [8. Interview Scenario: Library Selection]
# Discuss choosing TensorFlow libraries for a project.
print("\nInterview Scenario: Library Selection")
print("1. TensorFlow Datasets: For curated, preprocessed datasets.")
print("2. TensorFlow Hub: For quick transfer learning with pre-trained models.")
print("3. Keras: For rapid prototyping and simple model building.")
print("4. TensorFlow Lite: For mobile/edge deployment with low latency.")
print("5. TensorFlow.js: For browser-based ML with WebGL acceleration.")

# %% [9. Practical Application: Combined Workflow]
# Combine TensorFlow Datasets, Hub, and Keras for transfer learning.
if hub and tfds:
    ds = tfds.load('cifar10', split='train', as_supervised=True)
    def preprocess_hub(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, [96, 96])
        label = tf.one_hot(label, 10)
        return image, label
    hub_ds = ds.map(preprocess_hub).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    combined_model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5", 
                       input_shape=(96, 96, 3), trainable=False),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("\nCombined Workflow Model Summary:")
    combined_model.summary()
    combined_history = combined_model.fit(hub_ds, epochs=3, validation_data=test_ds, verbose=1)
    print("Combined Workflow Test Accuracy:", combined_history.history['val_accuracy'][-1].round(4))
else:
    print("\nCombined Workflow Skipped: Requires tensorflow-datasets and tensorflow-hub")

# %% [10. Visualizing Training Progress]
# Plot validation accuracy for Keras and Hub models.
plt.figure()
plt.plot(keras_history.history['val_accuracy'], label='Keras CNN')
plt.plot(hub_history.history['val_accuracy'], label='TensorFlow Hub')
if hub and tfds:
    plt.plot(combined_history.history['val_accuracy'], label='Combined Workflow')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Specialized Libraries Comparison')
plt.legend()
plt.savefig('specialized_libraries_comparison.png')