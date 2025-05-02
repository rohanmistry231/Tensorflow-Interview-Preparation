import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import os

# %% [1. Introduction to Deployment]
# Deployment involves exporting, serving, and running TensorFlow models in production.
# Covers SavedModel, TensorFlow Serving, and TensorFlow Lite/JS.

print("TensorFlow version:", tf.__version__)

# %% [2. Preparing the Dataset and Model]
# Load and preprocess MNIST dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32).prefetch(tf.data.AUTOTUNE)
print("\nMNIST Dataset:")
print("Train Shape:", x_train.shape, "Test Shape:", x_test.shape)

# Train a simple CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=3, validation_data=test_ds, verbose=1)
print("\nModel Test Accuracy:", model.evaluate(test_ds, verbose=0)[1].round(4))

# %% [3. Model Export: SavedModel]
# Export the model as SavedModel.
saved_model_path = "saved_model/mnist_cnn"
model.save(saved_model_path)
print("\nSavedModel Exported to:", saved_model_path)

# Load and test SavedModel
loaded_model = tf.keras.models.load_model(saved_model_path)
print("Loaded SavedModel Test Accuracy:", loaded_model.evaluate(test_ds, verbose=0)[1].round(4))

# %% [4. Serving with TensorFlow Serving]
# Note: TensorFlow Serving requires separate installation and setup.
# Instructions for serving the SavedModel.
print("\nTensorFlow Serving Instructions:")
print("1. Install TensorFlow Serving: `docker pull tensorflow/serving`")
print(f"2. Serve model: `docker run -p 8501:8501 --mount type=bind,source={os.path.abspath(saved_model_path)},target=/models/mnist_cnn -e MODEL_NAME=mnist_cnn -t tensorflow/serving`")
print("3. Query model: Use REST API at http://localhost:8501/v1/models/mnist_cnn:predict")

# %% [5. Edge Deployment: TensorFlow Lite]
# Convert model to TensorFlow Lite.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_path = "mnist_cnn.tflite"
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
    if np.argmax(pred) == y:
        correct += 1
    total += 1
print("TFLite Model Accuracy (Subset):", (correct / total).round(4))

# %% [6. Edge Deployment: TensorFlow.js]
# Note: TensorFlow.js conversion requires `tensorflowjs` package.
print("\nTensorFlow.js Conversion Instructions:")
print("1. Install: `pip install tensorflowjs`")
print(f"2. Convert: `tensorflowjs_converter --input_format=tf_saved_model {saved_model_path} tfjs_model`")
print("3. Use in browser: Load `tfjs_model/model.json` with TensorFlow.js")

# %% [7. Visualizing Predictions]
# Visualize predictions from the original model.
predictions = model.predict(x_test[:5])
plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i, :, :, 0], cmap='gray')
    plt.title(f"Pred: {np.argmax(predictions[i])}\nTrue: {y_test[i]}")
    plt.axis('off')
plt.savefig('deployment_predictions.png')

# %% [8. Interview Scenario: Model Deployment]
# Discuss deploying a model for production.
print("\nInterview Scenario: Model Deployment")
print("1. SavedModel: Standard format for TensorFlow Serving.")
print("2. TensorFlow Serving: Scalable REST/gRPC API for production.")
print("3. TensorFlow Lite: Lightweight for mobile/edge devices.")
print("4. TensorFlow.js: Browser-based inference with WebGL.")