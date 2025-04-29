import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# %% [1. Introduction to Customization]
# TensorFlow allows customization via custom layers, loss functions, and model APIs.
# This file covers custom layers/losses, Functional/Subclassing APIs, and gradient debugging.

print("TensorFlow version:", tf.__version__)

# %% [2. Preparing the Dataset]
# Load and preprocess MNIST dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train[..., np.newaxis]  # Shape: (60000, 28, 28, 1)
x_test = x_test[..., np.newaxis]    # Shape: (10000, 28, 28, 1)
print("\nMNIST Dataset:")
print("Train Shape:", x_train.shape, "Test Shape:", x_test.shape)

# Create tf.data.Dataset pipelines
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32).prefetch(tf.data.AUTOTUNE)
print("Dataset Pipelines Created")

# %% [3. Custom Layers]
# Define a custom layer that applies a learnable scaling factor to a Dense layer.
class ScaledDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None):
        super(ScaledDense, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(self.units)
        self.scale = self.add_weight('scale', shape=(), initializer='ones', trainable=True)
    
    def call(self, inputs):
        x = self.dense(inputs)
        x = x * self.scale
        if self.activation is not None:
            x = self.activation(x)
        return x

# Test custom layer in a simple model
custom_layer_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    ScaledDense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
custom_layer_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\nCustom Layer Model Summary:")
custom_layer_model.summary()
custom_layer_history = custom_layer_model.fit(train_ds, epochs=3, validation_data=test_ds, verbose=1)
print("Custom Layer Test Accuracy:", custom_layer_history.history['val_accuracy'][-1].round(4))

# %% [4. Custom Loss Functions]
# Define a custom loss function: weighted categorical crossentropy.
class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)
    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -tf.reduce_sum(
            tf.one_hot(y_true, depth=tf.shape(y_pred)[-1]) * tf.math.log(y_pred), axis=-1)
        weights = tf.gather(self.class_weights, y_true)
        return tf.reduce_mean(cross_entropy * weights)

# Test custom loss (emphasize class 0)
class_weights = [2.0] + [1.0] * 9  # Weight class 0 higher
custom_loss = WeightedCategoricalCrossentropy(class_weights)
custom_loss_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
custom_loss_model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
print("\nCustom Loss Model Summary:")
custom_loss_model.summary()
custom_loss_history = custom_loss_model.fit(train_ds, epochs=3, validation_data=test_ds, verbose=1)
print("Custom Loss Test Accuracy:", custom_loss_history.history['val_accuracy'][-1].round(4))

# %% [5. Functional API]
# Build a model using the Functional API for more complex architectures.
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
functional_model = tf.keras.Model(inputs, outputs)
functional_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\nFunctional API Model Summary:")
functional_model.summary()
functional_history = functional_model.fit(train_ds, epochs=3, validation_data=test_ds, verbose=1)
print("Functional API Test Accuracy:", functional_history.history['val_accuracy'][-1].round(4))

# %% [6. Subclassing API]
# Define a custom model using the Subclassing API for maximum flexibility.
class CustomCNN(tf.keras.Model):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

subclass_model = CustomCNN()
subclass_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\nSubclassing API Model Summary:")
subclass_model.summary()
subclass_history = subclass_model.fit(train_ds, epochs=3, validation_data=test_ds, verbose=1)
print("Subclassing API Test Accuracy:", subclass_history.history['val_accuracy'][-1].round(4))

# %% [7. Debugging Gradient Issues]
# Demonstrate common gradient issues and debugging techniques.
# Case 1: Non-differentiable operation (tf.cast to int).
x = tf.Variable(1.0)
with tf.GradientTape() as tape:
    y = tf.cast(x, tf.int32)  # Non-differentiable
    loss = tf.square(y)
grad = tape.gradient(loss, x)
print("\nGradient Debugging - Non-Differentiable Operation:")
print("Operation: y = cast(x to int), loss = y^2")
print("Gradient:", grad)  # Expected: None
print("Fix: Avoid non-differentiable ops (e.g., use float operations).")

# Case 2: Disconnected graph (no dependency).
x = tf.Variable(1.0)
with tf.GradientTape() as tape:
    y = tf.stop_gradient(x)  # Blocks gradient flow
    loss = tf.square(y)
grad = tape.gradient(loss, x)
print("\nGradient Debugging - Disconnected Graph:")
print("Operation: y = stop_gradient(x), loss = y^2")
print("Gradient:", grad)  # Expected: None
print("Fix: Ensure variables are part of the computational graph.")

# Case 3: Monitor gradient norms.
optimizer = tf.keras.optimizers.Adam()
gradient_norms = []
for epoch in range(2):
    with tf.GradientTape() as tape:
        logits = subclass_model(x_train[:32], training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_train[:32], logits)
    gradients = tape.gradient(loss, subclass_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, subclass_model.trainable_variables))
    grad_norm = tf.sqrt(sum(tf.norm(g) ** 2 for g in gradients if g is not None))
    gradient_norms.append(grad_norm.numpy())
print("\nGradient Norms:", gradient_norms)

# %% [8. Visualizing Training Progress]
# Plot validation accuracy for all models.
plt.figure()
plt.plot(custom_layer_history.history['val_accuracy'], label='Custom Layer')
plt.plot(custom_loss_history.history['val_accuracy'], label='Custom Loss')
plt.plot(functional_history.history['val_accuracy'], label='Functional API')
plt.plot(subclass_history.history['val_accuracy'], label='Subclassing API')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Customization Model Comparison')
plt.legend()
plt.savefig('customization_comparison.png')

# Plot gradient norms
plt.figure()
plt.plot(gradient_norms, label='Gradient Norm')
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norm During Training')
plt.legend()
plt.savefig('gradient_norms.png')

# %% [9. Interview Scenario: Custom Layer Implementation]
# Implement a custom layer with a learnable polynomial transformation.
class PolynomialLayer(tf.keras.layers.Layer):
    def __init__(self, degree):
        super(PolynomialLayer, self).__init__()
        self.degree = degree
    
    def build(self, input_shape):
        self.coefficients = self.add_weight('coefficients', shape=(self.degree + 1,), 
                                           initializer='random_normal', trainable=True)
    
    def call(self, inputs):
        x = inputs
        result = 0
        for i in range(self.degree + 1):
            result += self.coefficients[i] * tf.pow(x, float(i))
        return result

# Test polynomial layer
poly_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    PolynomialLayer(degree=2),
    tf.keras.layers.Dense(10, activation='softmax')
])
poly_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\nInterview Scenario: Polynomial Layer Model Summary:")
poly_model.summary()
poly_history = poly_model.fit(train_ds, epochs=3, validation_data=test_ds, verbose=1)
print("Polynomial Layer Test Accuracy:", poly_history.history['val_accuracy'][-1].round(4))

# %% [10. Practical Application: Combining Customizations]
# Combine custom layer, loss, and Subclassing API for MNIST classification.
class CustomCombinedModel(tf.keras.Model):
    def __init__(self):
        super(CustomCombinedModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.scaled_dense = ScaledDense(32, activation='relu')
        self.dense = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.scaled_dense(x)
        return self.dense(x)

combined_model = CustomCombinedModel()
combined_model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
print("\nCombined Custom Model Summary:")
combined_model.summary()
combined_history = combined_model.fit(train_ds, epochs=3, validation_data=test_ds, verbose=1)
print("Combined Custom Model Test Accuracy:", combined_history.history['val_accuracy'][-1].round(4))