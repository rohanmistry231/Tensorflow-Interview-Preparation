import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# %% [1. Introduction to Custom Extensions]
# Custom extensions enhance TensorFlow with custom gradients, addons, and optimizers.
# This file demonstrates custom gradient functions, TensorFlow Addons, and a custom optimizer.

print("TensorFlow version:", tf.__version__)

# %% [2. Preparing the Dataset]
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

# %% [3. Custom Gradient Functions]
# Define a custom gradient for a clipping operation.
@tf.custom_gradient
def clip_by_value(x, clip_min, clip_max):
    y = tf.clip_by_value(x, clip_min, clip_max)
    def grad(dy):
        return dy * tf.where((x >= clip_min) & (x <= clip_max), 1.0, 0.0), None, None
    return y, grad

# Test custom gradient in a model
model_custom_grad = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
optimizer = tf.keras.optimizers.Adam()
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model_custom_grad(x)
        logits = clip_by_value(logits, 1e-7, 1.0 - 1e-7)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits)
    gradients = tape.gradient(loss, model_custom_grad.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_custom_grad.trainable_variables))
    return loss

print("\nCustom Gradient Training:")
for epoch in range(3):
    for x, y in train_ds:
        loss = train_step(x, y)
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy():.4f}")

# %% [4. TensorFlow Addons]
# Note: TensorFlow Addons is deprecated; use Keras 3 or alternatives for advanced metrics/losses.
# Example: Custom loss inspired by Addons (e.g., focal loss).
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_mean(weight * ce)

model_addons = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model_addons.compile(optimizer='adam', loss=FocalLoss(), metrics=['accuracy'])
print("\nFocal Loss Model Summary:")
model_addons.summary()
addons_history = model_addons.fit(train_ds, epochs=3, validation_data=test_ds, verbose=1)
print("Focal Loss Test Accuracy:", addons_history.history['val_accuracy'][-1].round(4))

# %% [5. Custom Optimizers]
# Define a custom optimizer with momentum.
class CustomMomentumOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, name="CustomMomentum"):
        super().__init__(name=name)
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'velocity', initializer='zeros')
    
    def _resource_apply_dense(self, grad, var, apply_state=None):
        velocity = self.get_slot(var, 'velocity')
        velocity_t = velocity * self.momentum - self.learning_rate * grad
        var_t = var + velocity_t
        velocity.assign(velocity_t)
        var.assign(var_t)
        return tf.no_op()

model_custom_opt = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model_custom_opt.compile(optimizer=CustomMomentumOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\nCustom Optimizer Model Summary:")
model_custom_opt.summary()
custom_opt_history = model_custom_opt.fit(train_ds, epochs=3, validation_data=test_ds, verbose=1)
print("Custom Optimizer Test Accuracy:", custom_opt_history.history['val_accuracy'][-1].round(4))

# %% [6. Visualizing Training Progress]
# Plot validation accuracy for models.
plt.figure()
plt.plot(addons_history.history['val_accuracy'], label='Focal Loss')
plt.plot(custom_opt_history.history['val_accuracy'], label='Custom Optimizer')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Custom Extensions Comparison')
plt.legend()
plt.savefig('custom_extensions_comparison.png')

# %% [7. Interview Scenario: Custom Gradient]
# Discuss implementing a custom gradient for a non-standard operation.
print("\nInterview Scenario: Custom Gradient")
print("Use @tf.custom_gradient to define forward and backward passes.")
print("Example: Clip operation with gradients only in valid range.")
print("Key: Ensure gradient function matches operation’s logic.")