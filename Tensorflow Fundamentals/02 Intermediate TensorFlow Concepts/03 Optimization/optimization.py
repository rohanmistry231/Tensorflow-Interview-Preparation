import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.mixed_precision as mixed_precision
import os

# %% [1. Introduction to Optimization]
# Optimization in TensorFlow involves tuning hyperparameters, applying regularization,
# using mixed precision training, and model quantization for performance and efficiency.

print("TensorFlow version:", tf.__version__)

# %% [2. Preparing the Dataset]
# Load and preprocess CIFAR-10 dataset.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
print("\nCIFAR-10 Dataset:")
print("Train Shape:", x_train.shape, "Test Shape:", x_test.shape)

# Create tf.data.Dataset pipelines
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32).prefetch(tf.data.AUTOTUNE)
print("Dataset Pipelines Created")

# %% [3. Base Model Definition]
# Define a CNN model for CIFAR-10 classification.
def create_cnn_model(dropout_rate=0.0, l2_lambda=0.0):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3),
                               kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# %% [4. Hyperparameter Tuning: Learning Rate and Batch Size]
# Manually tune learning rate and batch size.
learning_rates = [0.001, 0.0001]
batch_sizes = [32, 64]
tuning_results = []

for lr in learning_rates:
    for bs in batch_sizes:
        print(f"\nTuning: Learning Rate = {lr}, Batch Size = {bs}")
        train_ds_tune = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(bs).prefetch(tf.data.AUTOTUNE)
        model = create_cnn_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(train_ds_tune, epochs=5, validation_data=test_ds, verbose=0)
        val_acc = history.history['val_accuracy'][-1]
        tuning_results.append({'lr': lr, 'bs': bs, 'val_acc': val_acc})
        print(f"Validation Accuracy: {val_acc:.4f}")

# Select best hyperparameters
best_result = max(tuning_results, key=lambda x: x['val_acc'])
print("\nBest Hyperparameters:", best_result)

# %% [5. Regularization: Dropout and L2]
# Train model with dropout and L2 regularization using best hyperparameters.
dropout_rate = 0.3
l2_lambda = 0.01
model_reg = create_cnn_model(dropout_rate=dropout_rate, l2_lambda=l2_lambda)
model_reg.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_result['lr']),
                  loss='categorical_crossentropy', metrics=['accuracy'])
print("\nRegularized Model Summary:")
model_reg.summary()
reg_history = model_reg.fit(train_ds, epochs=5, validation_data=test_ds, verbose=1)
print("Regularized Model Test Accuracy:", reg_history.history['val_accuracy'][-1].round(4))

# %% [6. Mixed Precision Training]
# Enable mixed precision training for faster computation.
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print("\nMixed Precision Policy:", policy.name)

# Train model with mixed precision
model_mp = create_cnn_model(dropout_rate=dropout_rate, l2_lambda=l2_lambda)
model_mp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_result['lr']),
                 loss='categorical_crossentropy', metrics=['accuracy'])
model_mp.output.dtype = tf.float32  # Ensure output is float32 for compatibility
print("\nMixed Precision Model Summary:")
model_mp.summary()
mp_history = model_mp.fit(train_ds, epochs=5, validation_data=test_ds, verbose=1)
print("Mixed Precision Test Accuracy:", mp_history.history['val_accuracy'][-1].round(4))

# Reset policy to default
mixed_precision.set_global_policy('float32')

# %% [7. Model Quantization]
# Quantize the regularized model for deployment using TensorFlow Lite.
converter = tf.lite.TFLiteConverter.from_keras_model(model_reg)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save quantized model
quantized_model_path = 'quantized_model.tflite'
with open(quantized_model_path, 'wb') as f:
    f.write(tflite_model)
print("\nQuantized Model Saved to:", quantized_model_path)

# Evaluate quantized model (simplified evaluation)
interpreter = tf.lite.Interpreter(model_path=quantized_model_path)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
correct = 0
total = 0
for x, y in test_ds.unbatch().take(100):  # Evaluate on subset
    x = x.numpy()[np.newaxis, ...]
    interpreter.set_tensor(input_index, x)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)
    if np.argmax(pred) == np.argmax(y):
        correct += 1
    total += 1
quantized_acc = correct / total
print("Quantized Model Accuracy (Subset):", quantized_acc.round(4))

# %% [8. Visualizing Training Progress]
# Plot validation accuracy for regularized and mixed precision models.
plt.figure()
plt.plot(reg_history.history['val_accuracy'], label='Regularized')
plt.plot(mp_history.history['val_accuracy'], label='Mixed Precision')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Optimization Comparison')
plt.legend()
plt.savefig('optimization_comparison.png')

# Plot hyperparameter tuning results
plt.figure()
for result in tuning_results:
    plt.scatter(result['lr'], result['bs'], s=100, c=result['val_acc'], cmap='viridis')
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Batch Size')
plt.title('Hyperparameter Tuning Results')
plt.colorbar(label='Validation Accuracy')
plt.savefig('hyperparameter_tuning.png')

# %% [9. Interview Scenario: Optimization Strategy]
# Discuss optimization strategies for a high-accuracy, efficient model.
print("\nInterview Scenario: Optimization Strategy")
print("1. Hyperparameter Tuning: Test learning rates (e.g., 1e-3, 1e-4) and batch sizes (e.g., 32, 64).")
print("2. Regularization: Use dropout (0.3) and L2 (0.01) to prevent overfitting.")
print("3. Mixed Precision: Enable mixed_float16 for faster training on GPUs.")
print("4. Quantization: Apply post-training quantization for efficient deployment.")
print("Tools: KerasTuner for automated hyperparameter tuning.")

# %% [10. Practical Application: Combined Optimization]
# Train a model with all optimizations combined.
model_combined = create_cnn_model(dropout_rate=dropout_rate, l2_lambda=l2_lambda)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
model_combined.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_result['lr']),
                       loss='categorical_crossentropy', metrics=['accuracy'])
model_combined.output.dtype = tf.float32
print("\nCombined Optimization Model Summary:")
model_combined.summary()
combined_history = model_combined.fit(train_ds, epochs=5, validation_data=test_ds, verbose=1)
print("Combined Optimization Test Accuracy:", combined_history.history['val_accuracy'][-1].round(4))

# Save and quantize combined model
converter = tf.lite.TFLiteConverter.from_keras_model(model_combined)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_combined_model = converter.convert()
with open('quantized_combined_model.tflite', 'wb') as f:
    f.write(tflite_combined_model)
print("Quantized Combined Model Saved to: quantized_combined_model.tflite")

# Reset policy
mixed_precision.set_global_policy('float32')