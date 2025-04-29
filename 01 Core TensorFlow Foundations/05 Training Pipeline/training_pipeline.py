import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# %% [1. Introduction to Training Pipeline]
# A training pipeline manages model training, evaluation, checkpointing, and monitoring.
# TensorFlow supports training/evaluation loops, model.save/load, GPU/TPU training, and TensorBoard.

print("TensorFlow version:", tf.__version__)

# %% [2. Preparing the Dataset]
# Load and preprocess MNIST dataset.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
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

# %% [3. Defining the Model]
# Create a CNN model using tf.keras.Sequential.
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
print("\nModel Summary:")
model.summary()

# %% [4. Training/Evaluation Loops]
# Compile and train the model with a custom training loop.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_acc_metric.update_state(y, logits)
    return loss

@tf.function
def test_step(x, y):
    logits = model(x, training=False)
    val_acc_metric.update_state(y, logits)

# Training loop
epochs = 5
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(log_dir)
history = {'loss': [], 'accuracy': [], 'val_accuracy': []}

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_loss = 0.0
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()
    
    # Training
    for step, (x_batch, y_batch) in enumerate(train_ds):
        loss = train_step(x_batch, y_batch)
        train_loss += loss
        if step % 200 == 0:
            print(f"Step {step}, Loss: {loss.numpy():.4f}, Accuracy: {train_acc_metric.result().numpy():.4f}")
    
    # Evaluation
    for x_batch, y_batch in test_ds:
        test_step(x_batch, y_batch)
    
    # Log metrics to TensorBoard
    with summary_writer.as_default():
        tf.summary.scalar('loss', train_loss / (step + 1), step=epoch)
        tf.summary.scalar('accuracy', train_acc_metric.result(), step=epoch)
        tf.summary.scalar('val_accuracy', val_acc_metric.result(), step=epoch)
    
    history['loss'].append(train_loss.numpy() / (step + 1))
    history['accuracy'].append(train_acc_metric.result().numpy())
    history['val_accuracy'].append(val_acc_metric.result().numpy())
    print(f"Epoch {epoch + 1}, Loss: {history['loss'][-1]:.4f}, Accuracy: {history['accuracy'][-1]:.4f}, Val Accuracy: {history['val_accuracy'][-1]:.4f}")

# %% [5. Model Checkpointing]
# Save and load the model using model.save and model.load.
checkpoint_dir = "checkpoints/mnist_cnn"
os.makedirs(checkpoint_dir, exist_ok=True)
model.save(os.path.join(checkpoint_dir, "model"))
print("\nModel Saved to:", checkpoint_dir)

# Load and evaluate the saved model
loaded_model = tf.keras.models.load_model(os.path.join(checkpoint_dir, "model"))
loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
loss, acc = loaded_model.evaluate(test_ds, verbose=0)
print("Loaded Model Test Loss:", loss.round(4), "Test Accuracy:", acc.round(4))

# %% [6. GPU/TPU Training with tf.device]
# Train on GPU if available, otherwise CPU.
device = '/CPU:0'
if tf.config.list_physical_devices('GPU'):
    device = '/GPU:0'
    print("\nTraining on GPU")
elif tf.config.list_physical_devices('TPU'):
    device = '/TPU:0'
    print("\nTraining on TPU (Note: Typically requires cloud environment like Colab)")
else:
    print("\nTraining on CPU")

# Re-train a small model on the selected device for demonstration
small_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
small_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

with tf.device(device):
    small_model.fit(train_ds, epochs=3, validation_data=test_ds, verbose=1)
print("Device Training Completed")

# %% [7. Monitoring with TensorBoard]
# TensorBoard logs are saved in log_dir.
print("\nTensorBoard Monitoring:")
print(f"Run: tensorboard --logdir {log_dir}")
print("Then open http://localhost:6006 in your browser to view metrics.")

# %% [8. Visualizing Training Progress]
# Plot training and validation accuracy.
plt.figure()
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('MNIST CNN Training Progress')
plt.legend()
plt.savefig('training_progress.png')

# Plot training loss
plt.figure()
plt.plot(history['loss'], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MNIST CNN Training Loss')
plt.legend()
plt.savefig('training_loss.png')

# %% [9. Interview Scenario: Custom Training Loop]
# Implement a custom training loop with gradient clipping.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)  # Gradient clipping
model_clip = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

@tf.function
def train_step_clip(x, y):
    with tf.GradientTape() as tape:
        logits = model_clip(x, training=True)
        loss = loss_fn(y, logits)
    gradients = tape.gradient(loss, model_clip.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_clip.trainable_variables))
    train_acc_metric.update_state(y, logits)
    return loss

print("\nInterview Scenario: Custom Training Loop with Gradient Clipping")
for epoch in range(2):  # Short loop for demonstration
    train_acc_metric.reset_states()
    for x_batch, y_batch in train_ds:
        loss = train_step_clip(x_batch, y_batch)
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy():.4f}, Accuracy: {train_acc_metric.result().numpy():.4f}")

# %% [10. Practical Application: Checkpoint Callback]
# Use ModelCheckpoint callback to save the best model during training.
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "best_model"),
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)
model.fit(train_ds, epochs=3, validation_data=test_ds, callbacks=[checkpoint_callback], verbose=1)
print("\nBest Model Saved with ModelCheckpoint Callback")