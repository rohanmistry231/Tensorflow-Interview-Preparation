import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# %% [1. Introduction to Distributed Training]
# Distributed training scales TensorFlow models across multiple GPUs/TPUs.
# Covers Data Parallelism, Multi-GPU/TPU Training, and Distributed Datasets.

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
batch_size = 64
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# %% [3. Data Parallelism with MirroredStrategy]
# Use MirroredStrategy for data parallelism across GPUs.
mirrored_strategy = tf.distribute.MirroredStrategy()
print("\nMirroredStrategy Devices:", mirrored_strategy.num_replicas_in_sync)

# Define and compile model within strategy scope
with mirrored_strategy.scope():
    model_mirrored = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model_mirrored.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nMirroredStrategy Model Summary:")
model_mirrored.summary()
mirrored_history = model_mirrored.fit(train_ds, epochs=5, validation_data=test_ds, verbose=1)
print("MirroredStrategy Test Accuracy:", mirrored_history.history['val_accuracy'][-1].round(4))

# %% [4. Multi-GPU/TPU Training with TPUStrategy]
# Use TPUStrategy (fallback to MirroredStrategy if TPU unavailable).
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    tpu_strategy = tf.distribute.TPUStrategy(resolver)
    print("\nTPUStrategy Initialized")
except ValueError:
    tpu_strategy = mirrored_strategy
    print("\nTPU Unavailable, Using MirroredStrategy")

with tpu_strategy.scope():
    model_tpu = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model_tpu.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nTPUStrategy Model Summary:")
model_tpu.summary()
tpu_history = model_tpu.fit(train_ds, epochs=5, validation_data=test_ds, verbose=1)
print("TPUStrategy Test Accuracy:", tpu_history.history['val_accuracy'][-1].round(4))

# %% [5. Distributed Datasets]
# Optimize dataset for distributed training.
dist_train_ds = mirrored_strategy.experimental_distribute_dataset(train_ds)
dist_test_ds = mirrored_strategy.experimental_distribute_dataset(test_ds)
print("\nDistributed Dataset Created")

# Custom training loop for distributed dataset
@tf.function
def dist_train_step(inputs):
    def step_fn(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            logits = model_mirrored(x, training=True)
            loss = tf.keras.losses.categorical_crossentropy(y, logits)
        gradients = tape.gradient(loss, model_mirrored.trainable_variables)
        model_mirrored.optimizer.apply_gradients(zip(gradients, model_mirrored.trainable_variables))
        return loss
    per_replica_losses = mirrored_strategy.run(step_fn, args=(inputs,))
    return mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

print("\nCustom Distributed Training Loop:")
for epoch in range(2):
    total_loss = 0.0
    num_batches = 0
    for inputs in dist_train_ds:
        total_loss += dist_train_step(inputs)
        num_batches += 1
    print(f"Epoch {epoch + 1}, Average Loss: {total_loss / num_batches:.4f}")

# %% [6. Visualizing Training Progress]
# Plot validation accuracy for MirroredStrategy and TPUStrategy.
plt.figure()
plt.plot(mirrored_history.history['val_accuracy'], label='MirroredStrategy')
plt.plot(tpu_history.history['val_accuracy'], label='TPUStrategy')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Distributed Training Comparison')
plt.legend()
plt.savefig('distributed_training_comparison.png')

# %% [7. Interview Scenario: Scaling Training]
# Discuss strategies for scaling training to multiple devices.
print("\nInterview Scenario: Scaling Training")
print("1. MirroredStrategy: Synchronous data parallelism for multi-GPU setups.")
print("2. TPUStrategy: Optimized for TPU clusters in cloud environments.")
print("3. Distributed Datasets: Use experimental_distribute_dataset for efficient data sharding.")
print("Key: Ensure model and data pipeline are compatible with strategy scope.")