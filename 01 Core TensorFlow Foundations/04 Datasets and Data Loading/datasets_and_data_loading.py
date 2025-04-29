import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
try:
    import tensorflow_datasets as tfds
except ImportError:
    tfds = None
from sklearn.preprocessing import StandardScaler

# %% [1. Introduction to Datasets and Data Loading]
# Efficient data loading and preprocessing are critical for ML training.
# TensorFlow provides tf.keras.datasets, tfds.load, tf.data.Dataset, and tf.keras.preprocessing.

print("TensorFlow version:", tf.__version__)

# %% [2. Built-in Datasets with tf.keras.datasets]
# Load MNIST dataset from tf.keras.datasets.
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()
print("\nMNIST Dataset:")
print("Train Shape:", x_train_mnist.shape, "Test Shape:", x_test_mnist.shape)
print("Label Example:", y_train_mnist[:5])

# Normalize pixel values to [0, 1]
x_train_mnist = x_train_mnist.astype('float32') / 255.0
x_test_mnist = x_test_mnist.astype('float32') / 255.0
print("Normalized Train Data (first sample, first row):", x_train_mnist[0, 0, :5])

# %% [3. TensorFlow Datasets with tfds.load]
# Load CIFAR-10 dataset using tensorflow-datasets (if installed).
if tfds is not None:
    ds_cifar, info = tfds.load('cifar10', with_info=True, as_supervised=True)
    ds_cifar_train = ds_cifar['train']
    ds_cifar_test = ds_cifar['test']
    print("\nCIFAR-10 Dataset Info:")
    print("Features:", info.features)
    print("Number of Training Examples:", info.splits['train'].num_examples)
    
    # Example: Extract one batch
    for image, label in ds_cifar_train.take(1):
        print("Sample Image Shape:", image.shape, "Label:", label.numpy())
else:
    print("\ntensorflow-datasets not installed. Install with: pip install tensorflow-datasets")
    ds_cifar_train = None

# %% [4. Data Pipeline with tf.data.Dataset]
# Create a tf.data.Dataset pipeline for MNIST.
mnist_train_ds = tf.data.Dataset.from_tensor_slices((x_train_mnist, y_train_mnist))

# Apply transformations: shuffle, batch, and preprocess
def preprocess_mnist(image, label):
    image = tf.image.random_brightness(image, max_delta=0.1)  # Data augmentation
    image = tf.expand_dims(image, axis=-1)  # Add channel dimension: (28, 28) -> (28, 28, 1)
    label = tf.cast(label, tf.int32)
    return image, label

mnist_train_ds = (mnist_train_ds
                  .map(preprocess_mnist, num_parallel_calls=tf.data.AUTOTUNE)
                  .shuffle(buffer_size=1000)
                  .batch(batch_size=32)
                  .prefetch(tf.data.AUTOTUNE))
print("\nMNIST tf.data.Dataset Pipeline Created:")
for image, label in mnist_train_ds.take(1):
    print("Batch Shape:", image.shape, "Label Shape:", label.shape)

# %% [5. Preprocessing with tf.keras.preprocessing]
# Use tf.keras.preprocessing for data augmentation on MNIST.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
])
# Apply augmentation to a sample image
sample_image = x_train_mnist[0:1][..., np.newaxis]  # Shape: (1, 28, 28, 1)
augmented_image = data_augmentation(sample_image)
print("\nAugmented Image Shape:", augmented_image.shape)

# Visualize original vs. augmented image
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(sample_image[0, :, :, 0], cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(augmented_image[0, :, :, 0], cmap='gray')
plt.title('Augmented Image')
plt.savefig('augmentation_comparison.png')

# %% [6. Handling Large Datasets]
# Simulate a large dataset with synthetic data and create an efficient pipeline.
np.random.seed(42)
large_x = np.random.rand(100000, 10).astype(np.float32)
large_y = np.random.randint(0, 2, 100000).astype(np.int32)
large_ds = tf.data.Dataset.from_tensor_slices((large_x, large_y))

# Preprocessing function
def preprocess_large(x, y):
    x = tf.cast(x, tf.float32)
    x = (x - tf.reduce_mean(x, axis=0)) / tf.math.reduce_std(x, axis=0)  # Standardize
    y = tf.cast(y, tf.int32)
    return x, y

# Efficient pipeline for large dataset
large_ds = (large_ds
            .map(preprocess_large, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(buffer_size=10000)
            .batch(batch_size=64)
            .prefetch(tf.data.AUTOTUNE))
print("\nLarge Dataset Pipeline:")
for x, y in large_ds.take(1):
    print("Batch Shape:", x.shape, "Label Shape:", y.shape)
    print("Standardized Features (first sample, first 5):", x[0, :5].numpy().round(4))

# %% [7. Practical Application: Training a Model with tf.data]
# Train a simple CNN on the MNIST pipeline.
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\nCNN Training on MNIST Pipeline:")
history = cnn_model.fit(mnist_train_ds, epochs=5, validation_data=(x_test_mnist[..., np.newaxis], y_test_mnist), verbose=1)
print("Final Validation Accuracy:", history.history['val_accuracy'][-1].round(4))

# %% [8. Visualizing Training Progress]
# Plot training and validation accuracy.
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('MNIST CNN Training Progress')
plt.legend()
plt.savefig('mnist_training_progress.png')

# %% [9. Interview Scenario: Optimizing Data Pipelines]
# Optimize a pipeline for a large image dataset.
def optimized_pipeline(dataset, batch_size=32):
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.random_flip_left_right(image)
        label = tf.cast(label, tf.int32)
        return image, label
    return (dataset
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()  # Cache in memory for small datasets
            .shuffle(buffer_size=1000)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))

print("\nInterview Scenario: Optimized Pipeline")
print("Key Optimizations: cache(), prefetch(), parallel map, appropriate shuffle buffer.")
if tfds is not None:
    sample_ds = tfds.load('cifar10', split='train', as_supervised=True)
    optimized_ds = optimized_pipeline(sample_ds)
    for image, label in optimized_ds.take(1):
        print("Optimized Batch Shape:", image.shape, "Label Shape:", label.shape)

# %% [10. Custom Preprocessing Function]
# Create a custom preprocessing function for a regression dataset.
np.random.seed(42)
X_reg = np.random.rand(1000, 5).astype(np.float32)
y_reg = np.sum(X_reg, axis=1) + np.random.normal(0, 0.1, 1000).astype(np.float32)
reg_ds = tf.data.Dataset.from_tensor_slices((X_reg, y_reg))

def custom_preprocess(x, y):
    x = tf.cast(x, tf.float32)
    x = (x - tf.reduce_mean(x)) / tf.math.reduce_std(x)  # Normalize
    y = tf.cast(y, tf.float32)
    return x, y

reg_ds = (reg_ds
          .map(custom_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
          .shuffle(buffer_size=100)
          .batch(batch_size=16)
          .prefetch(tf.data.AUTOTUNE))
print("\nCustom Preprocessing for Regression Dataset:")
for x, y in reg_ds.take(1):
    print("Batch Shape:", x.shape, "Label Shape:", y.shape)
    print("Normalized Features (first sample, first 5):", x[0, :5].numpy().round(4))