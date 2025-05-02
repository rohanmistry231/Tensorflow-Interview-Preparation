import tensorflow as tf
import numpy as np

# %% [1. Introduction to Tensors and Operations]
# Tensors are multi-dimensional arrays, the core data structure in TensorFlow.
# TensorFlow provides functions for tensor creation, manipulation, and operations.

print("TensorFlow version:", tf.__version__)

# %% [2. Tensor Creation]
# Create tensors using tf.constant, tf.zeros, tf.ones, and tf.random.
# tf.constant: Creates a tensor from a fixed value.
const_tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
print("\nConstant Tensor:")
print(const_tensor)

# tf.zeros: Creates a tensor filled with zeros.
zeros_tensor = tf.zeros([2, 3], dtype=tf.int32)
print("\nZeros Tensor:")
print(zeros_tensor)

# tf.random: Creates a tensor with random values.
random_tensor = tf.random.normal([2, 2], mean=0.0, stddev=1.0, seed=42)
print("\nRandom Normal Tensor:")
print(random_tensor)

# %% [3. Tensor Attributes]
# Tensors have attributes: shape, dtype, and device.
print("\nTensor Attributes for const_tensor:")
print("Shape:", const_tensor.shape)
print("Data Type:", const_tensor.dtype)
print("Device:", const_tensor.device)

# %% [4. Indexing and Slicing]
# Access tensor elements using indexing and slicing.
tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\nOriginal Tensor:")
print(tensor)
print("First Row:", tensor[0].numpy())
print("Element at [1, 2]:", tensor[1, 2].numpy())
print("Slice [0:2, 1:3]:")
print(tensor[0:2, 1:3].numpy())

# %% [5. Reshaping]
# Reshape tensors using tf.reshape.
reshaped_tensor = tf.reshape(tensor, [1, 9])
print("\nReshaped Tensor (1x9):")
print(reshaped_tensor)
reshaped_tensor_2 = tf.reshape(tensor, [9, 1])
print("Reshaped Tensor (9x1):")
print(reshaped_tensor_2)

# %% [6. Matrix Multiplication]
# Perform matrix multiplication using tf.matmul.
matrix_a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
matrix_b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
matmul_result = tf.matmul(matrix_a, matrix_b)
print("\nMatrix A:")
print(matrix_a)
print("Matrix B:")
print(matrix_b)
print("Matrix Multiplication (A @ B):")
print(matmul_result)

# %% [7. Broadcasting]
# Broadcasting allows operations on tensors of different shapes.
scalar = tf.constant(2.0, dtype=tf.float32)
broadcast_result = matrix_a * scalar
print("\nBroadcasting (Matrix A * Scalar):")
print(broadcast_result)

# Example with different shapes
tensor_1 = tf.constant([[1, 2, 3]], dtype=tf.float32)
tensor_2 = tf.constant([[4], [5], [6]], dtype=tf.float32)
broadcast_sum = tensor_1 + tensor_2
print("Broadcasting (1x3 + 3x1):")
print(broadcast_sum)

# %% [8. CPU/GPU Interoperability]
# TensorFlow automatically places tensors on GPU if available, or CPU otherwise.
# Explicitly place operations on CPU or GPU using tf.device.
with tf.device('/CPU:0'):
    cpu_tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    cpu_result = tf.matmul(cpu_tensor, cpu_tensor)
print("\nCPU Tensor Device:", cpu_tensor.device)
print("CPU Matmul Result:")
print(cpu_result)

if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        gpu_tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        gpu_result = tf.matmul(gpu_tensor, gpu_tensor)
    print("GPU Tensor Device:", gpu_tensor.device)
    print("GPU Matmul Result:")
    print(gpu_result)
else:
    print("No GPU available, skipping GPU test.")

# %% [9. NumPy Integration]
# Convert between TensorFlow tensors and NumPy arrays.
numpy_array = np.array([[1, 2], [3, 4]])
tensor_from_numpy = tf.convert_to_tensor(numpy_array, dtype=tf.float32)
print("\nTensor from NumPy Array:")
print(tensor_from_numpy)

# Convert tensor back to NumPy
numpy_from_tensor = tensor_from_numpy.numpy()
print("NumPy Array from Tensor:")
print(numpy_from_tensor)

# Perform NumPy-style operations
numpy_result = np.matmul(numpy_array, numpy_array)
tensor_result = tf.matmul(tensor_from_numpy, tensor_from_numpy)
print("NumPy Matmul Result:")
print(numpy_result)
print("TensorFlow Matmul Result:")
print(tensor_result)

# %% [10. Interview Scenario: Tensor Manipulation]
# Demonstrate a practical tensor manipulation task.
# Task: Create a 3x3 tensor, extract its diagonal, and compute its sum.
interview_tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
diagonal = tf.linalg.diag_part(interview_tensor)
diagonal_sum = tf.reduce_sum(diagonal)
print("\nInterview Task:")
print("Original Tensor:")
print(interview_tensor)
print("Diagonal:", diagonal.numpy())
print("Sum of Diagonal:", diagonal_sum.numpy())

# Visualize tensor as a heatmap
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(interview_tensor, cmap='viridis')
plt.colorbar()
plt.title('Tensor Heatmap')
plt.savefig('tensor_heatmap.png')