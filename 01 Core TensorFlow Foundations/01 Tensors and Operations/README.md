# Tensors and Operations (`tensorflow`)

## 📖 Introduction
Tensors are the core data structure in TensorFlow, representing multi-dimensional arrays. This guide covers tensor creation, attributes, operations, CPU/GPU interoperability, and NumPy integration, with practical examples and interview insights.

## 🎯 Learning Objectives
- Understand TensorFlow tensors and their properties.
- Master tensor creation (`tf.constant`, `tf.zeros`, `tf.random`) and manipulation.
- Perform operations like indexing, reshaping, matrix multiplication, and broadcasting.
- Explore CPU/GPU interoperability and NumPy integration.

## 🔑 Key Concepts
- **Tensor Creation**: Use `tf.constant`, `tf.zeros`, `tf.ones`, `tf.random` to create tensors.
- **Attributes**: Shape, dtype, and device define tensor properties.
- **Operations**: Indexing, reshaping, matrix multiplication (`tf.matmul`), and broadcasting.
- **CPU/GPU Interoperability**: TensorFlow manages device placement (`tf.device`).
- **NumPy Integration**: Seamless conversion between tensors and NumPy arrays.

## 📝 Example Walkthrough
The `tensors_and_operations.py` file demonstrates:
1. **Tensor Creation**: Creating constant, zero, and random tensors.
2. **Attributes**: Inspecting shape, dtype, and device.
3. **Operations**: Indexing, reshaping, matrix multiplication, and broadcasting.
4. **Interoperability**: Running operations on CPU/GPU and converting to/from NumPy.
5. **Visualization**: Plotting a tensor as a heatmap.

Example code:
```python
import tensorflow as tf
const_tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
matmul_result = tf.matmul(const_tensor, const_tensor)
```

## 🛠️ Practical Tasks
1. Create a 2x3 tensor using `tf.random.normal` and print its shape and dtype.
2. Reshape a 4x4 tensor into a 2x8 tensor and verify the result.
3. Perform matrix multiplication on two 3x3 tensors and check the output.
4. Convert a NumPy array to a TensorFlow tensor, perform an operation, and convert back.
5. Run a matrix multiplication on CPU and GPU (if available) using `tf.device`.

## 💡 Interview Tips
- **Common Questions**:
  - What is a TensorFlow tensor, and how does it differ from a NumPy array?
  - How does broadcasting work in TensorFlow?
  - Why is device placement important in TensorFlow?
- **Tips**:
  - Explain tensor attributes (shape, dtype, device) clearly.
  - Highlight broadcasting’s role in handling shape mismatches.
  - Be ready to code a tensor manipulation task (e.g., extract diagonal, compute sum).

## 📚 Resources
- [TensorFlow Core Guide](https://www.tensorflow.org/guide/tensor)
- [TensorFlow API Documentation](https://www.tensorflow.org/api_docs/python/tf)
- [Kaggle: TensorFlow Tutorials](https://www.kaggle.com/learn/intro-to-deep-learning)