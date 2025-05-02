# Automatic Differentiation (`tensorflow`)

## 📖 Introduction
Automatic differentiation is a cornerstone of TensorFlow, enabling gradient-based optimization for machine learning models. This guide covers computational graphs, gradient computation with `tf.GradientTape`, gradient application using `optimizer.apply_gradients`, and no-gradient context with `tf.stop_gradient`, with practical examples and interview insights.

## 🎯 Learning Objectives
- Understand computational graphs and their role in differentiation.
- Master `tf.GradientTape` for gradient computation.
- Apply gradients using optimizers for model training.
- Use `tf.stop_gradient` to control gradient flow.

## 🔑 Key Concepts
- **Computational Graphs**: Track operations for automatic differentiation.
- **tf.GradientTape**: Records operations dynamically to compute gradients.
- **Gradient Application**: Optimizers (e.g., SGD, Adam) update variables using gradients.
- **tf.stop_gradient**: Prevents gradients from flowing through specified tensors.

## 📝 Example Walkthrough
The `automatic_differentiation.py` file demonstrates:
1. **Computational Graphs**: Computing gradients for a polynomial.
2. **Gradient Computation**: Gradients for a linear layer using `tf.GradientTape`.
3. **Higher-Order Gradients**: Second derivatives with nested tapes.
4. **Gradient Application**: Optimizing a linear regression model.
5. **No-Gradient Context**: Using `tf.stop_gradient` to block gradient flow.
6. **Visualization**: Plotting loss curves and model fits.

Example code:
```python
import tensorflow as tf
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**2
dy_dx = tape.gradient(y, x)
```

## 🛠️ Practical Tasks
1. Compute the gradient of `y = x^3 + 2x` at `x = 2` using `tf.GradientTape`.
2. Train a linear regression model with `tf.GradientTape` and Adam optimizer.
3. Use `tf.stop_gradient` to block gradients for a portion of a computation graph.
4. Compute second-order gradients for `y = x^4` and verify the result.
5. Debug a case where gradients are `None` due to non-differentiable operations.

## 💡 Interview Tips
- **Common Questions**:
  - How does `tf.GradientTape` work in TensorFlow?
  - What causes gradients to be `None`, and how do you debug it?
  - When would you use `tf.stop_gradient`?
- **Tips**:
  - Explain the dynamic computation graph in `tf.GradientTape`.
  - Highlight common gradient issues (e.g., non-differentiable ops like `tf.cast`).
  - Be ready to code a gradient computation for a simple function or a training loop.

## 📚 Resources
- [TensorFlow Automatic Differentiation Guide](https://www.tensorflow.org/guide/autodiff)
- [TensorFlow API Documentation](https://www.tensorflow.org/api_docs/python/tf/GradientTape)
- [Kaggle: TensorFlow Tutorials](https://www.kaggle.com/learn/intro-to-deep-learning)