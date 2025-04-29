# Neural Networks (`tf.keras`)

## 📖 Introduction
`tf.keras` is TensorFlow’s high-level API for building and training neural networks. This guide covers defining models (`tf.keras.Sequential`, `tf.keras.Model`), layers (Dense, Convolutional, Pooling, Normalization), activations (ReLU, Sigmoid, Softmax), loss functions (MSE, Categorical Crossentropy), optimizers (SGD, Adam, RMSprop), and learning rate schedules, with practical examples and interview insights.

## 🎯 Learning Objectives
- Understand `tf.keras` for building neural networks.
- Master model definition with `Sequential` and `Model` APIs.
- Apply layers, activations, loss functions, and optimizers.
- Implement learning rate schedules for improved training.

## 🔑 Key Concepts
- **Model Definition**: `tf.keras.Sequential` for linear stacks, `tf.keras.Model` for custom architectures.
- **Layers**: Dense (fully connected), Conv2D (convolutional), MaxPooling2D (pooling), BatchNormalization.
- **Activations**: ReLU (non-linearity), Sigmoid (binary), Softmax (multi-class).
- **Loss Functions**: MSE (regression), Categorical Crossentropy (classification).
- **Optimizers**: SGD (gradient descent), Adam (adaptive), RMSprop (momentum-based).
- **Learning Rate Schedules**: Adjust learning rate dynamically (e.g., exponential decay).

## 📝 Example Walkthrough
The `neural_networks_keras.py` file demonstrates:
1. **Sequential Model**: Regression with Dense layers.
2. **Custom Model**: Classification using `tf.keras.Model` subclassing.
3. **CNN**: Image classification with Conv2D, Pooling, and BatchNormalization.
4. **Activations and Losses**: ReLU, Sigmoid, Softmax, MSE, and Crossentropy.
5. **Optimizers**: Comparing SGD, Adam, RMSprop.
6. **Learning Rate Schedules**: Exponential decay for regression.
7. **Visualization**: Plotting training loss curves.

Example code:
```python
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
```

## 🛠️ Practical Tasks
1. Build a `Sequential` model for regression on synthetic data and evaluate MSE.
2. Create a custom `tf.keras.Model` for multi-class classification and train it.
3. Design a CNN with Conv2D, MaxPooling2D, and BatchNormalization for image data.
4. Compare SGD, Adam, and RMSprop on a regression task.
5. Implement an exponential decay learning rate schedule and plot the loss curve.

## 💡 Interview Tips
- **Common Questions**:
  - What is the difference between `Sequential` and `Model` APIs?
  - When would you use ReLU vs. Sigmoid?
  - How does Adam differ from SGD?
- **Tips**:
  - Explain the role of activations in introducing non-linearity.
  - Highlight Adam’s adaptive learning rate for faster convergence.
  - Be ready to code a simple neural network or CNN architecture.

## 📚 Resources
- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)
- [TensorFlow API Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Kaggle: TensorFlow Tutorials](https://www.kaggle.com/learn/intro-to-deep-learning)