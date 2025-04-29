# Customization (`tensorflow`)

## 📖 Introduction
TensorFlow’s customization capabilities enable flexible model design and optimization. This guide covers custom layers and loss functions, Functional and Subclassing APIs, and debugging gradient issues, with practical examples and interview insights.

## 🎯 Learning Objectives
- Understand how to create custom layers and loss functions.
- Master Functional and Subclassing APIs for complex models.
- Learn to debug gradient issues in TensorFlow.

## 🔑 Key Concepts
- **Custom Layers**: Extend `tf.keras.layers.Layer` for specialized operations.
- **Custom Loss Functions**: Define task-specific loss functions.
- **Functional API**: Build models with flexible, non-sequential architectures.
- **Subclassing API**: Create fully customizable models via class inheritance.
- **Gradient Debugging**: Identify and fix issues like `None` gradients.

## 📝 Example Walkthrough
The `customization.py` file demonstrates:
1. **Custom Layer**: A `ScaledDense` layer with a learnable scaling factor.
2. **Custom Loss**: Weighted categorical crossentropy for class imbalance.
3. **Functional API**: A CNN with explicit input-output connections.
4. **Subclassing API**: A custom CNN with modular layers.
5. **Gradient Debugging**: Handling non-differentiable ops and disconnected graphs.
6. **Visualization**: Comparing model accuracy and gradient norms.

Example code:
```python
import tensorflow as tf
class ScaledDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(self.units)
        self.scale = self.add_weight('scale', shape=(), initializer='ones', trainable=True)
    
    def call(self, inputs):
        x = self.dense(inputs)
        x = x * self.scale
        return self.activation(x) if self.activation else x
```

## 🛠️ Practical Tasks
1. Create a custom layer that applies a polynomial transformation and test it on MNIST.
2. Implement a custom loss function that penalizes specific classes and train a model.
3. Build a model using the Functional API with multiple branches.
4. Use the Subclassing API to create a CNN with custom logic.
5. Debug a `None` gradient issue caused by a non-differentiable operation.

## 💡 Interview Tips
- **Common Questions**:
  - How do you implement a custom layer in TensorFlow?
  - What causes `None` gradients, and how do you debug them?
  - When would you use the Functional API over Subclassing?
- **Tips**:
  - Explain the `build` and `call` methods in custom layers.
  - Highlight common gradient issues (e.g., `tf.cast`, `tf.stop_gradient`).
  - Be ready to code a custom layer or debug a gradient issue.

## 📚 Resources
- [TensorFlow Custom Layers Guide](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
- [TensorFlow Functional API Guide](https://www.tensorflow.org/guide/keras/functional_api)
- [TensorFlow Gradient Debugging](https://www.tensorflow.org/guide/autodiff)
- [Kaggle: TensorFlow Tutorials](https://www.kaggle.com/learn/intro-to-deep-learning)