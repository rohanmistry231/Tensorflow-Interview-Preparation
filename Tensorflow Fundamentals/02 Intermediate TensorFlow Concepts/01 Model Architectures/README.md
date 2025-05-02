# Model Architectures (`tensorflow`)

## 📖 Introduction
TensorFlow supports a variety of neural network architectures tailored to specific tasks. This guide covers Feedforward Neural Networks (FNNs), Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs, LSTMs, GRUs), and Transfer Learning with `tf.keras.applications`, with practical examples and interview insights.

## 🎯 Learning Objectives
- Understand the structure and use cases of FNNs, CNNs, RNNs, and transfer learning.
- Master building and training these architectures with `tf.keras`.
- Apply transfer learning using pre-trained models.
- Compare architectures for different data types and tasks.

## 🔑 Key Concepts
- **FNNs**: Fully connected layers for tabular data or simple tasks.
- **CNNs**: Convolutional and pooling layers for image data.
- **RNNs (LSTMs, GRUs)**: Recurrent layers for sequential or time-series data.
- **Transfer Learning**: Use pre-trained models (e.g., MobileNetV2) for efficient training.

## 📝 Example Walkthrough
The `model_architectures.py` file demonstrates:
1. **FNN**: Classification on MNIST with Dense layers.
2. **CNN**: Classification on MNIST with Conv2D and MaxPooling2D.
3. **RNNs**: LSTM and GRU for synthetic sequence classification.
4. **Transfer Learning**: Fine-tuning MobileNetV2 on CIFAR-10.
5. **Visualization**: Comparing validation accuracy and visualizing CNN predictions.

Example code:
```python
import tensorflow as tf
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

## 🛠️ Practical Tasks
1. Build an FNN for MNIST classification and evaluate its accuracy.
2. Create a CNN for MNIST with additional Conv2D layers and compare performance.
3. Train an LSTM or GRU on a synthetic sequence dataset for binary classification.
4. Fine-tune a pre-trained MobileNetV2 model on CIFAR-10 and evaluate improvements.
5. Visualize predictions and compare parameter counts across architectures.

## 💡 Interview Tips
- **Common Questions**:
  - When would you use a CNN over an FNN?
  - What are the differences between LSTMs and GRUs?
  - How does transfer learning improve training efficiency?
- **Tips**:
  - Explain CNN’s spatial feature extraction vs. FNN’s fully connected layers.
  - Highlight LSTM’s memory cells for long-term dependencies.
  - Be ready to code a simple CNN or fine-tune a pre-trained model.

## 📚 Resources
- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)
- [TensorFlow Applications Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
- [Kaggle: TensorFlow Tutorials](https://www.kaggle.com/learn/intro-to-deep-learning)
- [TensorFlow Official Tutorials](https://www.tensorflow.org/tutorials)