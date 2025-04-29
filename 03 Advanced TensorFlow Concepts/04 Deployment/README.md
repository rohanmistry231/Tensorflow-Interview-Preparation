# Deployment (`tensorflow`)

## 📖 Introduction
Deploying TensorFlow models enables production use. This guide covers model export (SavedModel), serving (TensorFlow Serving), and edge deployment (TensorFlow Lite, TensorFlow.js), with practical examples and interview insights.

## 🎯 Learning Objectives
- Export models as SavedModel for serving.
- Serve models with TensorFlow Serving.
- Deploy models on edge devices with TensorFlow Lite/JS.

## 🔑 Key Concepts
- **SavedModel**: Standard TensorFlow format for export.
- **TensorFlow Serving**: Scalable serving via REST/gRPC.
- **TensorFlow Lite**: Lightweight models for mobile/edge.
- **TensorFlow.js**: Browser-based inference.

## 📝 Example Walkthrough
The `deployment.py` file demonstrates:
1. **Model**: Training a CNN on MNIST.
2. **SavedModel**: Exporting the model.
3. **TensorFlow Serving**: Instructions for serving.
4. **TensorFlow Lite/JS**: Converting and evaluating.
5. **Visualization**: Visualizing predictions.

Example code:
```python
import tensorflow as tf
model = tf.keras.Sequential([...])
model.save("saved_model/mnist_cnn")
```

## 🛠️ Practical Tasks
1. Export a trained model as SavedModel.
2. Set up TensorFlow Serving for the model.
3. Convert a model to TensorFlow Lite and evaluate it.
4. Prepare a model for TensorFlow.js deployment.
5. Visualize predictions from a deployed model.

## 💡 Interview Tips
- **Common Questions**:
  - What is the SavedModel format?
  - How does TensorFlow Serving handle requests?
  - Why use TensorFlow Lite for edge devices?
- **Tips**:
  - Explain SavedModel’s portability.
  - Highlight TensorFlow Lite’s quantization benefits.
  - Be ready to describe a deployment pipeline.

## 📚 Resources
- [TensorFlow SavedModel Guide](https://www.tensorflow.org/guide/saved_model)
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [TensorFlow.js](https://www.tensorflow.org/js)