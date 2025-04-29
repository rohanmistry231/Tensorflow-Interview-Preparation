# Distributed Training (`tensorflow`)

## 📖 Introduction
Distributed training scales TensorFlow models across multiple devices (GPUs/TPUs). This guide covers Data Parallelism (`MirroredStrategy`), Multi-GPU/TPU Training (`TPUStrategy`), and Distributed Datasets, with practical examples and interview insights.

## 🎯 Learning Objectives
- Understand data parallelism with `MirroredStrategy`.
- Implement multi-GPU/TPU training with `TPUStrategy`.
- Optimize datasets for distributed training.

## 🔑 Key Concepts
- **Data Parallelism**: `MirroredStrategy` replicates model across GPUs with synchronized gradients.
- **Multi-GPU/TPU Training**: `TPUStrategy` leverages TPUs for high-performance training.
- **Distributed Datasets**: Shard datasets across devices for efficient processing.

## 📝 Example Walkthrough
The `distributed_training.py` file demonstrates:
1. **Dataset**: Loading and preprocessing CIFAR-10.
2. **MirroredStrategy**: Training a CNN with data parallelism.
3. **TPUStrategy**: Training with TPU support (fallback to GPUs).
4. **Distributed Datasets**: Custom training loop with distributed datasets.
5. **Visualization**: Comparing accuracy across strategies.

Example code:
```python
import tensorflow as tf
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = tf.keras.Sequential([...])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## 🛠️ Practical Tasks
1. Train a CNN on CIFAR-10 using `MirroredStrategy`.
2. Adapt the model for `TPUStrategy` in a cloud environment (e.g., Colab).
3. Create a distributed dataset and implement a custom training loop.
4. Compare training speed and accuracy across strategies.

## 💡 Interview Tips
- **Common Questions**:
  - How does `MirroredStrategy` synchronize gradients?
  - What are the benefits of `TPUStrategy`?
  - How do you shard datasets for distributed training?
- **Tips**:
  - Explain gradient aggregation in data parallelism.
  - Highlight TPU’s matrix multiplication efficiency.
  - Be ready to code a model with `MirroredStrategy`.

## 📚 Resources
- [TensorFlow Distributed Training Guide](https://www.tensorflow.org/guide/distributed_training)
- [TensorFlow TPU Guide](https://www.tensorflow.org/guide/tpu)
- [Kaggle: TensorFlow Tutorials](https://www.kaggle.com/learn/intro-to-deep-learning)