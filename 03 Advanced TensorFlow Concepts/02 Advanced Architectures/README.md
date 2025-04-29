# Advanced Architectures (`tensorflow`)

## 📖 Introduction
Advanced architectures push TensorFlow’s capabilities for complex tasks. This guide covers Transformers (Vision Transformers), Generative Models (VAEs), and Reinforcement Learning (TF-Agents), with practical examples and interview insights.

## 🎯 Learning Objectives
- Understand Vision Transformers for image tasks.
- Implement VAEs for generative modeling.
- Apply TF-Agents for reinforcement learning.

## 🔑 Key Concepts
- **Vision Transformers**: Patch-based attention for images.
- **VAEs**: Encoder-decoder with latent space for generation.
- **Reinforcement Learning**: DQN with TF-Agents for decision-making.

## 📝 Example Walkthrough
The `advanced_architectures.py` file demonstrates:
1. **Vision Transformer**: Simplified ViT for CIFAR-10.
2. **VAE**: Generating CIFAR-10 images.
3. **DQN**: Training on CartPole with TF-Agents.
4. **Visualization**: Comparing original and generated images.

Example code:
```python
import tensorflow as tf
class ViT(tf.keras.Model):
    def __init__(self, num_classes, patch_size, num_patches, d_model, num_heads):
        super().__init__()
        self.transformer = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')
```

## 🛠️ Practical Tasks
1. Train a Vision Transformer on CIFAR-10.
2. Build a VAE for image generation and visualize outputs.
3. Implement a DQN agent for CartPole and evaluate rewards.
4. Experiment with Transformer hyperparameters (e.g., num_heads).

## 💡 Interview Tips
- **Common Questions**:
  - How do Transformers differ from CNNs?
  - What is the role of the latent space in VAEs?
  - How does DQN balance exploration and exploitation?
- **Tips**:
  - Explain ViT’s patch embedding process.
  - Highlight VAE’s reconstruction and KL losses.
  - Be ready to code a simple RL agent.

## 📚 Resources
- [TensorFlow Transformers Guide](https://www.tensorflow.org/text/tutorials/transformer)
- [TensorFlow Agents](https://www.tensorflow.org/agents)
- [Kaggle: TensorFlow Tutorials](https://www.kaggle.com/learn/intro-to-deep-learning)