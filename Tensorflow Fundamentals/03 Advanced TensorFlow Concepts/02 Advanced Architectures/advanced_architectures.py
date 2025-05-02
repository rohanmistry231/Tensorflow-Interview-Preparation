import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
import tensorflow_datasets as tfds
import tensorflow_agents as tfa
from tensorflow_agents.environments import suite_gym, tf_py_environment
from tensorflow_agents.agents.dqn import dqn_agent
from tensorflow_agents.networks import q_network
from tensorflow_agents.policies import random_tf_policy
from tensorflow_agents.replay_buffers import tf_uniform_replay_buffer
from tensorflow_agents.utils import common

# %% [1. Introduction to Advanced Architectures]
# Advanced architectures include Transformers, Generative Models, Graph Neural Networks, and Reinforcement Learning.
# This file demonstrates a Vision Transformer, VAE, and DQN with TF-Agents.

print("TensorFlow version:", tf.__version__)

# %% [2. Preparing Datasets]
# Load CIFAR-10 for Vision Transformer and VAE.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
print("\nCIFAR-10 Dataset:")
print("Train Shape:", x_train.shape, "Test Shape:", x_test.shape)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32).prefetch(tf.data.AUTOTUNE)

# %% [3. Vision Transformer (ViT)]
# Simplified Vision Transformer for CIFAR-10.
class PatchExtractor(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    
    def call(self, images):
        patches = tf.image.extract_patches(
            images=images, sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding='VALID')
        return tf.reshape(patches, [tf.shape(images)[0], -1, patches.shape[-1]])

class ViT(tf.keras.Model):
    def __init__(self, num_classes, patch_size, num_patches, d_model, num_heads):
        super().__init__()
        self.patch_extractor = PatchExtractor(patch_size)
        self.pos_embedding = self.add_weight('pos_embedding', shape=(1, num_patches + 1, d_model))
        self.cls_token = self.add_weight('cls_token', shape=(1, 1, d_model))
        self.transformer = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        patches = self.patch_extractor(inputs)
        batch_size = tf.shape(inputs)[0]
        cls_tokens = tf.repeat(self.cls_token, batch_size, axis=0)
        x = tf.concat([cls_tokens, patches], axis=1)
        x += self.pos_embedding
        x = self.transformer(x, x)
        x = x[:, 0, :]  # Take CLS token
        return self.dense(x)

vit_model = ViT(num_classes=10, patch_size=8, num_patches=(32//8)**2, d_model=64, num_heads=4)
vit_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("\nVision Transformer Training:")
vit_history = vit_model.fit(train_ds, epochs=3, validation_data=test_ds, verbose=1)
print("ViT Test Accuracy:", vit_history.history['val_accuracy'][-1].round(4))

# %% [4. Generative Model: Variational Autoencoder (VAE)]
# VAE for generating CIFAR-10-like images.
class VAE(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(16 + 16)  # Mean + log variance
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(16,)),
            tf.keras.layers.Dense(32 * 32 * 32, activation='relu'),
            tf.keras.layers.Reshape((32, 32, 32)),
            tf.keras.layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')
        ])
    
    def call(self, inputs):
        mean, logvar = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        epsilon = tf.random.normal(tf.shape(mean))
        z = mean + tf.exp(0.5 * logvar) * epsilon
        return self.decoder(z)

vae_model = VAE()
vae_optimizer = tf.keras.optimizers.Adam()
@tf.function
def vae_loss(x, x_recon):
    mean, logvar = tf.split(vae_model.encoder(x), num_or_size_splits=2, axis=1)
    recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, x_recon))
    kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
    return recon_loss + kl_loss

@tf.function
def train_vae_step(x):
    with tf.GradientTape() as tape:
        x_recon = vae_model(x)
        loss = vae_loss(x, x_recon)
    gradients = tape.gradient(loss, vae_model.trainable_variables)
    vae_optimizer.apply_gradients(zip(gradients, vae_model.trainable_variables))
    return loss

print("\nVAE Training:")
for epoch in range(3):
    for x, _ in train_ds:
        loss = train_vae_step(x)
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy():.4f}")

# Generate and save sample images
generated = vae_model(x_test[:5])
plt.figure()
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title("Original")
    plt.axis('off')
    plt.subplot(2, 5, i + 6)
    plt.imshow(generated[i])
    plt.title("Generated")
    plt.axis('off')
plt.savefig('vae_samples.png')

# %% [5. Reinforcement Learning with TF-Agents]
# DQN for CartPole environment.
env = suite_gym.load('CartPole-v0')
train_env = tf_py_environment.TFPyEnvironment(env)
eval_env = tf_py_environment.TFPyEnvironment(env)

q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=(100,)
)
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=tf.Variable(0)
)
agent.initialize()

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=1000
)
collect_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
def collect_step(env, policy):
    time_step = env.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = env.step(action_step.action)
    traj = tfa.trajectories.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)

print("\nDQN Training on CartPole:")
for _ in range(100):
    collect_step(train_env, collect_policy)
dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=64, num_steps=2).prefetch(3)
iterator = iter(dataset)
for _ in range(1000):
    trajectories, _ = next(iterator)
    agent.train(trajectories)

# Evaluate DQN
total_reward = 0
for _ in range(5):
    time_step = eval_env.reset()
    episode_reward = 0
    while not time_step.is_last():
        action_step = agent.policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        episode_reward += time_step.reward
    total_reward += episode_reward
print("Average Reward:", (total_reward / 5).numpy())