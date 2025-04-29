import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %% [1. Introduction to Neural Networks with tf.keras]
# tf.keras is TensorFlow's high-level API for building and training neural networks.
# Covers model definition, layers, activations, losses, optimizers, and learning rate schedules.

print("TensorFlow version:", tf.__version__)

# %% [2. Defining Models with tf.keras.Sequential]
# tf.keras.Sequential creates a linear stack of layers for simple models.
# Example: Regression model for synthetic data.
X_reg, y_reg = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_reg_train = scaler.fit_transform(X_reg_train)
X_reg_test = scaler.transform(X_reg_test)

seq_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # No activation for regression
])
seq_model.compile(optimizer='adam', loss='mse')
print("\nSequential Model Summary:")
seq_model.summary()

# Train the model
history_seq = seq_model.fit(X_reg_train, y_reg_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
print("Sequential Model Final Validation Loss:", history_seq.history['val_loss'][-1].round(4))

# %% [3. Defining Models with tf.keras.Model]
# tf.keras.Model allows custom models via subclassing for complex architectures.
# Example: Classification model for synthetic data.
X_clf, y_clf = make_classification(n_samples=1000, n_features=10, n_classes=3, n_informative=8, random_state=42)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
X_clf_train = scaler.fit_transform(X_clf_train)
X_clf_test = scaler.transform(X_clf_test)
y_clf_train_cat = tf.keras.utils.to_categorical(y_clf_train)
y_clf_test_cat = tf.keras.utils.to_categorical(y_clf_test)

class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(3, activation='softmax')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

custom_model = CustomModel()
custom_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("\nCustom Model Training:")
custom_model.fit(X_clf_train, y_clf_train_cat, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
loss, acc = custom_model.evaluate(X_clf_test, y_clf_test_cat, verbose=0)
print("Custom Model Test Loss:", loss.round(4), "Test Accuracy:", acc.round(4))

# %% [4. Layers: Dense, Convolutional, Pooling, Normalization]
# Example: CNN for synthetic image-like data (simplified).
X_img = np.random.rand(100, 28, 28, 1).astype(np.float32)  # 100 samples, 28x28x1
y_img = np.random.randint(0, 2, 100)  # Binary classification
X_img_train, X_img_test, y_img_train, y_img_test = train_test_split(X_img, y_img, test_size=0.2, random_state=42)

cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nCNN Model Summary:")
cnn_model.summary()
cnn_model.fit(X_img_train, y_img_train, epochs=5, batch_size=16, validation_split=0.2, verbose=0)
cnn_loss, cnn_acc = cnn_model.evaluate(X_img_test, y_img_test, verbose=0)
print("CNN Model Test Loss:", cnn_loss.round(4), "Test Accuracy:", cnn_acc.round(4))

# %% [5. Activations: ReLU, Sigmoid, Softmax]
# Demonstrate activation functions in a small network.
act_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),  # ReLU
    tf.keras.layers.Dense(8, activation='sigmoid'),  # Sigmoid
    tf.keras.layers.Dense(3, activation='softmax')  # Softmax
])
print("\nActivation Functions Model Summary:")
act_model.summary()

# %% [6. Loss Functions: MSE, Categorical Crossentropy]
# MSE for regression (used in seq_model).
# Categorical Crossentropy for classification (used in custom_model).
print("\nLoss Functions Used:")
print("MSE for Regression (Sequential Model):", history_seq.history['loss'][-1].round(4))
print("Categorical Crossentropy for Classification (Custom Model):", loss.round(4))

# %% [7. Optimizers: SGD, Adam, RMSprop]
# Compare optimizers on the regression task.
optimizers = {
    'SGD': tf.keras.optimizers.SGD(learning_rate=0.01),
    'Adam': tf.keras.optimizers.Adam(learning_rate=0.001),
    'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=0.001)
}
results = {}
for name, opt in optimizers.items():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=opt, loss='mse')
    history = model.fit(X_reg_train, y_reg_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    results[name] = history.history['val_loss'][-1]
print("\nOptimizer Comparison (Validation Loss):")
for name, val_loss in results.items():
    print(f"{name}: {val_loss:.4f}")

# %% [8. Learning Rate Schedules]
# Use a decaying learning rate schedule for the regression task.
initial_lr = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_lr, decay_steps=100, decay_rate=0.9, staircase=True
)
model_lr = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
model_lr.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='mse')
history_lr = model_lr.fit(X_reg_train, y_reg_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
print("\nLearning Rate Schedule Model Final Validation Loss:", history_lr.history['val_loss'][-1].round(4))

# %% [9. Visualizing Training Progress]
# Plot loss curves for Sequential and Learning Rate Schedule models.
plt.figure()
plt.plot(history_seq.history['loss'], label='Sequential (Adam)')
plt.plot(history_lr.history['loss'], label='Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curves')
plt.legend()
plt.savefig('loss_curves.png')

# %% [10. Interview Scenario: Model Design]
# Design a CNN for a small image classification task.
interview_cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
interview_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("\nInterview Scenario: CNN Model Summary:")
interview_cnn.summary()
print("Explanation: Conv2D extracts features, MaxPooling reduces dimensions, BatchNorm stabilizes training, Softmax outputs class probabilities.")