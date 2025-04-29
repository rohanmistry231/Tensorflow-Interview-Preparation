import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# %% [1. Introduction to Automatic Differentiation]
# Automatic differentiation computes gradients for optimization in TensorFlow.
# Key components: computational graphs, tf.GradientTape, optimizer.apply_gradients, and tf.stop_gradient.

print("TensorFlow version:", tf.__version__)

# %% [2. Computational Graphs]
# TensorFlow builds computational graphs to track operations for gradient computation.
# tf.GradientTape records operations dynamically for automatic differentiation.
x = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x)  # Ensure x is tracked
    y = x**2 + 2*x + 1  # Polynomial: y = x^2 + 2x + 1
dy_dx = tape.gradient(y, x)
print("\nComputational Graph Example:")
print("Function: y = x^2 + 2x + 1, x =", x.numpy())
print("Gradient dy/dx =", dy_dx.numpy())  # Expected: 2x + 2 = 8 at x=3

# %% [3. Gradient Computation with tf.GradientTape]
# Compute gradients for a simple neural network layer: y = Wx + b.
W = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
b = tf.Variable([1.0, 1.0])
x = tf.constant([[1.0], [2.0]])
with tf.GradientTape() as tape:
    y = tf.matmul(W, x) + b  # Linear transformation
    loss = tf.reduce_sum(y**2)  # Dummy loss: sum of squared outputs
grad_W, grad_b = tape.gradient(loss, [W, b])
print("\nGradient Computation (Linear Layer):")
print("W:\n", W.numpy())
print("b:", b.numpy())
print("x:\n", x.numpy())
print("Loss:", loss.numpy())
print("Gradient w.r.t. W:\n", grad_W.numpy())
print("Gradient w.r.t. b:", grad_b.numpy())

# %% [4. Higher-Order Gradients]
# Compute second-order gradients (e.g., Hessian) using nested tapes.
x = tf.constant(2.0)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        inner_tape.watch(x)
        y = x**3  # Function: y = x^3
    dy_dx = inner_tape.gradient(y, x)  # First derivative: 3x^2
d2y_dx2 = outer_tape.gradient(dy_dx, x)  # Second derivative: 6x
print("\nHigher-Order Gradients:")
print("Function: y = x^3, x =", x.numpy())
print("First Derivative (dy/dx):", dy_dx.numpy())  # Expected: 3x^2 = 12 at x=2
print("Second Derivative (d2y/dx2):", d2y_dx2.numpy())  # Expected: 6x = 12 at x=2

# %% [5. Gradient Application with Optimizer]
# Use an optimizer to update variables based on gradients.
W = tf.Variable([[1.0, 2.0]], name='W')
b = tf.Variable([0.0], name='b')
x = tf.constant([[1.0, 2.0]])
y_true = tf.constant([5.0])
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
for _ in range(3):  # Simulate 3 optimization steps
    with tf.GradientTape() as tape:
        y_pred = tf.matmul(W, x, transpose_b=True) + b  # y = Wx + b
        loss = tf.reduce_mean((y_pred - y_true)**2)  # MSE loss
    grad_W, grad_b = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip([grad_W, grad_b], [W, b]))
    print(f"\nStep {_+1} - Loss: {loss.numpy():.4f}, W: {W.numpy().flatten()}, b: {b.numpy()}")

# %% [6. No-Gradient Context with tf.stop_gradient]
# tf.stop_gradient prevents gradients from flowing through a tensor.
x = tf.constant(2.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x**2  # y = x^2
    z = tf.stop_gradient(y)  # Treat y as a constant
    w = z * x  # w = y * x
dw_dx = tape.gradient(w, x)
print("\nNo-Gradient Context:")
print("Function: w = (x^2) * x, with x^2 stopped")
print("Gradient dw/dx:", dw_dx.numpy())  # Expected: y = x^2 = 4 at x=2

# %% [7. Practical Application: Linear Regression]
# Train a linear regression model using tf.GradientTape.
np.random.seed(42)
X = np.random.rand(100, 1).astype(np.float32)
y = 3 * X + 2 + np.random.normal(0, 0.1, (100, 1)).astype(np.float32)
W = tf.Variable([[0.0]], name='weight')
b = tf.Variable([0.0], name='bias')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
losses = []
for epoch in range(50):
    with tf.GradientTape() as tape:
        y_pred = tf.matmul(X, W) + b
        loss = tf.reduce_mean(tf.square(y_pred - y))
    grad_W, grad_b = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip([grad_W, grad_b], [W, b]))
    losses.append(loss.numpy())
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")
print("\nLearned Parameters: W =", W.numpy().flatten(), "b =", b.numpy())

# %% [8. Visualizing Training Progress]
# Plot loss curve for linear regression.
plt.figure()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Linear Regression Loss Curve')
plt.savefig('loss_curve.png')

# Plot predictions
plt.figure()
plt.scatter(X, y, label='Data')
plt.plot(X, tf.matmul(X, W) + b, color='red', label='Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.savefig('linear_fit.png')

# %% [9. Interview Scenario: Gradient Debugging]
# Debug a case where gradients are None due to non-differentiable operations.
x = tf.Variable(1.0)
with tf.GradientTape() as tape:
    y = tf.cast(x, tf.int32)  # Non-differentiable operation
    loss = y**2
grad = tape.gradient(loss, x)
print("\nGradient Debugging:")
print("Operation: y = cast(x to int), loss = y^2")
print("Gradient:", grad)  # Expected: None due to non-differentiable cast
print("Fix: Ensure operations are differentiable (e.g., use float operations).")

# %% [10. Custom Gradient Computation]
# Compute gradients for a custom function: f(x) = sin(x) + x^2.
x = tf.Variable(1.0)
with tf.GradientTape() as tape:
    y = tf.sin(x) + x**2  # f(x) = sin(x) + x^2
dy_dx = tape.gradient(y, x)
print("\nCustom Gradient:")
print("Function: f(x) = sin(x) + x^2, x =", x.numpy())
print("Gradient: df/dx =", dy_dx.numpy())  # Expected: cos(x) + 2x = cos(1) + 2