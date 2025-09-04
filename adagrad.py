import numpy as np

# Define the function and its gradients
def f(x, y):
    return x**2 + y**2

def grad_f(x, y):
    return np.array([2 * x, 2 * y])

# Parameters
eta = 0.1  # Learning rate
epsilon = 1e-8  # Convergence threshold
x = -0.1659  # Initial x
y = 0.4406   # Initial y
bounds = [-1, 1]  # Bounds for x and y

# Initialize sum of squared gradients
sum_sq_grad = np.array([0.0, 0.0])

# Iteration counter
iteration = 0
max_iterations = 10000  # Safety limit

while True:
    # Compute gradients
    grad = grad_f(x, y)

    # Update sum of squared gradients
    sum_sq_grad += grad**2

    # Compute adaptive learning rate
    adaptive_lr = eta / (np.sqrt(sum_sq_grad) + epsilon)

    # Update parameters
    x -= adaptive_lr[0] * grad[0]
    y -= adaptive_lr[1] * grad[1]

    # Enforce bounds
    x = np.clip(x, bounds[0], bounds[1])
    y = np.clip(y, bounds[0], bounds[1])

    # Increment iteration
    iteration += 1

    # Print progress at each iteration
    print(f"Iteration {iteration}: x = {x:.8f}, y = {y:.8f}, f(x, y) = {f(x, y):.8f}")

    # Check convergence
    if np.linalg.norm(grad) < epsilon or iteration >= max_iterations:
        break

# Output results
print(f"\nFinal x: {x:.8f}, y: {y:.8f}")
print(f"Function value: {f(x, y):.8f}")
print(f"Total Iterations: {iteration}")




'''import numpy as np

def xavier_init(n_in, n_out, shape):
    """ Xavier (Glorot) initialization with normal distribution """
    std = np.sqrt(2.0 / (n_in + n_out))
    return np.random.normal(0, std, shape)

def he_init(n_in, shape):
    """ He (Kaiming) initialization with normal distribution """
    std = np.sqrt(2.0 / n_in)
    return np.random.normal(0, std, shape)

# Example dimensions
n_in = 64
n_out = 32

# Xavier initialization
w1_xavier = xavier_init(n_in, n_out, (n_in, n_out))
w2_xavier = xavier_init(n_out, n_in, (n_out, n_in))

# He initialization
w1_he = he_init(n_in, (n_in, n_out))
w2_he = he_init(n_out, (n_out, n_in))

print("Xavier w1:", w1_xavier.shape, " | std ≈", np.std(w1_xavier))
print("He w1:", w1_he.shape, " | std ≈", np.std(w1_he))'''
