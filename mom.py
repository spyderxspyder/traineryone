import numpy as np

# The objective function
def objective(x, y):
    return x**2.0 + y**2.0

# The derivative of the objective function
def derivative(x, y):
    return np.array([x * 2.0, y * 2.0])

# Gradient descent algorithm with momentum
def momentum_gradient_descent(objective, derivative, x,y, n_iter, learning_rate, momentum):
    # Generate an initial point
    # Initialize the change vector
    changex = 0.0
    changey= 0.0
    # Run the gradient descent
    for i in range(n_iter):
        # Calculate the gradient
        gradient = derivative(x, y)
        # Calculate the new change vector
        new_changex = learning_rate * gradient[0] + momentum * changex
        new_changey = learning_rate * gradient[1] + momentum * changey
        # Update the solution
        x-=new_changex
        y-=new_changey
        # Store the new change
        changex = new_changex
        changey = new_changey
        # Evaluate the candidate point
        # Report progress
        print(x,y)
    return [x, y]

# Seed the pseudo-random number generator
np.random.seed(1)
# Define the range for input
x = -0.1659  # Initial x
y = 0.4406   # Initial y
# Define the total iterations
n_iter = 30
# Define the learning rate
learning_rate = 0.1
# Define the momentum
momentum = 0.3
# Perform the gradient descent search with momentum
best, score = momentum_gradient_descent(objective, derivative, x,y, n_iter, learning_rate, momentum)
print('Done!')
print('f(%s) = %f' % (best, score))
