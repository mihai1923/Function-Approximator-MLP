function dy = leaky_relu_prime(x)
  # Derivative of Leaky ReLU activation function
  # Returns 1 for x > 0, and alpha for x <= 0
  # Critical for backpropagation with Leaky ReLU networks

  alpha = 0.01;  # Same leakiness parameter as in leaky_relu
  dy = (x > 0) + alpha * (x <= 0);
endfunction

