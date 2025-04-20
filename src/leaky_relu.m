function y = leaky_relu(x)
  # Leaky ReLU activation function - returns max(alpha*x, x)
  # Improves on standard ReLU by allowing small negative values
  # Alpha parameter controls the 'leakiness' (default 0.01)

  alpha = 0.01;  # Leakiness parameter
  y = max(alpha * x, x);
endfunction

