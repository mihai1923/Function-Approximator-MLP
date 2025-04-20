function y = relu(x)
  # ReLU activation function - returns max(0, x)
  # Better for deep networks than tanh/sigmoid due to no vanishing gradient
  # Takes input x (can be vector or matrix) and applies element-wise ReLU
  
  y = max(0, x);
endfunction