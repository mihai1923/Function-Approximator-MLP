function y = softmax(x)
  # Softmax activation function for multi-class classification
  # Computes softmax along rows (for each class)
  # Includes numerical stability improvements
  
  # Shift values for numerical stability (prevents overflow)
  # Subtracting the max doesn't change the output due to softmax properties
  shifted_x = x - max(x, [], 1);
  
  # Compute exp with the shifted values
  exp_x = exp(shifted_x);
  
  # Normalize to get probabilities
  y = exp_x ./ (sum(exp_x, 1) + eps);
endfunction