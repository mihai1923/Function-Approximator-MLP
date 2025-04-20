function [output] = sigmoid(z)
  # Sigmoid activation function
  output = 1 ./ (1 + exp(-z));
endfunction