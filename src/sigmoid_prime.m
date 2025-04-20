function [output] = sigmoid_prime(z)
  # Derivative of sigmoid function
  s = sigmoid(z);
  output = s .* (1 - s);
endfunction