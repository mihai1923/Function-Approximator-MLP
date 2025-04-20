function [output] = tanh_prime(z)
  # Derivative of tanh function
  output = 1 - tanh(z).^2;
endfunction