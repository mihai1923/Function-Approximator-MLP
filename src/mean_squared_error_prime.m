function [gradient] = mean_squared_error_prime(y_pred, y_true)
  # Derivative of MSE loss function
  gradient = 2 * (y_pred - y_true) / size(y_true, 1);
endfunction