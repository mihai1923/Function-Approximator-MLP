function [cost] = mean_squared_error(y_pred, y_true)
  # Mean squared error loss function
  cost = mean(mean((y_pred - y_true).^2, 2));
endfunction