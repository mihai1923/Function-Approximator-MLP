function [predictions] = predict(model, X)
  # Make predictions using the trained model
  [a, ~] = forward_pass(model, X);
  predictions = a{end};
endfunction