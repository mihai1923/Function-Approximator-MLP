function [grad_numerical] = check_gradients(model, X, y)
  # Numerical gradient checking to verify backpropagation
  epsilon = 1e-7;

  # Get analytical gradients
  [a, z] = forward_pass(model, X);
  [~, grad_w, grad_b] = backward_pass(model, a, z, y);

  # Initialize numerical gradients
  grad_numerical = struct("weights", cell(model.num_layers-1, 1), "biases", cell(model.num_layers-1, 1));

  # Check gradients for each layer
  for l = 1:(model.num_layers-1)
    # Check weights
    grad_numerical.weights{l} = zeros(size(model.weights{l}));

    for i = 1:size(model.weights{l}, 1)
      for j = 1:size(model.weights{l}, 2)
        # Temporarily modify weight
        model.weights{l}(i, j) += epsilon;
        [a_plus, ~] = forward_pass(model, X);
        loss_plus = model.loss_function(a_plus{end}, y);

        model.weights{l}(i, j) -= 2 * epsilon;
        [a_minus, ~] = forward_pass(model, X);
        loss_minus = model.loss_function(a_minus{end}, y);

        # Restore original weight
        model.weights{l}(i, j) += epsilon;

        # Compute numerical gradient
        grad_numerical.weights{l}(i, j) = (loss_plus - loss_minus) / (2 * epsilon);
      endfor
    endfor

    # Check biases
    grad_numerical.biases{l} = zeros(size(model.biases{l}));

    for i = 1:size(model.biases{l}, 1)
      # Temporarily modify bias
      model.biases{l}(i) += epsilon;
      [a_plus, ~] = forward_pass(model, X);
      loss_plus = model.loss_function(a_plus{end}, y);

      model.biases{l}(i) -= 2 * epsilon;
      [a_minus, ~] = forward_pass(model, X);
      loss_minus = model.loss_function(a_minus{end}, y);

      # Restore original bias
      model.biases{l}(i) += epsilon;

      # Compute numerical gradient
      grad_numerical.biases{l}(i) = (loss_plus - loss_minus) / (2 * epsilon);
    endfor
  endfor
endfunction