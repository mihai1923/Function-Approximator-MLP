function [delta, grad_w, grad_b] = backward_pass(model, a, z, y)
  # Backward propagation to compute gradients
  # a: activations from forward pass
  # z: weighted inputs from forward pass
  # y: true output values

  num_layers = model.num_layers;
  delta = cell(num_layers, 1);
  grad_w = cell(num_layers - 1, 1);
  grad_b = cell(num_layers - 1, 1);

  # Output layer error
  delta{num_layers} = model.loss_function_prime(a{num_layers}, y) .* model.output_activation_prime(z{num_layers});

  # Backpropagate error
  for l = (num_layers-1):-1:2
    delta{l} = (model.weights{l}' * delta{l+1}) .* model.activation_prime(z{l});
  endfor

  # Compute gradients for weights and biases
  for l = 1:(num_layers-1)
    grad_w{l} = delta{l+1} * a{l}' + model.reg_lambda * model.weights{l};  # Add regularization
    grad_b{l} = sum(delta{l+1}, 2);
  endfor
endfunction