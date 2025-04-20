function [a, z] = forward_pass(model, x)
  # Forward propagation through the network
  # x: input data (features)
  # Returns activations and weighted inputs for each layer

  num_layers = model.num_layers;
  a = cell(num_layers, 1);
  z = cell(num_layers, 1);

  # Input layer
  a{1} = x;

  # Hidden and output layers
  for i = 2:num_layers
    z{i} = model.weights{i-1} * a{i-1} + model.biases{i-1};

    # Apply activation function (different for output layer)
    if i == num_layers
      a{i} = model.output_activation(z{i});
    else
      a{i} = model.activation(z{i});
    endif
  endfor
endfunction