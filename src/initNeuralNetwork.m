function [model] = initNeuralNetwork(layer_sizes)
  # Initialize neural network with specified layer sizes
  # layer_sizes: array containing the number of neurons in each layer

  model = struct();
  model.num_layers = length(layer_sizes);
  model.layer_sizes = layer_sizes;
  model.weights = cell(model.num_layers - 1, 1);
  model.biases = cell(model.num_layers - 1, 1);

  # Initialize weights and biases with random values
  for i = 1:(model.num_layers - 1)
    # Xavier/Glorot initialization for better convergence
    model.weights{i} = randn(layer_sizes(i+1), layer_sizes(i)) * sqrt(2 / (layer_sizes(i) + layer_sizes(i+1)));
    model.biases{i} = zeros(layer_sizes(i+1), 1);
  endfor

  # Set default hyperparameters
  model.learning_rate = 0.01;
  model.reg_lambda = 0.01;  # L2 regularization parameter
  model.max_iter = 1000;
  model.tolerance = 1e-6;

  # Set activation functions
  model.activation = @sigmoid;
  model.activation_prime = @sigmoid_prime;
  model.output_activation = @identity;
  model.output_activation_prime = @identity_prime;
  model.loss_function = @mean_squared_error;
  model.loss_function_prime = @mean_squared_error_prime;

endfunction