# Neural Network implementation from scratch
# This file contains the core neural network functionality

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

function [output] = sigmoid(z)
  # Sigmoid activation function
  output = 1 ./ (1 + exp(-z));
endfunction

function [output] = sigmoid_prime(z)
  # Derivative of sigmoid function
  s = sigmoid(z);
  output = s .* (1 - s);
endfunction

function [output] = relu(z)
  # ReLU activation function
  output = max(0, z);
endfunction

function [output] = relu_prime(z)
  # Derivative of ReLU function
  output = (z > 0);
endfunction

function [output] = tanh_prime(z)
  # Derivative of tanh function
  output = 1 - tanh(z).^2;
endfunction

function [output] = identity(z)
  # Identity function for regression output layer
  output = z;
endfunction

function [output] = identity_prime(z)
  # Derivative of identity function
  output = ones(size(z));
endfunction

function [cost] = mean_squared_error(y_pred, y_true)
  # Mean squared error loss function
  cost = mean(mean((y_pred - y_true).^2, 2));
endfunction

function [gradient] = mean_squared_error_prime(y_pred, y_true)
  # Derivative of MSE loss function
  gradient = 2 * (y_pred - y_true) / size(y_true, 1);
endfunction

function [cost] = cross_entropy(y_pred, y_true)
  # Cross entropy loss function for classification
  epsilon = 1e-10;  # Prevent log(0)
  y_pred = max(min(y_pred, 1 - epsilon), epsilon);
  cost = -mean(sum(y_true .* log(y_pred) + (1 - y_true) .* log(1 - y_pred), 2));
endfunction

function [gradient] = cross_entropy_prime(y_pred, y_true)
  # Derivative of cross entropy loss
  epsilon = 1e-10;
  y_pred = max(min(y_pred, 1 - epsilon), epsilon);
  gradient = -(y_true ./ y_pred - (1 - y_true) ./ (1 - y_pred)) / size(y_true, 1);
endfunction

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

function [model, history] = train(model, X, y, val_X=[], val_y=[])
  # Train the neural network using gradient descent
  # X: training features
  # y: training labels
  # val_X, val_y: optional validation data

  history = struct("train_loss", [], "val_loss", []);

  n_samples = size(X, 2);
  batch_size = min(n_samples, 32);  # Default batch size

  for iter = 1:model.max_iter
    # Randomly sample a batch if dataset is large
    if n_samples > batch_size
      batch_idx = randperm(n_samples, batch_size);
      X_batch = X(:, batch_idx);
      y_batch = y(:, batch_idx);
    else
      X_batch = X;
      y_batch = y;
    endif

    # Forward pass
    [a, z] = forward_pass(model, X_batch);

    # Compute loss
    train_loss = model.loss_function(a{end}, y_batch);

    # Add regularization term to loss
    reg_term = 0;
    for i = 1:(model.num_layers-1)
      reg_term += 0.5 * model.reg_lambda * sum(sum(model.weights{i}.^2));
    endfor
    train_loss += reg_term / size(X_batch, 2);

    history.train_loss(end+1) = train_loss;

    # Compute validation loss if validation data is provided
    if !isempty(val_X) && !isempty(val_y)
      [val_a, ~] = forward_pass(model, val_X);
      val_loss = model.loss_function(val_a{end}, val_y);
      history.val_loss(end+1) = val_loss;
    endif

    # Check for convergence
    if iter > 1 && abs(history.train_loss(end) - history.train_loss(end-1)) < model.tolerance
      printf("Converged at iteration %d\n", iter);
      break;
    endif

    # Backward pass to get gradients
    [~, grad_w, grad_b] = backward_pass(model, a, z, y_batch);

    # Update weights and biases
    for i = 1:(model.num_layers-1)
      model.weights{i} -= model.learning_rate * grad_w{i};
      model.biases{i} -= model.learning_rate * grad_b{i};
    endfor

    # Print progress
    if mod(iter, 100) == 0
      printf("Iteration %d: loss = %f\n", iter, train_loss);
    endif
  endfor
endfunction

function [predictions] = predict(model, X)
  # Make predictions using the trained model
  [a, ~] = forward_pass(model, X);
  predictions = a{end};
endfunction

function [accuracy] = compute_accuracy(y_pred, y_true)
  # Compute classification accuracy
  [~, pred_labels] = max(y_pred, [], 1);
  [~, true_labels] = max(y_true, [], 1);
  accuracy = mean(pred_labels == true_labels);
endfunction

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