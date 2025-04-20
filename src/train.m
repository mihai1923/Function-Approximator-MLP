function [model, history] = train(model, X, y, val_X=[], val_y=[])
  # Train the neural network using gradient descent
  # X: training features
  # y: training labels
  # val_X, val_y: optional validation data

  history = struct("train_loss", [], "val_loss", []);

  n_samples = size(X, 2);
  batch_size = min(n_samples, 32);  # Default batch size

  # Check if clip_threshold is set, otherwise use a default
  if (!isfield(model, "clip_threshold"))
    model.clip_threshold = Inf;  # No clipping by default
  endif

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

    # Check for NaN loss and break if found
    if (isnan(train_loss))
      printf("Warning: NaN loss detected at iteration %d. Stopping training.\n", iter);
      break;
    endif

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

    # Apply gradient clipping to prevent exploding gradients
    for i = 1:(model.num_layers-1)
      # Clip gradients for weights
      grad_norm = sqrt(sum(sum(grad_w{i}.^2)));
      if (grad_norm > model.clip_threshold)
        grad_w{i} = grad_w{i} * (model.clip_threshold / grad_norm);
      endif

      # Clip gradients for biases
      bias_norm = sqrt(sum(grad_b{i}.^2));
      if (bias_norm > model.clip_threshold)
        grad_b{i} = grad_b{i} * (model.clip_threshold / bias_norm);
      endif
    endfor

    # Update weights and biases
    for i = 1:(model.num_layers-1)
      model.weights{i} -= model.learning_rate * grad_w{i};
      model.biases{i} -= model.learning_rate * grad_b{i};

      # Check for NaN or Inf in weights and biases
      if (any(isnan(model.weights{i}(:))) || any(isinf(model.weights{i}(:))) ||
          any(isnan(model.biases{i})) || any(isinf(model.biases{i})))
        printf("Warning: NaN or Inf values detected in weights/biases at iteration %d. Stopping training.\n", iter);
        return;
      endif
    endfor

    # Print progress
    if mod(iter, 100) == 0
      printf("Iteration %d: loss = %f\n", iter, train_loss);
    endif
  endfor
endfunction