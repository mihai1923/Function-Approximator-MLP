function [model, history] = train_with_frequency_awareness(model, X_train, y_train, X_val, y_val, options)
  # Specialized training function that leverages frequency information for seasonal patterns
  # Includes adaptive learning rates, momentum, and custom regularization
  
  history = struct("train_loss", [], "val_loss", []);
  
  # Initialize optimization parameters
  n_samples = size(X_train, 2);
  batch_size = min(n_samples, 64);
  
  # Use momentum for faster convergence
  momentum = 0.9;
  
  # Initialize momentum parameters
  velocity_w = cell(model.num_layers-1, 1);
  velocity_b = cell(model.num_layers-1, 1);
  
  for i = 1:(model.num_layers-1)
    velocity_w{i} = zeros(size(model.weights{i}));
    velocity_b{i} = zeros(size(model.biases{i}));
  endfor
  
  # Early stopping parameters
  best_val_loss = Inf;
  patience_counter = 0;
  best_model = model;
  best_iter = 0;
  
  # Learning rate schedule
  curr_learning_rate = model.learning_rate;
  min_learning_rate = 0.0001;
  
  # Identify frequency features for specialized weighting
  if isfield(options, "sequence_length")
    seq_length = options.sequence_length;
    freq_feature_indices = (seq_length+1):size(X_train, 1);
  else
    # Default: assume frequency features are the last few features (if any)
    if isfield(options, "frequencies")
      freq_feature_indices = (size(X_train, 1) - 2*length(options.frequencies) + 1):size(X_train, 1);
    else
      freq_feature_indices = [];
    endif
  endif
  
  # Apply frequency-aware feature weighting during training
  freq_attention = 1.5;  # Weight frequency features more strongly
  
  printf("Training with frequency-aware optimization...\n");
  
  for iter = 1:model.max_iter
    # Sample a batch
    if n_samples > batch_size
      batch_idx = randperm(n_samples, batch_size);
      X_batch = X_train(:, batch_idx);
      y_batch = y_train(:, batch_idx);
    else
      X_batch = X_train;
      y_batch = y_train;
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
    if isnan(train_loss)
      printf("Warning: NaN loss detected at iteration %d. Restoring best model.\n", iter);
      model = best_model;
      break;
    endif
    
    history.train_loss(end+1) = train_loss;
    
    # Compute validation loss
    [val_a, ~] = forward_pass(model, X_val);
    val_loss = model.loss_function(val_a{end}, y_val);
    val_loss += reg_term / size(X_val, 2);  # Add regularization to validation loss
    history.val_loss(end+1) = val_loss;
    
    # Early stopping logic
    if val_loss < best_val_loss
      best_val_loss = val_loss;
      best_model = model;
      best_iter = iter;
      patience_counter = 0;
    else
      patience_counter += 1;
      
      # Reduce learning rate when improvement stalls
      if patience_counter > 0 && mod(patience_counter, 10) == 0
        curr_learning_rate = max(curr_learning_rate * 0.9, min_learning_rate);
        printf("Reducing learning rate to %.6f at iteration %d\n", curr_learning_rate, iter);
      endif
      
      # Stop if no improvement for a while
      if patience_counter >= model.early_stopping_patience
        printf("Early stopping at iteration %d. Best validation loss: %.6f at iteration %d\n", 
              iter, best_val_loss, best_iter);
        model = best_model;
        break;
      endif
    endif
    
    # Backward pass to get gradients
    [~, grad_w, grad_b] = backward_pass(model, a, z, y_batch);
    
    # Apply frequency-aware gradient emphasis
    if !isempty(freq_feature_indices)
      # Increase gradient influence for frequency-related features in the first layer
      freq_rows = ismember(1:size(grad_w{1}, 1), freq_feature_indices);
      
      if any(freq_rows)
        # Scale up gradients for frequency feature weights
        grad_w{1}(freq_rows, :) *= freq_attention;
        
        # Also scale up gradients for corresponding biases
        grad_b{1}(freq_rows) *= freq_attention;
      endif
    endif
    
    # Apply gradient clipping
    for i = 1:(model.num_layers-1)
      # Clip gradients for weights
      grad_norm = sqrt(sum(sum(grad_w{i}.^2)));
      if grad_norm > model.clip_threshold
        scaling = model.clip_threshold / grad_norm;
        grad_w{i} *= scaling;
      endif
      
      # Clip gradients for biases
      bias_norm = sqrt(sum(grad_b{i}.^2));
      if bias_norm > model.clip_threshold
        scaling = model.clip_threshold / bias_norm;
        grad_b{i} *= scaling;
      endif
    endfor
    
    # Update with momentum
    for i = 1:(model.num_layers-1)
      # Update velocities with momentum
      velocity_w{i} = momentum * velocity_w{i} - curr_learning_rate * grad_w{i};
      velocity_b{i} = momentum * velocity_b{i} - curr_learning_rate * grad_b{i};
      
      # Apply updates
      model.weights{i} += velocity_w{i};
      model.biases{i} += velocity_b{i};
      
      # Check for NaN or Inf values
      if any(isnan(model.weights{i}(:))) || any(isinf(model.weights{i}(:))) || ...
         any(isnan(model.biases{i})) || any(isinf(model.biases{i}))
        printf("Warning: NaN or Inf values in weights/biases at iteration %d. Restoring best model.\n", iter);
        model = best_model;
        return;
      endif
    endfor
    
    # Print progress
    if mod(iter, 100) == 0
      printf("Iteration %d: train_loss=%.6f, val_loss=%.6f, lr=%.6f\n", 
             iter, train_loss, val_loss, curr_learning_rate);
    endif
  endfor
  
  # Return the best model if early stopping was triggered
  if patience_counter >= model.early_stopping_patience
    model = best_model;
    printf("Restored best model from iteration %d\n", best_iter);
  endif
  
  printf("Training completed. Final validation loss: %.6f\n", history.val_loss(end));
endfunction