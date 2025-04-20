function [model, history] = train_enhanced(model, X, y, val_X=[], val_y=[])
  # Enhanced training function with adaptive learning rate, momentum, and early stopping
  # Optimized for high-precision predictions with complex functions
  # X: training features
  # y: training labels
  # val_X, val_y: optional validation data

  history = struct("train_loss", [], "val_loss", []);
  
  # Initialize parameters for enhanced training
  n_samples = size(X, 2);
  
  # Use model-provided batch size if available, otherwise use default
  if (isfield(model, "batch_size"))
    batch_size = model.batch_size;
    printf("Using custom batch size: %d\n", batch_size);
  else
    batch_size = min(n_samples, 64);  # Larger batch size for stability
  endif
  
  # Set early stopping parameters if not already set
  if (!isfield(model, "early_stopping_patience"))
    model.early_stopping_patience = 50;  # Default patience
  endif
  
  # Set learning rate decay if not already set
  if (!isfield(model, "learning_rate_decay"))
    model.learning_rate_decay = 0.9999;  # Default decay rate
  endif
  
  # Initialize momentum for faster convergence (use model-provided value if available)
  if (isfield(model, "momentum"))
    momentum = model.momentum;
  else
    momentum = 0.9;  # Default momentum
  endif
  
  # Check for adaptive learning rate flag
  adaptive_lr = false;
  if (isfield(model, "adaptive_lr"))
    adaptive_lr = model.adaptive_lr;
    if (adaptive_lr)
      printf("Using adaptive learning rate scheduling\n");
    endif
  endif
  
  printf("Using momentum value: %.2f\n", momentum);
  
  # Initialize velocities for momentum
  velocity_w = cell(model.num_layers-1, 1);
  velocity_b = cell(model.num_layers-1, 1);
  
  for i = 1:(model.num_layers-1)
    velocity_w{i} = zeros(size(model.weights{i}));
    velocity_b{i} = zeros(size(model.biases{i}));
  endfor
  
  # Variables for early stopping
  best_val_loss = Inf;
  patience_counter = 0;
  best_model = model;
  best_iter = 0;
  
  # Learning rate schedule parameters
  curr_learning_rate = model.learning_rate;
  min_learning_rate = model.learning_rate * 0.001;  # Don't let LR go below 0.1% of initial
  
  # Track improvements for periodic reporting
  last_report_loss = Inf;
  
  printf("Starting enhanced training with adaptive techniques...\n");
  
  # Enable high-precision mode if the model has many iterations
  high_precision = model.max_iter > 10000;
  if (high_precision)
    report_interval = 500;  # Report less frequently in high-precision mode
    printf("Using high-precision training mode...\n");
  else
    report_interval = 100;  # Standard reporting interval
  endif
  
  for iter = 1:model.max_iter
    # Apply learning rate decay
    curr_learning_rate *= model.learning_rate_decay;
    
    # Ensure learning rate doesn't get too small
    curr_learning_rate = max(curr_learning_rate, min_learning_rate);
    
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
    
    # Compute loss with L2 regularization
    train_loss = model.loss_function(a{end}, y_batch);
    
    # Add regularization term to loss
    reg_term = 0;
    for i = 1:(model.num_layers-1)
      reg_term += 0.5 * model.reg_lambda * sum(sum(model.weights{i}.^2));
    endfor
    train_loss += reg_term / size(X_batch, 2);
    
    # Check for NaN loss and break if found
    if (isnan(train_loss))
      printf("Warning: NaN loss detected at iteration %d. Restoring best model.\n", iter);
      model = best_model;
      break;
    endif
    
    history.train_loss(end+1) = train_loss;
    
    # Compute validation loss if validation data is provided
    if !isempty(val_X) && !isempty(val_y)
      [val_a, ~] = forward_pass(model, val_X);
      val_loss = model.loss_function(val_a{end}, val_y);
      
      # Add regularization to validation loss for consistent comparison
      val_reg_term = 0;
      for i = 1:(model.num_layers-1)
        val_reg_term += 0.5 * model.reg_lambda * sum(sum(model.weights{i}.^2));
      endfor
      val_loss += val_reg_term / size(val_X, 2);
      
      history.val_loss(end+1) = val_loss;
      
      # Early stopping logic with improvements
      if val_loss < best_val_loss
        # Calculate improvement percentage
        if best_val_loss != Inf
          improvement = (best_val_loss - val_loss) / best_val_loss * 100;
        else
          improvement = 100;
        endif
        
        best_val_loss = val_loss;
        best_model = model;
        best_iter = iter;
        patience_counter = 0;
        
        # Report significant improvements
        if val_loss < last_report_loss * 0.95  # Report 5% or better improvements
          printf("Significant improvement at iter %d: val_loss=%.6f (%.2f%% better)\n", 
                 iter, val_loss, improvement);
          last_report_loss = val_loss;
        endif
      else
        patience_counter += 1;
        
        # Apply adaptive learning rate reduction based on plateau length
        if high_precision
          # More aggressive LR reduction for high-precision mode
          if patience_counter > 0 && mod(patience_counter, 25) == 0
            # Reduce learning rate according to plateau length
            reduction_factor = max(0.5, 1.0 - (patience_counter / (2 * model.early_stopping_patience)));
            curr_learning_rate *= reduction_factor;
            printf("High-precision LR adjustment to %.8f at iteration %d (patience: %d)\n", 
                   curr_learning_rate, iter, patience_counter);
          endif
        else
          # Standard LR reduction
          if patience_counter > 10 && mod(patience_counter, 10) == 0
            curr_learning_rate *= 0.8;  # Reduce learning rate by 20%
            printf("Reducing learning rate to %.6f at iteration %d\n", curr_learning_rate, iter);
          endif
        endif
        
        # Stop if no improvement for 'patience' iterations
        if patience_counter >= model.early_stopping_patience
          printf("Early stopping at iteration %d. Best was at iteration %d.\n", 
                 iter, best_iter);
          model = best_model;
          break;
        endif
      endif
    endif
    
    # Backward pass to get gradients
    [~, grad_w, grad_b] = backward_pass(model, a, z, y_batch);
    
    # Apply gradient clipping to prevent exploding gradients
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
    
    # Apply adaptive learning rate scheduling if enabled
    if adaptive_lr && !isempty(val_X) && !isempty(val_y)
      # Reduce learning rate by factor based on patience counter
      if patience_counter > 0
        # More aggressive reduction for longer plateaus
        if patience_counter > 100
          adaptive_factor = 0.1 + 0.9 * exp(-patience_counter / 100);
        else
          adaptive_factor = 0.3 + 0.7 * exp(-patience_counter / 50);
        endif
        
        # Adjust the learning rate for this iteration
        iter_learning_rate = curr_learning_rate * adaptive_factor;
        
        # Report significant learning rate adjustments
        if mod(patience_counter, 50) == 0
          printf("Adaptive LR adjustment: %.8f at iter %d (patience: %d)\n", 
                 iter_learning_rate, iter, patience_counter);
        endif
      else
        # No plateau, use normal learning rate
        iter_learning_rate = curr_learning_rate;
      endif
    else
      # Standard learning rate
      iter_learning_rate = curr_learning_rate;
    endif
    
    # Update with momentum
    for i = 1:(model.num_layers-1)
      # Calculate velocity updates with momentum
      velocity_w{i} = momentum * velocity_w{i} - iter_learning_rate * grad_w{i};
      velocity_b{i} = momentum * velocity_b{i} - iter_learning_rate * grad_b{i};
      
      # Apply updates
      model.weights{i} += velocity_w{i};
      model.biases{i} += velocity_b{i};
      
      # Check for NaN or Inf in weights and biases
      if any(isnan(model.weights{i}(:))) || any(isinf(model.weights{i}(:))) || ...
         any(isnan(model.biases{i})) || any(isinf(model.biases{i}))
        printf("Warning: NaN or Inf values detected at iteration %d. Restoring best model.\n", iter);
        model = best_model;
        return;
      endif
    endfor
    
    # Print progress at appropriate intervals
    if mod(iter, report_interval) == 0
      if !isempty(val_X) && !isempty(val_y)
        printf("Iter %d: train_loss=%.6f, val_loss=%.6f, lr=%.8f\n", 
               iter, train_loss, val_loss, curr_learning_rate);
      else
        printf("Iter %d: train_loss=%.6f, lr=%.8f\n", 
               iter, train_loss, curr_learning_rate);
      endif
    endif
  endfor
  
  # If early stopping was used, restore the best model
  if !isempty(val_X) && !isempty(val_y) && patience_counter >= model.early_stopping_patience
    model = best_model;
    printf("Restored best model from iteration %d with validation loss %.6f\n", 
           best_iter, best_val_loss);
  endif
  
  # Final report
  if !isempty(val_X) && !isempty(val_y)
    printf("Training completed. Final validation loss: %.8f\n", history.val_loss(end));
  else
    printf("Training completed. Final training loss: %.8f\n", history.train_loss(end));
  endif
endfunction