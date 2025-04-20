function loss = compute_loss(model, y_pred, y_true)
  # Compute loss based on model configuration
  
  # Check if a custom loss function is provided
  if isfield(model, "loss_function") && isa(model.loss_function, "function_handle")
    loss = model.loss_function(y_pred, y_true);
    return;
  endif
  
  # Default: MSE for regression, cross-entropy for classification
  if isfield(model, "task") && strcmp(model.task, "classification")
    # Cross-entropy loss for classification
    epsilon = 1e-15;
    y_pred = max(min(y_pred, 1 - epsilon), epsilon);
    loss = -mean(sum(y_true .* log(y_pred), 1));
  else
    # Default to MSE for regression
    loss = mean(sum((y_pred - y_true).^2, 1));
  endif
endfunction