function loss = cross_entropy(y_pred, y_true)
  # Cross-entropy loss function for classification tasks
  # y_pred: predicted probabilities (output of softmax)
  # y_true: true labels in one-hot encoded format
  
  # Add small epsilon to avoid log(0)
  epsilon = 1e-15;
  y_pred = max(min(y_pred, 1 - epsilon), epsilon);
  
  # Compute cross entropy
  loss = -mean(sum(y_true .* log(y_pred), 1));
endfunction