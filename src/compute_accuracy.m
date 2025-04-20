function accuracy = compute_accuracy(y_pred, y_true)
  # Compute accuracy for classification tasks
  # y_pred: model predictions (output of softmax)
  # y_true: true labels in one-hot encoded format
  
  # Convert probabilities to predicted class labels
  [~, pred_labels] = max(y_pred, [], 1);
  
  # Convert one-hot encoded true labels to class indices
  [~, true_labels] = max(y_true, [], 1);
  
  # Compute accuracy
  accuracy = mean(pred_labels == true_labels);
endfunction