function d_loss = cross_entropy_prime(y_pred, y_true)
  # Derivative of cross-entropy loss with respect to y_pred
  # When used with softmax output, this simplifies to (y_pred - y_true)
  
  # Add small epsilon for numerical stability
  epsilon = 1e-15;
  y_pred = max(min(y_pred, 1 - epsilon), epsilon);
  
  # The gradient of cross-entropy with respect to softmax outputs is simply (y_pred - y_true)
  d_loss = y_pred - y_true;
endfunction