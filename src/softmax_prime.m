function dy = softmax_prime(x)
  # Derivative of softmax activation function
  # For softmax, this is a Jacobian matrix, but we typically use it with 
  # cross-entropy loss which simplifies the gradient to (y_pred - y_true)
  # This function is included for completeness, but often unused directly
  
  # When used with cross-entropy loss, the derivative calculation
  # is handled in cross_entropy_prime.m, which returns (y_pred - y_true)
  
  # We return a dummy value since the actual Jacobian is complex
  # and not needed when using cross-entropy loss
  dy = ones(size(x));
endfunction