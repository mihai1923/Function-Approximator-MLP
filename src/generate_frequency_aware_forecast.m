function forecast = generate_frequency_aware_forecast(model, last_sequence, n_steps, frequencies, last_x, X_mean, X_std, y_mean, y_std)
  # Specialized forecasting function for seasonal patterns
  # Uses known frequencies to maintain phase coherence in long-term forecasts
  # This prevents the typical phase drift seen in standard recurrent forecasting
  
  # Extract sequence length from the input features (excluding frequency features)
  seq_length = length(last_sequence) - 2 * length(frequencies);
  
  # Initialize forecast array
  forecast = zeros(1, n_steps);
  
  # Create the initial sequence for forecasting (normalized)
  current_seq = last_sequence;
  
  # Get the last value of x from the input series
  current_x = last_x;
  
  # Step size in the x domain (assume uniform sampling)
  step_size = 1;  # Default step size
  
  # Generate each step of the forecast
  for i = 1:n_steps
    # Update the current x value for the new forecast point
    current_x = current_x + step_size;
    
    # Make prediction using current sequence
    pred = predict(model, current_seq);
    
    # Handle potential NaN values
    if isnan(pred)
      printf("Warning: NaN prediction at step %d. Using last valid prediction.\n", i);
      if i > 1
        pred = forecast(i-1);  # Use last valid prediction
      else
        pred = 0;  # Default if first prediction is NaN
      endif
    endif
    
    # Store the prediction (still normalized)
    forecast(i) = pred;
    
    # Update the sequence for next prediction by shifting values
    # Remove oldest value and add new prediction
    current_seq(1:end-1) = current_seq(2:end);
    
    # If we have frequency information, update the special frequency features
    # This is the key innovation - maintain accurate phase for seasonal components
    if !isempty(frequencies)
      # Update time-series part of the sequence
      current_seq(seq_length) = pred;
      
      # Update frequency features based on the new x value
      feature_idx = seq_length + 1;
      for freq_idx = 1:length(frequencies)
        freq = frequencies(freq_idx);
        
        # Update sine component (normalized)
        sine_val = sin(freq * current_x);
        # Normalize using stored mean and std
        norm_sine = (sine_val - X_mean(feature_idx)) / X_std(feature_idx);
        current_seq(feature_idx) = norm_sine;
        
        # Update cosine component (normalized)
        cosine_val = cos(freq * current_x);
        # Normalize using stored mean and std
        norm_cosine = (cosine_val - X_mean(feature_idx+1)) / X_std(feature_idx+1);
        current_seq(feature_idx+1) = norm_cosine;
        
        feature_idx += 2;
      endfor
    else
      # Without frequency features, just update the last value
      current_seq(end) = pred;
    endif
  endfor
  
  # Denormalize the forecast
  forecast = forecast * y_std + y_mean;
  
  return;
endfunction