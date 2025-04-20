function [forecast] = multi_step_forecast(model, last_sequence, n_steps, X_mean, X_std, y_mean, y_std)
  # Multi-step forecasting for time series prediction
  # Predicts multiple steps into the future using the trained model
  # Enhanced with error correction and prediction smoothing

  forecast = zeros(1, n_steps);
  current_seq = last_sequence;
  
  # Create a window of recent predictions for smoothing
  smooth_window_size = 3;
  recent_preds = [];

  for i = 1:n_steps
    # Predict the next value
    pred = predict(model, current_seq);

    # Handle potential NaN values
    if isnan(pred)
      printf("Warning: NaN prediction at step %d. Using last valid prediction.\n", i);
      if i > 1
        pred = forecast(i-1);  # Use the last valid prediction
      else
        pred = 0;  # Default value if first prediction is NaN
      endif
    endif
    
    # Apply moving average smoothing if we have enough predictions
    if length(recent_preds) >= smooth_window_size
      # Use exponential smoothing with more weight on the current prediction
      alpha = 0.7;  # Smoothing factor (higher = more weight on current prediction)
      smoothed_pred = alpha * pred + (1 - alpha) * mean(recent_preds);
      pred = smoothed_pred;
    endif
    
    # Add current prediction to recent predictions window
    recent_preds = [recent_preds, pred];
    if length(recent_preds) > smooth_window_size
      recent_preds = recent_preds(2:end);  # Keep only the most recent predictions
    endif

    forecast(i) = pred;

    # Update the sequence by removing the first element and adding the prediction
    current_seq = [current_seq(2:end); pred];
  endfor

  # Apply trend correction to final forecast (helps with long-term predictions)
  if n_steps > 5
    # Calculate the trend from the first few predictions
    initial_trend = mean(diff(forecast(1:min(5, n_steps))));
    
    # Apply gradual trend dampening for longer-term forecasts
    dampening_factor = 0.9;
    for i = 6:n_steps
      correction = initial_trend * dampening_factor^(i-5);
      forecast(i) = forecast(i) + correction;
    endfor
  endif

  # Denormalize forecast
  forecast = forecast * y_std + y_mean;
endfunction