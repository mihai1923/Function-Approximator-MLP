function run_enhanced_seasonal_prediction(ts_func, ts_func_str, options)
  # Enhanced seasonal pattern prediction with specialized multi-frequency handling
  # Optimized specifically for capturing patterns like 2*sin(0.1*x) + sin(x)
  # With support for decomposing and predicting both seasonal and rapid oscillations
  
  printf("\nRunning enhanced seasonal pattern prediction for: %s\n\n", ts_func_str);
  
  # Set defaults for options if not provided
  if !isfield(options, "sequence_length")
    options.sequence_length = 30;  # Longer sequence to capture seasonal patterns
  endif
  
  if !isfield(options, "frequencies")
    # Automatically detect frequencies if not provided (for option 3, we know them)
    options.frequencies = [0.1, 1];
  endif
  
  # Generate a more specialized dataset for seasonal patterns
  printf("Generating optimized seasonal pattern dataset...\n");
  
  # Use a longer time range to better capture seasonal patterns
  n_points = 2000;  # More data points for better frequency resolution
  seq_length = options.sequence_length;
  
  # Generate a time range that captures multiple full cycles of the slowest frequency
  # For a frequency of 0.1, a full cycle is 2π/0.1 ≈ 63 time units
  min_freq = min(options.frequencies);
  cycle_length = 2*pi / min_freq;
  x_range = cycle_length * 10;  # Capture multiple full cycles
  
  x = linspace(0, x_range, n_points);
  
  # Generate the time series with controlled noise
  base_series = ts_func(x);
  
  # Add minimal noise to the base series (very small for better prediction)
  noise_level = 0.02;
  base_series = base_series + noise_level * randn(1, n_points);
  
  # Create sequences for input and output
  X = zeros(seq_length, n_points - seq_length);
  y = zeros(1, n_points - seq_length);
  
  for i = 1:(n_points - seq_length)
    X(:, i) = base_series(i:(i+seq_length-1))';
    y(:, i) = base_series(i+seq_length);
  endfor
  
  # Perform frequency domain analysis to improve feature extraction
  printf("Performing frequency domain analysis...\n");
  
  # Use FFT to analyze the dominant frequencies in the data
  fft_result = fft(base_series);
  power_spectrum = abs(fft_result(1:floor(n_points/2))).^2;
  frequencies = (0:(n_points/2-1)) / n_points * 2*pi;  # Convert to angular frequencies
  
  # Find dominant frequencies (peaks in power spectrum)
  [~, peak_indices] = sort(power_spectrum, 'descend');
  dominant_freqs = frequencies(peak_indices(1:5));  # Get top 5 frequencies
  
  printf("Detected dominant frequencies: [");
  printf("%.3f ", dominant_freqs(1:min(5, length(dominant_freqs))));
  printf("]\n");
  
  # Create enhanced features based on frequency analysis
  # Add engineered features that capture sine and cosine components at key frequencies
  enhanced_X = zeros(seq_length + 2*length(options.frequencies), size(X, 2));
  
  # Copy original sequence data
  enhanced_X(1:seq_length, :) = X;
  
  # Add engineered frequency features
  feature_idx = seq_length + 1;
  for freq_idx = 1:length(options.frequencies)
    freq = options.frequencies(freq_idx);
    
    # Phase features at this specific frequency
    for i = 1:size(X, 2)
      # Add sine and cosine features at this frequency
      # Start position for this sequence
      start_pos = i;
      
      # Add sine feature (capturing phase)
      enhanced_X(feature_idx, i) = sin(freq * x(start_pos));
      
      # Add cosine feature (capturing phase)
      enhanced_X(feature_idx+1, i) = cos(freq * x(start_pos));
    endfor
    
    feature_idx += 2;
  endfor
  
  printf("Added %d frequency-aware features to input data\n", feature_idx - seq_length - 1);
  
  # Split data into training and test sets (80/20)
  train_size = floor(0.8 * size(enhanced_X, 2));
  X_train = enhanced_X(:, 1:train_size);
  y_train = y(:, 1:train_size);
  X_test = enhanced_X(:, train_size+1:end);
  y_test = y(:, train_size+1:end);
  
  # Normalize the data
  X_mean = mean(X_train, 2);
  X_std = std(X_train, 0, 2);
  X_std(X_std < 1e-8) = 1;  # Avoid division by zero
  
  X_train_norm = (X_train - X_mean) ./ X_std;
  X_test_norm = (X_test - X_mean) ./ X_std;
  
  y_mean = mean(y_train);
  y_std = std(y_train);
  y_train_norm = (y_train - y_mean) ./ y_std;
  y_test_norm = (y_test - y_mean) ./ y_std;
  
  # Create optimized network architecture for seasonal patterns
  # We use a specialized architecture with explicit frequency encoding
  input_size = size(X_train_norm, 1);
  
  # Network with specialized structure for seasonal patterns
  layer_sizes = [input_size, 128, 64, 32, 1];
  model = initNeuralNetwork(layer_sizes);
  
  # Configure model for seasonal pattern prediction
  model.activation = @tanh;
  model.activation_prime = @tanh_prime;
  model.output_activation = @identity;
  model.output_activation_prime = @identity_prime;
  model.loss_function = @mean_squared_error;
  model.loss_function_prime = @mean_squared_error_prime;
  
  # Use optimized hyperparameters for seasonal patterns
  model.learning_rate = 0.002;
  model.reg_lambda = 0.0001;  # Very light regularization for seasonal patterns
  model.max_iter = 4000;
  model.clip_threshold = 1.0;
  model.early_stopping_patience = 100;
  
  # Print architecture summary
  printf("\nEnhanced Seasonal Pattern Model Architecture:\n");
  printf("- Input size: %d (includes %d frequency features)\n", 
         input_size, input_size - seq_length);
  printf("- Hidden layers: [");
  printf("%d ", layer_sizes(2:end-1));
  printf("]\n");
  printf("- Output size: %d\n", layer_sizes(end));
  printf("- Learning rate: %.4f\n", model.learning_rate);
  printf("- Max iterations: %d\n", model.max_iter);
  
  # Train the model
  printf("\nTraining enhanced seasonal pattern model...\n");
  [model, history] = train_with_frequency_awareness(model, X_train_norm, y_train_norm, 
                                                   X_test_norm, y_test_norm, options);
  
  # Evaluate on test set
  predictions = predict(model, X_test_norm);
  mse = mean((predictions - y_test_norm).^2);
  
  # Denormalize predictions
  predictions_orig = predictions * y_std + y_mean;
  y_test_orig = y_test_norm * y_std + y_mean;
  
  # Calculate error metrics on denormalized data
  mse_orig = mean((predictions_orig - y_test_orig).^2);
  mae_orig = mean(abs(predictions_orig - y_test_orig));
  rmse_orig = sqrt(mse_orig);
  
  printf("\nTest Metrics:\n");
  printf("- Mean Squared Error: %.6f\n", mse_orig);
  printf("- Mean Absolute Error: %.6f\n", mae_orig);
  printf("- Root Mean Squared Error: %.6f\n", rmse_orig);
  
  # Calculate relative error
  mean_abs_y = mean(abs(y_test_orig));
  relative_error = mae_orig / mean_abs_y * 100;
  printf("- Relative Error: %.2f%%\n", relative_error);
  
  # Plot training history
  figure(1);
  plot(1:length(history.train_loss), history.train_loss, 'b-', 'LineWidth', 2);
  hold on;
  plot(1:length(history.val_loss), history.val_loss, 'r-', 'LineWidth', 2);
  title("Training History - Seasonal Pattern Model");
  xlabel("Iterations");
  ylabel("Loss");
  legend("Training Loss", "Validation Loss");
  grid on;
  
  # Plot predictions vs actual values
  test_length = 200;  # Number of points to visualize
  start_idx = 1;
  end_idx = min(test_length, length(predictions_orig));
  
  figure(2);
  hold on;
  plot(1:length(y_test_orig(start_idx:end_idx)), y_test_orig(start_idx:end_idx), 'b-', 'LineWidth', 2);
  plot(1:length(predictions_orig(start_idx:end_idx)), predictions_orig(start_idx:end_idx), 'r--', 'LineWidth', 2);
  title(sprintf("Seasonal Pattern Prediction: %s", ts_func_str));
  xlabel("Time Step");
  ylabel("Value");
  legend("Actual", "Predicted");
  grid on;
  
  # Generate multi-step forecast
  printf("\nGenerating multi-step forecast...\n");
  forecast_horizon = 100;  # Longer horizon for seasonal patterns
  
  # Get the last test sequence for forecasting
  last_sequence = X_test_norm(:, end);
  
  # Create enhanced forecast with frequency awareness
  forecast = generate_frequency_aware_forecast(model, last_sequence, forecast_horizon, 
                                              options.frequencies, x(end), X_mean, X_std, y_mean, y_std);
  
  # Plot the forecast with some history for context
  history_steps = 100;
  last_known_values = y_test_orig(end-history_steps:end);
  
  figure(3);
  hold on;
  plot(1:length(last_known_values), last_known_values, 'b-', 'LineWidth', 2);
  plot((length(last_known_values)+1):(length(last_known_values)+forecast_horizon), forecast, 'r--', 'LineWidth', 2);
  title(sprintf("Multi-step Seasonal Forecast: %s", ts_func_str));
  xlabel("Time Step");
  ylabel("Value");
  legend("Historical Data", "Forecast");
  grid on;
  
  # Create a safe filename for saving results
  safe_name = strrep(ts_func_str, "*", "star");
  safe_name = strrep(safe_name, " ", "_");
  safe_name = strrep(safe_name, "+", "plus");
  safe_name = strrep(safe_name, "^", "pow");
  safe_name = strrep(safe_name, "(", "");
  safe_name = strrep(safe_name, ")", "");
  safe_name = strrep(safe_name, "noise", "");
  
  # Make sure plots directory exists
  if !exist("../plots", "dir")
    mkdir("../plots");
  endif
  
  # Save plots with descriptive names
  print(figure(1), "-dpng", sprintf("../plots/seasonal_training_history_%s.png", safe_name));
  print(figure(2), "-dpng", sprintf("../plots/seasonal_predictions_%s.png", safe_name));
  print(figure(3), "-dpng", sprintf("../plots/seasonal_forecast_%s.png", safe_name));
  
  # Save the model
  if !exist("../data", "dir")
    mkdir("../data");
  endif
  
  # Save model with seasonal pattern specific name
  save("-binary", sprintf("../data/seasonal_model_%s.mat", safe_name), "model", "options");
  
  printf("\nSeasonal pattern model saved and results plotted.\n");
endfunction