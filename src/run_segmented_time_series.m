function run_segmented_time_series(ts_func, ts_func_str, precision_mode="standard")
  # Helper function to run segmented time series with custom function
  
  printf("\nGenerating segmented time series using function: %s\n\n", ts_func_str);
  
  # Check if we're using a string-based function that might need element-wise operators
  if ischar(ts_func) || strcmp(class(ts_func), "inline function")
    # First, remove any spaces around mathematical operators to normalize the expression
    clean_ts_func_str = strrep(ts_func_str, " ^ ", "^");
    clean_ts_func_str = strrep(clean_ts_func_str, " ^", "^");
    clean_ts_func_str = strrep(clean_ts_func_str, "^ ", "^");
    clean_ts_func_str = strrep(clean_ts_func_str, " * ", "*");
    clean_ts_func_str = strrep(clean_ts_func_str, " *", "*");
    clean_ts_func_str = strrep(clean_ts_func_str, "* ", "*");
    clean_ts_func_str = strrep(clean_ts_func_str, " / ", "/");
    clean_ts_func_str = strrep(clean_ts_func_str, " /", "/");
    clean_ts_func_str = strrep(clean_ts_func_str, "/ ", "/");
    
    # Create a safe version of the function that ensures element-wise operations
    safe_ts_func_str = strrep(clean_ts_func_str, "^", ".^");
    safe_ts_func_str = strrep(safe_ts_func_str, "*", ".*");
    safe_ts_func_str = strrep(safe_ts_func_str, "/", "./");
    
    # Fix operations that might already have element-wise operators to avoid double dots
    safe_ts_func_str = strrep(safe_ts_func_str, "..^", ".^");
    safe_ts_func_str = strrep(safe_ts_func_str, "..*", ".*");
    safe_ts_func_str = strrep(safe_ts_func_str, "../", "./");

    printf("Input function: %s\n", ts_func_str);
    printf("Clean function: %s\n", clean_ts_func_str);
    printf("Element-wise operations: %s\n", safe_ts_func_str);
    safe_ts_func = inline(safe_ts_func_str, "x");
  else
    # If it's already a proper function handle, use it directly
    safe_ts_func = ts_func;
  endif

  # Create a shorter safe name for filenames
  if (length(ts_func_str) > 20)
    # Use hash function to create a unique but shorter identifier
    safe_name = ["custom_ts_", num2str(sum(ts_func_str))];
    printf("Function name too long for filename. Using: %s\n", safe_name);
  else
    safe_name = strrep(ts_func_str, "*", "star");
    safe_name = strrep(safe_name, " ", "_");
    safe_name = strrep(safe_name, "+", "plus");
    safe_name = strrep(safe_name, "^", "pow");
    safe_name = strrep(safe_name, "(", "");
    safe_name = strrep(safe_name, ")", "");
  endif

  # Generate data with higher resolution for better precision
  printf("Generating high-precision dataset for segmented time series forecasting...\n");
  n_points = 2000;  # Number of points for better resolution
  seq_length = 40;  # Longer sequences for capturing more complex patterns
  
  if ischar(ts_func) || strcmp(class(ts_func), "inline function")
    [X, y, time_series] = generate_custom_time_series_full(n_points, seq_length, safe_ts_func);
  else
    [X, y, time_series] = generate_custom_time_series_full(n_points, seq_length, ts_func);
  endif

  # Generate time points (just for visualization)
  time_points = 1:length(time_series);
  
  # Plot the full time series
  figure(1);
  plot(time_points, time_series, 'b-', 'LineWidth', 1.5);
  title(sprintf("Full Time Series Data: %s", ts_func_str));
  xlabel("Time Step");
  ylabel("Value");
  grid on;
  
  # Define forecast horizon
  forecast_horizon = 150;
  
  # Create segmented forecasting options
  options = struct();
  options.segment_detection_method = "adaptive";  # Can be "adaptive", "equal", or "pattern"
  options.n_segments = 3;                         # Number of segments to use
  options.seq_length = seq_length;                # Sequence length for the models
  options.min_segment_size = 120;                 # Minimum points per segment
  
  printf("\n===== Segmented Time Series Forecasting =====\n");
  printf("Using adaptive segmentation to find pattern changes in the time series\n");
  
  # Run segmented forecasting
  tic();  # Start timing
  [segmented_forecast, segment_info] = segmented_time_series_forecast(time_series, forecast_horizon, options);
  segmented_time = toc();  # Record time
  
  printf("Segmented forecasting completed in %.2f minutes.\n", segmented_time/60);
  
  # For comparison, also run standard (non-segmented) forecasting
  printf("\nFor comparison, running standard (non-segmented) forecasting...\n");
  
  # Split into training and test sets (80/20 split)
  n_samples = size(X, 2);
  split_idx = floor(0.8 * n_samples);
  X_train = X(:, 1:split_idx);
  y_train = y(:, 1:split_idx);
  X_test = X(:, (split_idx+1):end);
  y_test = y(:, (split_idx+1):end);

  # Normalize data
  X_mean = mean(X_train(:));
  X_std = std(X_train(:)) + 1e-8;
  X_train = (X_train - X_mean) ./ X_std;
  X_test = (X_test - X_mean) ./ X_std;

  y_mean = mean(y_train);
  y_std = std(y_train) + 1e-8;
  y_train = (y_train - y_mean) ./ y_std;
  y_test = (y_test - y_mean) ./ y_std;
  
  # Standard model configuration
  layer_sizes = [seq_length, 128, 64, 32, 1];
  
  # Initialize neural network
  model = initNeuralNetwork(layer_sizes);
  model.activation = @tanh;
  model.activation_prime = @tanh_prime;
  model.output_activation = @identity;
  model.output_activation_prime = @identity_prime;
  model.loss_function = @mean_squared_error;
  model.loss_function_prime = @mean_squared_error_prime;
  model.learning_rate = 0.001;
  model.max_iter = 3000;
  model.early_stopping_patience = 100;
  model.momentum = 0.9;
  
  # Train standard model
  tic();  # Start timing
  printf("Training standard model (non-segmented approach)...\n");
  [model, history] = train(model, X_train, y_train, X_test, y_test);
  standard_time = toc();  # Record time
  
  printf("Standard model training completed in %.2f minutes.\n", standard_time/60);
  
  # Evaluate on test set
  predictions = predict(model, X_test);
  mse = mean((predictions - y_test).^2);
  printf("Test MSE for standard model: %.6f\n", mse);
  
  # Generate standard forecast
  last_sequence = time_series(end-seq_length+1:end)';
  norm_last_sequence = (last_sequence - X_mean) ./ X_std;
  
  # Multi-step forecasting with standard model
  standard_forecast = zeros(1, forecast_horizon);
  current_seq = norm_last_sequence;
  
  for i = 1:forecast_horizon
    # Make prediction
    pred = predict(model, current_seq);
    
    # Denormalize
    standard_forecast(i) = pred * y_std + y_mean;
    
    # Update sequence for next step
    current_seq = [current_seq(2:end); pred];
  endfor
  
  # Visualize the forecasts
  # Plot original series with forecasts
  n_history = min(200, length(time_series));
  historical_times = time_points(end-n_history+1:end);
  time_step = 1;  # Since we're using integer indices
  forecast_times = time_points(end) + time_step * (1:length(segmented_forecast));
  
  # Create comparison figure
  figure(2, 'position', [100, 100, 1000, 600]);
  
  # Plot historical data and forecasts
  subplot(2, 1, 1);
  hold on;
  
  # Historical data
  plot(historical_times, time_series(end-n_history+1:end), 'b-', 'linewidth', 2);
  
  # Forecasts
  plot(forecast_times, segmented_forecast, 'r-', 'linewidth', 2);
  plot(forecast_times, standard_forecast, 'g--', 'linewidth', 1.5);
  
  # Add vertical line at forecast start
  xline(time_points(end), 'k--', 'linewidth', 1.5);
  
  title('Comparison of Segmented vs Standard Forecasting');
  xlabel('Time');
  ylabel('Value');
  legend('Historical Data', 'Segmented Forecast', 'Standard Forecast', 'Forecast Start');
  grid on;
  
  # Plot the original time series with segment boundaries
  subplot(2, 1, 2);
  plot(time_points, time_series, 'b-', 'linewidth', 1.5);
  hold on;
  
  # Add vertical lines for segment boundaries
  boundaries = segment_info.boundaries;
  for i = 1:length(boundaries)
    if boundaries(i) < length(time_series)
      xline(time_points(boundaries(i)), 'r--', 'linewidth', 1.5);
    endif
  endfor
  
  title('Original Time Series with Detected Segments');
  xlabel('Time');
  ylabel('Value');
  grid on;
  
  # Make sure plots directory exists
  if !exist("../plots", "dir")
    mkdir("../plots");
  endif
  
  # Save the visualization
  segmented_plot_path = ["../plots/segmented_forecasting_", safe_name, ".png"];
  print(figure(2), "-dpng", "-r300", segmented_plot_path);
  printf("Segmented forecast visualization saved to '%s'\n", segmented_plot_path);
  
  # Calculate forecast accuracy metrics
  # We don't have actual future values, so we'll compare the approaches
  # by looking at how well each predicts the end of the known data
  test_horizon = min(50, length(y_test));
  
  # Prepare the sequences for evaluation
  eval_mse_standard = zeros(1, test_horizon);
  eval_mse_segmented = zeros(1, test_horizon);
  
  # Get values for comparison
  actual_end = time_series(end-test_horizon+1:end);
  
  # Generate forecasts starting from earlier point
  early_sequence = time_series(end-test_horizon-seq_length+1:end-test_horizon)';
  
  # Standard forecast from earlier point
  norm_early_sequence = (early_sequence - X_mean) ./ X_std;
  early_standard_forecast = zeros(1, test_horizon);
  current_seq = norm_early_sequence;
  
  for i = 1:test_horizon
    pred = predict(model, current_seq);
    early_standard_forecast(i) = pred * y_std + y_mean;
    current_seq = [current_seq(2:end); pred];
  endfor
  
  # Segmented forecast from earlier point
  early_segmented_forecast = segmented_time_series_forecast(time_series(1:end-test_horizon), test_horizon, options);
  
  # Calculate error metrics
  standard_mse = mean((early_standard_forecast - actual_end).^2);
  segmented_mse = mean((early_segmented_forecast - actual_end).^2);
  
  standard_mae = mean(abs(early_standard_forecast - actual_end));
  segmented_mae = mean(abs(early_segmented_forecast - actual_end));
  
  # Print comparison results
  printf("\n===== Forecast Accuracy Comparison =====\n");
  printf("Metrics calculated on the last %d known data points\n", test_horizon);
  printf("Standard approach MSE: %.6f\n", standard_mse);
  printf("Segmented approach MSE: %.6f\n", segmented_mse);
  printf("Standard approach MAE: %.6f\n", standard_mae);
  printf("Segmented approach MAE: %.6f\n", segmented_mae);
  
  # Calculate improvement percentage
  if standard_mse > 0
    mse_improvement = (standard_mse - segmented_mse) / standard_mse * 100;
    printf("MSE Improvement with segmented approach: %.2f%%\n", mse_improvement);
  endif
  
  if standard_mae > 0
    mae_improvement = (standard_mae - segmented_mae) / standard_mae * 100;
    printf("MAE Improvement with segmented approach: %.2f%%\n", mae_improvement);
  endif
  
  # Compare computation time
  time_ratio = segmented_time / standard_time;
  printf("Computation time ratio (Segmented/Standard): %.2fx\n", time_ratio);
  
  # Plot the accuracy comparison
  figure(3, 'position', [100, 100, 1000, 400]);
  hold on;
  
  plot(1:test_horizon, actual_end, 'b-', 'linewidth', 2);
  plot(1:test_horizon, early_standard_forecast, 'g--', 'linewidth', 1.5);
  plot(1:test_horizon, early_segmented_forecast, 'r-', 'linewidth', 1.5);
  
  title('Forecast Accuracy Comparison on Known Data');
  xlabel('Time Step');
  ylabel('Value');
  legend('Actual', 'Standard Forecast', 'Segmented Forecast');
  grid on;
  
  # Save the accuracy comparison
  accuracy_plot_path = ["../plots/forecast_accuracy_", safe_name, ".png"];
  print(figure(3), "-dpng", "-r300", accuracy_plot_path);
  printf("Forecast accuracy comparison saved to '%s'\n", accuracy_plot_path);
  
  # Save segmented forecast model
  if !exist("../data", "dir")
    mkdir("../data");
  endif
  
  model_path = ["../data/segmented_forecast_model_", safe_name, ".mat"];
  save("-binary", model_path, "segment_info");
  printf("Segmented forecast model saved to '%s'\n", model_path);
  
  # Return the segmented forecast
  return;
endfunction

# Helper function to generate full time series plus sequences
function [X, y, full_series] = generate_custom_time_series_full(n_points, seq_length, func)
  # Generate a time series based on a custom function with noise
  
  # Create base series with custom function
  x = linspace(0, 10, n_points);
  
  # Evaluate the function to generate the time series
  # Use try-catch to handle potential errors with function evaluation
  try
    base_series = func(x);
    
    # Add small random noise to make it more realistic
    noise_level = 0.05 * max(abs(base_series));
    base_series = base_series + noise_level * randn(size(base_series));
    
    # Return the full series first
    full_series = base_series;
    
    # Create sequence pairs for supervised learning
    X = zeros(seq_length, n_points - seq_length);
    y = zeros(1, n_points - seq_length);
    
    for i = 1:(n_points - seq_length)
      X(:, i) = base_series(i:(i+seq_length-1))';
      y(:, i) = base_series(i+seq_length);
    endfor
    
  catch
    err = lasterror;
    fprintf("Error evaluating function: %s\n", err.message);
    X = [];
    y = [];
    full_series = [];
  end_try_catch
endfunction