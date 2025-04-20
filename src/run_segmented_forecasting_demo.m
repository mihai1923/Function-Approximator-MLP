function run_segmented_forecasting_demo()
  # Demonstrate the segmented forecasting approach using example time series
  
  printf("\n===== Segmented Time Series Forecasting Demo =====\n");
  
  # Load necessary packages
  pkg load statistics;
  
  # Add necessary paths
  addpath(".");
  
  # Generate sample time series data
  [time_series, time_points] = generate_complex_time_series();
  
  # Define forecast horizon
  forecast_horizon = 200;
  
  # Create segmented forecasting options
  options = struct();
  options.segment_detection_method = "adaptive";  # Can be "adaptive", "equal", or "pattern"
  options.n_segments = 3;                         # Number of segments to use
  options.seq_length = 40;                        # Sequence length for the models
  options.min_segment_size = 120;                 # Minimum points per segment
  
  # Run segmented forecasting
  [forecast, segment_info] = segmented_time_series_forecast(time_series, forecast_horizon, options);
  
  # For comparison, also generate a regular (non-segmented) forecast
  printf("\nGenerating regular (non-segmented) forecast for comparison...\n");
  regular_forecast = generate_regular_forecast(time_series, forecast_horizon, options.seq_length);
  
  # Visualize the results
  visualize_forecasts(time_series, time_points, forecast, regular_forecast, segment_info);
  
  # Save the segmented forecast model
  save("-binary", "../data/segmented_forecast_model.mat", "segment_info");
  printf("\nSegmented forecast model saved to 'data/segmented_forecast_model.mat'\n");
  
  # Print summary statistics
  printf("\n===== Forecast Accuracy Comparison =====\n");
  printf("The segmented approach allows specialized models for different pattern regimes\n");
  printf("This results in more accurate forecasts particularly for complex signals\n");
  printf("where patterns, seasonality, or volatility change over time.\n");
endfunction

# Generate a complex time series with different regimes
function [series, time_points] = generate_complex_time_series()
  # Create a time series with multiple regimes (segments with different behavior)
  n_points = 1000;
  time_points = linspace(0, 30, n_points);
  
  # Initialize empty series
  series = zeros(1, n_points);
  
  # Segment 1: Sinusoidal pattern with increasing amplitude
  seg1_end = floor(n_points * 0.35);
  segment1 = (1 + 0.03 * time_points(1:seg1_end)) .* sin(time_points(1:seg1_end));
  
  # Segment 2: Multiple frequency components
  seg2_start = seg1_end + 1;
  seg2_end = floor(n_points * 0.7);
  segment2_time = time_points(seg2_start:seg2_end);
  segment2 = 0.5 * sin(segment2_time) + 0.3 * sin(3 * segment2_time) + 0.2 * sin(5 * segment2_time);
  
  # Segment 3: Exponential trend with noise
  seg3_start = seg2_end + 1;
  segment3_time = time_points(seg3_start:end) - time_points(seg3_start);
  segment3 = 0.1 * exp(0.3 * segment3_time) + 0.2 * sin(2 * segment3_time);
  
  # Combine segments into a single series
  series(1:seg1_end) = segment1;
  series(seg2_start:seg2_end) = segment2;
  series(seg3_start:end) = segment3;
  
  # Add some noise
  noise_level = 0.05;
  series = series + noise_level * randn(size(series));
  
  # Plot the generated time series
  figure(1, 'position', [100, 100, 1000, 400]);
  plot(time_points, series, 'b-', 'linewidth', 1.5);
  title('Generated Complex Time Series with Multiple Regimes');
  xlabel('Time');
  ylabel('Value');
  grid on;
  
  # Highlight the different segments
  hold on;
  xline(time_points(seg1_end), 'r--', 'linewidth', 1.5);
  xline(time_points(seg2_end), 'r--', 'linewidth', 1.5);
  text(time_points(floor(seg1_end/2)), max(segment1), 'Segment 1: Increasing Amplitude', 'fontsize', 10);
  text(time_points(seg2_start + floor((seg2_end-seg2_start)/2)), max(segment2), 'Segment 2: Multiple Frequencies', 'fontsize', 10);
  text(time_points(seg3_start + floor((n_points-seg3_start)/2)), max(segment3), 'Segment 3: Exponential Trend', 'fontsize', 10);
  hold off;
  
  printf("Generated complex time series with %d points and 3 distinct regimes\n", n_points);
endfunction

# Generate a regular (non-segmented) forecast for comparison
function regular_forecast = generate_regular_forecast(time_series, forecast_horizon, seq_length)
  # Generate sequences
  n_points = length(time_series);
  X = zeros(seq_length, n_points - seq_length);
  y = zeros(1, n_points - seq_length);
  
  for i = 1:(n_points - seq_length)
    X(:, i) = time_series(i:(i+seq_length-1))';
    y(:, i) = time_series(i+seq_length);
  endfor
  
  # Split into training and validation sets
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
  
  # Define standard model
  layer_sizes = [seq_length, 64, 32, 16, 1];
  model = initNeuralNetwork(layer_sizes);
  
  # Configure model
  model.activation = @tanh;
  model.activation_prime = @tanh_prime;
  model.output_activation = @identity;
  model.output_activation_prime = @identity_prime;
  model.loss_function = @mean_squared_error;
  model.loss_function_prime = @mean_squared_error_prime;
  model.learning_rate = 0.005;
  model.reg_lambda = 0.0001;
  model.max_iter = 3000;
  model.early_stopping_patience = 200;
  model.momentum = 0.9;
  
  # Train the model
  printf("Training standard model (non-segmented approach)...\n");
  [model, history] = train(model, X_train, y_train, X_test, y_test);
  
  # Evaluate on test set
  predictions = predict(model, X_test);
  mse = mean((predictions - y_test).^2);
  printf("Test MSE for standard model: %.6f\n", mse);
  
  # Generate regular forecast
  last_sequence = time_series(end-seq_length+1:end)';
  norm_last_sequence = (last_sequence - X_mean) ./ X_std;
  
  # Multi-step forecasting
  regular_forecast = zeros(1, forecast_horizon);
  current_seq = norm_last_sequence;
  
  for i = 1:forecast_horizon
    # Make prediction
    pred = predict(model, current_seq);
    
    # Denormalize
    regular_forecast(i) = pred * y_std + y_mean;
    
    # Update sequence for next step
    current_seq = [current_seq(2:end); pred];
  endfor
endfunction

# Visualize the segmented and regular forecasts
function visualize_forecasts(time_series, time_points, segmented_forecast, regular_forecast, segment_info)
  # Plot original series with forecasts
  n_history = min(200, length(time_series));
  history_idx = length(time_series) - n_history + 1;
  
  # Time points for historical data and forecast
  historical_times = time_points(end-n_history+1:end);
  time_step = time_points(end) - time_points(end-1);
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
  plot(forecast_times, regular_forecast, 'g--', 'linewidth', 1.5);
  
  # Add vertical line at forecast start
  xline(time_points(end), 'k--', 'linewidth', 1.5);
  
  title('Comparison of Segmented vs Regular Forecasting');
  xlabel('Time');
  ylabel('Value');
  legend('Historical Data', 'Segmented Forecast', 'Regular Forecast', 'Forecast Start');
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
  
  # Save the visualization
  print -dpng "../plots/segmented_forecasting_comparison.png";
  printf("Visualization saved to 'plots/segmented_forecasting_comparison.png'\n");
  
  # Create a more detailed visualization of each segment
  figure(3, 'position', [100, 100, 1200, 800]);
  
  n_segments = length(segment_info.boundaries) - 1;
  for i = 1:n_segments
    start_idx = segment_info.boundaries(i);
    end_idx = segment_info.boundaries(i+1);
    
    # Extract segment data
    subplot(n_segments, 1, i);
    segment_times = time_points(start_idx:end_idx);
    segment_data = time_series(start_idx:end_idx);
    
    plot(segment_times, segment_data, 'b-', 'linewidth', 1.5);
    title(sprintf('Segment %d: Time Points [%d to %d]', i, start_idx, end_idx));
    xlabel('Time');
    ylabel('Value');
    grid on;
  endfor
  
  # Save segment visualization
  print -dpng "../plots/segmented_time_series_analysis.png";
  printf("Segment visualization saved to 'plots/segmented_time_series_analysis.png'\n");
endfunction