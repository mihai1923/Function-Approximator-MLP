function run_custom_time_series(ts_func, ts_func_str, precision_mode="super")
  # Run time series forecast with custom function
  
  printf("\nGenerating time series using function: %s\n\n", ts_func_str);
  
  # Force super precision mode
  precision_mode = "super";
  
  ultra_precision = strcmp(precision_mode, "ultra");
  super_precision = strcmp(precision_mode, "super");
  high_precision = strcmp(precision_mode, "high") || ultra_precision || super_precision;
  
  printf("*** SUPER-PRECISION MODE ENABLED (Default) ***\n");

  # Convert string functions to element-wise operations
  if ischar(ts_func) || strcmp(class(ts_func), "inline function")
    # Clean spaces around operators
    clean_ts_func_str = strrep(ts_func_str, " ^ ", "^");
    clean_ts_func_str = strrep(clean_ts_func_str, " ^", "^");
    clean_ts_func_str = strrep(clean_ts_func_str, "^ ", "^");
    clean_ts_func_str = strrep(clean_ts_func_str, " * ", "*");
    clean_ts_func_str = strrep(clean_ts_func_str, " *", "*");
    clean_ts_func_str = strrep(clean_ts_func_str, "* ", "*");
    clean_ts_func_str = strrep(clean_ts_func_str, " / ", "/");
    clean_ts_func_str = strrep(clean_ts_func_str, " /", "/");
    clean_ts_func_str = strrep(clean_ts_func_str, "/ ", "/");
    
    # Convert to element-wise operations
    safe_ts_func_str = strrep(clean_ts_func_str, "^", ".^");
    safe_ts_func_str = strrep(safe_ts_func_str, "*", ".*");
    safe_ts_func_str = strrep(safe_ts_func_str, "/", "./");
    
    # Fix double dots
    safe_ts_func_str = strrep(safe_ts_func_str, "..^", ".^");
    safe_ts_func_str = strrep(safe_ts_func_str, "..*", ".*");
    safe_ts_func_str = strrep(safe_ts_func_str, "../", "./");

    printf("Input function: %s\n", ts_func_str);
    printf("Clean function: %s\n", clean_ts_func_str);
    printf("Element-wise operations: %s\n", safe_ts_func_str);
    safe_ts_func = inline(safe_ts_func_str, "x");
  endif

  # Generate dataset
  printf("Generating high-precision dataset for accurate modeling...\n");
  n_points = 2000;
  seq_length = 30;
  
  if ischar(ts_func) || strcmp(class(ts_func), "inline function")
    [X, y] = generate_custom_time_series(n_points, seq_length, safe_ts_func);
  else
    [X, y] = generate_custom_time_series(n_points, seq_length, ts_func);
  endif

  # Plot sample time series
  figure(1);
  sample_seq = [X(:, 1); y(1)];
  plot(1:length(sample_seq), sample_seq, 'b-', 'LineWidth', 2);
  hold on;
  plot_length = min(180, length(y));
  plot((length(X(:,1))+1):(length(X(:,1))+plot_length), y(1:plot_length), 'r-', 'LineWidth', 2);
  title(sprintf("Time Series Data Sample: %s", ts_func_str));
  xlabel("Time Step");
  ylabel("Value");
  legend("Input Sequence", "Target Output");
  grid on;

  # Split train/test
  n_samples = size(X, 2);
  split_idx = floor(0.8 * n_samples);
  X_train = X(:, 1:split_idx);
  y_train = y(:, 1:split_idx);
  X_test = X(:, (split_idx+1):end);
  y_test = y(:, (split_idx+1):end);

  # Normalize data
  eps = 1e-8;
  X_mean = mean(X_train(:));
  X_std = std(X_train(:)) + eps;
  X_train = (X_train - X_mean) ./ X_std;
  X_test = (X_test - X_mean) ./ X_std;

  y_mean = mean(y_train);
  y_std = std(y_train) + eps;
  y_train = (y_train - y_mean) ./ y_std;
  y_test = (y_test - y_mean) ./ y_std;

  # Detect function characteristics for architecture selection
  has_trig = false;
  has_product = false;
  has_polynomial = false;
  
  if ischar(ts_func_str)
    if any(strfind(ts_func_str, "sin")) || any(strfind(ts_func_str, "cos"))
      has_trig = true;
    endif
    
    if any(strfind(ts_func_str, "*x")) || any(strfind(ts_func_str, "x*"))
      has_product = true;
    endif
    
    if any(strfind(ts_func_str, "^"))
      has_polynomial = true;
    endif
  endif

  # Create architecture based on function type
  if has_trig && (has_product || has_polynomial)
    layer_sizes = [seq_length, 256, 192, 128, 96, 64, 32, 1];
    use_high_precision = true;
  elseif has_trig
    layer_sizes = [seq_length, 200, 150, 100, 50, 1];
    use_high_precision = true;
  elseif has_polynomial
    layer_sizes = [seq_length, 150, 100, 75, 50, 25, 1];
    use_high_precision = false;
  else
    layer_sizes = [seq_length, 128, 64, 32, 1];
    use_high_precision = false;
  endif

  # Initialize neural network
  model = initNeuralNetwork(layer_sizes);

  # Configure model
  model.activation = @tanh;
  model.activation_prime = @tanh_prime;
  model.output_activation = @identity;
  model.output_activation_prime = @identity_prime;
  model.loss_function = @mean_squared_error;
  model.loss_function_prime = @mean_squared_error_prime;
  model.learning_rate = 0.0005;
  
  # Set regularization based on function type
  if has_trig && (has_product || has_polynomial)
    model.reg_lambda = 0.0001;
  else
    model.reg_lambda = 0.001;
  endif
  
  # Set training parameters based on precision mode
  if ultra_precision
    # Ultra-high precision settings
    layer_sizes = [seq_length, 512, 384, 256, 192, 128, 96, 64, 32, 1];
    
    # Recreate model with ultra-deep architecture
    model = initNeuralNetwork(layer_sizes);
    model.activation = @tanh;
    model.activation_prime = @tanh_prime;
    model.output_activation = @identity;
    model.output_activation_prime = @identity_prime;
    model.loss_function = @mean_squared_error;
    model.loss_function_prime = @mean_squared_error_prime;
    
    model.max_iter = 50000;
    model.learning_rate = 0.0001;
    model.reg_lambda = 0.00005;
    model.early_stopping_patience = 1000;
    model.learning_rate_decay = 0.9998;
    model.momentum = 0.98;
    model.clip_threshold = 0.1;
    model.batch_size = 32;
    model.adaptive_lr = true;
    
    use_high_precision = true;
  elseif super_precision
    # Super-precision settings
    layer_sizes = [seq_length, 1024, 768, 512, 384, 256, 192, 128, 96, 64, 32, 1];
    
    # Recreate model with super-deep architecture
    model = initNeuralNetwork(layer_sizes);
    model.activation = @tanh;
    model.activation_prime = @tanh_prime;
    model.output_activation = @identity;
    model.output_activation_prime = @identity_prime;
    model.loss_function = @mean_squared_error;
    model.loss_function_prime = @mean_squared_error_prime;
    
    model.max_iter = 100000;
    model.learning_rate = 0.00005;
    model.reg_lambda = 0.00001;
    model.early_stopping_patience = 2000;
    model.learning_rate_decay = 0.9999;
    model.momentum = 0.99;
    model.clip_threshold = 0.05;
    model.batch_size = 16;
    model.adaptive_lr = true;
    
    use_high_precision = true;
  elseif use_high_precision
    model.max_iter = 15000;
  else
    model.max_iter = 5000;
  endif
  
  # Common parameters
  model.clip_threshold = 0.25;
  
  if use_high_precision
    model.early_stopping_patience = 200;
  else
    model.early_stopping_patience = 50;
  endif
  
  model.learning_rate_decay = 0.9995;
  model.momentum = 0.95;
  
  # Print configuration
  printf("\nNeural Network Configuration:\n");
  printf("- Architecture: ");
  printf("%d-", layer_sizes(1:end-1));
  printf("%d\n", layer_sizes(end));
  printf("- Learning rate: %.6f\n", model.learning_rate);
  printf("- Regularization: %.6f\n", model.reg_lambda);
  printf("- Maximum iterations: %d\n", model.max_iter);
  
  printf("\nStarting training...\n");
  
  # Train model
  tic();
  [model, history] = train_enhanced(model, X_train, y_train, X_test, y_test);
  training_time = toc();
  
  printf("Training completed in %.2f minutes.\n", training_time/60);

  # Evaluate on test set
  predictions = predict(model, X_test);
  mse = mean((predictions - y_test).^2);
  printf("Test MSE: %.6f\n", mse);
  
  # Calculate metrics
  mae = mean(abs(predictions - y_test));
  rmse = sqrt(mse);
  r_squared = 1 - sum((y_test - predictions).^2) / sum((y_test - mean(y_test)).^2);
  
  printf("Mean Absolute Error: %.6f\n", mae);
  printf("Root Mean Squared Error: %.6f\n", rmse);
  printf("R-squared: %.6f\n", r_squared);
  
  # Calculate relative error
  mean_y_test = mean(abs(y_test));
  if mean_y_test > 0
    relative_error = mae / mean_y_test * 100;
    printf("Relative Error: %.2f%%\n", relative_error);
  endif

  # Plot training history
  figure(2);
  plot(1:length(history.train_loss), history.train_loss, 'b-', 'LineWidth', 2);
  hold on;
  plot(1:length(history.val_loss), history.val_loss, 'r-', 'LineWidth', 2);
  title("Training History");
  xlabel("Iterations");
  ylabel("Loss");
  legend("Training Loss", "Validation Loss");
  grid on;

  # Plot predictions vs actual values
  test_length = 200;
  start_idx = 1;
  end_idx = min(test_length, length(predictions));

  # Denormalize the data
  y_test_orig = y_test(start_idx:end_idx) * y_std + y_mean;
  pred_orig = predictions(start_idx:end_idx) * y_std + y_mean;

  figure(3);
  hold on;
  plot(1:length(y_test_orig), y_test_orig, 'b-', 'LineWidth', 2);
  plot(1:length(pred_orig), pred_orig, 'r--', 'LineWidth', 2);
  title(sprintf("Time Series Prediction: %s", ts_func_str));
  xlabel("Time Step");
  ylabel("Value");
  legend("Actual", "Predicted");
  grid on;
  
  # Create comparison visualization
  figure(5, 'position', [200, 200, 1000, 800]);
  
  # Actual time series
  subplot(2, 2, 1);
  hold on;
  plot(1:length(y_test_orig), y_test_orig, 'b-', 'LineWidth', 2.5);
  title("Actual Time Series", 'FontSize', 12, 'FontWeight', 'bold');
  xlabel("Time Step", 'FontSize', 10);
  ylabel("Value", 'FontSize', 10);
  grid on;
  
  # Predicted time series
  subplot(2, 2, 2);
  hold on;
  plot(1:length(pred_orig), pred_orig, 'r-', 'LineWidth', 2.5);
  title("Neural Network Prediction", 'FontSize', 12, 'FontWeight', 'bold');
  xlabel("Time Step", 'FontSize', 10);
  ylabel("Value", 'FontSize', 10);
  grid on;
  
  # Combined view with error
  subplot(2, 2, [3, 4]);
  hold on;
  plot(1:length(y_test_orig), y_test_orig, 'b-', 'LineWidth', 2, 'DisplayName', 'Actual');
  plot(1:length(pred_orig), pred_orig, 'r--', 'LineWidth', 2, 'DisplayName', 'Predicted');
  
  # Error curve
  error_curve = pred_orig - y_test_orig;
  plot(1:length(error_curve), error_curve, 'g-', 'LineWidth', 1.5, 'DisplayName', 'Error');
  
  title(sprintf("Time Series Comparison: %s", ts_func_str), 'FontSize', 14, 'FontWeight', 'bold');
  xlabel("Time Step", 'FontSize', 12);
  ylabel("Value", 'FontSize', 12);
  legend('Location', 'northeast');
  grid on;
  
  # Add error metrics text
  stats_text = sprintf('MSE: %.6f\nMAE: %.6f\nRMSE: %.6f\nR²: %.4f\nRelative Error: %.2f%%', 
                      mse, mae, rmse, r_squared, relative_error);
  
  # Position text
  x_pos = length(y_test_orig) * 0.05;
  y_range = max(y_test_orig) - min(y_test_orig);
  y_pos = max(y_test_orig) - 0.15 * y_range;
  
  # Create background for text
  h_rect = patch([x_pos, x_pos+length(y_test_orig)*0.4, x_pos+length(y_test_orig)*0.4, x_pos], 
               [y_pos-0.15*y_range, y_pos-0.15*y_range, y_pos+0.15*y_range, y_pos+0.15*y_range],
               'w', 'FaceAlpha', 0.7, 'EdgeColor', 'k');
  
  # Add text
  text(x_pos+5, y_pos, stats_text, 'FontSize', 10);
  
  # Create safe filename
  if (length(ts_func_str) > 20)
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
  
  # Save comparison figure
  comparison_plot_path = ["../plots/time_series_comparison_", safe_name, ".png"];
  print(figure(5), "-dpng", "-r300", comparison_plot_path);
  printf("Time series comparison plot saved to %s\n", comparison_plot_path);

  # Multi-step forecast
  last_known_sequence = X_test(:, end);
  forecast_horizon = 150;
  forecast = multi_step_forecast(model, last_known_sequence, forecast_horizon, X_mean, X_std, y_mean, y_std);

  # Plot forecast
  history_steps = 100;
  last_known_values = (X_test(:, end-history_steps:end) * X_std + X_mean)(:);

  figure(4);
  hold on;
  plot(1:length(last_known_values), last_known_values, 'b-', 'LineWidth', 2);
  plot(length(last_known_values):(length(last_known_values)+forecast_horizon-1), forecast, 'r--', 'LineWidth', 2);
  title(sprintf("Multi-step Time Series Forecast: %s", ts_func_str));
  xlabel("Time Step");
  ylabel("Value");
  legend("Historical Data", "Forecast");
  grid on;

  # Create safe filename
  if (length(ts_func_str) > 20)
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

  # Make plots directory
  if !exist("../plots", "dir")
    mkdir("../plots");
  endif

  # Save plots
  sample_plot_path = ["../plots/time_series_sample_", safe_name, ".png"];
  prediction_plot_path = ["../plots/time_series_prediction_", safe_name, ".png"];
  forecast_plot_path = ["../plots/time_series_forecast_", safe_name, ".png"];
  history_plot_path = ["../plots/time_series_training_history_", safe_name, ".png"];

  print(figure(1), "-dpng", sample_plot_path);
  print(figure(2), "-dpng", history_plot_path);
  print(figure(3), "-dpng", prediction_plot_path);
  print(figure(4), "-dpng", forecast_plot_path);

  # Save model
  if !exist("../data", "dir")
    mkdir("../data");
  endif

  model_path = ["../data/time_series_model_", safe_name, ".mat"];
  save("-binary", model_path, "model");
endfunction
