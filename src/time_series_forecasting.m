# Time Series Forecasting with Neural Networks
# This script demonstrates how to use neural networks for time series prediction

# Load necessary packages
pkg load statistics;

# Instead of sourcing the file, we'll use addpath to make the functions available
addpath(".");  # Add current directory to the path

# Generate a time series dataset
function [X, y] = generate_time_series(n_points=1000, seq_length=20)
  # Generate a synthetic time series
  x = linspace(0, 30, n_points);
  base_series = sin(x) + 0.2 * sin(5 * x) + 0.1 * randn(1, n_points);

  # Create sequences for input and output
  X = zeros(seq_length, n_points - seq_length);
  y = zeros(1, n_points - seq_length);

  for i = 1:(n_points - seq_length)
    X(:, i) = base_series(i:(i+seq_length-1))';
    y(:, i) = base_series(i+seq_length);
  endfor

  return;
endfunction

# Generate time series data
[X, y] = generate_time_series(1000, 20);

# Plot a segment of the time series
figure(1);
# Fix the vector length mismatch by using proper ranges
sample_seq = [X(:, 1); y(1)];  # This has 21 elements (seq_length + 1)
plot(1:length(sample_seq), sample_seq, 'b-', 'LineWidth', 2);
hold on;
# Only plot as many y values as we have
plot_length = min(180, length(y));
plot((length(X(:,1))+1):(length(X(:,1))+plot_length), y(1:plot_length), 'r-', 'LineWidth', 2);
title("Time Series Data Sample");
xlabel("Time Step");
ylabel("Value");
legend("Input Sequence", "Target Output");
grid on;
print -dpng "../plots/time_series_sample.png";

# Split into training and test sets (80/20 split)
n_samples = size(X, 2);
split_idx = floor(0.8 * n_samples);
X_train = X(:, 1:split_idx);
y_train = y(:, 1:split_idx);
X_test = X(:, (split_idx+1):end);
y_test = y(:, (split_idx+1):end);

# Normalize data
X_mean = mean(X_train(:));
X_std = std(X_train(:));
X_train = (X_train - X_mean) ./ X_std;
X_test = (X_test - X_mean) ./ X_std;

y_mean = mean(y_train);
y_std = std(y_train);
y_train = (y_train - y_mean) ./ y_std;
y_test = (y_test - y_mean) ./ y_std;

# Define network architecture
# Input: sequence_length, Hidden: [32, 16], Output: 1
layer_sizes = [20, 32, 16, 1];

# Initialize neural network
model = initNeuralNetwork(layer_sizes);

# Configure model for time series prediction with improved stability
model.activation = @tanh;
model.activation_prime = @tanh_prime;
model.output_activation = @identity;
model.output_activation_prime = @identity_prime;
model.loss_function = @mean_squared_error;
model.loss_function_prime = @mean_squared_error_prime;
model.learning_rate = 0.005;  # Reduced learning rate for stability
model.reg_lambda = 0.01;      # Increased regularization
model.max_iter = 2000;
model.clip_threshold = 1.0;   # Add gradient clipping

# Train the model
printf("Training neural network for time series prediction...\n");
[model, history] = train(model, X_train, y_train, X_test, y_test);

# Evaluate on test set
predictions = predict(model, X_test);
mse = mean((predictions - y_test).^2);
printf("Test MSE: %.6f\n", mse);

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
print -dpng "../plots/time_series_training_history.png";

# Plot predictions vs actual values
test_length = 200; # Number of points to visualize
start_idx = 1;
end_idx = min(test_length, length(predictions));

# Denormalize the data
y_test_orig = y_test(start_idx:end_idx) * y_std + y_mean;
pred_orig = predictions(start_idx:end_idx) * y_std + y_mean;

figure(3);
hold on;
plot(1:length(y_test_orig), y_test_orig, 'b-', 'LineWidth', 2);
plot(1:length(pred_orig), pred_orig, 'r--', 'LineWidth', 2);
title("Time Series Prediction");
xlabel("Time Step");
ylabel("Value");
legend("Actual", "Predicted");
grid on;
print -dpng "../plots/time_series_prediction.png";

# Multi-step forecasting (predict the next 50 steps)
function [forecast] = multi_step_forecast(model, last_sequence, n_steps, X_mean, X_std, y_mean, y_std)
  # last_sequence: The last known sequence (normalized)
  # n_steps: Number of steps to forecast

  forecast = zeros(1, n_steps);
  current_seq = last_sequence;

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

    forecast(i) = pred;

    # Update the sequence by removing the first element and adding the prediction
    current_seq = [current_seq(2:end); pred];
  endfor

  # Denormalize forecast
  forecast = forecast * y_std + y_mean;
endfunction

# Get the last sequence from the test set
last_known_sequence = X_test(:, end);

# Forecast the next 150 steps
forecast_horizon = 150;
forecast = multi_step_forecast(model, last_known_sequence, forecast_horizon, X_mean, X_std, y_mean, y_std);

# Plot the forecast with some history for context
history_steps = 100;
last_known_values = (X_test(:, end-history_steps:end) * X_std + X_mean)(:);

figure(4);
hold on;
plot(1:length(last_known_values), last_known_values, 'b-', 'LineWidth', 2);
plot(length(last_known_values):(length(last_known_values)+forecast_horizon-1), forecast, 'r--', 'LineWidth', 2);
title("Multi-step Time Series Forecast");
xlabel("Time Step");
ylabel("Value");
legend("Historical Data", "Forecast");
grid on;
print -dpng "../plots/time_series_forecast.png";

# Save the model
save("-binary", "../data/time_series_model.mat", "model");