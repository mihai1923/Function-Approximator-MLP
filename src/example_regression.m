# Example script to demonstrate the neural network on a regression problem
# This script trains a neural network to approximate a complex function

# Load necessary packages
pkg load statistics;

# Instead of sourcing the file, we'll use addpath to make the functions available
addpath(".");  # Add current directory to the path

# Generate a dataset for regression
function [X, y] = generate_regression_dataset(n_samples=500, noise=0.15)
  # Sample input points with better distribution
  X = 4 * rand(1, n_samples) - 2;  # Uniformly in range [-2, 2]

  # Generate target values with slightly less noise for better fitting
  y = sin(2 * pi * X) .* X + 0.5 * X.^2 + noise * randn(1, n_samples);

  return;
endfunction

# Generate more data points with less noise
[X, y] = generate_regression_dataset(800, 0.15);

# Split into training and test sets (80/20 split)
n_samples = size(X, 2);
idx = randperm(n_samples);
train_idx = idx(1:round(0.8 * n_samples));
test_idx = idx(round(0.8 * n_samples)+1:end);

X_train = X(:, train_idx);
y_train = y(:, train_idx);
X_test = X(:, test_idx);
y_test = y(:, test_idx);

# Normalize data
X_mean = mean(X_train, 2);
X_std = std(X_train, 0, 2);
X_train = (X_train - X_mean) ./ X_std;
X_test = (X_test - X_mean) ./ X_std;

y_mean = mean(y_train, 2);
y_std = std(y_train, 0, 2);
y_train = (y_train - y_mean) ./ y_std;
y_test = (y_test - y_mean) ./ y_std;

# Define a deeper network architecture for better regression performance
layer_sizes = [1, 100, 50, 25, 1];

# Initialize neural network
model = initNeuralNetwork(layer_sizes);

# Configure model for regression with enhanced parameters
model.activation = @tanh;  # Using tanh for better performance in regression
model.activation_prime = @tanh_prime;
model.output_activation = @identity;  # Identity function for regression output
model.output_activation_prime = @identity_prime;
model.loss_function = @mean_squared_error;
model.loss_function_prime = @mean_squared_error_prime;
model.learning_rate = 0.003;  # Finely tuned learning rate
model.reg_lambda = 0.005;     # Balanced regularization
model.max_iter = 5000;        # More iterations
model.tolerance = 1e-6;       # Stricter convergence criterion
model.clip_threshold = 2.0;   # Appropriate gradient clipping

# Train the model
printf("Training neural network for function approximation...\n");
[model, history] = train(model, X_train, y_train, X_test, y_test);

# Evaluate on test set
predictions = predict(model, X_test);
mse = mean((predictions - y_test).^2);
printf("Test MSE: %.6f\n", mse);

# Plot regression results
figure(1);
hold on;

# Sort for smoother plot line
[X_test_sorted, sort_idx] = sort(X_test);
y_test_sorted = y_test(sort_idx);
pred_sorted = predictions(sort_idx);

# Denormalize data for plotting
X_test_orig = X_test_sorted * X_std + X_mean;
y_test_orig = y_test_sorted * y_std + y_mean;
pred_orig = pred_sorted * y_std + y_mean;

# Generate curve for true function (without noise)
x_curve = linspace(min(X_test_orig), max(X_test_orig), 1000);
y_curve = sin(2 * pi * x_curve) .* x_curve + 0.5 * x_curve.^2;

# Plot data and predictions
plot(x_curve, y_curve, 'k-', 'LineWidth', 2);  # True function
plot(X_test_orig, y_test_orig, 'bo', 'MarkerSize', 6);  # Test data
plot(X_test_orig, pred_orig, 'r-', 'LineWidth', 2);  # Predictions

title("Neural Network Regression");
xlabel("Input x");
ylabel("Output y");
legend("True Function", "Test Data", "Neural Network Prediction");
grid on;
hold off;

# Plot training history
figure(2);
plot(1:length(history.train_loss), history.train_loss, 'b-', 'LineWidth', 2);
hold on;
if !isempty(history.val_loss)
  plot(1:length(history.val_loss), history.val_loss, 'r-', 'LineWidth', 2);
  legend("Training Loss", "Validation Loss");
else
  legend("Training Loss");
endif
title("Training History");
xlabel("Iterations");
ylabel("Loss");
grid on;

# Save figures
print -dpng "../plots/regression_result.png";
print -dpng "../plots/regression_training_history.png";

# Optional: Save model to file
save("-binary", "../data/regression_model.mat", "model");