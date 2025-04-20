function [X, y] = generate_custom_regression_dataset(n_samples=800, noise=0.15, func, data_range=[-2, 2])
  # Generate a dataset for regression using the provided function

  # Sample input points
  X = (data_range(2) - data_range(1)) * rand(1, n_samples) + data_range(1);  # Uniformly in specified range

  # Generate target values using custom function with noise
  y = func(X) + noise * randn(1, n_samples);

  return;
endfunction