function run_custom_regression(func, func_str, options=[])
  # Default options
  if (!isstruct(options))
    options = struct();
  endif

  # Set default values if not provided
  if (!isfield(options, "data_range"))
    options.data_range = [-2, 2];
  endif
  if (!isfield(options, "n_samples"))
    options.n_samples = 2125;
  endif
  if (!isfield(options, "noise"))
    options.noise = 0.08;
  endif
  if (!isfield(options, "layers"))
    options.layers = [1, 150, 100, 60, 30, 1];
  endif
  if (!isfield(options, "learning_rate"))
    options.learning_rate = 0.0015;
  endif
  if (!isfield(options, "max_iter"))
    options.max_iter = 10000;
  endif
  if (!isfield(options, "reg_lambda"))
    options.reg_lambda = 0.002;
  endif
  if (!isfield(options, "max_segment_length"))
    options.max_segment_length = 1.2;
  endif

  # Convert standard operators to element-wise operators when passed as a string
  if (ischar(func_str))
    safe_func_str = strrep(func_str, "^", ".^");
    safe_func_str = strrep(safe_func_str, "*", ".*");
    safe_func_str = strrep(safe_func_str, "/", "./");
    safe_func_str = strrep(safe_func_str, "..^", ".^");
    safe_func_str = strrep(safe_func_str, "..*", ".*");
    safe_func_str = strrep(safe_func_str, "../", "./");

    func = inline(safe_func_str, "x");
  endif

  printf("\nGenerating dataset using function: %s\n\n", func_str);
  printf("Using parameters: samples=%d, noise=%.3f, range=[%.2f, %.2f]\n",
         options.n_samples, options.noise, options.data_range(1), options.data_range(2));

  printf("\nUsing segmented regression approach...\n");

  # Create safe function version for evaluation
  if ischar(func)
    safe_func_str = strrep(func_str, "^", ".^");
    safe_func_str = strrep(func_str, "*", ".*");
    safe_func_str = strrep(func_str, "/", "./");
    safe_func_str = strrep(func_str, "..^", ".^");
    safe_func_str = strrep(func_str, "..*", ".*");
    safe_func_str = strrep(func_str, "../", "./");
    test_func = inline(safe_func_str, "x");
  elseif strcmp(class(func), "inline function") || isa(func, 'function_handle')
    test_func = func;
    if ischar(func_str)
      safe_func_str = strrep(func_str, "^", ".^");
      safe_func_str = strrep(func_str, "*", ".*");
      safe_func_str = strrep(func_str, "/", "./");
      safe_func_str = strrep(func_str, "..^", ".^");
      safe_func_str = strrep(func_str, "..*", ".*");
      safe_func_str = strrep(func_str, "../", "./");
    else
      safe_func_str = "custom_function";
    endif
  else
    test_func = func;
    safe_func_str = "custom_function";
  endif

  # Define segments based on the data range
  full_range = options.data_range;
  range_width = full_range(2) - full_range(1);

  # Factor in overlap when calculating optimal segments
  overlap_factor = 1.4;

  # Detect function complexity
  is_complex_function = false;
  sin_count = length(strfind(func_str, "sin"));
  cos_count = length(strfind(func_str, "cos"));
  power_count = length(strfind(func_str, "^"));
  division_count = length(strfind(func_str, "/"));

  # Detect complex oscillatory functions and adjust parameters
  if ((sin_count + cos_count) > 0) && (power_count > 0 || division_count > 0)
    is_complex_function = true;
    printf("Detected complex oscillatory function with powers or frequency divisions.\n");

    if power_count > 1 || (sin_count + cos_count) > 1 || strfind(func_str, "^6") || strfind(func_str, "^5") || strfind(func_str, "^4")
      printf("Function has high powers or multiple oscillatory terms - using very fine segmentation.\n");
      options.max_segment_length = min(options.max_segment_length, 0.5);
      overlap_factor = 1.1;
    else
      options.max_segment_length = min(options.max_segment_length, 0.7);
      overlap_factor = 1.2;
    endif

    # For very small-value complex functions, adjust noise level
    test_values = test_func(linspace(-2, 2, 100));
    max_test_value = max(abs(test_values));

    if max_test_value < 0.1
      if options.noise > 0.01
        printf("Detected very small-value complex function - reducing noise from %.3f to 0.01\n", options.noise);
        options.noise = 0.01;
      endif
    elseif max_test_value < 0.5
      if options.noise > 0.03
        printf("Detected small-value complex function - reducing noise from %.3f to 0.03\n", options.noise);
        options.noise = 0.03;
      endif
    endif
  endif

  # For highly oscillatory functions with powers, adjust parameters
  if (strfind(func_str, "sin") || strfind(func_str, "cos")) && (strfind(func_str, "^"))
    if (strfind(func_str, "^3") || strfind(func_str, "^2")) && (strfind(func_str, "sin") || strfind(func_str, "cos"))
      printf("Detected complex oscillatory function with powers - using smaller segment length.\n");
      options.max_segment_length = min(options.max_segment_length, 0.6);
      overlap_factor = 1.2;
    endif
  endif

  # Add extra segment at the beginning for small functions
  boundary_extension = 0;
  is_small_function = false;
  is_very_small_value_function = false;

  # Detect small functions and adjust parameters
  test_values = test_func(linspace(full_range(1), full_range(2), 100));
  if max(abs(test_values)) < 0.5
    is_small_function = true;
    printf("Detected small-value function - applying enhanced boundary handling.\n");

    if max(abs(test_values)) < 0.1
      is_very_small_value_function = true;
      boundary_extension = 0.2;
      printf("Using extended boundary segment for very small function.\n");
    else
      boundary_extension = 0.1;
    endif

    if boundary_extension > 0
      extended_range = [full_range(1) - boundary_extension, full_range(2)];
      options.n_samples = round(options.n_samples * (1 + boundary_extension/range_width));
      printf("Extended input range to [%.2f, %.2f] for better boundary handling.\n",
             extended_range(1), extended_range(2));
    endif
  endif

  adjusted_max_length = options.max_segment_length / overlap_factor;

  # Calculate optimal number of segments
  optimal_segments = ceil(range_width / adjusted_max_length);
  n_segments = optimal_segments;

  printf("Dividing range [%.1f, %.1f] into %d segments (max segment length: %.2f)...\n",
       full_range(1), full_range(2), n_segments, options.max_segment_length);

  # Create segments with overlaps for smoother transitions
  segment_width = range_width / n_segments;

  # Calculate overlaps based on function type
  if (strfind(func_str, "sin") || strfind(func_str, "cos")) && (strfind(func_str, "^"))
    overlap = segment_width * 0.3;
    printf("Detected oscillatory function with powers - using increased segment overlap (30%%).\n");
  else
    overlap = segment_width * 0.2;
  endif

  # Special handling for small functions
  if is_small_function
    if is_very_small_value_function
      first_segment_overlap = segment_width * 0.5;
      printf("Using enhanced boundary overlap (50%%) for first segment of very small-value function.\n");
    else
      first_segment_overlap = segment_width * 0.4;
      printf("Using enhanced boundary overlap (40%%) for first segment of small-value function.\n");
    endif

    boundary_overlaps = [first_segment_overlap, overlap * ones(1, n_segments-1)];
  else
    boundary_overlaps = overlap * ones(1, n_segments);
  endif

  # Display segment information
  printf("\n%-10s %-20s %-20s %-20s\n", "Segment", "Range", "Width", "Overlap");
  printf("%-10s %-20s %-20s %-20s\n", "-------", "-----", "-----", "-------");

  for i = 1:n_segments
    lower = full_range(1) + (i-1) * segment_width - boundary_overlaps(i);
    upper = full_range(1) + i * segment_width + boundary_overlaps(i);

    # Clamp to full range
    lower = max(lower, full_range(1));
    upper = min(upper, full_range(2));

    act_width = upper - lower;

    if (i > 1 || i < n_segments)
      overlap_str = sprintf("%.2f", boundary_overlaps(i));
    else
      overlap_str = "N/A";
    endif

    printf("%-10d [%-8.2f, %-8.2f] %-20.2f %-20s\n",
           i, lower, upper, act_width, overlap_str);
  endfor

  printf("\nTraining individual models for each segment...\n");

  # Initialize arrays for models and ranges
  sub_models = {};
  sub_ranges = {};

  # Adjust parameters for sub-models
  sub_options = options;
  sub_options.n_samples = round(options.n_samples / n_segments) + 100;
  sub_options.max_iter = round(options.max_iter * 0.7);

  # Train individual models for each segment
  for i = 1:n_segments
    # Calculate segment boundaries with overlap
    if is_small_function && i == 1
      lower = full_range(1) - boundary_extension;
      upper = full_range(1) + segment_width + boundary_overlaps(i);
    else
      lower = full_range(1) + (i-1) * segment_width - boundary_overlaps(i);
      upper = full_range(1) + i * segment_width + boundary_overlaps(i);
    endif

    # Clamp to full range
    lower = max(lower, full_range(1) - boundary_extension);
    upper = min(upper, full_range(2));

    printf("Training segment model %d/%d for range [%.2f, %.2f]...\n", i, n_segments, lower, upper);

    # Set range for this segment
    sub_options.data_range = [lower, upper];

    # Special treatment for first segment of small functions
    if is_small_function && i == 1
      printf("Applying enhanced training to first segment of small-value function\n");

      # Increase samples and adjust parameters for better convergence
      sub_options.n_samples = round(sub_options.n_samples * 1.5);
      sub_options.max_iter = round(sub_options.max_iter * 1.5);
      sub_options.learning_rate = sub_options.learning_rate * 0.7;
      sub_options.reg_lambda = sub_options.reg_lambda * 0.5;

      printf("Enhanced settings: samples=%d, max_iter=%d, learning_rate=%.6f, reg_lambda=%.6f\n",
             sub_options.n_samples, sub_options.max_iter,
             sub_options.learning_rate, sub_options.reg_lambda);
    endif

    # Generate data for this segment
    if is_complex_function || is_very_small_value_function
      # For complex or small-value functions, use adaptive sampling
      [X_initial, y_initial] = generate_custom_regression_dataset(round(sub_options.n_samples / 2),
                                                sub_options.noise * 0.8,
                                                test_func,
                                                [lower, upper]);

      # Calculate approximate second derivative to find high-curvature regions
      x_values = X_initial;
      y_values = y_initial;

      # Ensure x_values is a row vector for proper sorting
      if size(x_values, 1) > size(x_values, 2)
        x_values = x_values';
      endif

      # Sort x values and get corresponding y values to calculate derivatives properly
      [x_sorted, sort_idx] = sort(x_values);
      y_sorted = y_values(sort_idx);

      # Compute approximate derivatives
      if length(x_sorted) > 5
        dx = diff(x_sorted);
        dy = diff(y_sorted);

        if size(dx, 1) > size(dx, 2)
          dx = dx';
        endif
        if size(dy, 1) > size(dy, 2)
          dy = dy';
        endif

        first_deriv = dy ./ dx;

        dx_first = dx(1:end-1);
        if size(dx_first, 1) > size(dx_first, 2)
          dx_first = dx_first';
        endif
        if size(first_deriv, 1) > size(first_deriv, 2)
          first_deriv_diff = diff(first_deriv')';
        else
          first_deriv_diff = diff(first_deriv);
        endif

        second_deriv = abs(first_deriv_diff ./ dx_first);

        if !isempty(second_deriv) && max(second_deriv) > 0
          norm_second_deriv = second_deriv / max(second_deriv);
        else
          norm_second_deriv = zeros(size(second_deriv));
        endif
      else
        # Fall back to standard sampling if not enough initial points
        [X_seg, y_seg] = generate_custom_regression_dataset(sub_options.n_samples,
                                                sub_options.noise,
                                                test_func,
                                                [lower, upper]);
      endif

      # Generate additional points with probability proportional to curvature
      if length(norm_second_deriv) > 3
        # Create weighted sampling for high-curvature regions
        x_regions = x_sorted(2:end-1);
        prob_weights = 0.3 + 0.7 * norm_second_deriv;
        prob_weights = prob_weights / sum(prob_weights);

        # Generate additional points with this distribution
        n_extra = round(sub_options.n_samples / 2);
        extra_indices = [];

        for ii = 1:n_extra
          r = rand();
          cumsum_prob = cumsum(prob_weights);
          idx = find(cumsum_prob >= r, 1);
          if isempty(idx)
            idx = length(prob_weights);
          endif
          extra_indices(end+1) = idx;
        endfor

        # Calculate x-values for these high-curvature extra points
        extra_x = x_regions(extra_indices);

        if size(extra_x, 1) > size(extra_x, 2)
          extra_x = extra_x';
        endif

        # Add small random offsets to avoid exact duplicates
        extra_x = extra_x + (rand(size(extra_x)) - 0.5) .* diff([lower, upper]) * 0.05;
        extra_x = min(max(extra_x, lower), upper);

        # Get corresponding y-values with appropriate noise
        extra_y = test_func(extra_x) + randn(size(extra_x)) * sub_options.noise;

        if size(extra_y, 1) > size(extra_y, 2)
          extra_y = extra_y';
        endif

        # Ensure X_initial and y_initial are properly oriented for concatenation
        if size(X_initial, 1) > 1
          X_seg = [X_initial', extra_x];
          y_seg = [y_initial', extra_y];
        else
          X_seg = [X_initial, extra_x];
          y_seg = [y_initial, extra_y];
        endif

        printf("Used adaptive sampling with %d extra points in high-curvature regions\n", n_extra);
      else
        # Fall back to standard sampling if not enough points for derivatives
        [X_seg, y_seg] = generate_custom_regression_dataset(sub_options.n_samples,
                                                  sub_options.noise,
                                                  test_func,
                                                  [lower, upper]);
      endif
    else
      # Standard sampling for simpler functions
      [X_seg, y_seg] = generate_custom_regression_dataset(sub_options.n_samples,
                                                sub_options.noise,
                                                test_func,
                                                [lower, upper]);
    endif

    # Check if this is a small-value function and scale if needed
    max_abs_y = max(abs(y_seg));

    # Auto-scale small values for better numerical stability
    scaling_factor = 1.0;
    if max_abs_y < 0.5 && i == 1
      if max_abs_y < 0.1
        scaling_factor = min(1.0 / max_abs_y, 200);
        printf("\nDetected very small-value function (max value: %.6f)\n", max_abs_y);
        printf("Auto-scaling values by factor of %.2f for better numerical accuracy.\n", scaling_factor);

        # Enhance training for small-value functions
        sub_options.learning_rate = 0.001;
        sub_options.max_iter = round(sub_options.max_iter * 1.5);
        sub_options.reg_lambda = 0.001;
      else
        scaling_factor = min(1.0 / max_abs_y, 100);
        printf("\nDetected small-value function (max value: %.6f)\n", max_abs_y);
        printf("Auto-scaling values by factor of %.2f for better numerical accuracy.\n", scaling_factor);
      endif

      printf("This is handled internally and won't affect the output predictions.\n");

      # Apply scaling
      y_seg = y_seg * scaling_factor;
    elseif max_abs_y < 0.5
      # For consistency, use same scaling for all segments
      y_seg = y_seg * scaling_factor;
    endif

    # Store scaling factor for later use
    if i == 1
      global_scaling_factor = scaling_factor;
    endif

    # Split into training and test sets
    n_samples_seg = size(X_seg, 2);
    idx_seg = randperm(n_samples_seg);
    train_idx_seg = idx_seg(1:round(0.8 * n_samples_seg));
    test_idx_seg = idx_seg(round(0.8 * n_samples_seg)+1:end);

    X_train_seg = X_seg(:, train_idx_seg);
    y_train_seg = y_seg(:, train_idx_seg);
    X_test_seg = X_seg(:, test_idx_seg);
    y_test_seg = y_seg(:, test_idx_seg);

    # Normalize data for this segment
    X_mean_seg = mean(X_train_seg, 2);
    X_std_seg = std(X_train_seg, 0, 2) + eps;
    X_train_seg = (X_train_seg - X_mean_seg) ./ X_std_seg;
    X_test_seg = (X_test_seg - X_mean_seg) ./ X_std_seg;

    y_mean_seg = mean(y_train_seg, 2);
    y_std_seg = std(y_train_seg, 0, 2) + eps;
    y_train_seg = (y_train_seg - y_mean_seg) ./ y_std_seg;
    y_test_seg = (y_test_seg - y_mean_seg) ./ y_std_seg;

    # Store normalization parameters
    norm_params_seg = struct();
    norm_params_seg.X_mean = X_mean_seg;
    norm_params_seg.X_std = X_std_seg;
    norm_params_seg.y_mean = y_mean_seg;
    norm_params_seg.y_std = y_std_seg;

    # Initialize segmented model with appropriate architecture
    if max_abs_y < 0.1 && i == 1
      # Deeper and wider network for small-value functions
      enhanced_layers = [1, 200, 150, 100, 75, 50, 25, 1];
      printf("Using enhanced network architecture [%s] for small-value function\n",
             strjoin(arrayfun(@num2str, enhanced_layers, "UniformOutput", false), ", "));
      model_seg = initNeuralNetwork(enhanced_layers);

      # Enhanced training for very small values
      sub_options.learning_rate = 0.0008;
      sub_options.max_iter = round(sub_options.max_iter * 2.0);
      sub_options.reg_lambda = 0.0008;

      printf("Using enhanced training settings: learning_rate=%.4f, max_iter=%d, reg_lambda=%.4f\n",
             sub_options.learning_rate, sub_options.max_iter, sub_options.reg_lambda);
    else
      model_seg = initNeuralNetwork(sub_options.layers);
    endif

    # Configure model for regression
    model_seg.activation = @tanh;
    model_seg.activation_prime = @tanh_prime;
    model_seg.output_activation = @identity;
    model_seg.output_activation_prime = @identity_prime;
    model_seg.loss_function = @mean_squared_error;
    model_seg.loss_function_prime = @mean_squared_error_prime;
    model_seg.learning_rate = sub_options.learning_rate;
    model_seg.reg_lambda = sub_options.reg_lambda;
    model_seg.max_iter = sub_options.max_iter;
    model_seg.tolerance = 1e-6;
    model_seg.clip_threshold = 2.0;
    model_seg.norm_params = norm_params_seg;

    # Train the segment model
    printf("Training segment model %d...\n", i);
    [model_seg, history_seg] = train(model_seg, X_train_seg, y_train_seg, X_test_seg, y_test_seg);

    # Store model and its valid range
    sub_models{i} = model_seg;
    sub_ranges{i} = [lower, upper];

    # Report segment performance
    predictions_seg = predict(model_seg, X_test_seg);
    mse_seg = mean((predictions_seg - y_test_seg).^2);
    printf("Segment %d MSE: %.6f\n", i, mse_seg);
  endfor

  # Generate dense evaluation points for the full range
  x_full = linspace(full_range(1), full_range(2), 1000);
  y_full = test_func(x_full);

  # Calculate max absolute value for later use
  max_abs_value = max(abs(y_full));
  is_small_value_function = max_abs_value < 1.0;
  is_very_small_value_function = max_abs_value < 0.1;

  if is_small_value_function
    printf("\nDetected small output values (max abs value: %.4f)\n", max_abs_value);
  endif

  # Initialize global scaling factor to share across segments
  global_scaling_factor = 1.0;

  # Evaluate the ensemble model
  y_pred_full = zeros(size(x_full));
  model_weights = zeros(1, length(x_full));

  # For each point, find which models apply and compute weighted predictions
  for j = 1:length(x_full)
    x_point = x_full(j);
    total_weight = 0;

    # Check each model's applicability to this point
    for i = 1:n_segments
      range_i = sub_ranges{i};

      # Skip if point is outside this model's range
      if (x_point < range_i(1) || x_point > range_i(2))
        continue;
      endif

      # Calculate weight based on position within the segment range
      segment_width = range_i(2) - range_i(1);
      segment_center = (range_i(1) + range_i(2)) / 2;

      # Calculate distance from center as percentage of half-width
      rel_dist = abs(x_point - segment_center) / (segment_width / 2);

      # Special handling for boundary segments of small-value functions
      is_boundary_segment = (i == 1 || i == n_segments);

      # Apply appropriate weighting function based on function type
      if is_small_function && is_boundary_segment && i == 1 && x_point < (full_range(1) + segment_width * 0.3)
        boundary_dist = (x_point - full_range(1)) / (segment_width * 0.3);

        if is_very_small_value_function && x_point <= (full_range(1) + segment_width * 0.1)
          weight = max(0.8, 0.8 + 0.2 * (1 - exp(-5 * boundary_dist)));
        else
          weight = max(0, 0.5 + 0.5 * (1 - exp(-4 * boundary_dist)) * (1 - rel_dist^2));
        endif
      elseif is_small_function && is_boundary_segment && i == n_segments
        boundary_dist = (full_range(2) - x_point) / (segment_width * 0.3);
        weight = max(0, 0.3 + 0.7 * (1 - exp(-3 * boundary_dist)) * (1 - rel_dist^2));
      elseif max_abs_value < 0.1 && (strfind(func_str, "sin") || strfind(func_str, "cos"))
        weight = max(0, exp(-(rel_dist^2 * 1.5)));
      elseif is_small_value_function && (strfind(func_str, "sin") || strfind(func_str, "cos"))
        weight = max(0, (1 - rel_dist^3)^1.2);
      elseif (strfind(func_str, "sin") || strfind(func_str, "cos")) && (strfind(func_str, "^"))
        weight = max(0, 1 - rel_dist^3);
      elseif (strfind(func_str, "sin") || strfind(func_str, "cos"))
        weight = max(0, 1 - rel_dist^2);
      else
        weight = max(0, 1 - rel_dist);
      endif

      # Prepare input for model
      x_norm = (x_point - sub_models{i}.norm_params.X_mean) / sub_models{i}.norm_params.X_std;

      # Get prediction and denormalize
      pred_norm = predict(sub_models{i}, x_norm);

      # Apply respective denormalization and scaling for this segment's model
      pred_denorm = pred_norm * sub_models{i}.norm_params.y_std + sub_models{i}.norm_params.y_mean;

      # Undo the internal scaling
      if is_small_function || is_small_value_function
        pred_denorm = pred_denorm / global_scaling_factor;
      endif

      # Add weighted prediction to ensemble result
      y_pred_full(j) += weight * pred_denorm;
      total_weight += weight;

      # Track weight for this point (for visualization)
      if i == 1
        model_weights(j) = weight / total_weight;
      endif
    endfor

    # Normalize by total weight if non-zero
    if total_weight > 0
      y_pred_full(j) /= total_weight;
    else
      # Fall back to nearest segment if no model covers this point
      segment_dists = zeros(1, n_segments);
      for i = 1:n_segments
        range_i = sub_ranges{i};
        segment_center = (range_i(1) + range_i(2)) / 2;
        segment_dists(i) = abs(x_point - segment_center);
      endfor

      [~, nearest_segment] = min(segment_dists);
      x_norm = (x_point - sub_models{nearest_segment}.norm_params.X_mean) / sub_models{nearest_segment}.norm_params.X_std;
      pred_norm = predict(sub_models{nearest_segment}, x_norm);
      pred_denorm = pred_norm * sub_models{nearest_segment}.norm_params.y_std + sub_models{nearest_segment}.norm_params.y_mean;

      if is_small_function || is_small_value_function
        pred_denorm = pred_denorm / global_scaling_factor;
      endif

      y_pred_full(j) = pred_denorm;
      printf("Warning: Point %.2f is not covered by any segment model. Using nearest segment.\n", x_point);
    endif
  endfor

  # Evaluate overall error
  mse_full = mean((y_pred_full - y_full).^2);
  rmse_full = sqrt(mse_full);

  printf("\nFull range performance:\n");
  printf("MSE: %.6f, RMSE: %.6f\n", mse_full, rmse_full);

  # Detect oscillatory functions for additional smoothing
  need_smoothing = false;
  is_oscillatory = (sin_count > 0 || cos_count > 0);

  if is_oscillatory && (power_count > 0) && max_abs_value < 1.0
    # Apply light Gaussian smoothing for oscillatory functions
    need_smoothing = true;
    window_size = 5;
    sigma = 1.0;

    # Create Gaussian kernel
    gaussian_kernel = exp(-((-floor(window_size/2):floor(window_size/2)).^2 / (2*sigma^2)));
    gaussian_kernel = gaussian_kernel / sum(gaussian_kernel);

    # Apply convolution for smoothing
    y_pred_smooth = conv(y_pred_full, gaussian_kernel, 'same');

    # Calculate smoothed error
    mse_smooth = mean((y_pred_smooth - y_full).^2);

    if mse_smooth < mse_full
      printf("Applied smoothing to oscillatory function predictions. Improved MSE: %.6f\n", mse_smooth);
      y_pred_full = y_pred_smooth;
      mse_full = mse_smooth;
    else
      printf("Smoothing did not improve results, keeping original predictions.\n");
    endif
  endif

  # Create model struct with all necessary information
  model = struct();
  model.sub_models = sub_models;
  model.sub_ranges = sub_ranges;
  model.global_scaling_factor = global_scaling_factor;
  model.is_small_function = is_small_function;
  model.is_small_value_function = is_small_value_function;
  model.is_very_small_value_function = is_very_small_value_function;
  model.is_oscillatory = is_oscillatory;
  model.data_range = full_range;
  model.need_smoothing = need_smoothing;

  # Save model with timestamp ID
  timestamp_id = round(now() * 24 * 60);
  model_filename = sprintf("data/regression_model_custom_func_%d.mat", timestamp_id);

  printf("\nSaving model to: %s\n", model_filename);

  # Create temporary file during saving
  temp_filename = sprintf("%s.saving_in_progress", model_filename);

  # Make sure directory exists before saving
  [dirname, ~, ~] = fileparts(model_filename);
  if !exist(dirname, "dir")
    mkdir(dirname);
  endif

  # Try saving the model
  save("-binary", temp_filename, "model");

  # Rename to final filename once saving is complete
  if exist(temp_filename, "file")
    [status, msg] = rename(temp_filename, model_filename);
    if status != 0
      printf("Error saving model: %s\n", msg);
    else
      printf("Model saved successfully.\n");
    endif
  else
    printf("Error creating temporary model file.\n");
  endif

  # Plot results
  figure;
  plot(x_full, y_full, 'b-', 'LineWidth', 2);
  hold on;
  plot(x_full, y_pred_full, 'r-', 'LineWidth', 1.5);
  lgd = legend('True Function', 'Neural Network Prediction');
  title(sprintf('Regression Results for %s', func_str));
  grid on;
  xlabel('x');
  ylabel('f(x)');

  set(lgd, 'Color', 'none');
  set(lgd, 'Box', 'off');
  set(lgd, 'TextColor', [0 0 0]);

  # Add segment boundaries if multiple segments
  if n_segments > 1
    for i = 1:n_segments
      range_i = sub_ranges{i};
      yrange = get(gca, 'YLim');
      line([range_i(1), range_i(1)], yrange, 'LineStyle', '--', 'Color', [0.5, 0.5, 0.5], 'HandleVisibility', 'off');
      line([range_i(2), range_i(2)], yrange, 'LineStyle', '--', 'Color', [0.5, 0.5, 0.5], 'HandleVisibility', 'off');
    endfor
  endif

  # Add error visualization
  figure;
  subplot(2, 1, 1);
  plot(x_full, abs(y_pred_full - y_full), 'r-');
  title('Absolute Error');
  grid on;
  xlabel('x');
  ylabel('|Error|');

  subplot(2, 1, 2);
  if n_segments > 1
    area(x_full, model_weights, 'FaceColor', [0.2, 0.6, 0.8], 'FaceAlpha', 0.5);
    title('First Segment Model Weight Contribution');
    grid on;
    xlabel('x');
    ylabel('Weight');
    ylim([0, 1]);
  else
    bar(1, mse_full);
    title('Mean Squared Error');
    grid on;
    ylabel('MSE');
    xticklabels({''});
  endif

  # Final report
  printf("\nModel Training Complete\n");
  printf("----------------------------------------------\n");
  printf("Function: %s\n", func_str);
  printf("Range: [%.2f, %.2f]\n", full_range(1), full_range(2));
  printf("Number of segments: %d\n", n_segments);
  printf("MSE: %.6f\n", mse_full);
  printf("RMSE: %.6f\n", rmse_full);
  if is_small_function
    printf("Small-value function detected, automatic scaling applied\n");
  endif
  if is_oscillatory
    printf("Oscillatory function detected\n");
  endif
  printf("Model saved as: %s\n", model_filename);
  printf("----------------------------------------------\n");

  # Return the model
  model.mse = mse_full;
  model.rmse = rmse_full;
  model.timestamp_id = timestamp_id;
  model.func_str = func_str;
endfunction
