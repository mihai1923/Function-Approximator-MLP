function [forecast, segment_info] = segmented_time_series_forecast(time_series, forecast_horizon, options)
  # Segmented Time Series Forecasting
  # Segments a time series into regions with similar patterns and trains
  # specialized models for each segment, improving forecast accuracy.
  #
  # Inputs:
  #   time_series      - The historical time series data (vector)
  #   forecast_horizon - Number of steps to forecast into the future
  #   options          - Structure with configuration options:
  #     .segment_detection_method - Method to detect segments ("adaptive", "equal", "pattern")
  #     .n_segments             - Target number of segments (may be adjusted)
  #     .seq_length             - Sequence length for the forecasting models
  #     .min_segment_size       - Minimum size for a valid segment
  #
  # Outputs:
  #   forecast     - Vector of forecasted values
  #   segment_info - Structure with segment information and models
  
  printf("Starting segmented time series forecasting...\n");
  
  # Set default options if not provided
  if nargin < 3
    options = struct();
  endif
  
  if !isfield(options, "segment_detection_method")
    options.segment_detection_method = "adaptive";
  endif
  
  if !isfield(options, "n_segments")
    options.n_segments = 3;
  endif
  
  if !isfield(options, "seq_length")
    options.seq_length = 40;
  endif
  
  if !isfield(options, "min_segment_size")
    options.min_segment_size = 120;
  endif
  
  # Step 1: Segment the time series
  printf("Detecting time series segments...\n");
  segment_boundaries = detect_segments(time_series, options);
  
  # Store segment info
  segment_info = struct();
  segment_info.boundaries = segment_boundaries;
  segment_info.models = cell(1, length(segment_boundaries) - 1);
  segment_info.norm_params = cell(1, length(segment_boundaries) - 1);
  
  # Step 2: Train specialized models for each segment
  printf("Training specialized models for %d detected segments...\n", length(segment_boundaries) - 1);
  
  for i = 1:(length(segment_boundaries) - 1)
    # Extract segment data
    start_idx = segment_boundaries(i);
    end_idx = segment_boundaries(i+1);
    segment_data = time_series(start_idx:end_idx);
    
    printf("Segment %d: Training with %d data points [%d to %d]\n", i, length(segment_data), start_idx, end_idx);
    
    # Skip if segment is too small for training
    if length(segment_data) < options.min_segment_size
      printf("Segment %d is too small for effective training, using global model instead\n", i);
      # For small segments, we'll use a model trained on the full dataset later
      continue;
    endif
    
    # Prepare data for time series forecasting
    [X_train, y_train, X_test, y_test, norm_params] = prepare_segment_data(segment_data, options.seq_length);
    
    # If we have enough data for train/test split
    if !isempty(X_test)
      # Train model for this segment
      segment_model = train_segment_model(X_train, y_train, X_test, y_test, i);
      
      # Store model and normalization parameters
      segment_info.models{i} = segment_model;
      segment_info.norm_params{i} = norm_params;
    else
      printf("Warning: Not enough data in segment %d for train/test split\n", i);
    endif
  endfor
  
  # Step 3: Train a global model for small segments or fallback
  printf("Training global model as fallback...\n");
  [X_global, y_global, X_global_test, y_global_test, global_norm] = prepare_segment_data(time_series, options.seq_length);
  global_model = train_segment_model(X_global, y_global, X_global_test, y_global_test, 0);
  
  segment_info.global_model = global_model;
  segment_info.global_norm = global_norm;
  
  # Step 4: Generate the forecast
  forecast = generate_segmented_forecast(time_series, forecast_horizon, segment_info, options);
  
  printf("Segmented forecasting complete.\n");
endfunction

# Detect segments in the time series
function segment_boundaries = detect_segments(time_series, options)
  n_points = length(time_series);
  method = options.segment_detection_method;
  n_segments = options.n_segments;
  min_segment_size = options.min_segment_size;
  
  # Initialize with default (all data in one segment)
  segment_boundaries = [1, n_points + 1];
  
  if strcmp(method, "equal")
    # Equal-sized segments (simplest approach)
    segment_size = floor(n_points / n_segments);
    
    if segment_size >= min_segment_size
      segment_boundaries = [1];
      for i = 1:n_segments-1
        segment_boundaries = [segment_boundaries, i * segment_size + 1];
      endfor
      segment_boundaries = [segment_boundaries, n_points + 1];
    else
      # If segments would be too small, reduce the number of segments
      adjusted_n_segments = floor(n_points / min_segment_size);
      
      if adjusted_n_segments > 1
        segment_size = floor(n_points / adjusted_n_segments);
        segment_boundaries = [1];
        for i = 1:adjusted_n_segments-1
          segment_boundaries = [segment_boundaries, i * segment_size + 1];
        endfor
        segment_boundaries = [segment_boundaries, n_points + 1];
        
        printf("Adjusted number of segments from %d to %d to ensure minimum segment size\n", 
               n_segments, adjusted_n_segments);
      endif
    endif
    
  elseif strcmp(method, "pattern")
    # Pattern-based segmentation (uses derivative changes)
    # This detects points where the behavior of the time series changes significantly
    
    # Calculate first differences (discrete derivative)
    diff_series = diff(time_series);
    
    # Smooth the differences to reduce noise
    window_size = min(31, floor(n_points / 10));
    if mod(window_size, 2) == 0
      window_size += 1; # Ensure odd window size for centered smoothing
    endif
    
    smooth_diff = smooth_series(diff_series, window_size);
    
    # Find sign changes in the smoothed derivative (indicating potential regime changes)
    sign_changes = find(diff(sign(smooth_diff)) != 0);
    
    # Filter sign changes to ensure minimum distance between segments
    sign_changes = filter_close_points(sign_changes, min_segment_size);
    
    # Limit to desired number of segments
    if length(sign_changes) > n_segments - 1
      # If we have too many sign changes, keep the most significant ones
      # (where the derivative changes the most)
      derivative_changes = abs(diff(smooth_diff(sign_changes)));
      
      # Get indices of the largest changes
      [~, top_indices] = sort(derivative_changes, 'descend');
      sign_changes = sort(sign_changes(top_indices(1:min(length(top_indices), n_segments - 1))));
    endif
    
    # Create segment boundaries
    segment_boundaries = [1, sign_changes + 1, n_points + 1];
    
  else # "adaptive" is the default method
    # Adaptive segmentation (uses statistical properties)
    # This method uses hierarchical clustering of windows based on statistical features
    
    # Choose window size (balance between too short and too long)
    window_size = min(90, max(30, floor(n_points / 20)));
    step_size = max(1, floor(window_size / 4));
    
    # Create overlapping windows
    n_windows = floor((n_points - window_size) / step_size) + 1;
    
    if n_windows >= 5 # Need enough windows for meaningful clustering
      # Extract features for each window
      features = zeros(n_windows, 4);
      
      for i = 1:n_windows
        start_idx = (i - 1) * step_size + 1;
        end_idx = start_idx + window_size - 1;
        window_data = time_series(start_idx:end_idx);
        
        # Calculate statistical features
        features(i, 1) = mean(window_data);
        features(i, 2) = std(window_data);
        features(i, 3) = skewness(window_data);
        
        # Simple trend (regression slope)
        X = [ones(length(window_data), 1), (1:length(window_data))'];
        b = X \ window_data(:);
        features(i, 4) = b(2); # Slope
      endfor
      
      # Normalize features
      features = (features - mean(features)) ./ (std(features) + 1e-8);
      
      # Hierarchical clustering of windows
      Z = linkage(features, 'ward');
      clusters = cluster(Z, 'maxclust', n_segments);
      
      # Find transition points between clusters
      transitions = find(diff(clusters) != 0);
      
      # Convert window transitions to data point indices
      segment_boundaries = [1];
      for i = 1:length(transitions)
        boundary = (transitions(i) * step_size) + floor(window_size / 2);
        segment_boundaries = [segment_boundaries, boundary];
      endfor
      segment_boundaries = [segment_boundaries, n_points + 1];
      
      # Filter boundaries to ensure minimum segment size
      segment_boundaries = filter_boundaries(segment_boundaries, min_segment_size);
    endif
  endif
  
  printf("Detected %d segments using '%s' method\n", length(segment_boundaries) - 1, method);
endfunction

# Apply moving average smoothing to a time series
function smoothed = smooth_series(series, window_size)
  n = length(series);
  half_window = floor(window_size / 2);
  smoothed = zeros(size(series));
  
  for i = 1:n
    # Determine window bounds
    window_start = max(1, i - half_window);
    window_end = min(n, i + half_window);
    
    # Calculate mean for this window
    smoothed(i) = mean(series(window_start:window_end));
  endfor
endfunction

# Filter points to ensure minimum distance between them
function filtered = filter_close_points(points, min_distance)
  if isempty(points)
    filtered = [];
    return;
  endif
  
  filtered = [points(1)];
  
  for i = 2:length(points)
    if points(i) - filtered(end) >= min_distance
      filtered = [filtered, points(i)];
    endif
  endfor
endfunction

# Filter boundaries to ensure minimum segment size
function filtered = filter_boundaries(boundaries, min_size)
  if length(boundaries) <= 2
    filtered = boundaries;
    return;
  endif
  
  filtered = [boundaries(1)];
  
  for i = 2:length(boundaries)-1
    if boundaries(i) - filtered(end) >= min_size && boundaries(i+1) - boundaries(i) >= min_size
      filtered = [filtered, boundaries(i)];
    endif
  endfor
  
  filtered = [filtered, boundaries(end)];
endfunction

# Prepare data for a segment
function [X_train, y_train, X_test, y_test, norm_params] = prepare_segment_data(segment, seq_length)
  n_points = length(segment);
  
  # If segment is too small for sequences, return empty
  if n_points <= seq_length + 10
    X_train = []; y_train = []; X_test = []; y_test = [];
    norm_params = struct();
    return;
  endif
  
  # Create sequences
  X = zeros(seq_length, n_points - seq_length);
  y = zeros(1, n_points - seq_length);
  
  for i = 1:(n_points - seq_length)
    X(:, i) = segment(i:(i+seq_length-1))';
    y(:, i) = segment(i+seq_length);
  endfor
  
  # Split into training and validation sets
  n_samples = size(X, 2);
  
  # If we have enough data for a meaningful split
  if n_samples >= 20
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
    
    # Store normalization parameters
    norm_params = struct();
    norm_params.X_mean = X_mean;
    norm_params.X_std = X_std;
    norm_params.y_mean = y_mean;
    norm_params.y_std = y_std;
  else
    # If not enough data, return empty arrays
    X_train = []; y_train = []; X_test = []; y_test = [];
    norm_params = struct();
  endif
endfunction

# Train model for a segment
function model = train_segment_model(X_train, y_train, X_test, y_test, segment_idx)
  # Check if we have valid training data
  if isempty(X_train) || isempty(y_train)
    model = [];
    return;
  endif
  
  # Define model architecture
  seq_length = size(X_train, 1);
  
  # For the first few segments, use a potentially more complex architecture
  if segment_idx <= 2
    layer_sizes = [seq_length, 64, 32, 16, 1];
  else
    # For later segments, use a potentially simpler architecture
    layer_sizes = [seq_length, 48, 24, 1];
  endif
  
  model = initNeuralNetwork(layer_sizes);
  
  # Configure model
  model.activation = @tanh;
  model.activation_prime = @tanh_prime;
  model.output_activation = @identity;
  model.output_activation_prime = @identity_prime;
  model.loss_function = @mean_squared_error;
  model.loss_function_prime = @mean_squared_error_prime;
  
  # For global model (segment_idx == 0), use more conservative learning rate
  if segment_idx == 0
    model.learning_rate = 0.003;
    model.max_iter = 3000;
  else
    model.learning_rate = 0.005;
    model.max_iter = 2000;
  endif
  
  model.reg_lambda = 0.0001;
  model.early_stopping_patience = 200;
  model.momentum = 0.9;
  
  # Train the model
  if segment_idx == 0
    printf("Training global model...\n");
  else
    printf("Training model for segment %d...\n", segment_idx);
  endif
  
  [model, history] = train(model, X_train, y_train, X_test, y_test);
  
  # Evaluate on test set
  predictions = predict(model, X_test);
  mse = mean((predictions - y_test).^2);
  
  if segment_idx == 0
    printf("Global model test MSE: %.6f\n", mse);
  else
    printf("Segment %d model test MSE: %.6f\n", segment_idx, mse);
  endif
endfunction

# Generate forecast using segmented approach
function forecast = generate_segmented_forecast(time_series, horizon, segment_info, options)
  # Initialize forecast
  forecast = zeros(1, horizon);
  
  # Get the most recent segment
  boundaries = segment_info.boundaries;
  last_segment_idx = length(boundaries) - 1;
  
  # Prepare the sequence from the end of the time series
  seq_length = options.seq_length;
  last_sequence = time_series(end-seq_length+1:end)';
  
  # Check if we have a valid model for the last segment
  if !isempty(segment_info.models{last_segment_idx}) && 
     !isempty(segment_info.norm_params{last_segment_idx})
    
    # Use the specialized model for the last segment
    printf("Using specialized model for segment %d for forecasting\n", last_segment_idx);
    
    norm_params = segment_info.norm_params{last_segment_idx};
    model = segment_info.models{last_segment_idx};
    
    # Normalize the sequence using segment-specific parameters
    norm_sequence = (last_sequence - norm_params.X_mean) ./ norm_params.X_std;
    
    # Generate forecast using the segment model
    for i = 1:horizon
      # Make prediction
      pred = predict(model, norm_sequence);
      
      # Denormalize for final forecast
      forecast(i) = pred * norm_params.y_std + norm_params.y_mean;
      
      # Update sequence for next step
      norm_sequence = [norm_sequence(2:end); pred];
    endfor
    
  else
    # Fallback to global model if no valid model for the last segment
    printf("No valid model for last segment, using global model for forecasting\n");
    
    norm_params = segment_info.global_norm;
    model = segment_info.global_model;
    
    # Check if global model is valid
    if !isempty(model) && !isempty(norm_params)
      # Normalize the sequence using global parameters
      norm_sequence = (last_sequence - norm_params.X_mean) ./ norm_params.X_std;
      
      # Generate forecast using the global model
      for i = 1:horizon
        # Make prediction
        pred = predict(model, norm_sequence);
        
        # Denormalize for final forecast
        forecast(i) = pred * norm_params.y_std + norm_params.y_mean;
        
        # Update sequence for next step
        norm_sequence = [norm_sequence(2:end); pred];
      endfor
    else
      # If even global model is not available, use naive forecast
      printf("Warning: No valid models available. Using naive forecast method.\n");
      forecast = naive_forecast(time_series, horizon);
    endif
  endif
endfunction

# Generate a naive forecast as fallback
function forecast = naive_forecast(time_series, horizon)
  # Simple seasonal naive forecast
  # Uses the average of last few observations
  
  last_values = time_series(end-min(30, length(time_series))+1:end);
  forecast = repmat(mean(last_values), 1, horizon);
  
  # Add some trend if detected
  if length(time_series) >= 60
    recent_trend = mean(diff(time_series(end-59:end)));
    forecast = forecast + recent_trend * (1:horizon);
  endif
endfunction