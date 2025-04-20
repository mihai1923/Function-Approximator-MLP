# Segmented Regression for Complex Functions
# Breaks down a complex function into segments and trains specialized models for each segment
function run_segmented_regression(func, func_str, options)
  # Set default options if not provided
  if nargin < 3
    options = struct();
  endif
  
  # Apply default values for any missing options
  if !isfield(options, "data_range")
    options.data_range = [-5, 5];
  endif
  
  if !isfield(options, "n_segments")
    options.n_segments = 5;
  endif
  
  if !isfield(options, "max_iter")
    options.max_iter = 5000;
  endif
  
  if !isfield(options, "learning_rate")
    options.learning_rate = 0.001;
  endif
  
  if !isfield(options, "noise")
    options.noise = 0.05;
  endif
  
  if !isfield(options, "n_samples")
    options.n_samples = 800;
  endif
  
  if !isfield(options, "layers")
    options.layers = [1, 150, 100, 75, 50, 1];
  endif
  
  if !isfield(options, "adaptive_segments")
    options.adaptive_segments = true;  # Enable adaptive segmentation by default
  endif
  
  if !isfield(options, "convergence_check")
    options.convergence_check = true;  # Enable convergence check by default
  endif
  
  if !isfield(options, "max_iterations")
    options.max_iterations = 3;  # Maximum number of refinement iterations
  endif
  
  # Extract parameters
  data_range = options.data_range;
  n_segments = options.n_segments;
  max_iter = options.max_iter;
  
  # Calculate segment width for initial segmentation
  range_width = data_range(2) - data_range(1);
  
  # Analyze function for complexity
  if options.adaptive_segments
    printf("Analyzing function complexity for adaptive segmentation...\n");
    # Sample the function densely across the range
    n_analysis_samples = 500;
    x_analysis = linspace(data_range(1), data_range(2), n_analysis_samples)';
    y_analysis = feval(func, x_analysis);
    
    # Calculate first and second derivatives
    first_deriv = diff(y_analysis) ./ diff(x_analysis);
    second_deriv = diff(first_deriv) ./ diff(x_analysis(1:end-1));
    
    # Identify regions of high complexity (high absolute second derivative)
    complexity = abs(second_deriv);
    mean_complexity = mean(complexity);
    std_complexity = std(complexity);
    
    # Define threshold for high complexity
    threshold = mean_complexity + std_complexity;
    
    # Find regions of high complexity
    complex_regions = find(complexity > threshold);
    
    if !isempty(complex_regions)
      # Create adaptive segment boundaries
      segment_boundaries = zeros(1, n_segments + 1);
      segment_boundaries(1) = data_range(1);
      segment_boundaries(end) = data_range(2);
      
      # Distribute segments with more segments in complex regions
      complex_points = x_analysis(complex_regions + 1);  # +1 due to derivatives
      
      if length(complex_points) >= n_segments - 2
        # If we have enough complex points, use them directly
        selected_indices = round(linspace(1, length(complex_points), n_segments - 1));
        segment_boundaries(2:end-1) = complex_points(selected_indices);
      else
        # Otherwise distribute remaining segments evenly
        selected_indices = round(linspace(1, length(complex_points), length(complex_points)));
        segment_boundaries(2:(1+length(complex_points))) = complex_points(selected_indices);
        
        # Distribute remaining segments evenly
        remaining_points = n_segments - 1 - length(complex_points);
        if remaining_points > 0
          remaining_boundaries = linspace(complex_points(end), data_range(2), remaining_points + 1);
          segment_boundaries((2+length(complex_points)):end) = remaining_boundaries(2:end);
        endif
      endif
      
      # Ensure boundaries are sorted and unique
      segment_boundaries = sort(segment_boundaries);
      
      printf("Created %d adaptive segments based on function complexity\n", n_segments);
    else
      # If no complex regions found, use equal segments
      segment_boundaries = linspace(data_range(1), data_range(2), n_segments + 1);
      printf("No complex regions detected, using %d equal segments\n", n_segments);
    endif
  else
    # Use equal segments if adaptive segmentation is disabled
    segment_boundaries = linspace(data_range(1), data_range(2), n_segments + 1);
  endif
  
  # Initialize storage for segment models
  segment_models = cell(1, n_segments);
  segment_info = struct();
  segment_info.boundaries = segment_boundaries;
  segment_info.models = segment_models;
  segment_info.original_func = func;
  segment_info.original_func_str = func_str;
  
  # List to store MSE for each segment
  segment_mses = zeros(1, n_segments);
  
  # List to store max errors for each segment
  segment_max_errors = zeros(1, n_segments);
  
  # Convergence parameters
  prev_ensemble_mse = Inf;
  current_iteration = 1;
  
  # Main training loop with optional refinement
  while current_iteration <= options.max_iterations
    printf("\n==== Iteration %d/%d ====\n", current_iteration, options.max_iterations);
    
    # Train a model for each segment
    for i = 1:n_segments
      segment_start = segment_boundaries(i);
      segment_end = segment_boundaries(i+1);
      segment_width = segment_end - segment_start;
      
      # Adjust segment boundaries slightly to ensure overlap for smooth transitions
      overlap = segment_width * 0.05;
      if i > 1
        adjusted_start = segment_start - overlap;
      else
        adjusted_start = segment_start;
      endif
      
      if i < n_segments
        adjusted_end = segment_end + overlap;
      else
        adjusted_end = segment_end;
      endif
      
      printf("\nTraining segment %d/%d [%.2f, %.2f]...\n", i, n_segments, segment_start, segment_end);
      
      # Generate data for this segment
      segment_range = [adjusted_start, adjusted_end];
      n_samples_segment = round(options.n_samples * (segment_width / range_width) * 1.5);
      n_samples_segment = max(n_samples_segment, 200); # Ensure minimum samples
      
      # Create segment-specific options
      segment_options = options;
      segment_options.data_range = segment_range;
      segment_options.n_samples = n_samples_segment;
      segment_options.quiet = true; # Suppress detailed output from training
      
      # Detect regions of rapid change for enhanced sampling
      sampling_points = linspace(adjusted_start, adjusted_end, 100);
      y_vals = feval(func, sampling_points);
      diffs = abs(diff(y_vals));
      mean_diff = mean(diffs);
      std_diff = std(diffs);
      
      # Check for regions of rapid change
      rapid_change_threshold = mean_diff + 2 * std_diff;
      rapid_change_regions = find(diffs > rapid_change_threshold);
      
      if !isempty(rapid_change_regions)
        n_rapid_regions = length(rapid_change_regions);
        printf("Detected %d regions of rapid change for enhanced sampling\n", n_rapid_regions);
        
        # For rapid change regions, increase samples and possibly adjust architecture
        if n_rapid_regions > 5
          segment_options.n_samples = n_samples_segment * 1.5;
          if !isfield(segment_options, "layers") || length(segment_options.layers) < 6
            segment_options.layers = [1, 200, 150, 100, 75, 50, 1];
          endif
        endif
      endif
      
      # Generate training data for this segment
      x_train = (adjusted_end - adjusted_start) * rand(segment_options.n_samples, 1) + adjusted_start;
      y_train = feval(func, x_train);
      
      # Add noise to training data
      if isfield(segment_options, "noise") && segment_options.noise > 0
        y_train = y_train + segment_options.noise * randn(size(y_train));
      endif
      
      # Configure neural network for this segment
      nn_config = struct();
      nn_config.input_dim = 1;
      nn_config.output_dim = 1;
      nn_config.hidden_layers = segment_options.layers(2:end-1);
      nn_config.activation = "tanh";
      nn_config.output_activation = "identity";
      nn_config.learning_rate = segment_options.learning_rate;
      nn_config.max_iter = segment_options.max_iter;
      nn_config.reg_lambda = 0.001;
      nn_config.convergence_threshold = 1e-6;
      nn_config.momentum = 0.9;
      nn_config.batch_size = 64;
      nn_config.verbose = true;
      
      # Initialize and train neural network
      nn = initNeuralNetwork(nn_config);
      printf("Training segment %d with %d samples...\n", i, length(x_train));
      [nn, training_history] = train(nn, x_train, y_train);
      
      # Store the trained model for this segment
      segment_models{i} = nn;
      
      # Evaluate segment performance
      x_eval = linspace(adjusted_start, adjusted_end, 200)';
      y_true = feval(func, x_eval);
      y_pred = predict(nn, x_eval);
      
      # Calculate MSE for this segment
      segment_mse = mean((y_true - y_pred).^2);
      segment_mses(i) = segment_mse;
      
      # Calculate maximum error
      segment_max_error = max(abs(y_true - y_pred));
      segment_max_errors(i) = segment_max_error;
      
      printf("Segment %d MSE: %f\n", i, segment_mse);
      printf("Segment %d Maximum error: %f\n", i, segment_max_error);
    endfor
    
    # Update segment info with trained models
    segment_info.models = segment_models;
    
    # Calculate overall MSE
    ensemble_mse = mean(segment_mses);
    printf("\nEnsemble MSE: %f\n", ensemble_mse);
    printf("Maximum error: %f\n", max(segment_max_errors));
    
    # Check for convergence
    if options.convergence_check
      improvement = (prev_ensemble_mse - ensemble_mse) / prev_ensemble_mse;
      printf("Improvement from previous iteration: %.2f%%\n", improvement * 100);
      
      # Break if improvement is minimal
      if improvement < 0.05 && current_iteration > 1
        printf("Minimal improvement detected (%.2f%%). Stopping iterations.\n", improvement * 100);
        break;
      endif
      
      # Check if any segment needs refinement (has higher than average error)
      if current_iteration < options.max_iterations
        high_error_segments = find(segment_mses > 1.5 * ensemble_mse);
        
        if !isempty(high_error_segments)
          printf("Refining segments with high error for next iteration...\n");
          
          # Refine the boundaries to allocate more segments to high-error regions
          new_boundaries = segment_boundaries;
          
          for seg_idx = high_error_segments
            if seg_idx < n_segments  # Can't split the last segment
              # Split this segment into two
              mid_point = (segment_boundaries(seg_idx) + segment_boundaries(seg_idx+1)) / 2;
              new_boundaries = [new_boundaries(1:seg_idx), mid_point, new_boundaries(seg_idx+1:end)];
              
              printf("Splitting segment %d [%.2f, %.2f] at %.2f\n", 
                     seg_idx, segment_boundaries(seg_idx), segment_boundaries(seg_idx+1), mid_point);
            endif
          endfor
          
          # Too many segments? Remove boundaries from low-error regions
          while length(new_boundaries) > n_segments + 1 + length(high_error_segments)
            # Find lowest error segment that has not been split
            remaining_segments = setdiff(1:n_segments, high_error_segments);
            [~, idx] = min(segment_mses(remaining_segments));
            remove_idx = remaining_segments(idx) + 1;  # +1 because we want the boundary
            
            # Don't remove first or last boundary
            if remove_idx > 1 && remove_idx < length(new_boundaries)
              printf("Removing boundary at %.2f from low-error region\n", new_boundaries(remove_idx));
              new_boundaries(remove_idx) = [];
            endif
          endwhile
          
          # Ensure we don't exceed the target number of segments
          if length(new_boundaries) > n_segments + 1
            # Keep only n_segments + 1 boundaries
            # Sort by error and keep boundaries of highest error segments
            [~, sorted_idx] = sort(segment_mses, "descend");
            keep_boundaries = [data_range(1), segment_boundaries(sorted_idx(1:n_segments-1)+1), data_range(2)];
            new_boundaries = sort(keep_boundaries);
          endif
          
          # Update boundaries for next iteration
          segment_boundaries = sort(new_boundaries);
          n_segments = length(segment_boundaries) - 1;
          
          # Reinitialize arrays for next iteration
          segment_models = cell(1, n_segments);
          segment_mses = zeros(1, n_segments);
          segment_max_errors = zeros(1, n_segments);
        endif
      endif
    endif
    
    # Update for next iteration
    prev_ensemble_mse = ensemble_mse;
    current_iteration += 1;
  endwhile
  
  # Final update of segment info
  segment_info.boundaries = segment_boundaries;
  segment_info.models = segment_models;
  
  # Save the ensemble model
  safe_func_str = strrep(func_str, "^", "_pow_");
  safe_func_str = strrep(safe_func_str, "*", "_star_");
  safe_func_str = strrep(safe_func_str, "/", "_div_");
  safe_func_str = strrep(safe_func_str, " ", "");
  safe_func_str = strrep(safe_func_str, "(", "");
  safe_func_str = strrep(safe_func_str, ")", "");
  model_filename = sprintf("../data/segmented_regression_model_%s.mat", safe_func_str);
  
  # Create a model info structure
  model_info = struct();
  model_info.segment_info = segment_info;
  model_info.func_str = func_str;
  model_info.data_range = data_range;
  model_info.n_segments = n_segments;
  model_info.segment_mses = segment_mses;
  model_info.ensemble_mse = ensemble_mse;
  model_info.segment_max_errors = segment_max_errors;
  
  # Save the model
  save("-binary", model_filename, "model_info");
  printf("Saved segmented model to %s\n", model_filename);
  
  # Generate high-quality plot of the segmented model predictions
  plot_segmented_model(segment_info, func_str, data_range);
  
  # Display summary
  printf("\nSegmented Regression Summary:\n");
  printf("----------------------------\n");
  printf("Function: %s\n", func_str);
  printf("Range: [%.2f, %.2f]\n", data_range(1), data_range(2));
  printf("Number of segments: %d\n", n_segments);
  printf("Overall MSE: %f\n", ensemble_mse);
  printf("Maximum error: %f\n", max(segment_max_errors));
  printf("Model saved to %s\n", model_filename);
endfunction

# Function to plot the segmented model predictions
function plot_segmented_model(segment_info, func_str, data_range)
  # Extract parameters
  segment_boundaries = segment_info.boundaries;
  segment_models = segment_info.models;
  original_func = segment_info.original_func;
  n_segments = length(segment_models);
  
  # Create a dense grid for plotting
  n_points = 1000;
  x_dense = linspace(data_range(1), data_range(2), n_points)';
  y_true = feval(original_func, x_dense);
  
  # Generate predictions using the segmented model
  y_pred = zeros(size(x_dense));
  
  for i = 1:length(x_dense)
    x = x_dense(i);
    # Find which segment this point belongs to
    segment_idx = find(x >= segment_boundaries(1:end-1) & x <= segment_boundaries(2:end), 1);
    if isempty(segment_idx)
      # Handle edge case
      if x < segment_boundaries(1)
        segment_idx = 1;
      else
        segment_idx = n_segments;
      endif
    endif
    
    # Get prediction from the appropriate segment model
    y_pred(i) = predict(segment_models{segment_idx}, x);
  endfor
  
  # Create a new figure
  figure("position", [100, 100, 1000, 600]);
  
  # Plot the function and predictions
  subplot(2, 1, 1);
  hold on;
  
  # Plot true function
  h_true = plot(x_dense, y_true, "b", "linewidth", 2);
  
  # Plot predictions
  h_pred = plot(x_dense, y_pred, "r", "linewidth", 2);
  
  # Plot segment boundaries
  for i = 2:length(segment_boundaries)-1
    boundary = segment_boundaries(i);
    plot([boundary, boundary], ylim, "k--", "linewidth", 1);
  endfor
  
  # Add legend and labels
  legend([h_true, h_pred], {"True function", "Neural network prediction"}, "location", "best");
  title(sprintf("Segmented Regression for f(x) = %s", func_str), "fontsize", 14);
  xlabel("x", "fontsize", 12);
  ylabel("f(x)", "fontsize", 12);
  grid on;
  
  # Plot error
  subplot(2, 1, 2);
  error = abs(y_true - y_pred);
  plot(x_dense, error, "m", "linewidth", 2);
  
  # Plot segment boundaries on error plot too
  hold on;
  for i = 2:length(segment_boundaries)-1
    boundary = segment_boundaries(i);
    plot([boundary, boundary], ylim, "k--", "linewidth", 1);
  endfor
  
  title("Absolute Error", "fontsize", 14);
  xlabel("x", "fontsize", 12);
  ylabel("|Error|", "fontsize", 12);
  grid on;
  
  # Add segment numbers
  for i = 1:n_segments
    segment_center = (segment_boundaries(i) + segment_boundaries(i+1)) / 2;
    text(segment_center, 0.9 * max(error), sprintf("Seg %d", i), "horizontalalignment", "center");
  endfor
  
  # Prepare filename for saving the plot
  safe_func_str = strrep(func_str, "^", "_pow_");
  safe_func_str = strrep(func_str, "*", "_star_");
  safe_func_str = strrep(func_str, "/", "_div_");
  safe_func_str = strrep(func_str, " ", "");
  safe_func_str = strrep(func_str, "(", "");
  safe_func_str = strrep(func_str, ")", "");
  plot_filename = sprintf("../plots/segmented_regression_result_%s.png", safe_func_str);
  
  # Save the plot
  print(plot_filename, "-dpng", "-r300");
  printf("Saved plot to %s\n", plot_filename);
  
  # Also save the training history plot for each segment
  history_plot_filename = sprintf("../plots/segmented_regression_training_history_%s.png", safe_func_str);
  figure("position", [100, 100, 800, 600]);
  
  # For each segment, plot a separate line
  colors = {"b", "r", "g", "m", "c", "y", "k", "b--", "r--", "g--", "m--", "c--", "y--", "k--"};
  
  hold on;
  legend_handles = [];
  legend_labels = {};
  
  for i = 1:n_segments
    segment_start = segment_boundaries(i);
    segment_end = segment_boundaries(i+1);
    
    # Evaluate segment performance
    x_eval = linspace(segment_start, segment_end, 200)';
    y_true_segment = feval(original_func, x_eval);
    y_pred_segment = predict(segment_models{i}, x_eval);
    
    # Calculate MSE for this segment
    segment_mse = mean((y_true_segment - y_pred_segment).^2);
    
    # Plot a point for this segment's MSE
    color_idx = mod(i-1, length(colors)) + 1;
    h = plot(i, segment_mse, [colors{color_idx}, "o"], "markersize", 10, "markerfacecolor", colors{color_idx}(1));
    legend_handles = [legend_handles, h];
    legend_labels{end+1} = sprintf("Segment %d [%.1f, %.1f]", i, segment_start, segment_end);
    
    # Add segment MSE as text
    text(i, segment_mse * 1.1, sprintf("%.4f", segment_mse), "horizontalalignment", "center");
  endfor
  
  # Add overall MSE
  h_overall = plot([0.5, n_segments+0.5], [mean(segment_mse), mean(segment_mse)], "k--", "linewidth", 2);
  legend_handles = [legend_handles, h_overall];
  legend_labels{end+1} = sprintf("Overall MSE: %.4f", mean(segment_mse));
  
  # Set axis and labels
  title(sprintf("MSE per Segment for f(x) = %s", func_str), "fontsize", 14);
  xlabel("Segment Number", "fontsize", 12);
  ylabel("Mean Squared Error", "fontsize", 12);
  grid on;
  xlim([0.5, n_segments+0.5]);
  set(gca, "xtick", 1:n_segments);
  
  # Add legend
  legend(legend_handles, legend_labels, "location", "best");
  
  # Save this plot
  print(history_plot_filename, "-dpng", "-r300");
  printf("Saved training history plot to %s\n", history_plot_filename);
endfunction