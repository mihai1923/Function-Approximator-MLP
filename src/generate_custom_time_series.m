function [X, y] = generate_custom_time_series(n_points=1000, seq_length=20, ts_func)
  # Generate a time series dataset with the provided function
  # Enhanced version with better handling for complex functions

  # Try to detect if the function contains exponential components
  # For exponential-like functions, use a smaller domain to prevent explosion
  contains_exp = false;
  
  # Use safer method to check function string that works with inline functions
  if isa(ts_func, "function_handle")
    # Try to get function as string safely
    try
      func_str = func2str(ts_func);
      # Check if it's an anonymous function with visible body
      if any(strfind(func_str, "exp")) || any(strfind(func_str, "e.^"))
        contains_exp = true;
        printf("Detected exponential component in function. Using restricted domain.\n");
      endif
    catch
      # If func2str fails, try to detect 'exp' in the original function string
      # This is passed as ts_func_str in the calling function and we can't access it directly
      printf("Could not analyze function directly. Assuming it may contain exponential terms.\n");
      contains_exp = true;  # Safest to assume exponential for complex functions
    end_try_catch
  else
    # For inline functions, which are actually objects in Octave
    try 
      # Try using the formula attribute of inline objects
      if isfield(ts_func, "formula") || isprop(ts_func, "formula")
        formula = ts_func.formula;
        if any(strfind(formula, "exp")) || any(strfind(formula, "e.^"))
          contains_exp = true;
          printf("Detected exponential component in inline function. Using restricted domain.\n");
        endif
      endif
    catch
      # If that fails too, just assume a potentially complex function
      printf("Could not analyze inline function. Using restricted domain to be safe.\n");
      contains_exp = true;
    end_try_catch
  endif
  
  # Adjust range based on function characteristics
  if contains_exp
    x = linspace(0, 10, n_points);  # Smaller range for exponential functions
    printf("Using restricted range [0, 10] for exponential function.\n");
  else
    x = linspace(0, 30, n_points);  # Standard range
  endif
  
  # Safely evaluate the function with error handling
  try
    # Try to evaluate the function on the x values
    base_series = ts_func(x);
    
    # Check for NaN or Inf values
    if any(isnan(base_series)) || any(isinf(base_series))
      printf("Error: Function produced NaN or Inf values.\n");
      # Try again with an even smaller range if it's an exponential function
      if contains_exp
        x = linspace(0, 5, n_points);
        printf("Retrying with smaller range [0, 5]...\n");
        base_series = ts_func(x);
        
        if any(isnan(base_series)) || any(isinf(base_series))
          error("Function evaluation still produced NaN/Inf values - exiting program.");
        endif
      else
        error("Function evaluation failed - exiting program.");
      endif
    endif
    
    # Detect if values are growing exponentially
    if max(base_series) > 1000 * min(abs(base_series(base_series != 0)))
      printf("Detected exponential growth pattern.\n");
      
      # Apply log transformation for exponential growth (preserving sign)
      sign_base = sign(base_series);
      log_base = sign_base .* log(abs(base_series) + 1);
      
      printf("Applied logarithmic transformation to manage exponential growth.\n");
      printf("Original range: [%.2f, %.2f], Transformed range: [%.2f, %.2f]\n", 
             min(base_series), max(base_series), min(log_base), max(log_base));
      
      base_series = log_base;
    else
      # Scale extremely large values to prevent training issues
      max_abs_value = max(abs(base_series));
      if max_abs_value > 100
        scaling_factor = 100 / max_abs_value;
        printf("Scaling very large values (max: %.2f) by factor of %.4f\n", max_abs_value, scaling_factor);
        base_series = base_series * scaling_factor;
      endif
      
      # Scale extremely small values to improve training
      max_abs_value = max(abs(base_series));
      if max_abs_value < 0.01 && max_abs_value > 0
        scaling_factor = min(0.1 / max_abs_value, 1000);
        printf("Scaling very small values (max: %.6f) by factor of %.2f\n", max_abs_value, scaling_factor);
        base_series = base_series * scaling_factor;
      endif
    endif
    
    # Standardize the data for better neural network training
    mean_val = mean(base_series);
    std_val = std(base_series);
    
    # Avoid division by zero
    if std_val < 1e-10
      std_val = 1;
    endif
    
    # Standardize to mean=0, std=1
    base_series = (base_series - mean_val) / std_val;
    printf("Standardized series to mean=0, std=1 for better training.\n");
    
  catch err
    # If evaluation fails, exit with error message
    printf("Error evaluating function: %s\n", err.message);
    error("Function evaluation failed - exiting program.");
  end_try_catch
  
  # Add controlled noise to the time series with adaptive noise level
  max_abs_value = max(abs(base_series));
  # Use much smaller noise for complex functions
  adaptive_noise = min(0.02, max(0.005, max_abs_value * 0.01));
  base_series = base_series + adaptive_noise * randn(1, n_points);
  printf("Added small controlled noise (amplitude: %.4f) for stability.\n", adaptive_noise);

  # Create sequences for input and output
  X = zeros(seq_length, n_points - seq_length);
  y = zeros(1, n_points - seq_length);

  for i = 1:(n_points - seq_length)
    X(:, i) = base_series(i:(i+seq_length-1))';
    y(:, i) = base_series(i+seq_length);
  endfor

  # Store preprocessing parameters for later denormalization
  X_meta.contains_exp = contains_exp;
  X_meta.x_range = [min(x), max(x)];
  X_meta.mean = mean_val;
  X_meta.std = std_val;
  X_meta.max_orig = max_abs_value;
  
  # Save meta information for proper denormalization
  save("-binary", "../data/last_preprocessing_meta.mat", "X_meta");
  
  return;
endfunction