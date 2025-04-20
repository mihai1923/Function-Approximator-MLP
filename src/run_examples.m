# Main script for neural network examples

# Add functions to path
addpath(pwd);

# Create directories if needed
if !exist("../data", "dir")
  mkdir("../data");
endif

if !exist("../plots", "dir")
  mkdir("../plots");
endif

# Display menu
printf("Neural Network From Scratch - Example Selector\n");
printf("=============================================\n");
printf("1. Classification Example (Spiral Dataset)\n");
printf("2. Regression Example (Function Approximation)\n");
printf("3. Time Series Forecasting\n");
printf("\n");

# Prompt for selection
selection = input("Enter example number to run (1-3): ");

switch(selection)
  case 1
    printf("\nRunning Classification Example...\n\n");
    example_classification;

  case 2
    printf("\nRegression Example - Function Approximation\n");
    printf("------------------------------------------\n");
    printf("You can choose a built-in function or define your own:\n");
    printf("1. Default: sin(2*pi*x)*x + 0.5*x^2\n");
    printf("2. Polynomial: x^3 - 2*x^2 + x - 3\n");
    printf("3. Trigonometric: 0.5*sin(3*x) + cos(x)\n");
    printf("4. Custom function\n\n");

    reg_selection = input("Enter your choice (1-4): ");

    if reg_selection == 4
      # Get custom function
      printf("\nEnter your custom function using Octave syntax.\n");
      printf("Use 'x' as the variable. Example: 2*x^2 + sin(x)\n");
      custom_func_str = input("Custom function: ", "s");
      
      # Convert to element-wise operations
      safe_func_str = strrep(custom_func_str, "^", ".^");
      safe_func_str = strrep(safe_func_str, "*", ".*");
      safe_func_str = strrep(safe_func_str, "/", "./");
      
      # Fix double dots
      safe_func_str = strrep(safe_func_str, "..^", ".^");
      safe_func_str = strrep(safe_func_str, "..*", ".*");
      safe_func_str = strrep(safe_func_str, "../", "./");
      
      # Custom range option
      printf("\n== Custom Range Configuration ==\n");
      printf("For complex functions, using a wider range with more segments can improve approximation.\n");
      printf("Would you like to specify a custom range? (Default is [-2, 2])\n");
      use_custom_range = input("Enter y/n: ", "s");
      
      options = struct();
      
      if strcmpi(use_custom_range, "y")
        printf("\nEnter the lower bound (e.g., -30): ");
        lower_bound = input("");
        printf("Enter the upper bound (e.g., 30): ");
        upper_bound = input("");
        
        # Validate range
        if upper_bound <= lower_bound
          printf("Warning: Upper bound must be greater than lower bound. Using default range.\n");
          options.data_range = [-2, 2];
        else
          options.data_range = [lower_bound, upper_bound];
          
          # Calculate range width for parameter adjustment
          range_width = upper_bound - lower_bound;
          
          printf("\n== Segmentation Configuration ==\n");
          printf("Segmentation divides the range into smaller parts for better approximation.\n");
          printf("Segments will be automatically calculated based on the optimal segment length.\n");
          
          # Calculate samples based on range size
          default_width = 4;
          scale_factor = max(1, range_width / default_width);
          
          # Scale samples for wider ranges
          base_samples = 1000;
          scaled_samples = round(base_samples * scale_factor);
          options.n_samples = min(scaled_samples, 10000);
          
          printf("\n== Training Configuration ==\n");
          printf("Automatically calculated parameters based on your range:\n");
          printf("- Range: [%.1f, %.1f] (width: %.1f)\n", lower_bound, upper_bound, range_width);
          printf("- Training samples: %d\n", options.n_samples);
          
          # Adjust network for complex functions
          if range_width > 10
            options.layers = [1, 150, 100, 50, 1];
            options.max_iter = 8000;
            printf("- Using enhanced network architecture for complex function\n");
          endif
        endif
      endif
      
      # Create function and start training
      custom_func = inline(safe_func_str, "x");
      printf("\nStarting regression training. This may take a while for complex functions...\n");
      run_custom_regression(custom_func, custom_func_str, options);
    else
      # Run with selected built-in function
      if reg_selection == 1
        func = @(x) sin(2*pi*x).*x + 0.5*x.^2;
        func_str = "sin(2*pi*x)*x + 0.5*x^2";
      elseif reg_selection == 2
        func = @(x) x.^3 - 2*x.^2 + x - 3;
        func_str = "x^3 - 2*x^2 + x - 3";
      elseif reg_selection == 3
        func = @(x) 0.5*sin(3*x) + cos(x);
        func_str = "0.5*sin(3*x) + cos(x)";
      else
        # Default option
        func = @(x) sin(2*pi*x).*x + 0.5*x.^2;
        func_str = "sin(2*pi*x)*x + 0.5*x.^2";
      endif

      run_custom_regression(func, func_str);
    endif

  case 3
    printf("\nTime Series Forecasting\n");
    printf("----------------------\n");
    printf("You can choose a built-in time series or define your own:\n");
    printf("1. Default: sin(x) + 0.2*sin(5*x) + noise\n");
    printf("2. Damped oscillation: exp(-0.1*x)*sin(x) + noise\n");
    printf("3. Seasonal pattern: 2*sin(0.1*x) + sin(x) + noise\n");
    printf("4. Custom time series function\n\n");

    ts_selection = input("Enter your choice (1-4): ");

    if ts_selection == 4
      # Get custom time series function
      printf("\nEnter your custom time series function using Octave syntax.\n");
      printf("Use 'x' as the time variable. Example: sin(0.5*x) + 0.1*x\n");
      custom_ts_str = input("Custom function: ", "s");
      
      # Normalize expression
      clean_ts_str = strrep(custom_ts_str, " ^ ", "^");
      clean_ts_str = strrep(clean_ts_str, " ^", "^");
      clean_ts_str = strrep(clean_ts_str, "^ ", "^");
      clean_ts_str = strrep(clean_ts_str, " * ", "*");
      clean_ts_str = strrep(clean_ts_str, " *", "*");
      clean_ts_str = strrep(clean_ts_str, "* ", "*");
      clean_ts_str = strrep(clean_ts_str, " / ", "/");
      clean_ts_str = strrep(clean_ts_str, " /", "/");
      clean_ts_str = strrep(clean_ts_str, "/ ", "/");
      
      # Convert to element-wise operations
      safe_ts_str = strrep(clean_ts_str, "^", ".^");
      safe_ts_str = strrep(safe_ts_str, "*", ".*");
      safe_ts_str = strrep(safe_ts_str, "/", "./");
      
      # Fix double dots
      safe_ts_str = strrep(safe_ts_str, "..^", ".^");
      safe_ts_str = strrep(safe_ts_str, "..*", ".*");
      safe_ts_str = strrep(safe_ts_str, "../", "./");
      
      # Create function handle
      custom_ts = inline(safe_ts_str, "x");
      
      printf("Input function: %s\n", custom_ts_str);
      printf("Clean function: %s\n", clean_ts_str);
      printf("Element-wise operations: %s\n", safe_ts_str);

      run_custom_time_series(custom_ts, custom_ts_str);
    else
      # Run with built-in function
      if ts_selection == 1
        ts_func = @(x) sin(x) + 0.2.*sin(5.*x);
        ts_func_str = "sin(x) + 0.2*sin(5*x) + noise";
        run_custom_time_series(ts_func, ts_func_str);
      elseif ts_selection == 2
        ts_func = @(x) exp(-0.1.*x).*sin(x);
        ts_func_str = "exp(-0.1*x)*sin(x) + noise";
        run_custom_time_series(ts_func, ts_func_str);
      elseif ts_selection == 3
        ts_func = @(x) 2.*sin(0.1.*x) + sin(x);
        ts_func_str = "2*sin(0.1*x) + sin(x) + noise";
        
        printf("\nUsing enhanced predictor for seasonal pattern with super precision...\n");
        
        # Set flags for seasonal pattern handling
        options = struct();
        options.is_seasonal = true;
        options.frequencies = [0.1, 1];
        options.amplitudes = [2, 1];
        options.use_enhanced_model = true;
        options.sequence_length = 30;
        
        run_enhanced_seasonal_prediction(ts_func, ts_func_str, options);
      else
        # Default option
        ts_func = @(x) sin(x) + 0.2.*sin(5.*x);
        ts_func_str = "sin(x) + 0.2*sin(5*x) + noise";
        run_custom_time_series(ts_func, ts_func_str);
      endif
    endif

  otherwise
    printf("\nInvalid selection. Please enter a number between 1 and 3.\n");
endswitch