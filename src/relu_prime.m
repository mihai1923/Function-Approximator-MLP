function dy = relu_prime(x)
  # Derivative of ReLU activation function
  # Returns 1 where x > 0, otherwise 0
  # Critical for backpropagation with ReLU networks
  
  # Handle batch normalization case where x is a struct instead of numeric
  if isstruct(x)
    # Extract the pre-batch normalization value from the struct
    if isfield(x, "pre_bn")
      x_val = x.pre_bn;
    else
      # Fallback to other possible field names
      if isfield(x, "normalized")
        x_val = x.normalized;
      elseif isfield(x, "raw")
        x_val = x.raw;
      else
        # If no expected fields exist, print available fields and use a default approach
        fields = fieldnames(x);
        warning("Unexpected batch normalization struct format. Available fields: %s", strjoin(fields, ", "));
        # Try to use the first numeric field or generate zeros
        for i = 1:length(fields)
          field_val = x.(fields{i});
          if isnumeric(field_val)
            x_val = field_val;
            break;
          endif
        endfor
        if !exist("x_val", "var")
          error("Could not find any numeric field in batch normalization struct");
        endif
      endif
    endif
    dy = (x_val > 0);
  else
    # Standard case - x is numeric
    dy = (x > 0);
  endif
endfunction