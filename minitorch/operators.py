"""Collection of the core mathematical operators used throughout the code base.

This module implements fundamental mathematical operations that serve as building blocks
for neural network computations in MiniTorch.
"""

# =============================================================================
# Task 0.1: Mathematical Operators
# =============================================================================

"""
Implementation of elementary mathematical functions.

FUNCTIONS TO IMPLEMENT:
    Basic Operations:
    - mul(x, y)     → Multiply two numbers
    - id(x)         → Return input unchanged (identity function)
    - add(x, y)     → Add two numbers
    - neg(x)        → Negate a number
    
    Comparison Operations:
    - lt(x, y)      → Check if x < y
    - eq(x, y)      → Check if x == y
    - max(x, y)     → Return the larger of two numbers
    - is_close(x, y) → Check if two numbers are approximately equal
    
    Activation Functions:
    - sigmoid(x)    → Apply sigmoid activation: 1/(1 + e^(-x))
    - relu(x)       → Apply ReLU activation: max(0, x)
    
    Mathematical Functions:
    - log(x)        → Natural logarithm
    - exp(x)        → Exponential function
    - inv(x)        → Reciprocal (1/x)
    
    Derivative Functions (for backpropagation):
    - log_back(x, d)  → Derivative of log: d/x
    - inv_back(x, d)  → Derivative of reciprocal: -d/(x²)
    - relu_back(x, d) → Derivative of ReLU: d if x>0, else 0

IMPORTANT IMPLEMENTATION NOTES:

Numerically Stable Sigmoid:
   To avoid numerical overflow, use different formulations based on input sign:
   
   For x ≥ 0:  sigmoid(x) = 1/(1 + exp(-x))
   For x < 0:  sigmoid(x) = exp(x)/(1 + exp(x))
   
   Why? This prevents computing exp(large_positive_number) which causes overflow.

is_close Function:
   Use tolerance: |x - y| < 1e-2
   This handles floating-point precision issues in comparisons.

Derivative Functions (Backpropagation):
   These compute: derivative_of_function(x) × upstream_gradient
   
   - log_back(x, d):  d/dx[log(x)] = 1/x  →  return d/x
   - inv_back(x, d):  d/dx[1/x] = -1/x**2   →  return -d/(x**2)
   - relu_back(x, d): d/dx[relu(x)] = 1 if x>0 else 0  →  return d if x>0 else 0
"""

# TODO: Implement all functions listed above for Task 0.1


# =============================================================================
# Task 0.3: Higher-Order Functions
# =============================================================================

"""
Implementation of functional programming concepts using higher-order functions.

These functions work with other functions as arguments, enabling powerful
abstractions for list operations.

CORE HIGHER-ORDER FUNCTIONS TO IMPLEMENT:

    map(fn, iterable):
        Apply function `fn` to each element of `iterable`
        Example: map(lambda x: x*2, [1,2,3]) → [2,4,6]
    
    zipWith(fn, list1, list2):
        Combine corresponding elements from two lists using function `fn`
        Example: zipWith(add, [1,2,3], [4,5,6]) → [5,7,9]
    
    reduce(fn, iterable, initial_value):
        Reduce iterable to single value by repeatedly applying `fn`
        Example: reduce(add, [1,2,3,4], 0) → 10

FUNCTIONS TO BUILD USING THE ABOVE:

    negList(lst):
        Negate all elements in a list
        Implementation hint: Use map with the neg function
        
    addLists(lst1, lst2):
        Add corresponding elements from two lists
        Implementation hint: Use zipWith with the add function
        
    sum(lst):
        Sum all elements in a list
        Implementation hint: Use reduce with add function and initial value 0
        
    prod(lst):
        Calculate product of all elements in a list
        Implementation hint: Use reduce with mul function and initial value 1
"""

# TODO: Implement all functions listed above for Task 0.3
