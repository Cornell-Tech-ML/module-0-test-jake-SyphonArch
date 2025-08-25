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

import math


def mul(x, y):
    return x * y

def id(x):
    return x

def add(x, y):
    return x + y

def neg(x):
    return -x

def lt(x, y):
    return x < y

def eq(x, y):
    return x == y

def max(x, y):
    return x if x > y else y

def is_close(x, y):
    return abs(x - y) < 1e-2

def sigmoid(x):
    if x >= 0:
        return 1 / (1 + exp(-x))
    else:
        return exp(x) / (1 + exp(x))

def relu(x):
    return max(0, x)

def log(x):
    return math.log(x)

def exp(x):
    return math.exp(x)

def inv(x):
    return 1 / x

def log_back(x, d):
    return d / x

def inv_back(x, d):
    return -d / (x ** 2)

def relu_back(x, d):
    return d if x > 0 else 0


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


def map(fn, iterable):
    return [fn(x) for x in iterable]

def zipWith(fn, list1, list2):
    return [fn(x, y) for x, y in zip(list1, list2)]

def reduce(fn, iterable, initial_value):
    result = initial_value
    for x in iterable:
        result = fn(result, x)
    return result

def negList(lst):
    return map(neg, lst)

def addLists(lst1, lst2):
    return zipWith(add, lst1, lst2)

def sum(lst):
    return reduce(add, lst, 0)

def prod(lst):
    return reduce(mul, lst, 1)
