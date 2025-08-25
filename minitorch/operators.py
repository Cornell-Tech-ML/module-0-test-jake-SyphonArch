"""Collection of the core mathematical operators used throughout the code base.

This module implements fundamental mathematical operations that serve as building blocks
for neural network computations in MiniTorch.
"""

from collections.abc import Callable
import math

# =============================================================================
# Task 0.1: Mathematical Operators
# =============================================================================

# TODO: Implement all functions listed above for Task 0.1


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Identity function: return input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> bool:
    """Check if x < y."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Check if x == y."""
    return x == y


def max(x: float, y: float) -> float:
    """Return the larger of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are approximately equal."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Apply sigmoid activation."""
    if x >= 0:
        return 1 / (1 + exp(-x))
    else:
        return exp(x) / (1 + exp(x))


def relu(x: float) -> float:
    """Apply ReLU activation."""
    return max(0, x)


def log(x: float) -> float:
    """Natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Exponential function."""
    return math.exp(x)


def inv(x: float) -> float:
    """Reciprocal function."""
    return 1 / x


def log_back(x: float, d: float) -> float:
    """Derivative of log: d/dx[log(x)] = 1/x."""
    return d / x


def inv_back(x: float, d: float) -> float:
    """Derivative of reciprocal: d/dx[1/x] = -1/x²."""
    return -d / (x**2)


def relu_back(x: float, d: float) -> float:
    """Derivative of ReLU: d/dx[relu(x)] = 1 if x>0 else 0."""
    return d if x > 0 else 0


# =============================================================================
# Task 0.3: Higher-Order Functions
# =============================================================================


def map(fn: Callable, iterable: list) -> list:
    """Apply function `fn` to each element of `iterable`."""
    return [fn(x) for x in iterable]


def zipWith(fn: Callable, list1: list, list2: list) -> list:
    """Combine corresponding elements from two lists using function `fn`."""
    return [fn(x, y) for x, y in zip(list1, list2)]


def reduce(fn: Callable, iterable: list, initial_value: float) -> float:
    """Reduce iterable to single value by repeatedly applying `fn`."""
    result = initial_value
    for x in iterable:
        result = fn(result, x)
    return result


def negList(lst: list) -> list:
    """Negate all elements in a list."""
    return map(neg, lst)


def addLists(lst1: list, lst2: list) -> list:
    """Add corresponding elements from two lists."""
    return zipWith(add, lst1, lst2)


def sum(lst: list) -> float:
    """Sum all elements in a list."""
    return reduce(add, lst, 0)


def prod(lst: list) -> float:
    """Calculate product of all elements in a list."""
    return reduce(mul, lst, 1)
