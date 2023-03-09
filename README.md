# Math Module
My own version of the Python Maths Module

# Algorithm Class
This is a Python class that implements various algorithms, including finding the greatest common divisor using Euclid's algorithm, computing logarithms with a specified base, counting the occurrences of words in a text, and performing operations with fractions.

addition(*args: Union[int, float]) -> Union[int, float]

    Returns the sum of integers and/or floats.

subtract(*args: Union[int, float]) -> Union[int, float]

    Returns integers or float of given numbers after being subtracted.

multiply(*args: Union[int, float]) -> Union[int, float]

    Returns an integer or float of given numbers multiplied.

division_float(dividend: Union[int, float], divisor: Union[int, float]) -> Union[int, float]

    Returns a float of dividend divided by divisor.

division_int(dividend: Union[int, float], divisor: Union[int, float]) -> Union[int, float]

    Returns an integer of dividend divided by divisor.

division_remainder(dividend: Union[int, float], divisor: Union[int, float]) -> Union[int, float]

    Returns the remainder of dividend divided by divisor.

power(base: Union[int, float], exponent: Union[int, float]) -> Union[int, float]

    Returns base to the power of exponent.
    
log(self, x, base=10)

Returns the logarithm of x with a specified base (default is 10)

    Args:
        x (int/float): The value for which to compute the logarithm
        base (int/float, optional): The base of the logarithm. Defaults to 10.
    Returns:
        The logarithm of x with the specified base

__ln(self, x)

Returns the natural logarithm of x (base e)

    Args:
        x (int/float): The value for which to compute the natural logarithm

    Returns:
        The natural logarithm of x

__log10(self, x)

Returns the logarithm of x (base 10)

    Args:
        x (int/float): The value for which to compute the logarithm

    Returns:
        The logarithm of x (base 10)

adding_fractions(self, *args)

Returns the sum of multiple fractions

    Args:
        *args (tuples): Multiple fractions represented as tuples of the form (numerator, denominator)

    Returns:
        A tuple representing the sum of all fractions in reduced form (numerator, denominator)

find_gcd(self, a, b)

Finds the greatest common divisor of two numbers using Euclid's algorithm.

    Args:
        a: An integer
        b: Another integer

    Returns:
        The greatest common divisor of a and b

count_words(text)

Returns a dictionary containing the count of each word in the given text.

    Args:
        text (str): The text to count the words in.

    Returns:
        A dictionary where the keys are the unique words in the text and the values are the count of each word.

multiplying_fractions(self, *args)

Returns the product of multiple fractions.

    Args:
        *args: An arbitrary number of arguments. Each argument must be a tuple with two values, the numerator and denominator of a fraction.

    Returns:
        A tuple containing the numerator and denominator of the product of the fractions.

    Raises:
        ValueError: If any of the arguments are not tuples of length 2 or if any of the denominators are 0.

divide_fractions(self, *args: tuple[int, int]) -> tuple[int, int]

Returns the result of dividing one fraction by another.

    Args:
        *args: Two tuples, each with two values, representing the numerator and denominator of the two fractions.

    Returns:
        A tuple containing the numerator and denominator of the quotient of the two fractions.

    Raises:
        ValueError: If any of the arguments are not tuples of length 2.
        ZeroDivisionError: If the denominator of the second fraction is zero.
        
proportion_rule(a: int, b: int, c: int = None, d: int = None) -> int

    This function returns the fourth proportional number given three proportional numbers. The function can be used in three different ways. First, if c and d are both None, then a and b are assumed to be the first two proportional numbers, and c and d are set to b and a respectively. Second, if d is None, then a and b are assumed to be the first two proportional numbers, and d is calculated from c using the formula d = (b * c) / a. Third, if c and d are both specified, then a and b are assumed to be the first two proportional numbers, and the function calculates the fourth proportional number x using the formula x = (b * d) / c.

percentage_to_fraction(x: float) -> float

    This function converts a percentage x to a fraction.

fraction_to_percentage(numerator: int, denominator: int) -> float

    This function converts a fraction given by numerator and denominator to a percentage.

linear_search(lst, target)

    This function searches for the target element in the given list lst and returns the index if found, otherwise returns -1.

binary_search(lst, target)

    This function searches for the target element in the given list lst using binary search and returns the index if found, otherwise returns -1.

bubble_sort(lst)

    This function sorts the given list lst in ascending order using bubble sort and returns the sorted list.

insertion_sort(lst)

    This function sorts the given list lst in ascending order using insertion sort and returns the sorted list.

merge_sort(lst)

    This function sorts the given list lst in ascending order using merge sort and returns the sorted list.

square_root(num)

    This function computes the square root of a given number num using the Babylonian method.

factorial(num):

    This function computes the factorial of a given number num.

fibonacci(n)

Compute the nth number in the Fibonacci sequence.
Arguments

    n (int): The index of the desired Fibonacci number.

Returns

    int: The nth number in the Fibonacci sequence.

is_prime(num)

Check whether a given number is prime.
Arguments

    num (int): The number to check for primality.

Returns

    bool: True if the number is prime, False otherwise.

gcd(*args)

Compute the greatest common divisor of two or more numbers.
Arguments

    *args (int): Two or more numbers to find the GCD of.

Returns

    int: The greatest common divisor of the given numbers.

lcm(*args)

Compute the least common multiple of two or more numbers.
Arguments

    *args (int): Two or more numbers to find the LCM of.

Returns

    int: The least common multiple of the given numbers.

sort_numbers(numbers: List[Union[int, float]], reverse: bool = False) -> List[Union[int, float]]

This function takes a list of numbers and returns a sorted list in ascending or descending order.
Arguments

    numbers (List[Union[int, float]]): A list of integers or floats to be sorted.
    reverse (bool, optional): If True, returns the list in descending order. Defaults to False.

Returns

    List[Union[int, float]]: A sorted list in ascending or descending order.

binary_search(numbers: List[Union[int, float]], target: Union[int, float]) -> int

This function takes a sorted list of numbers and a target number and returns the index of the target number, or -1 if it is not found.
Arguments

    numbers (List[Union[int, float]]): A sorted list of integers or floats.
    target (Union[int, float]): The number to search for in the list.

Returns

    int: The index of the target number in the list, or -1 if it is not found.

linear_regression(x, y)

Calculates the equation of the line of best fit (y = mx + b) for the given x and y values.
Arguments

    x (list): A list of x values.
    y (list): A list of corresponding y values.

Returns

    tuple: A tuple containing the slope (m) and y-intercept (b) of the line of best fit.
    
matrix_addition

This function takes in two matrices A and B of the same size, and returns their sum.

Arguments:

    A: A list of lists of floats representing the first matrix.
    B: A list of lists of floats representing the second matrix.

Returns:

    A list of lists of floats representing the sum of the matrices.

matrix_multiplication

This function multiplies two matrices A and B and returns the resulting matrix.

Arguments:

    A: The first matrix.
    B: The second matrix.

Returns:

    A list of lists of floats representing the product of matrices A and B.

matrix_inversion

This function inverts a matrix A and returns the resulting matrix.

Arguments:

    A: The matrix to be inverted.

Returns:

    A list of lists of floats representing the inverted matrix of A.



# Constant Class
This is a Python class that implements various
