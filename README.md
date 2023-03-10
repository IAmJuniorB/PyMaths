# Math Module
A library full of mathematical algorithms, constants, and functions.

# Algorithm Class
This is a Python class that implements various algorithms, including finding the greatest common divisor using Euclid's algorithm, computing logarithms with a specified base, counting the occurrences of words in a text, and performing operations with fractions.

### `addition(*args: Union[int, float]) -> Union[int, float]`
Returns the sum of integers and/or floats.

Arguments:

    *args (Union[int, float]): A variable-length argument list of integers and/or floats

Returns:

    Union[int, float]: The sum of the integers and/or floats in *args

    Returns the sum of integers and/or floats.
---
### `subtract(*args: Union[int, float]) -> Union[int, float]`
Returns integers or float of given numbers after being subtracted.

Arguments:

    *args (Union[int, float]): A variable-length argument list of integers and/or floats

Returns:

    Union[int, float]: The result of subtracting the integers and/or floats in *args from the first argument

    Returns integers or float of given numbers after being subtracted.
---
### `multiply(*args: Union[int, float]) -> Union[int, float]`
Returns an integer or float of given numbers multiplied.

Arguments:

    *args (Union[int, float]): A variable-length argument list of integers and/or floats
    
Returns:

    Union[int, float]: The product of the integers and/or floats in *args

    Returns an integer or float of given numbers multiplied.
---
### `division_float(dividend: Union[int, float], divisor: Union[int, float]) -> Union[int, float]`
Returns a float of dividend divided by divisor.

Arguments:

    dividend (Union[int, float]): The number to be divided (dividend)
    divisor (Union[int, float]): The number to divide by (divisor)
    
Returns:

    Union[int, float]: The result of dividing the dividend by the divisor, as a float

    Returns a float of dividend divided by divisor.
---
### `division_int(dividend: Union[int, float], divisor: Union[int, float]) -> Union[int, float]`
Returns an integer of dividend divided by divisor.

Arguments:

    dividend (Union[int, float]): The number to be divided (dividend)
    divisor (Union[int, float]): The number to divide by (divisor)

Returns:

    Union[int, float]: The result of dividing the dividend by the divisor, rounded down to the nearest integer

    Returns an integer of dividend divided by divisor.
---
### `division_remainder(dividend: Union[int, float], divisor: Union[int, float]) -> Union[int, float]`
Returns the remainder of dividend divided by divisor.

Arguments:

    dividend (Union[int, float]): The number to be divided (dividend)
    divisor (Union[int, float]): The number to divide by (divisor)

Returns:

    Union[int, float]: The remainder of dividing the dividend by the divisor

    Returns the remainder of dividend divided by divisor.
---
### `power(base: Union[int, float], exponent: Union[int, float]) -> Union[int, float]`
Returns base to the power of exponent.

Arguments:

    base (Union[int, float]): The base of the power operation
    exponent (Union[int, float]): The exponent of the power operation

Returns:

    Union[int, float]: The result of raising the base to the power of the exponent

    Returns base to the power of exponent.
---
### `log(self, x, base=10)`

Returns the logarithm of x with a specified base (default is 10)

Arguments:
    
        x (int/float): The value for which to compute the logarithm
        base (int/float, optional): The base of the logarithm. Defaults to 10.

Returns:

        Union[int, float]: The logarithm of x with the specified base
        The logarithm of x with the specified base
---
### `__ln(self, x)`

Returns the natural logarithm of x (base e)

    Args:
        x (int/float): The value for which to compute the natural logarithm

    Returns:
        The natural logarithm of x
---
### `__log10(self, x)`

Returns the logarithm of x (base 10)

    Args:
        x (int/float): The value for which to compute the logarithm

    Returns:
        The logarithm of x (base 10)
---
### `adding_fractions(self, *args)`

Returns the sum of multiple fractions

    Args:
        *args (tuples): Multiple fractions represented as tuples of the form (numerator, denominator)

    Returns:
        A tuple representing the sum of all fractions in reduced form (numerator, denominator)
---
### `find_gcd(self, a, b)`

Finds the greatest common divisor of two numbers using Euclid's algorithm.

    Args:
        a: An integer
        b: Another integer

    Returns:
        The greatest common divisor of a and b
---
### `count_words(text)`

Returns a dictionary containing the count of each word in the given text.

    Args:
        text (str): The text to count the words in.

    Returns:
        A dictionary where the keys are the unique words in the text and the values are the count of each word.
---
### `multiplying_fractions(self, *args)`

Returns the product of multiple fractions.

    Args:
        *args: An arbitrary number of arguments. Each argument must be a tuple with two values, the numerator and denominator of a fraction.

    Returns:
        A tuple containing the numerator and denominator of the product of the fractions.

    Raises:
        ValueError: If any of the arguments are not tuples of length 2 or if any of the denominators are 0.
---
### `divide_fractions(self, *args: tuple[int, int]) -> tuple[int, int]`

Returns the result of dividing one fraction by another.

    Args:
        *args: Two tuples, each with two values, representing the numerator and denominator of the two fractions.

    Returns:
        A tuple containing the numerator and denominator of the quotient of the two fractions.

    Raises:
        ValueError: If any of the arguments are not tuples of length 2.
        ZeroDivisionError: If the denominator of the second fraction is zero.
---        
### `proportion_rule(a: int, b: int, c: int = None, d: int = None) -> int`

    This function returns the fourth proportional number given three proportional numbers. The function can be used in three different ways. First, if c and d are both None, then a and b are assumed to be the first two proportional numbers, and c and d are set to b and a respectively. Second, if d is None, then a and b are assumed to be the first two proportional numbers, and d is calculated from c using the formula d = (b * c) / a. Third, if c and d are both specified, then a and b are assumed to be the first two proportional numbers, and the function calculates the fourth proportional number x using the formula x = (b * d) / c.
---
### `percentage_to_fraction(x: float) -> float`

This function converts a percentage `x` to a fraction.
---
### `fraction_to_percentage(numerator: int, denominator: int) -> float`

This function converts a fraction given by `numerator` and `denominator` to a percentage.
---
### `linear_search(lst, target)`

This function searches for the `target` element in the given list `lst` and returns the index if found, otherwise returns -1.
---
### `binary_search(lst, target)`

This function searches for the `target` element in the given list `lst` using binary search and returns the index if found, otherwise returns -1.
---
### `bubble_sort(lst)`

This function sorts the given list `lst` in ascending order using bubble sort and returns the sorted list.
---
### `insertion_sort(lst)`

This function sorts the given list `lst` in ascending order using insertion sort and returns the sorted list.
---
### `merge_sort(lst)`

This function sorts the given list `lst` in ascending order using merge sort and returns the sorted list.
---
### `square_root(num)`

This function computes the square root of a given number `num` using the Babylonian method.
---
### `factorial(num)`

This function computes the factorial of a given number `num`.
---
### `fibonacci(n)`

Compute the nth number in the Fibonacci sequence.

Arguments:

    n (int): The index of the desired Fibonacci number.

Returns:

    int: The nth number in the Fibonacci sequence.
---
### `is_prime(num)`

Check whether a given number is prime.

Arguments:

    num (int): The number to check for primality.

Returns:

    bool: True if the number is prime, False otherwise.
---
### `gcd(*args)`

Compute the greatest common divisor of two or more numbers.

Arguments:

    *args (int): Two or more numbers to find the GCD of.

Returns:

    int: The greatest common divisor of the given numbers.
---
### `lcm(*args)`

Compute the least common multiple of two or more numbers.

Arguments:

    *args (int): Two or more numbers to find the LCM of.

Returns:

    int: The least common multiple of the given numbers.
---
### `sort_numbers(numbers: List[Union[int, float]], reverse: bool = False) -> List[Union[int, float]]`

This function takes a list of numbers and returns a sorted list in ascending or descending order.

Arguments:

    numbers (List[Union[int, float]]): A list of integers or floats to be sorted.
    reverse (bool, optional): If True, returns the list in descending order. Defaults to False.

Returns:

    List[Union[int, float]]: A sorted list in ascending or descending order.
---
### `binary_search(numbers: List[Union[int, float]], target: Union[int, float]) -> int`

This function takes a sorted list of numbers and a target number and returns the index of the target number, or -1 if it is not found.

Arguments:

    numbers (List[Union[int, float]]): A sorted list of integers or floats.
    target (Union[int, float]): The number to search for in the list.

Returns:

    int: The index of the target number in the list, or -1 if it is not found.
---
### `linear_regression(x, y)`

Calculates the equation of the line of best fit (y = mx + b) for the given x and y values.

Arguments:

    x (list): A list of x values.
    y (list): A list of corresponding y values.

Returns:

    tuple: A tuple containing the slope (m) and y-intercept (b) of the line of best fit.
---    
### `matrix_addition`

This function takes in two matrices A and B of the same size, and returns their sum.

Arguments:

    A: A list of lists of floats representing the first matrix.
    B: A list of lists of floats representing the second matrix.

Returns:

    A list of lists of floats representing the sum of the matrices.
---
### `matrix_multiplication`

This function multiplies two matrices A and B and returns the resulting matrix.

Arguments:

    A: The first matrix.
    B: The second matrix.

Returns:

    A list of lists of floats representing the product of matrices A and B.
---
### `matrix_inversion`

This function inverts a matrix A and returns the resulting matrix.

Arguments:

    A: The matrix to be inverted.

Returns:

    A list of lists of floats representing the inverted matrix of A.
---
### `matrix_multiplication(A, B):`

Multiplies two matrices A and B and returns the resulting matrix.

Arguments:

        A (list[list[float]]): The first matrix.
        B (list[list[float]]): The second matrix.

Returns:

        list[list[float]]: The matrix product of A and B.

---
### `matrix_inversion(A):'

Inverts a matrix A and returns the resulting matrix.

Arguments:

        A (list[list[float]]): The matrix to be inverted.

Returns:

        list[list[float]]: The inverted matrix of A.
        
---        
### `newton_method(self, f, f_prime, x0, epsilon):`

Use Newton's method to find the root of a function f.

Arguments:
        
        - f (function): The function for which to find the root.
        - f_prime (function): The derivative of f.
        - x0 (float): The initial guess for the root.
        - epsilon (float): The desired level of accuracy.

Returns:
        
        - root (float): The estimated root of the function.
 ---   
### `gradient_descent(self, f, f_prime, x0, alpha, max_iters):`

Use gradient descent to find the minimum of a function f.

Arguments:
        
        - f (function): The function to minimize.
        - f_prime (function): The derivative of f.
        - x0 (float): The initial guess for the minimum.
        - alpha (float): The step size.
        - max_iters (int): The maximum number of iterations.

        Returns:
        - minimum (float): The estimated minimum of the function.

---   
### `monte_carlo_simulation(self, n, f):`

Use Monte Carlo simulation to estimate the probability of an event.

Arguments:
        
        - n (int): The number of simulations to run.
        - f (function): A function that returns True or False for a given sample.

Returns:
        
        - probability (float): The estimated probability of the event.
---
### `distance(point1, point2):`

Arguments:
        
        - point1 (int/float): The position of point 1
        - point2 (int/float): The position of point 2

Returns:

        The distance between the two points
---
### `random_seed(seed):`

A simple pseudorandom number generator based on the linear congruential method.

Arguments:

        - seed (int): The seed value used to initialize the generator.

Returns:

        - A float between 0 and 1.
---
### `k_means_clustering(self, data, k):`

Use k-means clustering to group data points into k clusters.

Arguments:

        - data (list): A list of data points.
        - k (int): The number of clusters to form.

Returns:

        - clusters (list): A list of k clusters, each containing the data points assigned to that cluster.
---
### `exp(self, num: Union[int, float]) -> Union[int, float]:`

Returns the exponential value of a number.

Arguments:

        - num: a number whose exponential value is to be calculated

Returns:

        The exponential value of the input number
---
### `absolute(self, num: Union[int, float]) -> Union[int, float]:`

Returns the absolute value of a number.

Arguments:

        - num: a number whose absolute value is to be calculated

Returns:

        The absolute value of the input number
---
### `modulo(self, dividend: Union[int, float], divisor: Union[int, float]) -> Union[int, float]:`

Returns the remainder of dividing the dividend by the divisor.

Arguments:
        - dividend: the number to be divided
        - divisor: the number to divide by

Returns:
        
        The remainder of dividing the dividend by the divisor
---
### `sin(self, num: Union[int, float]) -> Union[int, float]:`

Returns the sine value of a number.

Arguments:

        - num: a number in radians whose sine value is to be calculated

Returns:

        The sine value of the input number
---
### `cos(self, num: Union[int, float]) -> Union[int, float]:`

Returns the cosine value of a number.

Arguments:

        - num: a number in radians whose cosine value is to be calculated

Returns:

        The cosine value of the input number

---
### `tan(self, num: Union[int, float]) -> Union[int, float]:`

Returns the tangent value of a number.

Arguments:

        - num: a number in radians whose tangent value is to be calculated

Returns:
        
        The tangent value of the input number




# Constants Class
This is a Python class full of mathematical constants such a Pi or the speed of light.

### `speed_of_light(self):`
Returns the speed of light in meters per second

Arguments:

        - None

Returns:

        The speed of light in meters/second at 299_792_458
---
### `planck_constant(self):`
        pass
    
### `pi(self):`
The ratio of a circle's circumference to its diameter.

Arguments:

        - None
        
Returns:
        Pi, π, to the 20th decimal
        3.141_592_653_589_793_238_46
---
### `tau(self):`
the 19th letter of the Greek alphabet,
representing the voiceless dental or alveolar plosive IPA: [t].
In the system of Greek numerals, it has a value of 300.

Arguments:

        - None
        
Returns:
        tau, uppercase Τ, lowercase τ, or τ, to the 20th decimal
        6.283_185_307_179_586_476_92
---
### `phi(self):`
"The Golden Ratio"
In mathematics, two quantities are in the golden ratio
if their ratio is the same as the ratio of their sum
to the larger of the two quantities.

Arguments:

        - None

Returns:

        Uppercase Φ lowercase φ or ϕ: Value to the 20th decimal
        1.618_033_988_749_894_848_20
---
### `silver_ratio(self):`
"The Silver Ratio". Two quantities are in the silver ratio (or silver mean)
if the ratio of the smaller of those two quantities to the larger quantity
is the same as the ratio of the larger quantity to the sum of the
smaller quantity and twice the larger quantity

Arguments:

        - None
        
Returns:

        δS: Value to the 20th decimal
        2.414_213_562_373_095_048_80
---
### `supergolden_ratio(self):`
Returns the mathematical constant psi (the supergolden ratio).

Arguments:

        - None

Returns:
        
        ψ to the 25th decimal
        return 1.465_571_231_876_768_026_656_731_2
---
### `connective_constant(self):`
Returns the connective constant for the hexagonal lattice.

Arguments:

        - None
        
Returns:

        μ to the 4th decimal
        1.687_5
---
### `kepler_bouwkamp_constant(self):`
In plane geometry, the Kepler–Bouwkamp constant (or polygon inscribing constant)
is obtained as a limit of the following sequence.
Take a circle of radius 1. Inscribe a regular triangle in this circle.
Inscribe a circle in this triangle. Inscribe a square in it.
Inscribe a circle, regular pentagon, circle, regular hexagon and so forth.

Arguments:

        - None
        
Returns:

        K': to the 20th decimal
        0.114_942_044_853_296_200_70
