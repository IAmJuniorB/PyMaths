# **&Popf;&yopf;&Mopf;&aopf;&topf;&hopf;&sopf;**
══════════════════════
> A work in progress, incomplete, library full of mathematical algorithms, constants, and functions.

## Table of Contents

- [Algorithms](#algorithm-class)
- [Constants](#constants-class)
- [Functions](#functions-class)
- [Sequences](#sequences-class)
- [Hyperbolic Functions](#hyperbolicfunctions-class)
- [Complex Number](#complexnumber-class)
- [Real Number](#realnumber-class)
- [Rational Number](#rationalnumber-class)
- [Integral Number](#integralnumber-class)

---
## **Algorithm Class**
This is a Python class that implements various algorithms, including finding the greatest common divisor using Euclid's algorithm, computing logarithms with a specified base, counting the occurrences of words in a text, and performing operations with fractions.

### `addition Union[int, - loat]) -> Union[int, float]`
Returns the sum of integers and/or floats.

> ##### **Arguments:**

    *args (Union[int, float]): A variable-length argument list of integers and/or floats

> ##### **Returns:**

    Union[int, float]: The sum of the integers and/or floats in *args

---
### `subtract(int | float) -> int | float`
Returns integers or float of given numbers after being subtracted.

> ##### **Arguments:**

    *args (Union[int, float]): A variable-length argument list of integers and/or floats

> ##### **Returns:**

    Union[int, float]: The result of subtracting the integers and/or floats in *args from the first argument

---
### `multiply(int | float) -> int | float`
Returns an integer or float of given numbers multiplied.

> ##### **Arguments:**

    *args (Union[int, float]): A variable-length argument list of integers and/or floats
    
> ##### **Returns:**

    Union[int, float]: The product of the integers and/or floats in *args

---
### `division_float(dividend: Union[int, float], divisor: Union[int, float]) -> Union[int, float]`
Returns a float of dividend divided by divisor.

> ##### **Arguments:**

    dividend (Union[int, float]): The number to be divided (dividend)

    divisor (Union[int, float]): The number to divide by (divisor)
    
> ##### **Returns:**

    Union[int, float]: The result of dividing the dividend by the divisor, as a float

---
### `division_int(dividend: Union[int, float], divisor: Union[int, float]) -> Union[int, float]`

Returns an integer of dividend divided by divisor.

> ##### **Arguments:**

    dividend (Union[int, float]): The number to be divided (dividend)
    divisor (Union[int, float]): The number to divide by (divisor)

> ##### **Returns:**

    Union[int, float]: The result of dividing the dividend by the divisor, rounded down to the nearest integer

---
### `division_remainder(dividend: Union[int, float], divisor: Union[int, float]) -> Union[int, float]`
Returns the remainder of dividend divided by divisor.

> ##### **Arguments:**

    dividend (Union[int, float]): The number to be divided (dividend)
    divisor (Union[int, float]): The number to divide by (divisor)

> ##### **Returns:**

    Union[int, float]: The remainder of dividing the dividend by the divisor

---
### `power(base: Union[int, float], exponent: Union[int, float]) -> Union[int, float]`
Returns base to the power of exponent.

> ##### **Arguments:**

    base (Union[int, float]): The base of the power operation
    exponent (Union[int, float]): The exponent of the power operation

> ##### **Returns:**

    Union[int, float]: The result of raising the base to the power of the exponent

---
### `log(self, x, base=10)`

Returns the logarithm of x with a specified base (default is 10)

> ##### **Arguments:**
    
        x (int/float): The value for which to compute the logarithm
        base (int/float, optional): The base of the logarithm. Defaults to 10.

> ##### **Returns:**

        Union[int, float]: The logarithm of x with the specified base

---
### `__ln(self, x)`

Returns the natural logarithm of x (base e)

> ##### **Arguments:**

        x (int/float): The value for which to compute the natural logarithm

> ##### **Returns:**

        The natural logarithm of x
---
### `__log10(self, x)`

Returns the logarithm of x (base 10)

> ##### **Arguments:**

        x (int/float): The value for which to compute the logarithm

> ##### **Returns:**

        The logarithm of x (base 10)
        
---
### `adding_fractions(self, *args)`

Returns the sum of multiple fractions

> ##### **Arguments:**

        *args (tuples): Multiple fractions represented as tuples of the form (numerator, denominator)

> ##### **Returns:**

        A tuple representing the sum of all fractions in reduced form (numerator, denominator)
        
---
### `find_gcd(self, a, b)`

Finds the greatest common divisor of two numbers using Euclid's algorithm.

> ##### **Arguments:**

        a: An integer
        b: Another integer

> ##### **Returns:**

        The greatest common divisor of a and b
        
---
### `count_words(text)`

Returns a dictionary containing the count of each word in the given text.

> ##### **Arguments:**

        text (str): The text to count the words in.

> ##### **Returns:**

        A dictionary where the keys are the unique words in the text and the values are the count of each word.

---
### `multiplying_fractions(self, *args)`

Returns the product of multiple fractions.

> ##### **Arguments:**

    #### **arguments:** An arbitrar-  number of arguments. Each argument must be a tuple with two values, the numerator and denominator of a fraction.

> ##### **Returns:**

        A tuple containing the numerator and denominator of the product of the fractions.

> ##### **Raises:**

        ValueError: If any of the arguments are not tuples of length 2 or if any of the denominators are 0.

---
### `divide_fractions(self,#### **arguments:** tuple[int, - nt]) -> tuple[int, int]`

Returns the result of dividing one fraction by another.

> ##### **Arguments:**

    #### **arguments:** Two tuples,- each with two values, representing the numerator and denominator of the two fractions.

> ##### **Returns:**

        A tuple containing the numerator and denominator of the quotient of the two fractions.

> ##### **Raises:**

        ValueError: If any of the arguments are not tuples of length 2.
        ZeroDivisionError: If the denominator of the second fraction is zero.
        
---        
### `proportion_rule(a: int, b: int, c: int = None, d: int = None) -> int`

Returns the fourth proportional number given three proportional numbers.
    
> ##### **Arguments:**

        a (int): The first proportional number.
        b (int): The second proportional number.
        c (int, optional): The third proportional number. Defaults to None.
        d (int, optional): The fourth proportional number. Defaults to None.
        
> ##### **Returns:**

        int: The fourth proportional number calculated from the input.
        
        If both `c` and `d` are None, `a` and `b` are assumed to be the first two proportional numbers, and `c` and `d` are set to `b` and `a` respectively. If `d` is None, `a` and `b` are assumed to be the first two proportional numbers, and `d` is calculated from `c` using the formula `d = (b * c) / a`. If `c` and `d` are both specified, `a` and `b` are assumed to be the first two proportional numbers, and the function calculates the fourth proportional number `x` using the formula `x = (b * d) / c`.
        
---
### `percentage_to_fraction(x: float) -> float`

This function converts a percantage `x` to a fraction.

> ##### **Arguments:**

      x (float): percentage
            
> ##### **Returns:**

      The fraction form of a percentage

---
### `fraction_to_percentage(numerator: int, denominator: int) -> float`

This function converts a fraction given by `numerator` and `denominator` to a percentage.

> ##### **Arguments:**

      numerator: The numerator of the fraction.
      denominator: The denominator of the fraction.

> ##### **Returns:**

      The fraction as a percentage.
            
---
### `linear_search(lst, target)`

Searches for the target element in the given list and returns the index if found,
otherwise returns -1.

> ##### **Arguments:**

    - lst : list
        The list to be searched.

    - target : any
        The target element to be searched for in the list.

> ##### **Returns:**

    - int
        If the target is found in the list, the index of the target is returned.
        Otherwise, -1 is returned.

---
### `binary_search(lst, target)`

Searches for the target element in the given list using binary search and returns
the index if found, otherwise returns -1.

> ##### **Arguments:**

        - lst : list
            The list to be searched.

        - target : any
            The target element to be searched for in the list.

> ##### **Returns:**

        - int
            If the target is found in the list, the index of the target is returned.
            Otherwise, -1 is returned.

---
### `bubble_sort(lst)`

Sorts the given list in ascending order using bubble sort and returns the sorted list.

> ##### **Arguments:**

        - lst (list): The list to be sorted.

> ##### **Returns:**

        - list: The sorted list in ascending order.
        
---
### `insertion_sort(lst)`

Sorts the given list in ascending order using insertion sort and returns the sorted list.

> ##### **Arguments:**

        - lst (list): The list to be sorted.

> ##### **Returns:**

        - list: The sorted list in ascending order.

---
### `merge_sort(lst)`

Sorts the given list in ascending order using merge sort and returns the sorted list.

> ##### **Arguments:**

      - lst (list): The list to be sorted.

> ##### **Returns:**

      - list: The sorted list in ascending order.

---
### `square_root(num)`

This function computes the square root of a given number `num` using the Babylonian method.

> ##### **Arguments:**

      num (float): The number to find the square root of.

> ##### **Returns:**

      float: The square root of the given number.

---
### `factorial(num)`

This function computes the factorial of a given number `num`.

> ##### **Arguments:**
      
      num (int): The number to find the factorial of.

> ##### **Returns:**

      int: The factorial of the given number.
        
---
### `fibonacci(n)`

Compute the nth number in the Fibonacci sequence.

> ##### **Arguments:**

    n (int): The index of the desired Fibonacci number.

> ##### **Returns:**

    int: The nth number in the Fibonacci sequence.
    
---
### `is_prime(num)`

Check whether a given number is prime.

> ##### **Arguments:**

    num (int): The number to check for primality.

> ##### **Returns:**

    bool: True if the number is prime, False otherwise.
    
---
### `gcd(*args)`

Compute the greatest common divisor of two or more numbers.

> ##### **Arguments:**

    *args (int): Two or more numbers to find the GCD of.

> ##### **Returns:**

    int: The greatest common divisor of the given numbers.
    
---
### `lcm(*args)`

Compute the least common multiple of two or more numbers.

> ##### **Arguments:**

    *args (int): Two or more numbers to find the LCM of.

> ##### **Returns:**

    int: The least common multiple of the given numbers.
    
---
### `sort_numbers(numbers: List[Union[int, float]], reverse: bool = False) -> List[Union[int, float]]`

This function takes a list of numbers and returns a sorted list in ascending or descending order.

> ##### **Arguments:**

    numbers (List[Union[int, float]]): A list of integers or floats to be sorted.
    reverse (bool, optional): If True, returns the list in descending order. Defaults to False.

> ##### **Returns:**

    List[Union[int, float]]: A sorted list in ascending or descending order.
    
---
### `binary_search(numbers: List[Union[int, float]], target: Union[int, float]) -> int`

This function takes a sorted list of numbers and a target number and returns the index of the target number, or -1 if it is not found.

> ##### **Arguments:**

    numbers (List[Union[int, float]]): A sorted list of integers or floats.
    target (Union[int, float]): The number to search for in the list.

> ##### **Returns:**

    int: The index of the target number in the list, or -1 if it is not found.
    
---
### `linear_regression(x, y)`

Calculates the equation of the line of best fit (y = mx + b) for the given x and y values.

> ##### **Arguments:**

    x (list): A list of x values.
    y (list): A list of corresponding y values.

> ##### **Returns:**

    tuple: A tuple containing the slope (m) and y-intercept (b) of the line of best fit.
    
---    
### `matrix_addition`

This function takes in two matrices A and B of the same size, and returns their sum.

> ##### **Arguments:**

    A: A list of lists of floats representing the first matrix.
    B: A list of lists of floats representing the second matrix.

> ##### **Returns:**

    A list of lists of floats representing the sum of the matrices.
    
---
### `matrix_multiplication`

This function multiplies two matrices A and B and returns the resulting matrix.

> ##### **Arguments:**

    A: The first matrix.
    B: The second matrix.

> ##### **Returns:**

    A list of lists of floats representing the product of matrices A and B.
    
---
### `matrix_inversion`

This function inverts a matrix A and returns the resulting matrix.

> ##### **Arguments:**

    A: The matrix to be inverted.

> ##### **Returns:**

    A list of lists of floats representing the inverted matrix of A.
    
---
### `matrix_multiplication(A, B):`

Multiplies two matrices A and B and returns the resulting matrix.

> ##### **Arguments:**

        A (list[list[float]]): The first matrix.
        B (list[list[float]]): The second matrix.

> ##### **Returns:**

        list[list[float]]: The matrix product of A and B.

---
### `matrix_inversion(A):`

Inverts a matrix A and returns the resulting matrix.

> ##### **Arguments:**

        A (list[list[float]]): The matrix to be inverted.

> ##### **Returns:**

        list[list[float]]: The inverted matrix of A.
        
---        
### `newton_method(self, f, f_prime, x0, epsilon):`

Use Newton's method to find the root of a function f.

> ##### **Arguments:**
        
        - f (function): The function for which to find the root.
        - f_prime (function): The derivative of f.
        - x0 (float): The initial guess for the root.
        - epsilon (float): The desired level of accuracy.

> ##### **Returns:**
        
        - root (float): The estimated root of the function.
        
 ---   
### `gradient_descent(self, f, f_prime, x0, alpha, max_iters):`

Use gradient descent to find the minimum of a function f.

> ##### **Arguments:**
        
        - f (function): The function to minimize.
        - f_prime (function): The derivative of f.
        - x0 (float): The initial guess for the minimum.
        - alpha (float): The step size.
        - max_iters (int): The maximum number of iterations.

> ##### **Returns:**

        - minimum (float): The estimated minimum of the function.

---   
### `monte_carlo_simulation(self, n, f):`

Use Monte Carlo simulation to estimate the probability of an event.

> ##### **Arguments:**
        
        - n (int): The number of simulations to run.
        - f (function): A function that returns True or False for a given sample.

> ##### **Returns:**
        
        - probability (float): The estimated probability of the event.
        
---
### `distance(point1, point2):`

Calculates the Euclidean distance between two points in a two-dimensional space.

> ##### **Arguments:**
        
        - point1 (tuple): A tuple containing the coordinates of the first point as (x, y).
        - point2 (tuple): A tuple containing the coordinates of the second point as (x, y).

> ##### **Returns:**

        - float: The Euclidean distance between point1 and point2.

> ##### **Returns:**

        The distance between the two points
        
---
### `random_seed(seed):`

A simple pseudorandom number generator based on the linear congruential method.

> ##### **Arguments:**

        - seed (int): The seed value used to initialize the generator.

> ##### **Returns:**

        - A float between 0 and 1.
        
---
### `k_means_clustering(self, data, k):`

Use k-means clustering to group data points into k clusters.

> ##### **Arguments:**

        - data (list): A list of data points.
        - k (int): The number of clusters to form.

> ##### **Returns:**

        - clusters (list): A list of k clusters, each containing the data points assigned to that cluster.
        
---
### `exp(self, num: Union[int, float]) -> Union[int, float]:`

Returns the exponential value of a number.

> ##### **Arguments:**

        - num: a number whose exponential value is to be calculated

> ##### **Returns:**

        The exponential value of the input number
        
---
### `absolute(self, num: Union[int, float]) -> Union[int, float]:`

Returns the absolute value of a number.

> ##### **Arguments:**

        - num: a number whose absolute value is to be calculated

> ##### **Returns:**

        The absolute value of the input number
        
---
### `modulo(self, dividend: Union[int, float], divisor: Union[int, float]) -> Union[int, float]:`

Returns the remainder of dividing the dividend by the divisor.

> ##### **Arguments:**

        - dividend: the number to be divided
        - divisor: the number to divide by

> ##### **Returns:**
        
        The remainder of dividing the dividend by the divisor
        
---
### `sin(self, num: Union[int, float]) -> Union[int, float]:`

Returns the sine value of a number.

> ##### **Arguments:**

        - num: a number in radians whose sine value is to be calculated

> ##### **Returns:**

        The sine value of the input number
        
---
### `cos(self, num: Union[int, float]) -> Union[int, float]:`

Returns the cosine value of a number.

> ##### **Arguments:**

        - num: a number in radians whose cosine value is to be calculated

> ##### **Returns:**

        The cosine value of the input number

---
### `tan(self, num: Union[int, float]) -> Union[int, float]:`

Returns the tangent value of a number.

> ##### **Arguments:**

        - num: a number in radians whose tangent value is to be calculated

> ##### **Returns:**
        
        The tangent value of the input number


### `def next_prime(n):`

Finds the smallest prime number greater than n.

> #### **Arguments:**
      - n (int): A positive integer.

> #### **Returns:**

      - int: The smallest prime number greater than n.
            

### `def atan(x):`

Return the arc tangent of x, in radians.

> #### ***Arguments:**

      - x (float): The value whose arc tangent is to be returned.

> #### **Returns:**

      - float: The arc tangent of x, in radians.

### `def atan_helper(x):`

Helper function for atan. Computes the arc tangent of x in the interval [0, 1].

> #### **Arguments:**

      - x (float): The value whose arc tangent is to be returned.

> #### **Returns:**

      - float: The arc tangent of x, in radians.


### `def arctan(x):`

Calculates the arctangent of x using a Taylor series approximation.

> #### **Arguments:**

        - x (float): A real number.

> #### **Returns:**

        - float: The arctangent of x in radians.

        
### `def sieve_of_eratosthenes(n: int) -> List[int]:`

Returns a list of prime numbers up to n using the sieve of Eratosthenes algorithm.
        
> #### **Arguments:**

        - n (int): the upper limit of the list.
            
> #### **Returns:**

        - List[int]: a list of prime numbers up to n.

    
### `def zeta(s, zeta_1, n):`

Returns the value of the Riemann zeta function.

> #### **Arguments:**

        - s (float): The argument of the zeta function.
        - zeta_1 (complex): The initial value of the Riemann zeta function.
        - n (int): The number of iterations to perform.

> #### **Returns:**

        - complex: The value of the Riemann zeta function.
        

### `def histogram(data, num_bins):`

Compute the histogram of a list of data with a specified number of bins.

> #### **Arguments:**

        - data (list): A list of numeric data
        - num_bins (int): The number of bins to use in the histogram

> #### **Returns:**

        - tuple: A tuple containing the counts for each bin and the edges of the bins


### `def islice(iterable, start, stop, step=1):`

Returns an iterator that produces a slice of elements from the given iterable.

> #### **Arguments:**

        - iterable (iterable): The iterable to slice.
        - start (int): The index at which to start the slice.
        - stop (int): The index at which to stop the slice.
        - step (int, optional): The step size between slice elements. Defaults to 1.

> #### **Returns:**

        - iterator: An iterator that produces the slice of elements.

### `def normal_distribution_cdf(x):`

Calculates the cumulative distribution function (CDF) of a standard normal distribution at a given value.

> #### **Arguments:**

        - x (float): The value at which to calculate the CDF.

> #### **Returns:**

        - float: The CDF of the standard normal distribution at x, accurate to 10 decimal places.



---
## **Constants Class**

This is a Python class full of mathematical constants such a Pi or the speed of light.

### `speed_of_light(self):`

Returns the speed of light in meters per second

> ##### **Arguments:**

        - None

> ##### **Returns:**

        The speed of light in meters/second at 299_792_458
        
---
### `planck_constant(self):`

Returns the Planck constant, denoted as h, in joule-seconds.

The Planck constant is a physical constant that is fundamental to quantum mechanics.
It relates the energy of a photon to its frequency and is approximately 6.626 x 10^-34 J*s.

> ##### **Returns:**
            
            - float: The value of the Planck constant in joule-seconds.
    
### `pi(self):`

The ratio of a circle's circumference to its diameter.

> ##### **Arguments:**

        - None
        
> ##### **Returns:**

        Pi, π, to the 20th decimal
        3.141_592_653_589_793_238_46
        
---
### `tau(self):`

the 19th letter of the Greek alphabet,
representing the voiceless dental or alveolar plosive IPA: [t].
In the system of Greek numerals, it has a value of 300.

> ##### **Arguments:**

        - None
        
> ##### **Returns:**

        tau, uppercase Τ, lowercase τ, or τ, to the 20th decimal
        6.283_185_307_179_586_476_92
        
---
### `phi(self):`

"The Golden Ratio"
In mathematics, two quantities are in the golden ratio
if their ratio is the same as the ratio of their sum
to the larger of the two quantities.

> ##### **Arguments:**

        - None

> ##### **Returns:**

        Uppercase Φ lowercase φ or ϕ: Value to the 20th decimal
        1.618_033_988_749_894_848_20
        
---
### `silver_ratio(self):`

"The Silver Ratio". Two quantities are in the silver ratio (or silver mean)
if the ratio of the smaller of those two quantities to the larger quantity
is the same as the ratio of the larger quantity to the sum of the
smaller quantity and twice the larger quantity

> ##### **Arguments:**

        - None
        
> ##### **Returns:**

        δS: Value to the 20th decimal
        2.414_213_562_373_095_048_80
        
---
### `supergolden_ratio(self):`

Returns the mathematical constant psi (the supergolden ratio).

> ##### **Arguments:**

        - None

> ##### **Returns:**
        
        ψ to the 25th decimal
        return 1.465_571_231_876_768_026_656_731_2
        
---
### `connective_constant(self):`

Returns the connective constant for the hexagonal lattice.

> ##### **Arguments:**

        - None
        
> ##### **Returns:**

        μ to the 4th decimal
        1.687_5
        
---
### `kepler_bouwkamp_constant(self):`

In plane geometry, the Kepler–Bouwkamp constant (or polygon inscribing constant)
is obtained as a limit of the following sequence.
Take a circle of radius 1. Inscribe a regular triangle in this circle.
Inscribe a circle in this triangle. Inscribe a square in it.
Inscribe a circle, regular pentagon, circle, regular hexagon and so forth.

> ##### **Arguments:**

        - None
        
> ##### **Returns:**

        K': to the 20th decimal
        0.114_942_044_853_296_200_70
        
---
### `def wallis_constant(self):`

Returns Wallis's constant.

> ##### **Arguments:**

        - None
        
> ##### **Returns:**

      Value to the 20th decimal
      2.094_551_481_542_326_591_48
      
---
### `eulers_number(self):`

A mathematical constant approximately equal to 2.71828 that can be characterized in many ways.
It is the base of the natural logarithms.
It is the limit of (1 + 1/n)n as n approaches infinity, an expression that arises in the study of compound interest.
It can also be calculated as the sum of the infinite series

> ##### **Arguments:**

        - None
        
> ##### **Returns:**

      e: Value to the 20th decimal. math.e
      2.718_281_828_459_045_235_36
      
---
### `natural_log(self):`

Natural logarithm of 2.

> ##### **Arguments:**

        - None
        
> ##### **Returns:**
        
      ln 2: Value to the 30th decimal. math.log(2)
      0.693_147_180_559_945_309_417_232_121_458
      
---
### `lemniscate_constant(self):`

The ratio of the perimeter of Bernoulli's lemniscate to its diameter, analogous to the definition of π for the circle.

> ##### **Arguments:**

        - None
        
> ##### **Returns:**
        
      ϖ: Value to the 20th decimal. math.sqrt(2)
      2.622_057_554_292_119_810_46 
      
---
### `eulers_constant(self):`

Not to be confused with Euler's Number.
Defined as the limiting difference between the harmonic series and the natural logarithm

> ##### **Arguments:**

        - None
        
> ##### **Returns:**

      γ: Value to the 50th decimal
      0.577_215_664_901_532_860_606_512_090_082_402_431_042_159_335_939_92
      
---
### `Erdős_Borwein_constant(self):`

The sum of the reciprocals of the Mersenne numbers

> ##### **Arguments:**

        - None
        
> ##### **Returns:**
        
      E: Value to the 20th decimal. sum([1 / 2 ** (2 ** i) for i in range(40)])
      1.606_695_152_415_291_763_78
      
---
### `omega_constant(self):`

Defined as the unique real number that satisfies the equation Ωe**Ω = 1.

> ##### **Arguments:**

        - None
        
> ##### **Returns:**
        
      Ω: Value to the 30th decimal
      0.567_143_290_409_783_872_999_968_662_210
      
---
### `Apérys_constant(self):`

The sum of the reciprocals of the positive cubes.

> ##### **Arguments:**

        - None
        
> ##### **Returns:**
        
      ζ(3): Value to the 45th decimal
      1.202_056_903_159_594_285_399_738_161_511_449_990_764_986_292
      
---
### `laplace_limit(self):`

The maximum value of the eccentricity for which a solution to Kepler's equation, in terms of a power series in the eccentricity, converges.

> ##### **Arguments:**

        - None
        
> ##### **Returns:**

      Value to the 35th decimal
      0.662_743_419_349_181_580_974_742_097_109_252_90
      
---
### `ramanujan_soldner_constant(self):`

A mathematical constant defined as the unique positive zero of the logarithmic integral function.

> ##### **Arguments:**

        - None
        
> ##### **Returns:**

      μ ≈: Value to the 45th decimal
      1.451_369_234_883_381_050_283_968_485_892_027_449_493_032_28
      
---
### `gauss_constant(self):`

Transcendental mathematical constant that is the ratio of the perimeter of
Bernoulli's lemniscate to its diameter, analogous to the definition of π for the circle.

> ##### **Arguments:**

        - None
        
> ##### **Returns:**
        
      G == ϖ /π ≈ 0.8346268: Value to the 7th decimal
      0.834_626_8
      
---
### `second_hermite_constant(self):`

_summary_

> ##### **Arguments:**

        - None
        
> ##### **Returns:**
        
      γ2 : Value to the 20th decimal
      1.154_700_538_379_251_529_01
---
### `liouvilles_constant(self):`

A real number x with the property that, for every positive integer n,
there exists a pair of integers (p,q) with q>1.

> ##### **Arguments:**

        - None
        
> ##### **Returns:**

      L: Value to the 119th decimal
      0.110_001_000_000_000_000_000_001_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_01

---
### `first_continued_fraction(self):`

_summary_

> ##### **Arguments:**

        - None
        
> ##### **Returns:**
        
      C_{1}: _description_
      0.697_774_657_964_007_982_01
      
---
### `ramanujans_constant(self):`

The transcendental number, which is an almost integer, in that it is very close to an integer.

> ##### **Arguments:**

        - None
        
> ##### **Returns:**

      e**{{\pi {\sqrt {163}}}}: Value to the 18th decimal
      262_537_412_640_768_743.999_999_999_999_250_073
      
---
### `glaisher_kinkelin_constant(self):`

A mathematical constant, related to the K-function and the Barnes G-function.

> ##### **Arguments:**

        - None
        
> ##### **Returns:**

      A: Value to the 20th decimal
      1.282_427_129_100_622_636_87
      
---
### `catalans_constant(self):`

Computes the Catalan's constant to the specified number of decimal places using the formula:

> ##### **Arguments:**

      n (int): The number of terms to sum to approximate the constant.

> ##### **Returns:**

      float: The computed value of the Catalan's constant.

> ##### **Example:**

      >>> catalan_constant(1000000)
      0.915965594177219

---
### `dottie_number(self):`

Calculates the unique real root of the equation cos(x) = x, known as the Dottie number, to the 20th decimal place.

The Dottie number is a constant that arises in the study of iterative methods and has connections to chaos theory.

> ##### **Arguments:**

        - None
        
> ##### **Returns:**

         float: The Dottie number, i.e., the unique real root of the equation cos(x) = x, to the 20th decimal place.
    
> ##### **Example:**

       >>> dottie_number()
       0.73908513321516064165

---
### `meissel_mertens_constant(self):`

Returns the Meissel-Mertens constant M to the 40th decimal place.

The Meissel-Mertens constant M is defined as the sum of the reciprocal of the primes up to n, where n is an arbitrary positive integer. It has important connections to prime number theory and the Riemann hypothesis.

This function uses a precomputed value of M to return the constant to the 40th decimal place.

> ##### **Arguments:**

        - None
        
> ##### **Returns:**

      M (float): The Meissel-Mertens constant M to the 40th decimal place.
      return 0.261_497_212_847_642_783_755_426_838_608_695_859_051_6

> ##### **Example:**

        meissel_mertens_constant()
        0.2614972128476427837554268386086958590516

---
### `universal_parabolic_constant(self):

The ratio, for any parabola, of the arc length of the parabolic segment formed by the latus rectum to the focal parameter.

> ##### **Arguments:**

        - None
        
> ##### **Returns:**

      P: Value to the 20th decimal
      2.295_587_149_392_638_074_03
      
---
### `cahens_constant(self):`

The value of an infinite series of unit fractions with alternating signs.

> ##### **Arguments:**

        - None
        
> ##### **Returns:**

      C: Value to the 20th decimal
      0.643_410_546_288_338_026_18
      
---
### `gelfonds_constant(self):`

Calculates Gelfond's Constant, which is defined as e raised to the power of pi.

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of Gelfond's Constant, which is approximately 23.1406926327792690057292.

            - return self.eulers_constant**self.pi
    
---
### `gelfond_schneider_constant(self):`

Returns the Gelfond-Schneider constant, which is a transcendental number defined as the value of 
2^(1/2) raised to the power of itself, or approximately 2.6651441426902251886502972498731.

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of the Gelfond-Schneider constant.

            - pass

---
### `second_favard_constant(self):`

Returns the Second Favard constant, which is a mathematical constant defined as the limit of the 
arithmetic mean of the reciprocal of consecutive odd numbers, or approximately 0.661707182...

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of the Second Favard constant.

            - pass

---
### `golden_angle(self):`

Returns the golden angle constant, which is the angle subtended by the smaller of the two angles 
formed by dividing the circumference of a circle in the golden ratio. It is equal to 
(3 - sqrt(5)) * 180 degrees / pi, or approximately 137.5077640500378546463487 degrees.

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of the golden angle constant in degrees.

            - pass

---
### `sierpinskis_constant(self):`

Returns Sierpiński's constant, which is the fractal dimension of the Sierpiński triangle, a 
self-similar fractal shape. It is equal to log(3)/log(2), or approximately 1.585.

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of Sierpiński's constant.

            - pass

---
### `landau_ramanujan_constant(self):`

Returns the Landau-Ramanujan constant, which is a mathematical constant that appears in the 
asymptotic expansion of the partition function. It is equal to e^(pi * sqrt(163)), or approximately
2.2932021438344677e+17.

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of the Landau-Ramanujan constant.

            - pass

---
### `first_nielsen_ramanujan_constant(self):`

Returns the First Nielsen-Ramanujan constant, which is a mathematical constant that appears in 
certain partition identities. It is equal to the product of a series involving the gamma function, 
or approximately 0.866081804933.

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of the First Nielsen-Ramanujan constant.

            - pass

---
### `gieseking_constant(self):`

Returns Gieseking's constant, which is a mathematical constant that appears in the theory of 
harmonic analysis. It is equal to (2*pi)^(-3/4), or approximately 0.7511255444649425.

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of Gieseking's constant.

            - pass

---
### `bernsteins_constant(self):`

Returns Bernstein's constant, which is a mathematical constant that appears in the theory of 
Fourier analysis. It is equal to pi/sqrt(2), or approximately 2.221441469079183.

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of Bernstein's constant.

            - pass

---
### `tribonacci_constant(self):`

Returns the Tribonacci constant, which is a mathematical constant defined as the unique real root 
of the polynomial x^3 - x^2 - x - 1, or approximately 1.8392867552141612.

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of the Tribonacci constant.

            - pass

    
---
### `bruns_constant(self):`

Returns the limiting value of the sequence a(n) = sum(k=1 to n) 1/prime(k),
where prime(k) is the kth prime number.

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of Bruns constant, accurate to 42 decimal places.

            - pass

---
### `twin_primes_constant(self):`

Returns the limiting value of the sequence of twin primes (pairs of prime
numbers that differ by 2).

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of the twin primes constant, accurate to 36 decimal places.

            - pass

---
### `plastic_number(self):`

Returns the unique positive real root of x^3 = x + 1.

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of the plastic number, accurate to 32 decimal places.

            - pass

---
### `blochs_constant(self):`

Returns the limiting value of the sequence of numbers that represent the
Bloch wall widths in ferromagnets.

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of Bloch's constant, accurate to 34 decimal places.

            - pass

---
### `z_score_975_percentile(self):`

Returns the value that has 97.5% of the area under a standard normal distribution
to the left of it.

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of the z-score at the 97.5th percentile, accurate to 9 decimal places.

            - pass

---
### `landaus_constant(self):`

Returns the limiting value of the sequence of numbers that represent the
probability that a random permutation of n elements will have no cycle of length
greater than log(n).

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of Landau's constant, accurate to 19 decimal places.

            - pass

---
### `landaus_third_constant(self):`

Returns the limiting value of the sequence of numbers that represent the
probability that a random permutation of n elements will have no cycle of length
greater than sqrt(n) * log(n).

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of Landau's third constant, accurate to 20 decimal places.

            - pass

---
### `prouhet_thue_morse_constant(self):`

Returns the limiting value of the sequence of numbers that represent the
differences in density between the 0's and 1's in the Prouhet-Thue-Morse
sequence.

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of the Prouhet-Thue-Morse constant, accurate to 20 decimal places.

            - pass
    
---
### `golomb_dickman_constant(self):`

The Golomb-Dickman constant represents the limiting distribution of the ratio of the k-th smallest
number in a sample of n random numbers to n^(1/k) as n approaches infinity. It is denoted by G.

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of the Golomb-Dickman constant G, approximately 0.6243299885435508.

            - return 0.6243299885435508

---
### `lebesgue_asymptotic_behavior_constant(self):`

The Lebesgue asymptotic behavior constant describes the average size of the level sets
of a random walk in d dimensions. It is denoted by L(d).

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of the Lebesgue asymptotic behavior constant L(3), approximately 3.912023005428146.

            - return 3.912023005428146

---
### `feller_tornier_constant(self):`

The Feller-Tornier constant is the probability that a random walk on the integers
returns to the origin infinitely often. It is denoted by F.

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of the Feller-Tornier constant F, approximately 0.259183.

            - return 0.259183

---
### `base_10_champernowne_constant(self):`

The Champernowne constant is formed by concatenating the base 10 representations of
successive integers, and is represented by C_10. 

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of the base 10 Champernowne constant C_10, approximately 0.12345678910111213...

---
### `salem_constant(self):`

The Salem number is a complex number that is a root of a certain polynomial
with integer coefficients. It is denoted by s.

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - complex: The value of the Salem constant s, approximately (1+sqrt(2)) * e^(pi*sqrt(2)/4).

            - return (1 + 2 ** 0.5) * Algorithm.exp(Constants.pi * 2 ** 0.5 / 4)
    
---
### `khinchins_constant(self):`

The Khinchin constant is a number that appears in the theory of continued fractions. 
It is denoted by K.

> ##### **Arguments:**

            - none

> ##### **Returns:**
            - float: The value of the Khinchin constant K, approximately 2.6854520010653065.

            - return 2.6854520010653065

---
### `levys_constant(self):`

Levy's constant, also known as the Levy–Khinchin constant, is a mathematical constant that arises in the study of 
Levy processes, which are stochastic processes that exhibit properties such as long-range dependence and heavy tails. 
It is defined as the limit of the average absolute difference between two random variables divided by their 
root-mean-square difference, as the sample size tends to infinity. The value of Levy's constant is approximately 
1.3303872425, with high precision being 1.33038724246235217434246.
            
> ##### **Symbol:**

            - γ or K

> ##### **Arguments:**

            - none


> ##### **Returns:**

            - float: The value of Levy's constant.
    
            - return 1.330_387_242_462_352_174_342_46

---
### `levys_constant_two(self):`

Calculate the value of e to the power of Levy's constant.

> ##### **Arguments:**

            - none

> ##### **Returns:**

            - float: The value of e to the power of Levy's constant.

            - return Algorithm.exp(self.levys_constant)

---
### `copeland_erdos_constant(self):`

Copeland-Erdős constant is the smallest number that is not the sum of 
distinct non-negative integer powers of 2.

> ##### **Symbol:**
            - C_E

> ##### **Arguments:**

            - none

> ##### **Returns:**

            - float
    
---
### `gompertz_constant(self):`

Gompertz constant is a mathematical constant named after Benjamin Gompertz,
it is the limit of the ratio between the life expectancy of a certain age 
and the remaining life expectancy.

> ##### **Symbol:**
            - γ

> ##### **Arguments:**

            - none

> ##### **Returns:**

            - float
    
---
### `de_bruijn_newman_constant(self):`

De Bruijn–Newman constant is the limit of the sequence of coefficients a_n
such that the entire function f(z) = Π_(n=1)^∞ [(1 - z/a_n) * exp(z/a_n)] has
no zeros in the complex plane.

> ##### **Symbol:**

            - λ

> ##### **Arguments:**

            - none

> ##### **Returns:**

            - float

---
### `van_der_pauw_constant():`

The van der Pauw constant is a constant used in measuring resistance of flat samples,
and is defined as the ratio of the natural logarithm of the quotient of two measured
resistances to the constant π.

> ##### **Symbol:**

            - K

> ##### **Arguments:**

            - none

> ##### **Returns:**

            - float: The value of the van der Pauw constant to the highest precision.

            - return Algorithm.exp(Constants.pi * MathFunctions.copysign(1, MathFunctions.acos(1/Constants.pi)))

---
### `magic_angle(self):`

        Magic angle is an angle of rotation for the bilayer graphene where the
        electronic properties of the material exhibit a number of interesting
        phenomena.

> ##### **Symbol:**

            - θ

> ##### **Arguments:**

            - none

> ##### **Returns:**

            - float: The magic angle in radians.

            - return Constants.arctan(Algorithm.square_root(3))


---
### `arctan(x):

Calculates the arctangent of x using a Taylor series approximation.

> ##### **Arguments:**
            x (float): A real number.

> ##### **Returns:**
            - float: The arctangent of x in radians.


---
### `artins_constant(self):`

The Artin's constant is a number that appears in the formula to calculate the Artin-Mazur zeta function.
It is defined as the infinite product of (1 - p^(-s)) where p ranges over all prime numbers and s is the reciprocal
of the prime number.

> ##### **Arguments:**

            - none

> ##### **Returns:**

            - float: The value of the Artin's constant to the highest precision.

---
### `porters_constant(self):`

Porter's constant is a mathematical constant that appears in the field of information theory. It is defined as
the limit of the ratio of the maximum number of different words of length n over the number of possible words of
length n as n approaches infinity.

> ##### **Symbol:**

            -

> ##### **Arguments:**

            - none

> ##### **Returns:**

            - float: The value of Porter's constant to the highest precision.

            - return Algorithm.exp**(1/Constants.euler_mascheroni_constant)

---
### `euler_mascheroni_constant(self):`

Returns the Euler-Mascheroni constant, a mathematical constant that appears in many areas of mathematics.
It is defined as the limit of the difference between the harmonic series and the natural logarithm of n as n approaches infinity.
        
The function calculates the value of the Euler-Mascheroni constant using a sum of the harmonic series and the natural logarithm of n.
The sum is taken over a large number of terms to achieve a high degree of accuracy.
        
Note that the function uses the 'math' module to calculate the natural logarithm, so it must be imported before the function can be called.

> ##### **Arguments:**

            - none

> ##### **Returns:**

            - float: The value of the Euler-Mascheroni constant to a high degree of accuracy.

---
### `lochs_constant(self):`

Lochs' constant is a mathematical constant defined as the limiting ratio of the perimeter of an inscribed regular
decagon to its diameter.

> ##### **Symbol:**

            - 

> ##### **Arguments:**

            - none

> ##### **Returns:**

            - float: The value of Lochs' constant to the highest precision.

            - return Algorithm.square_root(2 + Algorithm.square_root(2 + Algorithm.square_root(2 + Algorithm.square_root(2 + Algorithm.square_root(2)))))


---
### `deviccis_tesseract_constant(self):`

The De Vries - De Vos - Barendrecht - De Klerk - Smit - Smit constant (also known as De Vries' tesseract constant)
is defined as the number that describes the maximum ratio of the content of a hypercube inscribed in a tesseract to
the content of the hypercube circumscribed about the tesseract.

> ##### **Symbol:**

            -

> ##### **Arguments:**

            - none

> ##### **Returns:**

            - float: The value of De Vries' tesseract constant to the highest precision.

            - return Algorithm.square_root(2 + Algorithm.square_root(2)) / (2 * Algorithm.square_root(2))


---
### `liebs_square_ice_constant(self):`

The Lieb's square ice constant is the infinite sum of alternating sign reciprocals of the squares of odd positive integers.
It appears in the square ice problem in statistical mechanics.

> ##### **Symbol:**

            - 

> ##### **Arguments:**

            - none

> ##### **Returns:**

            - float: The value of the Lieb's square ice constant to the highest precision.

            - return Constants.pi / (Algorithm.square_root(3) * Algorithm.log((3 + Algorithm.square_root(8)) / 2))

---
### `nivens_constant(self):`

Niven's constant is a mathematical constant that is the only known integer x that is divisible by the sum of its digits
when written in decimal base. The constant is also related to the convergence of certain infinite series.

> ##### **Symbol:**

            - 

> ##### **Arguments:**

            - none

> ##### **Returns:**

            - int: The value of Niven's constant to the highest precision.
    
---
### `mills_constant(self):`

Mills constant is the smallest positive real number A such that the 
floor function of the double exponential function is a prime number,
where the double exponential function is f(n) = A^(3^n).

> ##### **Symbol:**

            - A

> ##### **Arguments:**

            - none

> ##### **Returns:**

            - float

---
### `stephens_constant(self):`

Stephens' constant is a mathematical constant that arises in the study of prime numbers.

> ##### **Returns:**
            
            - float: The value of Stephens' constant.

            return 0.5364798721

---
### `regular_paperfolding_sequence(self):`

The regular paperfolding sequence is a binary sequence that arises in the study of fractal geometry.

> ##### **Returns:**

            - str: The regular paperfolding sequence as a string of 0s and 1s.

            return "110110011100100"

---
### `reciprocal_fibonacci_constant(self):`

The reciprocal Fibonacci constant is a real number that arises in the study of Fibonacci numbers.

> ##### **Returns:**

            - float: The value of the reciprocal Fibonacci constant.

            return 1.1319882488
    
---
### `chvatal_sankoff_constant(self):`

Chvátal–Sankoff constant for the binary alphabet.

> ##### **Symbol:**

            - \gamma_{2}

> ##### **Returns:**
            
            - float: The value of the Chvátal–Sankoff constant.

            return 1.7550327129
    
---
### `Feigenbaum_constant(self):`

Feigenbaum constant δ

> ##### **Symbol:**

            - \delta

> ##### **Returns:**

            - float: The value of the Feigenbaum constant.

            return 4.6692016091
    
---
### `chaitins_constant(self):`

Chaitin's constant is a real number that encodes the halting probability of a universal Turing machine.

> ##### **Symbol:**
            
            - \Omega

> ##### **Raises:**

            - ValueError: If the computation of the constant fails.

> ##### **Returns:**

            - float: The value of Chaitin's constant.
    
---
### `robbins_constant(self):`

Robbins' constant is a mathematical constant that arises in the study of mathematical analysis.

> ##### **Symbol:**
        
            - \Delta (3)

> ##### **Raises:**

            - ValueError: If the computation of the constant fails.

> ##### **Arguments:**

            - none

> ##### **Returns:**

            - float: The value of Robbins' constant.
    
---
### `weierstrass_constant(self):`

Weierstrass' constant is a mathematical constant that arises in the study of elliptic functions.

> ##### **Returns:**

            - float: The value of Weierstrass' constant.

            return 0.5174790617
    
---
### `fransen_robinson_constant(self):`

Returns Fransen-Robinson constant which is the smallest positive root of the following polynomial equation:
        
x^3 - x^2 - 1 = 0
        
> ##### **Symbol:**

            - F
        
> ##### **Raises:**
            
            - ValueError: If the root cannot be found
        
> ##### **Returns:**

            - float: The Fransen-Robinson constant
    
---
### `feigenbaum_constant(self):`

Returns Feigenbaum constant alpha which relates to the period-doubling bifurcation in chaotic systems.

> ##### **Symbol:**

            - \alpha 

> ##### **Raises:**

            - ValueError: If the constant cannot be computed

> ##### **Returns:**
            
            - float: The Feigenbaum constant alpha
    
---
### `second_du_bois_reymond_constant(self):`

Returns the Second du Bois-Reymond constant, which is defined as the supremum of the absolute values of the Fourier coefficients of a bounded variation function with period 1.

> ##### **Symbol:**
            
            - C_{2}
            
> ##### **Raises:**

            - ValueError: If the constant cannot be computed

> ##### **Returns:**

            - float: The Second du Bois-Reymond constant
    
---
### `erdos_tenenbaum_ford_constant(self):`

Returns the Erdős–Tenenbaum–Ford constant which is related to the distribution of prime numbers.

> ##### **Symbol:**
            
            - \delta
            
> ##### **Raises:**
            
            - ValueError: If the constant cannot be computed

> ##### **Returns:**
            
            - float: The Erdős–Tenenbaum–Ford constant
    
---
### `conways_constant(Self):`

Returns Conway's constant, which is the unique real root of the following polynomial equation:

x^3 - x - 1 = 0

> ##### **Symbol:**

            - \lambda
            
> ##### **Arguments:**
            
            - Self (object): The class instance

> ##### **Raises:**
            
            - ValueError: If the root cannot be found

> ##### **Returns:**
            
            - float: Conway's constant
    
---
### `backhouses_constant(self):`

Returns Backhouse's constant which is defined as the smallest k such that the inequality n! > k^n holds for all positive integers n.


> ##### **Symbol:**

            - B
            

> ##### **Raises:**

            - ValueError: If the constant cannot be computed


> ##### **Returns:**

            - float: Backhouse's constant

---
### `viswanath_constant(self):`

Returns Viswanath's constant, which is the limiting distribution of the ratios of successive gaps in the sequence of zeros of the Riemann zeta function.


> ##### **Symbol:**

            - \Omega_V
            

> ##### **Raises:**

            - ValueError: If the constant cannot be computed


> ##### **Returns:**

            - float: Viswanath's constant

---
### `komornik_loreti_constant(self):`

Returns Komornik-Loreti constant, which is the unique positive real root of the following polynomial equation:

        x^2 - x - 1 = 0


> ##### **Symbol:**

            - q
            

> ##### **Raises:**

            - ValueError: If the root cannot be found


> ##### **Returns:**

            - float: Komornik-Loreti constant
    
---
### `embree_trefethen_constant(self):`

Computes the Embree-Trefethen constant, which is defined as the supremum of the real parts
of the poles of a certain rational function.


> ##### **Symbol:**

            - {\displaystyle \beta ^{\star }}


> ##### **Raises:**

            - ValueError: If the computation fails to converge.


> ##### **Returns:**

            - float: The computed value of the Embree-Trefethen constant.

> ##### **References:**

            * Embree, M., & Trefethen, L. N. (1999). Growth and decay of random plane waves. 
            Communications on Pure and Applied Mathematics, 52(7), 757-788.
            * Trefethen, L. N. (2006). Spectral methods in MATLAB. SIAM.
    
---
### `heath_brown_moroz_constant(self):`

Computes the Heath-Brown-Moroz constant, which is defined as the product of the Euler-Mascheroni 
constant and the reciprocal of a certain infinite product.


> ##### **Symbol:**

            - C


> ##### **Raises:**

            - ValueError: If the computation fails to converge.


> ##### **Returns:**

            - float: The computed value of the Heath-Brown-Moroz constant.

> ##### **References:**

            * Heath-Brown, D. R. (1984). The fourth power moment of the Riemann zeta-function. 
            Proceedings of the London Mathematical Society, 49(2), 475-513.
            * Moroz, B. Z. (2001). Some constants associated with the Riemann zeta function. 
            Journal of Mathematical Analysis and Applications, 261(1), 235-251.
    
---
### `mrb_constant():`

Computes the MRB constant, which is defined as the sum of the alternating series obtained by 
raising the first n positive integers to their own powers and then summing them with alternating signs.


> ##### **Symbol:**

            - S


> ##### **Raises:**

            - ValueError: If the computation fails to converge.


> ##### **Returns:**

            - float: The computed value of the MRB constant.

> ##### **References:**

            * Borwein, J. M., Bradley, D. M., & Crandall, R. E. (1999). Computational strategies for 
            the Riemann zeta function. Journal of Computational and Applied Mathematics, 121(1-2), 247-296.
            * Bradley, D. M. (2004). Multiple q-zeta values. Ramanujan Journal, 8(1), 39-65.

---
### `prime_constant():`

Computes the Prime constant, which is defined as the product of the reciprocals of the primes 
minus ln(ln(2)).

> ##### **Symbol:**

            - \rho 

> ##### **Raises:**

            - ValueError: If the computation fails to converge.

> ##### **Returns:**

            - float: The computed value of the Prime constant.

> ##### **References:**

            * Meissel, L. (1879). Bestimmung einer zahl, welche zu der logaritmierten primzahlfunction 
            π(x) in näherung den nämlichen wert wie die zahl x selbst gibt. 
            Journal für die Reine und Angewandte Mathematik, 1879(88), 127-133.
            * Lehmer, D. H. (1959). List of computed values of the prime-counting function π(x) 
            from x= 10^6 to x= 10^20. U. S. National Bureau of Standards Applied Mathematics Series, 46.

---
### `somos_quadratic_recurrence_constant():`

Returns the Somos quadratic recurrence constant.


> ##### **Symbol:**

            - \sigma
                

> ##### **Raises:**

            - ValueError: If the calculation is not valid.


> ##### **Returns:**

            - float: The value of the Somos quadratic recurrence constant.

---
### `foias_constant(self):`

Returns the Foias constant.


> ##### **Symbol:**

            - \alpha
            

> ##### **Raises:**

            - ValueError: If the calculation is not valid.


> ##### **Returns:**

            - float: The value of the Foias constant.

---
### `logarithmic_capacity(self):`

Returns the logarithmic capacity of the unit disk.


> ##### **Raises:**

            - ValueError: If the calculation is not valid.


> ##### **Returns:**

            - float: The value of the logarithmic capacity.

            return 2

---
### `taniguchi_constant(self):`

Returns the Taniguchi constant.


> ##### **Raises:**

            - ValueError: If the calculation is not valid.


> ##### **Returns:**

            - float: The value of the Taniguchi constant.

---
## **Functions Class**

A class containing various mathematical functions.

### `gamma(self, x):`

Compute the value of the gamma function at the given value of x.

        Args:
            x (float): The value at which the gamma function is to be evaluated.

> ##### **Returns:**
        
        - float: The value of the gamma function at the given value of x.

> ##### **Raises:**
        
        - ValueError: If x is negative and not an integer.

> ##### **Notes:**

            The gamma function is defined as the integral from zero to infinity of t^(x-1) * exp(-t) dt.
            For positive integers, the gamma function can be computed recursively as (n-1)!.
            For x <= 0, the gamma function is undefined, but we return NaN to avoid raising an error.

### `area_of_circle(self, r: float) -> float:`

Calculates the area of a circle given its radius.

> ##### **Arguments:**

        - r: The radius of the circle.

> ##### **Returns:**

        - The area of the circle.
        - return 3.141592653589793238 * r ** 2

---
### `volume_of_sphere(self, r: float) -> float:`

Calculates the volume of a sphere given its radius.

> ##### **Arguments:**

        - r: The radius of the sphere.

> ##### **Returns:**

        - The volume of the sphere.
        - return 4 / 3 * 3.141592653589793238 * r ** 3

---
### `perimeter_of_rectangle(self, l: float, b: float) -> float:`

Calculates the perimeter of a rectangle given its length and breadth.

> ##### **Arguments:**

        - l: The length of the rectangle.
        - b: The breadth of the rectangle.

> ##### **Returns:**

        - The perimeter of the rectangle.
        - return 2 * (l + b)

---
### `pythagoras_theorem_length(self, a: float, b: float) -> float:`

Calculates the length of the hypotenuse of a right-angled triangle given the lengths of its two other sides.

> ##### **Arguments:**

        a: The length of one of the sides of the triangle.
        b: The length of the other side of the triangle.

> ##### **Returns:**

        The length of the hypotenuse of the triangle.
        return (a ** 2 + b ** 2) ** 0.5

---
### `square_root(self, x: float) -> float:`

Calculates the square root of a given number.

> ##### **Arguments:**

        x: The number to take the square root of.

> ##### **Returns:**

        The square root of x.
        return x ** 0.5

---
### `factorial(self, n: int) -> int:`

Calculates the factorial of a given number.

> ##### **Arguments:**

        n: The number to calculate the factorial of.

> ##### **Returns:**

        The factorial of n.

        fact = 1
        for i in range(1, n+1):
            fact *= i
        return fact

---
### `gcd(self, a: int, b: int) -> int:`

Calculates the greatest common divisor of two numbers.

> ##### **Arguments:**

        a: The first number.
        b: The second number.

> ##### **Returns:**

        The greatest common divisor of a and b.
        
        while(b):
            a, b = b, a % b
        return a

---
### `lcm(self, a: int, b: int) -> int:`

Calculates the least common multiple of two numbers.

> ##### **Arguments:**

        a: The first number.
        b: The second number.

> ##### **Returns:**

        The least common multiple of a and b.

        return a * b // self.gcd(a, b)

---
### `exponential(self, x: float) -> float:`

Calculates the value of e raised to a given power.

> ##### **Arguments:**

        x: The exponent.

> ##### **Returns:**

        The value of e raised to x.

        e = 2.718281828459045235
        return e ** x

---
### `logarithm(self, x: float, base: float) -> float:`

Calculates the logarithm of a given number to a given base.

> ##### **Arguments:**

        x: The number to take the logarithm of.
        base: The base of the logarithm.

> ##### **Returns:**

        The logarithm of x to the base.

        return (Functions.log(x) / Functions.log(base))

---
### `log(x):`

Calculates the natural logarithm of a given number.

> ##### **Arguments:**

        x: The number to take the natural logarithm of.

> ##### **Returns:**

        The natural logarithm of x.

        if x <= 0:
            return float('nan')
        elif x == 1:
            return 0.0
        else:
            return Functions.integrate(1/x, 1, x)

---
### `integrate(f, a, b):`

Approximates the definite integral of a function over a given interval using the trapezoidal rule.

> ##### **Arguments:**

        f: The function to integrate.
        a: The lower limit of the interval.
        b: The upper limit of the interval.

> ##### **Returns:**

        The approximate value of the definite integral of f over the interval [a, b].

        n = 1000 # Number of trapezoids to use
        dx = (b - a) / n
        x_values = [a + i * dx for i in range(n+1)]
        y_values = [f(x) for x in x_values]
        return (dx/2) * (y_values[0] + y_values[-1] + 2*sum(y_values[1:-1]))

---
### `surface_area_of_cylinder(self, r: float, h: float) -> float:`

Calculates the surface area of a cylinder given its radius and height.

> ##### **Arguments:**

        r: The radius of the cylinder.
        h: The height of the cylinder.

> ##### **Returns:**

        The surface area of the cylinder.

        return 2 * 3.14159265358979323846 * r * (r + h)

---
### `volume_of_cylinder(self, r: float, h: float) -> float:`

Calculates the volume of a cylinder given its radius and height.

> ##### **Arguments:**

        r: The radius of the cylinder.
        h: The height of the cylinder.

> ##### **Returns:**

        The volume of the cylinder.

        return 3.14159265358979323846 * r ** 2 * h

---
### `area_of_triangle(self, b: float, h: float) -> float:`

Calculates the area of a triangle given its base and height.

> ##### **Arguments:**

        b: The base of the triangle.
        h: The height of the triangle.

> ##### **Returns:**

        The area of the triangle.

        return 0.5 * b * h

---
### `sine(self, x: float) -> float:`

Calculates the sine of a given angle in radians.

> ##### **Arguments:**

        x: The angle in radians.

> ##### **Returns:**

        The sine of the angle.

        x = x % (2 * 3.141592653589793238)
        sign = 1 if x > 0 else -1
        x *= sign
        if x > 3.141592653589793238:
            x -= 2 * 3.141592653589793238
            sign *= -1
        return sign * (
            x - x ** 3 / 6 + x ** 5 / 120 - x ** 7 / 5040 + x ** 9 / 362880
        )
        
### `copysign(self, x, y):`

Return a float with the magnitude of x and the sign of y.

> ##### **Arguments:**

      - x (float): The magnitude of the result.
      - y (float): The sign of the result.

> ##### **Returns:**

      - float: A float with the magnitude of x and the sign of y.


---

### `acos(self, x):`

Return the arc cosine of x, in radians.

> ##### **Arguments:**

      - x (float): The value whose arc cosine is to be returned.

> ##### **Returns:**

      - float: The arc cosine of x, in radians.


---
---
## **Sequences class:**
    
---
### `harmonic_number(self, n: int) -> float:`

The nth harmonic number is the sum of the reciprocals of the first n natural numbers.

> ##### **Symbol:**

            - H_n

> ##### **Arguments:**

            - n (int): The number of terms to include in the sum.

> ##### **Returns:**

            - float: The value of the nth harmonic number.

            - return sum(1/i for i in range(1, n+1))
    
---
### `gregory_coefficients(self, n: int) -> float:`

The nth Gregory coefficient is a coefficient used in the Gregory series formula for pi,
which provides an approximate value of pi.

> ##### **Symbol:**

            - G_n

> ##### **Arguments:**

            - n (int): The index of the Gregory coefficient to be calculated.

> ##### **Returns:**

            - float: The value of the nth Gregory coefficient.

            if n == 0:
                return 1
            elif n % 2 == 0:
                return 0
            else:
                return -2 / (n * Constants.pi) * self.gregory_coefficients(n-1)
    
---
### `bernoulli_number(self, n: int) -> float:`

The nth Bernoulli number is a sequence of rational numbers with deep connections to number theory
and other areas of mathematics, including algebra and calculus.

> ##### **Symbol:**

            - B_n

> ##### **Arguments:**

            - n (int): The index of the Bernoulli number to be calculated.

> ##### **Returns:**

            - float: The value of the nth Bernoulli number.

            if n == 0:
                return 1
            elif n == 1:
                return -0.5
            else:
                sum_term = sum(MathFunctions.combination(n+1, k) * self.bernoulli_number(k) / (n+1-k) for k in range(1, n))
                return 1 - sum_term
    
---
### `hermite_constants(self, n: int) -> float:`

The nth Hermite constant is a constant that appears in the study of the quantum harmonic oscillator,
and is related to the normalization of the wave functions of the oscillator.

> ##### **Symbol:**

            - H_n

> ##### **Arguments:**

            - n (int): The index of the Hermite constant to be calculated.

> ##### **Returns:**

            - float: The value of the nth Hermite constant.

            if n == 0:
                return 1
            else:
                return (-1)**n * Algorithm.factorial(n-1)
    
---
### `hafner_sarnak_mccurley_constant(self, n: int) -> float:`

The nth Hafner-Sarnak-McCurley constant is a constant that appears in the study of prime numbers
and related topics in number theory.

> ##### **Symbol:**

            - C_n

> ##### **Arguments:**

            - n (int): The index of the Hafner-Sarnak-McCurley constant to be calculated.

> ##### **Returns:**

            - float: The value of the nth Hafner-Sarnak-McCurley constant.

            - return sum(Algorithm.exp(-n/p)/p for p in Algorithm.sieve_of_eratosthenes(2*n+1))
    
    
---
### `stieltjes_constants(self, n: int) -> float:`

Returns the nth Stieltjes constant.
        
> ##### **Arguments:**

            - n (int): the index of the sequence.
            
> ##### **Returns:**

            - float: the nth Stieltjes constant.
            - if n == 1:
                return 0.57721566490153286060651209  # gamma
            elif n == 2:
                return 1.20205690315959428539973816  # G
            elif n == 3:
                return 1.79175946922805500081247735  # pi^2/6
            elif n == 4:
                return 2.9456101084887218059356      # 7*G - 4*pi^2/3
            elif n == 5:
                return 4.4428829381583661417149      # 3*zeta(3) - 2*G
            else:
                raise ValueError("The index n should be between 1 and 5.")
            
> ##### **Reference:**

            - https://mathworld.wolfram.com/StieltjesConstants.html
            
---
### `favard_constants(self, n: int) -> float:`

Returns the nth Favard constant.
        
> ##### **Arguments:**

            - n (int): the index of the sequence.
            
> ##### **Returns:**

            - float: the nth Favard constant.

            if n < 1:
                raise ValueError("The index n should be a positive integer.")
            elif n == 1:
                return 1
            else:
                return sum([Sequences.favard_constants(self, i) * Sequences.favard_constants(self, n-i) / (i+n-i-1) 
                            for i in range(1, n)])
            
> ##### **Reference:**

            - https://mathworld.wolfram.com/FavardConstants.html
        
        
---
### `generalized_bruns_constant(self, n: int) -> float:`

Returns the nth generalized Bruns constant.
        
> ##### **Arguments:**

            - n (int): the index of the sequence.
            
> ##### **Returns:**

            - float: the nth generalized Bruns constant.

            if n < 1:
                raise ValueError("The index n should be a positive integer.")
            elif n == 1:
                return 1
            else:
                return sum([abs(Sequences.generalized_bruns_constant(self, i) - Sequences.generalized_bruns_constant(self, i-1)) 
                            for i in range(2, n+1)]) + 1
            
> ##### **Reference:**

            - https://mathworld.wolfram.com/GeneralizedBrunsConstant.html
        
---
### `champernowne_constants(self, n: int) -> float:`

Returns the nth Champernowne constant.
        
> ##### **Arguments:**

            - n (int): the index of the sequence.
            
> ##### **Returns:**

            - float: the nth Champernowne constant.

            if n < 1:
                raise ValueError("n should be a positive integer")
            if n == 1:
                return 0.12345678910111213141516171819202122
            else:
                prev = self.champernowne_constants(n-1)
                return float(str(prev) + str(n+8))
            
> ##### **Reference:**

            - https://mathworld.wolfram.com/ChampernowneConstant.html


---
### `lagrange_number(self, n: int) -> int:`

Returns the nth Lagrange number.
        
> ##### **Arguments:**

            - n (int): the index of the sequence.
            
> ##### **Returns:**

            - int: the nth Lagrange number.

            if n < 1:
                raise ValueError("n should be a positive integer")
            if n == 1:
                return 1
            else:
                return n * self.lagrange_number(n-1) - (-1)**n
            
> ##### **Reference:**

            - https://mathworld.wolfram.com/LagrangeNumber.html
    
---
### `fellers_coin_tossing_constants(self, n: int) -> float:`

Returns the nth Feller's coin-tossing constant.
        
> ##### **Arguments:**

            - n (int): the index of the sequence.
            
> ##### **Returns:**

            - float: the nth Feller's coin-tossing constant.

            result = 0
            for k in range(n + 1):
                result += (-1) ** k / (2 ** (2 ** k))
            return result
            
> ##### **Reference:**

            - https://mathworld.wolfram.com/FellersCoin-TossingConstants.html
    
---
### `stoneham_number(self, n: int) -> int:`

Returns the nth Stoneham number.
        
> ##### **Arguments:**

            - n (int): the index of the sequence.
            
> ##### **Returns:**

            - int: the nth Stoneham number.
            if n == 0:
                return 1
            else:
                return (3 * Sequences.stoneham_number(n - 1) + 1) // 2

> ##### **Reference:**

            - https://mathworld.wolfram.com/StonehamNumber.html
    
---
### `beraha_constants(self, n: int) -> float:`

Returns the nth Beraha constant.
        
> ##### **Arguments:**

            - n (int): the index of the sequence.
            
> ##### **Returns:**

            - float: the nth Beraha constant.

            if n == 0:
                return 1
            else:
                return 1 + 1 / Sequences.beraha_constants(n - 1)
            
> ##### **Reference:**

            - https://mathworld.wolfram.com/BerahasConstant.html
    
---
### `chvatal_sankoff_constants(self, n: int) -> float:`

Returns the nth Chvátal-Sankoff constant.
        
> ##### **Arguments:**

            - n (int): the index of the sequence.
            
> ##### **Returns:**

            - float: the nth Chvátal-Sankoff constant.

            result = 0
            for k in range(n + 1):
                binom = MathFunctions.comb(2 ** k, k)
                result += (-1) ** k * binom ** 2
            return result
            
> ##### **Reference:**

            - https://mathworld.wolfram.com/Chvatal-SankoffConstants.html
    
---
### `hyperharmonic_number(self, n: int, p: int) -> float:`

Computes the hyperharmonic number H(n,p), which is defined as the sum of the p-th powers of the reciprocals of
        the first n positive integers.
        
> ##### **Arguments:**

        - n - (int): The positive integer up to which to compute the sum.
        - p (int): The exponent to which to raise the reciprocals of the integers.
        
> ##### **Returns:**

        - H - (float): The hyperharmonic number H(n,p).

        H = 0
        for i in range(1, n+1):
            H += 1 / i ** p
        return H
        
> ##### **Symbols:**

        - H(n,p): hyperharmonic number of order p and degree n.

---
### `gregory_number(self, n: int) -> float:`

Computes the nth Gregory number, which is defined as the alternating sum of the reciprocals of the odd
positive integers, up to the nth term.
        
> ##### **Arguments:**

        - n - (int): The positive integer up to which to compute the alternating sum.
        
> ##### **Returns:**

        - G - (float): The nth Gregory number.

        G = 0
        for i in range(1, n+1):
            if i % 2 == 1:
                G += 1 / i
            else:
                G -= 1 / i
        return G
        
> ##### **Symbols:**

        - G(n): nth Gregory number.


---
### `metallic_mean(self, x: float) -> float:`

Computes the value of the metallic mean of x, which is the positive solution to the equation x = 1/(1+x).
        
> ##### **Arguments:**

        - x - (float): The value for which to compute the metallic mean.
        
> ##### **Returns:**

        - mm-  (float): The value of the metallic mean of x.

        mm = (1 + Algorithm.square_root(1 + 4*x)) / 2
        return mm
        
> ##### **Symbols:**

        - mm(x): metallic mean of x.

---
## **HyperbolicFunctions class:**

A class representing the six hyperbolic functions: sinh, cosh, tanh, coth, sech, and csch.

> ##### **References:**

        * Weisstein, E. W. (n.d.). Hyperbolic functions. MathWorld--A Wolfram Web Resource. 
        Retrieved October 11, 2021, from https://mathworld.wolfram.com/HyperbolicFunctions.html

---
### `sinh(x):`

Returns the hyperbolic sine of x.

> ##### **Arguments:**

            - x (float): The input value in radians.

> ##### **Returns:**

            - float: The hyperbolic sine of x.

            return (Algorithm.exp(x) - Algorithm.exp(-x)) / 2

---
### `cosh(x):`

Returns the hyperbolic cosine of x.

> ##### **Arguments:**

            - x (float): The input value in radians.

> ##### **Returns:**

            - float: The hyperbolic cosine of x.

            return (Algorithm.exp(x) + Algorithm.exp(-x)) / 2

---
### `tanh(x):`

Returns the hyperbolic tangent of x.

> ##### **Arguments:**

            - x (float): The input value in radians.

> ##### **Returns:**

            - float: The hyperbolic tangent of x.

            return HyperbolicFunctions.sinh(x) / HyperbolicFunctions.cosh(x)

---
### `coth(x):`

Returns the hyperbolic cotangent of x.

> ##### **Arguments:**

            - x (float): The input value in radians.

> ##### **Returns:**

            - float: The hyperbolic cotangent of x.

            return 1 / HyperbolicFunctions.tanh(x)

---
### `sech(x):`

Returns the hyperbolic secant of x.

> ##### **Arguments:**

            - x (float): The input value in radians.

> ##### **Returns:**

            - float: The hyperbolic secant of x.

            return 1 / HyperbolicFunctions.cosh(x)

---
### `csch(x):`

Returns the hyperbolic cosecant of x.

> ##### **Arguments:**

            - x (float): The input value in radians.

> ##### **Returns:**

            - float: The hyperbolic cosecant of x.

            return 1 / HyperbolicFunctions.sinh(x)


---
## **ComplexNumber class:**

A class representing a complex number.

> ##### **Attributes:**

        - real (float): The real part of the complex number.
        - imag (float): The imaginary part of the complex number.

---
### `__init__(self, real=0, imag=0):`

Initializes a complex number.

> ##### **Arguments:**

            - real (float): The real part of the complex number.
            - imag (float): The imaginary part of the complex number.

---
### `__repr__(self):`

Returns a string representation of the complex number.

> ##### **Returns:**

            - str: A string representation of the complex number.

            return f"{self.real} + {self.imag}j"

---
### `__add__(self, other):`

Adds two complex numbers.

> ##### **Arguments:**

            - other (ComplexNumber): The complex number to add.

> ##### **Returns:**

            - ComplexNumber: The sum of the two complex numbers.

            return ComplexNumber(self.real + other.real, self.imag + other.imag)

---
### `__sub__(self, other):`

Subtracts two complex numbers.

> ##### **Arguments:**

            - other (ComplexNumber): The complex number to subtract.

> ##### **Returns:**

            - ComplexNumber: The difference of the two complex numbers.

            return ComplexNumber(self.real - other.real, self.imag - other.imag)

---
### `__mul__(self, other):`

Multiplies two complex numbers.

> ##### **Arguments:**

            - other (ComplexNumber): The complex number to multiply.

> ##### **Returns:**

            - ComplexNumber: The product of the two complex numbers.

            real = self.real * other.real - self.imag * other.imag
            imag = self.real * other.imag + self.imag * other.real
            return ComplexNumber(real, imag)

---
### `__truediv__(self, other):`

Divides two complex numbers.

> ##### **Arguments:**

            - other (ComplexNumber): The complex number to divide.

> ##### **Returns:**

            - ComplexNumber: The quotient of the two complex numbers.

            denom = other.real**2 + other.imag**2
            real = (self.real * other.real + self.imag * other.imag) / denom
            imag = (self.imag * other.real - self.real * other.imag) / denom
            return ComplexNumber(real, imag)

---
### `conjugate(self):`

Computes the conjugate of the complex number.

> ##### **Returns:**

            - ComplexNumber: The conjugate of the complex number.

            return ComplexNumber(self.real, -self.imag)

---
### `modulus(self):`

Computes the modulus (magnitude) of the complex number.

> ##### **Returns:**

            - float: The modulus of the complex number.

            return (self.real**2 + self.imag**2)**0.5


##################
---
## **RealNumber class:**

A class representing a real number.
    
---
### `__init__(self, value):`

Initializes a new RealNumber object with the given value.

> ##### **Parameters:**

            - value (float): The value of the real number.

> ##### **Returns:**

            - RealNumber: A new RealNumber object.

            self.value = value

---
### `__str__(self):`

Returns a string representation of the real number.

> ##### **Returns:**

            - str: A string representation of the real number.

            return str(self.value)

---
### `__repr__(self):`

Returns a string representation of the real number.

> ##### **Returns:**

            - str: A string representation of the real number.

            return str(self.value)

---
### `__eq__(self, other):`

Checks whether the real number is equal to another object.

> ##### **Parameters:**

            - other (object): The object to compare with.

> ##### **Returns:**

            - bool: True if the real number is equal to the other object, False otherwise.

            if isinstance(other, RealNumber):
                return self.value == other.value
            elif isinstance(other, float):
                return self.value == other
            else:
                return False

---
### `__ne__(self, other):`

Checks whether the real number is not equal to another object.

> ##### **Parameters:**

            - other (object): The object to compare with.

> ##### **Returns:**

            - bool: True if the real number is not equal to the other object, False otherwise.

            return not self.__eq__(other)

---
### `__lt__(self, other):`

Checks whether the real number is less than another object.

> ##### **Parameters:**

            - other (object): The object to compare with.

> ##### **Returns:**

            - bool: True if the real number is less than the other object, False otherwise.

            if isinstance(other, RealNumber):
                return self.value < other.value
            elif isinstance(other, float):
                return self.value < other
            else:
                return NotImplemented

---
### `__le__(self, other):`

Checks whether the real number is less than or equal to another object.

> ##### **Parameters:**

            - other (object): The object to compare with.

> ##### **Returns:**

            - bool: True if the real number is less than or equal to the other object, False otherwise.

            if isinstance(other, RealNumber):
                return self.value <= other.value
            elif isinstance(other, float):
                return self.value <= other
            else:
                return NotImplemented

---
### `__gt__(self, other):`

Checks whether the real number is greater than another object.

> ##### **Parameters:**

            - other (object): The object to compare with.

> ##### **Returns:**

            - bool: True if the real number is greater than the other object, False otherwise.

            if isinstance(other, RealNumber):
                return self.value > other.value
            elif isinstance(other, float):
                return self.value > other
            else:
                return NotImplemented

---
### `__ge__(self, other):`

Checks whether the real number is greater than or equal to another object.

> ##### **Parameters:**

            - other (object): The object to compare with.

> ##### **Returns:**

            - bool: True if the real number is greater than or equal to the other object, False otherwise.

            if isinstance(other, RealNumber):
                return self.value >= other.value
            elif isinstance(other, float):
                return self.value >= other
            else:
                return NotImplemented
        
---
### `__add__(self, other):`

Adds two RealNumber objects.

> ##### **Parameters:**

            - other (RealNumber or float): The RealNumber object or float to add.

> ##### **Returns:**

            - RealNumber: A new RealNumber object with the sum of the two numbers.

            if isinstance(other, RealNumber):
                return RealNumber(self.value + other.value)
            elif isinstance(other, float):
                return RealNumber(self.value + other)
            else:
                return NotImplemented

---
### `__sub__(self, other):`

Subtracts two RealNumber objects.

> ##### **Parameters:**

            - other (RealNumber or float): The RealNumber object or float to subtract.

> ##### **Returns:**

            - RealNumber: A new RealNumber object with the difference of the two numbers.

            if isinstance(other, RealNumber):
                return RealNumber(self.value - other.value)
            elif isinstance(other, float):
                return RealNumber(self.value - other)
            else:
                return NotImplemented

---
### `__mul__(self, other):`

Multiplies two RealNumber objects.

> ##### **Parameters:**

            - other (RealNumber or float): The RealNumber object or float to multiply.

> ##### **Returns:**

            - RealNumber: A new RealNumber object with the product of the two numbers.

            if isinstance(other, RealNumber):
                return RealNumber(self.value * other.value)
            elif isinstance(other, float):
                return RealNumber(self.value * other)
            else:
                return NotImplemented

---
### `__truediv__(self, other):`

Divides two RealNumber objects.

> ##### **Parameters:**

            - other (RealNumber or float): The RealNumber object or float to divide by.

> ##### **Returns:**

            - RealNumber: A new RealNumber object with the quotient of the two numbers.

            if isinstance(other, RealNumber):
                return RealNumber(self.value / other.value)
            elif isinstance(other, float):
                return RealNumber(self.value / other)
            else:
                return NotImplemented

---
### `__abs__(self):`

Returns the absolute value of the RealNumber object.

> ##### **Returns:**

            - RealNumber: A new RealNumber object with the absolute value of the number.

            return RealNumber(abs(self.value))

---
### `__neg__(self):`

Returns the negation of the RealNumber object.

> ##### **Returns:**

            - RealNumber: A new RealNumber object with the negation of the number.

            return RealNumber(-self.value)

---
### `sqrt(self):`

Returns the square root of the RealNumber object.

> ##### **Returns:**

            - RealNumber: A new RealNumber object with the square root of the number.

            return RealNumber(self.value ** 0.5)
    
---
### `__pow__(self, other):`

Computes the power of the real number to the given exponent.

> ##### **Parameters:**

            - other (float): The exponent.

> ##### **Returns:**

            - RealNumber: A new RealNumber object with the result of the power operation.

            return RealNumber(self.value ** other)


---
## **RationalNumber class:**

A class representing a rational number.

> ##### **Attributes:**

        - numerator (int): The numerator of the rational number.
        - denominator (int): The denominator of the rational number.

> ##### **Methods:**

        - simplify: Simplifies the rational number.
        - add: Adds two rational numbers.
        - subtract: Subtracts two rational numbers.
        - multiply: Multiplies two rational numbers.
        - divide: Divides two rational numbers.

---
### `__init__(self, numerator, denominator):`

Initializes a rational number with the given numerator and denominator.

> ##### **Arguments:**

            - numerator (int): The numerator of the rational number.
            - denominator (int): The denominator of the rational number.

> ##### **Raises:**

            - ValueError: If the denominator is zero.

            if denominator == 0:
                raise ValueError("Denominator cannot be zero")
            self.numerator = numerator
            self.denominator = denominator
            self.simplify()

---
### `__str__(self):`

Returns the string representation of the rational number.

> ##### **Returns:**

        return f"{self.numerator}/{self.denominator}"

---
### `simplify(self):`

Simplifies the rational number.

> ##### **Returns:**

        gcd = self.gcd(self.numerator, self.denominator)
        self.numerator //= gcd
        self.denominator //= gcd

---
### `gcd(a, b):`

Computes the greatest common divisor of two numbers a and b.

> ##### **Arguments:**

            - a (int): The first number.
            - b (int): The second number.

> ##### **Returns:**

            - int: The greatest common divisor of a and b.

            while b:
                a, b = b, a % b
            return a

---
### `add(self, other):`

Adds two rational numbers.

> ##### **Arguments:**

            - other (RationalNumber): The other rational number.

> ##### **Returns:**

            - RationalNumber: The sum of the two rational numbers.

            numerator = self.numerator * other.denominator + other.numerator * self.denominator
            denominator = self.denominator * other.denominator
            return RationalNumber(numerator, denominator)

---
### `subtract(self, other):`

Subtracts two rational numbers.

> ##### **Arguments:**

            - other (RationalNumber): The other rational number.

> ##### **Returns:**

            - RationalNumber: The difference of the two rational numbers.

            numerator = self.numerator * other.denominator - other.numerator * self.denominator
            denominator = self.denominator * other.denominator
            return RationalNumber(numerator, denominator)

---
### `multiply(self, other):`

Multiplies two rational numbers.

> ##### **Arguments:**

            - other (RationalNumber): The other rational number.

> ##### **Returns:**

            - RationalNumber: The product of the two rational numbers.

            numerator = self.numerator * other.numerator
            denominator = self.denominator * other.denominator
            return RationalNumber(numerator, denominator)

---
### `divide(self, other):`

Divides two rational numbers.

> ##### **Arguments:**

            - other (RationalNumber): The other rational number.

> ##### **Returns:**

            - RationalNumber: The quotient of the two rational numbers.

            numerator = self.numerator * other.denominator
            denominator = self.denominator * other.numerator
            return RationalNumber(numerator, denominator)

##
---
## **IntegralNumber class:**

A class representing integral numbers.

> ##### **Attributes**

    - value : int
        The value of the integral number.

> ##### **Methods**

    __init__(self, value: int) -> None:
        Initializes a new instance of the IntegralNumber class with the specified integer value.

    __repr__(self) -> str:
        Returns a string representation of the IntegralNumber object.

    __eq__(self, other: 'IntegralNumber') -> bool:
        Determines if the current IntegralNumber object is equal to another IntegralNumber object.

    __lt__(self, other: 'IntegralNumber') -> bool:
        Determines if the current IntegralNumber object is less than another IntegralNumber object.

    __add__(self, other: 'IntegralNumber') -> 'IntegralNumber':
        Adds two IntegralNumber objects and returns a new IntegralNumber object.

    __sub__(self, other: 'IntegralNumber') -> 'IntegralNumber':
        Subtracts two IntegralNumber objects and returns a new IntegralNumber object.

    __mul__(self, other: 'IntegralNumber') -> 'IntegralNumber':
        Multiplies two IntegralNumber objects and returns a new IntegralNumber object.

    __truediv__(self, other: 'IntegralNumber') -> 'IntegralNumber':
        Divides two IntegralNumber objects and returns a new IntegralNumber object.

> ##### **Raises**

    - TypeError
        If the argument is not an instance of IntegralNumber.
    - ZeroDivisionError
        If the second IntegralNumber object is zero and division is attempted.

        Divides two IntegralNumber objects and returns a new IntegralNumber object.

> ##### **References**

    - https://en.wikipedia.org/wiki/Integer_(computer_science)
    - https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

---
### `__init__(self, value: int) -> None:`

Initializes a new instance of the IntegralNumber class with the specified integer value.

> ##### **Parameters**

        - value : int
            The integer value to initialize the IntegralNumber object with.

        self.value = value
        
---
### `__repr__(self) -> str:`

Returns a string representation of the IntegralNumber object.

> ##### **Returns**

        - str
            A string representation of the IntegralNumber object.

        return f"IntegralNumber({self.value})"
   
---
### `__eq__(self, other: 'IntegralNumber') -> bool:`

Determines if the current IntegralNumber object is equal to another IntegralNumber object.

> ##### **Parameters**

        - other : IntegralNumber
            The IntegralNumber object to compare to.

> ##### **Returns**

        - bool
            True if the objects are equal, False otherwise.

        if isinstance(other, IntegralNumber):
            return self.value == other.value
        return False
        
---
### `__lt__(self, other: 'IntegralNumber') -> bool:`

Determines if the current IntegralNumber object is less than another IntegralNumber object.

> ##### **Parameters**

        - other : IntegralNumber
            The IntegralNumber object to compare to.

> ##### **Returns**

        - bool
            True if the current object is less than the other object, False otherwise.

        if isinstance(other, IntegralNumber):
            return self.value < other.value
        return False
        
---
### `__add__(self, other: 'IntegralNumber') -> 'IntegralNumber':`

Adds two IntegralNumber objects.

> ##### **Parameters**

        - other : IntegralNumber
            The IntegralNumber object to be added to the current object.

> ##### **Returns**

        - IntegralNumber
            An IntegralNumber object which is the sum of the current object and the passed object.

> ##### **Raises**

        - TypeError
            If the passed object is not an IntegralNumber.


        if isinstance(other, IntegralNumber):
            return IntegralNumber(self.value + other.value)
        raise TypeError("Cannot add non-IntegralNumber object.")
        
---
### `__sub__(self, other: 'IntegralNumber') -> 'IntegralNumber':`

Subtracts two IntegralNumber objects.

> ##### **Parameters**

        - other : IntegralNumber
            The IntegralNumber object to be subtracted from the current object.

> ##### **Returns**

        - IntegralNumber
            An IntegralNumber object which is the difference between the current object and the passed object.

> ##### **Raises**

        - TypeError
            If the passed object is not an IntegralNumber.

        if isinstance(other, IntegralNumber):
            return IntegralNumber(self.value - other.value)
        raise TypeError("Cannot subtract non-IntegralNumber object.")
        
---
### `__mul__(self, other: 'IntegralNumber') -> 'IntegralNumber':`

Multiplies two IntegralNumber objects.

> ##### **Parameters**

        - other : IntegralNumber
            The IntegralNumber object to be multiplied with the current object.

> ##### **Returns**

        - IntegralNumber
            An IntegralNumber object which is the product of the current object and the passed object.

> ##### **Raises**

        - TypeError
            If the passed object is not an IntegralNumber.

        if isinstance(other, IntegralNumber):
            return IntegralNumber(self.value * other.value)
        raise TypeError("Cannot multiply non-IntegralNumber object.")
        
---
### `__truediv__(self, other: 'IntegralNumber') -> 'IntegralNumber':`

Divides two IntegralNumber objects.

> ##### **Parameters**

        - other : IntegralNumber
            The IntegralNumber object to be used as divisor for the current object.

> ##### **Returns**

        - IntegralNumber
            An IntegralNumber object which is the result of dividing the current object by the passed object.

> ##### **Raises**

        - TypeError
            If the passed object is not an IntegralNumber.
        ZeroDivisionError
            If the passed object has a value of zero.

        if isinstance(other, IntegralNumber):
            if other.value == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return IntegralNumber(self.value // other.value)
        raise TypeError("Cannot divide non-IntegralNumber object.")

---
### `add(self, other: 'IntegralNumber') -> 'IntegralNumber':`

Returns the sum of this number and `other`.

> ##### **Parameters:**

        - other : IntegralNumber
            The number to add to this number.

> ##### **Returns:**

        - IntegralNumber
            The sum of this number and `other`.

        return IntegralNumber(self.value + other.value)

---
### `subtract(self, other: 'IntegralNumber') -> 'IntegralNumber':`

Returns the difference between this number and `other`.

> ##### **Parameters:**

        - other : IntegralNumber
            The number to subtract from this number.

> ##### **Returns:**

        - IntegralNumber
            The difference between this number and `other`.

        return IntegralNumber(self.value - other.value)

---
### `multiply(self, other: 'IntegralNumber') -> 'IntegralNumber':`

Returns the product of this number and `other`.

> ##### **Parameters:**

        - other : IntegralNumber
            The number to multiply with this number.

> ##### **Returns:**

        - IntegralNumber
            The product of this number and `other`.

        return IntegralNumber(self.value * other.value)

---
### `divide(self, other: 'IntegralNumber') -> Union['IntegralNumber', None]:`

Returns the quotient of this number and `other`.

> ##### **Parameters:**

        - other : IntegralNumber
            The number to divide this number by.

> ##### **Returns:**

        - Union[IntegralNumber, None]
            The quotient of this number and `other`. Returns None if `other` is zero.

        if other.value == 0:
            return None
        return IntegralNumber(self.value // other.value)

---
### `__str__(self):`

> ##### **Returns:**

      - str(self.value)
