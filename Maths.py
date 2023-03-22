#from fractions import Fraction
from typing import List, Tuple, Union
#import math
#import random


class Algorithm:
    """Class full of mathematical functions"""
    
    def __init__(self):
        pass

    def addition(self, *args: Union[int, float]) -> Union[int, float]:
        """Returns the sum of integers and/or floats"""
        try:
            total = sum(args)
        except TypeError:
            raise TypeError("Input must be numbers")
        return total

    def subtract(self, *args: Union[int, float]) -> Union[int, float]:
        """Returns integers or float of given numbers after being subtracted"""
        try:
            total = args[0] - sum(args[1:])
        except TypeError:
            raise TypeError("Input must be numbers")
        return total

    def multiply(self, *args: Union[int, float]) -> Union[int, float]:
        """Returns an integer or float of given numbers multiplied"""
        try:
            total = 1
            for i in args:
                total *= i
        except TypeError:
            raise TypeError("Input must be numbers")
        return total

    def division_float(self, dividend: Union[int, float], divisor: Union[int, float]) -> Union[int, float]:
        """Returns a float of dividend divided by divisor"""
        try:
            result = dividend / divisor
        except ZeroDivisionError:
            raise ZeroDivisionError("Cannot divide by zero")
        except TypeError:
            raise TypeError("Input must be a number")
        return result

    def division_int(self, dividend: Union[int, float], divisor: Union[int, float]) -> Union[int, float]:
        """Returns an integer of dividend divided by divisor"""
        try:
            result = dividend // divisor
        except ZeroDivisionError:
            raise ZeroDivisionError("Cannot divide by zero")
        except TypeError:
            raise TypeError("Input must be a number")
        return result

    def division_remainder(self, dividend: Union[int, float], divisor: Union[int, float]) -> Union[int, float]:
        """Returns the remainder of dividend divided by divisor"""
        try:
            result = dividend % divisor
        except ZeroDivisionError:
            raise ZeroDivisionError("Cannot divide by zero")
        except TypeError:
            raise TypeError("Input must be a number")
        return result

    def power(self, base: Union[int, float], exponent: Union[int, float]) -> Union[int, float]:
        """Returns base to the power of exponent"""
        try:
            result = base ** exponent
        except TypeError:
            raise TypeError("Input must be a number")
        return result

    @staticmethod
    def log(self, x, base=10):
        """
        Returns the logarithm of x with a specified base (default is 10)

        Args:
            x (int/float): The value for which to compute the logarithm
            base (int/float, optional): The base of the logarithm. Defaults to 10.

        Returns:
            The logarithm of x with the specified base
        """
        if x <= 0 or base <= 0 or base == 1:
            raise ValueError("Invalid input. x must be a positive number and base must be greater than 0 and not equal to 1.")
        
        # Compute the logarithm using the change-of-base formula
        numerator = Algorithm.__ln(x)
        denominator = Algorithm.__ln(base)
        return numerator / denominator

    @staticmethod
    def __ln(self, x):
        """
        Returns the natural logarithm of x (base e)

        Args:
            x (int/float): The value for which to compute the natural logarithm

        Returns:
            The natural logarithm of x
        """
        if x == 1:
            return 0
        elif x < 1:
            return -Algorithm.__ln(1/x)
        else:
            return 1 + Algorithm.__ln(x/2)

    @staticmethod
    def __log10(self, x):
        """
        Returns the logarithm of x (base 10)

        Args:
            x (int/float): The value for which to compute the logarithm

        Returns:
            The logarithm of x (base 10)
        """
        return Algorithm.log(x, 10)

    def adding_fractions(self, *args):
        """
        Returns the sum of multiple fractions

        Args:
            *args (tuples): Multiple fractions represented as tuples of the form (numerator, denominator)

        Returns:
            A tuple representing the sum of all fractions in reduced form (numerator, denominator)
        """
        numerator = 0
        denominator = 1
        for fraction in args:
            if not isinstance(fraction, tuple) or len(fraction) != 2:
                raise ValueError("All arguments must be tuples of length 2.")
            numerator = numerator * fraction[1] + fraction[0] * denominator
            denominator = denominator * fraction[1]
        gcd = Algorithm.find_gcd(numerator, denominator)
        return (numerator // gcd, denominator // gcd)

    def find_gcd(self, a, b):
        """
        Finds the greatest common divisor of two numbers using Euclid's algorithm.

        Args:
            a: An integer
            b: Another integer

        Returns:
            The greatest common divisor of a and b
        """
        while b != 0:
            a, b = b, a % b
        return a

    def count_words(text):
        """
        Returns a dictionary containing the count of each word in the given text.

        Args:
            text (str): The text to count the words in.

        Returns:
            A dictionary where the keys are the unique words in the text and the values are the count of each word.
        """
        # Convert text to lowercase and split into words
        words = text.lower().split()

        # Create an empty dictionary to store the word counts
        word_counts = {}

        # Iterate over the words and update the word counts
        for word in words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

        return word_counts

    def multiplying_fractions(self, *args):
        """
        Returns the product of multiple fractions.

        Args:
            *args: An arbitrary number of arguments. Each argument must be a tuple with two values, the numerator
                and denominator of a fraction.

        Returns:
            A tuple containing the numerator and denominator of the product of the fractions.

        Raises:
            ValueError: If any of the arguments are not tuples of length 2 or if any of the denominators are 0.
        """
        numerator = 1
        denominator = 1
        for arg in args:
            if not isinstance(arg, tuple) or len(arg) != 2:
                raise ValueError("All arguments must be tuples of length 2.")
            if arg[1] == 0:
                raise ValueError("Cannot divide by zero.")
            numerator *= arg[0]
            denominator *= arg[1]
            
        # Find the greatest common divisor (GCD) manually
        a = numerator
        b = denominator
        while b:
            a, b = b, a % b
        
        return (numerator // a, denominator // a)

    def divide_fractions(self, *args: tuple[int, int]) -> tuple[int, int]:
        """
        Returns the result of dividing one fraction by another.

        Args:
            *args: Two tuples, each with two values, representing the numerator and denominator of the two fractions.

        Returns:
            A tuple containing the numerator and denominator of the quotient of the two fractions.

        Raises:
            ValueError: If any of the arguments are not tuples of length 2.
            ZeroDivisionError: If the denominator of the second fraction is zero.
        """
        if len(args) != 2 or not all(isinstance(arg, tuple) and len(arg) == 2 for arg in args):
            raise ValueError("Two tuples of length 2 are required.")
        if args[1][1] == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        numerator = args[0][0] * args[1][1]
        denominator = args[0][1] * args[1][0]
        return (numerator, denominator)

    def proportion_rule(self, a: int, b: int, c: int = None, d: int = None) -> int:
        """
    Returns the fourth proportional number given three proportional numbers.
    
    Args:
        a (int): The first proportional number.
        b (int): The second proportional number.
        c (int, optional): The third proportional number. Defaults to None.
        d (int, optional): The fourth proportional number. Defaults to None.
        
    Returns:
        int: The fourth proportional number calculated from the input.
    
    If both `c` and `d` are None, `a` and `b` are assumed to be the first two proportional numbers, 
    and `c` and `d` are set to `b` and `a` respectively. If `d` is None, `a` and `b` are assumed 
    to be the first two proportional numbers, and `d` is calculated from `c` using the formula 
    `d = (b * c) / a`. If `c` and `d` are both specified, `a` and `b` are assumed to be the first 
    two proportional numbers, and the function calculates the fourth proportional number `x` using 
    the formula `x = (b * d) / c`.
    """
        if c is None and d is None:
            # a:b = c:x
            c, d = b, a
        elif d is None:
            # a:b = c:d
            d = (b * c) / a
        else:
            # a:b = c:d, x is the fourth proportional number
            x = (b * d) / c
            return x
        return d

    def percentage_to_fraction(self, x: float) -> float:
        """Converts percentage to fraction"""
        return x / 100

    def fraction_to_percentage(self, numerator: int, denominator: int) -> float:
        """
        Converts a fraction to a percentage.

        Args:
            numerator: The numerator of the fraction.
            denominator: The denominator of the fraction.

        Returns:
            The fraction as a percentage.
        """
        if denominator == 0:
            raise ZeroDivisionError("Denominator cannot be zero.")
        return (numerator / denominator) * 100
    
    def linear_search(self, lst, target):
        """
        Searches for the target element in the given list and returns the index if found,
        otherwise returns -1.
        """
        for i, elem in enumerate(lst):
            if elem == target:
                return i
        return -1
    
    def binary_search(self, lst, target):
        """
        Searches for the target element in the given list using binary search and returns
        the index if found, otherwise returns -1.
        """
        low = 0
        high = len(lst) - 1
        while low <= high:
            mid = (low + high) // 2
            if lst[mid] == target:
                return mid
            elif lst[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return -1
    
    def bubble_sort(self, lst):
        """
        Sorts the given list in ascending order using bubble sort and returns the sorted list.
        """
        n = len(lst)
        for i in range(n):
            # Set a flag to check if any swap has been made in this iteration
            swapped = False
            for j in range(n-i-1):
                if lst[j] > lst[j+1]:
                    lst[j], lst[j+1] = lst[j+1], lst[j]
                    swapped = True
            # If no swap has been made in this iteration, the list is already sorted
            if not swapped:
                break
        return lst
    
    def insertion_sort(self, lst):
        """
        Sorts the given list in ascending order using insertion sort and returns the sorted list.
        """
        n = len(lst)
        for i in range(1, n):
            key = lst[i]
            j = i - 1
            while j >= 0 and lst[j] > key:
                lst[j+1] = lst[j]
                j -= 1
            lst[j+1] = key
        return lst

    def merge_sort(self, lst):
        """
        Sorts the given list in ascending order using merge sort and returns the sorted list.
        """
        if len(lst) <= 1:
            return lst

        # Divide the list into two halves
        mid = len(lst) // 2
        left_half = lst[:mid]
        right_half = lst[mid:]

        # Recursively sort the two halves
        left_half = Algorithm.merge_sort(left_half)
        right_half = Algorithm.merge_sort(right_half)

        # Merge the sorted halves
        merged_list = []
        i = j = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                merged_list.append(left_half[i])
                i += 1
            else:
                merged_list.append(right_half[j])
                j += 1

        # Append any remaining elements in the left or right half
        merged_list.extend(left_half[i:])
        merged_list.extend(right_half[j:])

        return merged_list
    
    def square_root(num):
        """
        Compute the square root of a given number using the Babylonian method.

        Args:
            num (float): The number to find the square root of.

        Returns:
            float: The square root of the given number.
        """
        if num < 0:
            raise ValueError("Cannot find square root of a negative number")
        if num == 0:
            return 0
        x = num
        while True:
            y = (x + num / x) / 2
            if abs(x - y) < 1e-6:
                return y
            x = y

    def factorial(num):
        """
        Compute the factorial of a given number.

        Args:
            num (int): The number to find the factorial of.

        Returns:
            int: The factorial of the given number.
        """
        if num < 0:
            raise ValueError("Cannot find factorial of a negative number")
        result = 1
        for i in range(1, num + 1):
            result *= i
        return result

    def fibonacci(n):
        """
        Compute the nth number in the Fibonacci sequence.

        Args:
            n (int): The index of the desired Fibonacci number.

        Returns:
            int: The nth number in the Fibonacci sequence.
        """
        if n < 0:
            raise ValueError("Index of Fibonacci sequence cannot be negative")
        if n <= 1:
            return n
        prev1 = 0
        prev2 = 1
        for i in range(2, n + 1):
            curr = prev1 + prev2
            prev1 = prev2
            prev2 = curr
        return curr

    def is_prime(num):
        """
        Check whether a given number is prime.

        Args:
            num (int): The number to check for primality.

        Returns:
            bool: True if the number is prime, False otherwise.
        """
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    def gcd(*args):
        """
        Compute the greatest common divisor of two or more numbers.

        Args:
            *args (int): Two or more numbers to find the GCD of.

        Returns:
            int: The greatest common divisor of the given numbers.
        """
        if len(args) < 2:
            raise ValueError("At least two numbers are required to find the GCD")
        result = args[0]
        for num in args[1:]:
            while num:
                result, num = num, result % num
        return result

    def lcm(*args):
        """
        Compute the least common multiple of two or more numbers.

        Args:
            *args (int): Two or more numbers to find the LCM of.

        Returns:
            int: The least common multiple of the given numbers.
        """
        if len(args) < 2:
            raise ValueError("At least two numbers are required to find the LCM")
        result = args[0]
        for num in args[1:]:
            result = result * num // gcd(result, num)
        return result

    def sort_numbers(numbers: List[Union[int, float]], reverse: bool = False) -> List[Union[int, float]]:
        """
        This function takes a list of numbers and returns a sorted list in ascending or descending order.

        Parameters:
        numbers (List[Union[int, float]]): A list of integers or floats to be sorted.
        reverse (bool, optional): If True, returns the list in descending order. Defaults to False.

        Returns:
        List[Union[int, float]]: A sorted list in ascending or descending order.

        Example:
        >>> sort_numbers([5, 2, 9, 1, 5.5])
        [1, 2, 5, 5.5, 9]
        >>> sort_numbers([5, 2, 9, 1, 5.5], True)
        [9, 5.5, 5, 2, 1]
        """
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                if reverse:
                    if numbers[j] > numbers[i]:
                        numbers[j], numbers[i] = numbers[i], numbers[j]
                else:
                    if numbers[j] < numbers[i]:
                        numbers[j], numbers[i] = numbers[i], numbers[j]
        return numbers

    def binary_search(numbers: List[Union[int, float]], target: Union[int, float]) -> int:
        """
        This function takes a sorted list of numbers and a target number and returns the index of the target number,
        or -1 if it is not found.

        Parameters:
        numbers (List[Union[int, float]]): A sorted list of integers or floats.
        target (Union[int, float]): The number to search for in the list.

        Returns:
        int: The index of the target number in the list, or -1 if it is not found.

        Example:
        >>> binary_search([1, 2, 3, 4, 5], 3)
        2
        >>> binary_search([1, 2, 3, 4, 5], 6)
        -1
        """
        start = 0
        end = len(numbers) - 1
        while start <= end:
            mid = (start + end) // 2
            if numbers[mid] == target:
                return mid
            elif numbers[mid] < target:
                start = mid + 1
            else:
                end = mid - 1
        return -1

    def linear_regression(x, y):
        """
        Calculates the equation of the line of best fit (y = mx + b) for the given x and y values.

        Args:
        x (list): A list of x values.
        y (list): A list of corresponding y values.

        Returns:
        tuple: A tuple containing the slope (m) and y-intercept (b) of the line of best fit.
        """
        if len(x) != len(y):
            raise ValueError("x and y lists must have the same length")

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_x_squared = sum([i**2 for i in x])
        sum_xy = sum([x[i] * y[i] for i in range(n)])

        m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
        b = (sum_y - m * sum_x) / n

        return m, b

    def matrix_addition(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """
        This function takes in two matrices A and B of the same size, and returns their sum.

        Args:
        A: A list of lists of floats representing the first matrix.
        B: A list of lists of floats representing the second matrix.

        Returns:
        A list of lists of floats representing the sum of the matrices.
        """

        n_rows = len(A)
        n_cols = len(A[0])
        C = [[0 for j in range(n_cols)] for i in range(n_rows)]

        for i in range(n_rows):
            for j in range(n_cols):
                C[i][j] = A[i][j] + B[i][j]

        return C


    def matrix_addition(A, B):
        """
        Adds two matrices A and B of the same size element-wise and returns the resulting matrix.

        Args:
        A (list[list[float]]): The first matrix.
        B (list[list[float]]): The second matrix.

        Returns:
        list[list[float]]: The matrix sum of A and B.
        """
        if len(A) != len(B) or len(A[0]) != len(B[0]):
            raise ValueError("Matrices must be of the same size.")
        result = []
        for i in range(len(A)):
            row = []
            for j in range(len(A[0])):
                row.append(A[i][j] + B[i][j])
            result.append(row)
        return result


    def matrix_multiplication(A, B):
        """
        Multiplies two matrices A and B and returns the resulting matrix.

        Args:
        A (list[list[float]]): The first matrix.
        B (list[list[float]]): The second matrix.

        Returns:
        list[list[float]]: The matrix product of A and B.
        """
        if len(A[0]) != len(B):
            raise ValueError("Number of columns in matrix A must be equal to number of rows in matrix B.")
        result = []
        for i in range(len(A)):
            row = []
            for j in range(len(B[0])):
                element = 0
                for k in range(len(B)):
                    element += A[i][k] * B[k][j]
                row.append(element)
            result.append(row)
        return result


    def matrix_inversion(A):
        """
        Inverts a matrix A and returns the resulting matrix.

        Args:
        A (list[list[float]]): The matrix to be inverted.

        Returns:
        list[list[float]]: The inverted matrix of A.
        """
        if len(A) != len(A[0]):
            raise ValueError("Matrix must be square.")
        n = len(A)
        A_augmented = [row + [1 if i == j else 0 for i in range(n)] for j, row in enumerate(A)]
        for i in range(n):
            pivot = A_augmented[i][i]
            if pivot == 0:
                raise ValueError("Matrix is not invertible.")
            for j in range(i + 1, n):
                factor = A_augmented[j][i] / pivot
                for k in range(2 * n):
                    A_augmented[j][k] -= factor * A_augmented[i][k]
        for i in range(n - 1, -1, -1):
            pivot = A_augmented[i][i]
            for j in range(i - 1, -1, -1):
                factor = A_augmented[j][i] / pivot
                for k in range(2 * n):
                    A_augmented[j][k] -= factor * A_augmented[i][k]
        A_inverse = [[A_augmented[i][j] / A_augmented[i][i + n] for j in range(n)] for i in range(n)]
        return A_inverse
    
    def newton_method(self, f, f_prime, x0, epsilon):
        """
        Use Newton's method to find the root of a function f.

        Args:
        - f (function): The function for which to find the root.
        - f_prime (function): The derivative of f.
        - x0 (float): The initial guess for the root.
        - epsilon (float): The desired level of accuracy.

        Returns:
        - root (float): The estimated root of the function.
        """

        x = x0
        while abs(f(x)) > epsilon:
            x = x - f(x)/f_prime(x)

        return x
    
    def gradient_descent(self, f, f_prime, x0, alpha, max_iters):
        """
        Use gradient descent to find the minimum of a function f.

        Args:
        - f (function): The function to minimize.
        - f_prime (function): The derivative of f.
        - x0 (float): The initial guess for the minimum.
        - alpha (float): The step size.
        - max_iters (int): The maximum number of iterations.

        Returns:
        - minimum (float): The estimated minimum of the function.
        """

        x = x0
        for i in range(max_iters):
            x = x - alpha * f_prime(x)

        return x
    
    def monte_carlo_simulation(self, n, f):
        """
        Use Monte Carlo simulation to estimate the probability of an event.

        Args:
        - n (int): The number of simulations to run.
        - f (function): A function that returns True or False for a given sample.

        Returns:
        - probability (float): The estimated probability of the event.
        """

        count = 0
        for i in range(n):
            if f():
                count += 1

        probability = count / n
        return probability
    
    def distance(point1, point2):
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

    def random_seed(seed):
        """
        A simple pseudorandom number generator based on the linear congruential method.

        Args:
        - seed (int): The seed value used to initialize the generator.

        Returns:
        - A float between 0 and 1.
        """
        a = 1103515245
        c = 12345
        m = 2**31
        x = seed
        while True:
            x = (a*x + c) % m
            yield x / m
    
    def k_means_clustering(self, data, k):
        """
        Use k-means clustering to group data points into k clusters.

        Args:
        - data (list): A list of data points.
        - k (int): The number of clusters to form.

        Returns:
        - clusters (list): A list of k clusters, each containing the data points assigned to that cluster.
        """

        # Initialize cluster centers randomly
        centers = alg.random_seed.sample(data, k)

        while True:
            # Assign each data point to the nearest center
            clusters = [[] for i in range(k)]
            for point in data:
                distances = [alg.distance(point, center) for center in centers]
                min_index = distances.index(min(distances))
                clusters[min_index].append(point)

            # Update cluster centers
            new_centers = []
            for cluster in clusters:
                if len(cluster) == 0:
                    new_centers.append(alg.random.choice(data))
                else:
                    center = [sum(coords)/len(coords) for coords in zip(*cluster)]
                    new_centers.append(center)

            # Check for convergence
            if new_centers == centers:
                break
            else:
                centers = new_centers

        return clusters

   
    def exp(self, num: Union[int, float]) -> Union[int, float]:
        """
        Returns the exponential value of a number.

        Args:
        - num: a number whose exponential value is to be calculated

        Returns:
        The exponential value of the input number
        """
        result = 1
        for i in range(num):
            result *= 2.71828
        return result

    def absolute(self, num: Union[int, float]) -> Union[int, float]:
        """
        Returns the absolute value of a number.

        Args:
        - num: a number whose absolute value is to be calculated

        Returns:
        The absolute value of the input number
        """
        if num < 0:
            return -num
        else:
            return num


    def modulo(self, dividend: Union[int, float], divisor: Union[int, float]) -> Union[int, float]:
        """
        Returns the remainder of dividing the dividend by the divisor.

        Args:
        - dividend: the number to be divided
        - divisor: the number to divide by

        Returns:
        The remainder of dividing the dividend by the divisor
        """
        return dividend % divisor


    def sin(self, num: Union[int, float]) -> Union[int, float]:
        """
        Returns the sine value of a number.

        Args:
        - num: a number in radians whose sine value is to be calculated

        Returns:
        The sine value of the input number
        """
        n = 10  # number of terms to calculate in Taylor series
        x = num % (2*con.pi)  # reduce to the range [0, 2*pi)
        result = 0
        for i in range(n):
            term = (-1)**i * x**(2*i+1) / alg.factorial(2*i+1)
            result += term
        return result


    def cos(self, num: Union[int, float]) -> Union[int, float]:
        """
        Returns the cosine value of a number.

        Args:
        - num: a number in radians whose cosine value is to be calculated

        Returns:
        The cosine value of the input number
        """
        n = 10  # number of terms to calculate in Taylor series
        x = num % (2*con.pi)  # reduce to the range [0, 2*pi)
        result = 0
        for i in range(n):
            term = (-1)**i * x**(2*i) / alg.factorial(2*i)
            result += term
        return result


    def tan(self, num: Union[int, float]) -> Union[int, float]:
        """
        Returns the tangent value of a number.

        Args:
        - num: a number in radians whose tangent value is to be calculated

        Returns:
        The tangent value of the input number
        """
        sin_val = self.sin(num)
        cos_val = self.cos(num)
        return sin_val / cos_val




class Constants:
    """A collection of mathematical constants."""
    
    def __init__(self):
        pass
    
    def speed_of_light(self):
        """Returns the speed of light in meters per second."""
        return 299_792_458
    
    def planck_constant(self):
        pass
    
    def pi(self):
        """The ratio of a circle's circumference to its diameter.
        Returns:
            Pi, π, to the 20th decimal
        """
        return 3.141_592_653_589_793_238_46
    
    def tau(self):
        """the 19th letter of the Greek alphabet,
        representing the voiceless dental or alveolar plosive IPA: [t].
        In the system of Greek numerals, it has a value of 300.
        
        Returns:
            tau, uppercase Τ, lowercase τ, or τ, to the 20th decimal
        """
        return 6.283_185_307_179_586_476_92
    
    def phi(self):
        """\"The Golden Ratio\".
        In mathematics, two quantities are in the golden ratio
        if their ratio is the same as the ratio of their sum
        to the larger of the two quantities.
        
        Returns:
            Uppercase Φ lowercase φ or ϕ: Value to the 20th decimal
        """
        return 1.618_033_988_749_894_848_20
    
    def silver_ratio(self):
        """\"The Silver Ratio\". Two quantities are in the silver ratio (or silver mean)
        if the ratio of the smaller of those two quantities to the larger quantity
        is the same as the ratio of the larger quantity to the sum of the
        smaller quantity and twice the larger quantity
        
        Returns:
            δS: Value to the 20th decimal
        """
        return 2.414_213_562_373_095_048_80
    
    def supergolden_ratio(self):
        """Returns the mathematical constant psi (the supergolden ratio).
        
        Returns:
            ψ to the 25th decimal
        """
        return 1.465_571_231_876_768_026_656_731_2
    
    def connective_constant(self):
        """Returns the connective constant for the hexagonal lattice.

        Returns:
            μ to the 4th decimal
        """
        return 1.687_5
    
    def kepler_bouwkamp_constant(self):
        """In plane geometry, the Kepler–Bouwkamp constant (or polygon inscribing constant)
        is obtained as a limit of the following sequence.
        Take a circle of radius 1. Inscribe a regular triangle in this circle.
        Inscribe a circle in this triangle. Inscribe a square in it.
        Inscribe a circle, regular pentagon, circle, regular hexagon and so forth.
        Returns:
            K': to the 20th decimal
        """
        return 0.114_942_044_853_296_200_70
    
    def wallis_constant(self):
        """Returns Wallis's constant.
        
        Returns:
            Value to the 20th decimal
        """
        return 2.094_551_481_542_326_591_48
    
    def eulers_number(self):
        """a mathematical constant approximately equal to 2.71828 that can be characterized in many ways.
        It is the base of the natural logarithms.
        It is the limit of (1 + 1/n)n as n approaches infinity, an expression that arises in the study of compound interest.
        It can also be calculated as the sum of the infinite series

        Returns:
            e: Value to the 20th decimal. math.e
        """
        return 2.718_281_828_459_045_235_36
    
    def natural_log(self):
        """Natural logarithm of 2.

        Returns:
            ln 2: Value to the 30th decimal. math.log(2)
        """
        return 0.693_147_180_559_945_309_417_232_121_458
    
    def lemniscate_constant(self):
        """The ratio of the perimeter of Bernoulli's lemniscate to its diameter, analogous to the definition of π for the circle.

        Returns:
            ϖ: Value to the 20th decimal. math.sqrt(2)
        """
        return 2.622_057_554_292_119_810_46 
    
    def eulers_constant(self):
        """Not to be confused with Euler's Number.
        Defined as the limiting difference between the harmonic series and the natural logarithm

        Returns:
            γ: Value to the 50th decimal
        """
        return 0.577_215_664_901_532_860_606_512_090_082_402_431_042_159_335_939_92
    
    def Erdős_Borwein_constant(self):
        """The sum of the reciprocals of the Mersenne numbers

        Returns:
            E: Value to the 20th decimal. sum([1 / 2 ** (2 ** i) for i in range(40)])
        """
        return 1.606_695_152_415_291_763_78
    
    def omega_constant(self):
        """Defined as the unique real number that satisfies the equation Ωe**Ω = 1.

        Returns:
            Ω: Value to the 30th decimal
        """
        return 0.567_143_290_409_783_872_999_968_662_210
    
    def Apérys_constant(self):
        """The sum of the reciprocals of the positive cubes.

        Returns:
            ζ(3): Value to the 45th decimal
        """
        return 1.202_056_903_159_594_285_399_738_161_511_449_990_764_986_292
    
    def laplace_limit(self):
        """The maximum value of the eccentricity for which a solution to Kepler's equation, in terms of a power series in the eccentricity, converges.

        Returns:
            Value to the 35th decimal
        """
        return 0.662_743_419_349_181_580_974_742_097_109_252_90
    
    def ramanujan_soldner_constant(self):
        """A mathematical constant defined as the unique positive zero of the logarithmic integral function.

        Returns:
            μ ≈: Value to the 45th decimal
        """
        return 1.451_369_234_883_381_050_283_968_485_892_027_449_493_032_28
        
    def gauss_constant(self):
        """transcendental mathematical constant that is the ratio of the perimeter of
        Bernoulli's lemniscate to its diameter, analogous to the definition of π for the circle.

        Returns:
            G == ϖ /π ≈ 0.8346268: Value to the 7th decimal
        """
        return 0.834_626_8
    
    def second_hermite_constant(self):
        """_summary_

        Returns:
            γ2 : Value to the 20th decimal
        """
        return  1.154_700_538_379_251_529_01
    
    def liouvilles_constant(self):
        """A real number x with the property that, for every positive integer n,
        there exists a pair of integers (p,q) with q>1.

        Returns:
            L: Value to the 119th decimal
        """
        return 0.110_001_000_000_000_000_000_001_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_01
    
    def first_continued_fraction(self):
        """_summary_

        Returns:
            C_{1}: _description_
        """
        return 0.697_774_657_964_007_982_01
    
    def ramanujans_constant(self):
        """The transcendental number, which is an almost integer, in that it is very close to an integer.

        Returns:
            e**{{\pi {\sqrt {163}}}}: Value to the 18th decimal
        """
        return  262_537_412_640_768_743.999_999_999_999_250_073
        
    def glaisher_kinkelin_constant(self):
        """A mathematical constant, related to the K-function and the Barnes G-function.

        Returns:
            A: Value to the 20th decimal
        """
        return 1.282_427_129_100_622_636_87
    
    def catalan_constant(n: int) -> float:
        """
        Computes the Catalan's constant to the specified number of decimal places using the formula:

        C = sum((-1)**k / (2*k + 1)**2 for k in range(n)) * 8 / 3

        Parameters:
        n (int): The number of terms to sum to approximate the constant.

        Returns:
        float: The computed value of the Catalan's constant.

        Example:
        >>> catalan_constant(1000000)
        0.915965594177219
   
        Returns:
            G: Value to the 39th decimal
        """
        # return 0.915_965_594_177_219_015_054_603_514_932_384_110_774
        constant = sum((-1)**k / (2*k + 1)**2 for k in range(n)) * 8 / 3
        return constant
    
    def dottie_number(self):
        """
            Calculates the unique real root of the equation cos(x) = x, known as the Dottie number, to the 20th decimal place.

            The Dottie number is a constant that arises in the study of iterative methods and has connections to chaos theory.

        Returns:
            float: The Dottie number, i.e., the unique real root of the equation cos(x) = x, to the 20th decimal place.
    
        Example:
            >>> dottie_number()
            0.73908513321516064165

        Arguments:
            None
        """
        return 0.739_085_133_215_160_641_65
    
    def meissel_mertens_constant(self):
        """_summary_

        Returns:
            M: Value to the 40th value
        """
        return 0.261_497_212_847_642_783_755_426_838_608_695_859_051_6
    
    def universal_parabolic_constant(self):
        """The ratio, for any parabola, of the arc length of the parabolic segment
        formed by the latus rectum to the focal parameter.

        Returns:
            P: Value to the 20th decimal
        """
        return  2.295_587_149_392_638_074_03
    
    def cahens_constant(self):
        """The value of an infinite series of unit fractions with alternating signs.

        Returns:
            C: Value to the 20th decimal
        """
        return  0.643_410_546_288_338_026_18
    
    
    
class MathFunctions:
    """
    A class containing various mathematical functions.
    """
    
    def __init__(self):
        pass

    def area_of_circle(self, r: float) -> float:
        """
        Calculates the area of a circle given its radius.

        Args:
        r: The radius of the circle.

        Returns:
        The area of the circle.
        """
        return 3.141592653589793238 * r ** 2

    def volume_of_sphere(self, r: float) -> float:
        """
        Calculates the volume of a sphere given its radius.

        Args:
        r: The radius of the sphere.

        Returns:
        The volume of the sphere.
        """
        return 4 / 3 * 3.141592653589793238 * r ** 3

    def perimeter_of_rectangle(self, l: float, b: float) -> float:
        """
        Calculates the perimeter of a rectangle given its length and breadth.

        Args:
        l: The length of the rectangle.
        b: The breadth of the rectangle.

        Returns:
        The perimeter of the rectangle.
        """
        return 2 * (l + b)

    def pythagoras_theorem_length(self, a: float, b: float) -> float:
        """
        Calculates the length of the hypotenuse of a right-angled triangle given the lengths of its two other sides.

        Args:
        a: The length of one of the sides of the triangle.
        b: The length of the other side of the triangle.

        Returns:
        The length of the hypotenuse of the triangle.
        """
        return (a ** 2 + b ** 2) ** 0.5

    def square_root(self, x: float) -> float:
        """
        Calculates the square root of a given number.

        Args:
        x: The number to take the square root of.

        Returns:
        The square root of x.
        """
        return x ** 0.5

    def factorial(self, n: int) -> int:
        """
        Calculates the factorial of a given number.

        Args:
        n: The number to calculate the factorial of.

        Returns:
        The factorial of n.
        """
        fact = 1
        for i in range(1, n+1):
            fact *= i
        return fact

    def gcd(self, a: int, b: int) -> int:
        """
        Calculates the greatest common divisor of two numbers.

        Args:
        a: The first number.
        b: The second number.

        Returns:
        The greatest common divisor of a and b.
        """
        while(b):
            a, b = b, a % b
        return a

    def lcm(self, a: int, b: int) -> int:
        """
        Calculates the least common multiple of two numbers.

        Args:
        a: The first number.
        b: The second number.

        Returns:
        The least common multiple of a and b.
        """
        return a * b // self.gcd(a, b)

    def exponential(self, x: float) -> float:
        """
        Calculates the value of e raised to a given power.

        Args:
        x: The exponent.

        Returns:
        The value of e raised to x.
        """
        e = 2.718281828459045235
        return e ** x
    
    def logarithm(self, x: float, base: float) -> float:
        """
        Calculates the logarithm of a given number to a given base.

        Args:
        x: The number to take the logarithm of.
        base: The base of the logarithm.

        Returns:
        The logarithm of x to the base.
        """
        return (Functions.log(x) / Functions.log(base))

    def log(x):
        """
        Calculates the natural logarithm of a given number.

        Args:
        x: The number to take the natural logarithm of.

        Returns:
        The natural logarithm of x.
        """
        if x <= 0:
            return float('nan')
        elif x == 1:
            return 0.0
        else:
            return Functions.integrate(1/x, 1, x)

    def integrate(f, a, b):
        """
        Approximates the definite integral of a function over a given interval using the trapezoidal rule.

        Args:
        f: The function to integrate.
        a: The lower limit of the interval.
        b: The upper limit of the interval.

        Returns:
        The approximate value of the definite integral of f over the interval [a, b].
        """
        n = 1000 # Number of trapezoids to use
        dx = (b - a) / n
        x_values = [a + i * dx for i in range(n+1)]
        y_values = [f(x) for x in x_values]
        return (dx/2) * (y_values[0] + y_values[-1] + 2*sum(y_values[1:-1]))

    def surface_area_of_cylinder(self, r: float, h: float) -> float:
        """
        Calculates the surface area of a cylinder given its radius and height.

        Args:
        r: The radius of the cylinder.
        h: The height of the cylinder.

        Returns:
        The surface area of the cylinder.
        """
        return 2 * 3.14159265358979323846 * r * (r + h)


    def volume_of_cylinder(self, r: float, h: float) -> float:
        """
        Calculates the volume of a cylinder given its radius and height.

        Args:
        r: The radius of the cylinder.
        h: The height of the cylinder.

        Returns:
        The volume of the cylinder.
        """
        return 3.14159265358979323846 * r ** 2 * h


    def area_of_triangle(self, b: float, h: float) -> float:
        """
        Calculates the area of a triangle given its base and height.

        Args:
        b: The base of the triangle.
        h: The height of the triangle.

        Returns:
        The area of the triangle.
        """
        return 0.5 * b * h


    def sine(self, x: float) -> float:
        """
        Calculates the sine of a given angle in radians.

        Args:
        x: The angle in radians.

        Returns:
        The sine of the angle.
        """
        x = x % (2 * 3.141592653589793238)
        sign = 1 if x > 0 else -1
        x *= sign
        if x > 3.141592653589793238:
            x -= 2 * 3.141592653589793238
            sign *= -1
        return sign * (
            x - x ** 3 / 6 + x ** 5 / 120 - x ** 7 / 5040 + x ** 9 / 362880
        )

alg = Algorithm()
con = Constants()
fun = MathFunctions()
