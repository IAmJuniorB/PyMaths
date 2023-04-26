from Constants import Constants
from typing import List


class Algorithm:
    """Class full of mathematical functions"""
    
    def __init__(self):
        pass

    def addition(self, *args: int | float) -> int | float:
        """Returns the sum of integers and/or floats"""
        try:
            total = sum(args)
        except TypeError:
            raise TypeError("Input must be numbers")
        return total

    def subtract(self, *args: int | float) -> int | float:
        """Returns integers or float of given numbers after being subtracted"""
        try:
            total = args[0] - sum(args[1:])
        except TypeError:
            raise TypeError("Input must be numbers")
        return total

    def multiply(self, *args: int | float) -> int | float:
        """Returns an integer or float of given numbers multiplied"""
        try:
            total = 1
            for i in args:
                total *= i
        except TypeError:
            raise TypeError("Input must be numbers")
        return total

    def division_float(self, dividend: int | float, divisor: int | float) -> int | float:
        """Returns a float of dividend divided by divisor"""
        try:
            result = dividend / divisor
        except ZeroDivisionError:
            raise ZeroDivisionError("Cannot divide by zero")
        except TypeError:
            raise TypeError("Input must be a number")
        return result

    def division_int(self, dividend: int | float, divisor: int | float) -> int | float:
        """Returns an integer of dividend divided by divisor"""
        try:
            result = dividend // divisor
        except ZeroDivisionError:
            raise ZeroDivisionError("Cannot divide by zero")
        except TypeError:
            raise TypeError("Input must be a number")
        return result

    def division_remainder(self, dividend: int | float, divisor: int | float) -> int | float:
        """Returns the remainder of dividend divided by divisor"""
        try:
            result = dividend % divisor
        except ZeroDivisionError:
            raise ZeroDivisionError("Cannot divide by zero")
        except TypeError:
            raise TypeError("Input must be a number")
        return result

    def power(self, base: int | float, exponent: int | float) -> int | float:
        """Returns base to the power of exponent"""
        try:
            result = base ** exponent
        except TypeError:
            raise TypeError("Input must be a number")
        return result

    @staticmethod
    def log(x, base=10):
        """
        Returns the logarithm of x with a specified base (default is 10)

        Args:
            x (int/float): The value for which to compute the logarithm
            base (int/float, optional): The base of the logarithm. Defaults to 10.

        Returns:
            The logarithm of x with the specified base
        """
        if not isinstance(x, (int, float)) or not isinstance(base, (int, float)):
            raise TypeError("Invalid input. x and base must be integers or floats.")
        if x <= 0:
            raise ValueError("Invalid input. x must be a positive number.")
        if base <= 0 or base == 1:
            raise ValueError("Invalid input. base must be greater than 0 and not equal to 1.")
        
        # Compute the logarithm using the change-of-base formula
        numerator = Algorithm.__ln(x)
        denominator = Algorithm.__ln(base)
        return numerator / denominator

    @staticmethod
    def __ln(x):
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
    def __log10(x):
        """
        Returns the logarithm of x (base 10)

        Args:
            x (int/float): The value for which to compute the logarithm

        Returns:
            The logarithm of x (base 10)
        """
        return Algorithm.log(x, 10)

    def add_fractions(*args):
        """
        Returns the sum of multiple fractions.

        Args:
            *args (tuples): Multiple fractions represented as tuples of the form (numerator, denominator).

        Returns:
            A tuple representing the sum of all fractions in reduced form (numerator, denominator).
        """
        numerator = 0
        denominator = 1
        for fraction in args:
            if not isinstance(fraction, tuple) or len(fraction) != 2:
                raise ValueError("Invalid input. Expected tuples of length 2.")
            if fraction[1] == 0:
                raise ValueError("Invalid input. Denominator cannot be zero.")
            numerator = numerator * fraction[1] + fraction[0] * denominator
            denominator = denominator * fraction[1]
        gcd = Algorithm.find_gcd(numerator, denominator)
        numerator //= gcd
        denominator //= gcd
        return (numerator, denominator)

    @staticmethod
    def find_gcd(a, b):
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

    @staticmethod
    def count_words(text):
        """
        Returns a dictionary of word frequencies in the input text.

        Args:
            text (str): The text to analyze.

        Returns:
            A dictionary where the keys are the words in the text and the values are the number of times each word appears.
        """
        if not isinstance(text, str):
            raise TypeError("Invalid input. text must be a string.")
        
        words = text.lower().split()
        
        word_counts = {}
        
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

#############################################################


    
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
            result = result * num // Algorithm.gcd(result, num)
        return result

    def sort_numbers(numbers: List[int | float], reverse: bool = False) -> List[int | float]:
        """
        This function takes a list of numbers and returns a sorted list in ascending or descending order.

        Parameters:
        numbers (List[int | float]): A list of integers or floats to be sorted.
        reverse (bool, optional): If True, returns the list in descending order. Defaults to False.

        Returns:
        List[int | float]: A sorted list in ascending or descending order.

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

    def binary_search(numbers: List[int | float], target: int | float) -> int:
        """
        This function takes a sorted list of numbers and a target number and returns the index of the target number,
        or -1 if it is not found.

        Parameters:
        numbers (List[int | float]): A sorted list of integers or floats.
        target (int | float): The number to search for in the list.

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
        """
        Calculates the Euclidean distance between two points in a two-dimensional space.

        Args:
            point1 (tuple): A tuple containing the coordinates of the first point as (x, y).
            point2 (tuple): A tuple containing the coordinates of the second point as (x, y).

        Returns:
            float: The Euclidean distance between point1 and point2.
        """
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

############################################################################

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
        centers = self.random_seed.sample(data, k)

        while True:
            # Assign each data point to the nearest center
            clusters = [[] for i in range(k)]
            for point in data:
                distances = [self.distance(point, center) for center in centers]
                min_index = distances.index(min(distances))
                clusters[min_index].append(point)

            # Update cluster centers
            new_centers = []
            for cluster in clusters:
                if len(cluster) == 0:
                    new_centers.append(Algorithm.random.choice(data))
                else:
                    center = [sum(coords)/len(coords) for coords in zip(*cluster)]
                    new_centers.append(center)

            # Check for convergence
            if new_centers == centers:
                break
            else:
                centers = new_centers

        return clusters
   
    def exp(self, num: int | float) -> int | float:
        """
        Returns the exponential value of a number.

        Args:
        - num: a number whose exponential value is to be calculated

        Returns:
        The exponential value of the input number
        """
        result = 1
        term = 1
        for i in range(1, 100):
            term *= num / i
            result += term
        return result

    def absolute(self, num: int | float) -> int | float:
        """
        Returns the absolute value of a number.

        Args:
        - num: a number whose absolute value is to be calculated

        Returns:
        The absolute value of the input number
        """
        return num if num >= 0 else -num


    def modulo(self, dividend: int | float, divisor: int | float) -> int | float:
        """
        Returns the remainder of dividing the dividend by the divisor.

        Args:
        - dividend: the number to be divided
        - divisor: the number to divide by

        Returns:
        The remainder of dividing the dividend by the divisor
        """
        quotient = int(dividend / divisor)
        return dividend - (quotient * divisor)


    def sin(self, num: int | float) -> int | float:
        """
        Returns the sine value of a number.

        Args:
        - num: a number in radians whose sine value is to be calculated

        Returns:
        The sine value of the input number
        """
        n = 10
        x = num % (2*Constants.pi)
        result = 0
        for i in range(n):
            term = (-1)**i * x**(2*i+1) / self.factorial(2*i+1)
            result += term
        return result


    def cos(self, num: int | float) -> int | float:
        """
        Returns the cosine value of a number.

        Args:
        - num: a number in radians whose cosine value is to be calculated

        Returns:
        The cosine value of the input number
        """
        n = 10
        x = num % (2*Constants.pi)
        result = 0
        for i in range(n):
            term = (-1)**i * x**(2*i) / self.factorial(2*i)
            result += term
        return result

    def tan(self, num: int | float) -> int | float:
        """
        Returns the tangent value of a number.

        Args:
        - num: a number in radians whose tangent value is to be calculated

        Returns:
        The tangent value of the input number
        """
        sin_val = self.sin(num)
        cos_val = self.cos(num)
        if cos_val == 0:
            return None
        return sin_val / cos_val

    @staticmethod
    def next_prime(n):
        """
        Finds the smallest prime number greater than n.

        Args:
            n (int): A positive integer.

        Returns:
            int: The smallest prime number greater than n.
        """
        if n < 2:
            return 2
        candidate = n + 1
        while True:
            is_prime = True
            for j in range(2, int(candidate**0.5) + 1):
                if candidate % j == 0:
                    is_prime = False
                    break
            if is_prime:
                return candidate
            candidate += 1
            
    @staticmethod
    def atan(x):
        """
        Return the arc tangent of x, in radians.

        Symbol:
            None

        Args:
            x (float): The value whose arc tangent is to be returned.

        Returns:
            float: The arc tangent of x, in radians.
        """
        if x == 0:
            return 0.0
        elif x == float('inf'):
            return Constants.pi / 2.0
        elif x == float('-inf'):
            return -Constants.pi / 2.0
        elif x > 0:
            return Algorithm.atan_helper(x)
        else:
            return -Algorithm.atan_helper(-x)

    @staticmethod
    def atan_helper(x):
        """
        Helper function for atan. Computes the arc tangent of x in the interval [0, 1].

        Args:
            x (float): The value whose arc tangent is to be returned.

        Returns:
            float: The arc tangent of x, in radians.
        """
        assert 0 <= x <= 1

        result = 0.0
        for n in range(1, 1000):
            term = (-1) ** (n - 1) * x ** (2 * n - 1) / (2 * n - 1)
            result += term
        return result
    
    @staticmethod
    def arctan(x):
        """
        Calculates the arctangent of x using a Taylor series approximation.

        Args:
            x (float): A real number.

        Returns:
            float: The arctangent of x in radians.
        """
        if x == 0:
            return 0.0
        elif x < 0:
            return -Algorithm.arctan(-x)
        elif x > 1:
            return Constants.pi/2 - Algorithm.arctan(1/x)
        else:
            result = 0.0
            term = x
            term_index = 1
            epsilon = 1e-10
            while abs(term) > epsilon:
                result += term
                term_index += 1
                term = -term * x * x * (2*term_index - 1) / ((2*term_index) * (2*term_index + 1))
            return result
        
    @staticmethod
    def sieve_of_eratosthenes(n: int) -> List[int]:
        """Returns a list of prime numbers up to n using the sieve of Eratosthenes algorithm.
        
        Args:
            n (int): the upper limit of the list.
            
        Returns:
            List[int]: a list of prime numbers up to n.
        """
        is_prime = [True] * (n + 1)
        is_prime[0], is_prime[1] = False, False
        
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        
        return [i for i in range(n + 1) if is_prime[i]]
    
    def zeta(s, zeta_1, n):
        """Returns the value of the Riemann zeta function.

        Args:
            s (float): The argument of the zeta function.
            zeta_1 (complex): The initial value of the Riemann zeta function.
            n (int): The number of iterations to perform.

        Returns:
            complex: The value of the Riemann zeta function.
        """
        # Initialize the Riemann zeta function with the initial value
        zeta = zeta_1
        # Loop over the desired number of iterations
        for i in range(n):
            # Update the value of the Riemann zeta function
            zeta += 1 / (i + 1) ** s
        # Return the final value of the Riemann zeta function
        return zeta

    def histogram(data, num_bins):
        """Compute the histogram of a list of data with a specified number of bins.

        Args:
            data (list): A list of numeric data
            num_bins (int): The number of bins to use in the histogram

        Returns:
            tuple: A tuple containing the counts for each bin and the edges of the bins

        """
        # Compute the minimum and maximum values in the data
        data_min = min(data)
        data_max = max(data)
        # Compute the width of each bin
        bin_width = (data_max - data_min) / num_bins
        # Initialize the counts for each bin to zero
        counts = [0] * num_bins
        # Loop over the data and increment the count for the appropriate bin
        for x in data:
            bin_index = int((x - data_min) / bin_width)
            if bin_index == num_bins:
                bin_index -= 1
            counts[bin_index] += 1
        # Compute the edges of the bins
        bin_edges = [data_min + i * bin_width for i in range(num_bins+1)]
        # Return the counts and edges as a tuple
        return counts, bin_edges
                
    def islice(iterable, start, stop, step=1):
        """
        Returns an iterator that produces a slice of elements from the given iterable.

        Args:
            iterable (iterable): The iterable to slice.
            start (int): The index at which to start the slice.
            stop (int): The index at which to stop the slice.
            step (int, optional): The step size between slice elements. Defaults to 1.

        Returns:
            iterator: An iterator that produces the slice of elements.
        """
        if start < 0:
            # If start is negative, convert it to a positive index
            start = len(iterable) + start
        if start >= stop:
            # If start is greater than or equal to stop, return an empty iterator
            return iter(())
        for i, x in enumerate(iterable):
            if i >= stop:
                break
            if i >= start and (i - start) % step == 0:
                yield x


    def normal_distribution_cdf(x):
        """
        Calculates the cumulative distribution function (CDF) of a standard normal distribution at a given value.

        Args:
            x (float): The value at which to calculate the CDF.

        Returns:
            float: The CDF of the standard normal distribution at x, accurate to 10 decimal places.
        """
        if x < -7:
            return 0.0
        elif x > 7:
            return 1.0
        else:
            term = x
            total = term
            i = 1
            while abs(term) > 0.0000000001:
                term *= -x * x / (2 * i + 1)
                total += term
                i += 1
            return round((total + 1) / 2, 10)
