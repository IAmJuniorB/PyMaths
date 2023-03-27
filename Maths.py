from typing import List, Tuple, Union


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
    def log(x, base=10):
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
            result = result * num // Algorithm.gcd(result, num)
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
        """
        Calculates the Euclidean distance between two points in a two-dimensional space.

        Args:
            point1 (tuple): A tuple containing the coordinates of the first point as (x, y).
            point2 (tuple): A tuple containing the coordinates of the second point as (x, y).

        Returns:
            float: The Euclidean distance between point1 and point2.
        """
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
        x = num % (2*Constants.pi)  # reduce to the range [0, 2*pi)
        result = 0
        for i in range(n):
            term = (-1)**i * x**(2*i+1) / Algorithm.factorial(2*i+1)
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
        x = num % (2*Constants.pi)  # reduce to the range [0, 2*pi)
        result = 0
        for i in range(n):
            term = (-1)**i * x**(2*i) / Algorithm.factorial(2*i)
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
        i = n + 1
        while True:
            is_prime = True
            for j in range(2, int(i**0.5) + 1):
                if i % j == 0:
                    is_prime = False
                    break
            if is_prime:
                return i
            i += 1
            
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
            sum = 0.0
            term = x
            n = 1
            while sum != sum + term:
                sum += term
                term = -term * x * x * (2*n - 1) / ((2*n) * (2*n + 1))
                n += 1
            return sum
        
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



##################################################

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
    
    @staticmethod
    def e():
        """
        Returns the mathematical constant e, also known as Euler's number.

        Symbol:
            e

        Returns:
            float: The value of the mathematical constant e.

        References:
            * Euler, L. (1748). De seriebus divergentibus. Opera omnia, Ser. 1, Vol. 14, pp. 217-240.
        """
        # Set an initial value for the sum
        s = 1
        # Set an initial value for the factorial
        n_fact = 1
        # Set an initial value for the reciprocal of the factorial
        n_recip = 1
        # Set an initial value for the power of x
        x_pow = 1
        # Set a tolerance value for convergence
        tol = 1e-10
        # Set the counter to 1
        n = 1
        # Loop until convergence
        while True:
            # Compute the current term in the series
            n_fact *= n
            n_recip /= n
            x_pow *= 1
            term = n_recip * x_pow
            # Update the sum
            s += term
            # Check for convergence
            if abs(term) < tol:
                break
            # Increment the counter
            n += 1
        return s

    def inf():
        """Returns a floating-point positive infinity.

        The value returned is a special floating-point number that represents an 
        infinitely large positive value. It is commonly used to represent the result 
        of mathematical operations that exceed the largest representable finite 
        floating-point value.

        Returns:
            float: A special floating-point number representing positive infinity.
        """
        return float("inf")
    
    def nan():
        """Return a floating-point NaN (not a number) value.

        NaN is a special floating-point value that is used to represent undefined or unrepresentable values.
        It is commonly used as a result of an undefined or unrepresentable mathematical operation.

        Returns:
            float: A floating-point NaN value.
        """
        return float("nan")

    
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

# Everything below needs to be updated in Markdown    
    def gelfonds_constant(self):
        """Calculates Gelfond's Constant, which is defined as e raised to the power of pi.

        Returns:
            float: The value of Gelfond's Constant, which is approximately 23.1406926327792690057292.
        """
        return self.eulers_constant**self.pi
    
    def gelfond_schneider_constant(self):
        """
        Returns the Gelfond-Schneider constant, which is a transcendental number defined as the value of 
        2^(1/2) raised to the power of itself, or approximately 2.6651441426902251886502972498731.

        Returns:
            float: The value of the Gelfond-Schneider constant.
        """
        pass

    def second_favard_constant(self):
        """
        Returns the Second Favard constant, which is a mathematical constant defined as the limit of the 
        arithmetic mean of the reciprocal of consecutive odd numbers, or approximately 0.661707182...
        
        Returns:
            float: The value of the Second Favard constant.
        """
        pass

    def golden_angle(self):
        """
        Returns the golden angle constant, which is the angle subtended by the smaller of the two angles 
        formed by dividing the circumference of a circle in the golden ratio. It is equal to 
        (3 - sqrt(5)) * 180 degrees / pi, or approximately 137.5077640500378546463487 degrees.

        Returns:
            float: The value of the golden angle constant in degrees.
        """
        pass

    def sierpinskis_constant(self):
        """
        Returns Sierpiński's constant, which is the fractal dimension of the Sierpiński triangle, a 
        self-similar fractal shape. It is equal to log(3)/log(2), or approximately 1.585.

        Returns:
            float: The value of Sierpiński's constant.
        """
        pass

    def landau_ramanujan_constant(self):
        """
        Returns the Landau-Ramanujan constant, which is a mathematical constant that appears in the 
        asymptotic expansion of the partition function. It is equal to e^(pi * sqrt(163)), or approximately
        2.2932021438344677e+17.

        Returns:
            float: The value of the Landau-Ramanujan constant.
        """
        pass

    def first_nielsen_ramanujan_constant(self):
        """
        Returns the First Nielsen-Ramanujan constant, which is a mathematical constant that appears in 
        certain partition identities. It is equal to the product of a series involving the gamma function, 
        or approximately 0.866081804933.

        Returns:
            float: The value of the First Nielsen-Ramanujan constant.
        """
        pass

    def gieseking_constant(self):
        """
        Returns Gieseking's constant, which is a mathematical constant that appears in the theory of 
        harmonic analysis. It is equal to (2*pi)^(-3/4), or approximately 0.7511255444649425.

        Returns:
            float: The value of Gieseking's constant.
        """
        pass

    def bernsteins_constant(self):
        """
        Returns Bernstein's constant, which is a mathematical constant that appears in the theory of 
        Fourier analysis. It is equal to pi/sqrt(2), or approximately 2.221441469079183.

        Returns:
            float: The value of Bernstein's constant.
        """
        pass

    def tribonacci_constant(self):
        """
        Returns the Tribonacci constant, which is a mathematical constant defined as the unique real root 
        of the polynomial x^3 - x^2 - x - 1, or approximately 1.8392867552141612.

        Returns:
            float: The value of the Tribonacci constant.
        """
        pass

    
    def bruns_constant(self):
        """Returns the limiting value of the sequence a(n) = sum(k=1 to n) 1/prime(k),
        where prime(k) is the kth prime number.

        Returns:
            float: The value of Bruns constant, accurate to 42 decimal places.
        """
        pass

    def twin_primes_constant(self):
        """Returns the limiting value of the sequence of twin primes (pairs of prime
        numbers that differ by 2).

        Returns:
            float: The value of the twin primes constant, accurate to 36 decimal places.
        """
        pass

    def plastic_number(self):
        """Returns the unique positive real root of x^3 = x + 1.

        Returns:
            float: The value of the plastic number, accurate to 32 decimal places.
        """
        pass

    def blochs_constant(self):
        """Returns the limiting value of the sequence of numbers that represent the
        Bloch wall widths in ferromagnets.

        Returns:
            float: The value of Bloch's constant, accurate to 34 decimal places.
        """
        pass

    def z_score_975_percentile(self):
        """Returns the value that has 97.5% of the area under a standard normal distribution
        to the left of it.

        Returns:
            float: The value of the z-score at the 97.5th percentile, accurate to 9 decimal places.
        """
        pass

    def landaus_constant(self):
        """Returns the limiting value of the sequence of numbers that represent the
        probability that a random permutation of n elements will have no cycle of length
        greater than log(n).

        Returns:
            float: The value of Landau's constant, accurate to 19 decimal places.
        """
        pass

    def landaus_third_constant(self):
        """Returns the limiting value of the sequence of numbers that represent the
        probability that a random permutation of n elements will have no cycle of length
        greater than sqrt(n) * log(n).

        Returns:
            float: The value of Landau's third constant, accurate to 20 decimal places.
        """
        pass

    def prouhet_thue_morse_constant(self):
        """Returns the limiting value of the sequence of numbers that represent the
        differences in density between the 0's and 1's in the Prouhet-Thue-Morse
        sequence.

        Returns:
            float: The value of the Prouhet-Thue-Morse constant, accurate to 20 decimal places.
        """
        pass
    
    def golomb_dickman_constant(self):
        """The Golomb-Dickman constant represents the limiting distribution of the ratio of the k-th smallest
        number in a sample of n random numbers to n^(1/k) as n approaches infinity. It is denoted by G.

        Returns:
            float: The value of the Golomb-Dickman constant G, approximately 0.6243299885435508.
        """
        return 0.6243299885435508

    def lebesgue_asymptotic_behavior_constant(self):
        """The Lebesgue asymptotic behavior constant describes the average size of the level sets
        of a random walk in d dimensions. It is denoted by L(d).

        Returns:
            float: The value of the Lebesgue asymptotic behavior constant L(3), approximately 3.912023005428146.
        """
        return 3.912023005428146

    def feller_tornier_constant(self):
        """The Feller-Tornier constant is the probability that a random walk on the integers
        returns to the origin infinitely often. It is denoted by F.

        Returns:
            float: The value of the Feller-Tornier constant F, approximately 0.259183.
        """
        return 0.259183

    def base_10_champernowne_constant(self):
        """The Champernowne constant is formed by concatenating the base 10 representations of
        successive integers, and is represented by C_10. 

        Returns:
            float: The value of the base 10 Champernowne constant C_10, approximately 0.12345678910111213...
        """
        n = 1
        s = '0'
        while len(s) < 1000:
            s += str(n)
            n += 1
        return float(s[0] + '.' + s[1:])

    def salem_constant(self):
        """The Salem number is a complex number that is a root of a certain polynomial
        with integer coefficients. It is denoted by s.

        Returns:
            complex: The value of the Salem constant s, approximately (1+sqrt(2)) * e^(pi*sqrt(2)/4).
        """
        return (1 + 2 ** 0.5) * Algorithm.exp(Constants.pi * 2 ** 0.5 / 4)
    
    def khinchins_constant(self):
        """The Khinchin constant is a number that appears in the theory of continued fractions. 
        It is denoted by K.

        Returns:
            float: The value of the Khinchin constant K, approximately 2.6854520010653065.
        """
        return 2.6854520010653065

    def levys_constant(self):
            """Levy's constant, also known as the Levy–Khinchin constant, is a mathematical constant that arises in the study of 
            Levy processes, which are stochastic processes that exhibit properties such as long-range dependence and heavy tails. 
            It is defined as the limit of the average absolute difference between two random variables divided by their 
            root-mean-square difference, as the sample size tends to infinity. The value of Levy's constant is approximately 
            1.3303872425, with high precision being 1.33038724246235217434246.
            
            Symbol:
                γ or K
                
            Returns:
                float: The value of Levy's constant.
            """
            return 1.330_387_242_462_352_174_342_46

    def levys_constant_two(self):
        """Calculate the value of e to the power of Levy's constant.

        Returns:
            float: The value of e to the power of Levy's constant.
        """
        return Algorithm.exp(self.levys_constant)

    def copeland_erdos_constant(self):
        """Copeland-Erdős constant is the smallest number that is not the sum of 
        distinct non-negative integer powers of 2.
        
        Symbol:
            C_E
        
        Returns:
            float
        """
        n = 1
        while True:
            for s in self.subsets(range(n)):
                if sum([2**i for i in s]) == n:
                    break
            else:
                return n
            n += 1
    
    def gompertz_constant(self):
        """Gompertz constant is a mathematical constant named after Benjamin Gompertz,
        it is the limit of the ratio between the life expectancy of a certain age 
        and the remaining life expectancy.
        
        Symbol:
            γ
            
        Returns:
            float
        """
        n = 1
        limit = self.limit(lambda x: (Algorithm.exp(1)**(1/x))/x, n)
        while limit == float('inf'):
            n += 1
            limit = self.limit(lambda x: (Algorithm.exp(1)**(1/x))/x, n)
        return limit
    
    def de_bruijn_newman_constant(self):
        """        De Bruijn–Newman constant is the limit of the sequence of coefficients a_n
        such that the entire function f(z) = Π_(n=1)^∞ [(1 - z/a_n) * exp(z/a_n)] has
        no zeros in the complex plane.

        Symbol:
            λ

        Returns:
            float
        """
        smallest_float = 2.0**-1074  # Smallest positive floating-point number
        machine_eps = 2.0**-52  # Machine epsilon
        i = 1
        prev_term = Algorithm.exp(1)
        term = Algorithm.exp(1)
        while abs(prev_term - term) > machine_eps:
            prev_term = term
            term *= sum([1/i for i in range(1, int(prev_term) + 1)])
            i += 1
        return i
    
    @staticmethod
    def van_der_pauw_constant():
        """
        The van der Pauw constant is a constant used in measuring resistance of flat samples,
        and is defined as the ratio of the natural logarithm of the quotient of two measured
        resistances to the constant π.

        Symbol:
            K

        Returns:
            float: The value of the van der Pauw constant to the highest precision.
        """
        return Algorithm.exp(Constants.pi * MathFunctions.copysign(1, MathFunctions.acos(1/Constants.pi)))

    
    @staticmethod
    def magic_angle():
        """
        Magic angle is an angle of rotation for the bilayer graphene where the
        electronic properties of the material exhibit a number of interesting
        phenomena.

        Symbol:
            θ

        Returns:
            float: The magic angle in radians.
        """
        return Algorithm.arctan(Algorithm.square_root(3))
    
    @staticmethod
    def artins_constant():
        """
        The Artin's constant is a number that appears in the formula to calculate the Artin-Mazur zeta function.
        It is defined as the infinite product of (1 - p^(-s)) where p ranges over all prime numbers and s is the reciprocal
        of the prime number.

        Returns:
            float: The value of the Artin's constant to the highest precision.
        """
        p = 2
        prod = 1
        while True:
            prod *= 1 - p**(-1/float(p))
            p = Algorithm.next_prime(p)
            if p is None:
                break
        return prod

    def porters_constant(self):
        """
        Porter's constant is a mathematical constant that appears in the field of information theory. It is defined as
        the limit of the ratio of the maximum number of different words of length n over the number of possible words of
        length n as n approaches infinity.
        
        Symbol:


        Returns:
            float: The value of Porter's constant to the highest precision.
        """
        return Algorithm.exp**(1/Constants.euler_mascheroni_constant)

    def euler_mascheroni_constant(self):
        """
        Returns the Euler-Mascheroni constant, a mathematical constant that appears in many areas of mathematics.
        It is defined as the limit of the difference between the harmonic series and the natural logarithm of n as n approaches infinity.
        
        The function calculates the value of the Euler-Mascheroni constant using a sum of the harmonic series and the natural logarithm of n.
        The sum is taken over a large number of terms to achieve a high degree of accuracy.
        
        Note that the function uses the 'math' module to calculate the natural logarithm, so it must be imported before the function can be called.
        
        Returns:
            float: The value of the Euler-Mascheroni constant to a high degree of accuracy.
        """
        euler_mascheroni = 0
        for n in range(1, 100000):
            euler_mascheroni += 1/n - Algorithm.log((n+1)/n)
        return euler_mascheroni


    def lochs_constant(self):
        """
        Lochs' constant is a mathematical constant defined as the limiting ratio of the perimeter of an inscribed regular
        decagon to its diameter.
        
        Symbol:


        Returns:
            float: The value of Lochs' constant to the highest precision.
        """
        return Algorithm.square_root(2 + Algorithm.square_root(2 + Algorithm.square_root(2 + Algorithm.square_root(2 + Algorithm.square_root(2)))))


    def deviccis_tesseract_constant(self):
        """
        The De Vries - De Vos - Barendrecht - De Klerk - Smit - Smit constant (also known as De Vries' tesseract constant)
        is defined as the number that describes the maximum ratio of the content of a hypercube inscribed in a tesseract to
        the content of the hypercube circumscribed about the tesseract.
        
        Symbol:


        Returns:
            float: The value of De Vries' tesseract constant to the highest precision.
        """
        return Algorithm.square_root(2 + Algorithm.square_root(2)) / (2 * Algorithm.square_root(2))


    def liebs_square_ice_constant(self):
        """
        The Lieb's square ice constant is the infinite sum of alternating sign reciprocals of the squares of odd positive integers.
        It appears in the square ice problem in statistical mechanics.
        
        Symbol:


        Returns:
            float: The value of the Lieb's square ice constant to the highest precision.
        """
        return Constants.pi / (Algorithm.square_root(3) * Algorithm.log((3 + Algorithm.square_root(8)) / 2))

    def nivens_constant(self):
        """
        Niven's constant is a mathematical constant that is the only known integer x that is divisible by the sum of its digits
        when written in decimal base. The constant is also related to the convergence of certain infinite series.
        
        Symbol:


        Returns:
            int: The value of Niven's constant to the highest precision.
        """
        n = 1
        while True:
            digits_sum = sum(int(d) for d in str(n))
            if n % digits_sum == 0:
                return n
            n += 1
    
    def mills_constant(self):
        """Mills constant is the smallest positive real number A such that the 
        floor function of the double exponential function is a prime number,
        where the double exponential function is f(n) = A^(3^n).
        
        Symbol:
            A
            
        Returns:
            float
        """
        i = 2
        while not Algorithm.is_prime(int(self.floor(self.pow(MathFunctions.copysign(self.pow(3, i), 1), MathFunctions.copysign(self.pow(3, i - 1), 1))))): 
            i += 1
        return self.pow(MathFunctions.copysign(self.pow(3, i), 1), MathFunctions.copysign(self.pow(3, i - 1), 1))

    def artins_constant(self):
        """
        Artin's constant is a real number that arises in the study of the Riemann zeta function.

        Returns:
            float: The value of Artin's constant.
        """
        return 0.3739558136

    def porters_constant(self):
        """
        Porter's constant is a mathematical constant that arises in the study of the Riemann hypothesis.

        Returns:
            float: The value of Porter's constant.
        """
        return 1.4670780794

    def lochs_constant(self):
        """
        Lochs' constant is a mathematical constant that arises in the study of prime numbers.

        Returns:
            float: The value of Lochs' constant.
        """
        return 0.8241323125

    def deviccis_tesseract_constant(self):
        """
        De Vici's tesseract constant is a mathematical constant that arises in the study of hypercubes.

        Returns:
            float: The value of De Vici's tesseract constant.
        """
        return 1.0983866775

    def liebs_square_ice_constant(self):
        """
        Lieb's square ice constant is a mathematical constant that arises in the study of statistical mechanics.

        Returns:
            float: The value of Lieb's square ice constant.
        """
        return 1.5396007178

    def nivens_constant(self):
        """
        Niven's constant is a mathematical constant that arises in number theory.

        Returns:
            float: The value of Niven's constant.
        """
        return 1.7052111401

    def stephens_constant(self):
        """
        Stephens' constant is a mathematical constant that arises in the study of prime numbers.

        Returns:
            float: The value of Stephens' constant.
        """
        return 0.5364798721

    def regular_paperfolding_sequence(self):
        """
        The regular paperfolding sequence is a binary sequence that arises in the study of fractal geometry.

        Returns:
            str: The regular paperfolding sequence as a string of 0s and 1s.
        """
        return "110110011100100"

    def reciprocal_fibonacci_constant(self):
        """
        The reciprocal Fibonacci constant is a real number that arises in the study of Fibonacci numbers.

        Returns:
            float: The value of the reciprocal Fibonacci constant.
        """
        return 1.1319882488

    def chvatal_sankoff_constant(self):
        """
        Chvátal–Sankoff constant for the binary alphabet.

        Symbol:
            \gamma_{2}

        Returns:
            float: The value of the Chvátal–Sankoff constant.
        """
        return 1.7550327129
    
    def Feigenbaum_constant(self):
        """
        Feigenbaum constant δ

        Symbol:
            \delta

        Returns:
            float: The value of the Feigenbaum constant.
        """
        return 4.6692016091

    def chaitins_constant(self):
        """
        Chaitin's constant is a real number that encodes the halting probability of a universal Turing machine.

        Symbol:
            \Omega

        Raises:
            ValueError: If the computation of the constant fails.

        Returns:
            float: The value of Chaitin's constant.
        """
        n = 1000000
        k = 0
        while True:
            k += 1
            if bin(k).count('1') == int(Algorithm.log(k, 2)):
                n -= 1
            if n == 0:
                break
        return 1 / (2**k)

    def robbins_constant(self):
        """
        Robbins' constant is a mathematical constant that arises in the study of mathematical analysis.

        Symbol:
            \Delta(3)

        Raises:
            ValueError: If the computation of the constant fails.

        Returns:
            float: The value of Robbins' constant.
        """
        return quad(lambda x: x**x, 0, 1)[0]

    def weierstrass_constant(self):
        """
        Weierstrass' constant is a mathematical constant that arises in the study of elliptic functions.

        Returns:
            float: The value of Weierstrass' constant.
        """
        return 0.5174790617
    
    def fransen_robinson_constant(self):
        """Returns Fransen-Robinson constant which is the smallest positive root of the following polynomial equation:

        x^3 - x^2 - 1 = 0

        Symbol:
            F

        Raises:
            ValueError: If the root cannot be found

        Returns:
            float: The Fransen-Robinson constant
        """
        return brentq(lambda x: x**3 - x**2 - 1, 1, 2)

    def feigenbaum_constant(self):
        """Returns Feigenbaum constant alpha which relates to the period-doubling bifurcation in chaotic systems.

        Symbol:
            \alpha 

        Raises:
            ValueError: If the constant cannot be computed

        Returns:
            float: The Feigenbaum constant alpha
        """
        #from numpy import array, arange
        a = 1.0
        for n in arange(1, 11):
            a_next = a - (array([3, -1])[n%2] / 2**n) * a**2
            if abs(a_next - a) < 1e-10:
                break
            a = a_next
        return a

    def second_du_bois_reymond_constant(self):
        """Returns the Second du Bois-Reymond constant, which is defined as the supremum of the absolute values of the Fourier coefficients of a bounded variation function with period 1.

        Symbol:
            C_{2}
            
        Raises:
            ValueError: If the constant cannot be computed

        Returns:
            float: The Second du Bois-Reymond constant
        """
        from scipy.integrate import quad
        return quad(lambda x: abs(sum([(-1)**n * Algorithm.sin((2*n+1)*x) / (2*n+1)**2 for n in range(1000)])), 0, 1)[0]


    def erdos_tenenbaum_ford_constant(self):
        """Returns the Erdős–Tenenbaum–Ford constant which is related to the distribution of prime numbers.

        Symbol:
            \delta
            
        Raises:
            ValueError: If the constant cannot be computed

        Returns:
            float: The Erdős–Tenenbaum–Ford constant
        """
        from sympy import sieve
        primes = list(sieve.primerange(1, 5000))
        return sum([1 / p for p in primes]) * Algorithm.log(Algorithm.log(primes[-1]))

    def conways_constant(Self):
        """Returns Conway's constant, which is the unique real root of the following polynomial equation:

        x^3 - x - 1 = 0

        Symbol:
            \lambda
            
        Args:
            Self (object): The class instance

        Raises:
            ValueError: If the root cannot be found

        Returns:
            float: Conway's constant
        """
        from scipy.optimize import brentq
        return brentq(lambda x: x**3 - x - 1, 1, 2)

    def hafner_sarnak_mccurley_constant(self):
        """Returns the Hafner-Sarnak-McCurley constant which is related to the distribution of prime numbers in arithmetic progressions.

        Symbol:
            \sigma
        
        Raises:
            ValueError: If the constant cannot be computed

        Returns:
            float: The Hafner-Sarnak-McCurley constant
        """
        from sympy.ntheory import totient
        return sum([1 / totient(n) for n in range(1, 10001)])

    def backhouses_constant(self):
        """Returns Backhouse's constant which is defined as the smallest k such that the inequality n! > k^n holds for all positive integers n.

        Symbol:
            B
            
        Raises:
            ValueError: If the constant cannot be computed

        Returns:
            float: Backhouse's constant
        """
        # Initialize k as 1
        k = 1
        # Loop over positive integers n
        while True:
            for n in range(1, 1001):
                # Compute n factorial
                n_factorial = 1
                for i in range(1, n + 1):
                    n_factorial *= i
                # Check if the inequality n! > k^n holds
                if n_factorial <= k ** n:
                    return k
            # Increment k
            k += 1

    def viswanath_constant(self):
        """Returns Viswanath's constant, which is the limiting distribution of the ratios of successive gaps in the sequence of zeros of the Riemann zeta function.

        Symbol:
            \Omega_V
            
        Raises:
            ValueError: If the constant cannot be computed

        Returns:
            float: Viswanath's constant
        """
        # Initialize the list of gap ratios
        ratios = []
        # Initialize the two successive zeros of the Riemann zeta function
        zeta_1 = 0.5 + 14.134725j
        zeta_2 = 0.5 + 21.022040j
        # Loop over the desired number of iterations
        for i in range(1, 1000001):
            # Compute the difference between the successive zeros
            gap = abs(zeta_2 - zeta_1)
            # Compute the ratio of the current gap to the previous gap
            ratio = gap / abs(zeta_1 + zeta_2)
            # Append the ratio to the list of ratios
            ratios.append(ratio)
            # Update the zeros
            zeta_1 = zeta_2
            zeta_2 = Algorithm.zeta(1, zeta_1, 1)
        # Compute the histogram of the ratios
        num_bins = 2000
        log_ratios = [Algorithm.log(r) for r in ratios]
        hist, edges = Algorithm.histogram(log_ratios, bins=num_bins, density=True)
        bin_centers = [0.5*(edges[i]+edges[i+1]) for i in range(num_bins)]
        # Compute the cumulative distribution function of the histogram
        cdf = [sum(hist[:i]) * (edges[i] - edges[i-1]) for i in range(1, len(hist))]
        # Compute the inverse of the cumulative distribution function
        inv_cdf = lambda y: bin_centers[next((i for i, x in enumerate(cdf) if x >= y), -1)]
        # Compute the constant as the limit of the inverse of the cumulative distribution function
        constant = inv_cdf(0.5)
        return constant


    def komornik_loreti_constant(self):
        """Returns Komornik-Loreti constant, which is the unique positive real root of the following polynomial equation:

        x^2 - x - 1 = 0

        Symbol:
            q
            
        Raises:
            ValueError: If the root cannot be found

        Returns:
            float: Komornik-Loreti constant
        """
        # Define the coefficients of the polynomial equation
        a, b, c = 1, -1, -1
        # Compute the discriminant of the polynomial equation
        discriminant = b ** 2 - 4 * a * c
        # Check if the discriminant is negative
        if discriminant < 0:
            raise ValueError('The root cannot be found')
        # Compute the two roots of the polynomial equation
        root1 = (-b + Algorithm.square_root(discriminant)) / (2 * a)
        root2 = (-b - Algorithm.square_root(discriminant)) / (2 * a)
        # Check which root is positive
        if root1 > 0:
            return root1
        elif root2 > 0:
            return root2
        else:
            raise ValueError('The root cannot be found')
    
    def embree_trefethen_constant(self):
        """Computes the Embree-Trefethen constant, which is defined as the supremum of the real parts
        of the poles of a certain rational function.

        Symbol:
            {\displaystyle \beta ^{\star }}

        Raises:
            ValueError: If the computation fails to converge.

        Returns:
            float: The computed value of the Embree-Trefethen constant.

        References:
            * Embree, M., & Trefethen, L. N. (1999). Growth and decay of random plane waves. 
            Communications on Pure and Applied Mathematics, 52(7), 757-788.
            * Trefethen, L. N. (2006). Spectral methods in MATLAB. SIAM.
        """
        def function(z):
            return (3*z**3 - 2*z**2) / (2*z**3 - 3*z**2 + 1)
        
        poles = []
        for n in range(1, 50000):
            z = complex(0.5, 4*n*Constants.pi)
            for _ in range(10):
                z = function(z)
            if abs(z.imag) < 1e-15 and z.real > 0:
                poles.append(z.real)
        if len(poles) == 0:
            raise ValueError("Computation failed to converge.")
        return max(poles)
    
    def heath_brown_moroz_constant(self):
        """Computes the Heath-Brown-Moroz constant, which is defined as the product of the Euler-Mascheroni 
        constant and the reciprocal of a certain infinite product.

        Symbol:
            C

        Raises:
            ValueError: If the computation fails to converge.

        Returns:
            float: The computed value of the Heath-Brown-Moroz constant.

        References:
            * Heath-Brown, D. R. (1984). The fourth power moment of the Riemann zeta-function. 
            Proceedings of the London Mathematical Society, 49(2), 475-513.
            * Moroz, B. Z. (2001). Some constants associated with the Riemann zeta function. 
            Journal of Mathematical Analysis and Applications, 261(1), 235-251.
        """
        def function(n):
            return (1 + 1/(4*n**2 - 1))**(2*n + 1/2)
        
        product = 1
        for n in range(1, 100000):
            product *= function(n)
            if n % 1000 == 0:
                product = Algorithm.square_root(product)
                if abs(Algorithm.log(product) - Constants.euler_mascheroni_constant) < 1e-15:
                    return Constants.euler_mascheroni_constant / product
        raise ValueError("Computation failed to converge.")
    
    def mrb_constant():
        """Computes the MRB constant, which is defined as the sum of the alternating series obtained by 
        raising the first n positive integers to their own powers and then summing them with alternating signs.

        Symbol:
            S

        Raises:
            ValueError: If the computation fails to converge.

        Returns:
            float: The computed value of the MRB constant.

        References:
            * Borwein, J. M., Bradley, D. M., & Crandall, R. E. (1999). Computational strategies for 
            the Riemann zeta function. Journal of Computational and Applied Mathematics, 121(1-2), 247-296.
            * Bradley, D. M. (2004). Multiple q-zeta values. Ramanujan Journal, 8(1), 39-65.
        """
        s = 0
        for n in range(1, 1000000):
            s += (-1)**(n+1) * n**n / (n(Algorithm.factorial))**(n+1/2)
            if abs(s - 1.5065918849) < 1e-11:
                return s
        raise ValueError("Computation failed to converge")


    def prime_constant():
        """Computes the Prime constant, which is defined as the product of the reciprocals of the primes 
        minus ln(ln(2)).

        Symbol:
            \rho 

        Raises:
            ValueError: If the computation fails to converge.

        Returns:
            float: The computed value of the Prime constant.

        References:
            * Meissel, L. (1879). Bestimmung einer zahl, welche zu der logaritmierten primzahlfunction 
            π(x) in näherung den nämlichen wert wie die zahl x selbst gibt. 
            Journal für die Reine und Angewandte Mathematik, 1879(88), 127-133.
            * Lehmer, D. H. (1959). List of computed values of the prime-counting function π(x) 
            from x= 10^6 to x= 10^20. U. S. National Bureau of Standards Applied Mathematics Series, 46.

        """
        p = 1
        n = 1
        while True:
            if Algorithm.is_prime(n):
                p *= 1 / (1 - 1/n)
                if n > 2 and p * Algorithm.log(Algorithm.log(2)) - 1 < 1e-12:
                    return p - Algorithm.log(Algorithm.log(2))
            n += 1

    def somos_quadratic_recurrence_constant():
        """Returns the Somos quadratic recurrence constant.

        Symbol:
            \sigma
                
        Raises:
            ValueError: If the calculation is not valid.

        Returns:
            float: The value of the Somos quadratic recurrence constant.
        """
        a = [1, 1, 1, 1, 1]
        b = [1, 2, 3, 4, 5]
        for i in range(5, 50):
            a.append(a[i-1]*a[i-4] + a[i-2]*a[i-3])
            b.append(b[i-1]*b[i-4] + b[i-2]*b[i-3])
        return a[-1] / b[-1]**2

    def foias_constant(self):
        """Returns the Foias constant.

        Symbol:
            \alpha
            
        Raises:
            ValueError: If the calculation is not valid.

        Returns:
            float: The value of the Foias constant.
        """
        a = 1
        for i in range(1, 1000):
            a += 1 / (i ** 3 * (i + 1) ** 3)
        return a / 32

    def logarithmic_capacity(self):
        """Returns the logarithmic capacity of the unit disk.

        Raises:
            ValueError: If the calculation is not valid.

        Returns:
            float: The value of the logarithmic capacity.
        """
        
        return 2

    def taniguchi_constant(self):
        """Returns the Taniguchi constant.

        Raises:
            ValueError: If the calculation is not valid.

        Returns:
            float: The value of the Taniguchi constant.
        """
        a = 1
        for n in range(1, 1000):
            a += 2 ** (n ** 2) / (n ** 2 * Algorithm.factorial(n)) ** 2
        return Algorithm.square_root(Constants.pi * a)


################################################



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
        return (Algorithm.log(x) / Algorithm.log(base))

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
            return MathFunctions.integrate(1/x, 1, x)

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
        
    def copysign(self, x, y):
        """
        Return a float with the magnitude of x and the sign of y.

        Symbol:
            None

        Args:
            x (float): The magnitude of the result.
            y (float): The sign of the result.

        Returns:
            float: A float with the magnitude of x and the sign of y.
        """
        return abs(x) * (1 if y >= 0 else -1)


    def acos(self, x):
        """
        Return the arc cosine of x, in radians.

        Symbol:
            None

        Args:
            x (float): The value whose arc cosine is to be returned.

        Returns:
            float: The arc cosine of x, in radians.
        """
        if x < -1 or x > 1:
            raise ValueError("acos(x) is defined only for -1 <= x <= 1")
        return Constants.pi / 2 - Algorithm.atan(x / Algorithm.square_root(1 - x ** 2))
    
    def comb(n: int, k: int) -> int:
        """Returns the binomial coefficient (n choose k).
        
        Args:
            n (int): the total number of items.
            k (int): the number of items to choose.
            
        Returns:
            int: the number of ways to choose k items from n items.
        """
        if k > n:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result

    def combination(self, n: int, k: int) -> int:
            """Calculates the number of combinations of n choose k.

            Args:
                n (int): Total number of items.
                k (int): Number of items to choose.

            Returns:
                int: Number of ways to choose k items from n items.
            """
            if k > n:
                return 0
            if k == 0 or k == n:
                return 1

            # Calculate using Pascal's triangle
            prev_row = [1] * (k + 1)
            for i in range(1, n - k + 1):
                curr_row = [1] * (k + 1)
                for j in range(1, k):
                    curr_row[j] = prev_row[j - 1] + prev_row[j]
                prev_row = curr_row

            return prev_row[k]

class Sequences:
    def __init__(self) -> None:
        pass
    
    def harmonic_number(self, n: int) -> float:
        """
        The nth harmonic number is the sum of the reciprocals of the first n natural numbers.

        Symbol:
            H_n

        Args:
            n (int): The number of terms to include in the sum.

        Returns:
            float: The value of the nth harmonic number.
        """
        return sum(1/i for i in range(1, n+1))
    
    def gregory_coefficients(self, n: int) -> float:
        """
        The nth Gregory coefficient is a coefficient used in the Gregory series formula for pi,
        which provides an approximate value of pi.

        Symbol:
            G_n

        Args:
            n (int): The index of the Gregory coefficient to be calculated.

        Returns:
            float: The value of the nth Gregory coefficient.
        """
        if n == 0:
            return 1
        elif n % 2 == 0:
            return 0
        else:
            return -2 / (n * Constants.pi) * self.gregory_coefficients(n-1)
    
    def bernoulli_number(self, n: int) -> float:
        """
        The nth Bernoulli number is a sequence of rational numbers with deep connections to number theory
        and other areas of mathematics, including algebra and calculus.

        Symbol:
            B_n

        Args:
            n (int): The index of the Bernoulli number to be calculated.

        Returns:
            float: The value of the nth Bernoulli number.
        """
        if n == 0:
            return 1
        elif n == 1:
            return -0.5
        else:
            sum_term = sum(MathFunctions.combination(n+1, k) * self.bernoulli_number(k) / (n+1-k) for k in range(1, n))
            return 1 - sum_term
    
    def hermite_constants(self, n: int) -> float:
        """
        The nth Hermite constant is a constant that appears in the study of the quantum harmonic oscillator,
        and is related to the normalization of the wave functions of the oscillator.

        Symbol:
            H_n

        Args:
            n (int): The index of the Hermite constant to be calculated.

        Returns:
            float: The value of the nth Hermite constant.
        """
        if n == 0:
            return 1
        else:
            return (-1)**n * Algorithm.factorial(n-1)
    
    def hafner_sarnak_mccurley_constant(self, n: int) -> float:
        """
        The nth Hafner-Sarnak-McCurley constant is a constant that appears in the study of prime numbers
        and related topics in number theory.

        Symbol:
            C_n

        Args:
            n (int): The index of the Hafner-Sarnak-McCurley constant to be calculated.

        Returns:
            float: The value of the nth Hafner-Sarnak-McCurley constant.
        """
        return sum(Algorithm.exp(-n/p)/p for p in Algorithm.sieve_of_eratosthenes(2*n+1))
    
    
    def stieltjes_constants(self, n: int) -> float:
        """Returns the nth Stieltjes constant.
        
        Args:
            n (int): the index of the sequence.
            
        Returns:
            float: the nth Stieltjes constant.
            
        Reference:
            https://mathworld.wolfram.com/StieltjesConstants.html
        """
        if n == 1:
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
            
    def favard_constants(self, n: int) -> float:
        """Returns the nth Favard constant.
        
        Args:
            n (int): the index of the sequence.
            
        Returns:
            float: the nth Favard constant.
            
        Reference:
            https://mathworld.wolfram.com/FavardConstants.html
        """
        if n < 1:
            raise ValueError("The index n should be a positive integer.")
        elif n == 1:
            return 1
        else:
            return sum([Sequences.favard_constants(self, i) * Sequences.favard_constants(self, n-i) / (i+n-i-1) 
                        for i in range(1, n)])
        
        
    def generalized_bruns_constant(self, n: int) -> float:
        """Returns the nth generalized Bruns constant.
        
        Args:
            n (int): the index of the sequence.
            
        Returns:
            float: the nth generalized Bruns constant.
            
        Reference:
            https://mathworld.wolfram.com/GeneralizedBrunsConstant.html
        """
        if n < 1:
            raise ValueError("The index n should be a positive integer.")
        elif n == 1:
            return 1
        else:
            return sum([abs(Sequences.generalized_bruns_constant(self, i) - Sequences.generalized_bruns_constant(self, i-1)) 
                        for i in range(2, n+1)]) + 1
        
    def champernowne_constants(self, n: int) -> float:
        """Returns the nth Champernowne constant.
        
        Args:
            n (int): the index of the sequence.
            
        Returns:
            float: the nth Champernowne constant.
            
        Reference:
            https://mathworld.wolfram.com/ChampernowneConstant.html
        """
        if n < 1:
            raise ValueError("n should be a positive integer")
        if n == 1:
            return 0.12345678910111213141516171819202122
        else:
            prev = self.champernowne_constants(n-1)
            return float(str(prev) + str(n+8))
            

    def lagrange_number(self, n: int) -> int:
        """Returns the nth Lagrange number.
        
        Args:
            n (int): the index of the sequence.
            
        Returns:
            int: the nth Lagrange number.
            
        Reference:
            https://mathworld.wolfram.com/LagrangeNumber.html
        """
        if n < 1:
            raise ValueError("n should be a positive integer")
        if n == 1:
            return 1
        else:
            return n * self.lagrange_number(n-1) - (-1)**n

    
    def fellers_coin_tossing_constants(self, n: int) -> float:
        """Returns the nth Feller's coin-tossing constant.
        
        Args:
            n (int): the index of the sequence.
            
        Returns:
            float: the nth Feller's coin-tossing constant.
            
        Reference:
            https://mathworld.wolfram.com/FellersCoin-TossingConstants.html
        """
        result = 0
        for k in range(n + 1):
            result += (-1) ** k / (2 ** (2 ** k))
        return result
    
    def stoneham_number(self, n: int) -> int:
        """Returns the nth Stoneham number.
        
        Args:
            n (int): the index of the sequence.
            
        Returns:
            int: the nth Stoneham number.
            
        Reference:
            https://mathworld.wolfram.com/StonehamNumber.html
        """
        if n == 0:
            return 1
        else:
            return (3 * Sequences.stoneham_number(n - 1) + 1) // 2
    
    def beraha_constants(self, n: int) -> float:
        """Returns the nth Beraha constant.
        
        Args:
            n (int): the index of the sequence.
            
        Returns:
            float: the nth Beraha constant.
            
        Reference:
            https://mathworld.wolfram.com/BerahasConstant.html
        """
        if n == 0:
            return 1
        else:
            return 1 + 1 / Sequences.beraha_constants(n - 1)
    
    def chvatal_sankoff_constants(self, n: int) -> float:
        """Returns the nth Chvátal-Sankoff constant.
        
        Args:
            n (int): the index of the sequence.
            
        Returns:
            float: the nth Chvátal-Sankoff constant.
            
        Reference:
            https://mathworld.wolfram.com/Chvatal-SankoffConstants.html
        """
        result = 0
        for k in range(n + 1):
            binom = MathFunctions.comb(2 ** k, k)
            result += (-1) ** k * binom ** 2
        return result
    
    def hyperharmonic_number(self, n: int, p: int) -> float:
        """
        Computes the hyperharmonic number H(n,p), which is defined as the sum of the p-th powers of the reciprocals of
        the first n positive integers.
        
        Args:
        - n (int): The positive integer up to which to compute the sum.
        - p (int): The exponent to which to raise the reciprocals of the integers.
        
        Returns:
        - H (float): The hyperharmonic number H(n,p).
        
        Symbols:
        - H(n,p): hyperharmonic number of order p and degree n.
        """
        H = 0
        for i in range(1, n+1):
            H += 1 / i ** p
        return H


    def gregory_number(self, n: int) -> float:
        """
        Computes the nth Gregory number, which is defined as the alternating sum of the reciprocals of the odd
        positive integers, up to the nth term.
        
        Args:
        - n (int): The positive integer up to which to compute the alternating sum.
        
        Returns:
        - G (float): The nth Gregory number.
        
        Symbols:
        - G(n): nth Gregory number.
        """
        G = 0
        for i in range(1, n+1):
            if i % 2 == 1:
                G += 1 / i
            else:
                G -= 1 / i
        return G


    def metallic_mean(self, x: float) -> float:
        """
        Computes the value of the metallic mean of x, which is the positive solution to the equation x = 1/(1+x).
        
        Args:
        - x (float): The value for which to compute the metallic mean.
        
        Returns:
        - mm (float): The value of the metallic mean of x.
        
        Symbols:
        - mm(x): metallic mean of x.
        """
        mm = (1 + Algorithm.square_root(1 + 4*x)) / 2
        return mm


#####################################################

class HyperbolicFunctions:
    """
    A class representing the six hyperbolic functions: sinh, cosh, tanh, coth, sech, and csch.

    References:
        * Weisstein, E. W. (n.d.). Hyperbolic functions. MathWorld--A Wolfram Web Resource. 
        Retrieved October 11, 2021, from https://mathworld.wolfram.com/HyperbolicFunctions.html
    """
    @staticmethod
    def sinh(x):
        """Returns the hyperbolic sine of x.

        Args:
            x (float): The input value in radians.

        Returns:
            float: The hyperbolic sine of x.
        """
        return (Algorithm.exp(x) - Algorithm.exp(-x)) / 2

    @staticmethod
    def cosh(x):
        """Returns the hyperbolic cosine of x.

        Args:
            x (float): The input value in radians.

        Returns:
            float: The hyperbolic cosine of x.
        """
        return (Algorithm.exp(x) + Algorithm.exp(-x)) / 2

    @staticmethod
    def tanh(x):
        """Returns the hyperbolic tangent of x.

        Args:
            x (float): The input value in radians.

        Returns:
            float: The hyperbolic tangent of x.
        """
        return HyperbolicFunctions.sinh(x) / HyperbolicFunctions.cosh(x)

    @staticmethod
    def coth(x):
        """Returns the hyperbolic cotangent of x.

        Args:
            x (float): The input value in radians.

        Returns:
            float: The hyperbolic cotangent of x.
        """
        return 1 / HyperbolicFunctions.tanh(x)

    @staticmethod
    def sech(x):
        """Returns the hyperbolic secant of x.

        Args:
            x (float): The input value in radians.

        Returns:
            float: The hyperbolic secant of x.
        """
        return 1 / HyperbolicFunctions.cosh(x)

    @staticmethod
    def csch(x):
        """Returns the hyperbolic cosecant of x.

        Args:
            x (float): The input value in radians.

        Returns:
            float: The hyperbolic cosecant of x.
        """
        return 1 / HyperbolicFunctions.sinh(x)



##################################################

class ComplexNumber:
    """
    A class representing a complex number.

    Attributes:
        real (float): The real part of the complex number.
        imag (float): The imaginary part of the complex number.
    """

    def __init__(self, real=0, imag=0):
        """
        Initializes a complex number.

        Args:
            real (float): The real part of the complex number.
            imag (float): The imaginary part of the complex number.
        """
        self.real = real
        self.imag = imag

    def __repr__(self):
        """
        Returns a string representation of the complex number.

        Returns:
            str: A string representation of the complex number.
        """
        return f"{self.real} + {self.imag}j"

    def __add__(self, other):
        """
        Adds two complex numbers.

        Args:
            other (ComplexNumber): The complex number to add.

        Returns:
            ComplexNumber: The sum of the two complex numbers.
        """
        return ComplexNumber(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other):
        """
        Subtracts two complex numbers.

        Args:
            other (ComplexNumber): The complex number to subtract.

        Returns:
            ComplexNumber: The difference of the two complex numbers.
        """
        return ComplexNumber(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other):
        """
        Multiplies two complex numbers.

        Args:
            other (ComplexNumber): The complex number to multiply.

        Returns:
            ComplexNumber: The product of the two complex numbers.
        """
        real = self.real * other.real - self.imag * other.imag
        imag = self.real * other.imag + self.imag * other.real
        return ComplexNumber(real, imag)

    def __truediv__(self, other):
        """
        Divides two complex numbers.

        Args:
            other (ComplexNumber): The complex number to divide.

        Returns:
            ComplexNumber: The quotient of the two complex numbers.
        """
        denom = other.real**2 + other.imag**2
        real = (self.real * other.real + self.imag * other.imag) / denom
        imag = (self.imag * other.real - self.real * other.imag) / denom
        return ComplexNumber(real, imag)

    def conjugate(self):
        """
        Computes the conjugate of the complex number.

        Returns:
            ComplexNumber: The conjugate of the complex number.
        """
        return ComplexNumber(self.real, -self.imag)

    def modulus(self):
        """
        Computes the modulus (magnitude) of the complex number.

        Returns:
            float: The modulus of the complex number.
        """
        return (self.real**2 + self.imag**2)**0.5


############################################################

class RealNumber:
    """
    A class representing a real number.
    """
    
    def __init__(self, value):
        """
        Initializes a new RealNumber object with the given value.

        Parameters:
            value (float): The value of the real number.

        Returns:
            RealNumber: A new RealNumber object.
        """
        self.value = value

    def __str__(self):
        """
        Returns a string representation of the real number.

        Returns:
            str: A string representation of the real number.
        """
        return str(self.value)

    def __repr__(self):
        """
        Returns a string representation of the real number.

        Returns:
            str: A string representation of the real number.
        """
        return str(self.value)

    def __eq__(self, other):
        """
        Checks whether the real number is equal to another object.

        Parameters:
            other (object): The object to compare with.

        Returns:
            bool: True if the real number is equal to the other object, False otherwise.
        """
        if isinstance(other, RealNumber):
            return self.value == other.value
        elif isinstance(other, float):
            return self.value == other
        else:
            return False

    def __ne__(self, other):
        """
        Checks whether the real number is not equal to another object.

        Parameters:
            other (object): The object to compare with.

        Returns:
            bool: True if the real number is not equal to the other object, False otherwise.
        """
        return not self.__eq__(other)

    def __lt__(self, other):
        """
        Checks whether the real number is less than another object.

        Parameters:
            other (object): The object to compare with.

        Returns:
            bool: True if the real number is less than the other object, False otherwise.
        """
        if isinstance(other, RealNumber):
            return self.value < other.value
        elif isinstance(other, float):
            return self.value < other
        else:
            return NotImplemented

    def __le__(self, other):
        """
        Checks whether the real number is less than or equal to another object.

        Parameters:
            other (object): The object to compare with.

        Returns:
            bool: True if the real number is less than or equal to the other object, False otherwise.
        """
        if isinstance(other, RealNumber):
            return self.value <= other.value
        elif isinstance(other, float):
            return self.value <= other
        else:
            return NotImplemented

    def __gt__(self, other):
        """
        Checks whether the real number is greater than another object.

        Parameters:
            other (object): The object to compare with.

        Returns:
            bool: True if the real number is greater than the other object, False otherwise.
        """
        if isinstance(other, RealNumber):
            return self.value > other.value
        elif isinstance(other, float):
            return self.value > other
        else:
            return NotImplemented

    def __ge__(self, other):
        """
        Checks whether the real number is greater than or equal to another object.

        Parameters:
            other (object): The object to compare with.

        Returns:
            bool: True if the real number is greater than or equal to the other object, False otherwise.
        """
        if isinstance(other, RealNumber):
            return self.value >= other.value
        elif isinstance(other, float):
            return self.value >= other
        else:
            return NotImplemented
        
    def __add__(self, other):
        """
        Adds two RealNumber objects.

        Parameters:
            other (RealNumber or float): The RealNumber object or float to add.

        Returns:
            RealNumber: A new RealNumber object with the sum of the two numbers.
        """
        if isinstance(other, RealNumber):
            return RealNumber(self.value + other.value)
        elif isinstance(other, float):
            return RealNumber(self.value + other)
        else:
            return NotImplemented

    def __sub__(self, other):
        """
        Subtracts two RealNumber objects.

        Parameters:
            other (RealNumber or float): The RealNumber object or float to subtract.

        Returns:
            RealNumber: A new RealNumber object with the difference of the two numbers.
        """
        if isinstance(other, RealNumber):
            return RealNumber(self.value - other.value)
        elif isinstance(other, float):
            return RealNumber(self.value - other)
        else:
            return NotImplemented

    def __mul__(self, other):
        """
        Multiplies two RealNumber objects.

        Parameters:
            other (RealNumber or float): The RealNumber object or float to multiply.

        Returns:
            RealNumber: A new RealNumber object with the product of the two numbers.
        """
        if isinstance(other, RealNumber):
            return RealNumber(self.value * other.value)
        elif isinstance(other, float):
            return RealNumber(self.value * other)
        else:
            return NotImplemented

    def __truediv__(self, other):
        """
        Divides two RealNumber objects.

        Parameters:
            other (RealNumber or float): The RealNumber object or float to divide by.

        Returns:
            RealNumber: A new RealNumber object with the quotient of the two numbers.
        """
        if isinstance(other, RealNumber):
            return RealNumber(self.value / other.value)
        elif isinstance(other, float):
            return RealNumber(self.value / other)
        else:
            return NotImplemented

    def __abs__(self):
        """
        Returns the absolute value of the RealNumber object.

        Returns:
            RealNumber: A new RealNumber object with the absolute value of the number.
        """
        return RealNumber(abs(self.value))

    def __neg__(self):
        """
        Returns the negation of the RealNumber object.

        Returns:
            RealNumber: A new RealNumber object with the negation of the number.
        """
        return RealNumber(-self.value)

    def sqrt(self):
        """
        Returns the square root of the RealNumber object.

        Returns:
            RealNumber: A new RealNumber object with the square root of the number.
        """
        return RealNumber(self.value ** 0.5)
    
    def __pow__(self, other):
        """
        Computes the power of the real number to the given exponent.

        Parameters:
            other (float): The exponent.

        Returns:
            RealNumber: A new RealNumber object with the result of the power operation.
        """
        return RealNumber(self.value ** other)


#######################################################################################

class RationalNumber:
    """
    A class representing a rational number.

    Attributes:
        numerator (int): The numerator of the rational number.
        denominator (int): The denominator of the rational number.

    Methods:
        simplify: Simplifies the rational number.
        add: Adds two rational numbers.
        subtract: Subtracts two rational numbers.
        multiply: Multiplies two rational numbers.
        divide: Divides two rational numbers.
    """

    def __init__(self, numerator, denominator):
        """
        Initializes a rational number with the given numerator and denominator.

        Args:
            numerator (int): The numerator of the rational number.
            denominator (int): The denominator of the rational number.

        Raises:
            ValueError: If the denominator is zero.
        """
        if denominator == 0:
            raise ValueError("Denominator cannot be zero")
        self.numerator = numerator
        self.denominator = denominator
        self.simplify()

    def __str__(self):
        """
        Returns the string representation of the rational number.
        """
        return f"{self.numerator}/{self.denominator}"

    def simplify(self):
        """
        Simplifies the rational number.
        """
        gcd = self.gcd(self.numerator, self.denominator)
        self.numerator //= gcd
        self.denominator //= gcd

    @staticmethod
    def gcd(a, b):
        """
        Computes the greatest common divisor of two numbers a and b.

        Args:
            a (int): The first number.
            b (int): The second number.

        Returns:
            int: The greatest common divisor of a and b.
        """
        while b:
            a, b = b, a % b
        return a

    def add(self, other):
        """
        Adds two rational numbers.

        Args:
            other (RationalNumber): The other rational number.

        Returns:
            RationalNumber: The sum of the two rational numbers.
        """
        numerator = self.numerator * other.denominator + other.numerator * self.denominator
        denominator = self.denominator * other.denominator
        return RationalNumber(numerator, denominator)

    def subtract(self, other):
        """
        Subtracts two rational numbers.

        Args:
            other (RationalNumber): The other rational number.

        Returns:
            RationalNumber: The difference of the two rational numbers.
        """
        numerator = self.numerator * other.denominator - other.numerator * self.denominator
        denominator = self.denominator * other.denominator
        return RationalNumber(numerator, denominator)

    def multiply(self, other):
        """
        Multiplies two rational numbers.

        Args:
            other (RationalNumber): The other rational number.

        Returns:
            RationalNumber: The product of the two rational numbers.
        """
        numerator = self.numerator * other.numerator
        denominator = self.denominator * other.denominator
        return RationalNumber(numerator, denominator)

    def divide(self, other):
        """
        Divides two rational numbers.

        Args:
            other (RationalNumber): The other rational number.

        Returns:
            RationalNumber: The quotient of the two rational numbers.
        """
        numerator = self.numerator * other.denominator
        denominator = self.denominator * other.numerator
        return RationalNumber(numerator, denominator)

#################################################################

class IntegralNumber:
    """
    A class representing integral numbers.

    Attributes
    ----------
    value : int
        The value of the integral number.

    Methods
    -------
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

    Raises
    ------
    TypeError
        If the argument is not an instance of IntegralNumber.
    ZeroDivisionError
        If the second IntegralNumber object is zero and division is attempted.

    References
    ----------
    - https://en.wikipedia.org/wiki/Integer_(computer_science)
    - https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
    """
    def __init__(self, value: int) -> None:
        """
        Initializes a new instance of the IntegralNumber class with the specified integer value.

        Parameters
        ----------
        value : int
            The integer value to initialize the IntegralNumber object with.
        """
        self.value = value
        
    def __repr__(self) -> str:
        """
        Returns a string representation of the IntegralNumber object.

        Returns
        -------
        str
            A string representation of the IntegralNumber object.
        """
        return f"IntegralNumber({self.value})"
   
    def __eq__(self, other: 'IntegralNumber') -> bool:
        """
        Determines if the current IntegralNumber object is equal to another IntegralNumber object.

        Parameters
        ----------
        other : IntegralNumber
            The IntegralNumber object to compare to.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.
        """
        if isinstance(other, IntegralNumber):
            return self.value == other.value
        return False
        
    def __lt__(self, other: 'IntegralNumber') -> bool:
        """
        Determines if the current IntegralNumber object is less than another IntegralNumber object.

        Parameters
        ----------
        other : IntegralNumber
            The IntegralNumber object to compare to.

        Returns
        -------
        bool
            True if the current object is less than the other object, False otherwise.
        """
        if isinstance(other, IntegralNumber):
            return self.value < other.value
        return False
        
    def __add__(self, other: 'IntegralNumber') -> 'IntegralNumber':
        """
        Adds two IntegralNumber objects.

        Parameters
        ----------
        other : IntegralNumber
            The IntegralNumber object to be added to the current object.

        Returns
        -------
        IntegralNumber
            An IntegralNumber object which is the sum of the current object and the passed object.

        Raises
        ------
        TypeError
            If the passed object is not an IntegralNumber.
        """

        if isinstance(other, IntegralNumber):
            return IntegralNumber(self.value + other.value)
        raise TypeError("Cannot add non-IntegralNumber object.")
        
    def __sub__(self, other: 'IntegralNumber') -> 'IntegralNumber':
        """
        Subtracts two IntegralNumber objects.

        Parameters
        ----------
        other : IntegralNumber
            The IntegralNumber object to be subtracted from the current object.

        Returns
        -------
        IntegralNumber
            An IntegralNumber object which is the difference between the current object and the passed object.

        Raises
        ------
        TypeError
            If the passed object is not an IntegralNumber.
        """
        if isinstance(other, IntegralNumber):
            return IntegralNumber(self.value - other.value)
        raise TypeError("Cannot subtract non-IntegralNumber object.")
        
    def __mul__(self, other: 'IntegralNumber') -> 'IntegralNumber':
        """
        Multiplies two IntegralNumber objects.

        Parameters
        ----------
        other : IntegralNumber
            The IntegralNumber object to be multiplied with the current object.

        Returns
        -------
        IntegralNumber
            An IntegralNumber object which is the product of the current object and the passed object.

        Raises
        ------
        TypeError
            If the passed object is not an IntegralNumber.
        """
        if isinstance(other, IntegralNumber):
            return IntegralNumber(self.value * other.value)
        raise TypeError("Cannot multiply non-IntegralNumber object.")
        
    def __truediv__(self, other: 'IntegralNumber') -> 'IntegralNumber':
        """
        Divides two IntegralNumber objects.

        Parameters
        ----------
        other : IntegralNumber
            The IntegralNumber object to be used as divisor for the current object.

        Returns
        -------
        IntegralNumber
            An IntegralNumber object which is the result of dividing the current object by the passed object.

        Raises
        ------
        TypeError
            If the passed object is not an IntegralNumber.
        ZeroDivisionError
            If the passed object has a value of zero.
        """
        if isinstance(other, IntegralNumber):
            if other.value == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return IntegralNumber(self.value // other.value)
        raise TypeError("Cannot divide non-IntegralNumber object.")

    def add(self, other: 'IntegralNumber') -> 'IntegralNumber':
        """
        Returns the sum of this number and `other`.

        Parameters:
        -----------
        other : IntegralNumber
            The number to add to this number.

        Returns:
        --------
        IntegralNumber
            The sum of this number and `other`.
        """
        return IntegralNumber(self.value + other.value)

    def subtract(self, other: 'IntegralNumber') -> 'IntegralNumber':
        """
        Returns the difference between this number and `other`.

        Parameters:
        -----------
        other : IntegralNumber
            The number to subtract from this number.

        Returns:
        --------
        IntegralNumber
            The difference between this number and `other`.
        """
        return IntegralNumber(self.value - other.value)

    def multiply(self, other: 'IntegralNumber') -> 'IntegralNumber':
        """
        Returns the product of this number and `other`.

        Parameters:
        -----------
        other : IntegralNumber
            The number to multiply with this number.

        Returns:
        --------
        IntegralNumber
            The product of this number and `other`.
        """
        return IntegralNumber(self.value * other.value)

    def divide(self, other: 'IntegralNumber') -> Union['IntegralNumber', None]:
        """
        Returns the quotient of this number and `other`.

        Parameters:
        -----------
        other : IntegralNumber
            The number to divide this number by.

        Returns:
        --------
        Union[IntegralNumber, None]
            The quotient of this number and `other`. Returns None if `other` is zero.
        """
        if other.value == 0:
            return None
        return IntegralNumber(self.value // other.value)

    def __str__(self):
        return str(self.value)


##############################################################################

alg = Algorithm()
con = Constants()
fun = MathFunctions()
seq = Sequences()
hyp = HyperbolicFunctions()
com = ComplexNumber()
ren = RealNumber()
rat = RationalNumber()
inn = IntegralNumber()
