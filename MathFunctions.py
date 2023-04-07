from Maths import *

class MathFunctions:
    """
    A class containing various mathematical functions.
    """
    
    def __init__(self):
        pass
    
    def gamma(self, x):
        """
        Compute the value of the gamma function at the given value of x.

        Args:
            x (float): The value at which the gamma function is to be evaluated.

        Returns:
            float: The value of the gamma function at the given value of x.

        Raises:
            ValueError: If x is negative and not an integer.

        Notes:
            The gamma function is defined as the integral from zero to infinity of t^(x-1) * exp(-t) dt.
            For positive integers, the gamma function can be computed recursively as (n-1)!.
            For x <= 0, the gamma function is undefined, but we return NaN to avoid raising an error.
        """
        if x.is_integer() and x <= 0:
            return float('nan')
        elif x == 1:
            return 1
        elif x < 1:
            return self.gamma(x + 1) / x
        else:
            y = (2 * x - 1) / 2
            result = 1
            while y > 1:
                result *= y
                y -= 1
            return result * MathFunctions.square_root(Constants.pi) / 2 ** x
        
    def prod(iterable):
        """
        Returns the product of all the elements in the given iterable.

        Args:
            iterable: An iterable of numeric values.

        Returns:
            The product of all the elements in the iterable. If the iterable is empty,
            returns 1.
        """
        result = 1
        for x in iterable:
            result *= x
        return result

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
