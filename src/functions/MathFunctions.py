from src.utils import Algorithm, Constants


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

    @staticmethod
    def quad(func, a, b, eps=1e-7, maxiter=50):
        """Approximates the definite integral of a function using the adaptive quadrature method.

        Parameters:
            func (callable): A function to integrate.
            a (float): Lower limit of integration.
            b (float): Upper limit of integration.
            eps (float): Desired accuracy.
            maxiter (int): Maximum number of iterations.

        Returns:
            float: The definite integral of the function over the interval [a, b].
        """
        def adaptivesimpson(a, b, eps, fa, fb, fc, level):
            c = (a + b) / 2
            h = b - a
            d = (a + c) / 2
            e = (c + b) / 2
            fd = func(d)
            fe = func(e)
            Sleft = h * (fa + 4 * fd + fc) / 6
            Sright = h * (fc + 4 * fe + fb) / 6
            S2 = Sleft + Sright
            if level >= maxiter or abs(S2 - Sleft - Sright) <= 15 * eps:
                return S2 + (S2 - Sleft - Sright) / 15
            return adaptivesimpson(a, c, eps / 2, fa, fc, fd, level + 1) + adaptivesimpson(c, b, eps / 2, fc, fb, fe, level + 1)

        fa = func(a)
        fb = func(b)
        fc = func((a + b) / 2)
        return adaptivesimpson(a, b, eps, fa, fb, fc, 0)
    
    @staticmethod
    def subsets(s):
        """Generates all possible non-empty subsets of a given iterable.
        
        Parameters:
        s (iterable): The iterable for which subsets are generated.
        
        Returns:
        generator: A generator that yields each subset as a tuple.
        """
        s = list(s)
        n = len(s)
        for i in range(1, 2**n):
            subset = tuple(s[j] for j in range(n) if (i >> j) & 1)
            yield subset

    @staticmethod
    def limit(func, x0, h=1e-8, max_iterations=1000, tol=1e-8):
        """Approximates the limit of a function f(x) as x approaches x0.

        Args:
            func (function): A function of one variable.
            x0 (float): The value of x that x approaches.
            h (float, optional): Step size for computing the numerical derivative. Defaults to 1e-8.
            max_iterations (int, optional): Maximum number of iterations for the numerical approximation. Defaults to 1000.
            tol (float, optional): Tolerance level for stopping iterations. Defaults to 1e-8.

        Returns:
            float: The numerical approximation of the limit.
        """
        f = func
        x = x0
        for i in range(max_iterations):
            f_x = f(x)
            f_x_plus_h = f(x + h)
            derivative = (f_x_plus_h - f_x) / h
            x -= f_x / derivative
            if abs(f(x)) < tol:
                return f(x)
        raise ValueError(f"Failed to converge to a limit within {max_iterations} iterations.")
    
    @staticmethod
    def floor(x):
        """
        Returns the greatest integer less than or equal to x.

        Args:
            x (float): A floating-point number.

        Returns:
            int: The greatest integer less than or equal to x.
        """
        return int(x) if x >= 0 else int(x) - 1

    @staticmethod
    def brentq(f, a, b, maxiter=100):
        """Find a root of a function in the interval [a, b] using Brent's method.
        
        Args:
            f (callable): The function to find the root of.
            a (float): The left endpoint of the interval.
            b (float): The right endpoint of the interval.
            maxiter (int, optional): The maximum number of iterations to perform. Defaults to 100.
            
        Returns:
            float: The root of the function.
            
        Raises:
            ValueError: If the root cannot be found within the maximum number of iterations.
        """
        eps = 1e-12
        tol = 1e-12
        
        fa = f(a)
        fb = f(b)
        
        if fa * fb > 0:
            raise ValueError("Root not bracketed in the interval [%f, %f]" % (a, b))
        
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        c = a
        fc = fa
        d = 0
        e = 0
        
        for i in range(maxiter):
            if fa != fc and fb != fc:
                s = (a*fb*fc)/((fa - fb)*(fa - fc)) + (b*fa*fc)/((fb - fa)*(fb - fc)) + (c*fa*fb)/((fc - fa)*(fc - fb))
            else:
                s = b - fb*(b - a)/(fb - fa)
            
            if s < (3*a + b)/4 or s > b or (e != 0 and abs(s - d) >= abs(e/2)):
                s = (a + b)/2
                e = d = b - a
            
            fs = f(s)
            d = e
            e = b - a
            
            if abs(fs) < abs(fb):
                a = b
                b = s
                fa = fb
                fb = fs
            else:
                a = s
                fa = fs
            
            if abs(fb) < tol or abs(b - a) < eps:
                return b
        
        raise ValueError("Failed to converge after %d iterations" % maxiter)

    @staticmethod
    def totient(n: int) -> int:
        """Returns the Euler's totient function for the given integer 'n',
        which counts the number of positive integers up to 'n' that are
        relatively prime to 'n'.

        Args:
            n (int): The integer to compute the totient function for

        Returns:
            int: The value of Euler's totient function for 'n'
        """
        if n == 1:
            return 1
        tot = n
        p = 2
        while p*p <= n:
            if n % p == 0:
                tot -= tot // p
                while n % p == 0:
                    n //= p
            p += 1
        if n > 1:
            tot -= tot // n
        return tot
