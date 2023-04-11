from src.utils.Algorithm import Algorithm


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
