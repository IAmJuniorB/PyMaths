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
