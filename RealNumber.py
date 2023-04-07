from Maths import *

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
