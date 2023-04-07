from Maths import *

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

    def divide(self, other: 'IntegralNumber') -> 'IntegralNumber' | None:
        """
        Returns the quotient of this number and `other`.

        Parameters:
        -----------
        other : IntegralNumber
            The number to divide this number by.

        Returns:
        --------
        IntegralNumber | None
            The quotient of this number and `other`. Returns None if `other` is zero.
        """
        if other.value == 0:
            return None
        return IntegralNumber(self.value // other.value)

    def __str__(self):
        return str(self.value)
