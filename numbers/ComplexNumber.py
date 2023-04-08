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
