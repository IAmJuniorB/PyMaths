from Algorithm import Algorithm
from Constants import Constants
from src.functions.MathFunctions import MathFunctions


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
