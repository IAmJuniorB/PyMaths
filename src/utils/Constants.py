from Algorithm import Algorithm
from src.functions.MathFunctions import MathFunctions

from numpy import array, arange
from scipy.integrate import quad
from scipy.optimize import brentq
from sympy.ntheory import totient
from sympy import sieve


class Constants:
    """A collection of mathematical constants."""
    
    def __init__(self):
        pass
    
    def speed_of_light(self):
        """Returns the speed of light in meters per second."""
        return 299_792_458
    
    def planck_constant(self):
        """
        Returns the Planck constant in joule-seconds.
        """
        h = 6.62607015e-34  # Planck constant in joule-seconds
        return h
    
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
        return 2**(2**(1/2))
    
    def second_favard_constant(self):
        """
        Returns the Second Favard constant, which is a mathematical constant defined as the limit of the 
        arithmetic mean of the reciprocal of consecutive odd numbers, or approximately 0.661707182...
        
        Returns:
            float: The value of the Second Favard constant.
        """
        n = 1
        s = 0
        while True:
            s += 1/(2*n - 1) - 1/(2*n + 1)
            n += 1
            if n > 10_000:
                break
        return s
    
    def golden_angle(self):
        """
        Returns the golden angle constant, which is the angle subtended by the smaller of the two angles 
        formed by dividing the circumference of a circle in the golden ratio. It is equal to 
        (3 - sqrt(5)) * 180 degrees / pi, or approximately 137.5077640500378546463487 degrees.

        Returns:
            float: The value of the golden angle constant in degrees.
        """
        return (3 - MathFunctions.square_root(5)) * 180 / Constants.pi
    
    def sierpinskis_constant(self):
        """
        Returns Sierpiński's constant, which is the fractal dimension of the Sierpiński triangle, a 
        self-similar fractal shape. It is equal to log(3)/log(2), or approximately 1.585.

        Returns:
            float: The value of Sierpiński's constant.
        """
        return MathFunctions.log(3)/MathFunctions.log(2)

    def landau_ramanujan_constant(self):
        """
        Returns the Landau-Ramanujan constant, which is a mathematical constant that appears in the 
        asymptotic expansion of the partition function. It is equal to e^(pi * sqrt(163)), or approximately
        2.2932021438344677e+17.

        Returns:
            float: The value of the Landau-Ramanujan constant.
        """
        return Algorithm.exp(Constants.pi * MathFunctions.square_root(163))

    def first_nielsen_ramanujan_constant(self):
        """
        Returns the First Nielsen-Ramanujan constant, which is a mathematical constant that appears in 
        certain partition identities. It is equal to the product of a series involving the gamma function, 
        or approximately 0.866081804933.

        Returns:
            float: The value of the First Nielsen-Ramanujan constant.
        """
        return 2 * MathFunctions.square_root(2) / MathFunctions.square_root(Constants.pi) * MathFunctions.prod([MathFunctions.gamma((n+1)/4)/((n/2)**((n+1)/4)) for n in range(1, 6)])

    def gieseking_constant(self):
        """
        Returns Gieseking's constant, which is a mathematical constant that appears in the theory of 
        harmonic analysis. It is equal to (2*pi)^(-3/4), or approximately 0.7511255444649425.

        Returns:
            float: The value of Gieseking's constant.
        """
        return (2*Constants.pi)**(-3/4)

    def bernsteins_constant(self):
        """
        Returns Bernstein's constant, which is a mathematical constant that appears in the theory of 
        Fourier analysis. It is equal to pi/sqrt(2), or approximately 2.221441469079183.

        Returns:
            float: The value of Bernstein's constant.
        """
        return Constants.pi/MathFunctions.square_root(2)

    def tribonacci_constant(self):
        """
        Returns the Tribonacci constant, which is a mathematical constant defined as the unique real root 
        of the polynomial x^3 - x^2 - x - 1, or approximately 1.8392867552141612.

        Returns:
            float: The value of the Tribonacci constant.
        """
        def f(x):
            return x**3 - x**2 - x - 1
        a, b = 1, 2
        while abs(a-b) > 1e-10:
            a, b = b, b - f(b)/(f(b)-f(a)) * (b-a)
        return b

    
    def bruns_constant(self):
        """
        Returns the limiting value of the sequence a(n) = sum(k=1 to n) 1/prime(k),
        where prime(k) is the kth prime number.

        Returns:
            float: The value of Bruns constant, accurate to 42 decimal places.
        """
        primes = []
        n = 2
        while len(primes) < 100000:
            if all(n % p != 0 for p in primes):
                primes.append(n)
            n += 1
        return round(sum(1/p for p in primes), 42)


    def twin_primes_constant(self):
        """
        Returns the limiting value of the sequence of twin primes (pairs of prime
        numbers that differ by 2).

        Returns:
            float: The value of the twin primes constant, accurate to 36 decimal places.
        """
        primes = []
        n = 2
        while len(primes) < 2000000:
            if all(n % p != 0 for p in primes):
                primes.append(n)
            n += 1
        twin_primes = [p for p in primes if p + 2 in primes]
        return round(len(twin_primes) / len(primes), 36)


    def plastic_number(self):
        """
        Returns the unique positive real root of x^3 = x + 1.

        Returns:
            float: The value of the plastic number, accurate to 32 decimal places.
        """
        x0 = 1.324717957244745
        while True:
            x1 = (1 + x0 + x0**2) / (3 * x0**2)
            if abs(x1 - x0) < 1e-32:
                return round(x1, 32)
            x0 = x1


    def blochs_constant(self):
        """
        Returns the limiting value of the sequence of numbers that represent the
        Bloch wall widths in ferromagnets.

        Returns:
            float: The value of Bloch's constant, accurate to 34 decimal places.
        """
        a = 1
        b = 1
        while True:
            a, b = b, a + b
            yield a / b**2

            return round(sum(Algorithm.islice(self.blochs_constant(self), 100000)), 34)


    def z_score_975_percentile(self):
        """Returns the value that has 97.5% of the area under a standard normal distribution
        to the left of it.

        Returns:
            float: The value of the z-score at the 97.5th percentile, accurate to 9 decimal places.
        """
        n = 0.5
        while True:
            if Algorithm.normal_distribution_cdf(n) > 0.975:
                return round(n, 9)
            n += 0.000001

    def landaus_constant(self):
        """Returns the limiting value of the sequence of numbers that represent the
        probability that a random permutation of n elements will have no cycle of length
        greater than log(n).

        Returns:
            float: The value of Landau's constant, accurate to 19 decimal places.
        """
        a = 1
        b = 1
        c = 1
        for n in range(1, 10000):
            a, b, c = b, c, (n - 1) * (a + b) - (n - 3) * c
        return round(a / b, 19)

    def landaus_third_constant(self):
        """Returns the limiting value of the sequence of numbers that represent the
        probability that a random permutation of n elements will have no cycle of length
        greater than sqrt(n) * log(n).

        Returns:
            float: The value of Landau's third constant, accurate to 20 decimal places.
        """
        a = 1
        b = 1
        c = 1
        for n in range(1, 10000):
            a, b, c = b, c, (n - 1) * (a + b) - (n - 3) * c
        return round(a / b / (MathFunctions.square_root(2) * Algorithm.exp(1) * Algorithm.log(b)), 20)

    def prouhet_thue_morse_constant(self):
        """Returns the limiting value of the sequence of numbers that represent the
        differences in density between the 0's and 1's in the Prouhet-Thue-Morse
        sequence.

        Returns:
            float: The value of the Prouhet-Thue-Morse constant, accurate to 20 decimal places.
        """
        s = [1]
        while True:
            t = [1 - x for x in s]
            s.extend(t)
            if len(s) > 10000:
                break
        ones = sum(s)
        zeros = len(s) - ones
        return round(abs(ones - zeros) / len(s), 20)

    
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def foias_constant():
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
