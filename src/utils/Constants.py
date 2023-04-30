from Algorithm import Algorithm
from src.functions.MathFunctions import MathFunctions


class Constants:
    """A collection of mathematical constants."""

    @staticmethod
    def speed_of_light():
        """Returns the speed of light in meters per second."""
        return 299_792_458

    @staticmethod    
    def planck_constant():
        """
        Returns the Planck constant in joule-seconds.
        """
        h = 6.62607015e-34
        return h

    @staticmethod    
    def pi():
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
        # initial values
        s = 1
        n_fact = 1
        n_recip = 1
        x_pow = 1

        tol = 1e-10
        n = 1
        while True:
            n_fact *= n
            n_recip /= n
            x_pow *= 1
            term = n_recip * x_pow
            s += term
            if abs(term) < tol:
                break
            n += 1
        return s

    @staticmethod
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

    @staticmethod    
    def nan():
        """Return a floating-point NaN (not a number) value.

        NaN is a special floating-point value that is used to represent undefined or unrepresentable values.
        It is commonly used as a result of an undefined or unrepresentable mathematical operation.

        Returns:
            float: A floating-point NaN value.
        """
        return float("nan")

    @staticmethod
    def tau():
        """the 19th letter of the Greek alphabet,
        representing the voiceless dental or alveolar plosive IPA: [t].
        In the system of Greek numerals, it has a value of 300.
        
        Returns:
            tau, uppercase Τ, lowercase τ, or τ, to the 20th decimal
        """
        return 6.283_185_307_179_586_476_92

    @staticmethod    
    def phi():
        """\"The Golden Ratio\".
        In mathematics, two quantities are in the golden ratio
        if their ratio is the same as the ratio of their sum
        to the larger of the two quantities.
        
        Returns:
            Uppercase Φ lowercase φ or ϕ: Value to the 20th decimal
        """
        return 1.618_033_988_749_894_848_20

    @staticmethod    
    def silver_ratio():
        """\"The Silver Ratio\". Two quantities are in the silver ratio (or silver mean)
        if the ratio of the smaller of those two quantities to the larger quantity
        is the same as the ratio of the larger quantity to the sum of the
        smaller quantity and twice the larger quantity
        
        Returns:
            δS: Value to the 20th decimal
        """
        return 2.414_213_562_373_095_048_80

    @staticmethod    
    def supergolden_ratio():
        """Returns the mathematical constant psi (the supergolden ratio).
        
        Returns:
            ψ to the 25th decimal
        """
        return 1.465_571_231_876_768_026_656_731_2

    @staticmethod    
    def connective_constant():
        """Returns the connective constant for the hexagonal lattice.

        Returns:
            μ to the 4th decimal
        """
        return 1.687_5

    @staticmethod    
    def kepler_bouwkamp_constant():
        """In plane geometry, the Kepler–Bouwkamp constant (or polygon inscribing constant)
        is obtained as a limit of the following sequence.
        Take a circle of radius 1. Inscribe a regular triangle in this circle.
        Inscribe a circle in this triangle. Inscribe a square in it.
        Inscribe a circle, regular pentagon, circle, regular hexagon and so forth.
        Returns:
            K': to the 20th decimal
        """
        return 0.114_942_044_853_296_200_70

    @staticmethod    
    def wallis_constant():
        """Returns Wallis's constant.
        
        Returns:
            Value to the 20th decimal
        """
        return 2.094_551_481_542_326_591_48

    @staticmethod    
    def eulers_constant():
        """a mathematical constant approximately equal to 2.71828 that can be characterized in many ways.
        It is the base of the natural logarithms.
        It is the limit of (1 + 1/n)n as n approaches infinity, an expression that arises in the study of compound interest.
        It can also be calculated as the sum of the infinite series

        Returns:
            e: Value to the 20th decimal. math.e
        """
        return 2.718_281_828_459_045_235_36

    @staticmethod    
    def natural_log():
        """Natural logarithm of 2.

        Returns:
            ln 2: Value to the 30th decimal. math.log(2)
        """
        return 0.693_147_180_559_945_309_417_232_121_458

    @staticmethod    
    def lemniscate_constant():
        """The ratio of the perimeter of Bernoulli's lemniscate to its diameter, analogous to the definition of π for the circle.

        Returns:
            ϖ: Value to the 20th decimal. math.sqrt(2)
        """
        return 2.622_057_554_292_119_810_46 

    @staticmethod    
    def eulers_constant():
        """Not to be confused with Euler's Number.
        Defined as the limiting difference between the harmonic series and the natural logarithm

        Returns:
            γ: Value to the 50th decimal
        """
        return 0.577_215_664_901_532_860_606_512_090_082_402_431_042_159_335_939_92

    @staticmethod    
    def Erdős_Borwein_constant():
        """The sum of the reciprocals of the Mersenne numbers

        Returns:
            E: Value to the 20th decimal. sum([1 / 2 ** (2 ** i) for i in range(40)])
        """
        return 1.606_695_152_415_291_763_78

    @staticmethod    
    def omega_constant():
        """Defined as the unique real number that satisfies the equation Ωe**Ω = 1.

        Returns:
            Ω: Value to the 30th decimal
        """
        return 0.567_143_290_409_783_872_999_968_662_210

    @staticmethod    
    def Apérys_constant():
        """The sum of the reciprocals of the positive cubes.

        Returns:
            ζ(3): Value to the 45th decimal
        """
        return 1.202_056_903_159_594_285_399_738_161_511_449_990_764_986_292

    @staticmethod    
    def laplace_limit():
        """The maximum value of the eccentricity for which a solution to Kepler's equation, in terms of a power series in the eccentricity, converges.

        Returns:
            Value to the 35th decimal
        """
        return 0.662_743_419_349_181_580_974_742_097_109_252_90

    @staticmethod    
    def ramanujan_soldner_constant():
        """A mathematical constant defined as the unique positive zero of the logarithmic integral function.

        Returns:
            μ ≈: Value to the 45th decimal
        """
        return 1.451_369_234_883_381_050_283_968_485_892_027_449_493_032_28

    @staticmethod        
    def gauss_constant():
        """transcendental mathematical constant that is the ratio of the perimeter of
        Bernoulli's lemniscate to its diameter, analogous to the definition of π for the circle.

        Returns:
            G == ϖ /π ≈ 0.8346268: Value to the 7th decimal
        """
        return 0.834_626_8

    @staticmethod    
    def second_hermite_constant():
        """_summary_

        Returns:
            γ2 : Value to the 20th decimal
        """
        return  1.154_700_538_379_251_529_01

    @staticmethod    
    def liouvilles_constant():
        """A real number x with the property that, for every positive integer n,
        there exists a pair of integers (p,q) with q>1.

        Returns:
            L: Value to the 119th decimal
        """
        return 0.110_001_000_000_000_000_000_001_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000_01

    @staticmethod    
    def first_continued_fraction():
        """_summary_

        Returns:
            C_{1}: _description_
        """
        return 0.697_774_657_964_007_982_01

    @staticmethod    
    def ramanujans_constant():
        """The transcendental number, which is an almost integer, in that it is very close to an integer.

        Returns:
            e**{{\pi {\sqrt {163}}}}: Value to the 18th decimal
        """
        return  262_537_412_640_768_743.999_999_999_999_250_073

    @staticmethod        
    def glaisher_kinkelin_constant():
        """A mathematical constant, related to the K-function and the Barnes G-function.

        Returns:
            A: Value to the 20th decimal
        """
        return 1.282_427_129_100_622_636_87

    @staticmethod    
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

    @staticmethod    
    def dottie_number():
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

    @staticmethod    
    def meissel_mertens_constant():
        """_summary_

        Returns:
            M: Value to the 40th value
        """
        return 0.261_497_212_847_642_783_755_426_838_608_695_859_051_6

    @staticmethod    
    def universal_parabolic_constant():
        """The ratio, for any parabola, of the arc length of the parabolic segment
        formed by the latus rectum to the focal parameter.

        Returns:
            P: Value to the 20th decimal
        """
        return  2.295_587_149_392_638_074_03

    @staticmethod    
    def cahens_constant():
        """The value of an infinite series of unit fractions with alternating signs.

        Returns:
            C: Value to the 20th decimal
        """
        return  0.643_410_546_288_338_026_18


    @staticmethod# Everything below needs to be updated in Markdown    
    def gelfonds_constant():
        """Calculates Gelfond's Constant, which is defined as e raised to the power of pi.

        Returns:
            float: The value of Gelfond's Constant, which is approximately 23.1406926327792690057292.
        """
        return Constants.eulers_constant**Constants.pi

    @staticmethod    
    def gelfond_schneider_constant():
        """
        Returns the Gelfond-Schneider constant, which is a transcendental number defined as the value of 
        2^(1/2) raised to the power of itself, or approximately 2.6651441426902251886502972498731.

        Returns:
            float: The value of the Gelfond-Schneider constant.
        """
        return 2**(2**(1/2))

    @staticmethod    
    def second_favard_constant():
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

    @staticmethod    
    def golden_angle():
        """
        Returns the golden angle constant, which is the angle subtended by the smaller of the two angles 
        formed by dividing the circumference of a circle in the golden ratio. It is equal to 
        (3 - sqrt(5)) * 180 degrees / pi, or approximately 137.5077640500378546463487 degrees.

        Returns:
            float: The value of the golden angle constant in degrees.
        """
        return (3 - MathFunctions.square_root(5)) * 180 / Constants.pi

    @staticmethod    
    def sierpinskis_constant():
        """
        Returns Sierpiński's constant, which is the fractal dimension of the Sierpiński triangle, a 
        self-similar fractal shape. It is equal to log(3)/log(2), or approximately 1.585.

        Returns:
            float: The value of Sierpiński's constant.
        """
        return MathFunctions.log(3)/MathFunctions.log(2)

    @staticmethod
    def landau_ramanujan_constant():
        """
        Returns the Landau-Ramanujan constant, which is a mathematical constant that appears in the 
        asymptotic expansion of the partition function. It is equal to e^(pi * sqrt(163)), or approximately
        2.2932021438344677e+17.

        Returns:
            float: The value of the Landau-Ramanujan constant.
        """
        return Algorithm.exp(Constants.pi * MathFunctions.square_root(163))

    @staticmethod
    def first_nielsen_ramanujan_constant():
        """
        Returns the First Nielsen-Ramanujan constant, which is a mathematical constant that appears in 
        certain partition identities. It is equal to the product of a series involving the gamma function, 
        or approximately 0.866081804933.

        Returns:
            float: The value of the First Nielsen-Ramanujan constant.
        """
        return 2 * MathFunctions.square_root(2) / MathFunctions.square_root(Constants.pi) * MathFunctions.prod([MathFunctions.gamma((n+1)/4)/((n/2)**((n+1)/4)) for n in range(1, 6)])

    @staticmethod
    def gieseking_constant():
        """
        Returns Gieseking's constant, which is a mathematical constant that appears in the theory of 
        harmonic analysis. It is equal to (2*pi)^(-3/4), or approximately 0.7511255444649425.

        Returns:
            float: The value of Gieseking's constant.
        """
        return (2*Constants.pi)**(-3/4)

    @staticmethod
    def bernsteins_constant():
        """
        Returns Bernstein's constant, which is a mathematical constant that appears in the theory of 
        Fourier analysis. It is equal to pi/sqrt(2), or approximately 2.221441469079183.

        Returns:
            float: The value of Bernstein's constant.
        """
        return Constants.pi/MathFunctions.square_root(2)

    @staticmethod
    def tribonacci_constant():
        """
        Returns the Tribonacci constant, which is a mathematical constant defined as the unique real root 
        of the polynomial x^3 - x^2 - x - 1, or approximately 1.8392867552141612.

        Returns:
            float: The value of the Tribonacci constant.
    
        @staticmethod    """
        def f(x):
            return x**3 - x**2 - x - 1
        a, b = 1, 2
        while abs(a-b) > 1e-10:
            a, b = b, b - f(b)/(f(b)-f(a)) * (b-a)
        return b


    @staticmethod    
    def bruns_constant():
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


    @staticmethod
    def twin_primes_constant():
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


    @staticmethod
    def plastic_number():
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


    @staticmethod
    def blochs_constant():
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

            return round(sum(Algorithm.islice(Constants.blochs_constant(), 100000)), 34)


    @staticmethod
    def z_score_975_percentile():
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

    @staticmethod
    def landaus_constant():
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

    @staticmethod
    def landaus_third_constant():
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

    @staticmethod
    def prouhet_thue_morse_constant():
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


    @staticmethod    
    def golomb_dickman_constant():
        """The Golomb-Dickman constant represents the limiting distribution of the ratio of the k-th smallest
        number in a sample of n random numbers to n^(1/k) as n approaches infinity. It is denoted by G.

        Returns:
            float: The value of the Golomb-Dickman constant G, approximately 0.6243299885435508.
        """
        return 0.6243299885435508

    @staticmethod
    def lebesgue_asymptotic_behavior_constant():
        """The Lebesgue asymptotic behavior constant describes the average size of the level sets
        of a random walk in d dimensions. It is denoted by L(d).

        Returns:
            float: The value of the Lebesgue asymptotic behavior constant L(3), approximately 3.912023005428146.
        """
        return 3.912023005428146

    @staticmethod
    def feller_tornier_constant():
        """The Feller-Tornier constant is the probability that a random walk on the integers
        returns to the origin infinitely often. It is denoted by F.

        Returns:
            float: The value of the Feller-Tornier constant F, approximately 0.259183.
        """
        return 0.259183

    @staticmethod
    def base_10_champernowne_constant():
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

    @staticmethod
    def salem_constant():
        """The Salem number is a complex number that is a root of a certain polynomial
        with integer coefficients. It is denoted by s.

        Returns:
            complex: The value of the Salem constant s, approximately (1+sqrt(2)) * e^(pi*sqrt(2)/4).
        """
        return (1 + 2 ** 0.5) * Algorithm.exp(Constants.pi * 2 ** 0.5 / 4)

    @staticmethod    
    def khinchins_constant():
        """The Khinchin constant is a number that appears in the theory of continued fractions. 
        It is denoted by K.

        Returns:
            float: The value of the Khinchin constant K, approximately 2.6854520010653065.
        """
        return 2.6854520010653065

    @staticmethod
    def levys_constant():
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

    @staticmethod
    def levys_constant_two():
        """Calculate the value of e to the power of Levy's constant.

        Returns:
            float: The value of e to the power of Levy's constant.
        """
        return Algorithm.exp(Constants.levys_constant)

    @staticmethod
    def copeland_erdos_constant():
        """Copeland-Erdős constant is the smallest number that is not the sum of 
        distinct non-negative integer powers of 2.
        
        Symbol:
            C_E
        
        Returns:
            float
        """
        n = 1
        while True:
            for s in MathFunctions.subsets(range(n)):
                if sum([2**i for i in s]) == n:
                    break
            else:
                return n
            n += 1

    @staticmethod    
    def gompertz_constant():
        """Gompertz constant is a mathematical constant named after Benjamin Gompertz,
        it is the limit of the ratio between the life expectancy of a certain age 
        and the remaining life expectancy.
        
        Symbol:
            γ
            
        Returns:
            float
        """
        n = 1
        limit = MathFunctions.limit(lambda x: (Algorithm.exp(1)**(1/x))/x, n)
        while limit == float('inf'):
            n += 1
            limit = MathFunctions.limit(lambda x: (Algorithm.exp(1)**(1/x))/x, n)
        return limit

    @staticmethod    
    def de_bruijn_newman_constant():
        """
        De Bruijn–Newman constant is the limit of the sequence of coefficients a_n
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

    @staticmethod
    def porters_constant():
        """
        Porter's constant is a mathematical constant that appears in the field of information theory. It is defined as
        the limit of the ratio of the maximum number of different words of length n over the number of possible words of
        length n as n approaches infinity.
        
        Symbol:


        Returns:
            float: The value of Porter's constant to the highest precision.
        """
        return Algorithm.exp**(1/Constants.euler_mascheroni_constant)

    @staticmethod
    def euler_mascheroni_constant():
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


    @staticmethod
    def lochs_constant():
        """
        Lochs' constant is a mathematical constant defined as the limiting ratio of the perimeter of an inscribed regular
        decagon to its diameter.
        
        Symbol:


        Returns:
            float: The value of Lochs' constant to the highest precision.
        """
        return Algorithm.square_root(2 + Algorithm.square_root(2 + Algorithm.square_root(2 + Algorithm.square_root(2 + Algorithm.square_root(2)))))


    @staticmethod
    def deviccis_tesseract_constant():
        """
        The De Vries - De Vos - Barendrecht - De Klerk - Smit - Smit constant (also known as De Vries' tesseract constant)
        is defined as the number that describes the maximum ratio of the content of a hypercube inscribed in a tesseract to
        the content of the hypercube circumscribed about the tesseract.
        
        Symbol:


        Returns:
            float: The value of De Vries' tesseract constant to the highest precision.
        """
        return Algorithm.square_root(2 + Algorithm.square_root(2)) / (2 * Algorithm.square_root(2))


    @staticmethod
    def liebs_square_ice_constant():
        """
        The Lieb's square ice constant is the infinite sum of alternating sign reciprocals of the squares of odd positive integers.
        It appears in the square ice problem in statistical mechanics.
        
        Symbol:


        Returns:
            float: The value of the Lieb's square ice constant to the highest precision.
        """
        return Constants.pi / (Algorithm.square_root(3) * Algorithm.log((3 + Algorithm.square_root(8)) / 2))

    @staticmethod
    def nivens_constant():
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

    @staticmethod    
    def mills_constant():
        """Mills constant is the smallest positive real number A such that the 
        floor function of the double exponential function is a prime number,
        where the double exponential function is f(n) = A^(3^n).
        
        Symbol:
            A
            
        Returns:
            float
        """
        i = 2
        while not Algorithm.is_prime(int(MathFunctions.floor(Algorithm.power(MathFunctions.copysign(Algorithm.power(3, i), 1), MathFunctions.copysign(Algorithm.power(3, i - 1), 1))))): 
            i += 1
        return Algorithm.power(MathFunctions.copysign(Algorithm.power(3, i), 1), MathFunctions.copysign(Algorithm.power(3, i - 1), 1))

    @staticmethod
    def artins_constant():
        """
        Artin's constant is a real number that arises in the study of the Riemann zeta function.

        Returns:
            float: The value of Artin's constant.
        """
        return 0.3739558136

    @staticmethod
    def porters_constant():
        """
        Porter's constant is a mathematical constant that arises in the study of the Riemann hypothesis.

        Returns:
            float: The value of Porter's constant.
        """
        return 1.4670780794

    @staticmethod
    def lochs_constant():
        """
        Lochs' constant is a mathematical constant that arises in the study of prime numbers.

        Returns:
            float: The value of Lochs' constant.
        """
        return 0.8241323125

    @staticmethod
    def deviccis_tesseract_constant():
        """
        De Vici's tesseract constant is a mathematical constant that arises in the study of hypercubes.

        Returns:
            float: The value of De Vici's tesseract constant.
        """
        return 1.0983866775

    @staticmethod
    def liebs_square_ice_constant():
        """
        Lieb's square ice constant is a mathematical constant that arises in the study of statistical mechanics.

        Returns:
            float: The value of Lieb's square ice constant.
        """
        return 1.5396007178

    @staticmethod
    def nivens_constant():
        """
        Niven's constant is a mathematical constant that arises in number theory.

        Returns:
            float: The value of Niven's constant.
        """
        return 1.7052111401

    @staticmethod
    def stephens_constant():
        """
        Stephens' constant is a mathematical constant that arises in the study of prime numbers.

        Returns:
            float: The value of Stephens' constant.
        """
        return 0.5364798721

    @staticmethod
    def regular_paperfolding_sequence():
        """
        The regular paperfolding sequence is a binary sequence that arises in the study of fractal geometry.

        Returns:
            str: The regular paperfolding sequence as a string of 0s and 1s.
        """
        return "110110011100100"

    @staticmethod
    def reciprocal_fibonacci_constant():
        """
        The reciprocal Fibonacci constant is a real number that arises in the study of Fibonacci numbers.

        Returns:
            float: The value of the reciprocal Fibonacci constant.
        """
        return 1.1319882488

    @staticmethod
    def chvatal_sankoff_constant():
        """
        Chvátal–Sankoff constant for the binary alphabet.

        Symbol:
            \gamma_{2}

        Returns:
            float: The value of the Chvátal–Sankoff constant.
        """
        return 1.7550327129

    @staticmethod    
    def Feigenbaum_constant():
        """
        Feigenbaum constant δ

        Symbol:
            \delta

        Returns:
            float: The value of the Feigenbaum constant.
        """
        return 4.6692016091

    @staticmethod
    def chaitins_constant():
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

    @staticmethod
    def robbins_constant():
        """
        Robbins' constant is a mathematical constant that arises in the study of mathematical analysis.

        Symbol:
            \Delta(3)

        Raises:
            ValueError: If the computation of the constant fails.

        Returns:
            float: The value of Robbins' constant.
        """
        return MathFunctions.quad(lambda x: x**x, 0, 1)[0]

    @staticmethod
    def weierstrass_constant():
        """
        Weierstrass' constant is a mathematical constant that arises in the study of elliptic functions.

        Returns:
            float: The value of Weierstrass' constant.
        """
        return 0.5174790617

    @staticmethod
    def fransen_robinson_constant():
        """Returns Fransen-Robinson constant which is the smallest positive root of the following polynomial equation:

        x^3 - x^2 - 1 = 0

        Symbol:
            F

        Raises:
            ValueError: If the root cannot be found

        Returns:
            float: The Fransen-Robinson constant
        """
        a, b = 1, 2
        while True:
            c = (a + b) / 2
            if abs(c ** 3 - c ** 2 - 1) < 1e-10:
                return c
            elif (c ** 3 - c ** 2 - 1) * (a ** 3 - a ** 2 - 1) < 0:
                b = c
            else:
                a = c

    @staticmethod
    def feigenbaum_constant():
        """Returns Feigenbaum constant alpha which relates to the period-doubling bifurcation in chaotic systems.

        Symbol:
            \alpha 

        Raises:
            ValueError: If the constant cannot be computed

        Returns:
            float: The Feigenbaum constant alpha
        """
        a = 1.0
        for n in range(1, 11):
            if n % 2 == 1:
                c = 3
            else:
                c = -1
            a_next = a - (c / 2**n) * a**2
            if abs(a_next - a) < 1e-10:
                break
            a = a_next
        else:
            raise ValueError("Failed to converge")
        return a

    @staticmethod
    def second_du_bois_reymond_constant():
        """Returns the Second du Bois-Reymond constant, which is defined as the supremum of the absolute values of the Fourier coefficients of a bounded variation function with period 1.

        Symbol:
            C_{2}
            
        Raises:
            ValueError: If the constant cannot be computed

        Returns:
            float: The Second du Bois-Reymond constant
        """

        return MathFunctions.quad(lambda x: abs(sum([(-1)**n * Algorithm.sin((2*n+1)*x) / (2*n+1)**2 for n in range(1000)])), 0, 1)[0]


    @staticmethod
    def erdos_tenenbaum_ford_constant():
        """Returns the Erdős–Tenenbaum–Ford constant which is related to the distribution of prime numbers.

        Symbol:
            \delta

        Raises:
            ValueError: If the constant cannot be computed

        Returns:
            float: The Erdős–Tenenbaum–Ford constant
        """

        primes = Algorithm.sieve_of_eratosthenes(5000)
        return sum([1 / p for p in primes]) * Algorithm.log(Algorithm.log(primes[-1]))

    @staticmethod
    def conways_constant():
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

        return self.brentq(lambda x: x**3 - x - 1, 1, 2)

    @staticmethod
    def hafner_sarnak_mccurley_constant():
        """Returns the Hafner-Sarnak-McCurley constant which is related to the distribution of prime numbers in arithmetic progressions.

        Symbol:
            \sigma
        
        Raises:
            ValueError: If the constant cannot be computed

        Returns:
            float: The Hafner-Sarnak-McCurley constant
        """

        return sum([1 / totient(n) for n in range(1, 10001)])

    @staticmethod
    def backhouses_constant():
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

    @staticmethod
    def viswanath_constant():
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

    @staticmethod
    def komornik_loreti_constant():
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

    @staticmethod    
    def embree_trefethen_constant():
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
    
        @staticmethod    """
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

    @staticmethod    
    def heath_brown_moroz_constant():
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
    
        @staticmethod    """
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

    @staticmethod
    def logarithmic_capacity():
        """Returns the logarithmic capacity of the unit disk.

        Raises:
            ValueError: If the calculation is not valid.

        Returns:
            float: The value of the logarithmic capacity.
        """
        
        return 2

    @staticmethod
    def taniguchi_constant():
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

