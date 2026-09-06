import ed, pid
import sympy as sp
import itertools


# This represents the ring of quadratic integers.
# Every integer in the ring can be written as z = {a + b*sqrt2 : a,b are integers}
# Coeffs correspond to [a,b]
# This ring is a Euclidean Domain and its norm is taken as defined in:
# https://math.dartmouth.edu/~lmcbeath/m71f23_110723.pdf

class ZSQRT2(ed.ED):
    dim = 2 #we require 2 coefficients to define an integer in the ring

    def __init__(self, content):
        if isinstance(content, ZSQRT2): #make a copy
            self.a = content.a
            self.b = content.b
            
        elif isinstance(content, list): #define zqsrt2 integer from list of two integers
            if len(content) != 2:
                raise pid.InvalidInitialContent
            if not (isinstance(content[0], int) and
                    isinstance(content[1], int)):
                print(type(content[0]))
                print(type(content[1]))

                raise pid.InvalidInitialContent
            else:
                self.a = content[0]
                self.b = content[1]

        else:
            raise pid.InvalidInitialContent

        self.coeffs = sp.Matrix([self.a, self.b])
        self.matrix = sp.Matrix([
                    [self.a, 2*self.b],[self.b,self.a],
                ])

    def __str__(self):
        """
        Pretty-print the real dyadic /sqrt integer in LaTeX form.
        """
        basis = ["", r"\sqrt{2}"]
        parts = []

        for coeff, name in zip(self.coeffs, basis):
            if coeff == 0:
                continue

            if name == "":
                parts.append(str(coeff))
            else:
                if coeff == 1:
                 parts.append(name)
                elif coeff == -1:
                    parts.append(f"-{name}")
                else:
                    parts.append(f"{coeff}{name}")

        if not parts:
            return "0"

        out = " + ".join(parts)
        out = out.replace("+ -", "- ")
        return f"${out}$"

    def __hash__(self):
        return hash(("com.ktjkeiller.matrixnormalforms.zsqrt2", self.a, self.b))

    def __eq__(self, y):
        return self.a == y.a and self.b == y.b

    def __ne__(self, y):
        return not self == y

    def __neg__(self):
        return ZSQRT2([-self.a, -self.b])

    def __add__(self, y):
        return ZSQRT2([self.a + y.a, self.b + y.b])

    def __sub__(self, y):
        return ZSQRT2([self.a - y.a, self.b - y.b])

    def __mul__(self, y):
        out = self.matrix * y.coeffs
        valid = [int(x) for x in out]
        return ZSQRT2(valid)

    # self = r + q * y, minimising norm of r

    def get_q(self, y):

        z = y.matrix.LUsolve(self.coeffs)
        best_q = None
        min_norm = float('inf')
        options = [[int(sp.floor(c)), int(sp.ceiling(c))] for c in z]

        for cand in itertools.product(*options):
            cand = [int(x) for x in cand]
            q_cand = ZSQRT2([cand[0],cand[1]])
            prod = q_cand*y
            r_cand = self - prod
            if r_cand.norm() < min_norm:
                min_norm = r_cand.norm()
                best_q = q_cand

        return best_q

    # get the remainder of division
    def get_r(self, y):
        q = self.get_q(y)
        return self - q*y


    # Returns the norm of this zsqrt2 object
    def norm(self):
        a, b = [int(x) for x in self.coeffs]
        x = a*a - 2*b*b
        return abs(x)

    # return the additive identity of the ring
    @staticmethod
    def getZero():
        return ZSQRT2([0, 0])

    # return the multiplicative identity of the ring
    @staticmethod
    def getOne():
        return ZSQRT2([1, 0])

    def isUnit(self):
        return self.norm() ==1
    
    def isEquivalent(self,y): #check if two elements are equivalent up to multiplication by a unit
        res = self.matrix.LUsolve(y.coeffs)
        x = [int(i) for i in res]

        return (type(self)(x).isUnit())