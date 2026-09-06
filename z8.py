import ed, pid
import sympy as sp
import itertools

# This represents the 8th cyclotomic ring of integers. 
# w is defined as the eighth root of unity
# Every integer in the ring can be written as z = {a + bw + cw^2 + dw^3 : a,b,c,d are integers}
# coeffs correspond to [a,b,c,d]
# In this ring, the standard euclidean algorithm (finding r,q such that b = aq + r) does not guarantee 
# that norm(r) < norm (b) but we can check integers in a radius to find an r that satisfies this condition.
# Norm and Euclidean algorithm for this ring as defined in:
# https://www.researchgate.net/publication/265424141_Cyclotomic_rings_with_simple_Euclidean_algorithm

class Z8(ed.ED):
    dim = 4

    def __init__(self, content):
        if isinstance(content, Z8):
            self.a = content.a
            self.b = content.b
            self.c = content.c
            self.d = content.d

        elif isinstance(content, list):
            if len(content) != 4:
                raise pid.InvalidInitialContent
            if not (isinstance(content[0], int) and
                    isinstance(content[1], int) and
                    isinstance(content[2], int) and
                    isinstance(content[3], int)):
                print(type(content[0]))
                print(type(content[1]))
                print(type(content[2]))
                print(type(content[3]))
                raise pid.InvalidInitialContent
            else:
                self.a = content[0]
                self.b = content[1]
                self.c = content[2]
                self.d = content[3]

        else:
            raise pid.InvalidInitialContent

        self.coeffs = sp.Matrix([self.a, self.b, self.c, self.d])
        self.matrix = sp.Matrix([
                    [self.a, -self.d, -self.c, -self.b],
                    [self.b, self.a, -self.d, -self.c],
                    [self.c, self.b, self.a, -self.d],
                    [self.d, self.c, self.b, self.a]
                ])

    def __str__(self):
        
        basis = ["", r"\omega", r"\omega^2", r"\omega^3"]
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
        return hash(("com.ktjkeiller.matrixnormalforms.z8", self.a, self.b, self.c, self.d))

    def __eq__(self, y):
        return self.a == y.a and self.b == y.b and self.c == y.c and self.d == y.d

    def __ne__(self, y):
        return not self == y

    def __neg__(self):
        return Z8([-self.a, -self.b, -self.c, -self.d])

    def __add__(self, y):
        return Z8([self.a + y.a, self.b + y.b, self.c + y.c, self.d + y.d])

    def __sub__(self, y):
        return Z8([self.a - y.a, self.b - y.b, self.c-y.c, self.d-y.d])

    def __mul__(self, y):
        out = self.matrix * y.coeffs
        valid = [int(x) for x in out]
        return Z8(valid)

    #return complex conjugate
    def com(self):
        return Z8([self.a, -self.d, -self.c, -self.b])

    #return the product of this with the complex conjugate of y
    def num(self, y):
        return self * y.com()

    #For information on the norm and euclidean algorithm in this ring:  
    #https://www.researchgate.net/publication/265424141_Cyclotomic_rings_with_simple_Euclidean_algorithm

    #self = r + y * q

    def get_q(self, y): 

        z = y.matrix.LUsolve(self.coeffs)
        best_q = None
        min_norm = float('inf')
        options = [[int(sp.floor(c)), int(sp.ceiling(c))] for c in z]

        for cand in itertools.product(*options):
            cand = [int(x) for x in cand]
            q_cand = Z8([cand[0],cand[1],cand[2],cand[3]])
            prod = q_cand*y
            r_cand = self - prod
            if r_cand.norm() < min_norm:
                min_norm = r_cand.norm()
                best_q = q_cand

        return best_q

    def get_r(self, y):
        q = self.get_q(y)
        
        return self - q*y


    #returns the norm of this z8 object
    def norm(self):
        a, b, c, d = [int(x) for x in self.coeffs]
        x2 = Z8([a,d,-c,b])
        x3 = Z8([a,-b,c,-d])
        x4 = Z8([a,-d,-c,-b])
        return (self*x2*x3*x4).a

    # return the additive identity of the ring
    @staticmethod
    def getZero():
        return Z8([0,0,0,0])

    # return the multiplicative identity of the ring
    @staticmethod
    def getOne():
        return Z8([1, 0,0,0])

    def isUnit(self):
        return self.norm() == 1

    def isEquivalent(self,y): #check if two elements are equivalent up to multiplication by a unit
        res = self.matrix.LUsolve(y.coeffs)
        x = [int(i) for i in res]

        return (type(self)(x).isUnit())