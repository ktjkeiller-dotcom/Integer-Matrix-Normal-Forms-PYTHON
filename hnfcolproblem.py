import matrix

#Computes the Hermite Normal Form of a matrix as defined in:
# https://doi.org/10.1145/101104.10110 - "Hermite canonical form and smith canonical form of a matrix over a PID" - Pasqualina Conti

# A matrix J is a column-wise Hermite Normal Form of matrix A if:
# The pivot element of each column is strictly below of pivot in the preceding column (leading to lower triangular
# matrices)
# The norm of each element in a row is smaller than the norm of the pivot for that row
# The transformation matrix S, such that  A * T = J, is a unimodular matrix (integer matrix with integer matrix inverse)

class HNFColProblem:

    # A is a matrix over a PID that we want to find the hermite normal form of.
    def __init__(self, A, debug=False):
        if isinstance(A, HNFColProblem):
            other = A
            self.A = other.A.copy()
            self.elementT = other.elementT
            self.J = other.J.copy()
            self.T = other.T.copy()
            self.debug = other.debug if debug is False else debug
        else:
            self.A = A.copy()
            self.elementT = type(A.get(0, 0))
            self.J = A.copy()
            self.T = matrix.Matrix.id(A.w, self.elementT)
            self.debug = debug

    def isValid(self):
        #Check whether the current HNFColProblem is a valid HNF.

        #Conditions:
        #1. S is unimodular.
        #2. A * T == J.
        #3. J is in row-echelon / lower-triangular form:
            #- entries to the right of the pivot in each row are zero;
            #- pivot positions in each column strictly below the pivot of the previous column.
        #4. Every nonzero column has a pivot.
        #5. Zero columns occur only after all nonzero columns.
        #6. Entries to the left of each pivot are reduced modulo the pivot:
        #    norm(entry) < norm(pivot).

        #We do not require the pivots themselves to be canonical
        #representatives of their associate classes."""

        zero = self.elementT.getZero()

        # 1. Check that S has the appropriate dimensions and is
        # unimodular.
        if self.T.h != self.A.h or self.T.w != self.A.h:
            print("T has incorrect dimensions")
            return False
        if not self.T.determinant().isUnit():
            print("T not unimodular")
            return False

        # 2. check multiplication identity
        if self.A * self.T != self.J:
            print("A * T != J")
            return False

        # 3. find the positions of the pivots in each column
        pivot_positions = []
        for col in range(self.J.w):
            pivot = None
            for row in range(self.J.h):
                if self.J.get(row, col) != zero:
                    pivot = row
                    break
            pivot_positions.append(pivot)

        # check the pivot in each column is strictly below the pivot in the previous column, 
        # and that all zero columns appear after all non-zero columns
        previous_pivot = None
        zero_col_seen = False
        for col, pivot in enumerate(pivot_positions):
            if pivot is None:
                zero_col_seen = True
                continue
            if zero_col_seen:
                print("Nonzero column occurs after a zero column")
                return False
            if previous_pivot is not None and pivot <= previous_pivot:
                print("Pivot rows are not strictly increasing")
                return False
            previous_pivot = pivot

            pivot_norm = self.J.get(pivot, col).norm()
            for left_col in range(col):
                entry = self.J.get(pivot, left_col)
                if entry != zero and not (entry.norm() < pivot_norm):
                    print("Entry to the left of pivot too large")
                    return False

        for col, pivot in enumerate(pivot_positions):
            if pivot is None:
                continue
            for row in range(pivot):
                if self.J.get(row, col) != zero:
                    print("Nonzero entry above pivot")
                    return False

        return True
    
    # swap columns i and j
    def cSwap(self, i, j):
        if self.debug:
            print("cSwap call")
        if i == j:
            return

        for row in range(self.J.h):
            temp = self.J.get(row, i)
            self.J.set(row, i, self.J.get(row, j))
            self.J.set(row, j, temp)

        adjustment = matrix.Matrix.id(self.T.h, self.elementT)
        adjustment.set(i, i, self.elementT.getZero())
        adjustment.set(j, j, self.elementT.getZero())
        adjustment.set(i, j, self.elementT.getOne())
        adjustment.set(j, i, self.elementT.getOne())
        self.T = self.T * adjustment

    # Perform a "column-wise linear combination" operation. Here we set the k
    # column of the matrix J to be a * the i column plus b times the j column. We
    # update the T matrix to ensure the relationship A*T = J continues to
    # hold.
    def cLC(self, row, i, j, a, b, gcd=None):
        if self.debug:
            print("cLC call")

        if gcd is None:
            c = self.elementT.getZero()
            d = self.elementT.getOne()
        else:
            c = -self.J.get(row, j) // gcd
            d = self.J.get(row, i) // gcd

        for current_row in range(self.J.h):
            old_i = self.J.get(current_row, i)
            self.J.set(current_row, i, a * old_i + b * self.J.get(current_row, j))
            self.J.set(current_row, j, c * old_i + d * self.J.get(current_row, j))

        adjustment = matrix.Matrix.id(self.T.h, self.elementT)
        adjustment.set(i, i, a)
        if i != j:
            adjustment.set(j, i, b)
            adjustment.set(i, j, c)
            adjustment.set(j, j, d)
        self.T = self.T * adjustment

    #  multiply a column by a unit
    def cScale(self, col, q):
        adjustment = matrix.Matrix.id(self.T.h, self.elementT)
        adjustment.set(col, col, q)
        for row in range(self.J.h):
            self.J.set(row, col, q * self.J.get(row, col))
        self.T = self.T * adjustment

    def computeColHNF(self):
        zero = self.elementT.getZero()
        one = self.elementT.getOne()
        pivot_col = 0

        # This is the transpose of the row algorithm: process rows top-down,
        # using right multiplication to clear entries in each row.
        for row in range(self.J.h):
            if pivot_col >= self.J.w:
                break

            pivot = None
            for col in range(pivot_col, self.J.w):
                if self.J.get(row, col) != zero:
                    pivot = col
                    break
            if pivot is None:
                continue

            if pivot != pivot_col:
                self.cSwap(pivot_col, pivot)

            for col in range(pivot_col + 1, self.J.w):
                pivot_value = self.J.get(row, pivot_col)
                entry = self.J.get(row, col)

                if entry == zero:
                    continue

                gcd, x, y = pivot_value.extended_gcd(entry)
                
                # if entry already divisible by pivot_value
                if entry % pivot_value == zero:
                    q = entry // pivot_value
                    self.cLC(row, col, pivot_col, one, -q)

                # If the current pivot is not divisible by pivot
                # replace the two rows by a Bezout transformation.
                else:
                    self.cLC(row,pivot_col,col,x,y,gcd)

            pivot_value = self.J.get(row, pivot_col)

            # use pivot to reduce elements to right of pivot

            for col in range(pivot_col):
                entry = self.J.get(row, col)
                if entry == zero:
                    continue
                q = entry // pivot_value
                self.cLC(row, col, pivot_col, one, -q)

            pivot_col += 1

    def colEquivalent(self, other):
        zero = self.elementT.getZero()
        one = self.elementT.getOne()
        working = HNFColProblem(self)
        other_working = HNFColProblem(other)

        for col in range(working.J.w - 1, -1, -1):
            q = working.J.get(col, col).get_q(other_working.J.get(col, col))
            if q * other_working.J.get(col, col) != working.J.get(col, col):
                return False
            other_working.cScale(col, q)

        for row in range(1, working.J.h):
            for col in range(row):
                diff = working.J.get(row, col) - other_working.J.get(row, col)
                if diff == zero:
                    continue
                q = diff.get_q(other_working.J.get(row, row))
                if q * other_working.J.get(row, row) != diff:
                    return False
                other_working.cLC(row, col, row, one, q)

        return working.J == other_working.J
