import copy
import matrix

#Computes the row-wise Hermite Normal Form of a matrix as defined in:
# https://doi.org/10.1145/101104.10110 - "Hermite canonical form and smith canonical form of a matrix over a PID" - Pasqualina Conti

# A matrix J is a row-wise Hermite Normal Form of matrix A if:
# The pivot element of each row is strictly to the left of pivot in the preceding row (leading to upper triangular
# matrices)
# The norm of each element in a column is smaller than the norm of the pivot for that column
# The transformation matrix S, such that S * A = J, is a unimodular matrix (integer matrix with integer matrix inverse)

class HNFRowProblem:

    # A is a matrix over a PID that we want to find the row-wise hermite normal form of.
    def __init__(self, A, debug=False):
        if isinstance(A, HNFRowProblem):
            other = A
            self.A = other.A.copy()
            self.elementT = other.elementT
            self.J = other.J.copy()
            self.S = other.S.copy()
            self.debug = other.debug if debug is False else debug
        else:
            self.A = A.copy()
            self.elementT = type(A.get(0, 0))
            self.J = A.copy()
            self.S = matrix.Matrix.id(A.h, type(A.get(0, 0)))
            self.debug = debug

    def isValid(self):
        #Check whether the current HNFRowProblem is a valid HNF.

        #Conditions:
        #1. S is unimodular.
        #2. A * S  == J.
        #3. J is in row-echelon / upper-triangular form:
            #- entries below the pivot in each row are zero;
            #- pivot positions strictly increase from row to row.
        #4. Every nonzero row has a pivot.
        #5. Zero rows occur only after all nonzero rows.
        #6. Entries above each pivot are reduced modulo the pivot:
        #    norm(entry) < norm(pivot).

        #We do not require the pivots themselves to be canonical
        #representatives of their associate classes."""

        zero = self.elementT.getZero()

        # Check that S has the appropriate dimensions and is
        # unimodular.
        if self.S.h != self.A.h or self.S.w != self.A.h:
            print("S has incorrect dimensions")
            return False

        if not self.S.determinant().isUnit():
            print("S not unimodular")
            return False

        #  # 2. Check S * A == J.
        if self.S * self.A != self.J:
            print("S * A != J")
            return False
        
        # 3. Find the pivot position of every row.
        pivot_positions = []
        for i in range(self.J.h):
            pivot = None
            for j in range(self.J.w):
                if self.J.get(i, j) != zero:
                    pivot = j
                    break
            pivot_positions.append(pivot)

        # 4. Check pivot ordering and zero-row placement.
        previous_pivot = None
        zero_row_seen = False
        for i, pivot in enumerate(pivot_positions):
            if pivot is None:
                zero_row_seen = True
                continue
            if zero_row_seen:
                print("Nonzero row occurs below a zero row")
                return False
            if previous_pivot is not None and pivot <= previous_pivot:
                print(
                    "Pivot in row", i,
                    "is not strictly to the right of the pivot in the previous row"
                )
                return False
            previous_pivot = pivot

        # 5. Check the upper-triangular / row-echelon condition.
        for i, pivot in enumerate(pivot_positions):
            if pivot is None:
                continue
            for j in range(pivot):
                if self.J.get(i, j) != zero:
                    print(
                        "Nonzero entry before pivot in row", i,
                        "at column", j
                    )
                    return False

        # 6. Check the reducedness condition.
        # For each pivot J[i, pivot], every entry above that pivot
        # in the same column must have strictly smaller norm.
        
        for i, pivot in enumerate(pivot_positions):
            if pivot is None:
                continue
            pivot_element = self.J.get(i, pivot)
            pivot_norm = pivot_element.norm()
            for r in range(i):
                entry = self.J.get(r, pivot)
                if entry != zero and not (entry.norm() < pivot_norm):
                    print(
                        "Entry above pivot too large at",
                        "row", r,
                        "column", pivot
                    )
                    return False
        return True
    
    #Perform a "row-swap". Here we modify the matrix J by swapping the rows
    # of index i and j. We adjust the matrix S to make sure the overall
    # relation of S * A = J continues to hold.
    def rSwap(self, i, j):
        if self.debug:
            print("r_swap call")

        if i == j:
            return

        # perform the row swap to self.J
        for k in range(self.J.w):
            temp = self.J.get(i, k)
            self.J.set(i, k, self.J.get(j, k))
            self.J.set(j, k, temp)

        # adjust the S matrix
        adjustment = matrix.Matrix.id(self.S.h, self.elementT)
        adjustment.set(i, j, self.elementT.getOne())
        adjustment.set(j, i, self.elementT.getOne())
        adjustment.set(i, i, self.elementT.getZero())
        adjustment.set(j, j, self.elementT.getZero())
        self.S = adjustment * self.S


    # Perform a "row-wise linear combination" operation. Here we set the k
    # row of the matrix J to be a * the i row plus b times the j row. We
    # update the S matrix to ensure the relationship S*A = J continues to
    # hold.
    def rLC(self, k, i, j, a, b, gcd=None):
        if self.debug:
            print("r_lc call")

        if gcd is None:
            c = self.elementT.getZero()
            d = self.elementT.getOne()
        else:
            c = -self.J.get(j, k) // gcd
            d = self.J.get(i, k) // gcd

        # perform the linear column application to self.J
        for k in range(self.J.w):
            temp = self.J.get(i, k)
            self.J.set(i, k, a * self.J.get(i, k) + b * self.J.get(j, k))
            self.J.set(j, k, c * temp + d * self.J.get(j, k))

        # adjust the self.S matrix
        adjustment = matrix.Matrix.id(self.S.h, self.elementT)
        adjustment.set(i, i, a)
        if i != j:
            adjustment.set(i, j, b)
            adjustment.set(j, i, c)
            adjustment.set(j, j, d)
        self.S = adjustment * self.S

    def rScale(self, i, q): #q must be a unit
        adjustment = matrix.Matrix.id(self.S.h, self.elementT)
        adjustment.set(i, i, q)
        for k in range(self.J.w):
            self.J.set(i, k, q * self.J.get(i, k))
        self.S = adjustment * self.S

    
    def computeRowHNF(self):
        zero = self.elementT.getZero()
        one = self.elementT.getOne()
        pivot_row = 0

        # Process columns from left to right.

        for col in range(self.J.w):
            if pivot_row >= self.J.h:
                break

            # Find a nonzero entry in this column at or below pivot_row.
            pivot = None

            for r in range(pivot_row, self.J.h): #if it finds a non-zero element when going down the column, increase the index pivot
                if self.J.get(r, col) != zero:
                    pivot = r
                    break

            if pivot is None:
                continue

            # Move the pivot into position.
            if pivot != pivot_row:
                self.rSwap(pivot_row, pivot)

            pivot_value = self.J.get(pivot_row, col)
            for r in range(pivot_row + 1, self.J.h):
                entry = self.J.get(r, col)

                if entry == zero:
                    continue

                gcd, x, y = pivot_value.extended_gcd(entry)

                # if entry already divisible by pivot_value, add multiple of pivot row to zero out
                if entry % pivot_value == zero:
                    q = entry // pivot_value
                    self.rLC(col,r,pivot_row,one,-q)

                # If the current pivot is not a multiple of pivot, 
                # use bezout coeffs and gcd to make pivot = gcd, and entry =0
                else:
                    self.rLC(col,pivot_row,r,x,y,gcd)

                pivot_value = self.J.get(pivot_row, col)
                
            # Reduce entries above the pivot.
            # Since the pivot is in column `col`, only rows above
            # `pivot_row` can have entries here.

            for r in range(pivot_row):
                entry = self.J.get(r, col)
                if entry == zero:
                    continue
                q = entry // pivot_value
                self.rLC(col,r,pivot_row,one,-q)

            # This row now has a valid pivot. Move to the next row.

            pivot_row += 1


    def rowEquivalent(self,other): #if not hnf equivalent, the transformation will fail and return False, if
        #it succeeds until the final check, it returns True as the transformation by pre-multiplying unimodular
        #is possible
        zero = self.elementT.getZero()
        one = self.elementT.getOne()
        working = HNFRowProblem(self) #use copies so as to not change the original
        s2 = HNFRowProblem(other)

        #if two matrices have same HNF, their diagonal elements will be the same up to multiplication by a unit
        #divide them to find this unit, the multiply the row by that unit so the diagonal element is exactly the same

        for i in range(working.J.h-1,-1,-1):
            q2 = working.J.get(i,i).get_q(s2.J.get(i,i))
            if (q2*s2.J.get(i,i)!=working.J.get(i,i)):
                return False
            s2.rScale(i, q2)

        #apply row operations of lower rows to transform s2 row into s1 row using allowed operations
        for j in range(1,working.J.h):
            for i in range(0,j):
                diff = working.J.get(i,j) - s2.J.get(i,j)
                if diff == zero:
                    continue
                q1 = diff.get_q(s2.J.get(j,j))
                if (q1*s2.J.get(j,j)!=diff):
                    return False
                s2.rLC(i,i,j,one,q1)

        #if a unimodular transformation from 'working' to 's2' is possible, they will be exactly the same
        if working.J != s2.J:
            return False

        #all checks correct
        return True


    