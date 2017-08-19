# coding=utf-8

from decimal import Decimal, getcontext
from vector import Vector

getcontext().prec = 30


class Line(object):

    NO_NONZERO_ELTS_FOUND_MSG = 'No nonzero elements found'

    def __init__(self, normal_vector=None, constant_term=None):
        self.dimension = 2

        if not normal_vector:
            all_zeros = ['0']*self.dimension
            normal_vector = Vector(all_zeros)
        self.normal_vector = normal_vector

        if not constant_term:
            constant_term = Decimal('0')
        self.constant_term = Decimal(constant_term)

        self.set_basepoint()


    # 基点，即(0, b) or (a, 0)
    def set_basepoint(self):
        try:
            n = self.normal_vector.coordinates
            c = self.constant_term
            basepoint_coords = ['0']*self.dimension

            initial_index = Line.first_nonzero_index(n)
            initial_coefficient = n[initial_index]

            basepoint_coords[initial_index] = c/initial_coefficient
            self.basepoint = Vector(basepoint_coords)

        except Exception as e:
            if str(e) == Line.NO_NONZERO_ELTS_FOUND_MSG:
                self.basepoint = None
            else:
                raise e


    # 是否平行
    def is_parallel_to(self, line2):
        return self.normal_vector.is_parallel_to(line2.normal_vector);


    # 是否是同一条直线
    def is_same(self, line2):
        if self.normal_vector.is_zero():
            if not line2.normal_vector.is_zero():
                return False
            else:
                diff = self.constant_term - line2.constant_term
                return MyDecimal(diff).is_near_zero()
        elif line2.normal_vector.is_zero():
            return False

        if (not self.is_parallel_to(line2)):
            return False;
        # n1 = self.normal_vector.coordinates
        # n2 = line2.normal_vector.coordinates
        # c1 = self.constant_term
        # c2 = line2.constant_term
        # initial_index1 = Line.first_nonzero_index(n1)
        # initial_index2 = Line.first_nonzero_index(n2)
        # initial_coefficient1 = n1[initial_index1]
        # initial_coefficient2 = n2[initial_index2]
        # delta = initial_coefficient1 / initial_coefficient2 - self.constant_term / line2.constant_term
        # return MyDecimal(delta).is_near_zero()

        x0 = self.basepoint
        y0 = line2.basepoint
        basepoint_difference = x0.minus(y0)
        n = self.normal_vector
        return n.is_orthogonal_to(basepoint_difference)


    # 交点
    def intersection_with(self, line2):
        if (self.is_same(line2)):
            print 'They are the same line, so they have countless intersection.'
        elif (self.is_parallel_to(line2)):
            print 'They are parallel, but not the same, so they have no intersection.'
        else:
            A = self.normal_vector.coordinates[0]
            B = self.normal_vector.coordinates[1]
            k1 = self.constant_term
            C = line2.normal_vector.coordinates[0]
            D = line2.normal_vector.coordinates[1]
            k2 = line2.constant_term
            x = (D * k1 - B * k2) / (A * D - B * C)
            y = (-C * k1 + A * k2) / (A * D - B * C)
            return [x, y]


    def __str__(self):

        num_decimal_places = 3

        def write_coefficient(coefficient, is_initial_term=False):
            coefficient = round(coefficient, num_decimal_places)
            if coefficient % 1 == 0:
                coefficient = int(coefficient)

            output = ''

            if coefficient < 0:
                output += '-'
            if coefficient > 0 and not is_initial_term:
                output += '+'

            if not is_initial_term:
                output += ' '

            if abs(coefficient) != 1:
                output += '{}'.format(abs(coefficient))

            return output

        n = self.normal_vector.coordinates

        try:
            initial_index = Line.first_nonzero_index(n)
            terms = [write_coefficient(n[i], is_initial_term=(i==initial_index)) + 'x_{}'.format(i+1)
                     for i in range(self.dimension) if round(n[i], num_decimal_places) != 0]
            output = ' '.join(terms)

        except Exception as e:
            if str(e) == self.NO_NONZERO_ELTS_FOUND_MSG:
                output = '0'
            else:
                raise e

        constant = round(self.constant_term, num_decimal_places)
        if constant % 1 == 0:
            constant = int(constant)
        output += ' = {}'.format(constant)

        return output


    # 第一个不为0的位置
    @staticmethod
    def first_nonzero_index(iterable):
        for k, item in enumerate(iterable):
            if not MyDecimal(item).is_near_zero():
                return k
        raise Exception(Line.NO_NONZERO_ELTS_FOUND_MSG)


class MyDecimal(Decimal):
    def is_near_zero(self, eps=1e-10):
        return abs(self) < eps


# line1 = Line(Vector([4.046, 2.836]), 1.21)
# line2 = Line(Vector([10.115, 7.09]), 3.025)

# line1 = Line(Vector([7.204, 3.182]), 8.68)
# line2 = Line(Vector([8.172, 4.114]), 9.883)

# line1 = Line(Vector([1.182, 5.562]), 6.744)
# line2 = Line(Vector([1.773, 8.343]), 9.525)

line1 = Line(Vector([0, 0]), 1.21)
line2 = Line(Vector([0, 0]), 1.21)

print line1.intersection_with(line2)
print line1.is_parallel_to(line2)
print line1.is_same(line2)
