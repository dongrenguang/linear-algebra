# coding=utf-8

from math import sqrt, acos, pi
from decimal import Decimal, getcontext

getcontext().prec = 30

class Vector(object):

    CANNOT_NORMALIZE_ZEOR_VECTOR_MSG = 'Cannot normalize the zero vector'

    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple([Decimal(x) for x in coordinates])
            self.dimension = len(self.coordinates)

        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be an iterable')


    def plus(self, v):
        new_coordinates = [x+y for x,y in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)


    def minus(self, v):
        new_coordinates = [x-y for x,y in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)


    def time_scalar(self, c):
        new_coordinates = [Decimal(str(c))*x for x in self.coordinates]
        return Vector(new_coordinates)


    # 长度
    def magnitude(self):
        coordinates_squared = [x**2 for x in self.coordinates]
        return Decimal(sqrt(sum(coordinates_squared)))


    # 单位化（即长度为1）
    def normalized(self):
        try:
            magnitude = self.magnitude()
            return self.time_scalar(Decimal('1.0')/magnitude)
        except ZeroDivisionError:
            raise Exception(CANNOT_NORMALIZE_ZEOR_VECTOR_MSG)


    # 内积（是一个值）
    def dot(self, v):
        s = sum([x*y for x,y in zip(self.coordinates, v.coordinates)])
        if s > 1 and MyDecimal(s - 1).is_near_zero():
            s = Decimal('1.0')
        elif s < -1 and MyDecimal(s + 1).is_near_zero():
            s = Decimal('-1.0')
        return s


    # 夹角
    def angle_with(self, v, in_degrees=False):
        try:
            u1 = self.normalized()
            u2 = v.normalized()
            angle_in_radians = Decimal(acos(u1.dot(u2)))

            if in_degrees:
                return angle_in_radians * (Decimal('180.0') / Decimal(pi))
            else:
                return angle_in_radians

        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZEOR_VECTOR_MSG:
                raise Exception(CANNOT_NORMALIZE_ZEOR_VECTOR_MSG)
            else:
                raise e


    # 是否正交
    def is_orthogonal_to(self, v, tolerance=1e-10):
        return abs(self.dot(v)) < tolerance


    # 是否平行
    def is_parallel_to(self, v, tolerance=1e-6):
        return self.is_zero() or v.is_zero() or self.angle_with(v) < Decimal(tolerance) or abs(self.angle_with(v) - Decimal(pi)) < Decimal(tolerance)


    # 是否是0向量
    def is_zero(self, tolerance=1e-10):
        return self.magnitude() < tolerance


    def component_parallel_to(self, b):
        try:
            u = b.normalized()
            return u.time_scalar(self.dot(u))
        except Exception as e:
            raise e


    def component_orthogonal_to(self, b):
        try:
            return self.minus(self.component_parallel_to(b))
        except Exception as e:
            raise e


    # 向量积（是一个向量）
    def cross(self, v):
        if (len(v.coordinates) > 3):
            return;
        x1 = self.coordinates[0]
        y1 = self.coordinates[1]
        z1 = self.coordinates[2]
        x2 = v.coordinates[0]
        y2 = v.coordinates[1]
        z2 = v.coordinates[2]
        return Vector([y1*z2-y2*z1, x2*z1-x1*z2, x1*y2-x2*y1])


    # 两个向量为边组成的平行四边形的面积
    def area(self, v):
        return self.cross(v).magnitude()


    # 两个向量为边组成的三角形的面积
    def helf_area(self, v):
        return self.area(v) / Decimal('2.0')


    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)


    def __eq__(self, v):
        return self.coordinates == v.coordinates


    def __getitem__(self, i):
        return self.coordinates[i]


    def __setitem__(self, i, x):
        self.coordinates[i] = x




class MyDecimal(Decimal):
    def is_near_zero(self, eps=1e-10):
        return abs(self) < eps


# v = Vector(['8.462', '7.893', '-8.187'])
# w = Vector(['6.984', '-5.975', '4.778'])
# print v.cross(w)
#
# v = Vector(['-8.987', '-9.838', '5.031'])
# w = Vector(['-4.268', '-1.861', '-8.866'])
# print v.area(w)
#
# v = Vector([pi, '9.547', '3.691'])
# w = Vector(['-6.007', '0.124', '5.772'])
# print v.cross(w)
# print v.helf_area(w)
