# coding=utf-8

from decimal import Decimal, getcontext
from copy import deepcopy

from vector import Vector
from plane import Plane

getcontext().prec = 30


class LinearSystem(object):

    ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG = 'All planes in the system should live in the same dimension'
    NO_SOLUTIONS_MSG = 'No solutions'
    INF_SOLUTIONS_MSG = 'Infinitely many solutions'

    def __init__(self, planes):
        try:
            d = planes[0].dimension
            for p in planes:
                assert p.dimension == d

            self.planes = planes
            self.dimension = d

        except AssertionError:
            raise Exception(self.ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG)


    def swap_rows(self, row1, row2):
        self[row1], self[row2] = self[row2], self[row1]


    def multiply_coefficient_and_row(self, coefficient, row):
        self[row] = self[row].time_scalar(coefficient)


    def add_multiple_times_row_to_row(self, coefficient, row_to_add, row_to_be_added_to):
        new_row = self[row_to_be_added_to].plus(self[row_to_add].time_scalar(coefficient))
        self[row_to_be_added_to] = new_row


    # 转化方程式为倒三角形矩阵
    def compute_triangular_form(self):
        system = deepcopy(self)
        l = len(system)
        for i in range(0, l - 1):
            indices = system.indices_of_first_nonzero_terms_in_each_row()
            # 交换
            if indices[i] != 0:
                first = -1
                for j in range(i + 1, l):
                    if indices[j] == 0:
                        first = j
                        break
                if first != -1:
                    system.swap_rows(i, first)
            # 处理i行以下的plane
            for k in range(i + 1, l):
                if indices[k] == i:
                    times = -system[k].normal_vector[i] / system[i].normal_vector[i]
                    system.add_multiple_times_row_to_row(times, i, k)

        return system


    # 计算最简形式的倒三角矩阵
    def compute_rref(self):
        tf = self.compute_triangular_form()
        num_equations = len(tf)
        pivot_indices = tf.indices_of_first_nonzero_terms_in_each_row()
        for i in range(num_equations)[::-1]:
            j = pivot_indices[i]
            if j < 0:
                continue
            tf.scale_row_to_make_coefficient_equal_one(i, j)
            tf.clear_coefficients_above(i, j)

        return tf

        ''' 我自己的方法
        tf = self.compute_triangular_form()
        l = len(tf)
        dimension = self.dimension
        i = l - 1
        min = l
        if dimension < min:
            min = dimension

        while i >= 0:
            # 归一化
            z = tf[i].normal_vector[indices[i]]
            if not MyDecimal(z).is_near_zero():
                times = Decimal('1.0') / z
                tf[i] = tf[i].time_scalar(times)
            # 去砸项
            if i < dimension - 1:
                for j in range(i + 1, min):
                    b = tf[i].normal_vector[j]
                    a = tf[j].normal_vector[j]
                    if not MyDecimal(a).is_near_zero():
                        times2 = - b / a
                        tf.add_multiple_times_row_to_row(times2, j, i)

            i -= 1

        return tf
        '''


    def scale_row_to_make_coefficient_equal_one(self, row, col):
        n = self[row].normal_vector
        beta = Decimal('1.0') / n[col]
        self.multiply_coefficient_and_row(beta, row)


    def clear_coefficients_above(self, row, col):
        for k in range(row)[::-1]:
            n = self[k].normal_vector
            alpha = -n[col]
            self.add_multiple_times_row_to_row(alpha, row, k)


    def do_gaussian_elimination_and_extract_solution(self):
        rref = self.compute_rref()
        rref.raise_exception_if_contradictory_equation()
        rref.raise_exception_if_too_few_pivots()

        num_variables = self.dimension
        solution_coordinates = [rref.planes[i].constant_term for i in range(num_variables)]
        return Vector(solution_coordinates)


    def raise_exception_if_contradictory_equation(self):
        for p in self.planes:
            try:
                p.first_nonzero_index(p.normal_vector)

            except Exception as e:
                if str(e) == 'No nonzero elements found':
                    constant_term = MyDecimal(p.constant_term)
                    if not constant_term.is_near_zero():
                        raise Exception(self.NO_SOLUTIONS_MSG)

                else:
                    raise e


    def raise_exception_if_too_few_pivots(self):
        pivot_indices = self.indices_of_first_nonzero_terms_in_each_row()
        num_pivots = sum([1 if index >= 0 else 0 for index in pivot_indices])
        num_variables = self.dimension
        if num_pivots < num_variables:
            raise Exception(self.INF_SOLUTIONS_MSG)


    def indices_of_first_nonzero_terms_in_each_row(self):
        num_equations = len(self)
        num_variables = self.dimension

        indices = [-1] * num_equations

        for i,p in enumerate(self.planes):
            try:
                indices[i] = p.first_nonzero_index(p.normal_vector)
            except Exception as e:
                if str(e) == Plane.NO_NONZERO_ELTS_FOUND_MSG:
                    continue
                else:
                    raise e

        return indices


    def __len__(self):
        return len(self.planes)


    def __getitem__(self, i):
        return self.planes[i]


    def __setitem__(self, i, x):
        try:
            assert x.dimension == self.dimension
            self.planes[i] = x

        except AssertionError:
            raise Exception(self.ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG)


    def __str__(self):
        ret = 'Linear System:\n'
        temp = ['Equation {}: {}'.format(i+1,p) for i,p in enumerate(self.planes)]
        ret += '\n'.join(temp)
        return ret


class MyDecimal(Decimal):
    def is_near_zero(self, eps=1e-10):
        return abs(self) < eps


class Parametrization(object):
    BASEPT_AND_DIR_VECTORS_MUST_BE_IN_SAME_DIM_MSG = 'The basepoint and direction vectors should all live in the same dimension'

    def __init__(self, basepoint, direction_vectors):
        self.basepoint = basepoint
        self.direction_vectors = direction_vectors
        self.dimension = self.basepoint.dimension

        try:
            for v in direction_vectors:
                assert v.dimension == self.dimension

        except AssertionError:
            raise Exception(BASEPT_AND_DIR_VECTORS_MUST_BE_IN_SAME_DIM_MSG)


# p0 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
# p1 = Plane(normal_vector=Vector(['0','1','0']), constant_term='2')
# p2 = Plane(normal_vector=Vector(['1','1','-1']), constant_term='3')
# p3 = Plane(normal_vector=Vector(['1','0','-2']), constant_term='2')
#
# s = LinearSystem([p0,p1,p2,p3])

# print s.indices_of_first_nonzero_terms_in_each_row()
# print '{},{},{},{}'.format(s[0],s[1],s[2],s[3])
# print len(s)
# print s

# s.swap_rows(0, 3)
# s.multiply_coefficient_and_row(2, 3)
# s.add_multiple_times_row_to_row(1, 3, 2)
# print s

# system = s.compute_triangular_form()
# print system




# p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
# p2 = Plane(normal_vector=Vector(['0','1','1']), constant_term='2')
# s = LinearSystem([p1,p2])
# r = s.compute_rref()
# if not (r[0] == Plane(normal_vector=Vector(['1','0','0']), constant_term='-1') and
#         r[1] == p2):
#     print 'test case 1 failed'
# else:
#     print 'Pass case 1'
#
# p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
# p2 = Plane(normal_vector=Vector(['1','1','1']), constant_term='2')
# s = LinearSystem([p1,p2])
# r = s.compute_rref()
# if not (r[0] == p1 and
#         r[1] == Plane(constant_term='1')):
#     print 'test case 2 failed'
# else:
#     print 'Pass case 2'
#
# p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
# p2 = Plane(normal_vector=Vector(['0','1','0']), constant_term='2')
# p3 = Plane(normal_vector=Vector(['1','1','-1']), constant_term='3')
# p4 = Plane(normal_vector=Vector(['1','0','-2']), constant_term='2')
# s = LinearSystem([p1,p2,p3,p4])
# r = s.compute_rref()
# if not (r[0] == Plane(normal_vector=Vector(['1','0','0']), constant_term='0') and
#         r[1] == p2 and
#         r[2] == Plane(normal_vector=Vector(['0','0','-2']), constant_term='2') and
#         r[3] == Plane()):
#     print 'test case 3 failed'
# else:
#     print 'Pass case 3'
#
#
# p1 = Plane(normal_vector=Vector(['0','1','1']), constant_term='1')
# p2 = Plane(normal_vector=Vector(['1','-1','1']), constant_term='2')
# p3 = Plane(normal_vector=Vector(['1','2','-5']), constant_term='3')
# s = LinearSystem([p1,p2,p3])
# r = s.compute_rref()
# if not (r[0] == Plane(normal_vector=Vector(['1','0','0']), constant_term=Decimal('23')/Decimal('9')) and
#         r[1] == Plane(normal_vector=Vector(['0','1','0']), constant_term=Decimal('7')/Decimal('9')) and
#         r[2] == Plane(normal_vector=Vector(['0','0','1']), constant_term=Decimal('2')/Decimal('9'))):
#     print 'test case 4 failed'
# else:
#     print 'Pass case 4'




# p0 = Plane(normal_vector=Vector(['5.862',' 1.178','-10.366']), constant_term='-8.15')
# p1 = Plane(normal_vector=Vector(['-2.931','-0.589','5.183']), constant_term='-4.075')

# p0 = Plane(normal_vector=Vector(['8.631','5.112','-1.816']), constant_term='-5.113')
# p1 = Plane(normal_vector=Vector(['4.315','11.132','-5.27']), constant_term='-6.775')
# p2 = Plane(normal_vector=Vector(['-2.158','3.01','-1.727']), constant_term='-0.831')

# p0 = Plane(normal_vector=Vector(['5.262','2.739','-9.878']), constant_term='-3.441')
# p1 = Plane(normal_vector=Vector(['5.111','6.358','7.638']), constant_term='-2.152')
# p2 = Plane(normal_vector=Vector(['2.016','-9.924','-1.367']), constant_term='-9.278')
# p3 = Plane(normal_vector=Vector(['2.167','-13.593','-18.883']), constant_term='-10.567')


# p0 = Plane(Vector(['0.786', '0.786', '0.588']), '-0.714')
# p1 = Plane(Vector(['-0.138', '-0.138', '0.244']), '0.319')

p0 = Plane(Vector(['8.631', '5.112', '-1.816']), '-5.113')
p1 = Plane(Vector(['4.315', '11.132', '-5.27']), '-6.775')
p2 = Plane(Vector(['-2.158', '3.01', '-1.727']), '-0.831')


#p1 = Plane(Vector([]), )# coding=utf-8

from decimal import Decimal, getcontext
from copy import deepcopy

from vector import Vector
from plane import Plane

getcontext().prec = 30


class LinearSystem(object):

    ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG = 'All planes in the system should live in the same dimension'
    NO_SOLUTIONS_MSG = 'No solutions'
    INF_SOLUTIONS_MSG = 'Infinitely many solutions'

    def __init__(self, planes):
        try:
            d = planes[0].dimension
            for p in planes:
                assert p.dimension == d

            self.planes = planes
            self.dimension = d

        except AssertionError:
            raise Exception(self.ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG)


    def swap_rows(self, row1, row2):
        self[row1], self[row2] = self[row2], self[row1]


    def multiply_coefficient_and_row(self, coefficient, row):
        self[row] = self[row].time_scalar(coefficient)


    def add_multiple_times_row_to_row(self, coefficient, row_to_add, row_to_be_added_to):
        new_row = self[row_to_be_added_to].plus(self[row_to_add].time_scalar(coefficient))
        self[row_to_be_added_to] = new_row


    # 转化方程式为倒三角形矩阵
    def compute_triangular_form(self):
        system = deepcopy(self)
        l = len(system)
        for i in range(0, l - 1):
            indices = system.indices_of_first_nonzero_terms_in_each_row()
            # 交换
            if indices[i] != 0:
                first = -1
                for j in range(i + 1, l):
                    if indices[j] == 0:
                        first = j
                        break
                if first != -1:
                    system.swap_rows(i, first)
            # 处理i行以下的plane
            for k in range(i + 1, l):
                if indices[k] == i:
                    times = -system[k].normal_vector[i] / system[i].normal_vector[i]
                    system.add_multiple_times_row_to_row(times, i, k)

        return system


    # 计算最简形式的倒三角矩阵
    def compute_rref(self):
        tf = self.compute_triangular_form()
        num_equations = len(tf)
        pivot_indices = tf.indices_of_first_nonzero_terms_in_each_row()
        for i in range(num_equations)[::-1]:
            j = pivot_indices[i]
            if j < 0:
                continue
            tf.scale_row_to_make_coefficient_equal_one(i, j)
            tf.clear_coefficients_above(i, j)

        return tf

        ''' 我自己的方法
        tf = self.compute_triangular_form()
        l = len(tf)
        dimension = self.dimension
        i = l - 1
        min = l
        if dimension < min:
            min = dimension

        while i >= 0:
            # 归一化
            z = tf[i].normal_vector[indices[i]]
            if not MyDecimal(z).is_near_zero():
                times = Decimal('1.0') / z
                tf[i] = tf[i].time_scalar(times)
            # 去砸项
            if i < dimension - 1:
                for j in range(i + 1, min):
                    b = tf[i].normal_vector[j]
                    a = tf[j].normal_vector[j]
                    if not MyDecimal(a).is_near_zero():
                        times2 = - b / a
                        tf.add_multiple_times_row_to_row(times2, j, i)

            i -= 1

        return tf
        '''


    def scale_row_to_make_coefficient_equal_one(self, row, col):
        n = self[row].normal_vector
        beta = Decimal('1.0') / n[col]
        self.multiply_coefficient_and_row(beta, row)


    def clear_coefficients_above(self, row, col):
        for k in range(row)[::-1]:
            n = self[k].normal_vector
            alpha = -n[col]
            self.add_multiple_times_row_to_row(alpha, row, k)


    def do_gaussian_elimination_and_extract_solution(self):
        rref = self.compute_rref()
        rref.raise_exception_if_contradictory_equation()
        rref.raise_exception_if_too_few_pivots()

        num_variables = self.dimension
        solution_coordinates = [rref.planes[i].constant_term for i in range(num_variables)]
        return Vector(solution_coordinates)


    def raise_exception_if_contradictory_equation(self):
        for p in self.planes:
            try:
                p.first_nonzero_index(p.normal_vector)

            except Exception as e:
                if str(e) == 'No nonzero elements found':
                    constant_term = MyDecimal(p.constant_term)
                    if not constant_term.is_near_zero():
                        raise Exception(self.NO_SOLUTIONS_MSG)

                else:
                    raise e


    def raise_exception_if_too_few_pivots(self):
        pivot_indices = self.indices_of_first_nonzero_terms_in_each_row()
        num_pivots = sum([1 if index >= 0 else 0 for index in pivot_indices])
        num_variables = self.dimension
        if num_pivots < num_variables:
            raise Exception(self.INF_SOLUTIONS_MSG)


    def indices_of_first_nonzero_terms_in_each_row(self):
        num_equations = len(self)
        num_variables = self.dimension

        indices = [-1] * num_equations

        for i,p in enumerate(self.planes):
            try:
                indices[i] = p.first_nonzero_index(p.normal_vector)
            except Exception as e:
                if str(e) == Plane.NO_NONZERO_ELTS_FOUND_MSG:
                    continue
                else:
                    raise e

        return indices


    def compute_solution(self):
        try:
            return self.do_gaussian_elimination_and_parametrize_solution()
        except Exception as e:
            if str(e) == self.NO_SOLUTIONS_MSG:
                return str(e)
            else:
                raise e


    def do_gaussian_elimination_and_parametrize_solution(self):
        rref = self.compute_rref()
        print rref

        rref.raise_exception_if_contradictory_equation()

        direction_vectors = rref.extract_direction_vectors_for_parametrization()
        basepoint = rref.extract_basepoint_vectors_for_parametrization()

        return Parametrization(basepoint, direction_vectors)


    def extract_direction_vectors_for_parametrization(self):
        num_variables = self.dimension
        pivot_indices = self.indices_of_first_nonzero_terms_in_each_row()
        free_variable_indices = set(range(num_variables)) - set(pivot_indices)

        direction_vectors = []

        for free_var in free_variable_indices:
            vector_coords = [0] * num_variables
            vector_coords[free_var] = 1
            for i, p in enumerate(self.planes):
                pivot_var = pivot_indices[i]
                if pivot_var < 0:
                    break
                vector_coords[i] = -p.normal_vector[free_var]
            direction_vectors.append(Vector(vector_coords))

        return direction_vectors


    def extract_basepoint_vectors_for_parametrization(self):
        num_variables = self.dimension
        pivot_indices = self.indices_of_first_nonzero_terms_in_each_row()

        basepoint_coords = [0] * num_variables
        for i, p in enumerate(self.planes):
            pivot_var = pivot_indices[i]
            if pivot_var < 0:
                break
            basepoint_coords[pivot_var] = p.constant_term

        return Vector(basepoint_coords)


    def __len__(self):
        return len(self.planes)


    def __getitem__(self, i):
        return self.planes[i]


    def __setitem__(self, i, x):
        try:
            assert x.dimension == self.dimension
            self.planes[i] = x

        except AssertionError:
            raise Exception(self.ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG)


    def __str__(self):
        ret = 'Linear System:\n'
        temp = ['Equation {}: {}'.format(i+1,p) for i,p in enumerate(self.planes)]
        ret += '\n'.join(temp)
        return ret


class MyDecimal(Decimal):
    def is_near_zero(self, eps=1e-10):
        return abs(self) < eps


class Parametrization(object):
    BASEPT_AND_DIR_VECTORS_MUST_BE_IN_SAME_DIM_MSG = 'The basepoint and direction vectors should all live in the same dimension'

    def __init__(self, basepoint, direction_vectors):
        self.basepoint = basepoint
        self.direction_vectors = direction_vectors
        self.dimension = self.basepoint.dimension

        try:
            for v in direction_vectors:
                assert v.dimension == self.dimension

        except AssertionError:
            raise Exception(BASEPT_AND_DIR_VECTORS_MUST_BE_IN_SAME_DIM_MSG)


    def __str__(self):
        result = 'basepoint:\n' + str(self.basepoint) + '\n\n'
        result += 'direction_vectors:\n'
        for v in self.direction_vectors:
            result += str(v) + '\n'

        return result


# p0 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
# p1 = Plane(normal_vector=Vector(['0','1','0']), constant_term='2')
# p2 = Plane(normal_vector=Vector(['1','1','-1']), constant_term='3')
# p3 = Plane(normal_vector=Vector(['1','0','-2']), constant_term='2')
#
# s = LinearSystem([p0,p1,p2,p3])

# print s.indices_of_first_nonzero_terms_in_each_row()
# print '{},{},{},{}'.format(s[0],s[1],s[2],s[3])
# print len(s)
# print s

# s.swap_rows(0, 3)
# s.multiply_coefficient_and_row(2, 3)
# s.add_multiple_times_row_to_row(1, 3, 2)
# print s

# system = s.compute_triangular_form()
# print system




# p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
# p2 = Plane(normal_vector=Vector(['0','1','1']), constant_term='2')
# s = LinearSystem([p1,p2])
# r = s.compute_rref()
# if not (r[0] == Plane(normal_vector=Vector(['1','0','0']), constant_term='-1') and
#         r[1] == p2):
#     print 'test case 1 failed'
# else:
#     print 'Pass case 1'
#
# p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
# p2 = Plane(normal_vector=Vector(['1','1','1']), constant_term='2')
# s = LinearSystem([p1,p2])
# r = s.compute_rref()
# if not (r[0] == p1 and
#         r[1] == Plane(constant_term='1')):
#     print 'test case 2 failed'
# else:
#     print 'Pass case 2'
#
# p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
# p2 = Plane(normal_vector=Vector(['0','1','0']), constant_term='2')
# p3 = Plane(normal_vector=Vector(['1','1','-1']), constant_term='3')
# p4 = Plane(normal_vector=Vector(['1','0','-2']), constant_term='2')
# s = LinearSystem([p1,p2,p3,p4])
# r = s.compute_rref()
# if not (r[0] == Plane(normal_vector=Vector(['1','0','0']), constant_term='0') and
#         r[1] == p2 and
#         r[2] == Plane(normal_vector=Vector(['0','0','-2']), constant_term='2') and
#         r[3] == Plane()):
#     print 'test case 3 failed'
# else:
#     print 'Pass case 3'
#
#
# p1 = Plane(normal_vector=Vector(['0','1','1']), constant_term='1')
# p2 = Plane(normal_vector=Vector(['1','-1','1']), constant_term='2')
# p3 = Plane(normal_vector=Vector(['1','2','-5']), constant_term='3')
# s = LinearSystem([p1,p2,p3])
# r = s.compute_rref()
# if not (r[0] == Plane(normal_vector=Vector(['1','0','0']), constant_term=Decimal('23')/Decimal('9')) and
#         r[1] == Plane(normal_vector=Vector(['0','1','0']), constant_term=Decimal('7')/Decimal('9')) and
#         r[2] == Plane(normal_vector=Vector(['0','0','1']), constant_term=Decimal('2')/Decimal('9'))):
#     print 'test case 4 failed'
# else:
#     print 'Pass case 4'




# p0 = Plane(normal_vector=Vector(['5.862',' 1.178','-10.366']), constant_term='-8.15')
# p1 = Plane(normal_vector=Vector(['-2.931','-0.589','5.183']), constant_term='-4.075')

# p0 = Plane(normal_vector=Vector(['8.631','5.112','-1.816']), constant_term='-5.113')
# p1 = Plane(normal_vector=Vector(['4.315','11.132','-5.27']), constant_term='-6.775')
# p2 = Plane(normal_vector=Vector(['-2.158','3.01','-1.727']), constant_term='-0.831')

# p0 = Plane(normal_vector=Vector(['5.262','2.739','-9.878']), constant_term='-3.441')
# p1 = Plane(normal_vector=Vector(['5.111','6.358','7.638']), constant_term='-2.152')
# p2 = Plane(normal_vector=Vector(['2.016','-9.924','-1.367']), constant_term='-9.278')
# p3 = Plane(normal_vector=Vector(['2.167','-13.593','-18.883']), constant_term='-10.567')




p0 = Plane(Vector(['0.786', '0.786', '0.588']), '-0.714')
p1 = Plane(Vector(['-0.138', '-0.138', '0.244']), '0.319')

# p0 = Plane(Vector(['8.631', '5.112', '-1.816']), '-5.113')
# p1 = Plane(Vector(['4.315', '11.132', '-5.27']), '-6.775')
# p2 = Plane(Vector(['-2.158', '3.01', '-1.727']), '-0.831')


# p0 = Plane(Vector(['0.935', '1.76', '-9.365']), '-9.955')
# p1 = Plane(Vector(['0.187', '0.352', '-1.873']), '-1.991')
# p2 = Plane(Vector(['0.374', '0.704', '-3.746']), '-3.982')
# p3 = Plane(Vector(['-0.561', '-1.056', '5.619']), '5.973')

s = LinearSystem([p0, p1])
r = s.compute_solution()
print r
