# coding=utf-8

def shape(M):
    rows = len(M)
    cols = len(M[0])
    return rows, cols


def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]


def scaleRow(M, r, scale):
    if scale == 0:
        raise ValueError

    M[r] = [M[r][i] * scale for i in range(len(M[r]))]


def addScaledRow(M, r1, r2, scale):
    M[r1] = [M[r1][i] + M[r2][i] * scale for i in range(len(M[r1]))]


def matxRound(M, decPts=4):
    for row in M:
        for i, c in enumerate(row):
            row[i] = round(c, decPts)


# TODO 构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):
    rows, cols = shape(A)
    res = [[0] * (cols + 1) for i in range(rows)]
    for r in range(rows):
        for c in range(cols):
            res[r][c] = A[r][c]
        res[r][cols] = b[r][0]

    return res


# 实现 Gaussain Jordan 方法求解 Ax = b
""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16

    返回列向量 x 使得 Ax = b
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""

"""
    步骤1 检查A，b是否行数相同
    步骤2 构造增广矩阵Ab
    步骤3 逐列转换Ab为化简行阶梯形矩阵 中文维基链接
    对于Ab的每一列（最后一列除外）
        当前列为列c
        寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
        如果绝对值最大值为0
            那么A为奇异矩阵，返回None （请在问题2.4中证明该命题）
        否则
            使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c）
            使用第二个行变换，将列c的对角线元素缩放为1
            多次使用第三个行变换，将列c的其他元素消为0
    步骤4 返回Ab的最后一列
"""

def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):
    matxRound(A, decPts)
    matxRound(b, decPts)

    rows, cols = shape(A)
    if rows != len(b):
        return None

    Ab = augmentMatrix(A, b)
    for c in range(cols):
        maxValue = abs(Ab[c][c])
        maxIndex = c
        for i in range(c, rows):
            if abs(Ab[i][c]) > maxValue:
                maxValue = abs(Ab[i][c])
                maxIndex = i

        if abs(maxValue - epsilon) < 0 :
            return None

        swapRows(Ab, maxIndex, c)
        scaleRow(Ab, c, 1 / Ab[c][c])

        for j in range(rows):
            if j != c:
                addScaledRow(Ab, j, c, -Ab[j][c])

    return [[Ab[r][cols]] for r in range(rows)]


# A = [[9, 1, 4, 0, -4, 9], [5, -5, 8, 3, 9, 9], [2, 5, 0, 4, -8, -7], [-6, 9, -6, 6, 6, 3], [-4, 4, -7, -5, 4, 0], [3, 8, -3, 4, -7, -1]]
# b = [[0], [1], [2], [3], [4], [5]]
#
# print gj_Solve(A, b)




def linearRegression(points):
    sum_x_2 = 0
    sum_x = 0
    sum_xy = 0
    sum_y = 0
    for point in points:
        sum_x_2 += point[0] ** 2
        sum_x += point[0]
        sum_xy += point[0] * point[1]
        sum_y += point[1]

    A = [[sum_x_2, sum_x], [sum_x, len(points)]]
    b = [[sum_xy], [sum_y]]
    result =  gj_Solve(A, b)
    return result[0][0], result[1][0]


# print linearRegression([(0, 0), (1, 1), (4,4.1)])




# TODO 构造线性函数
def linearFunction(x):
    y = 2.0 * x + 1.0
    return y

# TODO 构造 100 个线性函数上的点，加上适当的高斯噪音
import random

def pointsGenerator(num=100):
    points = []
    for i in range(num):
        x = i
        y = linearFunction(x) + random.uniform(-2, 2)
        points.append([x, y])
    return points

#TODO 对这100个点进行线性回归，将线性回归得到的函数和原线性函数比较
print linearRegression(pointsGenerator())
