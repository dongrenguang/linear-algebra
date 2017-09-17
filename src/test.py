def shape(M):
    rows = len(M)
    cols = len(M[0])
    return rows, cols


def transpose(M):
    rows, cols = shape(M)
    res = [[0] * rows for i in range(cols)]
    for c in range(cols):
        for r in range(rows):
            res[c][r] = M[r][c]

    return res


def matxMultiply(A, B):
    rows_a, cols_a = shape(A)
    rows_b, cols_b = shape(B)

    if rows_a != cols_b or cols_a != rows_b or rows_a == 0 or cols_a == 0:
        return None

    # res = [[0 for i in range(cols_b)] for j in range(rows_a)]
    res = [[0]*cols_b for j in range(rows_a)]
    for r_a in range(rows_a):
        for c_b in range(cols_b):
            sum = 0
            for r_b in range(rows_b):
                sum += A[r_a][r_b] * B[r_b][c_b]
            res[r_a][c_b] = sum

    return res

a= [[1,2],[2,3],[1,1]]
b = [[3,2,1], [1,1,1]]
# a = [[-2, -4, 8, -2, 7, -10, -7, -1, -2, 3, -7, 9, 3, -5, 5, 2, -3, -3, 4, 9, -2, -3, -9], [1, 3, 5, -5, -5, -4, -10, -9, 1, -7, 6, 3, 9, 7, 6, 4, 5, -5, 2, 6, -10, 9, 5], [-9, 6, 8, -9, -9, -6, 8, 3, 9, 4, 8, -10, 8, 6, -1, 7, 6, 9, -3, 0, 8, 3, -3], [3, -8, -3, -8, 0, 2, 6, 7, -1, 0, 1, 6, 3, 6, 4, 1, -3, 8, -7, 1, 3, 0, 8], [1, 6, 1, -4, 8, -6, -1, 5, 5, 8, -9, -4, 4, -9, 8, 5, 6, 8, 8, -1, 2, 1, -10], [-10, 8, 0, 4, -10, -2, -10, 8, 7, -1, -8, -8, -4, 6, -5, -3, -4, 2, 9, -9, -2, 8, 1]]
# b = [[-4, 1, -5, -4, -1, -3, -4, 1, -5], [0, 2, -5, -4, -3, -2, -4, -5, 3], [3, 1, 4, 3, -5, -3, -2, -4, -5], [-2, 4, -2, -3, -5, 4, -4, -5, 3], [-2, -1, 4, 0, -4, -2, -2, 4, -5], [-4, -2, -1, 0, -4, 1, 2, -2, 2], [2, -3, 1, 4, -5, 3, 2, -4, 1], [-3, 2, 4, -4, -2, -4, -4, -5, -4], [-2, 3, 3, -3, 4, 1, -1, 2, 3], [-3, 2, 2, -1, 0, 3, 2, 2, 3], [0, -5, -2, 3, -4, -2, 3, -2, 4], [-5, -1, 4, 1, -1, 3, 1, -5, -5], [4, 2, -1, 4, 3, -4, -1, 2, -3], [3, -4, -4, -4, 2, -2, -2, 2, 3], [3, 0, 3, -4, 0, -5, -5, 1, 1], [-4, -5, -1, 1, 1, 2, -2, 2, -3], [-5, 0, -4, -1, 1, 3, 1, 0, 4], [-3, -3, 2, 2, -5, -2, -4, 2, 4], [-5, -2, -3, -4, 3, -4, 3, -4, 3], [-3, -4, -2, 3, -1, 4, -4, -1, 3], [0, -3, 2, 2, 1, 3, 1, 4, -4], [2, -2, 4, 4, 3, 2, -5, -4, 1], [-5, -5, -1, -5, -3, -1, 3, 0, -5]]

# print matxMultiply(a,b)

def augmentMatrix(A, b):
    rows, cols = shape(A)
    res = [[0] * (cols + 1) for i in range(rows)]
    for r in range(rows):
        for c in range(cols):
            res[r][c] = A[r][c]
        res[r][cols] = b[r]

    return res

# A = [[1,2,3], [2,3,4]]
# b = [5,6]
# print augmentMatrix(A, b)

def scaleRow(M, r, scale):

    print r
    print scale
    print M
    if scale != 0:
        M[r] = [M[r][i] * scale for i in range(len(M[r]))]

# a = [[1,2], [3,4]]
# print a
# scaleRow(a, 1, 3)
# print a
