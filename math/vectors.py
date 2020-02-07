import numpy as np


# 用一维数组表示，无行列之分
a = np.array([1, 2, 3, 4])
a_t = a.transpose()
print("a: {}, a_t: {}".format(a, a_t))

# 用二维数组间接表示向量，行向量：1行4列；列向量：4行1列
A = np.array([[1, 2, 3, 4]])
A_T = A.T
print("A: {}\nA_T: {}".format(A, A_T))
print("A.shape: {}, A_T.shape: {}".format(A.shape, A_T.shape))

# 内积，必须用一维数组计算，用二维数组表示列向量也不行，因为这种方式的表示本质是矩阵
u = np.array([2, 3, 4])
v = np.array([3, 4, 5])
result = np.dot(u, v)
print("u dot v = ", result)

# 外积
u = np.array([2, 3])
v = np.array([1, 4])
result = np.cross(u, v)
print("u cross v = ", result)

# 线性组合
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
w = np.array([7, 8, 9])
print(3*u + 4*v + 5*w)

