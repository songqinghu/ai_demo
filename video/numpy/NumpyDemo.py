import numpy as np

# 创建ndarray
score = np.array(
    [[80, 89, 86, 67, 79],
     [78, 97, 89, 67, 81],
     [90, 94, 78, 67, 74],
     [91, 91, 90, 67, 69],
     [76, 87, 75, 67, 86],
     [70, 79, 84, 67, 84],
     [94, 92, 93, 67, 64],
     [86, 85, 83, 67, 80]])

print(score)

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([1, 2, 3, 4])
c = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

# 数组纬度的元组
print(a.shape)
print(b.shape)
print(c.shape)
# 数组维数
print(a.ndim)
print(b.ndim)
print(c.ndim)
# 数组中元素个数
print(a.size)
print(b.size)
print(c.size)
# 一个数组的长度
print(a.itemsize)
print(b.itemsize)
print(c.itemsize)
# 数组的数据类型
print(a.dtype)
print(b.dtype)
print(c.dtype)
