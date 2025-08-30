
import numpy as np

# 创建一个小的矩阵和向量来演示
np_matrix = np.arange(9).reshape(3, 3)
np_vector1 = np.array([10, 20, 30])
np_vector2 = np.array([1, 2, 3])

print("原始矩阵：\n", np_matrix)
print("\n原始向量1：", np_vector1)
print("\n原始向量2：", np_vector2)

# 第一步：矩阵和向量相减
result_subtraction = np_matrix - np_vector1
print("\n第一步的结果 (矩阵 - 向量1)：\n", result_subtraction)

# 第二步：结果矩阵和向量相乘
final_result = result_subtraction * np_vector2
print("\n最终结果 (结果矩阵 * 向量2)：\n", final_result)

print(final_result[1,2])

a = np.array([[1, 2, 3],
              [4, 5, 6]])

# 沿 axis=0 求和
col_sum = np.sum(a, axis=0)
print(col_sum)  # [5 7 9]
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
# 沿着行方向重复
b = np.repeat(a, 2, axis=0)
print(b)

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
# 沿着列方向求和
b = np.sum(a, axis=0)
print(b)