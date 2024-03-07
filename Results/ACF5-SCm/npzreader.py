import numpy as np

# 加载 npz 文件
data = np.load('res_PPO.npz')

# 查看 npz 文件中的所有数组名称
print("Arrays in npz file:", list(data.keys()))

# 逐个查看数组的内容
for array_name in data:
    print(f"Array '{array_name}':")
    array = data[array_name]
    print(array)  # 打印数组内容
    print("Shape:", array.shape)  # 打印数组形状
