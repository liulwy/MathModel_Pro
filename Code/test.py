# 在main函数中添加这行代码，确认文件路径
import os
print(f"当前工作目录: {os.getcwd()}")
print(f"目标文件路径: {os.path.abspath('../src/result1.xlsx')}")