# import pandas as pd
# df = pd.read_csv(r'C:\Users\Yumei\Music\Projects\Pycharm\scientificProject\源域输出\summary.csv')
# #统计一下故障类别这一列的情况
# print(df['通道'].value_counts())
# print(df['采样率'].value_counts())
# print(df['故障类别'].value_counts())
# print(df['尺寸(inch)'].value_counts())
# print(df['载荷'].value_counts())
import pandas as pd

# 所有维度的数据
all_data = {
    '采样通道': {'DE': 161, 'FE': 153, 'BA': 97},
    '采样率': {'12000.0': 299, '48000.0': 112},
    '故障类别': {'外圈': 203, '滚动体': 100, '内圈': 100, '正常': 8},
    '尺寸(inch)': {'0.007': 160, '0.021': 136, '0.014': 99, '0.028': 8},
    '载荷': {'0.0': 52, '1.0': 52, '2.0': 52, '3.0': 52}
}

# 转换为所需的格式
result = []
for dimension, values in all_data.items():
    value_str = ', '.join([f'{k}:{v}' for k, v in values.items()])
    result.append({'维度': dimension, '情况': value_str})

# 创建DataFrame
df = pd.DataFrame(result)
print(df)