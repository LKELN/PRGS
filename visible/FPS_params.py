import matplotlib.pyplot as plt
import pandas as pd
import numpy
#读入文件并给变量赋值
# file_path = "data.txt"
# df = pd.read_table(file_path, header=None)
# x = []
# y = []
# for i in range(len(df[0])):
#     x.append(int(df[0][i].split(',')[0]))
#     y.append(int(df[0][i].split(',')[1]))

#小样本手动赋值
# FPS x; Params y;

plt.figure(figsize=(4.7,4.2))
# FPS
x = [6.4, 3.29, 25.6,3.5 , 0.151, 0.054]
# param
# x = [6.19, 6.5, 5.03, 46.9, 43.7, 137, 43.2, 49.7, 83]
# accuracy
y = [78.1,77.2,79.5,86.8,89.7,89.9]

label = ["SP-SuperGlue", "Patch-NetVLAD-s", "Patch-NetVLAD-p", "TransVPR", "R^2Former","Ours"]
# offset for Param vs FPS
# offset = [(-13,7),(-29,7),(-52,-4),(-35,-4),(-14,7),(7,-4),(5,-4),(7,-4)]
# offset for acc vs FPS
offset = [(-12,-11),(-27,6),(-18,6),(-11,6),(-13,6),(-13,-13),(-13,6),(-19,6),(-4,6)]
# offset for acc vs param
# offset = [(5,-5),(6,-4),(-9,5),(-12,5),(4,-4),(-25,-13),(4,-4),(4,-4),(-18,5)]
#定义颜色变量
color = ['brown', 'b', 'g', 'c', 'm', 'y']
# mark = ['*', '.', '.', '.', '.', '.', '.']
# plt.yscale('log')
plt.xscale('log')

plt.xlim(0, 30)
# plt.xlim(3, 180)
# plt.ylim(0.88,0.91)

# acc axle
y_ = [a for a in numpy.linspace(0.75, 0.90, num=7)]
y_name = list(numpy.linspace(0.75, 0.90, num=7))
plt.yticks(y_,y_name)

# FPS axle
x_ = [a for a in range(0,30,1)]
# x_name = [1,2,3,4,5,6,7]
# plt.xticks(x_,x_name)

# param axle
# x_ = [3, 5, 10, 50,100,150]
# x_name = x_
# plt.xticks(x_,x_name)

# 用于fontdict参数项
# ours_font = {
#     #'fontsize': rcParams['axes.titlesize'], # 设置成和轴刻度标签一样的大小
#     'fontsize': 12,
#     #'fontweight': rcParams['axes.titleweight'], # 设置成和轴刻度标签一样的粗细
#     'fontweight': 'normal',
#     #'color': rcParams['axes.titlecolor'], # 设置成和轴刻度标签一样的颜色
#     'color': 'r',
# }

# 可选粗细有 ['light','normal','medium','semibold','bold','heavy','black']
plt.xlabel("Matching Time",fontsize=10,fontweight='semibold')
plt.ylabel("Recall@N(%)",fontsize=10,fontweight='semibold')
# plt.title("Figure")

plt.grid(axis='both',linestyle='--')
#画图
plt.scatter(x, y, c=color)



for i, txt in enumerate(label):
    if i==0:
        plt.annotate(txt, xy=(x[i], y[i]), xytext=offset[i], textcoords='offset points',fontsize=10,fontweight='semibold',color='r')
    else:
        plt.annotate(txt, xy=(x[i], y[i]), xytext=offset[i], textcoords='offset points')


plt.tight_layout()
plt.show()