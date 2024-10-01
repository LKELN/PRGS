import matplotlib.pyplot as plt
import numpy as np
# 统一设置
from proplot import rc
import matplotlib.ticker as ticker
from matplotlib.pyplot import MultipleLocator




# 显示图形
plt.show()
# 统一设置字体
rc["font.family"] = "Times New Roman"
# 统一设置轴刻度标签的字体大小
rc['tick.labelsize'] = 10
# 统一设置xy轴名称的字体大小
rc["axes.labelsize"] = 10
# 统一设置轴刻度标签的字体粗细
# rc["axes.labelweight"] = "light"
# 统一设置xy轴名称的字体粗细
# rc["tick.labelweight"] = "bold"

# fig,axes = plt.subplots(1,1,figsize=(8,8),dpi=100,facecolor="w")
# fig.subplots_adjust(left=0.2,bottom=0.2)

# ax.tick_params(bottom=False, top=False, left=False, right=False)

# axes.set_xlabel('X')
# axes.set_ylabel('Y')

plt.figure(dpi=300,figsize=(3.57,3.14))
plt.subplots_adjust(top=2, bottom=0, right=2, left=0, hspace=1, wspace=0.2)
plt.margins(0, 0)
ax=plt.subplot(1,1,1)
# 去除次要刻度
ax.xaxis.set_minor_locator(ticker.NullLocator())
ax.yaxis.set_minor_locator(ticker.NullLocator())
x=np.arange(1,26,step=1)
index=[1,4,9,14,15,16,17,18,19,24]
R_1=[85.1,84.7,85.4,85.3,85.8,85.6,85.3,85.0,83.9,85.3,85.2,84.7,84.7,85.7,85.8,86.1,85.4,84.6,86.4,84.9,84.9,84.8,85.3,84.7,85.1]
R_5=[92.5,92.8,92.6,93.0,92.5,92.8,92.6,93.1,92.2,92.5,92.8,92.8,92.4,93.1,92.3,93.6,93.2,92.8,93.5,92.3,92.8,92.9,92.9,92.6,92.7]
R_10=[94.8,94.8,94.6,95.2,94.7,94.7,94.7,95.0,94.4,94.6,94.7,94.6,94.5,94.9,94.6,95.2,95.2,94.8,95.4,94.3,94.5,95.1,95.1,94.8,94.7]
new_R_1=np.array(R_1)[index]
new_R_5=np.array(R_5)[index]
new_R_10=np.array(R_10)[index]
x=x[index]
plt.ylabel("Recall@N(%)")
plt.xlabel("Number")
plt.ylim((80,100))
plt.xlim(0,26)
plt.yticks(np.arange(80, 105, 5))

plt.plot(x,new_R_1,color="orange",linewidth=1,label="R@1")
plt.plot(x,new_R_5,color="red",linewidth=1,label="R@5")
plt.plot(x,new_R_10,color="green",linewidth=1,label="R@10")
ax.set_xticks([0,5,10,15,20,25])


plt.grid(b=True,axis="both")
plt.legend(loc=4,fontsize=6)
bbox_to_anchor=(1.19,0.115)
ax.set_yticks([80,85,90,95,100])

ax=plt.subplot(1,1,1)
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=None,hspace=None)
y_major_locator = MultipleLocator(1)
ax.xaxis.set_major_locator(y_major_locator)
# 去除次要刻度
ax.xaxis.set_minor_locator(ticker.NullLocator())
ax.yaxis.set_minor_locator(ticker.NullLocator())
# ax.set_xticks([50,75,100,125,150])
# ax.set_yticks([80,85,90,95,100])
x=np.arange(16,22,step=1)
y=[84.5,84.6,86.4,84.2,85.2]
R_1=[88.0,88.2,88.9,88.4,88.4,86.4]
R_5=[94.5,93.9,94.1,94.3,94.1,93.9]
R_10=[95.1,94.9,95.0,95.3,95.1,95.1]
plt.ylim((80,100))
# plt.xlim(15,22)
plt.yticks(np.arange(80, 105, 5))
plt.ylabel("Recall@N(%)")
plt.xlabel("Number")
p1=plt.plot(x,R_1,color="orange",linewidth=1,label="R@1")
p2=plt.plot(x,R_5,color="red",linewidth=1,label="R@5")
p3=plt.plot(x,R_10,color="green",linewidth=1,label="R@10")
plt.legend(loc=4,fontsize=6)
plt.grid(b=True,axis="both")

plt.savefig(r"C:\Users\LKN\Desktop\GCNrerank\vision_pdf\Figure_5_msls.jpg",dpi=300,bbox_inches='tight')
# plt.tick_params(bottom=False, top=False, left=False, right=False)
# ax.tick_params(bottom=False,top=False,left=False,right=False)
plt.show()


