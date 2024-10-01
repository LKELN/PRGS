import torch
import seaborn
import os
import re
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
datanames = os.listdir(r'../visible')
datanames=sorted(datanames,key=lambda x:os.path.getmtime(os.path.join("../visible",x)))
data_list=[]
i=1
fig = plt.figure(figsize=(12.8,9.6))
for j in range(1):
    for dataname in datanames:
            if os.path.splitext(dataname)[1] == '.pt':#目录下包含.json的文件
                data=torch.load(dataname,map_location="cpu").detach().numpy()
                pattern = r"(?<=\d).*?(?=\.)"
                match = re.search(pattern, dataname)
                if len(data.shape)==4:
                    data=data[0,1,:,:]
                if len(data.shape)==3:
                    data=data[0,:,:]
                plt.subplot(3,3,i)
                plt.title(match.group())
                plt.subplots_adjust(wspace=0.2,hspace=0.2)
                seaborn.heatmap(data,xticklabels=False,yticklabels=False)
                i+=1
    plt.savefig("heatmap.jpg",dpi=1000,bbox_inches='tight',pad_inches=0.1)
    plt.show()

        # data_list.append(data)



