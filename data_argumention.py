import torch.nn as nn
import numpy as np
import random

class Add_Noise(nn.Module):
    def __init__(self):
        super(Add_Noise,self).__init__()
        self.mean=0

    def forward(self,x,sigma=0.2,alpha=0):
        #加入高斯噪声，均值为0
        if sigma!=0:
            noise = np.random.normal(self.mean,sigma, x.shape)
            x=x+noise*x

        #随机mask
        if alpha!=0:
            num_channel=x.shape[1]
            mask_num=int(alpha*num_channel)
            sample_index=random.sample(list(range(num_channel)),mask_num)
            x[:,sample_index]=0
        return x
