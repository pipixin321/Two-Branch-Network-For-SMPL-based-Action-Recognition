import torch
from torch.utils.data import Dataset
import os
import utils
import numpy as np
import random
from data_argumention import Add_Noise

class Window_SMPLdataset_ALL(Dataset):
    def __init__(self,args,subset="train",stride=10,test_ratio=0.3):
        self.stride=stride
        self.subset=subset
        self.ActionLength=60
        self.test_ratio=test_ratio
        self.class_num=11
        self.use_random_data=False
        self.DA=Add_Noise()
        self.get_basedata(args)
        
    def get_basedata(self,args):
        data_folder=os.path.join(args.data_root,"stride"+str(self.stride))

        self.data_list=[]
        self.label_list=[]

        for file in os.listdir(data_folder):
            file_path=os.path.join(data_folder,file)
            label=int(file[1:3])
            if not self.use_random_data:
                if label==10:
                    continue
                
            if label>10:
                if label==11 or label==13:
                    label=10
                else:
                    continue

            for data in os.listdir(file_path):
                data_path=os.path.join(file_path,data)
                self.data_list.append(data_path)
                self.label_list.append(label)

        total_num=len(self.data_list)
        self.train_datalist,self.train_labellist=[],[]
        self.test_datalist,self.test_labellist=[],[]
        random.seed(0)
        test_idx=random.sample(range(0,total_num),int(total_num*self.test_ratio))
        for idx, data in enumerate(self.data_list):
            if idx in test_idx:  
                self.test_datalist.append(data)
                self.test_labellist.append(self.label_list[idx])
            else:
                self.train_datalist.append(data)
                self.train_labellist.append(self.label_list[idx])
        
        print("Total number of sample:{},train:{},val:{}".format(len(self.data_list),len(self.train_datalist),len(self.test_datalist)))

    def __len__(self):
        if self.subset == "train":
            return len(self.train_datalist)
        elif self.subset == 'val':
            return len(self.test_datalist)
        else:
            return 0

    def __getitem__(self,idx):
        if self.subset == 'train':
            datalist=self.train_datalist
            labellist=self.train_labellist
        elif self.subset == 'val':
            datalist=self.test_datalist
            labellist=self.test_labellist
        else:
            print("None subset")

        #加载数据并预处理
        pose=np.load(datalist[idx],allow_pickle=True)
        input_pose=self.preprocess(pose)

        c=labellist[idx]
        label=np.zeros(self.class_num,dtype=np.float32)
        label[c]=1

        return input_pose,torch.from_numpy(label)

    def preprocess(self,pose):
        if self.subset=="train":
            pose=self.DA(pose)
        pose = torch.from_numpy(pose).reshape(-1, 24, 3)  # 这里

        # 进行姿态格式转换: 三元轴角式->四元数->旋转矩阵->rot6D  data.shape=[60,24,6]
        # input_pose = utils.matrix_to_rotation_6d(utils.axis_angle_to_matrix(pose))
        input_pose = utils.matrix_to_rotation_6d(utils.axis_angle_to_matrix(pose))
        input_pose = input_pose.permute(1, 2, 0).contiguous()  # 将数据维度变成[24,6,ActionLength]
        input_pose = input_pose.float()

        return input_pose

if __name__ == "__main__":
    from opts import build_args
    args=build_args()
    dataset=Window_SMPLdataset_ALL(args,subset="train")
    data,label=dataset.__getitem__(0)
    print(data.shape,label)
