import os
from tqdm import tqdm
import torch
import numpy as np

def euler_angles_to_axis_angle(rot_vecs):
    rot_angle = torch.norm(rot_vecs + 1e-8, dim=0, keepdim=True)
    rot_axis = rot_vecs / rot_angle
    return rot_axis, rot_angle

def ChangeGlobalOrientation2forward(out_root):
    file_list = []
    for file in os.listdir(out_root):
        file_name=os.path.join(out_root,file[:-4])
        file_list.append(file_name)
    Pi = 3.1415926
    Pi_axis = torch.from_numpy(np.array([Pi, 0, 0]))

    for name in tqdm(file_list):
        file_name = name + '.npy'
        poses = np.load(file_name, allow_pickle=True)
        poses = poses.astype(np.float32)
        SmplParams_name = name + '_forward.npy'
        print(name)
        if not os.path.exists(SmplParams_name):
            for frame in range(0,poses.shape[0]):
                rot_vec = poses[frame, :3].astype(np.float32)
                rot_vec = torch.from_numpy(rot_vec)
                rot_axis, rot_angle = euler_angles_to_axis_angle(rot_vec)
                rot_vec_forward = Pi_axis*rot_angle
                poses[frame, :3] = rot_vec_forward
            np.save(SmplParams_name,poses)

def get_window_data(out_root,window_root,stride=60,n_frame=60):
    window_save_dir=os.path.join(window_root,"stride"+str(stride))
    if not os.path.exists(window_save_dir):
        os.makedirs(window_save_dir)
    file_list=[]
    for file in os.listdir(out_root):
        if file.endswith(".npy"):
            file_name=file.split(".")[0]
            if file_name[-7:]=="forward":
                file_list.append(file_name)

    for f in tqdm(file_list,total=len(file_list)):
        save_path=os.path.join(window_save_dir,f)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        overall_data=np.load(os.path.join(out_root,f+".npy"),allow_pickle=True)

        steps=[2]
        for step in steps:
            nframes = overall_data.shape[0]
            lastone = step * (n_frame - 1)
            shift_max = nframes - lastone - 1
            shift=0
            while shift<shift_max:
                frame_idx = shift + np.arange(0, lastone + 1, step)
                pose = overall_data[frame_idx, :].astype(np.float32)
                np.save(os.path.join(save_path,str(shift)+"_"+str(step)+".npy"),pose)

                #为保证数据平衡，a4和a6步距增倍
                shift+=stride
                if int(f[2]) in [4,6]:
                    shift+=stride


def main():
    #将视角转换为forward  out_root中存放romp算法提取出的romp参数
    # out_root="./video_romp_result"
    # ChangeGlobalOrientation2forward(out_root)

    #滑窗采样数据
    for stride in [10]:
        out_root="./video_romp_result"
        window_root="./window_data"
        get_window_data(out_root,window_root,stride,n_frame=60)

if __name__ =="__main__":
    main()