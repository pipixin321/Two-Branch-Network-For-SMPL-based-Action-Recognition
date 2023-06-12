'''
用于水平翻转视频,输入为视频文件夹,输出的视频存放在out_dir中
'''

import os
import multiprocessing

def get_cmds(video_dir, out_dir, val_subset=False):
    cmds=[]
    for file in os.listdir(video_dir):
        if file.endswith(".mp4"):
            file_name = file.split(".")[0]
            if file_name[-7:] != "forward":
                input_path = os.path.join(video_dir,file)
                if val_subset:
                    file_name = file_name.replace("c","g")
                output_path = os.path.join(out_dir, file_name + "_flip.mp4")
                cmd = "ffmpeg -i {} -vf hflip -y {}".format(input_path, output_path)
                if not os.path.exists(output_path):
                    cmds.append(cmd)
    return cmds

def f(cmds):
    for cmd in cmds:
        os.system(cmd)

if __name__ == "__main__":
    out_dir = "/mnt/data1/zhx/CIGS/dataset/video0713_flip"
    data_root = "/mnt/data1/zhx/CIGS/dataset"
    video_dir_list = ["video0713"]
    for dir in video_dir_list:
        video_dir = os.path.join(data_root, dir)

        cmds = get_cmds(video_dir, out_dir)
        print(len(cmds))
        print(cmds)
        num_process = 10
        for i in range(num_process):
            process = multiprocessing.Process(target=f, args=(cmds[i * len(cmds) // num_process : (i + 1) * len(cmds) // num_process], ))
            process.start()



