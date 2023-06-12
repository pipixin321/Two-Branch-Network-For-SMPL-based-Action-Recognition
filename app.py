import os
import utils
import torch
import numpy as np
import gradio as gr
from model import Net
from opts import build_args

class_lst=['Jumping jack','lunging','Left stretching','Raising hand and jumping','Wrist circling',
           'Single dumbbell raising','Dumbbell one-arm shoulder pressing',
           'Dumbbell shrugging','Pinching back','Shoulder abduction','Unkown']

def preprocess(pose):
    pose = torch.from_numpy(pose).reshape(-1, 24, 3)  # 这里
    # 进行姿态格式转换: 三元轴角式->四元数->旋转矩阵->rot6D  data.shape=[60,24,6]
    # input_pose = utils.matrix_to_rotation_6d(utils.axis_angle_to_matrix(pose))
    input_pose = utils.matrix_to_rotation_6d(utils.axis_angle_to_matrix(pose))
    input_pose = input_pose.permute(1, 2, 0).contiguous()  # 将数据维度变成[24,6,ActionLength]
    input_pose = input_pose.float()
    return input_pose

def action_classifier(input_file):
    input_file=input_file.name
    args=build_args()
    os.environ['CUDA_VIVIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    args.device=device

    graph_args={"layout": 'smpl',"strategy": 'spatial'}
    model=Net(in_channels=6,num_class=11,graph_args=graph_args,edge_importance_weighting=True)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint,"best_gcn_cnn.pth")),strict=False)
    model.to(device)

    pose=np.load(input_file,allow_pickle=True)
    input_pose=preprocess(pose)
    input_pose=input_pose.to(device)

    model.eval()
    with torch.no_grad():
        outputs=model(input_pose.unsqueeze(0))
    outputs=outputs[0].detach().cpu().numpy()

    outputs={class_lst[i]:float(outputs[i]) for i in range(outputs.shape[0])}

    print(outputs)
    return outputs


if __name__ == '__main__':
    demo = gr.Interface(fn=action_classifier, inputs=gr.File(), outputs=gr.Label(num_top_classes=3))
    demo.launch()