import argparse
import os

def build_args():
    parser = argparse.ArgumentParser("This script is used for the HUAWEI SMPL Action Classification.")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--data_root", default="./window_data", type=str)
    parser.add_argument("--checkpoint", default="./checkpoint", type=str)
    parser.add_argument("--output", default="./output", type=str)
    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--batch_size", default=16, type=int)#8
    parser.add_argument("--lr", default=0.00004, type=float)#0.00002
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--min_epoch", default=0, type=int) ###
    parser.add_argument("--step_size", default=50, type=int)
    parser.add_argument("--step_gamma", default=0.1, type=float)
    parser.add_argument("--train_model", default="gcn", type=str)

    args = parser.parse_args()
    return init_args(args)

def init_args(args):
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    # if not os.path.exists(args.output):
    #     os.makedirs(args.output)
    return args
