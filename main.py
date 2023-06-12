import os
import torch
import torch.nn as nn
from model import Net
from dataset import Window_SMPLdataset_ALL
from opts import build_args
from tqdm import tqdm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(epoch,model,train_loader,optimizer,criterion,device):
    model.train()
    running_loss = 0.0
    for data, label in tqdm(train_loader):
        data=data.to(device)
        label=label.to(device)
        outputs=model(data)

        loss = criterion(outputs, label)
        loss=loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
    print("train epoch[{}] loss:{:.5f}".format(epoch+1,running_loss/len(train_loader)))

def test(epoch,model,val_loader,criterion,device):
    model.eval()
    acc=0.0
    running_loss = 0.0
    with torch.no_grad():
        for data, label in tqdm(val_loader):
            data=data.to(device)
            label=label.to(device)

            outputs=model(data)

            loss = criterion(outputs, label)
            loss=loss.mean()
            running_loss+=loss.item()

            predict_y=torch.max(outputs,dim=1)[1]
            label_y=torch.argmax(label)
            if predict_y[0] == label_y:
                acc += 1

    val_acc=acc/len(val_loader)
    loss=running_loss/len(val_loader)
    print("test epoch[{}] loss:{:.5f}".format(epoch+1,loss))

    return val_acc,loss


def main(args):
    os.environ['CUDA_VIVIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    args.device=device

    graph_args={"layout": 'smpl',"strategy": 'spatial'}
    model=Net(in_channels=6,num_class=11,graph_args=graph_args,edge_importance_weighting=True)
    model.apply(weights_init)

    model.to(device)
    if args.train_model == "gcn":
        gcn_grad=True
        cnn_grad=False
        model_save_name="best_gcn.pth"
    elif args.train_model == "cnn":
        model.load_state_dict(torch.load(os.path.join(args.checkpoint,"best_gcn.pth")))
        gcn_grad=False
        cnn_grad=True
        model_save_name="best_gcn_cnn.pth"
    else:
        print("Unrecognize model")

    for k, v in model.named_parameters():
        if any(x in k.split('.') for x in ['data_bn','st_gcn_networks','edge_importance','fcn']):
            if not gcn_grad:
                print('freezing %s' % k)
            v.requires_grad_(gcn_grad)
            
        if any(x in k.split('.') for x in ['x_1d_b','x_1d_s']):
            if not cnn_grad:
                print('freezing %s' % k)
            v.requires_grad_(cnn_grad)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss(reduction='none')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)

    train_dataset=Window_SMPLdataset_ALL(args,subset="train")
    val_dataset=Window_SMPLdataset_ALL(args,subset="val")

    if args.mode=="train":
        best_acc=0.0
        best_test_loss=1e10
        for epoch in range(args.epochs):
            train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=1,drop_last=True)
            val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,num_workers=1)
            
            train(epoch,model,train_loader,optimizer,criterion,device)
            acc,test_loss=test(epoch,model,val_loader,criterion,device)

            scheduler.step()
            if acc >= best_acc:
                best_acc=acc
                if test_loss<best_test_loss:
                    best_test_loss=test_loss
                    torch.save(model.state_dict(),os.path.join(args.checkpoint,model_save_name))
                    print("pth at {} epoch saved".format(epoch+1))
            print("VAL:Current accuracy:{:.3f},Best accuracy:{:.3f},Best test loss:{:.5f}".format(acc,best_acc,best_test_loss))
            
    elif args.mode=="infer":
        epoch=0
        model.load_state_dict(torch.load(os.path.join(args.checkpoint,"best_gcn_cnn.pth")),strict=False)
        val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=32,shuffle=False,num_workers=1)
        acc1,loss1=test(epoch,model,val_loader,criterion,device)
        print("Test:Accuracy:{:.3f} loss:{:.5f}".format(acc1,loss1))

if __name__ == "__main__":
    args=build_args()
    main(args)