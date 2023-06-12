# SMPL Action Recognition Implementation 
## 准备工作
本项目开源地址为：https://github.com/pipixin321/Two-Branch-Network-For-SMPL-based-Action-Recognition.git

### 推荐的环境
* Python 3.7
* Pytorch 1.7
* CUDA 10.1
* 依赖包
```python
pip install -r ./requirements.txt
```

### 准备数据
方法一：\
使用已经提取好的姿态参数：https://pan.baidu.com/s/1CPZ8CN1nY9RgVCn4OllpSQ?pwd=lbaf

方法二: \
1.拍摄视频，视频命名格式为axx_cxx.mp4
- 如动作0，样本1：a00_c01.mp4

2.使用ROMP提取姿态参数
- ROMP: https://github.com/Arthur151/ROMP，
- input: video_folder
- output: ideo_romp_result

### 数据预处理
```python
python data_preprocess.py
```
- input:./video_romp_result
- output:./window_data


## 训练和测试模型

## 训练gcn部分
model.py line110 mode改为train_gcn

```python
python main.py --mode train  --train_model gcn
```

## 训练cnn部分
model.py line110 mode改为train_cnn
```python
python main.py --mode train  --train_model cnn
```

## 测试
model.py line110 mode改为infer
```python
python main.py --mode infer
```
### APP的使用
```python
python app.py
```
- 单击本地的URL链接进入ui界面、
- 将examples文件夹中的姿态参数序列上传并提交




