# SMPL Action Recognition Implementation 
## 准备工作

### 推荐的环境
* Python 3.7
* Pytorch 1.7
* CUDA 10.1
* 依赖包
```python
pip install -r ./requirements.txt
```

### 准备数据
1.拍摄视频，视频命名格式为axx_cxx.mp4
- 如动作0，样本1：a00_c01.mp4

2.使用ROMP提取姿态参数
- input:video_folder
- output:video_romp_result

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
单击本地的URL链接即可进入ui界面




