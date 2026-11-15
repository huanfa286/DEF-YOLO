
#-----------YOLOv11train-----------------

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':

  model = YOLO('ultralytics/cfg/models/11/improve1/FPN-DEConv/yolov11-FPN-DEConv.yaml')
  # model = YOLO('yolo11n.pt')

  model.load('yolo11n.pt')  #注释则不加载
  results = model.train(
    data='VisDrone2019/visdrone2019.yaml',  #数据集配置文件的路径
    epochs=150,  #训练轮次总数
    batch=4,  #批量大小，即单次输入多少图片训练
    imgsz=800,  #训练图像尺寸
    workers=8,  #加载数据的工作线程数
    device= 0,  #指定训练的计算设备，无nvidia显卡则改为 'cpu'
    optimizer='AdamW',  #训练使用优化器，可选 auto,SGD,Adam,AdamW 等
    amp= True,  #True 或者 False, 解释为：自动混合精度(AMP) 训练
    cache=False,  # True 在内存中缓存数据集图像，服务器推荐开启
    # resume = True,  # 关键参数：启用恢复训练模式
    project='VisDrone2019/runs/our',
    name='train1',

    lr0=0.001,  # 初始学习率，AdamW搭配0.001最优，SGD是0.01，适配改进模型
    lrf=0.01,  # 学习率衰减因子，余弦退火到初始的1%，防止后期震荡
    warmup_epochs=3,  # 热身训练5轮，防止初始学习率太大导致梯度爆炸
    weight_decay=0.0005,  # L2正则化，轻微权重衰减，防止过拟合（无人机数据集易过拟合）
    patience=30,  # 早停机制：20轮精度不涨则自动停止训练，避免无效训练+过拟合
    val=True,  # 训练中验证，实时监控精度，必开
    save=True,  # 保存最优模型，必开
    save_period=10,  # 每10轮保存一次权重，防止意外中断
    cos_lr=True,  # ✅ 开启余弦退火学习率：核心！学习率平滑下降，收敛更稳定，精度更高
    label_smoothing=0.05,  # ✅ 标签平滑：轻微平滑标签，防止过拟合，对小目标检测精度提升明显
    mosaic=0.8,  # ✅ 马赛克增强：从1.0→0.8，降低增强强度，无人机小目标避免被过度切割导致特征丢失
    mixup=0.0,  # ✅ 开启轻微mixup增强：提升泛化性，0.1足够，太大则小目标特征模糊
    copy_paste=0.1,  # ✅ 开启轻微复制粘贴增强：针对无人机小目标密集场景，提升小目标召回率
    single_cls=False,  # 数据集是多类则False，单类则True，根据你的UAV数据集修改
    rect=False,  # 1024 尺寸关闭矩形训练，避免小目标特征变形

)
  
# --------------RTDETR-----------------------

# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import RTDETR 
# if __name__ == '__main__':
#   model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml')
#   model.load('rtdetr-l.pt')  #注释则不加载
#   results = model.train(
#     data='DOTA/dota.yaml',  #数据集配置文件的路径
#     epochs=200,  #训练轮次总数
#     batch=4,  #批量大小，即单次输入多少图片训练
#     imgsz=640,  #训练图像尺寸
#     workers=4,  #加载数据的工作线程数
#     device= 0,  #指定训练的计算设备，无nvidia显卡则改为 'cpu'
#     optimizer='SGD',  #训练使用优化器，可选 auto,SGD,Adam,AdamW 等
#     amp= True,  #True 或者 False, 解释为：自动混合精度(AMP) 训练
#     cache=False,  # True 在内存中缓存数据集图像，服务器推荐开启
#     resume = True,  # 关键参数：启用恢复训练模式
#     project='DOTA/runs',
#     # name='1-1024sunny'
# )