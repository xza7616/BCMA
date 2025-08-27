import os
import time
#Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息
from tqdm import tqdm
#图像处理库PIL的Image
from PIL import Image
import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from toolbox import get_dataset
from toolbox import get_model, get_mutual_model
from toolbox import averageMeter, runningScore
from toolbox import class_to_RGB, load_ckpt


def evaluate(logdir, save_predict=False):
    # 加载配置文件cfg
    cfg = None
    for file in os.listdir(logdir):
        if file.endswith('.json'):
            with open(os.path.join(logdir, file), 'r') as fp:
                cfg = json.load(fp)
    assert cfg is not None

    device = torch.device('cuda')

    testset = get_dataset(cfg)[-1]
    test_loader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=cfg['num_workers'])

    model = get_model(cfg).to(device)
    # model = get_mutual_model(cfg).to(device)

    model = load_ckpt(logdir, model, kind='best_A')
    # model = load_ckpt(logdir, model, kind='best_B')

    # save_pth = os.path.join(logdir, '0.755.A')
    # model.load_state_dict(torch.load(save_pth))

    #cfg['n_classes']：用于初始化性能指标跟踪器，确保性能指标与类别数目相关连
    # running_metrics_val = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    running_metrics_val = runningScore(cfg['n_classes'], ignore_index=None)
    #计算和跟踪损失、准确度的平均值
    time_meter = averageMeter()
    #创建目录来保存模型的预测结果
    save_path = os.path.join(logdir, 'predictsA')
    # save_path = os.path.join(logdir, 'predictsB')
    if not os.path.exists(save_path) and save_predict:
        os.mkdir(save_path)

    #模型在测试数据上进行推断
    #上下文管理器，用于确保在其内部代码快中关闭梯度计算。测试阶段一般不需梯度计算，提可推断速度和节约空间
    with torch.no_grad():
        #切换到评估模式，评估模式模式下，模型的行为可能不同
        model.eval()
        #迭代测试数据集的循环；enumerate：同时获取数据和他的索引；tqdm：一个进度条来跟踪迭代进度
        for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
            time_start = time.time()
            depth = sample['depth'].to(device)
            image = sample['image'].to(device)
            label = sample['label'].to(device)

            # resize
            h, w = image.size(2), image.size(3)
            #将图像和深度图进行插值操作，确保他们具有与模型输入匹配的大下
            image = F.interpolate(image, size=(int((h // 32) * 32), int(w // 32) * 32), mode='bilinear',
                                  align_corners=True)
            depth = F.interpolate(depth, size=(int((h // 32) * 32), int(w // 32) * 32), mode='bilinear',
                                  align_corners=True)
            # predict = model(image)
            predict = model(image, depth)
            # print(predict[0])
            # print(predict[0].shape)
            # return to the original size
            #‘h’,‘w’分别是之前通过image.size(2)和 image.size(3)获取的
            # predict = F.interpolate(predict[0], size=(h, w), mode='bilinear', align_corners=True)

            #处理预测结果
            #predict.max(1)用于计算每个像素点的最大预测值及其对应的类别索引
            #'[1]'用于从提取类别索引；.cpu().numpy()：将结果从GPU转到CPU上，并且转换成NUMPY数组
            predict = predict[0].max(1)[1].cpu().numpy()  # [1, h, w]  ###########
            # predict = predict[2].max(1)[1].cpu().numpy()
            label = label.cpu().numpy()
            #方便更新性能指标
            running_metrics_val.update(label, predict)
            # running_metrics_val.update(label, predict)

            #用于测量和记录模型推断每个样本所需要的时间
            time_meter.update(time.time() - time_start, n=image.size(0))

            if save_predict:
                #如果需要保存预测结果，则将predict的第一个纬度压缩掉
                predict = predict.squeeze(0)  # [1, h, w] -> [h, w]
                #将预测结果从类别索引转换成彩色图像
                #N表示类别总数，camp表示一个包含颜色信息的数据结构的类别映射
                predict = class_to_RGB(predict, N=len(testset.cmap), cmap=testset.cmap)  # 如果数据集没有给定cmap,使用默认cmap
                predict = Image.fromarray(predict)  #将numpy数组转换成PIL图像
                predict.save(os.path.join(save_path, sample['label_path'][0]))

    metrics = running_metrics_val.get_scores()
    for k, v in metrics[0].items():
        print(k, v)     #打印每个性能指标名称和值
    for k, v in metrics[1].items():
        print(k, v)
    for k, v in metrics[2].items():
        print(k, v)
    print('inference time per image: ', time_meter.avg)
    print('inference fps: ', 1 / time_meter.avg)
    #return metrics[0]['mIou: ']

#从指定目录中加载配置文件，在使用加载的配置文件来设置评估所需要的参数，最后运行评估过程
def msc_evaluate(logdir, save_predict=False):
    # 加载配置文件cfg
    cfg = None
    #遍历指定目录中的所有文件
    for file in os.listdir(logdir):
        if file.endswith('.json'):
            with open(os.path.join(logdir, file), 'r') as fp:
                #将配置文件的内容加载到cfg中，这样cfg中将包含配置文件中的参数和设置
                cfg = json.load(fp)
    assert cfg is not None

    device = torch.device('cuda')

    #获取测试数据集列表的最后一个
    testset = get_dataset(cfg)[-1]
    #数据加载器
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])

    model = get_model(cfg).to(device)
    model = load_ckpt(logdir, model,kind='epoch')

    running_metrics_val = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    time_meter = averageMeter()

    save_path = os.path.join(logdir, 'predicts')
    if not os.path.exists(save_path) and save_predict:
        os.mkdir(save_path)

    with torch.no_grad():
        model.eval()
        eval_scales = tuple(float(i) for i in cfg['eval_scales'].split(' '))
        eval_flip = cfg['eval_flip'] == 'true'
        for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
            time_start = time.time()
            depth = sample['depth'].to(device)
            image = sample['image'].to(device)
            label = sample['label'].to(device)
            # resize
            h, w = image.size(2), image.size(3)

            #创建一个全零的张量用于存储最终的预测结果。
            #张量的形状：1, cfg['n_classes'], h, w
            predicts = torch.zeros((1, cfg['n_classes'], h, w), requires_grad=False).to(device)
            #迭代不同的尺度
            for scale in eval_scales:
                #计算当前尺度下，图像的新H、W
                newHW = (int((h * scale // 32) * 32), int((w * scale // 32) * 32))
                new_image = F.interpolate(image, newHW, mode='bilinear', align_corners=True)
                new_depth = F.interpolate(depth, newHW, mode='bilinear', align_corners=True)
                out = model(new_image, new_depth)
                #将预测结果还原为原始图像大小
                out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
                #获得每个像素点不同类别的概率分布
                prob = F.softmax(out, 1)
                #用于综合不同尺度下的预测结果
                predicts += prob
                #是否进行水平反转
                if eval_flip:
                    #对图像和深度进行水平反转
                    out = model(torch.flip(new_image, dims=(3,)), torch.flip(new_depth, dims=(3,)))
                    out = torch.flip(out, dims=(3,))
                    out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
                    prob = F.softmax(out, 1)
                    predicts += prob
            predict = predicts.max(1)[1].cpu().numpy()
            label = label.cpu().numpy()
            running_metrics_val.update(label, predict)

            time_meter.update(time.time() - time_start, n=image.size(0))

            if save_predict:
                predict = predict.squeeze(0)  # [1, h, w] -> [h, w]
                predict = class_to_RGB(predict, N=len(testset.cmap), cmap=testset.cmap)  # 如果数据集没有给定cmap,使用默认cmap
                predict = Image.fromarray(predict)
                predict.save(os.path.join(save_path, sample['label_path'][0]))

    metrics = running_metrics_val.get_scores()
    for k, v in metrics[0].items():
        print(k, v)
    print('iou for each class.....')

    # for k, v in metrics[1].items():
    #     print(k, v)

    print('inference time per image: ', time_meter.avg)
    print('inference fps: ', 1 / time_meter.avg)


if __name__ == '__main__':
    #parse自带的命令行参数解析包，可以用来方便地读取命令行参数，当你的代码需要频繁地修改参数的时候，使用这个工具可以将参数和代码分离开来
    import argparse

    parser = argparse.ArgumentParser(description="evaluate")
    #####
    parser.add_argument("--logdir", type=str, help="run logdir", default="/home/pc/xza/run/2025-04-15-10-26-Test_kld+glo(lambda_=0.00005)_batch=6")    #########
    parser.add_argument("-s", type=bool, default=True, help="save predict or not")
    args = parser.parse_args()
    # print(args.logdir)
    evaluate(args.logdir, save_predict=args.s)
    # msc_evaluate(args.logdir, save_predict=args.s)
