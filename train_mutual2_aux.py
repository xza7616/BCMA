
#os 库是Python标准库，包含几百个函数，常用的有路径操作、进程管理、环境参数等。
import os
#高级的 文件、文件夹、压缩包 处理模块
import shutil
#JSON(JavaScript Object Notation, JS 对象简谱) 是一种轻量级的数据交换格式。
import json
import time
#加速
# from apex import amp
from torch.cuda import amp
import tqdm
# import apex
import numpy as np
#分布式通信包
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
#寻找最适合当前配置的高效算法，来达到优化运行效率的问题
import torch.backends.cudnn as cudnn
#调整学习率（learning rate）的方法

from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader

from toolbox import MscCrossEntropyLoss
from KD_loss.loss import KLDLoss
from KD_loss.Global_loss import BCMALoss as global_loss

from toolbox import get_dataset
from toolbox import get_logger
from toolbox import get_model
from toolbox import get_mutual_model
# from toolbox import get_model_t
from toolbox import averageMeter, runningScore
from toolbox import ClassWeight, save_ckpt,load_ckpt
from toolbox import Ranger
# from toolbox.kdlosses import *
torch.manual_seed(123)
#程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
cudnn.benchmark = True

def kl_divergence_loss(log_A,log_B):
    p_A = F.log_softmax(log_A,dim=1)
    p_B = F.log_softmax(log_B,dim=1)
    KL_loss = F.kl_div(p_A,p_B,reduction='batchmean')
    return KL_loss

def run(args):
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)
    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}-Test'

    args.logdir = logdir

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    if args.local_rank == 0:
        logger.info(f'Conf | use logdir {logdir}')

    model_A = get_model(cfg)

    model_A.load_pre('/home/pc/xza/Pretrain/mit_b2.pth')

    print('****************mutual_PTH loading Finish!*************')

    model_B = get_mutual_model(cfg)
    
    print('****************mutual_PTH loading Finish!*************')

    trainset, *testset = get_dataset(cfg)
    device = torch.device('cuda:0')
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.local_rank == 0:
            print(f"WORLD_SIZE is {os.environ['WORLD_SIZE']}")

    train_sampler = None
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()

        # model = apex.parallel.convert_syncbn_model(model)
        model_A = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_A)
        model_B = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_B)

        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    model_A.to(device)
    model_B.to(device)

    # teacher.to(device)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=(train_sampler is None),
                              num_workers=cfg['num_workers'], pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(testset[0], batch_size=1, shuffle=False,num_workers=cfg['num_workers'],pin_memory=True, drop_last=True)
    params_list_A = model_A.parameters()
    params_list_B = model_B.parameters()

    optimizer_A = torch.optim.SGD(params_list_A, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'], momentum=cfg['momentum'])
    optimizer_B = torch.optim.SGD(params_list_B, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'], momentum=cfg['momentum'])


    Scaler_A = amp.GradScaler()
    Scaler_B = amp.GradScaler()

    #optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay']
    scheduler_A = LambdaLR(optimizer_A, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)
    scheduler_B = LambdaLR(optimizer_B, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)


    # class weight 计算
    if hasattr(trainset, 'class_weight'):
        print('using classweight in dataset')
        class_weight = trainset.class_weight
    else:
        classweight = ClassWeight(cfg['class_weight'])
        class_weight = classweight.get_weight(train_loader, cfg['n_classes'])

    class_weight = torch.from_numpy(class_weight).float().to(device)

    # class_weight[cfg['id_unlabel']] = 0


    criterion = MscCrossEntropyLoss(weight=class_weight).to(device)

    criterion_kld = KLDLoss(tau=1).to(device)
    criterion_glo1 = global_loss(
            s_channels=64,
            t_channels=96,
            gamma=0.5,
            lambda_=1
        ).to(device)
    criterion_glo2 = global_loss(
            s_channels=128,
            t_channels=192,
            gamma=0.5,
            lambda_=1
        ).to(device)
    criterion_glo3 = global_loss(
            s_channels=320,
            t_channels=384,
            gamma=0.5,
            lambda_=1
        ).to(device)
    criterion_glo4 = global_loss(
            s_channels=512,
            t_channels=768,
            gamma=0.5,
            lambda_=1
        ).to(device)
    

    train_loss_meter_A = averageMeter()
    train_loss_meter_B = averageMeter()
    train_loss_meter = averageMeter()

    val_loss_meter_A = averageMeter()
    val_loss_meter_B = averageMeter()

    running_metrics_val_A = runningScore(cfg['n_classes'], ignore_index=None)
    running_metrics_val_B = runningScore(cfg['n_classes'], ignore_index=None)

    flag = True 
    miou_A = 0
    miou_B = 0

    for ep in range(cfg['epochs']):
        if args.distributed:
            train_sampler.set_epoch(ep)

        # training
        model_A.train()
        model_B.train()
        # train_loss_meter_A.reset()
        # train_loss_meter_B.reset()
        train_loss_meter.reset()
        # teacher.eval()

        for i, sample in enumerate(train_loader):
            optimizer_A.zero_grad()  # 梯度清零
            optimizer_B.zero_grad()  # 梯度清零


            ################### train edit #######################
            depth = sample['depth'].to(device)
            image = sample['image'].to(device)
            label = sample['label'].to(device)
            # label = sample['labelcxk'].to(device)
            # print(i,set(label.cpu().reshape(-1).tolist()),'label')


            with amp.autocast():

        
                predict_A = model_A(image, depth)  ########
               
                with torch.no_grad():
                    predict_B = model_B(image, depth)
                loss_A = criterion(predict_A[0], label)  #######################1
               
                lossA_B = criterion_kld(predict_A[0],predict_B[0].detach())

                lossFT_A_B1 = criterion_glo1(predict_A[1], predict_B[1].clone())
                lossFT_A_B2 = criterion_glo2(predict_A[2], predict_B[2].clone())
                lossFT_A_B3 = criterion_glo3(predict_A[3], predict_B[3].clone())
                lossFT_A_B4 = criterion_glo4(predict_A[4], predict_B[4].clone())

                
                total_A = loss_A + lossA_B #+ lossFT_A_B1 + lossFT_A_B2 +lossFT_A_B3+lossFT_A_B4

            ####################################################

           
            Scaler_A.scale(total_A).backward()
            Scaler_A.step(optimizer_A)
            Scaler_A.update()
           
            train_loss_meter.update(total_A.item())

            with amp.autocast():
                predict_B = model_B(image, depth)  ########
                with torch.no_grad():
                    # predict_A_detach0 = predict_A[0].detach()
                    predict_A = model_A(image, depth)
                    # predict_A_detach1 = predict_A[1].detach()

                loss_B = criterion(predict_B[0], label)  #######################1
                
                lossB_A = criterion_kld(predict_B[0], predict_A[0].detach())
                lossFT_B_A1 = criterion_glo1(predict_A[1].clone(), predict_B[1])
                lossFT_B_A2 = criterion_glo2(predict_A[2].clone(), predict_B[2])
                lossFT_B_A3 = criterion_glo3(predict_A[3].clone(), predict_B[3])
                lossFT_B_A4 = criterion_glo4(predict_A[4].clone(), predict_B[4])
               
                total_B = loss_B + lossB_A #+ lossFT_B_A1 + lossFT_B_A2 + lossFT_B_A3 + lossFT_B_A4

            Scaler_B.scale(total_B).backward()
            Scaler_B.step(optimizer_B)
            Scaler_B.update()   
            train_loss_meter.update(total_B.item())
           

            if args.distributed:
                reduced_loss_A = total_A.clone()
                dist.all_reduce(reduced_loss_A, op=dist.ReduceOp.SUM)
                reduced_loss_A /= args.world_size
                train_loss_meter.update(reduced_loss_A.item())

                reduced_loss_B = total_B.clone()
                dist.all_reduce(reduced_loss_B, op=dist.ReduceOp.SUM)
                reduced_loss_B /= args.world_size
                train_loss_meter.update(reduced_loss_B.item())
            else:
                train_loss_meter.update(total_A.item())
                train_loss_meter.update(total_B.item())

        scheduler_A.step(ep)
        scheduler_B.step(ep)

        # val
        with torch.no_grad():
            model_A.eval()
            model_B.eval()

            running_metrics_val_A.reset()
            running_metrics_val_B.reset()

            val_loss_meter_A.reset()
            val_loss_meter_B.reset()

            ################### val edit #######################
            for i, sample in enumerate(val_loader):
                depth = sample['depth'].to(device)
                image = sample['image'].to(device)
                label = sample['label'].to(device)

                predict_A= model_A(image, depth)
                predict_B = model_B(image, depth)

                loss_A = criterion(predict_A[0], label)         #############################
                loss_B = criterion(predict_B[0], label)  #############################2

                val_loss_meter_A.update(loss_A.item())
                val_loss_meter_B.update(loss_B.item())

                predict_A = predict_A[0].max(1)[1].cpu().numpy()  # [1, h, w]
                predict_B = predict_B[0].max(1)[1].cpu().numpy()  # [1, h, w]

                label = label.cpu().numpy()

            ###################edit end#########################
                running_metrics_val_A.update(label, predict_A)
                running_metrics_val_B.update(label, predict_B)


        if args.local_rank == 0:
            logger.info(
                f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] '
                f'  Model A train/val loss={train_loss_meter.avg:.5f}/{val_loss_meter_A.avg:.5f}, '
                f', mAcc={running_metrics_val_A.get_scores()[0]["mAcc: "]:.3f}'
                f', miou={running_metrics_val_A.get_scores()[0]["mIou: "]:.3f}'
                f', best_miou={miou_A:.3f}'
                f'  Model B train/val loss={train_loss_meter.avg:.5f}/{val_loss_meter_B.avg:.5f}, '
                f', mAcc={running_metrics_val_B.get_scores()[0]["mAcc: "]:.3f}'
                f', miou={running_metrics_val_B.get_scores()[0]["mIou: "]:.3f}'
                f', best_miou={miou_B:.3f}'
            )
            save_ckpt(logdir, model_A, kind='end_A')
            save_ckpt(logdir, model_B, kind='end_B')
            newmiou_A = running_metrics_val_A.get_scores()[0]["mIou: "]
            newmiou_B = running_metrics_val_B.get_scores()[0]["mIou: "]

            if newmiou_A > miou_A:
                save_ckpt(logdir, model_A, kind='best_A')  #消融可能不一样
                miou_A = newmiou_A

            if newmiou_B > miou_B:
                save_ckpt(logdir, model_B, kind='best_B')  #消融可能不一样
                miou_B = newmiou_B

    save_ckpt(logdir, model_A, kind='end_A')  #保存最后一个模型参数
    save_ckpt(logdir, model_B, kind='end_B')  #保存最后一个模型参数

if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/SUIM.json",
        # default="configs/sunrgbd.json",
        # default="configs/WE3DS.json",
        help="Configuration file to use",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--opt_level",
        type=str,
        default='O1',
    )

    args = parser.parse_args()
    run(args)
