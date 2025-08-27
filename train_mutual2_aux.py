
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
#实现自由的数据读取,dataloadateset读取训练集dataset (Dataset): 加载数据的数据集
# * batch_size (int, optional): 每批加载多少个样本
# * shuffle (bool, optional): 设置为“真”时,在每个epoch对数据打乱.（默认：False）
# * sampler (Sampler, optional): 定义从数据集中提取样本的策略,返回一个样本
# * batch_sampler (Sampler, optional): like sampler, but returns a batch of indices at a time 返回一批样本. 与atch_size, shuffle, sampler和 drop_last互斥.
# * num_workers (int, optional): 用于加载数据的子进程数。0表示数据将在主进程中加载​​。（默认：0）
# * collate_fn (callable, optional): 合并样本列表以形成一个 mini-batch.  #　callable可调用对象
# * pin_memory (bool, optional): 如果为 True, 数据加载器会将张量复制到 CUDA 固定内存中,然后再返回它们.
# * drop_last (bool, optional): 设定为 True 如果数据集大小不能被批量大小整除的时候, 将丢掉最后一个不完整的batch,(默认：False).
# * timeout (numeric, optional): 如果为正值，则为从工作人员收集批次的超时值。应始终是非负的。（默认：0）
# * worker_init_fn (callable, optional): If not None, this will be called on each worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as input, after seeding and before data loading. (default: None)．

#from toolbox.datasets.nyuv2 import train_collate_fn
#from lib.data_fetcher import DataPrefetcher
from torch.utils.data import DataLoader

from toolbox import MscCrossEntropyLoss
from KD_loss.Knowledge_Distillation.kd_losses.at import AT
from KD_loss.TorchDistiller.SemSeg.utils.criterion import CriterionKD
from KD_loss.Knowledge_Distillation.kd_losses.Pair_wise import CriterionPairWiseforWholeFeatAfterPool
from KD_loss.multipw_kld import Multipwkld
# from KD_loss.CosineSimilarityLoss import FeatureSimilarityLoss
from KD_loss.loss import KLDLoss
from KD_loss.Global_loss import GlobalDistillLoss as global_loss
from KD_loss.CosineSimilarityLoss import FeatureSimilarityLoss
from toolbox.loss import lovaszSoftmax
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
#载configs下的配置文件
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)
    #用于保存日志文件或其他的与时间相关的数据
    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}-Test_kld+rela_loss+'


    #logdir = 'run/2020-12-23-18-38'
    args.logdir = logdir

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    #将源文件路径复制到logdir
    shutil.copy(args.config, logdir)

    #方便调试维护代码
    logger = get_logger(logdir)
    if args.local_rank == 0:
        logger.info(f'Conf | use logdir {logdir}')

    model_A = get_model(cfg)

    # model._load_resnet_pretrained()
    # model._freeze_parameters()
    # model._unfreeze_bn()
    # model_A.load_pre_sa('/home/wjy/桌面/yzy/yzyall/yzy/mymodelnew/toolbox/models/segformer/confg_pth/segformer.b4.512x512.ade.160k.pth')
    model_A.load_pre('/home/pc/xza/Pretrain/mit_b2.pth')
    # model.load_pre_sa('/home/wjy/桌面/yzy/yzyall/yzy/mymodelnew/toolbox/models/segformer/confg_pth/segformer.b4.512x512.ade.160k.pth')
    # model.initialize_weights()
    # model.load_pre('/home/hjk/桌面/mymodelnew/toolbox/models/BBSNet/resnet50-0676ba61.pth')
    # model.load_state_dict(torch.load('/home/hjk/桌面/mymodelnew/toolbox/models/CIRNet/CIRNet_R50.pth'), strict=False)
    print('****************student_PTH loading Finish!*************')

    model_B = get_mutual_model(cfg)
    # model_B.load_pre_sa('/media/maps/shuju/yzy/mymodelnew/toolbox/models/segformer/confg_pth/segformer.b2.512x512.ade.160k.pth')
    # model_B.load_pre_sa('/media/maps/shuju/yzy/mymodelnew/Backbone/p2t_segmentation/ADE20K/sem_fpn_p2t_s_ade20k_80k.pth')
    print('****************mutual_PTH loading Finish!*************')

    #将get_dataset返回的对象分别传给train、test
    trainset, *testset = get_dataset(cfg)
#torch.device代表将torch.Tensor分配到的设备的对象
    device = torch.device('cuda:0')
    args.distributed = False
#environ是一个字符串所对应环境的映像对象
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
##################################加载PTH预训练############################################
    # args.model_pretrained = True
    # if args.model_pretrained:
    #         print('load weights from ckpt_pretrained  ...')
    #         save_model = torch.load('/home/yangenquan/aabyby/ckpt_pretrained/model_SegformerB2_best.pth')
    #         model_dict = model.state_dict()
    #         state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys() and (v.shape == model_dict[k].shape)}
    #         model_dict.update(state_dict)
    #         model.load_state_dict(model_dict, strict=False)
    #         print('加载OK!!!')
    # else:
    #     assert 'unsupported model'
###########################################################################################
    model_A.to(device)
    model_B.to(device)

    # teacher.to(device)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=(train_sampler is None),
                              num_workers=cfg['num_workers'], pin_memory=True, sampler=train_sampler, drop_last=True)
    #                                             drop_last=True解决照片留单然后导致batch变成1
    val_loader = DataLoader(testset[0], batch_size=1, shuffle=False,num_workers=cfg['num_workers'],pin_memory=True, drop_last=True)
    params_list_A = model_A.parameters()
    params_list_B = model_B.parameters()
    # criterion_Multipwkld1 = Multipwkld(initial_scale=0.5, initial_alpha=0.5).to(device)
    # criterion_Multipwkld2 = Multipwkld(initial_scale=0.5, initial_alpha=0.5).to(device)



    # criterion_PW = CriterionPairWiseforWholeFeatAfterPool(initial_scale=0.5).to(device)
    # criterion_PW1 = CriterionPairWiseforWholeFeatAfterPool(initial_scale=0.5).to(device)
    # # loss_cos = FeatureSimilarityLoss(initial_alpha=0.5).to(device)

    # wd_params, non_wd_params = model.get_params()
    # params_list = [{'params': wd_params, },
    #                {'params': non_wd_params, 'weight_decay': 0}]
    #weight_decy是放在正则项（regularization）前面的一个系数,调节模型复杂度对损失函数的影响,防止过拟合
    # optimizer = torch.optim.Adam(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    optimizer_A = torch.optim.SGD(params_list_A, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'], momentum=cfg['momentum'])
    optimizer_B = torch.optim.SGD(params_list_B, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'], momentum=cfg['momentum'])


    Scaler_A = amp.GradScaler()
    Scaler_B = amp.GradScaler()

    #optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay']
    scheduler_A = LambdaLR(optimizer_A, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)
    scheduler_B = LambdaLR(optimizer_B, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)

    # model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    # if args.distributed:
    #   model = torch.nn.parallel.DistributedDataParallel(model)

    # class weight 计算
    if hasattr(trainset, 'class_weight'):
        print('using classweight in dataset')
        class_weight = trainset.class_weight
    else:
        classweight = ClassWeight(cfg['class_weight'])
        class_weight = classweight.get_weight(train_loader, cfg['n_classes'])

    class_weight = torch.from_numpy(class_weight).float().to(device)

    # class_weight[cfg['id_unlabel']] = 0


    # 损失函数 & 类别权重平衡 & 训练时ignore unlabel
    # criterion_kdbase = CriterionKD().to(device)
    # criterion_kl = DistillKL().to(device)
    # criterion_skd = CriterionStructuralKD().to(device)
    # criterion_at = CriterionAT().to(device)
    criterion = MscCrossEntropyLoss(weight=class_weight).to(device)
    # criterion_AT = AT(p=2).to(device)
    # criterion_CriterionKD = CriterionKD(temperature=2).to(device)
    # criterion_pw = CriterionPairWiseforWholeFeatAfterPool(scale=0.5)
    criterion_kld = KLDLoss(tau=1).to(device)
    criterion_glo1 = global_loss(
            student_channels=64,
            teacher_channels=96,
            gamma=0.001,
            lambda_=0.00005
        ).to(device)
    criterion_glo2 = global_loss(
            student_channels=128,
            teacher_channels=192,
            gamma=0.001,
            lambda_=0.00005
        ).to(device)
    criterion_glo3 = global_loss(
            student_channels=320,
            teacher_channels=384,
            gamma=0.001,
            lambda_=0.00005
        ).to(device)
    criterion_glo4 = global_loss(
            student_channels=512,
            teacher_channels=768,
            gamma=0.001,
            lambda_=0.00005
        ).to(device)
    # criterion_kld = nn.KLDivLoss(reduction="batchmean").to(device)
   # criterion = MscCrossEntropyLoss().to(device)
    # criterion = lovaszSoftmax(ignore_index=0).to(device)

    # 指标 包含unlabel
    train_loss_meter_A = averageMeter()
    train_loss_meter_B = averageMeter()
    train_loss_meter = averageMeter()

    val_loss_meter_A = averageMeter()
    val_loss_meter_B = averageMeter()

    # running_metrics_val_A = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    # running_metrics_val_B = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    #
    running_metrics_val_A = runningScore(cfg['n_classes'], ignore_index=None)
    running_metrics_val_B = runningScore(cfg['n_classes'], ignore_index=None)
    # 每个epoch迭代循环

    flag = True #为了先保存一次模型做的判断
    #设置一个初始miou
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

                # predict = model(torch.cat((image.unsqueeze(2), depth.unsqueeze(2)), dim=2))

                # predict = model(torch.squeeze(image, dim=1), depth)
                # predict = model(image, depth)  ########
                predict_A = model_A(image, depth)  ########
                # predict_np = predict[0].cpu().detach().numpy()
                # label_np =label.cpu().detach().numpy()
                # mi = mutual_info_score(predict_np, label_np)
                # predict = model(image)
                # print('predict', predict.shape)
                # loss = criterion(predict[0], label) + criterion(predict[1], label) + criterion(predict[2], label) +\
                # criterion(predict[3], label)
                # original_loss = criterion(predict, label)  #######################1
                # loss =  original_loss + 0.1 * loss_dhfr #######################1
                with torch.no_grad():
                    predict_B = model_B(image, depth)
                loss_A = criterion(predict_A[0], label)  #######################1
                # loss = criterion(predict[0], label)  #######################1
                # lossA_B = criterion_kld(predict_A[0], predict_B[0].detach())
                #
                lossA_B = criterion_kld(predict_A[0],predict_B[0].detach())

                lossFT_A_B1 = criterion_glo1(predict_A[1], predict_B[1].clone())
                lossFT_A_B2 = criterion_glo2(predict_A[2], predict_B[2].clone())
                lossFT_A_B3 = criterion_glo3(predict_A[3], predict_B[3].clone())
                lossFT_A_B4 = criterion_glo4(predict_A[4], predict_B[4].clone())

                # lossFT_A_B = criterion_AT(predict_A[1],predict_B[6].detach())
                # total_A = loss_A
                total_A = loss_A + lossA_B #+ lossFT_A_B1 + lossFT_A_B2 +lossFT_A_B3+lossFT_A_B4

            ####################################################

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            Scaler_A.scale(total_A).backward()
            Scaler_A.step(optimizer_A)
            Scaler_A.update()
            # optimizer_A.zero_grad()  # 梯度清零
            # optimizer.step()

            # if args.distributed:
            #     reduced_loss = total_A.clone()
            #     dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            #     reduced_loss /= args.world_size
            # else:
            #     reduced_loss = total_A
            # train_loss_meter_A.update(reduced_loss.item())
            train_loss_meter.update(total_A.item())

            with amp.autocast():
                predict_B = model_B(image, depth)  ########
                with torch.no_grad():
                    # predict_A_detach0 = predict_A[0].detach()
                    predict_A = model_A(image, depth)
                    # predict_A_detach1 = predict_A[1].detach()

                loss_B = criterion(predict_B[0], label)  #######################1
                # lossB_A = criterion_kld(predict_B[0], predict_A[0].detach())
                lossB_A = criterion_kld(predict_B[0], predict_A[0].detach())
                lossFT_B_A1 = criterion_glo1(predict_A[1].clone(), predict_B[1])
                lossFT_B_A2 = criterion_glo2(predict_A[2].clone(), predict_B[2])
                lossFT_B_A3 = criterion_glo3(predict_A[3].clone(), predict_B[3])
                lossFT_B_A4 = criterion_glo4(predict_A[4].clone(), predict_B[4])
                # lossFT_B_A = criterion_AT(predict_B[6],predict_A_detach1)
                # total_B = loss_B
                total_B = loss_B + lossB_A #+ lossFT_B_A1 + lossFT_B_A2 + lossFT_B_A3 + lossFT_B_A4

            Scaler_B.scale(total_B).backward()
            Scaler_B.step(optimizer_B)
            Scaler_B.update()
            # optimizer_B.zero_grad()  # 梯度清零
            # optimizer.step()
            train_loss_meter.update(total_B.item())
            # train_loss_meter_B.update(total_B.item())

            if args.distributed:
                reduced_loss_A = total_A.clone()
                dist.all_reduce(reduced_loss_A, op=dist.ReduceOp.SUM)
                reduced_loss_A /= args.world_size
                # train_loss_meter_A.update(reduced_loss_A.item())
                train_loss_meter.update(reduced_loss_A.item())

                reduced_loss_B = total_B.clone()
                dist.all_reduce(reduced_loss_B, op=dist.ReduceOp.SUM)
                reduced_loss_B /= args.world_size
                # train_loss_meter_B.update(reduced_loss_B.item())
                train_loss_meter.update(reduced_loss_B.item())
            else:
                # train_loss_meter_A.update(total_A.item())
                # train_loss_meter_B.update(total_B.item())
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

                # predict = model(torch.cat((image.unsqueeze(2), depth.unsqueeze(2)), dim=2))
                # loss = criterion(predict, label)
                # predict = model(image, depth)
                predict_A= model_A(image, depth)
                predict_B = model_B(image, depth)

                loss_A = criterion(predict_A[0], label)         #############################2
                # loss = criterion(predict[0], label)         #############################2
                loss_B = criterion(predict_B[0], label)  #############################2

                val_loss_meter_A.update(loss_A.item())
                val_loss_meter_B.update(loss_B.item())

                # predict = predict[0].cpu().numpy()  # [1, h, w]     #############################3
                # print('predict',predict.shape)
                predict_A = predict_A[0].max(1)[1].cpu().numpy()  # [1, h, w]
                predict_B = predict_B[0].max(1)[1].cpu().numpy()  # [1, h, w]

                # predict = predict[0].max(1)[1].cpu().numpy()  # [1, h, w]
                label = label.cpu().numpy()

            ###################edit end#########################
                running_metrics_val_A.update(label, predict_A)
                running_metrics_val_B.update(label, predict_B)

                # if isinstance(label,np.ndarray):
                #     label = torch.from_numpy(label).to(device)

                # predict_B = model_B(image, depth)
                #
                # loss_B = criterion(predict_B, label)  #############################2
                # # loss = criterion(predict[0], label)         #############################2
                #
                # val_loss_meter_B.update(loss_B.item())
                # predict_B = predict_B.max(1)[1].cpu().numpy()  # [1, h, w]
                # # predict = predict[0].max(1)[1].cpu().numpy()  # [1, h, w]
                # label = label.cpu().numpy()
                # running_metrics_val_B.update(label, predict_B)

        if args.local_rank == 0:
            logger.info(
                f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] '
                f'  Model A train/val loss={train_loss_meter.avg:.5f}/{val_loss_meter_A.avg:.5f}, '
                # f', PA={running_metrics_val_A.get_scores()[0]["pixel_acc: "]:.3f}'
                # f', CA={running_metrics_val_A.get_scores()[0]["class_acc: "]:.3f}'
                f', mAcc={running_metrics_val_A.get_scores()[0]["mAcc: "]:.3f}'
                f', miou={running_metrics_val_A.get_scores()[0]["mIou: "]:.3f}'
                # f', mi={running_metrics_val.get_scores()[0]["mi: "]:.3f}'
                f', best_miou={miou_A:.3f}'
                f'  Model B train/val loss={train_loss_meter.avg:.5f}/{val_loss_meter_B.avg:.5f}, '
                # f', PA={running_metrics_val_B.get_scores()[0]["pixel_acc: "]:.3f}'
                # f', CA={running_metrics_val_B.get_scores()[0]["class_acc: "]:.3f}'
                f', mAcc={running_metrics_val_B.get_scores()[0]["mAcc: "]:.3f}'
                f', miou={running_metrics_val_B.get_scores()[0]["mIou: "]:.3f}'
                # f', mi={running_metrics_val.get_scores()[0]["mi: "]:.3f}'
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

            if newmiou_A > 0.730:
                state = model_A.module.state_dict() if hasattr(model_A, 'module') else model_A.state_dict()
                torch.save(state, os.path.join(logdir, f'{newmiou_A:.3f}.A'))

            if newmiou_B > 0.730:
                state = model_B.module.state_dict() if hasattr(model_B, 'module') else model_B.state_dict()
                torch.save(state, os.path.join(logdir, f'{newmiou_B:.3f}.B'))

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