# # coding: utf-8
# import os
#
#
# def createFilelist(images_path, text_save_path):
#     # 打开图片列表清单txt文件
#     file_name = open(text_save_path, "w")
#     # 查看文件夹下的图片
#     images_name = os.listdir(images_path)
#     # 遍历所有文件
#     for eachname in images_name:
#         # 按照需要的格式写入目标txt文件
#         file_name.write(images_path + '/' + eachname + '\n')
#
#     print('生成txt成功！')
#
#     file_name.close()
#
#
# if __name__ == "__main__":
#     # txt文件存放目录
#     txt_path = './sunrgbd/train'
#     # 图片存放目录
#     images_path = './sunrgbd/train/image'
#     # 生成图片列表文件命名
#     txt_name = 'train.txt'
#     if not os.path.exists(txt_name):
#         os.mkdir(txt_name)
#     # 生成图片列表文件的保存目录
#     text_save_path = txt_path + '/' + txt_name
#     # 生成txt文件
#     createFilelist(images_path, text_save_path)
import torch.utils.model_zoo as model_zoo
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
pretrain_dict = model_zoo.load_url(model_urls['resnet50'])
for k, v in pretrain_dict.items():
    print(k,"\n",v)