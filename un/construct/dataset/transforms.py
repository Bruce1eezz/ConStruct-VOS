import numpy as np
import torch


class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            label = torch.from_numpy(pic).long()
        elif pic.mode == '1':
            label = torch.from_numpy(np.array(pic, np.uint8, copy=False)).long().view(1, pic.size[1], pic.size[0])
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            if pic.mode == 'LA':
                label = label.view(pic.size[1], pic.size[0], 2)
                label = label.transpose(0, 1).transpose(0, 2).contiguous().long()[0]
                label = label.view(1, label.size(0), label.size(1))
            else:
                label = label.view(pic.size[1], pic.size[0], -1)
                label = label.transpose(0, 1).transpose(0, 2).contiguous().long()
        label[label == 255] = 0
        return label

'''
# 导入NumPy库，用于处理数组
import numpy as np
# 导入PyTorch库，用于创建张量
import torch

# 定义一个名为 LabelToLongTensor 的类
# 此类是一个自定义的图像变换，用于将输入的图片或数据转为类型为 long 的 PyTorch 张量
class LabelToLongTensor(object):
    # 定义类的 __call__ 方法，使类实例可以像函数一样被调用
    def __call__(self, pic):
        # 如果输入的图片是NumPy数组
        if isinstance(pic, np.ndarray):
            # 将NumPy数组转换为PyTorch张量，并将数据类型更改为 long（长整型）
            label = torch.from_numpy(pic).long()
        # 如果输入图片是 PIL 图像且其模式为 '1'（即二值化图像）
        elif pic.mode == '1':
            # 将PIL图像转换为NumPy数组，再将其转换为 PyTorch 长整型张量
            # 并且调整张量的形状为 (1, 高度, 宽度)
            label = torch.from_numpy(np.array(pic, np.uint8, copy=False)).long().view(1, pic.size[1], pic.size[0])
        # 如果输入图片是其他模式，如 'LA' 或一般RGB图片
        else:
            # 将图片缓冲区的数据转换为PyTorch字节类型张量
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            # 如果图片模式是 'LA'（灰度图像带alpha通道）
            if pic.mode == 'LA':
                # 将张量调整为形状 (高度, 宽度, 2)，表示灰度通道和alpha通道
                label = label.view(pic.size[1], pic.size[0], 2)
                # 调整张量的通道顺序，以确保适合PyTorch的格式
                label = label.transpose(0, 1).transpose(0, 2).contiguous().long()[0]
                # 重新调整张量形状为 (1, 高度, 宽度)
                label = label.view(1, label.size(0), label.size(1))
            # 如果图片是其他模式
            else:
                # 根据图片的尺寸，将缓冲区数据转换为 (高度, 宽度, 通道数) 的张量
                label = label.view(pic.size[1], pic.size[0], -1)
                # 调整通道的顺序，使其适合PyTorch的格式：(通道, 高度, 宽度)
                label = label.transpose(0, 1).transpose(0, 2).contiguous().long()
        # 后处理：将值为 255 的像素替换为 0
        # 通常在分割任务中，255代表“忽略”或“背景”像素，这里将其替换为0是为了方便模型处理
        label[label == 255] = 0
        # 返回处理后的张量
        return label

'''