#模型工厂
import os
import torch
from torch.optim import Adam
from core.models import  predrnn_v2
class Model(object):
    def __init__(self, configs):#初始化信息，构造器
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]#单个隐藏层
        self.num_layers = len(self.num_hidden)#隐藏层块

        Network =predrnn_v2.RNN #模型框架RNN
        self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)#模型框架RNN->网络模型（隐藏层块,单个隐藏层,配置信息）
        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)#优化器：Adam（网络参数,自定义学习率）

    def save(self, itr):##保存模型
        stats = {}#定义字典（'net_param'：模型信息）
        stats['net_param'] = self.network.state_dict()#返回包含模型整个状态的字典。
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt'+'-'+str(itr))#设检查点（保存文件夹路径+model.ckpt+第几次迭代次数）
        torch.save(stats, checkpoint_path)#保存模型信息（模型状态，保存的文件路径） torch.save：保存一个序列化（serialized）的目标到磁盘
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):##从保存的文件路径处加载模型信息
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path)#从保存的文件路径处加载模型信息  torch.load：解压来反序列化文件到对应存储设备上
        self.network.load_state_dict(stats['net_param'])#参数和缓冲区从：attr:`state_dict`复制到此模块及其子模块 加载模型参数



    def train(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()
        next_frames, loss = self.network(frames_tensor, mask_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames, _ = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()