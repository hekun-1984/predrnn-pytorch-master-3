__author__ = 'yunbo'

import os
import shutil
import argparse#解析器 配置信息
import numpy as np
import math
from core.data_provider import datasets_factory
from core.models.model_factory import Model
from core.utils import preprocess
import core.trainer as trainer

# -----------------------------------------------------------------------------
  #argparse 模块可以让人轻松编写用户友好的命令行接口。程序定义它需要的参数，然后 argparse 将弄清如何从 sys.argv 解析出那些参数。
  # argparse 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。
parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN') #1.创建解析器对象

# training/test
parser.add_argument('--is_training', type=int, default=1) #2.添加参数
parser.add_argument('--device', type=str, default='cpu:0')

# data
parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--train_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-train.npz')
parser.add_argument('--valid_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-valid.npz')
parser.add_argument('--save_dir', type=str, default='checkpoints/mnist_predrnn')
parser.add_argument('--gen_frm_dir', type=str, default='results/mnist_predrnn')
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_width', type=int, default=64)
parser.add_argument('--img_channel', type=int, default=1)

# model
parser.add_argument('--model_name', type=str, default='predrnn')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--layer_norm', type=int, default=1)
parser.add_argument('--decouple_beta', type=float, default=0.1)

# reverse scheduled sampling
parser.add_argument('--reverse_scheduled_sampling', type=int, default=0)
parser.add_argument('--r_sampling_step_1', type=float, default=25000)
parser.add_argument('--r_sampling_step_2', type=int, default=50000)
parser.add_argument('--r_exp_alpha', type=int, default=5000)
# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

# optimization
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_iterations', type=int, default=80000)
parser.add_argument('--display_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=5000)
parser.add_argument('--snapshot_interval', type=int, default=5000)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--n_gpu', type=int, default=1)

# visualization of memory decoupling
parser.add_argument('--visual', type=int, default=0)
parser.add_argument('--visual_path', type=str, default='./decoupling_visual')

# action-based predrnn
parser.add_argument('--injection_action', type=str, default='concat')
parser.add_argument('--conv_on_input', type=int, default=0, help='conv on input')
parser.add_argument('--res_on_conv', type=int, default=0, help='res on conv')
parser.add_argument('--num_action_ch', type=int, default=4, help='num action ch')

args = parser.parse_args()#3.解析参数args
print(args)


def reserve_schedule_sampling_exp(itr): ##反计划采样
    if itr < args.r_sampling_step_1:
        r_eta = 0.5
    elif itr < args.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < args.r_sampling_step_1:
        eta = 0.5
    elif itr < args.r_sampling_step_2:
        eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample(
        (args.batch_size, args.input_length - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)

    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))

    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - 2):
            if j < args.input_length - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (args.input_length - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - 2,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return real_input_flag


def schedule_sampling(eta, itr): ##计划采样
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag


def train_wrapper(model):##训练封装
    if args.pretrained_model:#'--pretrained_model', type=str, default=''
        model.load(args.pretrained_model)#加载模型
    ## load data
       #def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size,
       #                  img_width, seq_length, injection_action, is_training=True):
    train_input_handle, test_input_handle = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
        seq_length=args.total_length, injection_action=args.injection_action, is_training=True)

    eta = args.sampling_start_value#'--sampling_start_value', type=float, default=1.0 采样开始值

    for itr in range(1, args.max_iterations + 1):#'--max_iterations', type=int, default=80000迭代80000次
        if train_input_handle.no_batch_left():#train_input_handle变量=datasets_factory.data_provider()方法
            train_input_handle.begin(do_shuffle=True) #begin()函数功能？？？？
        ims = train_input_handle.get_batch()
        ims = preprocess.reshape_patch(ims, args.patch_size)

        if args.reverse_scheduled_sampling == 1:#反计划采样
            real_input_flag = reserve_schedule_sampling_exp(itr)
        else:
            eta, real_input_flag = schedule_sampling(eta, itr)#计划采样

        trainer.train(model, ims, real_input_flag, args, itr) #训练，见trainer类train方法

        if itr % args.snapshot_interval == 0: #'--snapshot_interval', type=int, default=5000
            model.save(itr) #每迭代5000次保存一次模型

        if itr % args.test_interval == 0: #'--test_interval', type=int, default=5000
            trainer.test(model, test_input_handle, args, itr) #每5000次迭代用测试数据test一次

        train_input_handle.next() #跳到下一个输入数据


def test_wrapper(model):##测试封装
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
        seq_length=args.total_length, injection_action=args.injection_action, is_training=False)
    trainer.test(model, test_input_handle, args, 'test_result')


if os.path.exists(args.save_dir):##创建checkpoints/mnist_predrnn文件夹并保存训练信息此路径下
    shutil.rmtree(args.save_dir)#递归删除整个文件夹下所有文件，包括此文件夹；
os.makedirs(args.save_dir)#创建文件夹

if os.path.exists(args.gen_frm_dir):##创建results/mnist_predrnn文件夹并保存图片结果到此路径下
    shutil.rmtree(args.gen_frm_dir)
os.makedirs(args.gen_frm_dir)

print('Initializing models')

model = Model(args)##模型工厂用args配置信息确定详细模型model

if args.is_training:#is_training默认值1，
    train_wrapper(model)#训练模型model
else:
    test_wrapper(model)#测试模型model
