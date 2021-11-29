#定义训练过程模板
import os.path
import datetime
import cv2
import numpy as np

#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim  #处理图像的包，使用原生numpy数组作为图像对象 结构相似性指数（SSIM）函数

from core.utils import preprocess, metrics
import lpips
import torch

loss_fn_alex = lpips.LPIPS(net='alex') #LPIPS（一种图像相似性度量标准，小好，可选vgg、alex) best forward scores


def train(model, ims, real_input_flag, configs, itr): ##训练（模型，ims为数组？？，真实输入帧，配置，第几次迭代）
    cost = model.train(ims, real_input_flag) #启用batch normalization和drop out。
    if configs.reverse_input: #反计划采样和计划采样一起用，看配置信息种的reverse_input
        ims_rev = np.flip(ims, axis=1).copy() #flip将数组在左右方向上翻转；copy()属于深拷贝,拷贝前的地址和拷贝后的地址不一样
        cost += model.train(ims_rev, real_input_flag) #？？？
        cost = cost / 2 #翻转取平均，作用？？？   cost为loss值，考虑到了正反计划采样

    if itr % configs.display_interval == 0: #训练过程概中迭代100次输出一次（当前时间，迭代次数itr，loss值cost代价)
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
        print('training loss: ' + str(cost))


def test(model, test_input_handle, configs, itr): ##测试（模型，测试输入，配置，第几次迭代）
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    test_input_handle.begin(do_shuffle=False) #begin函数作用？？？
    res_path = os.path.join(configs.gen_frm_dir, str(itr)) #连接两个或更多的路径名组件，
    os.mkdir(res_path)#创建目录
    avg_mse = 0#
    batch_id = 0#批次id
    img_mse, ssim, psnr = [], [], [] #，图像计算常用指标（img_mse？？ ， 结构相似性指数，psnr？？）
    lp = []#lp?

    for i in range(configs.total_length - configs.input_length):#作用？--total_length', type=int, default=20;--input_length', type=int, default=10
        img_mse.append(0)#每批全添0
        ssim.append(0)
        psnr.append(0)
        lp.append(0)

    # reverse schedule sampling 反计划采样
    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1#作用？
    else:
        mask_input = configs.input_length#默认值10

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - mask_input - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))#real_input_flag全0数组，作用？

    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, :configs.input_length - 1, :, :] = 1.0#？

    while (test_input_handle.no_batch_left() == False):
        batch_id = batch_id + 1
        test_ims = test_input_handle.get_batch()
        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)
        test_ims = test_ims[:, :, :, :, :configs.img_channel]
        img_gen = model.test(test_dat, real_input_flag)

        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length 
        img_out = img_gen[:, -output_length:]

        # MSE per frame
        for i in range(output_length):
            x = test_ims[:, i + configs.input_length, :, :, :]
            gx = img_out[:, i, :, :, :]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse
            # cal lpips
            img_x = np.zeros([configs.batch_size, 3, configs.img_width, configs.img_width])
            if configs.img_channel == 3:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 1]
                img_x[:, 2, :, :] = x[:, :, :, 2]
            else:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 0]
                img_x[:, 2, :, :] = x[:, :, :, 0]
            img_x = torch.FloatTensor(img_x)
            img_gx = np.zeros([configs.batch_size, 3, configs.img_width, configs.img_width])
            if configs.img_channel == 3:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 1]
                img_gx[:, 2, :, :] = gx[:, :, :, 2]
            else:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 0]
                img_gx[:, 2, :, :] = gx[:, :, :, 0]
            img_gx = torch.FloatTensor(img_gx)
            lp_loss = loss_fn_alex(img_x, img_gx)
            lp[i] += torch.mean(lp_loss).item()

            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)

            psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
            for b in range(configs.batch_size):
                score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True, multichannel=True)
                ssim[i] += score

        # save prediction examples
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            for i in range(configs.total_length):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0, i, :, :, :] * 255)
                cv2.imwrite(file_name, img_gt)
            for i in range(output_length):
                name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_out[0, i, :, :, :]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)
        test_input_handle.next()

    avg_mse = avg_mse / (batch_id * configs.batch_size)
    print('mse per seq: ' + str(avg_mse))
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size))

    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])

    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])

    lp = np.asarray(lp, dtype=np.float32) / batch_id
    print('lpips per frame: ' + str(np.mean(lp)))
    for i in range(configs.total_length - configs.input_length):
        print(lp[i])
