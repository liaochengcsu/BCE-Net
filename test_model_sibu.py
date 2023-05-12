import os
import time
import cv2
import random
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.seg_metric import Pixel_A
from dataset.cd_dataload_512 import Mydataset
from Testmodel.CDResSIBU import Baseline34
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    seed_torch(seed=1024)
    # val_path :the base filename of the images and the corresponding labels of the test set, recorded in a csv file'
    val_path = r'./dataset/test_sibu.csv'
    # save_pred_path :the path to save the predicted results, including new, remove, and existing building for each samples'
    save_pred_path = r'./Results/res-sibu'
    # trained_model :the path to the trained model.pth'
    trained_model = r'./weights/checkpoint-best-sibu.pth'


    batch_size = 4
    normMean = [0, 0, 0]
    normStd = [1, 1, 1]
    normTransfrom = transforms.Normalize(normMean, normStd)
    transform = transforms.Compose([
        transforms.ToTensor(),
        normTransfrom
    ])
    val_data = Mydataset(path=val_path, transform=transform, augment=False)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
    print("valid data iter:", len(val_loader))

    net = Baseline34(pretrained=True).cuda()
    net = torch.nn.DataParallel(net)
    net.load_state_dict(
        torch.load(trained_model)['state_dict'])
    # from thop import profile
    # input = torch.randn(1, 3, 256, 256).cuda()
    # label = torch.randn(1, 256, 256).cuda()
    # flops, params = profile(net, inputs=(input, label))
    # print("total parameters is: %.2fM" % (params / 1e6))
    # print("total flops is: %.2fG" % (flops / 1e9))

    if not os.path.exists(save_pred_path):
        os.mkdir(save_pred_path)
    savebase = 'sibu'

    tmp_save_name_n = os.path.join(save_pred_path, "new_" + savebase)
    tmp_save_name_m = os.path.join(save_pred_path, "mov_" + savebase)
    tmp_save_name_b = os.path.join(save_pred_path, "bui_" + savebase)
    tmp_save_name_c = os.path.join(save_pred_path, "chg_" + savebase)

    if not os.path.exists(tmp_save_name_m):
        os.mkdir(tmp_save_name_m)
    if not os.path.exists(tmp_save_name_n):
        os.mkdir(tmp_save_name_n)
    if not os.path.exists(tmp_save_name_b):
        os.mkdir(tmp_save_name_b)
    if not os.path.exists(tmp_save_name_c):
        os.mkdir(tmp_save_name_c)

    start_time = time.time()
    net.eval()

    TP = 0
    FP = 0
    FN = 0

    nTP = 0
    nFP = 0
    nFN = 0

    anTP = 0
    anFP = 0
    anFN = 0

    predict_time = 0
    for i, data in enumerate(val_loader):
        # labels_o{1,3},labels_n{2},labels_m{3},labels{1,2}
        inputs, labels_o, labels_n, labels_m, labels, img_name = data
        change = labels_n + labels_m
        change = Variable(change.cuda())
        inputs = Variable(inputs.cuda())
        labels_o = Variable(labels_o.float().cuda())
        labels_n = Variable(labels_n.float().cuda())
        labels = Variable(labels.float().cuda())

        with torch.no_grad():
            start_time1 = time.time()
            predicts_b, predicts_mov, predicts_new, _, _ = net.forward(inputs, labels_o)
            start_time2 = time.time()
            predict_time+=(start_time2-start_time1)
            predictsn = torch.sigmoid(predicts_new)
            predictsm = torch.sigmoid(predicts_mov)
            predictsb = torch.sigmoid(predicts_b)

            predictsn[predictsn < 0.5] = 0
            predictsn[predictsn >= 0.5] = 1
            resultn = np.squeeze(predictsn)

            # predictsm[predictsm < 0.5] = 0
            # predictsm[predictsm >= 0.5] = 1
            resultm = np.squeeze(predictsm)

            predictsb[predictsb < 0.5] = 0
            predictsb[predictsb >= 0.5] = 1
            resultb = np.squeeze(predictsb)

            if len(img_name) == 1:
                labels_o[0][labels_o[0] > 0] = 1
                movefeat_i = resultm.cpu().detach().numpy() * labels_o[0].cpu().detach().numpy()
                out_mov = np.zeros((512, 512), dtype=np.uint8)
                old_contours, _ = cv2.findContours(np.array(255 * labels_o[0].cpu().detach().numpy(), dtype=np.uint8),
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for old_i in old_contours:
                    chg_coord = np.array(old_i.transpose(1, 0, 2))
                    changi = np.zeros((512, 512), dtype=np.uint8)
                    changi = cv2.fillPoly(changi, chg_coord, 1)

                    if np.sum(changi * movefeat_i) / np.sum(changi) > 0.5:
                        out_mov = cv2.fillPoly(out_mov, chg_coord, 1)

                cv2.imwrite(os.path.join(tmp_save_name_m, img_name[0].replace('tif', 'png')), out_mov * 255)
                tp, fp, fn = Pixel_A(out_mov, labels_m[0].cpu().detach().numpy())

                # cv2.imwrite(os.path.join(tmp_save_name_m, img_name[0].replace('tif', 'png')), resultm[0].cpu().detach().numpy() * 255)
                # tp, fp, fn = Pixel_A(resultm[0].cpu().detach().numpy(), labels_m[0].cpu().detach().numpy())

                cv2.imwrite(os.path.join(tmp_save_name_n, img_name[0].replace('tif', 'png')), resultn[0].cpu().detach().numpy() * 255)
                ntp, nfp, nfn = Pixel_A(resultn.cpu().detach().numpy(), labels_n[0].cpu().detach().numpy())
                cv2.imwrite(os.path.join(tmp_save_name_c, img_name[0].replace('tif', 'png')),
                            resultm[0].cpu().detach().numpy() * 255 + resultn[0].cpu().detach().numpy() * 255)
                cv2.imwrite(os.path.join(tmp_save_name_b, img_name[0].replace('tif', 'png')), resultb[0].cpu().detach().numpy() * 255)
                TP += tp
                FP += fp
                FN += fn
                nTP += ntp
                nFP += nfp
                nFN += nfn
                break
            for index in range(len(img_name)):
                labels_o[index][labels_o[index] > 0] = 1
                movefeat_i = resultm[index].cpu().detach().numpy() * labels_o[index].cpu().detach().numpy()
                out_mov = np.zeros((512, 512), dtype=np.uint8)
                old_contours, _ = cv2.findContours(
                    np.array(255 * labels_o[index].cpu().detach().numpy(), dtype=np.uint8), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
                for old_i in old_contours:
                    chg_coord = np.array(old_i.transpose(1, 0, 2))
                    changi = np.zeros((512, 512), dtype=np.uint8)
                    changi = cv2.fillPoly(changi, chg_coord, 1)
                    if np.sum(changi * movefeat_i) / np.sum(changi) > 0.5:
                        out_mov = cv2.fillPoly(out_mov, chg_coord, 1)

                cv2.imwrite(os.path.join(tmp_save_name_m, img_name[index].replace('tif', 'png')), out_mov * 255)
                tp, fp, fn = Pixel_A(out_mov, labels_m[index].cpu().detach().numpy())
                # cv2.imwrite(os.path.join(tmp_save_name_m, img_name[index].replace('tif', 'png')), resultm[index].cpu().detach().numpy() * 255)
                # tp, fp, fn = Pixel_A(resultm[index].cpu().detach().numpy(), labels_m[index].cpu().detach().numpy())

                cv2.imwrite(os.path.join(tmp_save_name_n, img_name[index].replace('tif', 'png')), resultn[index].cpu().detach().numpy() * 255)
                ntp, nfp, nfn = Pixel_A(resultn[index].cpu().detach().numpy(), labels_n[index].cpu().detach().numpy())

                cv2.imwrite(os.path.join(tmp_save_name_c, img_name[index].replace('tif', 'png')), resultm[index].cpu().detach().numpy() * 255+resultn[index].cpu().detach().numpy() * 255)

                cv2.imwrite(os.path.join(tmp_save_name_b, img_name[index].replace('tif', 'png')),resultb[index].cpu().detach().numpy() * 255)

                TP += tp
                FP += fp
                FN += fn
                nTP += ntp
                nFP += nfp
                nFN += nfn


    # val_p = TP / (TP + FP + 1)
    # val_r = TP / (TP + FN + 1)
    # val_iou = val_p * val_r / (val_p + val_r - val_p * val_r + 0.01)
    # val_acc = 2 * val_p * val_r / (val_p + val_r + 0.01)
    # print('#######################')
    # print("mvalid F1:", val_acc)
    # print("mvalid IoU:", val_iou)
    # # print("mPrecision:", val_p)
    # # print("mRecall:", val_r)
    # print('------------------')
    #
    # nval_p = nTP / (nTP + nFP)
    # nval_r = nTP / (nTP + nFN)
    # nval_iou = nval_p * nval_r / (nval_p + nval_r - nval_p * nval_r)
    # nval_acc = 2 * nval_p * nval_r / (nval_p + nval_r)
    # print("nvalid F1:", nval_acc)
    # print("nvalid IoU:", nval_iou)
    # # print("nPrecision:", nval_p)
    # # print("nRecall:", nval_r)
    # print('------------------')
    #
    anTP = nTP + TP
    anFN = nFN + FN
    anFP = nFP + FP
    anval_p = anTP / (anTP + anFP)
    anval_r = anTP / (anTP + anFN)
    anval_iou = anval_p * anval_r / (anval_p + anval_r - anval_p * anval_r)
    anval_acc = 2 * anval_p * anval_r / (anval_p + anval_r)
    print("M_valid F1:", anval_acc)
    print("M_valid IoU:", anval_iou)
    # # print("nPrecision:", anval_p)
    # # print("nRecall:", anval_r)
    print("predict time:", predict_time)
    print("total time:", time.time()-start_time)