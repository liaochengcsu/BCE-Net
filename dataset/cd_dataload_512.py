# # # import os
# # # import cv2
# # # import numpy as np
# # # import pandas as pd
# # # from PIL import Image, ImageEnhance
# # # from albumentations import Compose, MotionBlur
# # # from torch.utils.data import Dataset
# # #
# # #
# # # def random_color_jitter(cv_img, saturation_range, brightness_range, contrast_range, u=0.5):
# # #     def saturation_jitter(cv_img, jitter_range):
# # #         """
# # #         调节图像饱和度
# # #         Args:
# # #             cv_img(numpy.ndarray): 输入图像
# # #             jitter_range(float): 调节程度，0-1
# # #         Returns:
# # #             饱和度调整后的图像
# # #         """
# # #         greyMat = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
# # #         greyMat = greyMat[:, :, None] * np.ones(3, dtype=int)[None, None, :]
# # #         cv_img = cv_img.astype(np.float32)
# # #         cv_img = cv_img * (1 - jitter_range) + jitter_range * greyMat
# # #         cv_img = np.where(cv_img > 255, 255, cv_img)
# # #         cv_img = cv_img.astype(np.uint8)
# # #         return cv_img
# # #
# # #     def brightness_jitter(cv_img, jitter_range):
# # #         """
# # #         调节图像亮度
# # #         Args:
# # #             cv_img(numpy.ndarray): 输入图像
# # #             jitter_range(float): 调节程度，0-1
# # #         Returns:
# # #             亮度调整后的图像
# # #         """
# # #         cv_img = cv_img.astype(np.float32)
# # #         cv_img = cv_img * (1.0 - jitter_range)
# # #         cv_img = np.where(cv_img > 255, 255, cv_img)
# # #         cv_img = cv_img.astype(np.uint8)
# # #         return cv_img
# # #
# # #     def contrast_jitter(cv_img, jitter_range):
# # #         """
# # #         调节图像对比度
# # #         Args:
# # #             cv_img(numpy.ndarray): 输入图像
# # #             jitter_range(float): 调节程度，0-1
# # #         Returns:
# # #             对比度调整后的图像
# # #         """
# # #         greyMat = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
# # #         mean = np.mean(greyMat)
# # #         cv_img = cv_img.astype(np.float32)
# # #         cv_img = cv_img * (1 - jitter_range) + jitter_range * mean
# # #         cv_img = np.where(cv_img > 255, 255, cv_img)
# # #         cv_img = cv_img.astype(np.uint8)
# # #         return cv_img
# # #     """
# # #     图像亮度、饱和度、对比度调节，在调整范围内随机获得调节比例，并随机顺序叠加三种效果
# # #     Args:
# # #         cv_img(numpy.ndarray): 输入图像
# # #         saturation_range(float): 饱和对调节范围，0-1
# # #         brightness_range(float): 亮度调节范围，0-1
# # #         contrast_range(float): 对比度调节范围，0-1
# # #     Returns:
# # #         亮度、饱和度、对比度调整后图像
# # #     """
# # #     if np.random.random() < u:
# # #         saturation_ratio = np.random.uniform(-saturation_range, saturation_range)
# # #         brightness_ratio = np.random.uniform(-brightness_range, brightness_range)
# # #         contrast_ratio = np.random.uniform(-contrast_range, contrast_range)
# # #         order = [0, 1, 2]
# # #         np.random.shuffle(order)
# # #         for i in range(3):
# # #             if order[i] == 0:
# # #                 cv_img = saturation_jitter(cv_img, saturation_ratio)
# # #             if order[i] == 1:
# # #                 cv_img = brightness_jitter(cv_img, brightness_ratio)
# # #             if order[i] == 2:
# # #                 cv_img = contrast_jitter(cv_img, contrast_ratio)
# # #         return cv_img
# # #     return cv_img
# # #
# # #
# # # def randomHueSaturationValue(image1, image2, hue_shift_limit=(-180, 180),sat_shift_limit=(-255, 255),val_shift_limit=(-255, 255), u=0.5):
# # #     if np.random.random() < u:
# # #         image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
# # #         h, s, v = cv2.split(image1)
# # #         hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
# # #         hue_shift = np.uint8(hue_shift)
# # #         h += hue_shift
# # #         sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
# # #         s = cv2.add(s, sat_shift)
# # #         val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
# # #         v = cv2.add(v, val_shift)
# # #         image1 = cv2.merge((h, s, v))
# # #         image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2BGR)
# # #
# # #         image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
# # #         h, s, v = cv2.split(image2)
# # #         hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
# # #         hue_shift = np.uint8(hue_shift)
# # #         h += hue_shift
# # #         sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
# # #         s = cv2.add(s, sat_shift)
# # #         val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
# # #         v = cv2.add(v, val_shift)
# # #         image2 = cv2.merge((h, s, v))
# # #         image2 = cv2.cvtColor(image2, cv2.COLOR_HSV2BGR)
# # #     return image1, image2
# # #
# # #
# # # def randomShiftScaleRotate(image1, image2, mask,shift_limit=(-0.0, 0.0),scale_limit=(-0.0, 0.0),rotate_limit=(-0.0, 0.0),
# # #                            aspect_limit=(-0.0, 0.0),borderMode=cv2.BORDER_CONSTANT, u=0.5):
# # #     if np.random.random() < u:
# # #         height, width, channel = image1.shape
# # #
# # #         angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
# # #         scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
# # #         aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
# # #         sx = scale * aspect / (aspect ** 0.5)
# # #         sy = scale / (aspect ** 0.5)
# # #         dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
# # #         dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)
# # #
# # #         cc = np.math.cos(angle / 180 * np.math.pi) * sx
# # #         ss = np.math.sin(angle / 180 * np.math.pi) * sy
# # #         rotate_matrix = np.array([[cc, -ss], [ss, cc]])
# # #
# # #         box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
# # #         box1 = box0 - np.array([width / 2, height / 2])
# # #         box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])
# # #
# # #         box0 = box0.astype(np.float32)
# # #         box1 = box1.astype(np.float32)
# # #         mat = cv2.getPerspectiveTransform(box0, box1)
# # #         image1 = cv2.warpPerspective(image1, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
# # #                                     borderValue=(0, 0,0,))
# # #         image2 = cv2.warpPerspective(image2, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
# # #                                      borderValue=(0, 0, 0,))
# # #         mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
# # #                                    borderValue=(
# # #                                        0, 0,
# # #                                        0,))
# # #     return image1, image2, mask
# # #
# # #
# # # def randomHorizontalFlip(image1, image2, mask, u=0.5):
# # #     if np.random.random() < u:
# # #         image1 = cv2.flip(image1, 1)
# # #         image2 = cv2.flip(image2, 1)
# # #         mask = cv2.flip(mask, 1)
# # #     return image1,image2, mask
# # #
# # #
# # # def randomVerticleFlip(image1,image2, mask, u=0.5):
# # #     if np.random.random() < u:
# # #         image1 = cv2.flip(image1, 0)
# # #         image2 = cv2.flip(image2, 0)
# # #         mask = cv2.flip(mask, 0)
# # #     return image1, image2, mask
# # #
# # #
# # # def randomRotate90(image1,image2, mask, u=0.5):
# # #     if np.random.random() < u:
# # #         angle = np.random.randint(1,4)
# # #         for i in range(angle):
# # #             image1 = np.rot90(image1)
# # #             image2 = np.rot90(image2)
# # #             mask = np.rot90(mask)
# # #         return image1,image2, mask
# # #     return image1,image2, mask
# # #
# # #
# # # def resize(image1,image2, gt,insize, outsize):
# # #
# # #     x = np.random.randint(-192, 192)
# # #     y = np.random.randint(-192, 192)
# # #     if x < 0:
# # #         if y < 0:
# # #             image1 = image1[0:x, 0:y, :]
# # #             image2 = image2[0:x, 0:y, :]
# # #             gt = gt[0:x, 0:y]
# # #         else:
# # #             image1 = image1[0:x, y:insize, :]
# # #             image2 = image2[0:x, y:insize, :]
# # #             gt = gt[0:x, y:insize]
# # #     else:
# # #         if y < 0:
# # #             image1 = image1[x:insize, 0:y, :]
# # #             image2 = image2[x:insize, 0:y, :]
# # #             gt = gt[x:insize, 0:y]
# # #         else:
# # #             image1 = image1[x:insize, y:insize, :]
# # #             image2 = image2[x:insize, y:insize, :]
# # #             gt = gt[x:insize, y:insize]
# # #     image1 = cv2.resize(image1, (outsize, outsize), interpolation=cv2.INTER_LINEAR)
# # #     image2 = cv2.resize(image2, (outsize, outsize), interpolation=cv2.INTER_LINEAR)
# # #     gt = cv2.resize(gt, (outsize, outsize), interpolation=cv2.INTER_NEAREST)
# # #     # x = np.random.randint(0, 128)
# # #     # y = np.random.randint(0, 128)
# # #     # image = image[x:x+outsize, y:y+outsize, :]
# # #     # gt = gt[x:x+outsize, y:y+outsize]
# # #
# # #     return image1, image2,gt
# # #
# # #
# # # def motionblur(image, gt, blur=7, p=0.5):
# # #     aug = Compose([MotionBlur(blur_limit=blur, p=p)])
# # #     augmented = aug(image=image, mask=gt)
# # #     image_MotionBlur = augmented['image']
# # #     gt_MotionBlur = augmented['mask']
# # #     return image_MotionBlur, gt_MotionBlur
# # #
# # #
# # # def grade(img):
# # #     x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
# # #     y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
# # #     absX = cv2.convertScaleAbs(x)
# # #     absY = cv2.convertScaleAbs(y)
# # #     dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# # #     mi = np.min(dst)
# # #     ma = np.max(dst)
# # #     res = (dst - mi) / (0.000000001 + (ma - mi))
# # #     res[np.isnan(res)] = 0
# # #     return res
# # #
# # #
# # # class Mydataset(Dataset):
# # #     def __init__(self, path, augment=False, transform=None, target_transform=None):
# # #
# # #         self.aug = augment
# # #         self.file_path = os.path.dirname(path)
# # #         data = pd.read_csv(path)  # 获取csv表中的数据
# # #         imgs = []
# # #         for i in range(len(data)):
# # #             imgs.append((data.iloc[i, 0], data.iloc[i, 1]))
# # #         self.imgs = imgs
# # #         self.transform = transform
# # #         self.target_transform = target_transform
# # #
# # #     def __getitem__(self, item):
# # #         if self.aug == False:
# # #             fn, lab = self.imgs[item]
# # #             # fn = os.path.join(self.file_path, "image_A/" + fn)
# # #             # label = os.path.join(self.file_path, "image_A/" + lab)
# # #             fn1 = os.path.join(self.file_path, "image1/" + fn)
# # #             fn2 = os.path.join(self.file_path, "image2/" + fn)
# # #             label = os.path.join(self.file_path, "label/" + lab)
# # #
# # #             bgr_img = cv2.imread(fn1, -1)
# # #             img = Image.fromarray(bgr_img)
# # #             if self.transform is not None:
# # #                 img = self.transform(img)
# # #
# # #             bgr_img2 = cv2.imread(fn2, -1)
# # #             img2 = Image.fromarray(bgr_img2)
# # #             if self.transform is not None:
# # #                 img2 = self.transform(img2)
# # #
# # #             gt = cv2.imread(label, 0)
# # #             return img, img2, gt, lab
# # #
# # #
# # #         else:
# # #             # 进行数据增强
# # #             fn, lab = self.imgs[item]
# # #             # train with data.cvs
# # #             fn1 = os.path.join(self.file_path, "image1/" + fn)
# # #             fn2 = os.path.join(self.file_path, "image2/" + fn)
# # #             label = os.path.join(self.file_path, "label/" + lab)
# # #
# # #             gt = cv2.imread(label, 0)
# # #             # gt = gt[:,:,0]/255
# # #             image1 = cv2.imread(fn1, -1)
# # #             image2 = cv2.imread(fn2, -1)
# # #
# # #             # batch=[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
# # #             # sort1 = batch[np.random.randint(0,6)]
# # #             # sort2 = batch[np.random.randint(0, 6)]
# # #             # image1 = cv2.merge([image1[:, :, sort1[0]], image1[:, :, sort1[1]], image1[:, :, sort1[2]]])
# # #             # image2 = cv2.merge([image2[:, :, sort2[0]], image2[:, :, sort2[1]], image2[:, :, sort2[2]]])
# # #
# # #
# # #             # image, gt = motionblur(image, gt, blur=5, p=0.5)
# # #             # image = random_color_jitter(image, saturation_range=0.5, brightness_range=0.5, contrast_range=0.5, u=0.5)
# # #             # image = randomHueSaturationValue(image,hue_shift_limit=(-30, 30),sat_shift_limit=(-5, 5),val_shift_limit=(-15, 15))
# # #             # image, gt = randomShiftScaleRotate(image, gt, shift_limit=(-0.1, 0.1), scale_limit=(-0.0, 0.0),
# # #             #                                    aspect_limit=(-0.1, 0.1), rotate_limit=(-5, 5))
# # #             # image, gt = randomHorizontalFlip(image, gt, u=0.5)
# # #             # image, gt = randomVerticleFlip(image, gt, u=0.5)
# # #             # image, gt = randomRotate90(image, gt, u=0.5)
# # #             # image, gt = resize(image, gt, 1024, 640)
# # #
# # #             image1,image2 = randomHueSaturationValue(image1, image2,
# # #                                              hue_shift_limit=(-30, 30),
# # #                                              sat_shift_limit=(-35, 35),
# # #                                              val_shift_limit=(-35, 35))
# # #             # image1,image2, gt = randomShiftScaleRotate(image1, image2, gt,
# # #             #                                    shift_limit=(-0.15, 0.15),
# # #             #                                    scale_limit=(-0.25, 0.25),
# # #             #                                    aspect_limit=(-0.15, 0.15),
# # #             #                                    rotate_limit=(-10, 10))
# # #
# # #             image1, image2, gt = randomHorizontalFlip(image1,image2, gt)
# # #             image1, image2, gt = randomVerticleFlip(image1, image2, gt)
# # #             image1, image2, gt = randomRotate90(image1, image2, gt)
# # #             image1, image2, gt = resize(image1, image2, gt, 512,512)
# # #             # image, gt = resize(image, gt, 512)
# # #
# # #             # image = image[..., ::-1]  # bgr2rgb
# # #             # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # #             # grad = (255 * grade(gray)).astype(np.uint8)
# # #
# # #             # batch=[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
# # #             # sort = batch[np.random.randint(0,6)]
# # #             # image = cv2.merge([image[:,:,sort[0]], image[:,:,sort[1]], image[:,:,sort[2]]])
# # #
# # #             img1 = Image.fromarray(image1)
# # #             if self.transform is not None:
# # #                 img1 = self.transform(img1.copy())
# # #
# # #             img2 = Image.fromarray(image2)
# # #             if self.transform is not None:
# # #                 img2 = self.transform(img2.copy())
# # #             if np.random.random() < 0.5:
# # #                 return img1, img2, gt.copy(), lab
# # #             else:
# # #                 return img2, img1, gt.copy(), lab
# # #
# # #     def __len__(self):
# # #         return len(self.imgs)
# #
# #
# # import os
# # import cv2
# # import numpy as np
# # import pandas as pd
# # import random
# # from PIL import Image, ImageEnhance
# # from skimage.measure import label,regionprops
# # # from albumentations import Compose, MotionBlur
# # from torch.utils.data import Dataset
# #
# #
# # def random_color_jitter(cv_img, saturation_range, brightness_range, contrast_range, u=0.5):
# #     def saturation_jitter(cv_img, jitter_range):
# #         """
# #         调节图像饱和度
# #         Args:
# #             cv_img(numpy.ndarray): 输入图像
# #             jitter_range(float): 调节程度，0-1
# #         Returns:
# #             饱和度调整后的图像
# #         """
# #         greyMat = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
# #         greyMat = greyMat[:, :, None] * np.ones(3, dtype=int)[None, None, :]
# #         cv_img = cv_img.astype(np.float32)
# #         cv_img = cv_img * (1 - jitter_range) + jitter_range * greyMat
# #         cv_img = np.where(cv_img > 255, 255, cv_img)
# #         cv_img = cv_img.astype(np.uint8)
# #         return cv_img
# #
# #     def brightness_jitter(cv_img, jitter_range):
# #         """
# #         调节图像亮度
# #         Args:
# #             cv_img(numpy.ndarray): 输入图像
# #             jitter_range(float): 调节程度，0-1
# #         Returns:
# #             亮度调整后的图像
# #         """
# #         cv_img = cv_img.astype(np.float32)
# #         cv_img = cv_img * (1.0 - jitter_range)
# #         cv_img = np.where(cv_img > 255, 255, cv_img)
# #         cv_img = cv_img.astype(np.uint8)
# #         return cv_img
# #
# #     def contrast_jitter(cv_img, jitter_range):
# #         """
# #         调节图像对比度
# #         Args:
# #             cv_img(numpy.ndarray): 输入图像
# #             jitter_range(float): 调节程度，0-1
# #         Returns:
# #             对比度调整后的图像
# #         """
# #         greyMat = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
# #         mean = np.mean(greyMat)
# #         cv_img = cv_img.astype(np.float32)
# #         cv_img = cv_img * (1 - jitter_range) + jitter_range * mean
# #         cv_img = np.where(cv_img > 255, 255, cv_img)
# #         cv_img = cv_img.astype(np.uint8)
# #         return cv_img
# #     """
# #     图像亮度、饱和度、对比度调节，在调整范围内随机获得调节比例，并随机顺序叠加三种效果
# #     Args:
# #         cv_img(numpy.ndarray): 输入图像
# #         saturation_range(float): 饱和对调节范围，0-1
# #         brightness_range(float): 亮度调节范围，0-1
# #         contrast_range(float): 对比度调节范围，0-1
# #     Returns:
# #         亮度、饱和度、对比度调整后图像
# #     """
# #     if np.random.random() < u:
# #         saturation_ratio = np.random.uniform(-saturation_range, saturation_range)
# #         brightness_ratio = np.random.uniform(-brightness_range, brightness_range)
# #         contrast_ratio = np.random.uniform(-contrast_range, contrast_range)
# #         order = [0, 1, 2]
# #         np.random.shuffle(order)
# #         for i in range(3):
# #             if order[i] == 0:
# #                 cv_img = saturation_jitter(cv_img, saturation_ratio)
# #             if order[i] == 1:
# #                 cv_img = brightness_jitter(cv_img, brightness_ratio)
# #             if order[i] == 2:
# #                 cv_img = contrast_jitter(cv_img, contrast_ratio)
# #         return cv_img
# #     return cv_img
# #
# #
# # def randomHueSaturationValue(image1, image2, hue_shift_limit=(-180, 180),sat_shift_limit=(-255, 255),val_shift_limit=(-255, 255), u=0.5):
# #     if np.random.random() < u:
# #         image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
# #         h, s, v = cv2.split(image1)
# #         hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
# #         hue_shift = np.uint8(hue_shift)
# #         h += hue_shift
# #         sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
# #         s = cv2.add(s, sat_shift)
# #         val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
# #         v = cv2.add(v, val_shift)
# #         image1 = cv2.merge((h, s, v))
# #         image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2BGR)
# #
# #         image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
# #         h, s, v = cv2.split(image2)
# #         hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
# #         hue_shift = np.uint8(hue_shift)
# #         h += hue_shift
# #         sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
# #         s = cv2.add(s, sat_shift)
# #         val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
# #         v = cv2.add(v, val_shift)
# #         image2 = cv2.merge((h, s, v))
# #         image2 = cv2.cvtColor(image2, cv2.COLOR_HSV2BGR)
# #     return image1, image2
# #
# #
# # def randomShiftScaleRotate(image1, image2, mask,shift_limit=(-0.0, 0.0),scale_limit=(-0.0, 0.0),rotate_limit=(-0.0, 0.0),
# #                            aspect_limit=(-0.0, 0.0),borderMode=cv2.BORDER_CONSTANT, u=0.5):
# #     if np.random.random() < u:
# #         height, width, channel = image1.shape
# #
# #         angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
# #         scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
# #         aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
# #         sx = scale * aspect / (aspect ** 0.5)
# #         sy = scale / (aspect ** 0.5)
# #         dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
# #         dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)
# #
# #         cc = np.math.cos(angle / 180 * np.math.pi) * sx
# #         ss = np.math.sin(angle / 180 * np.math.pi) * sy
# #         rotate_matrix = np.array([[cc, -ss], [ss, cc]])
# #
# #         box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
# #         box1 = box0 - np.array([width / 2, height / 2])
# #         box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])
# #
# #         box0 = box0.astype(np.float32)
# #         box1 = box1.astype(np.float32)
# #         mat = cv2.getPerspectiveTransform(box0, box1)
# #         image1 = cv2.warpPerspective(image1, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
# #                                     borderValue=(0, 0,0,))
# #         image2 = cv2.warpPerspective(image2, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
# #                                     borderValue=(0, 0,0,))
# #         mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
# #                                    borderValue=(0, 0,0,))
# #     return image1, image2, mask
# #
# #
# # def randomHorizontalFlip(image1, image2, mask, u=0.5):
# #     if np.random.random() < u:
# #         image1 = cv2.flip(image1, 1)
# #         image2 = cv2.flip(image2, 1)
# #         mask = cv2.flip(mask, 1)
# #     return image1,image2, mask
# #
# #
# # def randomVerticleFlip(image1,image2, mask, u=0.5):
# #     if np.random.random() < u:
# #         image1 = cv2.flip(image1, 0)
# #         image2 = cv2.flip(image2, 0)
# #         mask = cv2.flip(mask, 0)
# #     return image1, image2, mask
# #
# #
# # def randomRotate90(image1,image2, mask, u=0.5):
# #     angle = np.random.randint(0,4)
# #     if angle:
# #         for i in range(angle):
# #             image1 = np.rot90(image1)
# #             image2 = np.rot90(image2)
# #             mask = np.rot90(mask)
# #         return image1,image2, mask
# #     return image1,image2, mask
# #
# # def randomclip(img1,img2,gt,outsize=384,multiscale=False):
# #     w,h = img1.shape[0],img1.shape[1]
# #     bbox_range=[0,0,w,h]
# #     lab = gt*255
# #     labeled_img, num = label(lab, background=0, return_num=True, connectivity=1)
# #     if num>0:
# #         bboxs = regionprops(labeled_img)
# #     if num==1:
# #         minr, minc, maxr, maxc = bboxs[0].bbox
# #         bbox_range=[minr, minc, maxr, maxc]
# #     if num>1:
# #         minr, minc, _, _ = bboxs[0].bbox
# #         bbox_range[0] = max(bbox_range[0], minr)
# #         bbox_range[1] = max(bbox_range[1], minc)
# #         _, _, maxr, maxc = bboxs[-1].bbox
# #         bbox_range[2] = min(bbox_range[2], maxr)
# #         bbox_range[3] = min(bbox_range[3], maxc)
# #     cr = (bbox_range[0] + bbox_range[2]) // 2
# #     cc = (bbox_range[1] + bbox_range[3]) // 2
# #     if (bbox_range[2]-bbox_range[0])<outsize and (bbox_range[3]-bbox_range[1])<outsize:
# #         # return aera (outsize,outsize) contain the bbox
# #         box=[cr-outsize//2,cc-outsize//2,cr+outsize//2,cc+outsize//2]
# #         if box[0]<0:
# #             box[0]=0
# #             box[2]=outsize
# #         if box[1]<0:
# #             box[1]=0
# #             box[3]=outsize
# #         if box[2]>w:
# #             box[0]=w-outsize
# #             box[2]=w
# #         if box[3]>h:
# #             box[1]=h-outsize
# #             box[3]=h
# #         img1 = img1[box[0]:box[2], box[1]:box[3], :]
# #         img2 = img2[box[0]:box[2], box[1]:box[3], :]
# #         gt = gt[box[0]:box[2], box[1]:box[3]]
# #         return img1,img2,gt
# #
# #     elif (bbox_range[2] - bbox_range[0]) < outsize or (bbox_range[3] - bbox_range[1]) < outsize:
# #         if (bbox_range[2] - bbox_range[0]) < outsize:
# #             diff = (outsize-(bbox_range[2] - bbox_range[0]))//2
# #             bbox_range[2] += diff
# #             bbox_range[0] -= diff
# #             if bbox_range[0]<0:
# #                 bbox_range[2] -= bbox_range[0]-1
# #                 bbox_range[0] = 0
# #             if bbox_range[2] > w:
# #                 bbox_range[0] -= bbox_range[2] - w
# #                 bbox_range[2] = w
# #
# #         if (bbox_range[3] - bbox_range[1]) < outsize:
# #             diff = (outsize-(bbox_range[3] - bbox_range[1]))//2
# #             bbox_range[3] += diff
# #             bbox_range[1] -= diff
# #             if bbox_range[1]<0:
# #                 bbox_range[3] -= bbox_range[1]-1
# #                 bbox_range[1] = 0
# #             if bbox_range[3] > h:
# #                 bbox_range[1] -= bbox_range[3] - h
# #                 bbox_range[3] = h
# #     if multiscale:
# #         min_size = min(bbox_range[3] - bbox_range[1], bbox_range[2] - bbox_range[0])
# #         dert = random.randint(256,min(480,min_size))
# #         x = random.randint(bbox_range[0], bbox_range[0]+(bbox_range[2] - bbox_range[0]-dert))
# #         y = random.randint(bbox_range[1], bbox_range[1]+(bbox_range[3] - bbox_range[1]-dert))
# #         img1 = img1[x:x + dert, y:y + dert, :]
# #         img2 = img2[x:x + dert, y:y + dert, :]
# #         gt = gt[x:x + dert, y:y + dert]
# #         img1 = cv2.resize(img1, (outsize, outsize), interpolation=cv2.INTER_LINEAR)
# #         img2 = cv2.resize(img2, (outsize, outsize), interpolation=cv2.INTER_LINEAR)
# #         gt = cv2.resize(gt, (outsize, outsize), interpolation=cv2.INTER_NEAREST)
# #         return img1, img2, gt
# #     else:
# #         x = random.randint(bbox_range[0], bbox_range[2] - outsize)
# #         y = random.randint(bbox_range[1], bbox_range[3] - outsize)
# #         img1 = img1[x:x + outsize, y:y + outsize, :]
# #         img2 = img2[x:x + outsize, y:y + outsize, :]
# #         gt = gt[x:x + outsize, y:y + outsize]
# #         return img1, img2, gt
# #
# # def resize(image1, image2, gt, insize, outsize):
# #
# #     x = np.random.randint(-512, 512)
# #     y = np.random.randint(-512, 512)
# #     if x < 0:
# #         if y < 0:
# #             image1 = image1[0:x, 0:y, :]
# #             image2 = image2[0:x, 0:y, :]
# #             gt = gt[0:x, 0:y]
# #         else:
# #             image1 = image1[0:x, y:insize, :]
# #             image2 = image2[0:x, y:insize, :]
# #             gt = gt[0:x, y:insize]
# #     else:
# #         if y < 0:
# #             image1 = image1[x:insize, 0:y, :]
# #             image2 = image2[x:insize, 0:y, :]
# #             gt = gt[x:insize, 0:y]
# #         else:
# #             image1 = image1[x:insize, y:insize, :]
# #             image2 = image2[x:insize, y:insize, :]
# #             gt = gt[x:insize, y:insize]
# #     image1 = cv2.resize(image1, (outsize, outsize), interpolation=cv2.INTER_LINEAR)
# #     image2 = cv2.resize(image2, (outsize, outsize), interpolation=cv2.INTER_LINEAR)
# #     gt = cv2.resize(gt, (outsize, outsize), interpolation=cv2.INTER_NEAREST)
# #     # x = np.random.randint(0, 128)
# #     # y = np.random.randint(0, 128)
# #     # image = image[x:x+outsize, y:y+outsize, :]
# #     # gt = gt[x:x+outsize, y:y+outsize]
# #
# #     return image1, image2, gt
# #
# #
# # def Balance_Random_Sample(img1, img2, gt, inputsize=1024, outsize=512, multiscale=False):
# #     w, h = img1.shape[0], img1.shape[1]
# #     bbox_range = [0, 0, w, h]
# #     half_outsz = outsize//2
# #     # the input label with value {0,1...}
# #     lab = gt * 255
# #     labeled_img, num = label(lab, background=0, return_num=True, connectivity=1)
# #     # get the boundary or the positive label
# #     if num > 0:
# #         bbox_range = [w, h, 0, 0]
# #         bboxs = regionprops(labeled_img)
# #         for index in range(num):
# #             minr, minc, maxr, maxc = bboxs[index].bbox
# #             bbox_range[0] = min(bbox_range[0], minr)
# #             bbox_range[1] = min(bbox_range[1], minc)
# #             bbox_range[2] = max(bbox_range[2], maxr)
# #             bbox_range[3] = max(bbox_range[3], maxc)
# #     # if num==1:
# #     #     minr, minc, maxr, maxc = bboxs[0].bbox
# #     #     bbox_range=[minr, minc, maxr, maxc]
# #     # if num>1:
# #     #     minr, minc, _, _ = bboxs[0].bbox
# #     #     bbox_range[0] = max(bbox_range[0], minr)
# #     #     bbox_range[1] = max(bbox_range[1], minc)
# #     #     _, _, maxr, maxc = bboxs[-1].bbox
# #     #     bbox_range[2] = min(bbox_range[2], maxr)
# #     #     bbox_range[3] = min(bbox_range[3], maxc)
# #     ph = (bbox_range[2] - bbox_range[0])
# #     pw = (bbox_range[3] - bbox_range[1])
# #
# #     # clip for the 100% background image
# #     if num == 0:
# #         if multiscale:
# #             dert = int(outsize*(0.6+random.random()))
# #             halfdsz = dert//2
# #             ccx = random.randint(halfdsz, inputsize - halfdsz)
# #             ccy = random.randint(halfdsz, inputsize - halfdsz)
# #             image1 = img1[ccx - halfdsz:ccx+ halfdsz, ccy- halfdsz:ccy + halfdsz, :]
# #             image2 = img2[ccx - halfdsz:ccx+ halfdsz, ccy- halfdsz:ccy + halfdsz, :]
# #             mask = gt[ccx - halfdsz:ccx+ halfdsz, ccy- halfdsz:ccy + halfdsz]
# #             image1 = cv2.resize(image1, (outsize, outsize), interpolation=cv2.INTER_LINEAR)
# #             image2 = cv2.resize(image2, (outsize, outsize), interpolation=cv2.INTER_LINEAR)
# #             mask = cv2.resize(mask, (outsize, outsize), interpolation=cv2.INTER_NEAREST)
# #             return image1, image2, mask
# #         else:
# #             ccx = random.randint(half_outsz, inputsize - half_outsz)
# #             ccy = random.randint(half_outsz, inputsize - half_outsz)
# #             image1 = img1[ccx - half_outsz:ccx + half_outsz, ccy - half_outsz:ccy + half_outsz, :]
# #             image2 = img2[ccx - half_outsz:ccx + half_outsz, ccy - half_outsz:ccy + half_outsz, :]
# #             mask = gt[ccx - half_outsz:ccx + half_outsz, ccy - half_outsz:ccy + half_outsz]
# #             return image1, image2, mask
# #
# #     # clip for the higher background image
# #     elif pw < outsize or ph < outsize:
# #         background = random.random()
# #         if background>0.5:
# #             if multiscale:
# #                 dert = int(outsize * (0.6 + random.random()))
# #                 halfdsz = dert//2
# #                 ccx = random.randint(halfdsz, inputsize - halfdsz)
# #                 ccy = random.randint(halfdsz, inputsize - halfdsz)
# #                 image1 = img1[ccx - halfdsz:ccx + halfdsz, ccy - halfdsz:ccy + halfdsz, :]
# #                 image2 = img2[ccx - halfdsz:ccx + halfdsz, ccy - halfdsz:ccy + halfdsz, :]
# #                 mask = gt[ccx - halfdsz:ccx + halfdsz, ccy - halfdsz:ccy + halfdsz]
# #                 image1 = cv2.resize(image1, (outsize, outsize), interpolation=cv2.INTER_LINEAR)
# #                 image2 = cv2.resize(image2, (outsize, outsize), interpolation=cv2.INTER_LINEAR)
# #                 mask = cv2.resize(mask, (outsize, outsize), interpolation=cv2.INTER_NEAREST)
# #                 return image1, image2, mask
# #             else:
# #                 ccx = random.randint(half_outsz, inputsize - half_outsz)
# #                 ccy = random.randint(half_outsz, inputsize - half_outsz)
# #                 image1 = img1[ccx - half_outsz:ccx + half_outsz, ccy - half_outsz:ccy + half_outsz, :]
# #                 image2 = img2[ccx - half_outsz:ccx + half_outsz, ccy - half_outsz:ccy + half_outsz, :]
# #                 mask = gt[ccx - half_outsz:ccx + half_outsz, ccy - half_outsz:ccy + half_outsz]
# #                 return image1, image2, mask
# #         else:
# #             if multiscale:
# #                 dert = int(outsize * (0.6 + random.random()))
# #                 halfdsz = dert // 2
# #                 if bbox_range[0] < halfdsz:
# #                     bbox_range[0] = halfdsz
# #                 if bbox_range[1] < halfdsz:
# #                     bbox_range[1] = halfdsz
# #                 if bbox_range[2] > inputsize - halfdsz:
# #                     bbox_range[2] = inputsize - halfdsz
# #                 if bbox_range[3] > inputsize - halfdsz:
# #                     bbox_range[3] = inputsize - halfdsz
# #
# #                 if (bbox_range[0] >= bbox_range[2]):
# #                     ccx = bbox_range[0]
# #                 else:
# #                     ccx = random.randint(bbox_range[0], bbox_range[2])
# #                 if (bbox_range[1] >= bbox_range[3]):
# #                     ccy = bbox_range[1]
# #                 else:
# #                     ccy = random.randint(bbox_range[1], bbox_range[3])
# #
# #                 image1 = img1[ccx - halfdsz:ccx, ccy:ccy + halfdsz, :]
# #                 image2 = img2[ccx - halfdsz:ccx, ccy:ccy + halfdsz, :]
# #                 mask = gt[ccx - halfdsz:ccx, ccy:ccy + halfdsz]
# #                 image1 = cv2.resize(image1, (outsize, outsize), interpolation=cv2.INTER_LINEAR)
# #                 image2 = cv2.resize(image2, (outsize, outsize), interpolation=cv2.INTER_LINEAR)
# #                 mask = cv2.resize(mask, (outsize, outsize), interpolation=cv2.INTER_NEAREST)
# #                 return image1, image2, mask
# #             else:
# #                 if bbox_range[0] < half_outsz:
# #                     bbox_range[0] = half_outsz
# #                 if bbox_range[1] < half_outsz:
# #                     bbox_range[1] = half_outsz
# #                 if bbox_range[2] > inputsize - half_outsz:
# #                     bbox_range[2] = inputsize - half_outsz
# #                 if bbox_range[3] > inputsize - half_outsz:
# #                     bbox_range[3] = inputsize - half_outsz
# #                 if (bbox_range[0] >= bbox_range[2]):
# #                     ccx = bbox_range[0]
# #                 else:
# #                     ccx = random.randint(bbox_range[0], bbox_range[2])
# #                 if (bbox_range[1] >= bbox_range[3]):
# #                     ccy = bbox_range[1]
# #                 else:
# #                     ccy = random.randint(bbox_range[1], bbox_range[3])
# #                 image1 = img1[ccx - half_outsz:ccx+ half_outsz, ccy- half_outsz:ccy + half_outsz, :]
# #                 image2 = img2[ccx - half_outsz:ccx+ half_outsz, ccy- half_outsz:ccy + half_outsz, :]
# #                 mask = gt[ccx - half_outsz:ccx+ half_outsz, ccy- half_outsz:ccy + half_outsz]
# #                 return image1, image2, mask
# #
# #     # clip for the lower background image
# #     else:
# #         if multiscale:
# #             dert = int(outsize * (0.6 + random.random()))
# #             halfdsz = dert//2
# #             if bbox_range[0]<halfdsz:
# #                 bbox_range[0]=halfdsz
# #             if bbox_range[1]<halfdsz:
# #                 bbox_range[1]=halfdsz
# #             if bbox_range[2]> inputsize-halfdsz:
# #                 bbox_range[2]=inputsize-halfdsz
# #             if bbox_range[3]> inputsize-halfdsz:
# #                 bbox_range[3]= inputsize-halfdsz
# #
# #             if (bbox_range[0] >= bbox_range[2]):
# #                 ccx = bbox_range[0]
# #             else:
# #                 ccx = random.randint(bbox_range[0], bbox_range[2])
# #             if (bbox_range[1] >= bbox_range[3]):
# #                 ccy = bbox_range[1]
# #             else:
# #                 ccy = random.randint(bbox_range[1], bbox_range[3])
# #
# #             image1 = img1[ccx - halfdsz:ccx + halfdsz, ccy - halfdsz:ccy + halfdsz, :]
# #             image2 = img2[ccx - halfdsz:ccx + halfdsz, ccy - halfdsz:ccy + halfdsz, :]
# #             mask = gt[ccx - halfdsz:ccx + halfdsz, ccy - halfdsz:ccy + halfdsz]
# #
# #             image1 = cv2.resize(image1, (outsize, outsize), interpolation=cv2.INTER_LINEAR)
# #             image2 = cv2.resize(image2, (outsize, outsize), interpolation=cv2.INTER_LINEAR)
# #             mask = cv2.resize(mask, (outsize, outsize), interpolation=cv2.INTER_NEAREST)
# #             return image1, image2, mask
# #         else:
# #             if bbox_range[0]<half_outsz:
# #                 bbox_range[0]=half_outsz
# #             if bbox_range[1]<half_outsz:
# #                 bbox_range[1]=half_outsz
# #             if bbox_range[2]> inputsize-half_outsz:
# #                 bbox_range[2]=inputsize-half_outsz
# #             if bbox_range[3]> inputsize-half_outsz:
# #                 bbox_range[3]= inputsize-half_outsz
# #
# #             ccx = random.randint(bbox_range[0], bbox_range[2])
# #             ccy = random.randint(bbox_range[1], bbox_range[3])
# #             image1 = img1[ccx - half_outsz:ccx + half_outsz, ccy - half_outsz:ccy + half_outsz, :]
# #             image2 = img2[ccx - half_outsz:ccx + half_outsz, ccy - half_outsz:ccy + half_outsz, :]
# #             mask = gt[ccx - half_outsz:ccx + half_outsz, ccy - half_outsz:ccy + half_outsz]
# #             return image1, image2, mask
# #
# # # def motionblur(image, gt, blur=7, p=0.5):
# # #     aug = Compose([MotionBlur(blur_limit=blur, p=p)])
# # #     augmented = aug(image=image, mask=gt)
# # #     image_MotionBlur = augmented['image']
# # #     gt_MotionBlur = augmented['mask']
# # #     return image_MotionBlur, gt_MotionBlur
# #
# #
# # def grade(img):
# #     x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
# #     y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
# #     absX = cv2.convertScaleAbs(x)
# #     absY = cv2.convertScaleAbs(y)
# #     dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# #     mi = np.min(dst)
# #     ma = np.max(dst)
# #     res = (dst - mi) / (0.000000001 + (ma - mi))
# #     res[np.isnan(res)] = 0
# #     return res
# #
# # def mixup(timg1,timg2,gt):
# #     lab = gt.copy()
# #     labeled_img, num = label(255*lab, background=0, return_num=True, connectivity=1)
# #     bboxs = regionprops(labeled_img)
# #     if num==0:
# #         return timg1,timg2,gt
# #     else:
# #         mask = np.zeros(timg1.shape[:2],np.uint8)
# #         for index in range(num):
# #             if bboxs[index].area>400:
# #                 if random.random()<0.75:
# #                     coord = bboxs[index].coords
# #                     coord = np.array([coord])
# #                     cv2.polylines(mask,coord,1,255)
# #                     cv2.fillPoly(mask,coord,255)
# #         mask = np.transpose(mask)
# #         back = 255 - mask
# #         dst1 = cv2.bitwise_and(timg1, timg1,mask=mask)
# #         dst2 = cv2.bitwise_and(timg2, timg2, mask=mask)
# #         bg1 = cv2.bitwise_and(timg1, timg1, mask=back)
# #         bg2 = cv2.bitwise_and(timg2, timg2, mask=back)
# #
# #         dist_image1 = dst2+bg1
# #         dist_image2 = dst1+bg2
# #         return dist_image1,dist_image2,gt
# #
# #
# # def v_adapt(bgrimg):
# #     hsv = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2HSV)
# #     h, s, v = cv2.split(hsv)
# #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7, 7))
# #     res = clahe.apply(v)
# #     v2 = np.mean(res)
# #     res = np.array(res + (100.0 - v2), dtype=np.uint8)
# #     hsv = cv2.merge([h, s, res])
# #     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# #
# # class Mydataset(Dataset):
# #     def __init__(self, path, augment=False, transform=None, target_transform=None):
# #
# #         self.aug = augment
# #         self.file_path = os.path.dirname(path)
# #         data = pd.read_csv(path)  # 获取csv表中的数据
# #         imgs = []
# #         for i in range(len(data)):
# #             imgs.append((data.iloc[i, 0], data.iloc[i, 1]))
# #         self.imgs = imgs
# #         self.transform = transform
# #         self.target_transform = target_transform
# #
# #     def __getitem__(self, item):
# #         if self.aug == False:
# #             fn, lab = self.imgs[item]
# #             # fn = os.path.join(self.file_path, "image_A/" + fn)
# #             # label = os.path.join(self.file_path, "image_A/" + lab)
# #             fn1 = os.path.join(self.file_path, "Image1/" + fn)
# #             fn2 = os.path.join(self.file_path, "Image2/" + fn)
# #             # print(fn2)
# #             label = os.path.join(self.file_path, "label/" + lab)
# #
# #             bgr_img = cv2.imread(fn1, -1)
# #             bgr_img = v_adapt(bgr_img)
# #             img = Image.fromarray(bgr_img)
# #             if self.transform is not None:
# #                 img = self.transform(img)
# #
# #             bgr_img2 = cv2.imread(fn2, -1)
# #             bgr_img2 = v_adapt(bgr_img2)
# #             img2 = Image.fromarray(bgr_img2)
# #             if self.transform is not None:
# #                 img2 = self.transform(img2)
# #
# #             gt = cv2.imread(label, 0)//255
# #             return img, img2, gt, lab
# #
# #
# #         else:
# #             # 进行数据增强
# #             fn, lab = self.imgs[item]
# #             # train with data.cvs
# #             fn1 = os.path.join(self.file_path, "img1/" + fn)
# #             fn2 = os.path.join(self.file_path, "img2/" + fn)
# #             label = os.path.join(self.file_path, "label/" + lab)
# #
# #             gt = cv2.imread(label, 0)//255
# #             # gt = gt[:,:,0]/255
# #             image1 = cv2.imread(fn1, -1)
# #             image2 = cv2.imread(fn2, -1)
# #
# #             # batch=[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
# #             # sort1 = batch[np.random.randint(0,6)]
# #             # # sort2 = batch[np.random.randint(0, 6)]
# #             # image1 = cv2.merge([image1[:, :, sort1[0]], image1[:, :, sort1[1]], image1[:, :, sort1[2]]])
# #             # image2 = cv2.merge([image2[:, :, sort1[0]], image2[:, :, sort1[1]], image2[:, :, sort1[2]]])
# #
# #
# #             # image, gt = motionblur(image, gt, blur=5, p=0.5)
# #             # image = random_color_jitter(image, saturation_range=0.5, brightness_range=0.5, contrast_range=0.5, u=0.5)
# #             # image = randomHueSaturationValue(image,hue_shift_limit=(-30, 30),sat_shift_limit=(-5, 5),val_shift_limit=(-15, 15))
# #             # image, gt = randomShiftScaleRotate(image, gt, shift_limit=(-0.1, 0.1), scale_limit=(-0.0, 0.0),
# #             #                                    aspect_limit=(-0.1, 0.1), rotate_limit=(-5, 5))
# #             # image, gt = randomHorizontalFlip(image, gt, u=0.5)
# #             # image, gt = randomVerticleFlip(image, gt, u=0.5)
# #             # image, gt = randomRotate90(image, gt, u=0.5)
# #
# #
# #             image1,image2 = randomHueSaturationValue(image1, image2,
# #                                              hue_shift_limit=(-30, 30),
# #                                              sat_shift_limit=(-25, 25),
# #                                              val_shift_limit=(-5, 5))
# #             image1,image2, gt = randomShiftScaleRotate(image1, image2, gt,
# #                                                shift_limit=(-0.15, 0.15),
# #                                                scale_limit=(-0.25, 0.5),
# #                                                aspect_limit=(-0.1, 0.1),
# #                                                rotate_limit=(-10, 10))
# #             # image1, image2, gt = randomHorizontalFlip(image1,image2, gt)
# #             image1, image2, gt = randomVerticleFlip(image1, image2, gt)
# #             image1, image2, gt = randomRotate90(image1, image2, gt)
# #
# #             image1, image2, gt = resize(image1, image2, gt, 1024, 768)
# #             # image1, image2, gt = Balance_Random_Sample(image1, image2, gt, inputsize=1024, outsize=768, multiscale=True)
# #             # image1, image2, gt = randomclip(image1, image2, gt, outsize=512, multiscale=True)
# #             # image1, image2, gt = mixup(image1, image2, gt)
# #             # batch=[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
# #             # sort1 = batch[np.random.randint(0,6)]
# #             # sort2 = batch[np.random.randint(0, 6)]
# #             # image1 = cv2.merge([image1[:, :, sort1[0]], image1[:, :, sort1[1]], image1[:, :, sort1[2]]])
# #             # image2 = cv2.merge([image2[:, :, sort2[0]], image2[:, :, sort2[1]], image2[:, :, sort2[2]]])
# #
# #             # image = image[..., ::-1]  # bgr2rgb
# #             # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #             # grad = (255 * grade(gray)).astype(np.uint8)
# #
# #             # batch=[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
# #             # sort = batch[np.random.randint(0,6)]
# #             # image = cv2.merge([image[:,:,sort[0]], image[:,:,sort[1]], image[:,:,sort[2]]])
# #
# #             img1 = Image.fromarray(image1)
# #             if self.transform is not None:
# #                 img1 = self.transform(img1.copy())
# #
# #             img2 = Image.fromarray(image2)
# #             if self.transform is not None:
# #                 img2 = self.transform(img2.copy())
# #             if np.random.random() < 0.5:
# #                 return img1, img2, gt.copy(), lab
# #             else:
# #                 return img2, img1, gt.copy(), lab
# #             # return img1, img2, gt.copy(), lab
# #
# #     def __len__(self):
# #         return len(self.imgs)
#
#
# import os
# import cv2
# import numpy as np
# import pandas as pd
# from PIL import Image, ImageEnhance
# # from albumentations import Compose, MotionBlur
# from torch.utils.data import Dataset
#
#
# def random_color_jitter(cv_img, saturation_range, brightness_range, contrast_range, u=0.5):
#     def saturation_jitter(cv_img, jitter_range):
#         """
#         调节图像饱和度
#         Args:
#             cv_img(numpy.ndarray): 输入图像
#             jitter_range(float): 调节程度，0-1
#         Returns:
#             饱和度调整后的图像
#         """
#         greyMat = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
#         greyMat = greyMat[:, :, None] * np.ones(3, dtype=int)[None, None, :]
#         cv_img = cv_img.astype(np.float32)
#         cv_img = cv_img * (1 - jitter_range) + jitter_range * greyMat
#         cv_img = np.where(cv_img > 255, 255, cv_img)
#         cv_img = cv_img.astype(np.uint8)
#         return cv_img
#
#     def brightness_jitter(cv_img, jitter_range):
#         """
#         调节图像亮度
#         Args:
#             cv_img(numpy.ndarray): 输入图像
#             jitter_range(float): 调节程度，0-1
#         Returns:
#             亮度调整后的图像
#         """
#         cv_img = cv_img.astype(np.float32)
#         cv_img = cv_img * (1.0 - jitter_range)
#         cv_img = np.where(cv_img > 255, 255, cv_img)
#         cv_img = cv_img.astype(np.uint8)
#         return cv_img
#
#     def contrast_jitter(cv_img, jitter_range):
#         """
#         调节图像对比度
#         Args:
#             cv_img(numpy.ndarray): 输入图像
#             jitter_range(float): 调节程度，0-1
#         Returns:
#             对比度调整后的图像
#         """
#         greyMat = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
#         mean = np.mean(greyMat)
#         cv_img = cv_img.astype(np.float32)
#         cv_img = cv_img * (1 - jitter_range) + jitter_range * mean
#         cv_img = np.where(cv_img > 255, 255, cv_img)
#         cv_img = cv_img.astype(np.uint8)
#         return cv_img
#     """
#     图像亮度、饱和度、对比度调节，在调整范围内随机获得调节比例，并随机顺序叠加三种效果
#     Args:
#         cv_img(numpy.ndarray): 输入图像
#         saturation_range(float): 饱和对调节范围，0-1
#         brightness_range(float): 亮度调节范围，0-1
#         contrast_range(float): 对比度调节范围，0-1
#     Returns:
#         亮度、饱和度、对比度调整后图像
#     """
#     if np.random.random() < u:
#         saturation_ratio = np.random.uniform(-saturation_range, saturation_range)
#         brightness_ratio = np.random.uniform(-brightness_range, brightness_range)
#         contrast_ratio = np.random.uniform(-contrast_range, contrast_range)
#         order = [0, 1, 2]
#         np.random.shuffle(order)
#         for i in range(3):
#             if order[i] == 0:
#                 cv_img = saturation_jitter(cv_img, saturation_ratio)
#             if order[i] == 1:
#                 cv_img = brightness_jitter(cv_img, brightness_ratio)
#             if order[i] == 2:
#                 cv_img = contrast_jitter(cv_img, contrast_ratio)
#         return cv_img
#     return cv_img
#
#
# def randomHueSaturationValue(image1, image2, hue_shift_limit=(-180, 180),sat_shift_limit=(-255, 255),val_shift_limit=(-255, 255), u=0.5):
#     if np.random.random() < u:
#         image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
#         h, s, v = cv2.split(image1)
#         hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
#         hue_shift = np.uint8(hue_shift)
#         h += hue_shift
#         sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
#         s = cv2.add(s, sat_shift)
#         val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
#         v = cv2.add(v, val_shift)
#         image1 = cv2.merge((h, s, v))
#         image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2BGR)
#
#         image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
#         h, s, v = cv2.split(image2)
#         hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
#         hue_shift = np.uint8(hue_shift)
#         h += hue_shift
#         sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
#         s = cv2.add(s, sat_shift)
#         val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
#         v = cv2.add(v, val_shift)
#         image2 = cv2.merge((h, s, v))
#         image2 = cv2.cvtColor(image2, cv2.COLOR_HSV2BGR)
#     return image1, image2
#
#
# def randomShiftScaleRotate(image1, image2, mask,gt1,gt2,shift_limit=(-0.0, 0.0),scale_limit=(-0.0, 0.0),rotate_limit=(-0.0, 0.0),
#                            aspect_limit=(-0.0, 0.0),borderMode=cv2.BORDER_CONSTANT, u=0.5):
#     if np.random.random() < u:
#         height, width, channel = image1.shape
#
#         angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
#         scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
#         aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
#         sx = scale * aspect / (aspect ** 0.5)
#         sy = scale / (aspect ** 0.5)
#         dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
#         dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)
#
#         cc = np.math.cos(angle / 180 * np.math.pi) * sx
#         ss = np.math.sin(angle / 180 * np.math.pi) * sy
#         rotate_matrix = np.array([[cc, -ss], [ss, cc]])
#
#         box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
#         box1 = box0 - np.array([width / 2, height / 2])
#         box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])
#
#         box0 = box0.astype(np.float32)
#         box1 = box1.astype(np.float32)
#         mat = cv2.getPerspectiveTransform(box0, box1)
#         image1 = cv2.warpPerspective(image1, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
#                                     borderValue=(0, 0,0,))
#         image2 = cv2.warpPerspective(image2, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
#                                      borderValue=(0, 0, 0,))
#         mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST, borderMode=borderMode,
#                                    borderValue=(
#                                        0, 0,
#                                        0,))
#         gt1 = cv2.warpPerspective(gt1, mat, (width, height), flags=cv2.INTER_NEAREST, borderMode=borderMode,
#                                    borderValue=(
#                                        0, 0,
#                                        0,))
#         gt2 = cv2.warpPerspective(gt2, mat, (width, height), flags=cv2.INTER_NEAREST, borderMode=borderMode,
#                                    borderValue=(
#                                        0, 0,
#                                        0,))
#     return image1, image2, mask,gt1,gt2
#
#
# def randomHorizontalFlip(image1, image2, mask, u=0.5):
#     if np.random.random() < u:
#         image1 = cv2.flip(image1, 1)
#         image2 = cv2.flip(image2, 1)
#         mask = cv2.flip(mask, 1)
#     return image1,image2, mask
#
#
# def randomVerticleFlip(image1,image2, mask,gt1,gt2, u=0.5):
#     if np.random.random() < u:
#         image1 = cv2.flip(image1, 0)
#         image2 = cv2.flip(image2, 0)
#         mask = cv2.flip(mask, 0)
#         gt1 = cv2.flip(gt1, 0)
#         gt2 = cv2.flip(gt2, 0)
#     return image1, image2, mask,gt1,gt2
#
#
# def randomRotate90(image1,image2, mask,gt1,gt2, u=0.5):
#     if np.random.random() < u:
#         angle = np.random.randint(1,4)
#         for i in range(angle):
#             image1 = np.rot90(image1)
#             image2 = np.rot90(image2)
#             mask = np.rot90(mask)
#             gt1 = np.rot90(gt1)
#             gt2 = np.rot90(gt2)
#         return image1,image2, mask,gt1,gt2
#     return image1,image2, mask,gt1,gt2
#
#
# def resize(image1,image2, gt,insize, outsize):
#
#     x = np.random.randint(-192, 192)
#     y = np.random.randint(-192, 192)
#     if x < 0:
#         if y < 0:
#             image1 = image1[0:x, 0:y, :]
#             image2 = image2[0:x, 0:y, :]
#             gt = gt[0:x, 0:y]
#         else:
#             image1 = image1[0:x, y:insize, :]
#             image2 = image2[0:x, y:insize, :]
#             gt = gt[0:x, y:insize]
#     else:
#         if y < 0:
#             image1 = image1[x:insize, 0:y, :]
#             image2 = image2[x:insize, 0:y, :]
#             gt = gt[x:insize, 0:y]
#         else:
#             image1 = image1[x:insize, y:insize, :]
#             image2 = image2[x:insize, y:insize, :]
#             gt = gt[x:insize, y:insize]
#     image1 = cv2.resize(image1, (outsize, outsize), interpolation=cv2.INTER_LINEAR)
#     image2 = cv2.resize(image2, (outsize, outsize), interpolation=cv2.INTER_LINEAR)
#     gt = cv2.resize(gt, (outsize, outsize), interpolation=cv2.INTER_NEAREST)
#     # x = np.random.randint(0, 128)
#     # y = np.random.randint(0, 128)
#     # image = image[x:x+outsize, y:y+outsize, :]
#     # gt = gt[x:x+outsize, y:y+outsize]
#
#     return image1, image2,gt
#
#
# # def motionblur(image, gt, blur=7, p=0.5):
# #     aug = Compose([MotionBlur(blur_limit=blur, p=p)])
# #     augmented = aug(image=image, mask=gt)
# #     image_MotionBlur = augmented['image']
# #     gt_MotionBlur = augmented['mask']
# #     return image_MotionBlur, gt_MotionBlur
#
#
# def grade(img):
#     x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
#     y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
#     absX = cv2.convertScaleAbs(x)
#     absY = cv2.convertScaleAbs(y)
#     dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
#     mi = np.min(dst)
#     ma = np.max(dst)
#     res = (dst - mi) / (0.000000001 + (ma - mi))
#     res[np.isnan(res)] = 0
#     return res
#
#
# class Mydataset(Dataset):
#     def __init__(self, path, augment=False, transform=None, target_transform=None,lab_smooth=0):
#
#         self.aug = augment
#         self.file_path = os.path.dirname(path)
#         data = pd.read_csv(path)  # 获取csv表中的数据
#         imgs = []
#         for i in range(len(data)):
#             imgs.append((data.iloc[i, 0], data.iloc[i, 1]))
#         self.imgs = imgs
#         self.transform = transform
#         self.target_transform = target_transform
#         self.lab_smooth=lab_smooth
#
#     def __getitem__(self, item):
#         if self.aug == False:
#             fn, lab = self.imgs[item]
#             # fn = os.path.join(self.file_path, "image_A/" + fn)
#             # label = os.path.join(self.file_path, "image_A/" + lab)
#             fn1 = os.path.join(self.file_path, "A/" + fn)
#             fn2 = os.path.join(self.file_path, "B/" + fn)
#             label = os.path.join(self.file_path, "label/" + lab)
#
#             bgr_img = cv2.imread(fn1, -1)
#             img = Image.fromarray(bgr_img)
#             if self.transform is not None:
#                 img = self.transform(img)
#
#             bgr_img2 = cv2.imread(fn2, -1)
#             img2 = Image.fromarray(bgr_img2)
#             if self.transform is not None:
#                 img2 = self.transform(img2)
#
#             gt = cv2.imread(label, 0)/255
#             # gt = gt[:, :, 0] / 255
#             return img, img2, gt, lab
#
#
#         else:
#             # 进行数据增强
#             fn, lab = self.imgs[item]
#             fn1 = os.path.join(self.file_path, "A/" + fn)
#             fn2 = os.path.join(self.file_path, "B/" + fn)
#             label = os.path.join(self.file_path, "label/" + lab)
#             lab1 = os.path.join(self.file_path, "lab1/" + lab)
#             lab2 = os.path.join(self.file_path, "lab2/" + lab)
#
#             if self.lab_smooth>0:
#                 gt = cv2.GaussianBlur(cv2.imread(label, 0) / 255, ksize=(self.lab_smooth, self.lab_smooth),sigmaX=self.lab_smooth)
#                 gt1 = cv2.GaussianBlur(cv2.imread(lab1, 0) / 255, ksize=(self.lab_smooth, self.lab_smooth),sigmaX=self.lab_smooth)
#                 gt2 = cv2.GaussianBlur(cv2.imread(lab2, 0) / 255, ksize=(self.lab_smooth, self.lab_smooth),sigmaX=self.lab_smooth)
#             else:
#                 gt = cv2.imread(label, 0)/255
#                 gt1 = cv2.imread(lab1, 0) / 255
#                 gt2 = cv2.imread(lab2, 0) / 255
#
#             image1 = cv2.imread(fn1, -1)
#             image2 = cv2.imread(fn2, -1)
#
#             # batch=[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
#             # sort1 = batch[np.random.randint(0,6)]
#             # sort2 = batch[np.random.randint(0, 6)]
#             # image1 = cv2.merge([image1[:, :, sort1[0]], image1[:, :, sort1[1]], image1[:, :, sort1[2]]])
#             # image2 = cv2.merge([image2[:, :, sort2[0]], image2[:, :, sort2[1]], image2[:, :, sort2[2]]])
#
#
#             # image, gt = motionblur(image, gt, blur=5, p=0.5)
#             # image = random_color_jitter(image, saturation_range=0.5, brightness_range=0.5, contrast_range=0.5, u=0.5)
#             # image = randomHueSaturationValue(image,hue_shift_limit=(-30, 30),sat_shift_limit=(-5, 5),val_shift_limit=(-15, 15))
#             # image, gt = randomShiftScaleRotate(image, gt, shift_limit=(-0.1, 0.1), scale_limit=(-0.0, 0.0),
#             #                                    aspect_limit=(-0.1, 0.1), rotate_limit=(-5, 5))
#             # image, gt = randomHorizontalFlip(image, gt, u=0.5)
#             # image, gt = randomVerticleFlip(image, gt, u=0.5)
#             # image, gt = randomRotate90(image, gt, u=0.5)
#             # image, gt = resize(image, gt, 1024, 640)
#
#             image1,image2 = randomHueSaturationValue(image1, image2,
#                                              hue_shift_limit=(-30, 30),
#                                              sat_shift_limit=(-35, 35),
#                                              val_shift_limit=(-35, 35))
#             image1,image2, gt,gt1,gt2 = randomShiftScaleRotate(image1, image2, gt,gt1,gt2,
#                                                shift_limit=(-0.15, 0.15),
#                                                scale_limit=(-0.25, 0.5),
#                                                aspect_limit=(-0.15, 0.15),
#                                                rotate_limit=(-10, 10))
#
#             # image1, image2, gt = randomHorizontalFlip(image1,image2, gt)
#             image1, image2, gt,gt1,gt2 = randomVerticleFlip(image1, image2, gt,gt1,gt2)
#             image1, image2, gt,gt1,gt2 = randomRotate90(image1, image2, gt,gt1,gt2)
#             # image1, image2, gt = resize(image1, image2, gt, 512,512)
#             # image, gt = resize(image, gt, 512)
#
#             # image = image[..., ::-1]  # bgr2rgb
#             # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             # grad = (255 * grade(gray)).astype(np.uint8)
#
#             # batch=[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
#             # sort = batch[np.random.randint(0,6)]
#             # image = cv2.merge([image[:,:,sort[0]], image[:,:,sort[1]], image[:,:,sort[2]]])
#
#             img1 = Image.fromarray(image1)
#             if self.transform is not None:
#                 img1 = self.transform(img1.copy())
#
#             img2 = Image.fromarray(image2)
#             if self.transform is not None:
#                 img2 = self.transform(img2.copy())
#
#             # return img1, img2, gt.copy(),gt1.copy(), gt2.copy(),lab
#             if np.random.random() < 0.5:
#                 return img1, img2, gt.copy(),gt1.copy(), gt2.copy(),lab
#             else:
#                 return img2, img1, gt.copy(),gt2.copy(), gt1.copy(), lab
#
#     def __len__(self):
#         return len(self.imgs)
#




import os
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
# from albumentations import Compose, MotionBlur
from torch.utils.data import Dataset
from skimage.measure import label,regionprops

def random_color_jitter(cv_img, saturation_range, brightness_range, contrast_range, u=0.5):
    def saturation_jitter(cv_img, jitter_range):
        """
        调节图像饱和度
        Args:
            cv_img(numpy.ndarray): 输入图像
            jitter_range(float): 调节程度，0-1
        Returns:
            饱和度调整后的图像
        """
        greyMat = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        greyMat = greyMat[:, :, None] * np.ones(3, dtype=int)[None, None, :]
        cv_img = cv_img.astype(np.float32)
        cv_img = cv_img * (1 - jitter_range) + jitter_range * greyMat
        cv_img = np.where(cv_img > 255, 255, cv_img)
        cv_img = cv_img.astype(np.uint8)
        return cv_img

    def brightness_jitter(cv_img, jitter_range):
        """
        调节图像亮度
        Args:
            cv_img(numpy.ndarray): 输入图像
            jitter_range(float): 调节程度，0-1
        Returns:
            亮度调整后的图像
        """
        cv_img = cv_img.astype(np.float32)
        cv_img = cv_img * (1.0 - jitter_range)
        cv_img = np.where(cv_img > 255, 255, cv_img)
        cv_img = cv_img.astype(np.uint8)
        return cv_img

    def contrast_jitter(cv_img, jitter_range):
        """
        调节图像对比度
        Args:
            cv_img(numpy.ndarray): 输入图像
            jitter_range(float): 调节程度，0-1
        Returns:
            对比度调整后的图像
        """
        greyMat = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        mean = np.mean(greyMat)
        cv_img = cv_img.astype(np.float32)
        cv_img = cv_img * (1 - jitter_range) + jitter_range * mean
        cv_img = np.where(cv_img > 255, 255, cv_img)
        cv_img = cv_img.astype(np.uint8)
        return cv_img
    """
    图像亮度、饱和度、对比度调节，在调整范围内随机获得调节比例，并随机顺序叠加三种效果
    Args:
        cv_img(numpy.ndarray): 输入图像
        saturation_range(float): 饱和对调节范围，0-1
        brightness_range(float): 亮度调节范围，0-1
        contrast_range(float): 对比度调节范围，0-1
    Returns:
        亮度、饱和度、对比度调整后图像
    """
    if np.random.random() < u:
        saturation_ratio = np.random.uniform(-saturation_range, saturation_range)
        brightness_ratio = np.random.uniform(-brightness_range, brightness_range)
        contrast_ratio = np.random.uniform(-contrast_range, contrast_range)
        order = [0, 1, 2]
        np.random.shuffle(order)
        for i in range(3):
            if order[i] == 0:
                cv_img = saturation_jitter(cv_img, saturation_ratio)
            if order[i] == 1:
                cv_img = brightness_jitter(cv_img, brightness_ratio)
            if order[i] == 2:
                cv_img = contrast_jitter(cv_img, contrast_ratio)
        return cv_img
    return cv_img


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),sat_shift_limit=(-255, 255),val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def randomShiftScaleRotate(image, mask, shift_limit=(-0.0, 0.0),scale_limit=(-0.0, 0.0),rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),borderMode=cv2.BORDER_REFLECT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape
        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,borderValue=(0, 0,0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST, borderMode=borderMode,borderValue=(0, 0, 0,))
    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        angle = np.random.randint(1,4)
        for i in range(angle):
            image = np.rot90(image)
            mask = np.rot90(mask)
        # return image, mask
    return image, mask


def resize(image, gt,insize, outsize):
    x = np.random.randint(-128, 128)
    y = np.random.randint(-128, 128)
    if x < 0:
        if y < 0:
            image = image[0:x, 0:y, :]
            gt = gt[0:x, 0:y]
        else:
            image = image[0:x, y:insize, :]
            gt = gt[0:x, y:insize]
    else:
        if y < 0:
            image = image[x:insize, 0:y, :]
            gt = gt[x:insize, 0:y]
        else:
            image = image[x:insize, y:insize, :]
            gt = gt[x:insize, y:insize]
    image = cv2.resize(image, (outsize, outsize), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (outsize, outsize), interpolation=cv2.INTER_NEAREST)

    return image,gt


def grade(img):
    x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    mi = np.min(dst)
    ma = np.max(dst)
    res = (dst - mi) / (0.000000001 + (ma - mi))
    res[np.isnan(res)] = 0
    return res


def Raster2Bbox(rasterlab):
    bbox=[]
    lab = rasterlab.copy()
    labeled_img, num = label(255 * lab, background=0, return_num=True, connectivity=1)
    bboxs = regionprops(labeled_img)
    for index in range(num):
        if bboxs[index].area>400:
            bbox.append(bboxs[index].bbox)
    return bbox


class Mydataset(Dataset):
    def __init__(self, path, augment=False, transform=None, target_transform=None,lab_smooth=0):
        self.aug = augment
        self.file_path = os.path.dirname(path)
        data = pd.read_csv(path)  # 获取csv表中的数据
        imgs = []
        for i in range(len(data)):
            imgs.append((data.iloc[i, 0], data.iloc[i, 1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.lab_smooth=lab_smooth

    def __getitem__(self, item):
        if self.aug == False:
            imgn, labn = self.imgs[item]
            imgn = os.path.join(self.file_path, "images/" + imgn)
            label = os.path.join(self.file_path, "labels/" + labn)
            bgr_img = cv2.imread(imgn, -1)
            image = Image.fromarray(bgr_img)
            if self.transform is not None:
                image = self.transform(image)
            # gt = cv2.imread(label, 0)/255
            gt = cv2.imread(label, 0)
            gt_old = np.zeros((512,512), dtype=np.int)
            gt_new = np.zeros((512,512), dtype=np.int)
            gt_mov = np.zeros((512,512), dtype=np.int)
            gt_b = np.zeros((512, 512), dtype=np.int)


            gt_old[gt == 1] = 1
            gt_old[gt == 3] = 1
            gt_new[gt == 2] = 1
            gt_mov[gt == 3] = 1
            gt_b[gt == 1] = 1
            gt_b[gt == 2] = 1
            return image, gt_old, gt_new,gt_mov,gt_b, labn

        else:
            # 进行数据增强
            imgn, labn = self.imgs[item]
            imgn = os.path.join(self.file_path, "images/" + imgn)
            labn = os.path.join(self.file_path, "labels/" + labn)
            # if self.lab_smooth>0:
            #     gt = cv2.GaussianBlur(cv2.imread(labn, 0), ksize=(self.lab_smooth, self.lab_smooth),sigmaX=self.lab_smooth)
            # else:
            gt = cv2.imread(labn, 0)
                # gt = cv2.imread(labn, 0)/255
                # gt[gt > 2] = 0
                # gt[gt == 2] = 1
            image = cv2.imread(imgn, -1)


            # gt = cv2.imread(label, 0)/255
            # gt1 = cv2.imread(lab1, 0) / 255
            # gt2 = cv2.imread(lab2, 0) / 255
            # image1 = cv2.imread(fn1, -1)
            # image2 = cv2.imread(fn2, -1)

            # batch=[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
            # sort1 = batch[np.random.randint(0,6)]
            # sort2 = batch[np.random.randint(0, 6)]
            # image1 = cv2.merge([image1[:, :, sort1[0]], image1[:, :, sort1[1]], image1[:, :, sort1[2]]])
            # image2 = cv2.merge([image2[:, :, sort2[0]], image2[:, :, sort2[1]], image2[:, :, sort2[2]]])


            # image, gt = motionblur(image, gt, blur=5, p=0.5)
            # image = random_color_jitter(image, saturation_range=0.5, brightness_range=0.5, contrast_range=0.5, u=0.5)
            # image = randomHueSaturationValue(image,hue_shift_limit=(-30, 30),sat_shift_limit=(-5, 5),val_shift_limit=(-15, 15))
            # image, gt = randomShiftScaleRotate(image, gt, shift_limit=(-0.1, 0.1), scale_limit=(-0.0, 0.0),
            #                                    aspect_limit=(-0.1, 0.1), rotate_limit=(-5, 5))
            # image, gt = randomHorizontalFlip(image, gt, u=0.5)
            # image, gt = randomVerticleFlip(image, gt, u=0.5)
            # image, gt = randomRotate90(image, gt, u=0.5)
            # image, gt = resize(image, gt, 1024, 640)
            #
            image = randomHueSaturationValue(image,
                                             hue_shift_limit=(-35, 35),
                                             sat_shift_limit=(-35, 35),
                                             val_shift_limit=(-35, 35))
            # image, gt = randomShiftScaleRotate(image, gt,
            #                                    shift_limit=(-0.15, 0.15),
            #                                    scale_limit=(-0.25, 0.5),
            #                                    aspect_limit=(-0.15, 0.15),
            #                                    rotate_limit=(-10, 10))

            # image1, image2, gt = randomHorizontalFlip(image1,image2, gt)
            image, gt = randomVerticleFlip(image, gt)
            image, gt = randomRotate90(image, gt)
            # image1, image2, gt = resize(image1, image2, gt, 512,512)
            # image, gt = resize(image, gt, 512, 512)

            # image = image[..., ::-1]  # bgr2rgb
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # grad = (255 * grade(gray)).astype(np.uint8)


            gt_old = np.zeros((512,512), dtype=np.uint8)
            gt_new = np.zeros((512,512), dtype=np.uint8)
            gt_mov = np.zeros((512,512), dtype=np.uint8)
            gt_b = np.zeros((512, 512), dtype=np.uint8)


            # gt_1 = np.zeros((512, 512), dtype=np.uint8)
            # gt_1[gt == 1] = 255
            # gt_agu = gt.copy()
            # builds, _ = cv2.findContours(gt_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # for build in builds:
            #     if random.random() > 0.5:
            #         build_coord = np.array(build.transpose(1, 0, 2))
            #         gt_agu = cv2.fillPoly(gt_agu, build_coord, 2)
            image, gt = randomShiftScaleRotate(image, gt, u=0.75,
                                                   shift_limit=(-0.15, 0.15),
                                                   scale_limit=(-0.25, 0.5),
                                                   aspect_limit=(-0.15, 0.15),
                                                   rotate_limit=(-15, 15))
            # gt_old[gt_agu == 1] = 1
            # gt_old[gt_agu == 3] = 1
            # gt_new[gt_agu == 2] = 1
            # gt_mov[gt_agu == 3] = 1
            # gt_b[gt_agu == 1] = 1
            # gt_b[gt_agu == 2] = 1

            # cv2.imwrite(r'H:\Projects\BE_Net\Result\05-16_20-53-00\ga.png', 80 * gt_agu)
            # cv2.imwrite(r'H:\Projects\BE_Net\Result\05-16_20-53-00\gt.png', 80 * gt)
            # gt_old[gt == 1] = 1
            # gt_old[gt == 3] = 1
            # gt_new[gt == 2] = 1
            # gt_mov[gt == 3] = 1
            # gt_b[gt == 1] = 1
            # gt_b[gt == 2] = 1

            if random.random() > 0.8:
                gt_old[gt == 3] = 1
                gt_new[gt == 1] = 1
                gt_new[gt == 2] = 1
                gt_mov[gt == 3] = 1
                gt_b[gt == 1] = 1
                gt_b[gt == 2] = 1
            else:
                gt_old[gt == 1] = 1
                gt_old[gt == 3] = 1
                gt_new[gt == 2] = 1
                gt_mov[gt == 3] = 1
                gt_b[gt == 1] = 1
                gt_b[gt == 2] = 1

            if random.random() > 0.8:
                bboxs = []
                for i in range(5):
                    x0 = random.randint(0, 448)
                    y0 = random.randint(0, 384)
                    x_size = random.randint(32, 64)
                    y_size = random.randint(x_size // 2, 2 * x_size)

                    # x_size = random.randint(32, 96)
                    # y_size = random.randint(x_size // 2, 2 * x_size)
                    # x0 = random.randint(0, 511-x_size)
                    # y0 = random.randint(0, 511-y_size)
                    bboxs.append([[x0, y0], [x0, y0 + y_size], [x0 + x_size, y0 + y_size]])

                gen_remove = np.zeros((512, 512), dtype=np.uint8)
                move_area = np.array(bboxs)
                gen_remove = cv2.fillPoly(gen_remove, move_area, 255)
                angle = random.randint(0, 90) - 45
                M = cv2.getRotationMatrix2D((256, 256), angle, 1)
                gen_remove = cv2.warpAffine(gen_remove, M, (512, 512))

                ker = np.ones((5, 5), np.uint8)
                gen_remove = cv2.erode(gen_remove, ker, iterations=1) // 255
                jmov = (1 - gt_b) * gen_remove
                gt_mov[jmov == 1] = 1


            img = Image.fromarray(image)
            if self.transform is not None:
                img = self.transform(img.copy())
            return img, gt_old.copy(), gt_new.copy(),gt_mov.copy(),gt_b.copy(), labn

    def __len__(self):
        return len(self.imgs)

