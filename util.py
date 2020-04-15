import cv2
import os, sys
import numpy as np
import re
import torch
import matplotlib.pyplot as plt

path = "DATA/IR VIS HAK/"

# sort by id frames (not have 'frame10' before 'frame2' for example)
def sortOrder(name):
    return int(re.findall("\d+", name)[0])

# load dataset.
# return all the RGB images and IR images
def loadDataset(path, img_rgb_path, img_ir_path):
    img_src_ir = os.listdir(path + img_ir_path)
    img_src_rgb = os.listdir(path + img_rgb_path)

    img_src_ir.sort(key = sortOrder)
    img_src_rgb.sort(key = sortOrder)

    imgs_rgb = []
    imgs_ir = []

    for ir, rgb in zip(img_src_ir, img_src_rgb):
        img_rgb = cv2.imread(path + img_rgb_path + "/" + rgb, cv2.IMREAD_GRAYSCALE)
        img_ir = cv2.imread(path + img_ir_path + "/" + ir, cv2.IMREAD_GRAYSCALE)
        imgs_rgb.append(img_rgb)
        imgs_ir.append(img_ir)

    imgs_rgb = np.array(imgs_rgb)
    imgs_ir = np.array(imgs_ir)
    
    return imgs_rgb, imgs_ir
    

def loadDataset2(path, img_rgb_path, img_ir_path):
    img_src_ir = os.listdir(path + img_ir_path)
    img_src_rgb = os.listdir(path + img_rgb_path)

    img_src_ir.sort(key = sortOrder)
    img_src_rgb.sort(key = sortOrder)

    imgs_rgb = []
    imgs_ir = []

    for ir, rgb in zip(img_src_ir, img_src_rgb):
        img_rgb = cv2.imread(path + img_rgb_path + "/" + rgb)
        img_ir = cv2.imread(path + img_ir_path + "/" + ir)
        imgs_rgb.append(img_rgb)
        imgs_ir.append(img_ir)

    imgs_rgb = np.array(imgs_rgb)
    imgs_ir = np.array(imgs_ir)
    
    return imgs_rgb, imgs_ir
    
# apply gradient magnitude filter on img
def gradMagnitude(img, ksize = 3):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=ksize)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=ksize)
    sobelxy = np.sqrt(sobelx*sobelx + sobely*sobely)
    
    return sobelxy
    
# apply all the needed filters on imgs_rgb and imgs_ir 
def applyFilterOnDataset(imgs_rgb, imgs_ir):
    imgs_rgb_filtered = []
    imgs_ir_filtered = []

    for i in range(imgs_rgb.shape[0]):
        imgs_rgb_gauss = cv2.GaussianBlur(imgs_rgb[i], (7, 7), 0.5)
        grad_magn_rgb = gradMagnitude(imgs_rgb_gauss)
        imgs_rgb_filtered.append(grad_magn_rgb)
        
        imgs_ir_gauss = cv2.GaussianBlur(imgs_ir[i], (7, 7), 0.5)
        grad_magn_ir = gradMagnitude(imgs_ir_gauss)
        imgs_ir_filtered.append(grad_magn_ir)
        
    imgs_rgb_filtered = np.array(imgs_rgb_filtered)
    imgs_ir_filtered = np.array(imgs_ir_filtered)
    
    return imgs_rgb_filtered, imgs_ir_filtered
    
# save imgs in path
def saveImages(imgs, path):
    for i in range(imgs.shape[0]):
        cv2.imwrite(path+"/frame"+str(i)+".jpg", imgs[i])
        
# split imgs in two parts: the first part is imgs without black images 
# and the second part is imgs only composed of black images (to check which images are blacks)
# id_noBlack_frames is used to keep the real numero of the frames 
def splitBlackImages(imgs, threshold=20, prob=0.95):
    imgs_rgb_noBlack = []
    imgs_rgb_black = []
    id_noBlack_frames = []
    for i in range(imgs.shape[0]):
        if (imgs[i] < threshold).sum()/(3*imgs.shape[1]*imgs.shape[2]) > prob:
            imgs_rgb_black.append(imgs[i])
        else:
            imgs_rgb_noBlack.append(imgs[i])
            id_noBlack_frames.append(i)
    
    imgs_rgb_nb = np.array(imgs_rgb_noBlack)
    imgs_rgb_b = np.array(imgs_rgb_black)

    return imgs_rgb_nb, imgs_rgb_b, id_noBlack_frames
    
# prepare the data to can put it in the network
def parseImages(imgs, size):
    imgsDimSup = torch.from_numpy(imgs).float()
    imgsDimSup = imgsDimSup[None, ...]
    imgsDimSup = imgsDimSup[:, :size, :, :]
    imgsDimSup = imgsDimSup.permute(1, 0, 2, 3)
    imgsDimSup = imgsDimSup/255
    
    return imgsDimSup
    
# print the images 
def printResult(imRGB, imIR, warp, cmap='gray'):
    plt.figure(figsize=(15, 15))
    plt.subplot(151)
    plt.title("Warp")
    warp = warp[0, ...].detach().numpy()
    warp = warp[0, ...]
    plt.imshow(warp, cmap='gray')

    plt.subplot(152)
    plt.title("RGB Image")
    plt.imshow(imRGB[0, ...].numpy(), cmap='gray')

    plt.subplot(153)
    plt.title("IR Image")
    plt.imshow(imIR[0, ...].numpy(), cmap='gray')

    plt.subplot(154)
    plt.title("Diff RGB and IR")
    plt.imshow(imIR[0, ...].numpy() - imRGB[0, ...].numpy(), cmap=cmap)

    plt.subplot(155)
    plt.title("Diff RGB and Warp")
    plt.imshow(-imRGB[0, ...].numpy() + warp, cmap=cmap)
    
# get the images with no idImage
def getImagesById(imgsRGB, imgsIR, idImage, model):
    warp, flow = model(imgsIR[None, idImage])
    return imgsRGB[idImage], imgsIR[idImage], warp
    
def getImagesById2(imgsRGB, imgsIR, idImage, model):
    warp, flow = model(imgsIR[None, idImage], imgsRGB[None, idImage])
    return imgsRGB[idImage], imgsIR[idImage], warp
    
    
def saveVideo(frames, fps, width, height, outputName, fourCC='DIVX'):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*fourCC)
    out = cv2.VideoWriter(outputName, fourcc, fps, (width, height))

    for i in range(frames.shape[0]):
    
        # write the frame
        out.write(frames[i])

        cv2.imshow('frame', frames[i])

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
    

def superposeTwoImages(imgsDown, imgsUp, alpha=0.65, beta=0.35):
    res = []
    for i in range(imgsDown.shape[0]):
        res.append(cv2.addWeighted(imgsDown[i], alpha, imgsUp[i], beta, 0))
    imFinal = np.array(res)
    
    return imFinal