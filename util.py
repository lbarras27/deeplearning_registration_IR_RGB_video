import cv2
import os, sys
import numpy as np
import re
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sort by id frames (not have 'frame10' before 'frame2' for example)
def sortOrder(name):
    return int(re.findall("\d+", name)[0])

""" 
    Load all the rgb and ir frames.
    
    @path: the path where the RGB and IR dir containing the frames is.
    @img_rgb_path: name of the dir containing the rgb frames.
    @img_ir_path: name of the dir containing the ir frames.
    @color: if true, load the frames in color, otherwise load in grayscale
    
    @return all the RGB images and IR images
"""
def loadDataset(path, img_rgb_path, img_ir_path, color=True):
    img_src_ir = os.listdir(path + img_ir_path)
    img_src_rgb = os.listdir(path + img_rgb_path)

    img_src_ir.sort(key = sortOrder)
    img_src_rgb.sort(key = sortOrder)
    
    if color:
        mode = cv2.IMREAD_COLOR
    else:
        mode = cv2.IMREAD_GRAYSCALE

    imgs_rgb = []
    imgs_ir = []

    for ir, rgb in zip(img_src_ir, img_src_rgb):
        img_rgb = cv2.imread(path + img_rgb_path + "/" + rgb, mode)
        img_ir = cv2.imread(path + img_ir_path + "/" + ir, mode)
        imgs_rgb.append(img_rgb)
        imgs_ir.append(img_ir)

    imgs_rgb = np.array(imgs_rgb)
    imgs_ir = np.array(imgs_ir)
    
    return imgs_rgb, imgs_ir
 
"""    
    Apply gradient magnitude filter on img.
    
    @img: the image(s) we want to apply the filter.
    @ksize: the filter size (n x n)
    
    @return: The image(s) after to have apply the filter on them.
 
"""
def gradMagnitude(img, ksize = 3):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=ksize)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=ksize)
    sobelxy = np.sqrt(sobelx*sobelx + sobely*sobely)
    
    return sobelxy
    
"""
    Apply all the needed filters  to get contours on imgs_rgb and imgs_ir.
    
    @imgs_rgb: the rgb images.
    @imgs_ir: the ir images.
    
    @return: all the rgb and ir images after to have apply the filter on them (the contours of the images).
"""
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
    
""" 
    Save all the images in the dir specified by the path.
    
    @imgs: the images we want to save.
    @path: the path where we want save the images.
"""
def saveImages(imgs, path):
    for i in range(imgs.shape[0]):
        cv2.imwrite(path+"/frame"+str(i)+".jpg", imgs[i])
        
""" Split imgs in two parts: the first part is imgs without black images 
    and the second part is imgs only composed of black images (to check which images are blacks).
    
    @imgs: the images where we want to remove the black images.
    @threshold: under this threshold the pixels values are considered as black.
    @prob: the ratio of black pixels.
    
    @return: the images whitout black images, the black images and id_noBlack_frames is used to keep the real numero of the frames.
"""
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

"""    
    Prepare the data to can put it in the network.
    
    @imgs: the images that we want to parse.
    @size: the number of images we want to keep (0:size)
    
    @return the parsed images.
"""
def parseImages(imgs, size):
    imgsDimSup = torch.from_numpy(imgs).float()
    imgsDimSup = imgsDimSup[None, ...]
    imgsDimSup = imgsDimSup[:, :size, :, :]
    imgsDimSup = imgsDimSup.permute(1, 0, 2, 3)
    #normalize the pixel value between [0, 1]
    imgsDimSup = imgsDimSup/255
    
    return imgsDimSup
    
"""
    Print the results of the networks. 
    RGB image, IR image, warp image, difference between RGB and IR and difference between RGB and warp respectively.
    
    @imRGB: the rgb image.
    @imIR: the ir image.
    @warp: the output image of the network (IR after transformation).
    @cmap: how print the images.
"""
def printResult(imRGB, imIR, warp, cmap='gray'):
    plt.figure(figsize=(15, 15))
    
    plt.subplot(151)
    plt.title("RGB Image")
    plt.imshow(imRGB[0, ...].numpy(), cmap='gray')

    plt.subplot(152)
    plt.title("IR Image")
    plt.imshow(imIR[0, ...].numpy(), cmap='gray')
    
    plt.subplot(153)
    plt.title("Warp")
    warp = warp[0, ...].detach().numpy()
    warp = warp[0, ...]
    plt.imshow(warp, cmap='gray')

    plt.subplot(154)
    plt.title("Diff RGB and IR")
    plt.imshow(imIR[0, ...].numpy() - imRGB[0, ...].numpy(), cmap=cmap)

    plt.subplot(155)
    plt.title("Diff RGB and Warp")
    plt.imshow(-imRGB[0, ...].numpy() + warp, cmap=cmap)
    

"""
    Get the images with no idImage.
"""
def getImagesById(imgsRGB, imgsIR, idImage, model):
    warp, flow = model(imgsIR[None, idImage], imgsRGB[None, idImage])
    return imgsRGB[idImage], imgsIR[idImage], warp
    
"""
    Save the frames in a video with the number of fps indicate by fps.
    
    @frames: the frames we want to save in a video.
    @fps: the number of frames per second.
    @width: the width of the frames.
    @height: the height of the frames.
    @outputName: name of the video.
    @fourCC: the codec to use.
"""  
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
    
"""
    Superpose two images in applying alpha blending.
    
    @imgsDown: the main image(s).
    @imgsUp: the image(s) we want to superpose on imgsDown.
    @alpha: the ratio for the first image.
    @beta: the ratio for the second image.
    
    @return: the superposition of the two images.
"""
def superposeTwoImages(imgsDown, imgsUp, alpha=0.65, beta=0.35):
    res = []
    for i in range(imgsDown.shape[0]):
        res.append(cv2.addWeighted(imgsDown[i], alpha, imgsUp[i], beta, 0))
    imFinal = np.array(res)
    
    return imFinal
    
"""
    Apply a threshold on the images imgs_ir.

    @imgs_ir: is the images we want to apply the threshold on them (N, H, W, C=3).
    @number_images: is the numbers of images we want to apply the threshold.
    @thresh_value: is the threshold value we apply on the blue component of the images.
    
    @return: the threshold version of the images.
"""
def thresh_ir_images(imgs_ir, number_images, thresh_value=160):
    thresh_img = []
    for n in range(0, number_images):
        test_thresh = imgs_ir[n, :, :, 0].copy()
        for i in range(0, test_thresh.shape[0]):
            for j in range(0, test_thresh.shape[1]):
                if test_thresh[i, j] > thresh_value:
                    test_thresh[i, j] = 0
                else:
                    test_thresh[i, j] = 255
        thresh_img.append(test_thresh)
    thresh_img = np.array(thresh_img)
    
    return thresh_img
    
"""
    Useful function to have the good parameters for the canny filter according to the image 
    and apply canny filter.
    
    @image: the image that we want to apply the canny filter.
    @sigma: parameter to regulate the results.
    
    @return: the image containing only the edges.
"""
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged
    
    
"""
    Convert all the video in the path_DATA directory in frames in the new directory path_dst.
    
    @path_DATA: path to the directory DATA. (the path must contain DATA: ex: /home/DATA).
    @path_dst: path where the frames will be save.
    @resize_ir: if True so we resize the ir frames in size_ir.
    @size_ir: the size of the new ir frames.
"""
def convert_all_videos_in_frames(path_DATA, path_dst="DATA2", resize_ir = True, size_ir = (320, 240)):
    path_dst_rgb = path_dest+"/"+"RGB_frames"
    path_dst_ir = path_dest+"/"+"IR_frames"
    if not os.path.exists(path_dest):
        os.makedirs(path_dest)
        os.makedirs(path_dst_rgb)
        os.makedirs(path_dst_ir)
     
    l = os.listdir(path_DATA)
    direct = []
    for elem in l:
        if "IR" in elem:
            direct.append(elem)
            
    count_rgb = 0
    count_ir = 0
    for d in direct:
        video = os.listdir(path_DATA + d)
        rgb_video = []
        ir_video = []
        for e in video:
            if "visible" in e and ".avi" in e:
                rgb_video.append(e)
            elif "ir0" in e and ".avi":
                ir_video.append(e)

        for vid in rgb_video:
            vidcap = cv2.VideoCapture(path_DATA + d + "/" + vid)
            success,image = vidcap.read()
            while success:
                cv2.imwrite(path_dst_rgb + "/frame%d.jpg" % count_rgb, image)
                success,image = vidcap.read()
                count_rgb += 1
        
        for vid in ir_video:
            vidcap = cv2.VideoCapture(path_DATA + d + "/" + vid)
            success,image = vidcap.read()
            if resize_ir:
                image = cv2.resize(image, size_ir)
            while success:
                cv2.imwrite(path_dst_ir + "/frame%d.jpg" % count_ir, image)
                success,image = vidcap.read()
                if success:
                    if resize_ir:
                        image = cv2.resize(image, size_ir)
                count_ir += 1
                
"""
These following functions are useful for applying the output transformation of the network 
on the originals images (colored images)
"""

def parseColoredImages(imgs_get, beg, end):
    imgs = imgs_get[beg:end, ...]
    imgs = torch.from_numpy(imgs).float()
    imgs = imgs/255
    imgs = imgs.permute(0, 3, 1, 2)
    
    return imgs
    
def deParseColoredImages(imgs):
    res = imgs.permute(0, 2, 3, 1)
    res = res.detach().numpy()

    return res

def applyFlowToColoredImages(imgs, flow, beg, end):
    res_ir_gray = parseColoredImages(imgs, beg, end)
    
    x = F.grid_sample(res_ir_gray, flow)

    x = deParseColoredImages(x)
    
    return x

def applyFlowToColoredImages_netMain(imgs, flow, beg, end, model):
    res_ir_gray = parseColoredImages(imgs, beg, end)
    
    x = model.spatial_transform(res_ir_gray, flow)

    x = deParseColoredImages(x)
    
    return x

    
def getAllFlows(imgs_ir, imgs_rgb, model):
    flows = []
    for i in range(0, imgs_ir.shape[0]):
        _, flow = model(imgs_ir[None, i, ...].cpu(), imgs_rgb[None, i, ...].cpu())
        flow = flow[0].detach().numpy()
        flows.append(flow)
    flows = torch.tensor(flows)
    
    return flows

def printFinal(warp, img_rgb, img_ir, warp_rgb, rgb_ir):
    plt.figure(figsize=(15, 15))

    plt.subplot(151)
    plt.title("Image RGB")
    plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB), cmap='gray')

    plt.subplot(152)
    plt.title("Image IR")
    plt.imshow(cv2.cvtColor(img_ir, cv2.COLOR_BGR2RGB), cmap='gray')
    
    plt.subplot(153)
    plt.title("Warp")
    plt.imshow(cv2.cvtColor(warp, cv2.COLOR_BGR2RGB), cmap='gray')
    
    plt.subplot(154)
    plt.title("IR and RGB")
    plt.imshow(cv2.cvtColor(rgb_ir, cv2.COLOR_BGR2RGB), cmap='gray')
    
    plt.subplot(155)
    plt.title("Warp and RGB")
    plt.imshow(cv2.cvtColor(warp_rgb, cv2.COLOR_BGR2RGB), cmap='gray')

"""
    Print the loss function graph.
    
    @epoch: list containing loss for each epoch
    @title: plot tittle 
"""
def printGraphLoss(epoch, title):
    x = list(range(0, len(epoch)))
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(x, epoch);
    plt.title(title)
    ax = ax.set(xlabel='epoch', ylabel='loss')
    
    
"""
    Split the dataset in train and test set.
    
    @imgs_rgb: the rgb images to split.
    @imgs_ir: the ir images to split.
    @imgs_mask_rgb: the rgb masks to split.
    @imgs_mask_ir: the ir masks to split.
    @split: the indice where we want to split.
    @shuffle: if shuffle is true, the images are shuffles.
    
    @return: the rgb train, ir train set and the rgb, ir test set.
"""
def splitDataset(imgs_rgb, imgs_ir, imgs_mask_rgb, imgs_mask_ir, split, shuffle=False):
    if shuffle:
        indice_train = np.random.choice(imgs_rgb.shape[0], split, replace=False)
        mask_test = np.ones(imgs_rgb.shape[0], dtype=bool)
        mask_test[indice_train] = False
        mask_train = ~mask_test
        
        imgs_rgb_train = imgs_rgb[mask_train]
        imgs_ir_train = imgs_ir[mask_train]
        imgs_rgb_test = imgs_rgb[mask_test]
        imgs_ir_test = imgs_ir[mask_test]
        
        imgs_mask_rgb_train = imgs_mask_rgb[mask_train]
        imgs_mask_ir_train = imgs_mask_ir[mask_train]
        imgs_mask_rgb_test = imgs_mask_rgb[mask_test]
        imgs_mask_ir_test = imgs_mask_ir[mask_test]
    else:
        imgs_rgb_train = imgs_rgb[0:split]
        imgs_ir_train = imgs_ir[0:split]
        imgs_rgb_test = imgs_rgb[split:imgs_rgb.shape[0]]
        imgs_ir_test = imgs_ir[split:imgs_ir.shape[0]]
        
        imgs_mask_rgb_train = imgs_mask_rgb[0:split]
        imgs_mask_ir_train = imgs_mask_ir[0:split]
        imgs_mask_rgb_test = imgs_mask_rgb[split:imgs_mask_rgb.shape[0]]
        imgs_mask_ir_test = imgs_mask_ir[split:imgs_mask_ir.shape[0]]
    
    return imgs_rgb_train, imgs_ir_train, imgs_rgb_test, imgs_ir_test, imgs_mask_rgb_train, imgs_mask_ir_train, imgs_mask_rgb_test, imgs_mask_ir_test