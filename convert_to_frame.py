import cv2
import os
import sys

def Video_to_Frames(Video_file):
    head, tail = os.path.split(Video_file)
    output_location = head + "/" + tail[:-4]+ "_frames"
    os.mkdir(output_location)

    vidcap = cv2.VideoCapture(Video_file) # name of the video
    success, image = vidcap.read()
    count = 0
    while success: 
        cv2.imwrite(output_location+"/frame%d.jpg" % count, image)     # save frame as JPEG file
        success, image = vidcap.read()
        #resized  = cv2.resize(image, (256, 256))
        #cv2.imwrite(output_location+"/frame%d.jpg" % count, resized)
        print('Read a new frame%d: '% count, success)
        count += 1

def resize(src, size, output_folder):
    head, tail = os.path.split(src)
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    #print('Original Dimensions : ', img.shape)

    resized = cv2.resize(img, size)
    #print('Resized Dimensions ', tail, ": ", resized.shape)
    cv2.imwrite(output_folder+"/"+tail, resized)

root = "/Volumes/IEMPROG/Education/data/data_frames/"
list_directories = os.listdir(root)
list_directories = list_directories[1:]
print("Main data folders: ", list_directories)

#print([name for name in os.listdir(path) if os.path.isdir(name)])
#path = "/Users/user/Desktop/UPMC/EPFL/Research/data/IR VIS VHA/SAVE_6_ir0_frames"

for dir_name in list_directories:
    path = root + dir_name
    print("\n--- Getting inside main folder: ", path)
    list_folders = os.listdir(path)
    list_folders = list_folders[1:]                                 #Problem with sys files like: .DS_Store

    for img_folder_name in list_folders:
        path_imgs = path + "/" + img_folder_name

        head, tail = os.path.split(path_imgs)
        print("head: ", head)
        print("tail: ", tail)
        list_images = os.listdir(path_imgs)
        list_images = list_images[1:]
        output_loc = head  + "/" + tail + "_resized/"
        os.mkdir(output_loc)
        print("\n------------- Started resizing imgs in ", path_imgs)
        for img in list_images:
            path_tail = path_imgs + "/" + img
            try:
                resize(path_tail, (256, 256), output_loc)
            except OError:
                print("Error with: ", path_tail)
            else:
                pass
        print("\n------------- DONE with ", path_tail)
    print("\n--- Out of main  folder: ", path)


#video_path = path + "/SAVE_1_visible.avi"
#Video_to_Frame(video_path)
"""
list_videos = os.listdir(path)
print(list_videos)
for video in list_videos:
    if video[-3:] == "avi":
        video_path = path+"/"+video
        print("We are in: ", video_path)
        Video_to_Frames(video_path)
        print("Done extracting frames from ", video)
        print("-----------------------------------------------------------------------")
"""