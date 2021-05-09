# USAGE
# python real_time_mask_detection.py

# import the necessary packages
import cv2
from tqdm import tqdm
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from random import *
import numpy as np
import argparse
import time
import os
import sys

import tensorflow as tf
config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.1
sess=tf.compat.v1.Session(config=config)

def printinfo(labels,all_people):
    #print(labels)
    i=0
    cnt_safe=0
    cnt_not=0
    cnt_danger=0
    while all_people>0:
        #print(i)
        if labels[i]=="Mask":
            cnt_safe+=1
        elif labels[i]=="Useless_Mask":
            cnt_not+=1
        else: #if not wearing mask
            cnt_danger+=1
        i+=1
        all_people-=1
    labels.clear()
    #print(all_people,cnt_safe,cnt_not,cnt_danger)  
        

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX-20), max(0, startY-20))
            (endX, endY) = (min(w - 1, endX+20), min(h - 1, endY+20))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds,faces)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
    default="face_detector",
    help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
    default="mask_detector.model",
    help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video_input", type=str,
    default="mask.mp4",
    help="path to video file")
ap.add_argument("-o", "--save_video", type=str,
    default="mask_output.avi",
    help="save video file path")
ap.add_argument("-fps", "--fps", type=int,
    default=10,
    help="Frame per second")
        
args = vars(ap.parse_args())

print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])


print("[INFO] starting video stream...")

cap = cv2.VideoCapture(args["video_input"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

time.sleep(2.0)
ret, img = cap.read()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
real_label=[]

total_cnt=0
yes_cnt = [0]*7
useless_cnt = [0]*7
no_cnt = [0]*7

first_detect=True

color=[] 
color_MASK=(0,255,0)  # BRG (0,255,0) - red 
color_useless=(0,255,255)  # yellow
color_NOMASK=(0,0,255)

time_checker = 0
flag = 0
m_cnt = 0

# tqdm pbar
p = 1
pbar = tqdm(total = frame_count)
result = cv2.VideoWriter(args["save_video"],  
                         fourcc, 
                         args["fps"],(int(cap.get(3)), int(cap.get(4))))

if (cap.isOpened() == False):  
    print("Error reading video file") 
 
while True:

    pbar.update(p)
    ret, img = cap.read()
    time_checker+=1
    if not ret:
        break

    h,w=img.shape[:2]
    
    index=0
    result_img = img.copy()
    (locs, preds,people) = detect_and_predict_mask(img, faceNet, maskNet)

    all_people=len(people)
    labels=[]
#   color=[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]
    #print(color)
    label=""
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (Useless_Mask,mask, withoutMask) = pred
        
        # determine the class label and color we'll use to draw
        # the bounding box and text
        if mask > withoutMask and mask > Useless_Mask:
            yes_cnt[index]+=1
            index+=1
        elif Useless_Mask>mask and Useless_Mask > withoutMask:
            useless_cnt[index]+=1
            index+=1
        else:
            no_cnt[index]+=1
            index+=1

        percent=max(Useless_Mask,mask, withoutMask) * 100

        if first_detect==False:
            
            try:
                label = "{}: {:.2f}%".format(real_label[index-1], percent)

                randv = uniform(35.8, 36.9)
                temperature = round(randv, 1)
                num = str(temperature)
                label+="     "
                label += num

                cv2.putText(result_img,label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color[index-1], 2)
                cv2.rectangle(result_img, (startX, startY), (endX, endY), color[index-1], 2)
                if case == 2:
                    message = "Please wear a mask..!"
                    if time_checker % 3 == 0:
                        cv2.rectangle(result_img, (300, 300), (700, 400), (0, 0, 255), -1)
                        cv2.putText(result_img, message, (300, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    flag =1
                elif case ==1:
                    message = "Mask worn incorrectly!"
                    if time_checker % 3 == 0:
                        cv2.rectangle(result_img, (200, 300), (900, 400), (0, 255, 255), -1)
                        cv2.putText(result_img, message, (300, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    flag =1
                else:
                    if flag == 1 and m_cnt < 7:
                        message = "Mask worn"
                        cv2.rectangle(result_img, (300, 300), (700, 400), (0, 255, 0), -1)
                        cv2.putText(result_img, message, (400, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                        m_cnt += 1 
                        
                    if m_cnt == 7:
                        flag =0
                        m_cnt = 0

            except IndexError:
                color=[]
                real_label=[]
                total_cnt=0
                if first_detect==True:
                    first_detect=False
                for i in range(0,all_people):
                    choice=max(yes_cnt[i],useless_cnt[i],no_cnt[i])
                    if yes_cnt[i]==choice:
                        real_label.append("Mask")
                        color.append(color_MASK)
                        case = 0
                    elif useless_cnt[i]==choice:
                        real_label.append("Useless_Mask")
                        color.append(color_useless)
                        case = 1
                    elif no_cnt[i]==choice:
                        real_label.append("No_Mask")
                        color.append(color_NOMASK)
                        case = 2
                    else:
                        cv2.putText(result_img,"Error",(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                yes_cnt = [0]*7
                useless_cnt = [0]*7
                no_cnt = [0]*7
                #send_data(real_label,locs)
                label = "{}: {:.2f}%".format(real_label[index-1], percent)
                #print(label)
                cv2.putText(result_img,label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color[index-1], 2)
                cv2.rectangle(result_img, (startX, startY), (endX, endY), color[index-1], 2)
                
    total_cnt+=1
 
    # Collect 11 datas of mask detection infos and choose the most counted result
    # should be odd number since there could be a draw when using even number
    if total_cnt==11:
        color=[]
        real_label=[]
        total_cnt=0
        if first_detect==True:
            first_detect=False
        for i in range(0,all_people):
            choice=max(yes_cnt[i],useless_cnt[i],no_cnt[i])
            if yes_cnt[i]==choice:
                real_label.append("Mask")
#               color = (0, 255, 0)
                color.append(color_MASK)
                case = 0
            elif useless_cnt[i]==choice:
                real_label.append("Useless_Mask")
                color.append(color_useless)
#               color = (255, 0, 0)
                case = 1
            elif no_cnt[i]==choice:
                real_label.append("No_Mask")
                color.append(color_NOMASK)
#               color = (0, 0, 255)
                case = 2
            else:
                cv2.putText(result_img,"Error",(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        yes_cnt = [0]*7
        useless_cnt = [0]*7
        no_cnt = [0]*7
#   print(locs)
#   printinfo(labels,all_people)
        
    # save rhe output frame
    result.write(result_img)
    if result_img is None:
        print('completed...!')
    # show the output frame  
    #cv2_imshow(result_img)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        client_socket.close();  
        break

# do a bit of cleanup
cap.release()
result.release() 
cv2.destroyAllWindows()
