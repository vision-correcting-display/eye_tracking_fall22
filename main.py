from __future__ import print_function
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
from torchvision import transforms
from PIL import Image
import glob
from mtcnn.mtcnn import MTCNN
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import dlib
import os

import zero_dce_model
import conventional

import time

###############################
#   DL Low Light Enhancement
###############################
# Zero DCE
def ZERO_DCE(DCE_net, dark_img_in):
    dark_img_in = (np.asarray(dark_img_in)/255.0)
    
    dark_img_in = torch.from_numpy(dark_img_in).float()
    dark_img_in = dark_img_in.permute(2,0,1)
    dark_img_in = dark_img_in.unsqueeze(0).to('cpu')
    
    _,enhanced_image,_ = DCE_net(dark_img_in)

    return enhanced_image
	
# EnlightenGAN
def ENLIGHTEN_GAN(dark_img_in, light_img_out):
    pass

def DEEP_RETINEX_DECOMP(dark_img_in, light_img_out):
    pass

#######################################
#   Conventional Low Light Enhancement
#######################################
# LIME
def LIME(dark_img_in, light_img_out):
    pass

def DUAL_ILLUMINATION(dark_img_in, light_img_out):
    pass

# Util Functions
def cvt_GRAY(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # frame_gray = cv.equalizeHist(frame_gray)
    
    return frame_gray
    
def cvt_RGB(frame):
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    return frame_rgb

def draw_FACE(frame, face_ROIs):
    if face_ROIs == None:
        return frame
    
    for faceROI in face_ROIs:
        x1, x2, y1, y2 = faceROI
        frame = cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

    return frame

def draw_EYE(frame, eyes_ROIs, method="68"):
    frame_drawn = frame
        
    if eyes_ROIs == None:
        return frame_drawn
    
    if method == "68":
        for eyesROI in eyes_ROIs:
            frame_drawn = cv.fillConvexPoly(frame, eyesROI, (255,0,0))
        
    else:
        for eyesROI in eyes_ROIs:
            x1, x2, y1, y2 = eyesROI
            w, h = x2-x1, y2-y1
            eye_center = ((x1+x2)//2, (y1+y2)//2)
            radius = int(round((w + h)*0.25))
            frame_drawn = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)
            # eyesROI =np.array([(x1,y1), (x1, y2), (x2, y1), (x2,y2)])
            # frame = cv.fillConvexPoly(frame, eyesROI, (255,0,0))
    
    return frame_drawn

def shape_to_np(shape, dtype='int'):
    coords = np.zeros((68,2), dtype=dtype)
    
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
        
    return coords

def eye_on_mask(shape, mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv.fillConvexPoly(mask, points, 255)
    return mask

def eyes_dilate_and_segment(frame, mask):
    kernel  = np.ones((9,9), np.uint8)
    mask    = cv.dilate(mask, kernel, 5)
    eyes    = cv.bitwise_and(frame, frame, mask=mask)
    
    mask    = (eyes == [0,0,0]).all(axis=2)
    eyes[mask] = [255, 255, 255]
    eyes_gray = cvt_GRAY(eyes)  
    
    return eyes_gray


def thresholding(eyes_gray):
    # threshold   = cv.getTrackbarPos('threshold', 'image')
    threshold   = 80
    # print(threshold)
    _, thresh   = cv.threshold(eyes_gray, threshold, 255, cv.THRESH_BINARY)
    thresh      = cv.erode(thresh, None, iterations=2)
    thresh      = cv.dilate(thresh, None, iterations=4)  
    thresh      = cv.medianBlur(thresh, 3)
    thresh      = cv.bitwise_not(thresh)
    
    return thresh
    
def contouring(frame, mid_point, thresh, right=False):
    cnts, _     = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # print(cnts)
    try:
        cnt         = max(cnts, key = cv.contourArea)
        M           = cv.moments(cnt)
        cx          = int(M['m10']/M['m00'])
        cy          = int(M['m01']/M['m00']) 
        
        if right: 
            cx  += mid_point     
        
        radius      = 5
        thickness   = 3
        frame = cv.circle(frame, (cx, cy), radius, (0, 0, 255), thickness)
         
    except:
        pass

    return frame

# Plotting Functions
def plot_image(img, title, size=(16, 8)):
    plt.figure(figsize= size)
    plt.title(title)
    plt.axis('off')
    plt.imshow(img, plt.cm.gray)

def plot_mul_images(imgs, titles, nrow, ncol, size= (8, 8)):
    fig     = plt.figure(figsize = size)
    total   = nrow* ncol 
    for idx in range(total):
        ax = plt.subplot(nrow, ncol, idx+1)
        
        if(titles[idx]):
            plt.title(titles[idx])
        plt.imshow(imgs[idx], cmap='gray')
        plt.axis('off')

    fig.tight_layout()
    plt.show()
    
def display_results(frame, faces, eyes):
    result = np.concatenate((frame, faces, eyes), axis=1)
    
    cv.imshow('Capture - Face and Eye detection', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
###################################################
#   Face Detector
#       Input   : One Frame
#       Output  : Faces in (x1, x2, y1, y2)
###################################################
# Face Detector: Haar Cascade
def face_haar_cascade(face_detector, frame):
    # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces      = face_detector.detectMultiScale(frame)
    #-- Store the faces' coordinates
    face_ROIs  = []
    for (x,y,w,h) in faces:
        face_cor = (x, x+w, y, y+h)
        # center = (x + w//2, y + h//2)
        # frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        # faceROI = frame[y:y+h,x:x+w]
        face_ROIs.append(face_cor)
    
    return face_ROIs

# Face Detector: HOG Based Frontal Face Detector
def face_hog(face_detector, upsample, frame_gray):
    #-- Detect faces
    faces      = face_detector(frame_gray, upsample)
    #-- Store the faces' coordinates
    face_ROIs  = []
    for result in faces:
        face_cor = (result.left(), result.right(), result.top(), result.bottom())
        face_ROIs.append(face_cor)
    
    return face_ROIs

# Face Detector: MTCNN
def face_mtcnn(face_detector, frame):
    #-- Detect faces
    faces      = face_detector.detect_faces(frame)
    #-- Store the faces' coordinates
    face_ROIs  = []
    for result in faces:
        x,y,w,h     = result['box']
        face_cor    = (x, x+w, y, y+h)
        face_ROIs.append(face_cor)
    
    return face_ROIs

# Face Detector: DNN
def face_dnn(face_detector, frame_blob, img_shape):
    h, w = img_shape[:2]
    #-- Detect faces
    face_detector.setInput(frame_blob)
    faces       = face_detector.forward() 
    #-- Store the faces' coordinates
    face_ROIs  = []
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            box = faces[0, 0, i, 3:7] * np.array([w,h,w,h])
            (x1, y1, x2, y2)    = box.astype("int")
            face_cor            = (x1, x2, y1, y2)
            face_ROIs.append(face_cor)
            
    return face_ROIs
########################################################
#   Eye Detector
#       Input   : Faces of one frame
#       Output  : Eyes in (x1, x2, y1, y2) for one frame
########################################################
# Eye Detector: Haar Cascade
def eye_haar_cascade(eyes_detector, frame, face_ROIs):
    #-- Store the eyes' coordinates
    eye_ROIs = []
    
    if face_ROIs == None:
        return eye_ROIs
    
    for (x1,x2,y1,y2) in face_ROIs:
        # center = (x + w//2, y + h//2)
        # frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame[y1:y2,x1:x2]
        #-- In each face, detect eyes
        eyes = eyes_detector.detectMultiScale(faceROI)
        
        for (x,y,w,h) in eyes:
            # eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            # radius = int(round((w2 + h2)*0.25))
            # frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
            eye_cor = (x1+x, x1+x+w, y1+y, y1+y+h)
            
            eye_ROIs.append(eye_cor)
        
    return eye_ROIs

# Eye Detector: 68 Facial Landmark
def eye_68_facial_landmark(eyes_detector, frame_gray, face_ROIs, left_cor, right_cor):
    frame_gray = cvt_GRAY(frame_gray)
    eye_ROIs = []
    facial_landmarks = []
    
    if face_ROIs == None:
        return eye_ROIs
    
    for (x1,x2,y1,y2) in face_ROIs:
        rect = dlib.rectangle(x1, y1, x2, y2)
        #-- In each face, detect 68 point facial landmark
        facial_landmarks= eyes_detector(frame_gray, rect)
        # get the coords for each facial landmarks
        facial_landmarks= shape_to_np(facial_landmarks)
        # get left eyes 6 points
        left_points     = np.array([facial_landmarks[i] for i in left_cor], dtype=np.int32)
        # get right eyes 6 points
        right_points    = np.array([facial_landmarks[i] for i in right_cor], dtype=np.int32)
        
        
        eye_ROIs.append(left_points)
        eye_ROIs.append(right_points)
        
    return facial_landmarks, eye_ROIs

# For Real Time DETECTING
def enhanceAndDisplay(frame, low_light_model):
    start_time = time.time()
    img_original= frame.copy()
    if(low_light_model == "LIME"):
        frame = conventional.enhance_image_exposure(frame, 0.6, 0.15, False,
                                            sigma=3, bc=1, bs=1, be=1, eps=1e-3)
    elif(low_light_model == "DUAL"): 
        frame = conventional.enhance_image_exposure(frame, 0.6, 0.15, True,
                                            sigma=3, bc=1, bs=1, be=1, eps=1e-3)
    elif(low_light_model == "DCE"):
        frame = torch.from_numpy(frame/255.0).float()
        frame = frame.permute(2,0,1)
        frame = frame.unsqueeze(0).to('cpu')
        
        _,frame,_ = DCE_net(frame)
        frame = frame.squeeze(0).permute(1,2,0).detach().numpy()
        
    process_time = time.time() - start_time
    frame = cv.putText(img= frame, text=f"{process_time:.2f} sec", org=(80, 80), 
            fontFace=cv.FONT_HERSHEY_DUPLEX, 
            fontScale=1.0, color=(125, 246, 55), thickness=3)
    
    print(f"process time = {process_time:.2f} sec")
    cv.imshow('Before Enhances', img_original)
    cv.imshow('Capture - Low Light Enhancement', frame)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def detectAndDisplay(frame, face_detector, eyes_detector, low_light_model):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_detector.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes = eyes_detector.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
    
    cv.imshow('Capture - Face detection', frame)
    cv.waitKey(0)
    cv.destroyAllWindows()


############################
# Initialing Low Light NN
############################
DCE_net = zero_dce_model.enhance_net_nopool().to('cpu')
DCE_net.load_state_dict(torch.load('Zero-DCE/Zero-DCE_code/snapshots/Epoch99.pth', map_location='cpu'))

############################
# Initialing Face Detector
############################
# Haar Cascade
face_haar_detector = cv.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')
# HOG
hog_upsample        = 0 # upsample the input image before the face detection: if face smaller than 8x80
face_hog_detector   = dlib.get_frontal_face_detector()
# MTCNN
face_mtcnn_detector = MTCNN()
# DNN
dnn_model_file      = "./models/res10_300x300_ssd_iter_140000.caffemodel"
dnn_config_file     = "./models/deploy.prototxt"   
face_dnn_detector   = cv.dnn.readNetFromCaffe(dnn_config_file, dnn_model_file)

############################
# Initialing Eye Detector
############################
# Haar Cascade
eye_haar_detector   = cv.CascadeClassifier('data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

# 68 Point Facial Landmark
eye_68_detector     = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
left_eye_cor        = [36, 37, 38, 39, 40, 41]
right_eye_cor       = [42, 43, 44, 45, 46, 47]

# 5 Point Facial Landmark [TBD]

def ALL_METHODS():
    ############################
    # Operation
    ############################
    faces_dict  = {}
    eyes_dict   = {} 
    facial_landmark_dict = {}

    cv.namedWindow('image')
    def nothing(x):
        pass
    cv.createTrackbar('threshold', 'image', 0, 255, nothing)

    images = os.listdir('my_images')   
    
    for image in images:
        if(image[-4:] != ".JPG" and image[-5:] != ".jpeg"): 
            continue
        if(image != "low8_real_A.jpeg"):
            continue
        
        print("Processing:", image)
        
        img = cv.imread(os.path.join('my_images', image))
        # detectAndDisplay(img)
        
        ###############################
        #   Face Detection
        ###############################
        # Haar Cascade
        img_haar        = img.copy() 
        face_haar_ROIs  = face_haar_cascade(face_haar_detector, img_haar)
        faces_dict["HAAR"] = face_haar_ROIs
        
        # HOG
        img_hog         = img.copy()
        img_hog_gray    = cvt_GRAY(img_hog)
        face_hog_ROIs   = face_hog(face_hog_detector, hog_upsample, img_hog_gray)
        faces_dict["HOG"] = face_hog_ROIs
        
        # MTCNN
        img_mtcnn           = img.copy()
        face_mtcnn_ROIs     = face_mtcnn(face_mtcnn_detector, img_mtcnn)
        faces_dict["MTCNN"] = face_mtcnn_ROIs
        
        # DNN: resize to (300x300) to get the best result?
        img_dnn             = img.copy()
        img_dnn_blob        = cv.dnn.blobFromImage(cv.resize(img_dnn, (300, 300)),
                                1.0, (300, 300), (104.0, 117.0, 123.0))
        face_dnn_ROIs       = face_dnn(face_dnn_detector, img_dnn_blob, img_dnn.shape)
        faces_dict["DNN"]   = face_dnn_ROIs
        
        ###############################
        #  Eye Detection
        ###############################
        # Haar Cascade Face + Haar Cascade Eye
        # eye_haar_haar_ROIs      = eye_haar_cascade(eye_haar_detector, img_haar, face_haar_ROIs)
        # eyes_dict["HAAR_HAAR"]  = eye_haar_haar_ROIs
        
        # HOG Face + Haar Cascade Eye
        # eye_hog_haar_ROIs       = eye_haar_cascade(eye_haar_detector, img_hog, face_hog_ROIs)
        # eyes_dict["HOG_HAAR"]   = eye_hog_haar_ROIs
        
        # MTCNN Face + Haar Cascade Eye
        # eye_mtcnn_haar_ROIs     = eye_haar_cascade(eye_haar_detector, img_mtcnn, face_mtcnn_ROIs)
        # eyes_dict["MTCNN_HAAR"] = eye_mtcnn_haar_ROIs
        
        # DNN Face + Haar Cascade Eye
        # eye_dnn_haar_ROIs       = eye_haar_cascade(eye_haar_detector, img_dnn, face_dnn_ROIs)
        # eyes_dict["DNN_HAAR"]   = eye_dnn_haar_ROIs
        
        # Haar Cascade Face + 68 Facial Landmarks Eye
        eye_haar_68_lm, eye_haar_68_ROIs    = eye_68_facial_landmark(eye_68_detector, img_haar, face_haar_ROIs, left_eye_cor, right_eye_cor)
        eyes_dict["HAAR_68"]                = eye_haar_68_ROIs
        facial_landmark_dict["HAAR_68"]     = eye_haar_68_lm
        
        # HOG Face + 68 Facial Landmarks Eye
        eye_hog_68_lm, eye_hog_68_ROIs      = eye_68_facial_landmark(eye_68_detector, img_hog, face_hog_ROIs, left_eye_cor, right_eye_cor)
        eyes_dict["HOG_68"]                 = eye_hog_68_ROIs
        facial_landmark_dict["HOG_68"]      = eye_hog_68_lm
        
        # MTCNN Face + 68 Facial Landmarks Eye
        eye_mtcnn_68_lm, eye_mtcnn_68_ROIs  = eye_68_facial_landmark(eye_68_detector, img_mtcnn, face_mtcnn_ROIs, left_eye_cor, right_eye_cor)
        eyes_dict["MTCNN_68"]               = eye_mtcnn_68_ROIs
        facial_landmark_dict["MTCNN_68"]    = eye_mtcnn_68_lm
        
        # DNN Face + 68 Facial Landmarks Eye
        eye_dnn_68_lm, eye_dnn_68_ROIs      = eye_68_facial_landmark(eye_68_detector, img_dnn, face_dnn_ROIs, left_eye_cor, right_eye_cor)
        eyes_dict["DNN_68"]                 = eye_dnn_68_ROIs
        facial_landmark_dict["DNN_68"]      = eye_dnn_68_lm
        ###############################
        #  Plotting
        ###############################
        face_methods    = ["HAAR", "HOG", "MTCNN", "DNN"]
        # face_methods    = ["DNN"]
        eyes_methods    = ["68"]
        img_results     = []
        img_titles      = [] 
        
        plot_offset     = 0  # if not detect
        
        for face_method in face_methods: 
            img_results.append(cvt_RGB(img))
            img_titles.append("Original")
            
            # Draw Face
            img_copy        = img.copy() 
            img_faces       = draw_FACE(img_copy, faces_dict[face_method])
            
            img_results.append(cvt_RGB(img_faces))
            img_titles.append(f"{face_method} Face Detector")
            
            for eyes_method in eyes_methods:
                img_faces_copy  = img_faces.copy()
                
                img_facial_landmark = draw_EYE(img_copy, eyes_dict[face_method+'_'+eyes_method], eyes_method)
                
                img_results.append(cvt_RGB(img_facial_landmark))
                img_titles.append(f"{eyes_method} Eye Detector")
                
                img_mask        = np.zeros(img_copy.shape[:2], dtype=np.uint8)
                img_eyes        = draw_EYE(img_mask, eyes_dict[face_method+'_'+eyes_method], eyes_method)
                
                img_results.append(img_eyes)
                img_titles.append(f"{eyes_method} Eye on Mask")
                
                # Dilate and Segment the eyes
                img_copy        = img.copy()
                img_dil_seg_eyes= eyes_dilate_and_segment(img_copy, img_eyes) 
                
                img_results.append(img_dil_seg_eyes)
                img_titles.append("Segment the eyes")
                
                # Thresholding
                thresh = thresholding(img_dil_seg_eyes)
                
                img_results.append(thresh)
                img_titles.append("After Thresholding")
                
                # Contouring
                if(len(facial_landmark_dict[face_method+'_'+eyes_method]) == 0):
                    print("Facial Landmark not detect.")
                    plot_offset += 1
                    break
                else:
                    mid_point = \
                        (facial_landmark_dict[face_method+'_'+eyes_method][42][0]+\
                        facial_landmark_dict[face_method+'_'+eyes_method][39][0]) //2
                
                    # Contour left eye
                    img_copy = img.copy()
                    img_left_eye    = contouring(img_copy, mid_point, thresh[:, 0:mid_point], False)
                    # Contour right eye
                    img_both_eyes   = contouring(img_left_eye, mid_point, thresh[:, mid_point:], True)
                    
                    img_results.append(cvt_RGB(img_both_eyes))
                    img_titles.append("Result")
        
        plot_mul_images(img_results, img_titles, len(face_methods), len(eyes_methods)+ 6 - plot_offset)
        
        break

def DETECTION(camera_device, low_light_model):
    ################################
    #   Using Camera
    ################################
    # -- 2. Read the video stream
    cap = cv.VideoCapture(camera_device)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)

    while True:
        ret, frame = cap.read()
        
        # image resize
        width       = 240
        height      = 240
        frame       = cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
        
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        enhanceAndDisplay(frame, low_light_model)
        # detectAndDisplay(frame, face_haar_detector, eye_haar_detector)
        if cv.waitKey(10) == 27:
            break
    
def REAL_TIME_TRACKING(camera_device, face_detector, low_light_model):
    cap         = cv.VideoCapture(camera_device)
    ret, img    = cap.read()
    
    # image resize
    width       = 320
    height      = 240
    img         = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)

    thresh      = img.copy()

    cv.namedWindow('image')
    kernel = np.ones((9, 9), np.uint8)

    def nothing(x):
        pass
    cv.createTrackbar('threshold', 'image', 0, 255, nothing)
    
    left    = [36, 37, 38, 39, 40, 41]
    right   = [42, 43, 44, 45, 46, 47]

    start_time  = time.time()
    avg_sec     = 5
    counter     = 0
    fps         = 0
    while(True):
        
        ret, img    = cap.read()
        
        
        # image resize
        width       = 320
        height      = 240
        # width       = 640
        # height      = 480
        
        img         = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)
        orig_img    = img.copy() 
        # low light enhancement
        if(low_light_model == "LIME"):
            img = conventional.enhance_image_exposure(img, 0.6, 0.15, False,
                                                sigma=3, bc=1, bs=1, be=1, eps=1e-3)
        elif(low_light_model == "DUAL"): 
            img = conventional.enhance_image_exposure(img, 0.6, 0.15, True,
                                                sigma=3, bc=1, bs=1, be=1, eps=1e-3)
        elif(low_light_model == "DCE"):
            img = torch.from_numpy(img/255.0).float()
            img = img.permute(2,0,1)
            img = img.unsqueeze(0).to('cpu')
            
            _,img,_ = DCE_net(img)
            img     = img.squeeze(0).permute(1,2,0).detach().numpy()

            
        img_gray    = cvt_GRAY(img)
        #Choose the detector
        if(face_detector == "DNN"):
            img_dnn             = img.copy()
            img_dnn_blob        = cv.dnn.blobFromImage(cv.resize(img_dnn, (300, 300)),
                                        1.0, (300, 300), (104.0, 117.0, 123.0))
            rects               = face_dnn(face_dnn_detector, img_dnn_blob, img_dnn.shape)
        elif(face_detector == "HOG"):
            rects               = face_hog(face_hog_detector, hog_upsample, img_gray)
        elif(face_detector == "MTCNN"):
            img_mtcnn           = img.copy()
            rects               = face_mtcnn(face_mtcnn_detector, img_mtcnn)
        elif(face_detector == "HAAR"): 
            img_haar            = img.copy() 
            rects               = face_haar_cascade(face_haar_detector, img_haar)
        else:
            print("Invalid Face Detector")
            return  
            
        # img       = draw_FACE(img, rects)
        
        for (x1,x2,y1,y2) in rects:
            rect  = dlib.rectangle(x1, y1, x2, y2)
            shape = eye_68_detector(img_gray, rect)
            shape = shape_to_np(shape)
            
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask = eye_on_mask(shape, mask, left)
            mask = eye_on_mask(shape, mask, right)
            
            
            
            mask = cv.dilate(mask, kernel, 5)
            eyes = cv.bitwise_and(img, img, mask=mask)
            mask = (eyes == [0, 0, 0]).all(axis=2)
            eyes[mask] = [255, 255, 255]
            mid = (shape[42][0] + shape[39][0]) // 2
            eyes_gray = cvt_GRAY(eyes)
            threshold = cv.getTrackbarPos('threshold', 'image')
            # threshold = 80
            _, thresh = cv.threshold(eyes_gray, threshold, 255, cv.THRESH_BINARY)
            thresh = cv.erode(thresh, None, iterations=2) #1
            thresh = cv.dilate(thresh, None, iterations=4) #2
            thresh = cv.medianBlur(thresh, 3) #3
            thresh = cv.bitwise_not(thresh)
            img    = contouring(img, mid, thresh[:, 0:mid])
            img    = contouring(img, mid, thresh[:, mid:], True)

        
        counter += 1
        if((time.time()-start_time)> avg_sec):   
            fps = counter/ (time.time()-start_time)
            counter = 0
            start_time = time.time()
            print("FPS: ", fps)
        
        img = cv.putText(img= img, text=f"FPS={fps:.2f}", org=(150, 80), 
                        fontFace=cv.FONT_HERSHEY_DUPLEX, 
                        fontScale=1.0, color=(125, 246, 55), thickness=1)
    
        # show the image with the face detections + facial landmarks
        cv.imshow('eyes', img)
        cv.imshow('orig', orig_img)
        cv.imshow("image", thresh)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv.destroyAllWindows()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
    parser.add_argument('--type', help='["ALL", "DETECT", "TRACK"]', type=str, default="ALL")
    parser.add_argument('--face', help='["DNN", "HOG", "MTCNN", "HAAR"]', type=str, default="DNN")
    parser.add_argument('--light', help='["DCE", "GAN", "LIME", "DUAL"]', type=str, default="DCE")
    args = parser.parse_args()

    if(args.type == "ALL"):
        print("RUN ALL THE METHOD.")
        ALL_METHODS()
    elif(args.type == "DETECT"):
        print("USING WEBCAM TO DO DETECTION")
        DETECTION(args.camera, args.light)
    else:
        print("USING WEBCAM TO DO REAL TIME TRACKING")
        REAL_TIME_TRACKING(args.camera, args.face, args.light)