from __future__ import print_function
from mtcnn.mtcnn import MTCNN
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import dlib
import os

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
        
        radius      = 40
        thickness   = 10
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

# def detectAndDisplay(frame):
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     frame_gray = cv.equalizeHist(frame_gray)
#     #-- Detect faces
#     faces = face_cascade.detectMultiScale(frame_gray)
#     for (x,y,w,h) in faces:
#         center = (x + w//2, y + h//2)
#         frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
#         faceROI = frame_gray[y:y+h,x:x+w]
#         #-- In each face, detect eyes
#         eyes = eyes_cascade.detectMultiScale(faceROI)
#         for (x2,y2,w2,h2) in eyes:
#             eye_center = (x + x2 + w2//2, y + y2 + h2//2)
#             radius = int(round((w2 + h2)*0.25))
#             frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
    
#     cv.imshow('Capture - Face detection', frame)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

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

###############################
#   DL Low Light Enhancement
###############################

enhancement_method = ""

# Englighten GAN
if enhancement_method == "EnlightenGAN":
    os.system("python ./low_light/EnlightenGAN/predict.py \
    --dataroot ./my_images \
    --checkpoints_dir ./low_light/EnlightenGANs/EnlightenGAN/checkpoints \
    --no_dropout \
    --self_attention \
    --times_residual \
    --which_epoch " + str(200))

 
for image in images:
    if(image[-4:] != ".JPG" and image[-5:] != ".jpeg"): 
        continue
    if(image != "low8_real_A.jpeg"):
        continue
    
    print("Processing:", image)
    
    img = cv.imread(os.path.join('my_images', image))
    # detectAndDisplay(img)
    ###############################
    #   DL Low Light Enhancement
    ###############################
    # Zero DCE
    def ZERO_DCE(dark_img_in, light_img_out):
        pass
    # EnlightenGAN
    def ENLIGHTEN_GAN(dark_img_in, light_img_out = None):
        # complete above
        # do directly on the entire directory instead of single images
        enlighten_images = os.listdir("./enlighten_images") 
        image_name = dark_img_in
        for enlighten_img in enlighten_images:
            if enlighten_img == image_name:
                light_img_out = cv.imread(os.path.join('./enlighten_images', enlighten_img))
                break
        return light_img_out

    
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


################################
#   Using Camera
################################

# camera_device = args.camera
#-- 2. Read the video stream
# cap = cv.VideoCapture(camera_device)
# if not cap.isOpened:
#     print('--(!)Error opening video capture')
#     exit(0)

# while True:
#     ret, frame = cap.read()
#     if frame is None:
#         print('--(!) No captured frame -- Break!')
#         break
#     detectAndDisplay(frame)
#     if cv.waitKey(10) == 27:
#         break