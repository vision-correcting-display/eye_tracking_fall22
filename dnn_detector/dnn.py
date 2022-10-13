import cv2 as cv
import math
import numpy as np
import argparse
from totxt import make_text_file
import time

#constants used in gaze angle calculations
#~3.80 pixels to 1 mm
pixel_conversion = 3.8
ave_eye_size = 30
#ideal distance for computer screen users equiv
#Jacob used 250 though
z_distance = 400

modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
# img_name = "dnn_test.jpg"

net = cv.dnn.readNetFromCaffe(configFile, modelFile)
# img = cv2.imread('test.jpg')
# h, w = img.shape[:2]
# blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 117.0, 123.0))
# net.setInput(blob)
# faces = net.forward()
# #to draw faces on image
# for i in range(faces.shape[2]):
#         confidence = faces[0, 0, i, 2]
#         if confidence > 0.5:
#             box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (x, y, x1, y1) = box.astype("int")
#             cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)

# cv2.imwrite(img_name, img)

# img = cv2.imread('test.jpg')
# h, w = img.shape[:2]


# Read the video stream
def detectAndDisplay(frame):
    height, width, _ = frame.shape
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # detect face 
    start = time.time()
    blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    end = time.time()

    run_time = end - start

    #instantiations
    left_eye = None
    right_eye = None
    pupilliary_distance = None
    left_pitch = None
    left_yaw = None
    right_pitch = None
    right_yaw = None

    has_face = 0
    
    #to draw faces on image
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            box = faces[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            frame = cv.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)

            faceROI = frame[y:y1, x:x1]
            
            has_face = 1
            #-- In each face, detect eyes
            eyes = eyes_cascade.detectMultiScale(faceROI)
            for (ex,ey,ew,eh) in eyes:
                #filtering potential false eyes (mouth and nose)
                if ey+eh < int(2 * (y1 - y) / 3):
                    #create eye region
                    eyeROI = frame[ey+y:ey+eh+y, ex+x:ex+ew+x]
                    eyeROI_center = (x + ex + ew//2, y + ey + eh//2)
                    radius = int(round((ew + eh)*0.25))
                    #eye drawing
                    #frame = cv.circle(frame, eyeROI_center, radius, (255, 0, 0 ), 4)
                    x_pos, y_pos = get_iris_region(eyeROI, x, y, ex, ey, frame)
                    if x_pos < x + ((x1 - x)//2) and x_pos != -1:
                        left_eye = (x_pos, y_pos)
                    elif x_pos != -1:
                        right_eye = (x_pos, y_pos)
                    
                    if left_eye != None and right_eye != None:
                        pupilliary_distance = get_pupil_distance(left_eye, right_eye, ew)
                        cv.line(frame, left_eye, right_eye, (0, 0, 255), 2)
                        left_pitch, left_yaw = get_pitch_yaw_angle(left_eye, height, width)
                        right_pitch, right_yaw = get_pitch_yaw_angle(right_eye, height, width)
                
                        #switch L/R to the users perspective (which is what Jacob has coded for from my understanding)
                        print("right_eye: (", math.degrees(left_pitch), math.degrees(left_yaw), ")")
                        print("left_eye: (", math.degrees(right_pitch), math.degrees(right_yaw), ")")
                        print(pupilliary_distance)

    cv.imshow('Capture - Face detection', frame)

    #will need to return pupilliary distance and eye locations in the future for binocular vision aspect
    #but for now, with just the 3D repo, this isn't needed.
    return left_pitch, left_yaw, right_pitch, right_yaw, run_time, has_face

def get_iris_region(eyeROI, face_x, face_y, eye_x, eye_y, frame):
	grayEye = cv.cvtColor(eyeROI, cv.COLOR_BGR2GRAY)
	grayEye = cv.GaussianBlur(grayEye, (7, 7), 0)
	_, threshold = cv.threshold(grayEye, 30, 255, cv.THRESH_BINARY_INV)
	contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=lambda x:cv.contourArea(x), reverse = True)
	
	for cnt in contours:
		(x, y, w, h) = cv.boundingRect(cnt)
		x_pos = x+int(w/2)+face_x+eye_x
		y_pos = y+int(h/2)+face_y+eye_y
		cv.circle(frame, (x_pos, y_pos), 5, (255, 255, 255), 3)
		return x_pos, y_pos
	
	return -1, -1

#let me know if this is logically flawed, but I think it should be right
#because it should just be to get a ratio of pixels to real life
#but I did this at like 3am one day, so please check
def get_pupil_distance(left, right, eye_width):
	pixel_dist = right[0] - left[0]
	screen_dist = pixel_dist / pixel_conversion
	ratio = eye_width / pixel_conversion / ave_eye_size
	return screen_dist / ratio

def get_pitch_yaw_angle(eye, screen_height, screen_width):
	pitch_angle = math.atanh(abs(eye[1] - int(screen_height/2)) / pixel_conversion / z_distance)
	yaw_angle = math.atanh(abs(eye[0] - int(screen_width/2)) / pixel_conversion / z_distance)
	if (eye[1] - int(screen_height/2)) < 0:
		#looking down
		pitch_angle = -pitch_angle
	if (eye[0] - int(screen_width/2)) > 0:
		#looking right by jacob's def
		yaw_angle = -yaw_angle
	return pitch_angle, yaw_angle

# For Sainan and Patricia, these are the configuration metrics that are needed by the code
# So I believe some more things that you are working on could be passed into this
def write_config(pitch_angle, yaw_angle, configPath):
	imagePath = "../test_images/checkerboard.png"	#arbitrary for now - using this image in simulation

	eyeFocalDist = 250
	eyeFocalLength = 20
	pupilDiameter = 6
	imageDist = 20

	displayPreset = "custom"
	displayPixelWidth = 750
	displayPixelHeight = 1334
	displayPPI = 326

	hardwareType = "pinhole"
	hardwareDensity = 5
	hardwareDepth = 6
	hardwarePinholeDiameter = 0.1

	displayOriginX = 0
	displayOriginY = 0
	displayOriginZ = z_distance
	displayDist = z_distance

	displayRoll = 0				#this angle is a bit odder in real-life scenarios -> set to 0
	displayYaw = yaw_angle
	displayPitch = pitch_angle

	margin = 10
	retinaNumPixelsWidth = 1300
	retinaNumPixelsHeight = 1300
	retinaWidth = 10.00
	retinaHeight = 10.00

	simulatorSampleRate = 256
	maxTolerance = 0.01
	nearbyPinholes = 9
	scheme = 2
	samplesPerPixel = 4
	samplesPerPinhole = 4

	testSuite = 'false'
	testName = "checkerboard_1"
	
	make_text_file(configPath, imagePath, eyeFocalDist, eyeFocalLength, pupilDiameter, imageDist, \
					displayPreset, displayPixelWidth, displayPixelHeight, displayPPI, hardwareType, \
					hardwareDensity, hardwareDepth, hardwarePinholeDiameter, displayOriginX, displayOriginY,\
					displayOriginZ, displayDist, displayRoll, displayYaw, displayPitch, margin, \
					retinaNumPixelsWidth, retinaNumPixelsHeight, retinaWidth, retinaHeight, simulatorSampleRate,\
					maxTolerance, nearbyPinholes, scheme, samplesPerPixel, samplesPerPinhole, testSuite, testName)

#Defining Haar Cascade classifier (eyes and face)
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
# parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)


args = parser.parse_args()
# face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
# face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
#-- 1. Load the cascades
# if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
#     print('--(!)Error loading face cascade')
#     exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
camera_device = args.camera

cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
    
frame_number = 0
face_number = 0
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    start_time = time.time()

    LP, LY, RP, RY, RT, face = detectAndDisplay(frame)

    frame_number += 1
    if face:
        face_number += 1


    end_time = time.time()
    total_runtime = end_time - start_time

    k = cv.waitKey(10)
    if k == 32: #space bar
        print("screen taken!")
        img_name = "captured_tracked_frame/opencv_frame.png"
        cv.imwrite(img_name, frame)
        print(f" {face_number} / {frame_number}")
        #switching left and right eye definitions because original calcs were from screen perspective
        write_config(round(RP, 2), round(RY, 2), "eye_tracking_config_files/left_eye_config.txt")
        write_config(round(LP, 2), round(LY, 2), "eye_tracking_config_files/right_eye_config.txt")
        print("screen saved!")
        print(f"Runtime of the face detection is {RT}")
        print(f"Runtime of the pupil detection is {total_runtime}")
        break
    elif k == 27: #escape key
        print("Escape hit, closing...")
        break

cap.release()
cv.destroyAllWindows()

