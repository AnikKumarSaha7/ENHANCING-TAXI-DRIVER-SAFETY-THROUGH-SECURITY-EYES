import cv2
import numpy as np
import urllib.request
import time
from geopy.geocoders import Nominatim
from twilio.rest import Client
from twilio.base.exceptions import TwilioException
import pygame
import os
account_sid = 'ACf83ecc9a7692fec4582573f4b30c37db'
auth_token = '2e4dfc730c5195f3d87304bb894f15c2'
twilio_whatsapp_number = 'whatsapp:+xxxxxxxxxxx'  
destination_whatsapp_number = 'whatsapp:+###########'  
esp32_camera_url = 'http://<ESP32_IP_ADDRESS>/stream'
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
classesfile = 'coco2.names'
classNames = []
with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
modelConfig = 'yolov3_training(5).cfg'
modelWeights = 'yolov3_training_last(8).weights'
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
cap = cv2.VideoCapture(esp32_camera_url)

def play_alarm_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("alarm.mp3")
    pygame.mixer.music.play()

def get_gps_coordinates():
    locator = Nominatim(user_agent="myGeocoder")
    location = locator.geocode("Name_of_City")
    if location:
        return location.latitude, location.longitude
    else:
        return None, None
def capture_image(image_path):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam")
        return None
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame")
        return None
    frame_resized = cv2.resize(frame, (192, 192))
    cv2.imwrite(image_path, frame_resized)
    cap.release()
    cv2.destroyAllWindows()
    return image_path

def upload_image(image_path):
    return "http://server.com/uploads/" + os.path.basename(image_path)

def send_whatsapp_with_image(image_path, latitude, longitude):
    client = Client(account_sid, auth_token)
    message_text = f"Detected Object. Latitude: {latitude}, Longitude: {longitude}"    
    try:
        media_url = upload_image(image_path)
        message = client.messages.create(
            body=message_text,
            from_=twilio_whatsapp_number,
            to=destination_whatsapp_number,
            media_url=[media_url]  
        )
        print(f"WhatsApp message sent to {destination_whatsapp_number} successfully with GPS coordinates!")
    except TwilioException as e:
        print(f"Twilio Exception: {e}")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from the camera.")
        break

    im = frame
    blob = cv2.dnn.blobFromImage(im, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layernames = net.getLayerNames()
    outputNames = [layernames[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    hT, wT, cT = im.shape
    bbox = []
    classIds = []
    confs = []
    found_knife = False
    found_gun = False
    found_chloroform = False
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        
        if classNames[classIds[i]] == 'knife':
            found_knife = True
        elif classNames[classIds[i]] == 'gun':
            found_gun = True
        elif classNames[classIds[i]] == 'chloroform':
            found_chloroform = True
        if found_knife or found_gun or found_chloroform:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(im, classNames[classIds[i]].upper(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 
    
    if found_knife or found_gun or found_chloroform:
        play_alarm_sound()
        latitude, longitude = get_gps_coordinates()
        if latitude and longitude:
            image_path = "captured_image.jpg"
            image_path = capture_image(image_path)
            if image_path:
                send_whatsapp_with_image(image_path, latitude, longitude)
            else:
                print("Failed to capture image.")
        else:
            print("Failed to obtain GPS coordinates.")
    
    cv2.imshow('Detection', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
