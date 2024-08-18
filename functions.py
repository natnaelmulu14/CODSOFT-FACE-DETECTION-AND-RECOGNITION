from tkinter import *
from ttkbootstrap.constants import *
import ttkbootstrap as tb
import cv2 as cv
import numpy as np
from tkinter import filedialog
import time
from PIL import Image
import json
import os


test_images = []
test_image = np.array([])
train_images = []
haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


def insert_train():
    global train_images
    images_path = filedialog.askopenfilenames(title="Select the image for the training",
        filetypes=(
            ("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.tiff"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg;*.jpeg"),
            ("BMP files", "*.bmp"),
            ("TIFF files", "*.tiff"),
            ("All files", "*.*")
        )
    )
    if images_path:
        for image in images_path:
            image = Image.open(image)
            image_array = np.array(image)
            gray = cv.cvtColor(image_array, cv.COLOR_RGB2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=4)
            for (x,y,w,h) in faces_rect:
                face = gray[y:y+h, x:x+w]
                train_images.append(face)
def insert_test():
    global test_image
    images_path = filedialog.askopenfilename(title="Select the image for the training",
        filetypes=(
            ("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.tiff"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg;*.jpeg"),
            ("BMP files", "*.bmp"),
            ("TIFF files", "*.tiff"),
            ("All files", "*.*")
        )
    )
    if images_path:

        image = Image.open(images_path)
        image_array = np.array(image)
        gray = cv.cvtColor(image_array, cv.COLOR_RGB2GRAY)
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=4)
        for (x,y,w,h) in faces_rect:
            face = gray[y:y+h, x:x+w]
            test_image = face
    print(test_image)
def insert_image(mode):
    if mode == 'train':
        insert_train()

    elif mode == 'test':
        insert_test()

def open_cam_train():
    global train_images
    vid= cv.VideoCapture(0)

    while True:
        isTrue, frame = vid.read()
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=4)
        for (x,y,w,h) in faces_rect:
            face = gray[y:y+h, x:x+w]
            train_images.append(face)
            cv.imshow('f', face)
        if len(train_images) == 300:
            break
        if cv.waitKey(20) and 0xFF == ord('d'):
            break
        time.sleep(0.01)

    vid.release()
    cv.destroyAllWindows()

def open_cam_test():
    global test_images
    test_images = []
    vid= cv.VideoCapture(0)
    # start_time = time.time()
    # duration = 5
    while True:
        isTrue, frame = vid.read()
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=4)
        for (x,y,w,h) in faces_rect:
            face = gray[y:y+h, x:x+w]
            test_images.append(face)
            cv.imshow('f', face)
        if len(test_images) == 10:
            break
        if cv.waitKey(20) and 0xFF == ord('d'):
            break
        time.sleep(0.1)
    print(test_images)
    vid.release()
    cv.destroyAllWindows()

def open_webcam(mode):
    if mode == 'train':
        open_cam_train()
    elif mode == 'test':
        open_cam_test()

def train(name, train_status):
    global train_images
    labels = []

    if name:

        with open ('labels.json', 'r+') as data:
            label = len(data.readlines())
            label = int(label)
            for i in range(len(train_images)):
                labels.append(label)
            row = {label:name}
            json.dump(row, data)
            data.write('\n')
            
            

        train_images = np.array(train_images, dtype='object')
        labels = np.array(labels)

        face_recognizer = cv.face.LBPHFaceRecognizer_create()
        model_path = 'face_recognizer.yml'

        if os.path.exists(model_path):
            face_recognizer.read('face_recognizer.yml')
            face_recognizer.update(train_images, labels)
            face_recognizer.save(model_path)
            train_status.config(text="Done training")
        else:
            face_recognizer.train(train_images, labels)
            face_recognizer.save('face_recognizer.yml')
            train_status.config(text="Done training")

def display(best_match, name_display):
    name = ''
    with open ('labels.json', 'r') as data:
        for line in data:
            line = json.loads(line.strip())
            for id in line:
                if id == str(best_match):
                    name = line[id]
                    name_display.config(text=name)



def test(name_display,status_label):
    global test_images, test_image
    matches = []
    best_match = 0
    count = 0
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    model_path = 'face_recognizer.yml'
    if os.path.exists(model_path):

        face_recognizer.read('face_recognizer.yml')
        status_label.config(text='')
        if test_images:
            for face in test_images:
                id, confidence= face_recognizer.predict(face)
                matches.append(id)
            if matches:    
                print(matches)
                for match in matches:
                    qua = matches.count(match)
                    if qua > count:
                        count = qua
                        best_match = match
                display(best_match, name_display)

        if test_image.size != 0:
            id, confidence= face_recognizer.predict(test_image)
            print(id)
            display(id, name_display)
    else:
        status_label.config(text='There is no any trained model. Please train one first before testing', bootstyle='danger')


def reset():
    model_path = 'face_recognizer.yml'
    if os.path.exists(model_path):
        os.remove(model_path)

    with open('labels.json', 'w') as labels:
        pass  



