from tkinter import *
from ttkbootstrap.constants import *
import ttkbootstrap as tb
import cv2 as cv
from tkinter import filedialog
import time
from PIL import Image
import numpy as np
from functions import *

root = tb.Window(themename="superhero")
root.title("Face Regonition Ai")
root.geometry('550x550')
haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


def insert():
    train(name.get(), train_status)

def check():
    test(name_display, status_label)

body = tb.Notebook(root)
body.pack(pady=10, fill=BOTH)

train_tab = tb.Frame(body)
test_tab = tb.Frame(body)

description_label = tb.Label(train_tab, text="Here you can train the ai using either images or webcam")
description_label.pack(pady=10)

ins_train_btn = tb.Button(train_tab, text="Using images",command= lambda:insert_image('train'))
ins_train_btn.pack(pady=20)

open_camera_train = tb.Button(train_tab, text="Using camera",command=lambda:open_webcam('train'))
open_camera_train.pack(pady=20)


name_frame = tb.Frame(train_tab)
name_frame.pack(pady=20)

name_label = tb.Label(name_frame, text="Persons's name:")
name_label.grid(column=0, row=0, sticky='w')

name = tb.Entry(name_frame, text="Enter the person's name")
name.grid(column=1, row=0,padx=20)

train_btn = tb.Button(train_tab,text="Train", command=insert)
train_btn.pack(pady=20, padx=20)

train_status = tb.Label(train_tab)
train_status.pack(pady=20)


description_label2 = tb.Label(test_tab, text="Here you can test the face recognition AI using either an image or your webcam")
description_label2.pack(pady=10)

ins_test_btn = tb.Button(test_tab, text="Insert image",command= lambda:insert_image('test'))
ins_test_btn.pack(pady=20)

open_camera_test = tb.Button(test_tab, text="Open camera",command=lambda:open_webcam('test'))
open_camera_test.pack(pady=20)

check_btn = tb.Button(test_tab, text="Check", command=check)
check_btn.pack(pady=20)

name_display = tb.Label(test_tab)
name_display.pack(pady=20)

body.add(train_tab, text="Train", sticky='snwe')
body.add(test_tab, text="Test AI", )

status_label = tb.Label(test_tab)
status_label.pack(pady=10)
reset_btn = tb.Button(root, text="Reset AI", bootstyle="danger", command=reset)
reset_btn.pack(pady=5)

root.mainloop()
