import tkinter as tk 
from tkinter import filedialog
from tkinter import *
import tensorflow as tf
# from model_eval_program import model
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import ImageTk, Image
import numpy as np 
import cv2

model = load_model('model.keras')

def mouth_detection():
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

top = tk.Tk()
top.geometry('256x256')
top.title('Mouth Detector')

label1 = Label(top, background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def detect(file_path):
    global Label_packed

    try:
        image = cv2.imread(file_path)
        resize = tf.image.resize(image, (256, 256))
        plt.imshow(resize.numpy().astype(int))
        yhat = model.predict(np.expand_dims(resize/255, 0))
        print(yhat)
        label1.configure(foreground='#011638', text= 'Open' if yhat > 0.5 else 'Close')

        if yhat > 0.5:
            print(f'Mouth is open')
        else:
            print(f'Mouth is closed')

    except:
        label1.configure(foreground='#011638', text='Unable to detect')

def show_detect_button(file_path):
    detect_b = Button(top, text='Detect Mouth', command=lambda : detect(file_path),pady=5, padx=10)
    detect_b.configure(background='#011638', foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.3), (top.winfo_height()/2.3)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label1.configure(text='')
        show_detect_button(file_path)
    except:
        pass

upload = Button(top,text='Upload Image', command=upload_image,padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top, text='Mouth Detection', pady=20, font=('arial', 10, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.protocol("WM_DELETE_WINDOW", top.quit)
top.mainloop()
