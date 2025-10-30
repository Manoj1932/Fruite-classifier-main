# fruit_classifier_gui_voice.py

import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import pyttsx3
import os

# ---------------- Load Model ---------------- #
try:
    model = load_model("final_fruit_classifier.h5")  # or your model name
except Exception as e:
    messagebox.showerror("Error", f"Failed to load model:\n{e}")
    exit()

# Class labels (adjust based on your dataset)
class_labels = ['apple', 'banana', 'orange']

# ---------------- Speech Setup ---------------- #
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        voices = engine.getProperty('voices')
        if len(voices) > 1:
            engine.setProperty('voice', voices[1].id)  # female (optional)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print("Speech Error:", e)

# ---------------- Prediction Function ---------------- #
def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(100, 100))  # match training size
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        return class_labels[class_idx], confidence
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Error during prediction:\n{e}")
        return None, None

# ---------------- GUI Functions ---------------- #
def upload_image():
    global img_path
    img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if img_path:
        try:
            img = Image.open(img_path)
            img = img.resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            panel.config(image=img_tk)
            panel.image = img_tk
            result_label.config(text="🔍 Predicting...", fg="blue")
            root.after(600, lambda: predict_and_display(img_path))
        except Exception as e:
            messagebox.showerror("Error", f"Unable to open image:\n{e}")

def predict_and_display(path):
    fruit_name, confidence = predict_image(path)
    if fruit_name:
        result_text = f"🍉 {fruit_name.upper()}\nConfidence: {confidence:.2f}%"
        result_label.config(text=result_text, fg="green", font=("Arial", 14, "bold"))
        # Always speak every prediction
        speak_prediction(fruit_name, confidence)

def speak_prediction(fruit, confidence):
    text = f"I am {confidence:.0f} percent sure this is a {fruit}."
    speak(text)

# ---------------- GUI Setup ---------------- #
root = tk.Tk()
root.title("🍎 Fruit Classifier with Voice")
root.geometry("420x520")
root.resizable(False, False)
root.configure(bg="#f7f7f7")

title_label = Label(root, text="Fruit Classifier using CNN",
                    font=("Helvetica", 18, "bold"), bg="#f7f7f7", fg="#2C3E50")
title_label.pack(pady=10)

panel = Label(root, bg="#f7f7f7")
panel.pack(pady=10)

upload_btn = Button(root, text="Upload Fruit Image", command=upload_image,
                    bg="#27AE60", fg="white", font=("Arial", 12, "bold"),
                    padx=10, pady=5)
upload_btn.pack(pady=10)

result_label = Label(root, text="", bg="#f7f7f7", fg="blue", font=("Arial", 12, "bold"))
result_label.pack(pady=15)

footer_label = Label(root, text="Developed by Jaddu Mohan Kishore",
                     font=("Arial", 9), bg="#f7f7f7", fg="#7F8C8D")
footer_label.pack(side="bottom", pady=5)

root.mainloop()
