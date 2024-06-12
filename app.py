import os
import cv2
import numpy as np
from skimage.feature import hog
import joblib
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk, ImageGrab
import pyperclip

# Function to preprocess the image and extract HOG features
def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (150, 150))
    feature, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, channel_axis=-1)
    return feature

# Function to predict the class of an input image
def predict_image_class(image, model):
    feature = preprocess_image(image)
    feature = feature.reshape(1, -1)  # Reshape to match the expected input shape of the model
    prediction = model.predict(feature)

    # prediction_proba = model.predict_proba(feature)

    # threshold = 0.4
    # if prediction_proba.max() < threshold:
    #     return "Unknown"
    # else:
    return prediction[0]  # Directly return the class label

# Load the trained SVM model
model_filename = 'svm_model.pkl'
best_model = joblib.load(model_filename)

# Define the function to upload and predict the image
def upload_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            img = Image.open(file_path)
            predicted_class = predict_image_class(img, best_model)
            
            # Display the uploaded image in its original size
            img.thumbnail((400, 400))  # Limit size for display purposes
            img_tk = ImageTk.PhotoImage(img)
            image_label.configure(image=img_tk)
            image_label.image = img_tk
            
            # Display the predicted class
            result_label.config(text=f'Predicted Class: {predicted_class}')
        except Exception as e:
            messagebox.showerror("Error", str(e))

# Define the function to paste and predict the image from clipboard
def paste_and_predict():
    try:
        # Grab the image from clipboard
        img = ImageGrab.grabclipboard()
        if isinstance(img, Image.Image):
            predicted_class = predict_image_class(img, best_model)
            
            # Display the pasted image in its original size
            img.thumbnail((400, 400))  # Limit size for display purposes
            img_tk = ImageTk.PhotoImage(img)
            image_label.configure(image=img_tk)
            image_label.image = img_tk
            
            # Display the predicted class
            result_label.config(text=f'Predicted Class: {predicted_class}')
        else:
            messagebox.showerror("Error", "No image found in clipboard.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main window
root = tk.Tk()
root.title("Sport Classification with SVM")
root.geometry("600x700")
root.configure(bg="#fafafa")

# Create a header frame
header_frame = tk.Frame(root, bg="#ffffff", height=50, padx=10, pady=10)
header_frame.pack(fill="x")
header_label = tk.Label(header_frame, text="Sport Classification", font=("Helvetica", 18, "bold"), bg="#ffffff")
header_label.pack()

# Create an image display frame
image_frame = tk.Frame(root, bg="#fafafa", padx=10, pady=10)
image_frame.pack(fill="both", expand=True)

image_label = tk.Label(image_frame, bg="#fafafa")
image_label.pack(expand=True)

# Create a result label
result_label = tk.Label(root, text="Predicted Class: ", font=("Helvetica", 14), bg="#fafafa")
result_label.pack(pady=10)

# Create a footer frame for upload and paste buttons
footer_frame = tk.Frame(root, bg="#ffffff", height=50, padx=10, pady=10)
footer_frame.pack(fill="x", side="bottom")
btn_upload = tk.Button(footer_frame, text="Upload Image and Predict", command=upload_and_predict, bg="#3897f0", fg="#ffffff", font=("Helvetica", 12, "bold"))
btn_upload.pack(side="left", padx=20, pady=10)
btn_paste = tk.Button(footer_frame, text="Paste Image and Predict", command=paste_and_predict, bg="#3897f0", fg="#ffffff", font=("Helvetica", 12, "bold"))
btn_paste.pack(side="right", padx=20, pady=10)

# Start the GUI loop
root.mainloop()
