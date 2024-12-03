import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Store the classification results globally
classification_results = []

# Function to classify the image
def classify_image(image_path):
    global classification_results
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to 224x224 for the model
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)

    # Decode predictions
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Get the top 3 predictions
    classification_results = decoded_predictions
    return decoded_predictions

# Function to handle the upload button
def upload_image():
    global img_label, predictions_label
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    
    if file_path:
        try:
            # Load and check image dimensions
            img = Image.open(file_path)
            if img.size[0] > 3000 or img.size[1] > 3000:
                messagebox.showerror("Error", "Please upload an image smaller than 3000x3000 pixels.")
                return

            # Display uploaded image
            img = img.resize((300, 200))  # Resize for display purposes
            img_tk = ImageTk.PhotoImage(img)
            img_label.config(image=img_tk)
            img_label.image = img_tk  # Keep a reference

            # Classify the image
            predictions = classify_image(file_path)

            # Format the result to include type, breed, and percentage
            result = ""
            for i, (imagenet_id, label, score) in enumerate(predictions):
                percentage = score * 100  # Convert to percentage
                result += f"{i+1}. {label} (Confidence: {percentage:.2f}%)\n"

            predictions_label.config(text=result.strip())

            # Clear previous graph when a new image is uploaded
            for widget in graph_container.winfo_children():
                widget.destroy()

        except Exception as e:
            messagebox.showerror("Error", str(e))

# Function to remove the uploaded image
def remove_image():
    img_label.config(image='')
    img_label.image = None
    predictions_label.config(text="")
    # Clear the graph when the image is removed
    for widget in graph_container.winfo_children():
        widget.destroy()

# Function to plot the bar graph
def plot_graph():
    if not classification_results:
        messagebox.showerror("Error", "No classification results to plot!")
        return
    
    # Clear the previous graph if it exists
    for widget in graph_container.winfo_children():
        widget.destroy()
    
    labels = [label for (_, label, _) in classification_results]
    scores = [score * 100 for (_, _, score) in classification_results]

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(5, 3))  # Adjusted graph width and height

    # Plot the bar chart
    ax.bar(labels, scores, color='skyblue', width=0.5)  # Adjust bar width

    # Set labels and title
    ax.set_ylabel('Confidence (%)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Animal Types', fontsize=10, fontweight='bold')
    ax.set_title('Top 3 Classification Results', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)  # Ensure y-axis goes from 0 to 100%

    # Ensure labels on the x-axis are horizontal
    ax.tick_params(axis='x', rotation=0, labelsize=9)

    # Display the plot in the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=graph_container)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Create the GUI window
root = tk.Tk()
root.title("Image Classifier")
root.geometry("1366x768")
root.state("zoomed")

# Background color for the entire window
root.config(bg="white")

# Styling variables
bg_color = "#ecf2f9"   # Sidebar background color
btn_color = "#132743"  # Button background color
fg_color = "white"     # Button text color
font = ("Arial", 12)

# Sidebar Frame
sidebar = tk.Frame(root, width=250, bg=bg_color, height=400, relief="sunken")
sidebar.grid(row=0, column=0, sticky="ns", padx=10, pady=(20, 0))

# Button creation helper function
def create_button(parent, text, command=None):
    button = tk.Button(parent, text=text,
                       bg=btn_color, fg=fg_color, font=font, relief="flat", activebackground=btn_color, activeforeground=fg_color,
                       highlightthickness=0, anchor="center", cursor="hand2", width=20, height=2, command=command)
    button.grid(pady=10, padx=10, sticky="ew")

# Create Sidebar buttons
buttons = ["Upload Image", "Remove Image", "Plot Graph", "Quit"]
for button_text in buttons:
    if button_text == "Upload Image":
        create_button(sidebar, button_text, upload_image)
    elif button_text == "Remove Image":
        create_button(sidebar, button_text, remove_image)
    elif button_text == "Plot Graph":
        create_button(sidebar, button_text, plot_graph)
    elif button_text == "Quit":
        create_button(sidebar, button_text, root.quit)

# Content Frame
content_frame = tk.Frame(root, bg="#ecf2f9")
content_frame.grid(row=0, column=1, sticky="nsew", padx=30, pady=20)

# Ensure the row and column can expand
root.grid_rowconfigure(0, weight=1)  # Allow row 0 to expand
root.grid_columnconfigure(1, weight=1)  # Allow column 1 to expand

# Heading label for Image Classification
label = tk.Label(content_frame, text="Classify Image", font=("Arial", 24), bg="#ecf2f9")
label.pack(pady=20)

# Container for image, graph, and results
container = tk.Frame(content_frame, bg="#f7f7f7")
container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Frame for the image and results
img_results_frame = tk.Frame(container, bg="#f7f7f7")
img_results_frame.pack(side=tk.LEFT, padx=20, pady=20, anchor="n")  # Anchor to top left

img_label = tk.Label(img_results_frame, bg="#f7f7f7")
img_label.pack(pady=(0, 20))  # Space between image and results

predictions_label = tk.Label(img_results_frame, text="", bg="#f7f7f7", fg="black", font=('Arial', 12), anchor="w", justify="left")
predictions_label.pack(pady=10)

# Frame for the graph
graph_container = tk.Frame(container, bg="#f7f7f7", height=300, width=900)  # Frame size unchanged
graph_container.pack_propagate(False)  # Prevent resizing based on content
graph_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Start the GUI main loop
root.mainloop()
