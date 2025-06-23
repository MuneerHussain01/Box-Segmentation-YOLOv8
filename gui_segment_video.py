import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("best.pt")

# ---------------------------------
# Global variables for video
# ---------------------------------
cap = None
out = None
fps = 30  # default fallback

# ---------------------------------
# Image segmentation
# ---------------------------------
def browse_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.jfif")]
    )
    entry_image.delete(0, tk.END)
    entry_image.insert(0, file_path)

def run_image_segmentation():
    img_path = entry_image.get()
    if not os.path.isfile(img_path):
        messagebox.showerror("Error", "Please select a valid image file.")
        return

    img = cv2.imread(img_path)
    if img is None:
        messagebox.showerror("Error", "Failed to read the image.")
        return

    results = model.predict(source=img, task="segment", conf=0.4, verbose=False)
    annotated = results[0].plot()

    output_path = "output_segmented_image.jpg"
    cv2.imwrite(output_path, annotated)

    # Show in GUI
    bgr_frame = cv2.resize(annotated, (640, 480))
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb_frame)
    img_tk = ImageTk.PhotoImage(img_pil)
    label_output.config(image=img_tk)
    label_output.image = img_tk

    messagebox.showinfo("Done", f"Image segmentation complete!\nSaved as: {output_path}")

# ---------------------------------
# Video segmentation
# ---------------------------------
def browse_video():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm")]
    )
    entry_video.delete(0, tk.END)
    entry_video.insert(0, file_path)

def run_video_segmentation():
    global cap, out, fps

    vid_path = entry_video.get()
    if not os.path.isfile(vid_path):
        messagebox.showerror("Error", "Please select a valid video file.")
        return

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output_segmented_video.mp4", fourcc, fps, (width, height))

    btn_video.config(state=tk.DISABLED)
    messagebox.showinfo("Processing", "Video segmentation started.\nClose the window or wait for it to finish.")

    process_next_frame()

def process_next_frame():
    global cap, out, fps

    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            results = model.predict(source=frame, task="segment", conf=0.4, verbose=False)
            annotated_frame = results[0].plot()

            out.write(annotated_frame)

            bgr_frame = cv2.resize(annotated_frame, (640, 480))
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_frame)
            img_tk = ImageTk.PhotoImage(img_pil)

            label_output.config(image=img_tk)
            label_output.image = img_tk

            delay = int(1000 / fps)
            root.after(delay, process_next_frame)
        else:
            finish_video_processing()
    else:
        finish_video_processing()

def finish_video_processing():
    global cap, out

    if cap:
        cap.release()
    if out:
        out.release()

    messagebox.showinfo("Done", "Video segmentation complete!\nSaved as: output_segmented_video.mp4")
    btn_video.config(state=tk.NORMAL)

# ---------------------------------
# GUI Layout
# ---------------------------------
root = tk.Tk()
root.title("YOLOv8 Segmentation GUI")
root.geometry("900x800")

# IMAGE section
label_image = tk.Label(root, text="Image File:")
label_image.pack()
entry_image = tk.Entry(root, width=80)
entry_image.pack()
btn_browse_image = tk.Button(root, text="Browse Image", command=browse_image)
btn_browse_image.pack()
btn_image = tk.Button(root, text="Run Image Segmentation", command=run_image_segmentation)
btn_image.pack()

# VIDEO section
label_video = tk.Label(root, text="Video File:")
label_video.pack()
entry_video = tk.Entry(root, width=80)
entry_video.pack()
btn_browse_video = tk.Button(root, text="Browse Video", command=browse_video)
btn_browse_video.pack()
btn_video = tk.Button(root, text="Run Video Segmentation (Show in GUI)", command=run_video_segmentation)
btn_video.pack()

# Shared output display
label_output = tk.Label(root)
label_output.pack()

root.mainloop()
