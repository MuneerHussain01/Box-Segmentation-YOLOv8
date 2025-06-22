import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import cv2
import os

# ---------- YOLO Model ----------
MODEL_PATH = "best.pt"  
model = YOLO(MODEL_PATH)

# ---------- Process Video ----------
def segment_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "output_segmented.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, task="segment", conf=0.4, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    return output_path

# ---------- GUI Logic ----------
def select_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
    )
    if file_path:
        entry_var.set(file_path)

def run_segmentation():
    video_path = entry_var.get()
    if not os.path.isfile(video_path):
        messagebox.showerror("Error", "Please select a valid video file.")
        return

    btn_run.config(state=tk.DISABLED)
    messagebox.showinfo("Processing", "Segmentation started. Please wait...")

    output = segment_video(video_path)

    messagebox.showinfo("Done", f"Segmentation complete!\nOutput saved as: {output}")
    btn_run.config(state=tk.NORMAL)

# ---------- Build GUI ----------
root = tk.Tk()
root.title("YOLOv8 Segmentation - Video Processor")

root.geometry("500x200")
root.resizable(False, False)

entry_var = tk.StringVar()

label = tk.Label(root, text="Select a video file to segment:")
label.pack(pady=10)

entry = tk.Entry(root, textvariable=entry_var, width=50)
entry.pack(pady=5)

btn_browse = tk.Button(root, text="Browse", command=select_file)
btn_browse.pack(pady=5)

btn_run = tk.Button(root, text="Run Segmentation", command=run_segmentation)
btn_run.pack(pady=20)

root.mainloop()
