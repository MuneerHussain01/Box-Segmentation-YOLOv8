import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import csv
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -----------------------------
# Load YOLOv8 model
# -----------------------------
model = YOLO("best.pt")

# -----------------------------
# Global variables
# -----------------------------
cap = None
out = None
fps = 30
frame_count = 0
total_boxes = 0
occupancy = 0
frame_numbers = []
total_boxes_list = []

LOG_FILE = "warehouse_log.csv"
if not os.path.isfile(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Boxes_Detected", "Total_Boxes", "Occupancy_%"])

# -----------------------------
# Matplotlib Figure for live chart
# -----------------------------
fig, ax = plt.subplots(figsize=(5, 3))
ax.set_title("Total Boxes Over Time")
ax.set_xlabel("Frame")
ax.set_ylabel("Total Boxes")
line, = ax.plot([], [], color='cyan', lw=2)
ax.grid(True)

# -----------------------------
# Image segmentation
# -----------------------------
def browse_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.jfif")]
    )
    entry_image.delete(0, tk.END)
    entry_image.insert(0, file_path)

def run_image_segmentation():
    global total_boxes, occupancy, frame_numbers, total_boxes_list

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

    # Update dashboard
    num_boxes = len(results[0].boxes)
    total_boxes += num_boxes
    occupancy = min(100, total_boxes // 5)
    frame_numbers.append(1)
    total_boxes_list.append(total_boxes)

    label_boxes.config(text=f"Boxes Detected: {num_boxes}")
    label_frames.config(text=f"Frames Processed: 1")
    label_total_boxes.config(text=f"Total Boxes: {total_boxes}")
    label_occupancy.config(text=f"Occupancy: {occupancy}%")

    # Log and update chart
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([1, num_boxes, total_boxes, occupancy])

    update_chart()

    messagebox.showinfo("Done", f"Image segmentation complete!\nSaved as: {output_path}")

# -----------------------------
# Video segmentation
# -----------------------------
def browse_video():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm")]
    )
    entry_video.delete(0, tk.END)
    entry_video.insert(0, file_path)

def run_video_segmentation():
    global cap, out, fps, frame_count, total_boxes, frame_numbers, total_boxes_list

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

    frame_count = 0
    total_boxes = 0
    frame_numbers = []
    total_boxes_list = []

    btn_video.config(state=tk.DISABLED)
    messagebox.showinfo("Processing", "Video segmentation started.\nClose the window or wait for it to finish.")

    process_next_frame()

def process_next_frame():
    global cap, out, fps, frame_count, total_boxes, occupancy

    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            results = model.predict(source=frame, task="segment", conf=0.4, verbose=False)
            annotated_frame = results[0].plot()

            out.write(annotated_frame)

            bgr_frame = cv2.resize(annotated_frame, (640, 480))
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_frame)
            img_tk = ImageTk.PhotoImage(img_pil)

            label_output.config(image=img_tk)
            label_output.image = img_tk

            num_boxes = len(results[0].boxes)
            total_boxes += num_boxes
            occupancy = min(100, total_boxes // 5)

            frame_numbers.append(frame_count)
            total_boxes_list.append(total_boxes)

            label_boxes.config(text=f"Boxes Detected: {num_boxes}")
            label_frames.config(text=f"Frames Processed: {frame_count}")
            label_total_boxes.config(text=f"Total Boxes: {total_boxes}")
            label_occupancy.config(text=f"Occupancy: {occupancy}%")

            with open(LOG_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([frame_count, num_boxes, total_boxes, occupancy])

            update_chart()

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

    messagebox.showinfo("Done", f"Video segmentation complete!\nSaved as: output_segmented_video.mp4\nLog saved in: {LOG_FILE}")
    btn_video.config(state=tk.NORMAL)

def update_chart():
    line.set_data(frame_numbers, total_boxes_list)
    ax.relim()
    ax.autoscale_view()
    canvas.draw()

# -----------------------------
# GUI Layout (modern theme)
# -----------------------------
root = tk.Tk()
root.title("YOLOv8 Warehouse Dashboard")
root.geometry("1200x750")
root.configure(bg="#2c3e50")

# LEFT dashboard frame
frame_dashboard = tk.Frame(root, bg="#34495e", padx=20, pady=20)
frame_dashboard.pack(side="left", fill="y")

label_title = tk.Label(frame_dashboard, text="📦 Warehouse Dashboard", font=("Helvetica", 18, "bold"), fg="white", bg="#34495e")
label_title.pack(pady=10)

label_boxes = tk.Label(frame_dashboard, text="Boxes Detected: 0", font=("Helvetica", 14), fg="white", bg="#34495e")
label_boxes.pack(pady=5)

label_frames = tk.Label(frame_dashboard, text="Frames Processed: 0", font=("Helvetica", 14), fg="white", bg="#34495e")
label_frames.pack(pady=5)

label_total_boxes = tk.Label(frame_dashboard, text="Total Boxes: 0", font=("Helvetica", 14), fg="white", bg="#34495e")
label_total_boxes.pack(pady=5)

label_occupancy = tk.Label(frame_dashboard, text="Occupancy: 0%", font=("Helvetica", 14), fg="white", bg="#34495e")
label_occupancy.pack(pady=5)

# Embed live chart
canvas = FigureCanvasTkAgg(fig, master=frame_dashboard)
canvas.get_tk_widget().pack(pady=20)

# RIGHT main frame
frame_main = tk.Frame(root, bg="#2c3e50", padx=20, pady=20)
frame_main.pack(side="right", expand=True, fill="both")

label_image = tk.Label(frame_main, text="Image File:", font=("Helvetica", 12), fg="white", bg="#2c3e50")
label_image.pack()
entry_image = tk.Entry(frame_main, width=80)
entry_image.pack()
btn_browse_image = tk.Button(frame_main, text="Browse Image", command=browse_image, bg="#2980b9", fg="white", padx=10, pady=5)
btn_browse_image.pack(pady=5)
btn_image = tk.Button(frame_main, text="Run Image Segmentation", command=run_image_segmentation, bg="#27ae60", fg="white", padx=10, pady=5)
btn_image.pack(pady=10)

label_video = tk.Label(frame_main, text="Video File:", font=("Helvetica", 12), fg="white", bg="#2c3e50")
label_video.pack()
entry_video = tk.Entry(frame_main, width=80)
entry_video.pack()
btn_browse_video = tk.Button(frame_main, text="Browse Video", command=browse_video, bg="#2980b9", fg="white", padx=10, pady=5)
btn_browse_video.pack(pady=5)
btn_video = tk.Button(frame_main, text="Run Video Segmentation (Show in GUI)", command=run_video_segmentation, bg="#e67e22", fg="white", padx=10, pady=5)
btn_video.pack(pady=10)

label_output = tk.Label(frame_main, bg="#2c3e50")
label_output.pack(pady=10)

root.mainloop()
