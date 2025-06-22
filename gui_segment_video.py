import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
from ultralytics import YOLO

# Load YOLOv8 segmentation model
model = YOLO("best.pt")  

# ---------------------------
# Image Segmentation Function
# ---------------------------
def segment_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    results = model.predict(source=img, task="segment", conf=0.4, verbose=False)
    annotated = results[0].plot()
    output_path = "output_segmented_image.jpg"
    cv2.imwrite(output_path, annotated)
    return output_path

def run_image_segmentation():
    img_path = entry_image.get()
    if not os.path.isfile(img_path):
        messagebox.showerror("Error", "Please select a valid image file.")
        return

    output = segment_image(img_path)

    # Show result in GUI
    img = Image.open(output)
    img = img.resize((400, 400))
    img_tk = ImageTk.PhotoImage(img)
    label_output.config(image=img_tk)
    label_output.image = img_tk
    messagebox.showinfo("Done", f"Image segmentation complete!\nSaved as: {output}")

# ---------------------------
# Live Video Segmentation
# ---------------------------
def segment_video_live(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "output_segmented_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, task="segment", conf=0.4, verbose=False)
        annotated_frame = results[0].plot()

        # Show in OpenCV window
        cv2.imshow("Live Segmentation - Press 'q' to stop", annotated_frame)

        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path

def run_video_segmentation():
    vid_path = entry_video.get()
    if not os.path.isfile(vid_path):
        messagebox.showerror("Error", "Please select a valid video file.")
        return

    btn_video.config(state=tk.DISABLED)
    messagebox.showinfo("Processing", "Video segmentation started.\nPress 'q' to stop early.")

    output = segment_video_live(vid_path)

    messagebox.showinfo("Done", f"Video segmentation complete!\nSaved as: {output}")
    btn_video.config(state=tk.NORMAL)

# ---------------------------
# Browse Functions
# ---------------------------
def browse_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.jfif")]
    )
    entry_image.delete(0, tk.END)
    entry_image.insert(0, file_path)

def browse_video():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm")]
    )
    entry_video.delete(0, tk.END)
    entry_video.insert(0, file_path)

# ---------------------------
# GUI Layout
# ---------------------------
root = tk.Tk()
root.title("YOLOv8 Segmentation GUI")
root.geometry("800x600")

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
btn_video = tk.Button(root, text="Run Video Segmentation (Live View)", command=run_video_segmentation)
btn_video.pack()

# Output display
label_output = tk.Label(root)
label_output.pack()

root.mainloop()
