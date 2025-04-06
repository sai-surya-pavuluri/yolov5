import os
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from detect import monitor_image_folder  # Import monitor_image_folder from detect.py
from redact import monitor_detections_folder  # Import monitor_detections_folder from redact.py

class ImageHandler(FileSystemEventHandler):
    def __init__(self, input_folder, detections_folder):
        self.input_folder = input_folder
        self.detections_folder = detections_folder

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.jpg', '.jpeg', '.png')):
            print(f"New image detected: {event.src_path}")
            monitor_image_folder(source=event.src_path, project=self.detections_folder)

def start_monitoring(input_folder, detections_folder):
    print("Started monitoring folder for new images and detections...")
    event_handler = ImageHandler(input_folder, detections_folder)
    observer = Observer()
    observer.schedule(event_handler, input_folder, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    input_folder = "C:/forms"  # Folder to monitor for new images
    detections_folder = "C:/forms/detections"  # Folder where detection results will be saved
    redactions_folder = "C:/forms/redactions"  # Folder to save redacted images

    # Ensure necessary folders exist
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    if not os.path.exists(detections_folder):
        os.makedirs(detections_folder)
        os.makedirs(os.path.join(detections_folder, "labels"))
    if not os.path.exists(redactions_folder):
        os.makedirs(redactions_folder)

    # Start monitoring the folders
    threading.Thread(target=start_monitoring, args=(input_folder, detections_folder)).start()
    threading.Thread(target=monitor_detections_folder, args=(detections_folder, redactions_folder)).start()
