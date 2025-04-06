import threading
import os
import time
from pathlib import Path
from shutil import move
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import torch
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (
    LOGGER,
    check_img_size,
    non_max_suppression,
    scale_boxes,
    xyxy2xywh,
    cv2,
)
from redact import start_monitoring_redactions_folder

# Paths
WATCH_FOLDER = "C:\\forms"
OUTPUT_FOLDER = "C:\\output"
LABELS_FOLDER = os.path.join(OUTPUT_FOLDER, "labels")

# Ensure output directories exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(LABELS_FOLDER, exist_ok=True)

# Set default arguments
conf_thres = 0.4  # Confidence threshold (default)
imgsz = 416       # Image size (default)
device = "cpu"    # Device for inference (can be set to 'cuda' if GPU is available)
weights = "runs/train/exp4/weights/best.pt"  # Path to weights file
project = "runs/detect"  # Folder for saving results
name = "results"  # Experiment name for saving results
save_txt = True   # Whether to save text labels
exist_ok = True   # Whether to overwrite results

# Load model once
device = select_device(device)
model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((imgsz, imgsz), s=stride)
model.warmup(imgsz=(1, 3, *imgsz))


def detect_single_image(image_path):
    """Detect objects in a single image."""
    dataset = LoadImages(image_path, img_size=imgsz, stride=stride, auto=pt)
    for path, im, im0s, _, _ in dataset:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()
        im /= 255.0
        if len(im.shape) == 3:
            im = im[None]

        pred = model(im, augment=False)
        pred = non_max_suppression(pred, conf_thres, 0.45, classes=None, agnostic=False)

        for i, det in enumerate(pred):
            p = Path(path)
            save_path = os.path.join(OUTPUT_FOLDER, p.name)  # Save image directly to OUTPUT_FOLDER
            txt_path = os.path.join(LABELS_FOLDER, p.stem + ".txt")
            gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]
            annotator = Annotator(im0s.copy(), line_width=3, example=str(names))

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()

                with open(txt_path, "w") as f:
                    for *xyxy, conf, cls in reversed(det):
                        # Get normalized coordinates for bounding box
                        x_center, y_center, width, height = xyxy2xywh(torch.tensor(xyxy)).view(-1).tolist()

                        # Normalize the bounding box values (x_center, y_center, width, height)
                        norm_coords = (cls.item(), x_center / im0s.shape[1], y_center / im0s.shape[0], width / im0s.shape[1], height / im0s.shape[0])

                        # Write only 5 values: class_id, x_center, y_center, width, height
                        f.write(" ".join([str(val) for val in norm_coords]) + "\n")

                # Annotate and save image with bounding boxes
                result_img = annotator.result()
                cv2.imwrite(save_path, result_img)
                LOGGER.info(f"Detection done: {p.name}, saved to {save_path}")

            # After detection, delete the image from the Test folder (since it's already processed)
            os.remove(image_path)  # Delete the image from the Test folder
            LOGGER.info(f"Deleted image from test folder: {image_path}")

            # No need to move since it's already in the Results folder, so skip moving it again
            LOGGER.info(f"Image already saved in results folder: {save_path}")

class ImageCreatedHandler(FileSystemEventHandler):
    def on_created(self, event):
        """Triggered when a new image is added to the monitored folder."""
        if event.is_directory:
            return
        if event.src_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            time.sleep(1)
            LOGGER.info(f"New image detected: {event.src_path}")
            detect_single_image(event.src_path)


def start_monitoring_detections_folder():
    """Starts monitoring for new images in the detection folder."""
    print("Started monitoring folder...")

    stop_event = threading.Event()
    redaction_thread = threading.Thread(
        target=start_monitoring_redactions_folder,
        args=(LABELS_FOLDER, OUTPUT_FOLDER, stop_event)
    )
    redaction_thread.start()

    event_handler = ImageCreatedHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_FOLDER, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        stop_event.set()

    observer.join()
    redaction_thread.join()


def run():
    """Main method to start detection and monitoring."""
    LOGGER.info("Starting detection and monitoring process.")
    start_monitoring_detections_folder()


if __name__ == "__main__":
    run() 