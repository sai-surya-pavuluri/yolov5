import os
import time
import logging
from pdf2image import convert_from_path
from fpdf import FPDF
from PIL import Image
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import torch
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import LOGGER, check_img_size, non_max_suppression, scale_boxes, xyxy2xywh, cv2

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
WATCH_FOLDER = "C:\\forms"
OUTPUT_FOLDER = "C:\\output"
DETECTIONS_FOLDER = "C:\\detections"
LABELS_FOLDER = os.path.join(DETECTIONS_FOLDER, "labels")

# Ensure output directories exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DETECTIONS_FOLDER, exist_ok=True)
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

# Dictionary to track pages and their redaction status
pdf_pages_status = {}

# Redaction logic for the images
def apply_redaction(image, labels_path):
    """Apply redaction to the image based on the labels."""
    redacted = False
    with open(labels_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = list(map(float, line.strip().split()))
        cls = int(parts[0])

        # English: Class 0 must have classes 1, 2, and 3 for redaction
        # Irish: Class 4 must have classes 5, 6, and 7 for redaction
        if cls in [0, 4]:
            # Check if the necessary classes are present for redaction
            class_ids = [int(part[0]) for part in lines]
            if cls == 0 and (all(x in class_ids for x in [1, 2, 3]) or all(x in class_ids for x in [5,6,7])):  # English classes rule
                redacted = True
            elif cls == 4 and (all(x in class_ids for x in [1, 2, 3]) or all(x in class_ids for x in [5,6,7])):  # Irish classes rule
                redacted = True

            if redacted:
                # Get bounding box coordinates
                x_center, y_center, width, height = parts[1], parts[2], parts[3], parts[4]
                # Convert to pixel coordinates
                x1 = int((x_center - width / 2) * image.shape[1])
                y1 = int((y_center - height / 2) * image.shape[0])
                x2 = int((x_center + width / 2) * image.shape[1])
                y2 = int((y_center + height / 2) * image.shape[0])

                # Redact the area by setting the pixels to black
                image[y1:y2, x1:x2] = (0, 0, 0)
                os.remove(labels_path)

    return image, redacted

def detect_and_redact(image_path, pdf_name, page_index):
    """Detect objects in a single image and apply redaction if necessary."""
    try:
        # Run the detection
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
                save_path = os.path.join(DETECTIONS_FOLDER, p.name)  # Save image directly to DETECTIONS_FOLDER
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

                # After detection, apply redaction if necessary
                image_with_redaction, redacted = apply_redaction(result_img, txt_path)
                if redacted:
                    # Overwrite the image with redacted content
                    cv2.imwrite(save_path, image_with_redaction)
                    logger.info(f"Redacted image saved: {save_path}")

                # If no redaction was applied, leave the image intact
                else:
                    cv2.imwrite(save_path, result_img)
                    logger.info(f"No redaction needed for {image_path}, saved as is.")
                    os.remove(txt_path)

                # Track the redacted image path for PDF merging
                if save_path not in pdf_pages_status[pdf_name]["pages"]:
                    pdf_pages_status[pdf_name]["pages"].append(save_path)   

                # Increment redacted pages count if redaction was applied
                if redacted:
                    pdf_pages_status[pdf_name]["redacted_pages"] += 1

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")

def process_redacted_pdf(pdf_name):
    """After redaction, merge the redacted images back into a single PDF."""
    try:
        redacted_images = pdf_pages_status[pdf_name]["pages"]
        output_pdf_path = os.path.join(OUTPUT_FOLDER, pdf_name)
        images_to_pdf(redacted_images, output_pdf_path)
        logger.info(f"Redacted PDF saved to: {output_pdf_path}")

        # After merging, delete the temporary images
        for image_path in redacted_images:
            if os.path.exists(image_path):
                os.remove(image_path)
                logger.info(f"Deleted temporary image: {image_path}")

        # Delete the original PDF and label files
        original_pdf_path = os.path.join(WATCH_FOLDER, pdf_name)
        if os.path.exists(original_pdf_path):
            os.remove(original_pdf_path)
            logger.info(f"Deleted original PDF: {original_pdf_path}")
        
        # Delete label files
        for image_path in redacted_images:
            label_file = image_path.replace(".png", ".txt")
            if os.path.exists(label_file):
                os.remove(label_file)
                logger.info(f"Deleted label file: {label_file}")

    except Exception as e:
        logger.error(f"Error processing redacted PDF for {pdf_name}: {str(e)}")

def images_to_pdf(images, output_pdf_path):
    """Convert a list of images back into a PDF, without resizing or cropping, while centering the image."""
    try:
        image_list = []
        for image_path in images:
            image = Image.open(image_path)
            image_list.append(image)

        if image_list:
            image_list[0].save(output_pdf_path, "PDF", save_all=True, append_images=image_list[1:])
            print(f"Redacted PDF saved to: {output_pdf_path}")
        else:
            print("No images to convert.")
    except Exception as e:
        print(f"Error generating PDF: {e}")

# Define the ImageCreatedHandler class for PDF monitoring
class ImageCreatedHandler(FileSystemEventHandler):
    def __init__(self, input_folder, output_folder):
        """Initialize the handler with input and output folders."""
        self.input_folder = input_folder
        self.output_folder = output_folder

    def on_created(self, event):
        """Triggered when a new PDF is added to the monitored folder."""
        if event.is_directory:
            return

        # Check if the event is a PDF
        if event.src_path.lower().endswith(".pdf"):
            logger.info(f"New PDF detected: {event.src_path}")
            self.process_pdf(event.src_path)  # Process PDF

    def process_pdf(self, pdf_path):
        """Convert PDF to images and process each page."""
        try:
            logger.info(f"Converting PDF {pdf_path} to images...")
            images = convert_from_path(pdf_path, dpi=300)  # Convert PDF to images (one per page)

            # Generate a unique key for the PDF
            pdf_name = os.path.basename(pdf_path)
            pdf_pages_status[pdf_name] = {"pages": [], "redacted_pages": 0, "total_pages": len(images)}
            count = 0
            # Process each image (one per page)
            for i, image in enumerate(images):
                timestamp = int(time.time() * 1000)  # Timestamp for uniqueness
                image_name = f"{os.path.splitext(pdf_name)[0]}_{timestamp}_page_{i}.png"
                image_path = os.path.join(DETECTIONS_FOLDER, image_name)  # Temporary image path for the page
                image.save(image_path, "PNG")
                logger.info(f"Processing page {i} as image...")

                # Detect and redact if necessary
                detect_and_redact(image_path, pdf_name, i)
                count += 1

                # Keep track of the page's path
                # pdf_pages_status[pdf_name]["pages"].append(image_path)
                

            # Check if all pages of the PDF have been processed and redacted
            if count == pdf_pages_status[pdf_name]["total_pages"]:
                process_redacted_pdf(pdf_name)  # Merge pages back into a PDF
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")

# Monitoring function for PDFs
def monitor_for_pdfs(input_folder, output_folder):
    """Monitor the input folder for new PDFs and process them."""
    try:
        event_handler = ImageCreatedHandler(input_folder, output_folder)
        observer = Observer()
        observer.schedule(event_handler, input_folder, recursive=False)
        observer.start()
        logger.info("Started monitoring PDFs in forms folder.")
        while True:
            time.sleep(1)
    except Exception as e:
        logger.error(f"Error monitoring PDFs: {str(e)}")

# Start monitoring PDFs in the forms folder
if __name__ == "__main__":
    try:
        monitor_for_pdfs(WATCH_FOLDER, OUTPUT_FOLDER)
    except Exception as e:
        logger.error(f"Error starting the detection process: {str(e)}")
