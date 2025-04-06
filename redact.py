import os
import time
import cv2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ImageHandler(FileSystemEventHandler):
    def __init__(self, label_folder, output_folder):
        self.label_folder = label_folder
        self.output_folder = output_folder

    def on_created(self, event):
        # Check if it's a new image file
        if event.is_directory:
            return
        if event.src_path.endswith(('.jpg', '.jpeg', '.png')):  # Image file extensions
            # if 'redacted_' in os.path.basename(event.src_path):
            #     print(f"Skipping redacted image: {event.src_path}")
            #     return
            
            print(f"New image detected: {event.src_path}")
            self.process_image(event.src_path)

    def process_image(self, image_path):
        # Get the name of the label file associated with the image
        label_file_path = os.path.join(self.label_folder, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
        if os.path.exists(label_file_path):
            print(f"Label file found: {label_file_path}")

            # Read all label lines
            with open(label_file_path, 'r') as f:
                lines = f.readlines()

            # Condition 1: require at least two detections
            if len(lines) < 2:
                print("Not enough detections in label file (need at least 2); skipping redaction.")
                os.remove(label_file_path)
                return

            # Parse lines to extract class ids (assuming each line has 5 values)
            class_ids = []
            for line in lines:
                try:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) < 5:
                        print(f"Skipping invalid label line: {line}")
                        continue
                    class_id = int(parts[0])
                    class_ids.append(class_id)
                except ValueError:
                    print(f"Skipping invalid label line: {line}")
                    continue

            # Condition 2: Check if either English (classes 0-3) or Irish (classes 4-7) side classes are present
            if not any(cls in [0, 1, 2, 3, 4, 5, 6, 7] for cls in class_ids):
                print("No relevant class detected (English or Irish); skipping redaction.")
                os.remove(label_file_path)
                return

            # Condition 3: Check for mandatory class presence for redaction
            redaction_condition_met = False
            if 0 in class_ids:  # English side class 0
                if all(cls in class_ids for cls in [1, 2, 3]):  # All English classes 1, 2, and 3 must be present
                    redaction_condition_met = True
            elif 4 in class_ids:  # Irish side class 4
                if all(cls in class_ids for cls in [5, 6, 7]):  # All Irish classes 5, 6, and 7 must be present
                    redaction_condition_met = True

            if not redaction_condition_met:
                print("Redaction conditions not met; skipping redaction.")
                os.remove(label_file_path)
                return

            # Read image
            image = cv2.imread(image_path)

            # Now, process each line – only redact boxes for class 0 or 4
            for line in lines:
                try:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) < 5:
                        print(f"Skipping invalid label line: {line}")
                        continue
                    cls, x_center, y_center, width, height = parts
                    cls = int(cls)
                    if cls not in [0, 4]:  # Only redact class 0 (English) or class 4 (Irish)
                        continue

                    # Convert normalized coordinates to pixel values
                    x_center = int(x_center * image.shape[1])
                    y_center = int(y_center * image.shape[0])
                    width = int(width * image.shape[1])
                    height = int(height * image.shape[0])

                    # Calculate bounding box coordinates
                    x1 = x_center - width // 2
                    y1 = y_center - height // 2
                    x2 = x_center + width // 2
                    y2 = y_center + height // 2

                    # Redact the region (black out the area)
                    image[y1:y2, x1:x2] = (0, 0, 0)
                except ValueError:
                    print(f"Skipping invalid label line: {line}")
                    continue

            # Save the redacted image directly to the results folder, overwriting the original image
            redacted_image_path = os.path.join(self.output_folder, os.path.basename(image_path))  # Save redacted image in place
            cv2.imwrite(redacted_image_path, image)
            print(f"Redacted image saved: {redacted_image_path}")

            # After successful redaction, delete the corresponding label file
            os.remove(label_file_path)  # Delete the label file
            print(f"Deleted label file: {label_file_path}")
        else:
            print(f"Label file for {image_path} not found, skipping redaction.")


def start_monitoring_redactions_folder(label_folder, output_folder, stop_event):
    print("Started monitoring folder... ")
    event_handler = ImageHandler(label_folder, output_folder)
    observer = Observer()
    observer.schedule(event_handler, output_folder, recursive=False)
    observer.start()

    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    label_folder = 'runs/detect/results/labels'  # Folder where the labels are saved
    output_folder = 'runs/detect/redact' # Folder to monitor for new images and redact
 
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Start monitoring the folder
    start_monitoring_redactions_folder(label_folder, output_folder)