import glob
import re
import cv2
import shutil
from pathlib import Path
import os
from os.path import basename, dirname


# Find center point coordinates
def midpoint(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def make_archive(source, destination):
    base = os.path.basename(destination)
    name = base.split('.')[0]
    format = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move('%s.%s' % (name, format), destination)


class ConvertOIDtoYOLO:
    def __init__(self):
        self.dataset_folder_path = r"/content/OIDv4_ToolKit/OID/Dataset/**/*.txt"
        self.classes_indexes = {}
        self.google_drive_dataset_dir = r"/content/gdrive/My Drive/yolo_dataset"

        self.yolo_dataset_directory = r"/content/OIDv4_ToolKit/OID/yolo_dataset"
        if not os.path.exists(self.yolo_dataset_directory):
            os.makedirs(self.yolo_dataset_directory)
        pass

    def get_class_index(self, class_name):
        class_index = self.classes_indexes.get(class_name, None)
        if class_index is None:
            self.classes_indexes[class_name] = len(self.classes_indexes)
            return len(self.classes_indexes) - 1
        return class_index

    def get_image_from_label(self, txt_path):
        """ This function loads the image, by its label path"""
        path_no_extension = os.path.splitext(txt_path)[0]
        image_path = path_no_extension + ".jpg"
        image = cv2.imread(image_path)
        return image

    def move_label_to_parent(self):
        """ Move label to parent directory to stay together with images"""
        txt_file_paths = glob.glob(self.dataset_folder_path, recursive=True)
        print("Found {} labels".format(len(txt_file_paths)))
        print("Moving labels to parents")
        # from Python 3.6

        for path in txt_file_paths:
            # Check if the parent is Label, and if so, then move it
            if basename(dirname(path)) == "Label":
                parent_dir = Path(path).parents[1]
                shutil.move(path, parent_dir)
        print("Labels moved to parent")

    def get_labels_path(self):
        txt_file_paths = glob.glob(self.dataset_folder_path, recursive=True)
        return txt_file_paths

    def convert_labels(self, file_path):
        text_converted = []
        numbers_found = False
        # get image size
        image = self.get_image_from_label(file_path)
        if image is not None:
            height_image, width_image, _ = image.shape
        else:
            print(" {} not found, skipping image")

        if image is not None:
            with open(file_path, "r") as f_o:
                lines = f_o.readlines()

                for line in lines:
                    names = re.findall("[A-Za-z]+", line)
                    numbers = re.findall("[0-9.]+", line)

                    if names:
                        class_name = names[0]
                        class_idx = self.get_class_index(class_name)
                        if numbers:
                            # Get box coordinates
                            x, y, x2, y2 = float(numbers[0]), float(numbers[1]), float(numbers[2]), float(numbers[3])

                            # Center point, width, height
                            box_cx, box_cy = midpoint(x, y, x2, y2)
                            box_width, box_height = x2 - x, y2 - y

                            # Convert to YOLO format
                            cx_yolo = box_cx / width_image
                            cy_yolo = box_cy / height_image
                            width_yolo = box_width / width_image
                            heigth_yolo = box_height / height_image

                            text = "{} {} {} {} {}".format(class_idx, cx_yolo, cy_yolo, width_yolo, heigth_yolo)

                            text_converted.append(text)

                            # Update to say that numbers were found
                            return True, text_converted
        return False, []

    def generate_yolo_dataset(self):
        txt_file_paths = glob.glob(self.dataset_folder_path, recursive=True)

        total_labels = len(txt_file_paths)
        label_idx = 0
        for i, file_path in enumerate(txt_file_paths):
            label_idx += 1
            print("Converting label {} of {}".format(label_idx, total_labels))

            ret, text_converted = self.convert_labels(file_path)
            # Rewrite teh file only if numbers found
            if ret is True:
                with open(file_path, 'w') as fp:
                    for item in text_converted:
                        fp.writelines("%s\n" % item)

                # Move the file and image to the YOLO dataset folder
                path_no_extension = os.path.splitext(file_path)[0]
                image_path = path_no_extension + ".jpg"

                # if the image associate to the label exists, move it
                if os.path.isfile(image_path):
                    # Move label
                    shutil.move(file_path, self.yolo_dataset_directory)

                    # Move image
                    shutil.move(image_path, self.yolo_dataset_directory)

    def zip_yolo_dataset_folder(self):
        print("Saving dataset inside Google Drive ...")

        # create the directory if it doesn't exist
        if not os.path.exists(self.google_drive_dataset_dir):
            os.makedirs(self.google_drive_dataset_dir)
        make_archive(self.yolo_dataset_directory, self.google_drive_dataset_dir + "/dataset.zip")
        print("Image dataset saved inside your google drive, on the folder yolo_dataset.")


if __name__ == "__main__":
    cot = ConvertOIDtoYOLO()
    cot.move_label_to_parent()
    cot.generate_yolo_dataset()
    cot.zip_yolo_dataset_folder()
    print("dataset.zip successfully created and ready to download")
