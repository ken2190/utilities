# This files prepare a custom Object detector for YOLO v4
#
import glob
import errno
import os
import argparse
import re
import zipfile
from pathlib import Path


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-c", "--n_classes", type=int, required=True, help="number of classes")
ap.add_argument("-b", "--backup", type=str, required=False, help="backup path")
#ap.add_argument("-a", "--method", type=str, default="t", choices=["t", "n"], help="test")
#ap.add_argument("-r", "--radius", type=int, default=3, help="in")
args = vars(ap.parse_args())


class CustomYOLODetector:
    def __init__(self):
        # Files location
        self.custom_cfg_path = "cfg/yolov4-custom.cfg"
        self.new_custom_cfg_path = "cfg/yolov4-custom-detector.cfg"
        self.obj_data_path = "data/obj.data"
        self.images_folder_path = "data/obj/"
        self.backup_folder_path = "backup/"

        # Argument import
        if args["backup"]:
            self.backup_folder_path = args["backup"]

        self.n_classes = 0

    def count_classes_number(self):
        # Detect number of Classes by reading the labels indexes
        # If there are missing indexes, normalize the number of classes by rewriting the indexes starting from 0
        txt_file_paths = glob.glob(self.images_folder_path + "/**/*.txt", recursive=True)
        print(txt_file_paths)
        # Count number of classes
        class_indexes = set()
        for i, file_path in enumerate(txt_file_paths):
            # get image size
            with open(file_path, "r") as f_o:
                lines = f_o.readlines()
                for line in lines:
                    numbers = re.findall("[0-9.]+", line)
                    if numbers:
                        # Define coordinates
                        class_idx = int(numbers[0])
                        class_indexes.add(class_idx)

        # Update classes number
        self.n_classes = len(class_indexes)

        # Verify if there are missing indexes
        print("Classes detected: {}".format(len(class_indexes)))
        if max(class_indexes) > len(class_indexes) - 1:
            print("Class indexes missing")
            print("Normalizing and rewriting classes indexes so that they have consecutive index number")
            # Assign consecutive indexes, if there are missing ones
            # for example if labels are 0, 1, 3, 4 so the index 2 is missing
            # rewrite labels with indexes 0, 1, 2, 3
            new_indexes = {}
            classes = sorted(class_indexes)
            for i in range(len(classes)):
                new_indexes[classes[i]] = i

            for i, file_path in enumerate(txt_file_paths):
                # get image size
                with open(file_path, "r") as f_o:
                    lines = f_o.readlines()
                    text_converted = []
                    for line in lines:
                        numbers = re.findall("[0-9.]+", line)
                        if numbers:
                            # Define coordinates
                            class_idx = new_indexes.get(int(numbers[0]))
                            class_indexes.add(class_idx)
                            text = "{} {} {} {} {}".format(0, numbers[1], numbers[2], numbers[3], numbers[4])
                            text_converted.append(text)
                    # Write file
                    with open(file_path, 'w') as fp:
                        for item in text_converted:
                            fp.writelines("%s\n" % item)

    def generate_yolo_custom_cfg(self):
        """
        This files loads the yolo
        :return:
        """
        # 1) Edit CFG file
        print("Generating YOLO Configuration file for {} classes".format(self.n_classes))
        #print("Classes number: {}")
        with open(self.custom_cfg_path, "r") as f_o:
            cfg_lines = f_o.readlines()

        # Edit subdivision, max_batches, classes and filters on configuration file
        # Max batches 2000*class, but a minimum of 6000
        max_batches = self.n_classes * 2000
        max_batches = 6000 if max_batches < 6000 else max_batches
        print("Max batches: {}".format(max_batches))

        cfg_lines[6] = "subdivisions=32\n"
        cfg_lines[19] = "max_batches = {}\n".format(max_batches)

        # Classes number
        cfg_lines[969] = "classes={}\n".format(self.n_classes)
        cfg_lines[1057] = "classes={}\n".format(self.n_classes)
        cfg_lines[1145] = "classes={}\n".format(self.n_classes)

        # Filters
        filters = (self.n_classes + 5) * 3
        cfg_lines[962] = "filters={}\n".format(filters)
        cfg_lines[1050] = "filters={}\n".format(filters)
        cfg_lines[1138] = "filters={}\n".format(filters)
        print("Filters: {}".format(filters))

        # Saving edited file
        with open(self.new_custom_cfg_path, "w") as f_o:
            f_o.writelines(cfg_lines)

    def generate_obj_data(self):
        obj_data = 'classes= {}\ntrain  = data/train.txt\nvalid  = data/test.txt\nnames = data/obj.names\nbackup = {}'\
            .format(self.n_classes, self.backup_folder_path)

        # Saving Obj data
        with open(self.obj_data_path, "w") as f_o:
            f_o.writelines(obj_data)

    def generate_train_val_files(self):
        print("Generating Train/Validation list")
        images_list = glob.glob(self.images_folder_path + "*.jpg")
        print("{} Images found".format(len(images_list)))
        if len(images_list) == 0:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), "Images list not found. Make sure that the images are with .jpg "
                                                         "format and inside the directory {}".format(self.images_folder_path))

        # Read labels
        for img_path in images_list:
            img_name_basename = os.path.basename(img_path)
            img_name = os.path.splitext(img_name_basename)[0]

        with open("data/train.txt", "w") as f_o:
            f_o.write("\n".join(images_list))
        print("Train.txt generated")

    def extract_zip_file(self, path_to_zip_file):
        print("Extracting Images")
        with zipfile.ZipFile(self.images_folder_path + "images.zip", 'r') as zip_ref:
            zip_ref.extractall(self.images_folder_path)

if "__main__" == __name__:
    cyd = CustomYOLODetector()

    # Extract images
    cyd.count_classes_number()
    cyd.generate_yolo_custom_cfg()
    cyd.generate_obj_data()
    cyd.generate_train_val_files()
