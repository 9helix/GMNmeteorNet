import argparse
import math
import os
import random
import shutil
from datetime import datetime
import tarfile
import time

from PIL import Image

from RMS.Formats import FFfile, FTPdetectinfo
from RMS.MLFilter import blackfill
random.seed(10)  # so that rerunning the script when creating random dataset gives same images
"""
This script is used for setting up ML dataset, specifically extracting station's .config files, FTPdetectinfo and .fits files on the server-side. Fits files are converted into pngs and cropped to the meteor detection (stored in FTPdetectinfo) according to its location on the original image. Default padding of 20px was added as Fiachra Feehilly saw improvements in ML model's performance when the meteor detection was not touching the edges of the image. 
Functions crop_detection, cropPNG  were taken from the MLFilter.py on the RMS repo and slightly modified.
repo link: https://github.com/CroatianMeteorNetwork/RMS/blob/master/RMS/MLFilter.py
"""


def crop_detection(detection_info, fits_dir, padding=20, should_crop=True):
    # taken from MLFilter.crop_detections
    fits_file_name = detection_info[0]
    # meteor_num = detection_info[2]
    # num_segments = detection_info[3]
    first_frame_info = detection_info[11][0]
    first_frame_no = first_frame_info[1]
    last_frame_info = detection_info[11][-1]
    last_frame_no = last_frame_info[1]
    # print(os.path.dirname(fits_dir), fits_file_name)
    fits_file = FFfile.read(os.path.dirname(fits_dir), fits_file_name, fmt="fits")

    # image array with background set to 0 so detections stand out more
    # TODO inlcude code to use mask for the camera, currently masks not available on the data given to me, Fiachra Feehilly (2021)
    if fits_file is None:
        return None
    detect_only = fits_file.maxpixel - fits_file.avepixel

    # set image to only include frames where detection occurs, reduces likelihood that there will then be multiple detections in the same cropped image
    detect_only_frames = FFfile.selectFFFrames(
        detect_only, fits_file, first_frame_no, last_frame_no
    )
    if not should_crop:
        return detect_only_frames
    # get size of the image
    row_size = detect_only_frames.shape[0]
    col_size = detect_only_frames.shape[1]

    # side 1, 2 are the left and right sides but still need to determine which is which
    # left side will be the lesser value as the value represents column number
    side_1 = first_frame_info[2]
    side_2 = last_frame_info[2]
    if side_1 > side_2:
        right_side = (
            math.ceil(side_1) + 1
        )  # rounds up and adds 1 to deal with Python slicing so that it includes everything rather than cutting off the last column
        left_side = math.floor(side_2)
    else:
        left_side = math.floor(side_1)
        right_side = math.ceil(side_2) + 1

    # side 3 and 4 are the top and bottom sides but still need to determine which is which
    # bottom side will be the higher value as the value represents the row number
    side_3 = first_frame_info[3]
    side_4 = last_frame_info[3]
    if side_3 > side_4:
        bottom_side = math.ceil(side_3) + 1
        top_side = math.floor(side_4)
    else:
        top_side = math.floor(side_3)
        bottom_side = math.ceil(side_4) + 1

    # add some space around the meteor detection so that its not touching the edges
    # leftover terms need to be set to 0 outside if statements otherwise they wont be set if there's nothing left over which will cause an error with the blackfill.blackfill() line
    left_side = left_side - padding
    leftover_left = 0
    if left_side < 0:
        # this will be used later to determine how to fill in the rest of the image to make it square but also have the meteor centered in the image
        leftover_left = 0 - left_side
        left_side = 0

    right_side = right_side + padding
    leftover_right = 0
    if right_side > col_size:
        leftover_right = right_side - col_size
        right_side = col_size

    top_side = top_side - padding
    leftover_top = 0
    if top_side < 0:
        leftover_top = 0 - top_side
        top_side = 0

    bottom_side = bottom_side + padding
    leftover_bottom = 0
    if bottom_side > row_size:
        leftover_bottom = bottom_side - row_size
        bottom_side = row_size

    # get cropped image of the meteor detection
    # first index set is for row selection, second index set is for column selection
    crop_image = detect_only_frames[top_side:bottom_side, left_side:right_side]
    square_crop_image = blackfill(
        crop_image, leftover_top, leftover_bottom, leftover_left, leftover_right
    )

    return square_crop_image


def cropPNG(fits_path: str, ftp_path: str, destination: str):
    ftp_dir = os.path.dirname(ftp_path)
    # image_dest = os.path.join(destination, "images")
    # os.makedirs(image_dest, exist_ok=True)

    meteor_list = FTPdetectinfo.readFTPdetectinfo(ftp_dir, os.path.basename(ftp_path))
    ct = 0
    for detection_entry in meteor_list:

        # Read FTPdetectinfo name and meteor number
        fits_file_name = detection_entry[0]
        meteor_num = detection_entry[2]

        png_name = (
            fits_file_name.strip(".fits").strip(".bin") + "_" + str(int(meteor_num))
        )
        # print(fits_file_name,os.path.basename(ftp_path))
        if fits_file_name == os.path.basename(fits_path):
            square_crop_image = crop_detection(
                detection_entry,
                fits_path,
                padding=args.p,
                should_crop=args.no_crop,
            )
            if square_crop_image is None:
                continue
            # save the Numpy array as a png using PIL
            im = Image.fromarray(square_crop_image)
            im = im.convert("L")  # converts to grescale
            im.save(os.path.join(destination, png_name + ".png"))
            ct += 1
    return ct


def extract_data(folder_path, limit=0):
    """
    Extracts relevant data for ML dataset from the given folder path.

    Args:
        folder_path (str): The path to the folder containing the data.
        limit (int, optional): The number of confirmed images to extract.
            If set to 0, there is no limit. Default is 0.

    Returns:
        None
    """
    current_destination = os.path.join(
        destination,
        "Meteors/" if "ConfirmedFiles" in folder_path else "Artifacts/",
    )
    os.makedirs(current_destination, exist_ok=True)

    # apply limits per class
    if "ConfirmedFiles" in folder_path:
        if args.k:
            limit = int(limit * 0.8217)
        else:
            limit = int(limit * 0.5)
    else:
        if args.k:
            limit = int(limit * 0.1783)
        else:
            limit = int(limit * 0.5)

    fits_count = 0
    png_count = 0

    folder = os.listdir(folder_path)
    if args.newest_first:
        folder = sorted(
            folder,
            key=lambda x: datetime.strptime(
                x.split("_")[1] + x.split("_")[2], "%Y%m%d%H%M%S"
            ),
            reverse=True,
        )
    else:
        random.shuffle(folder)

    stop = False
    for subfolder in folder:

        subfolder_path = os.path.join(folder_path, subfolder)

        # station_name = subfolder[:6]
        # stations_config_state[station_name] = False
        # filtered_subfolder_path = os.path.join(current_destination, subfolder)
        # os.makedirs(filtered_subfolder_path, exist_ok=True) saving all images in same folder for now
        unfiltered_imgs = []
        temp = []
        ftp_path = None
        print("Fecthing files in:", subfolder_path)

        files = os.listdir(subfolder_path)
        #if args.l > 0: not needed for now
        #    random.shuffle(files)
        for file in files:
            file_path = os.path.join(subfolder_path, file)

            # add relevant FF files to processing list
            if file.startswith("FF_") and file.endswith(".fits"):
                temp.append(file_path)

            # copy relevant ftpdetectinfo file
            if (
                file.startswith("FTPdetectinfo")
                and file.endswith(".txt")
                and len(file) == 47
                and file[14].isalpha()
                and file[15].isalpha()
            ):
                ftp_path = os.path.join(subfolder_path, file)

                """ might extract into separate function later, as these are  not required for training the model     
                ftp_path = os.path.join(filtered_subfolder_path, file)       
                os.makedirs(filtered_subfolder_path, exist_ok=True)
                if not os.path.exists(ftp_path):
                    shutil.copy(file_path, filtered_subfolder_path)
                else:
                    pass """
        if ftp_path is None:
            continue
        unfiltered_imgs.extend(temp)
        del temp
        for i in range(len(unfiltered_imgs)):
            # preproccess/crop the file here
            png_count += cropPNG(unfiltered_imgs[i], ftp_path, current_destination)
            # it can produce more than one image
            fits_count += 1
            if 0 < limit <= png_count:  # limit number of images processed
                stop = True
                break

            if "RejectedFiles" in ftp_path and i >= args.l - 1 >= 0:
                print("Limit reached for artifacts. Skipping the rest of the folder...")
                break

        print(f"{png_count}/{limit}")
        if stop:
            break

    print("\nTotal fits processed:", fits_count)
    print("Total pngs generated:", png_count)
    print()


def get_configs(path):
    """
    Retrieves the configurations for each station from the given path and copies the .config files to the appropriate destination.

    Args:
        path (str): The path to the directory containing the station folders.

    Returns:
        None
    """
    current_destination = os.path.join(destination, "configs")
    stations_config_state = {}
    ct = 0
    ct2 = 0
    for subfolder in os.listdir(path):
        station_name = subfolder[:6]

        subfolder_path = os.path.join(path, subfolder)
        if station_name not in stations_config_state:
            stations_config_state[station_name] = False

        for file in os.listdir(subfolder_path):

            if file == ".config":
                ct += 1
                file_path = os.path.join(subfolder_path, file)
                stations_config_state[station_name] = True

                station_path = os.path.join(current_destination, station_name)

                if not os.path.isdir(station_path):
                    print("Found .config for", station_name)

                    ct2 += 1
                    os.makedirs(station_path, exist_ok=True)
                    shutil.copy(file_path, station_path)
                else:
                    pass

    print("Total .config files found:", ct)
    print("Total .config files copied:", ct2)
    for i in stations_config_state:
        if stations_config_state[i] == False:
            print("Station", i, "is missing a .config file")

def get_ftps(path):
    """
    Retrieves the FTPdetectinfo files from the given path and copies them to the appropriate destination.

    Args:
        path (str): The path to the directory containing the station folders.

    Returns:
        None
    """
    current_destination = os.path.join(destination, "FTPdetectinfo")
    ct = 0
    for subfolder in os.listdir(path):
        subfolder_path = os.path.join(path, subfolder)
        for file in os.listdir(subfolder_path):
            if (
                file.startswith("FTPdetectinfo")
                and file.endswith(".txt")
                and len(file) == 47
                and file[14].isalpha()
                and file[15].isalpha()
            ):
                ct += 1
                file_path = os.path.join(subfolder_path, file)
                

                print("Found FTPdetectinfo for", subfolder)
                os.makedirs(current_destination, exist_ok=True)
                shutil.copy(file_path, current_destination)
               

    print("Total FTPdetectinfo files found:", ct)
# Create a parser for the command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "-c", action="store_true", help="Execute get_configs instead of extract_data"
)
parser.add_argument(
    "-f", action="store_true", help="Execute get_ftps instead of extract_data"
)
parser.add_argument(
    "-n",
    type=int,
    nargs="?",
    default=1000,
    help="Number of images to extract. May vary slightly due to different amount of detections in a single fits file. Use 0 to disable limit.",
)
parser.add_argument(
    "-p", type=int, nargs="?", default=20, help="Detection padding in px"
)
parser.add_argument("--no_crop", action="store_false", help="Disable image cropping")
parser.add_argument(
    "--newest-first",
    action="store_true",
    help="Extract files starting from newest first. Default is random order.",
)
parser.add_argument(
    "-k", action="store_true", help="Keeps class imbalance of the original dataset."
)
parser.add_argument(
    "-l",
    type=int,
    default=0,
    help="Limit of extracted images per folder for artifacts. Default is 0 (no limit).",
) #this is useful because some night might have very similar artifacts, so their number in a certain night is limited to avoid overfitting

# Parse the command-line arguments
args = parser.parse_args()

dirs = ["/home/mldataset/files/archived/ConfirmedFiles/", "/home/mldataset/files/archived/RejectedFiles/"]
destination = "datasets/"
dataset_name = f"GMN_n{args.n}_p{args.p}{f'_l{args.l}' if args.l>0 else ''}_{'newest' if args.newest_first else 'random'}{'_no_crop' if not args.no_crop else ''}{'_unbalanced' if args.k else ''}"
destination = os.path.join(destination, dataset_name)

if os.path.exists(destination):
    print(
        f"Dataset {dataset_name} already exists. Do you want to overwrite it? (y/n) ",
        end="",
    )
    if input().lower() != "y":
        print("Exiting...")
        exit()
    shutil.rmtree(destination)
print("Creating dataset", dataset_name, "...\n\n")
start_time = time.time()
for i in dirs:
    if args.c:
        print("Getting configs from", i)
        get_configs(i)
    elif args.f:
        print("Getting FTPdetectinfo from", i)
        get_ftps(i)
    else:
        print("Extracting data from", i, "\n")
        extract_data(i, args.n)

print("\nCompressing and archiving the dataset...")
with tarfile.open(f"{destination}.tar.bz2", "w:bz2") as tar:
    tar.add(destination, arcname=dataset_name)
end_time = time.time()
print(f"{dataset_name}.tar.bz2 has been created successfully.")
elapsed_time = (end_time - start_time) / 60
print(f"Total elapsed time: {elapsed_time:.2f} minutes")
