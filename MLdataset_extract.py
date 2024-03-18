import os
import shutil
from PIL import Image
import math
import numpy as np

from RMS.Formats import FTPdetectinfo
from RMS.Formats import FFfile

"""
This script is used for preprocessing ML datasets, specifically extracting station's .config files, FTPdetectinfo and .fits files. Fits files are converted to pngs and cropped to the meteor detection according to its location on the image stored in FTPdetectinfo. Default padding of 20px was added as Fiachra Feehilly saw improvements in ML model's performance when the meteor detection was not touching the edges of the image. 
add_zeros_row, add_zeros_col, blackfill, crop_detection, cropPNG are functions taken from the RMS repo and slightly modified
repo link: https://github.com/CroatianMeteorNetwork/RMS
"""


def add_zeros_row(image, top_or_bottom, num_rows_to_add):
    """adds rows of zeros to either the top or bottom of the numpy array
    if performance is important the same effect can be achieved using numpy slicing which is faster
    """
    image_shape = np.shape(image)
    # shape returns (num_rows, num_cols)
    # num_rows = image_shape[0]
    num_cols = image_shape[1]

    zero_rows = np.zeros((num_rows_to_add, num_cols))

    if top_or_bottom == "top":
        new_image = np.vstack((zero_rows, image))
        return new_image
    elif top_or_bottom == "bottom":
        new_image = np.vstack((image, zero_rows))
        return new_image
    # return None which will cause an error if invalid inputs have been used
    return


def add_zeros_col(image, left_or_right, num_cols_to_add):
    """adds columns of zeros to either the left or right of the numpy array
    if performance is important the same effect can be achieved using numpy slicing which is faster
    """
    image_shape = np.shape(image)
    # shape returns (num_rows, num_cols)
    num_rows = image_shape[0]
    # num_cols = image_shape[1]
    zero_cols = np.zeros((num_rows, num_cols_to_add))

    if left_or_right == "left":
        new_image = np.hstack((zero_cols, image))
        return new_image
    elif left_or_right == "right":
        new_image = np.hstack((image, zero_cols))
        return new_image
    # return None which will cause an error if invalid inputs have been used
    return


def blackfill(
    image, leftover_top=0, leftover_bottom=0, leftover_left=0, leftover_right=0
):
    """
    Padds the image with zeros based on leftovers.
    """

    if leftover_top > 0:
        image = add_zeros_row(image, "top", leftover_top)
    if leftover_bottom > 0:
        image = add_zeros_row(image, "bottom", leftover_bottom)
    if leftover_left > 0:
        image = add_zeros_col(image, "left", leftover_left)
    if leftover_right > 0:
        image = add_zeros_col(image, "right", leftover_right)

    if np.shape(image)[0] == np.shape(image)[1]:
        new_image = image[:, :]
        return new_image

    if np.shape(image)[0] < np.shape(image)[1]:
        rows_needed = np.shape(image)[1] - np.shape(image)[0]
        image = add_zeros_row(image, "top", math.floor(rows_needed / 2))
        image = add_zeros_row(image, "bottom", math.ceil(rows_needed / 2))
        new_image = image[:, :]
        return new_image

    if np.shape(image)[1] < np.shape(image)[0]:
        cols_needed = np.shape(image)[0] - np.shape(image)[1]
        image = add_zeros_col(image, "left", math.floor(cols_needed / 2))
        image = add_zeros_col(image, "right", math.ceil(cols_needed / 2))
        new_image = image[:, :]
        return new_image

    return


def crop_detection(detection_info, fits_dir, padding=20):
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
    detect_only = fits_file.maxpixel - fits_file.avepixel

    # set image to only include frames where detection occurs, reduces likelihood that there will then be multiple detections in the same cropped image
    detect_only_frames = FFfile.selectFFFrames(
        detect_only, fits_file, first_frame_no, last_frame_no
    )

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


def cropPNG(fits_path: str, ftp_path: str):
    destination = os.path.dirname(ftp_path)
    image_dest = os.path.join(destination, "images")
    os.makedirs(image_dest, exist_ok=True)

    meteor_list = FTPdetectinfo.readFTPdetectinfo(
        os.path.dirname(ftp_path), os.path.basename(ftp_path)
    )

    for detection_entry in meteor_list:

        # Read FTPdetectinfo name and meteor number
        fits_file_name = detection_entry[0]
        meteor_num = detection_entry[2]

        png_name = (
            fits_file_name.strip(".fits").strip(".bin") + "_" + str(int(meteor_num))
        )
        # print(fits_file_name,os.path.basename(ftp_path))
        if fits_file_name == os.path.basename(fits_path):
            square_crop_image = crop_detection(detection_entry, fits_path)

            # save the Numpy array as a png using PIL
            im = Image.fromarray(square_crop_image)
            im = im.convert("L")  # converts to grescale
            im.save(os.path.join(image_dest, png_name + ".png"))


def extract_data(folder_path, limit=0):
    # limit is number of confirmed images, 0 means no limit
    # stations_config_state = {}
    current_destination = os.path.join(
        destination,
        "ConfirmedFiles/" if "ConfirmedFiles" in folder_path else "RejectedFiles/",
    )
    if "ConfirmedFiles" not in folder_path:
        limit = limit / 4.61  # keep original unbalanced class ratio
    unfiltered_imgs = []

    img_count = 0
    for subfolder in os.listdir(folder_path):

        subfolder_path = os.path.join(folder_path, subfolder)

        # if there is no FTPdetectinfo file in the folder, skip it
        useful = any(
            item.startswith("FTPdetectinfo")
            and item.endswith(".txt")
            and len(item) == 47
            and item[14].isalpha()
            and item[15].isalpha()
            for item in os.listdir(subfolder_path)
        )
        if not useful:
            continue

        # station_name = subfolder[:6]
        # stations_config_state[station_name] = False
        filtered_subfolder_path = os.path.join(current_destination, "nights", subfolder)
        os.makedirs(filtered_subfolder_path, exist_ok=True)

        print("Tranforming FF files in:", subfolder_path)
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)

            # add relevant FF files to processing list
            if file.startswith("FF_") and file.endswith(".fits"):
                unfiltered_imgs.append(file_path)

            # copy relevant ftpdetectinfo file
            if (
                file.startswith("FTPdetectinfo")
                and file.endswith(".txt")
                and len(file) == 47
                and file[14].isalpha()
                and file[15].isalpha()
            ):
                ftp_path = os.path.join(filtered_subfolder_path, file)

                os.makedirs(filtered_subfolder_path, exist_ok=True)
                if not os.path.exists(ftp_path):
                    shutil.copy(file_path, filtered_subfolder_path)
                else:
                    pass
                    # print(
                    #    "FTPdetectinfo file already exists in",
                    #    filtered_subfolder_path,
                    # )
            """
            # copy the config file to the station folder
            if file == ".config":
                stations_config_state[station_name] = True
                station_path = os.path.join(current_destination, "configs",station_name)
                if not os.path.isdir(station_path):
                    os.mkdir(station_path)
                    shutil.copy(file_path, station_path)
                else:
                    print("Station folder already exists in", station_path)
            """

        for i in unfiltered_imgs:
            # preproccess/crop the file here
            cropPNG(i, ftp_path)
            unfiltered_imgs.remove(i)
            img_count += 1

        if limit != 0 and limit <= img_count:  # limit number of images processed
            break
    """
    configs = os.path.join(current_destination, "configs")
    for i in stations_config_state:
        if i not in os.listdir(configs):
            print("Station", i, "is missing a .config file")
    """
    print("\nTotal images processed:", img_count)
    print()


def get_configs(path):
    """
    Retrieves the configurations for each station from the given path and copies the .config files to the appropriate destination.

    Args:
        path (str): The path to the directory containing the station folders.

    Returns:
        None
    """
    current_destination = os.path.join(
        destination,
        (
            "ConfirmedFiles/configs/"
            if "ConfirmedFiles" in path
            else "RejectedFiles/configs/"
        ),
    )
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
                    os.makedirs(station_path)
                    shutil.copy(file_path, station_path)
                else:
                    pass
                    # print("Station folder already exists in", station_path)
    print("Total .config files found:", ct)
    print("Total .config files copied:", ct2)
    for i in stations_config_state:
        if stations_config_state[i] == False:
            print("Station", i, "is missing a .config file")


dirs = ["/home/mldataset/files/ConfirmedFiles/", "/home/mldataset/files/RejectedFiles/"]
destination = "/home/dgrzinic/mldataset/cropped/"
for i in dirs:
    print(i)
    # get_configs(i)
    extract_data(i, 821)