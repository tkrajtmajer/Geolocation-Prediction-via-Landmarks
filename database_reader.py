import os, sys
from pathlib import Path
from SIFT_transform import *


currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


def read_images(path):
    # get all image paths in the folder path
    image_names = os.listdir(path)

    # read all images
    images = []
    for name in image_names:
        image_path = path + "/" + name
        images.append((name, cv2.imread(image_path)))

    # return the image list
    return images


def create():
    db_path = "resources/db.txt"

    # open database or create a new database if not exist
    db_file = Path(db_path)
    db_file.touch(exist_ok=True)

    # read all images from the given folder
    images = read_images("resources/images/aaf")

    data = []
    # compute descriptor for each image
    for item in images:
        (name, image) = item
        if (name == ".DS_Store"):
            continue

        print("processing", name)
        kp, des = make_sift(image)
        data.append([name, des])

    # save the descriptors
    with open(db_path, 'w', encoding='UTF8', newline='') as f:
        cnt = 0
        f.write(str(len(data)) + "\n")
        for item in data:
            name = item[0]
            desc = item[1]

            # save name and dimension of the descriptor for reading again in the future
            f.write(name + " " + str(desc.shape[0]) + " " + str(desc.shape[1]) + "\n")

            # adjust the accuracy of saving here
            np.savetxt(f, desc, fmt='%.4e')

            # for debuging purpose
            cnt += 1
            print("written", cnt, "/", len(data), ":", name)


def read(path):
    print("Read database")
    result = []

    if os.stat(path).st_size == 0:
        return result

    with open(path, 'r') as f:
        line_content = f.readline()

        for i in range(int(line_content)):
            line_content = f.readline()
            contents = line_content.split(" ")
            name = contents[0]

            row = int(contents[1])
            col = int(contents[2])

            descriptors = []
            for j in range(row):
                values = list(map(np.float32, f.readline().split(" ")))
                descriptors.append(values)

            result.append([name, np.array(descriptors)])

    return result


# create()
