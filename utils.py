import math
import numpy as np
from PIL import ImageDraw, Image
import cv2
import matplotlib.pyplot as plt
import random


def determine_max(rect):

    x_min = math.inf
    x_max = 0

    y_min = math.inf
    y_max = 0

    for i in rect:

        if i[0] < x_min:

            x_min = i[0]

        if i[0] > x_max:

            x_max = i[0]

        if i[1] < y_min:

            y_min = i[1]

        if i[1] > y_max:

            y_max = i[1]

    # return ((x_min,y_min),(x_max,y_max))
    return ((y_min, x_min), (y_max, x_max))


def get_mask(bounding_boxes):

    matrix = np.zeros((50, 50))
    for box in bounding_boxes:
        if box != "name":
            # print("X:",int(int(bounding_boxes[box][1])/20))
            # print("Y:",int(int(bounding_boxes[box][3])/20))
            for j in range(
                int(int(bounding_boxes[box][1]) / 20),
                int(int(bounding_boxes[box][3]) / 20),
            ):
                for i in range(
                    int(int(bounding_boxes[box][0] / 20)),
                    int(int(bounding_boxes[box][2]) / 20),
                ):

                    if i >= 50:
                        i = 49
                    if j >= 50:
                        j = 49

                    matrix[j][i] = 1

    return matrix


def bounding_box(matrix, img):

    matrix = np.asarray(matrix).astype(np.uint8)

    # print(type(matrix))
    # print(matrix)

    contours, hierarchy = cv2.findContours(
        matrix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contour_store = []
    for i in contours:

        contour_store.append(i.reshape((-1, 2)))

    box_find = []

    for i in contour_store:

        f = determine_max(i)
        box_find.append(f)

    mat = np.zeros([50, 50])

    for j in box_find:

        if j[0] != j[1]:

            for a in range(j[0][0], j[1][0]):
                for b in range(j[0][1], j[1][1]):

                    mat[a][b] = 1

        else:

            mat[j[0][0] : j[1][0]] = 1

    draw = ImageDraw.Draw(img)
    for h in box_find:
        draw.rectangle(
            ((h[0][1] * 5, h[0][0] * 5), (h[1][1] * 5, h[1][0] * 5)),
            outline="black",
            width=3,
        )
    plt.imshow(img)
    plt.show()
    # return box_find


def plot_gallery(model, transform):

    imgs = [
        "06_11.jpg",
        "07_01.jpg",
        "07_08.jpg",
        "07_14.jpg",
        "08_15.jpg",
        "00_00.jpg",
    ]

    url = "/Users/hrishikesh/Hrishikesh/Projects/Street View Text Recognition/SVT/svt1/img/"

    for path in imgs:

        print("\n")

        er = model(
            transform(Image.open(url + path).resize((250, 250)))
            .view([-1, 3, 250, 250])
            .double()
        )
        e = er.detach().numpy()
        # print(max(e.flatten()))
        e = e.reshape([50, 50]) > 0.5
        plt.imshow(Image.open(url + path).resize((250, 250)))
        plt.show()
        im = Image.open(url + path).resize((250, 250))
        bounding_box(e, im)

        print("\n")


def crop_img(img, IMG_SIZE, factor=0.3):

    width, height = img.size

    new_arr = []

    for x in range(0, width, 80):
        for y in range(0, height, 80):

            size = random.randint(20, 150)
            new_arr.append(img.crop((x, y, x + size, y + size)))
            # display(img.crop((x,y,x+size,y+size)))

    return new_arr
