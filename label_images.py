import os
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import math
import random
import yaml
from train_utils import load_checkpoint


def inference(img_path, model, testdata_transform):

    # images=["18_11.jpg"]
    img = Image.open(img_path).resize((1000, 1000))

    img_x = []
    img_y = []
    img_score = []
    high_size = []

    width, height = img.size

    for x in range(0, width, 20):
        for y in range(0, height, 20):
            for size in range(0, 200, 20):
                im1 = img.crop((x, y, x + size, y + size)).resize((120, 120))
                img_t = testdata_transform(im1)
                batch_t = torch.unsqueeze(img_t, 0)
                out = model(batch_t.double())
                out = torch.nn.Softmax()(out)
                item = torch.argmax(out).item()
                score = torch.max(out).item()

                if item == 1 and score >= 0.90:

                    img_score.append(score)
                    img_x.append(x)
                    img_y.append(y)
                    # animal.append(item)
                    high_size.append(size)
                    # display(im1)

    for i in range(0, len(img_x)):
        im1 = img.crop(
            (img_x[i], img_y[i], img_x[i] + high_size[i], img_y[i] + high_size[i])
        )
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            ((img_x[i], img_y[i]), (img_x[i] + high_size[i], img_y[i] + high_size[i])),
            outline="black",
            width=10,
        )

    return img_x, img_y, high_size


def getcoordinates(img_x, img_y, high_size):
    arr = []
    for i in range(0, len(img_x)):
        arr.append(
            [img_x[i], img_y[i], img_x[i] + high_size[i], img_y[i] + high_size[i]]
        )
    return arr


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def rect_distance(p1, p2):

    x1 = p1[0]
    y1 = p1[1]
    x1b = p1[2]
    y1b = p1[3]

    x2 = p2[0]
    y2 = p2[1]
    x2b = p2[2]
    y2b = p2[3]

    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist((x1, y1b), (x2b, y2))
    elif left and bottom:
        return dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        return dist((x1b, y1), (x2, y2b))
    elif right and top:
        return dist((x1b, y1b), (x2, y2))
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:  # rectangles intersect
        return 0.0


def get_boundingboxes(arr):

    Rect_boxes = []

    index = 0

    for rect1 in arr:

        if len(Rect_boxes):

            added = False

            for rect2 in range(0, len(Rect_boxes)):

                arr1 = Rect_boxes[rect2]
                if (
                    rect_distance(
                        (rect1[0], rect1[1], rect1[2], rect1[3]),
                        (arr1[0], arr1[1], arr1[2], arr1[3]),
                    )
                    <= 0.0
                ):

                    Rect_boxes[rect2].append(rect1)
                    added = True
                    rect2 = len(Rect_boxes)

            if not added:

                Rect_boxes.append(rect1)

        else:

            Rect_boxes.append(rect1)

    index += 1

    return Rect_boxes


def lolify(Rect_boxes):

    new_rects = []

    for i in Rect_boxes:

        lol_rect = []

        rect_x = i[0]
        rect_y = i[1]
        rect_w = i[2]
        rect_h = i[3]

        lol_rect.append([rect_x, rect_y, rect_w, rect_h])

        for e in range(4, len(i)):

            lol_rect.append(i[e])

        new_rects.append(lol_rect)

    return new_rects


def suppression(new_rets):

    bounding_boxes = []

    for rect in new_rets:

        x1min = 1000000
        y1min = 1000000

        x2max = 0.0
        y2max = 0.0

        for rectangle in rect:

            if rectangle[0] < x1min:

                x1min = rectangle[0]

            if rectangle[1] < y1min:

                y1min = rectangle[1]

            if rectangle[2] > x2max:

                x2max = rectangle[2]

            if rectangle[3] > y2max:

                y2max = rectangle[3]

        bounding_boxes.append([x1min, y1min, x2max, y2max])

    return bounding_boxes


def infer_boundingbox(img_path, model, testdata_transform, index=0):

    img_x, img_y, high_size = inference(img_path, model, testdata_transform)
    arr = getcoordinates(img_x, img_y, high_size)

    new_rects = None
    bounding_boxes = None

    for i in range(0, 5):
        arr = get_boundingboxes(arr)
        new_rects = lolify(arr)
        bounding_boxes = suppression(new_rects)
        arr = bounding_boxes

    bounding_boxes = arr
    disp_image = Image.open(img_path).resize((1000, 1000))
    boxes_image = Image.open(img_path).resize((1000, 1000))

    final_boxes = []

    for box in bounding_boxes:

        draw = ImageDraw.Draw(disp_image)
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline="black", width=10)
        final_boxes.append(boxes_image.crop((box[0], box[1], box[2], box[3])))
        print(box)

    save_img = Image.open(img_path).resize((1000, 1000))

    disp_image.save("new_results/results/" + str(index) + str(".jpg"))
    save_img.save("new_results/images/" + str(index) + str(".jpg"))

    dic = {"name": str(img_path[9:])}

    for i in range(0, len(bounding_boxes)):

        dic["bounding_box" + str(i)] = bounding_boxes[i]

    print(dic)

    fp = open("new_results/annotations/" + str(index) + ".yaml", "w")
    yaml.dump(dic, fp)


def main():

    model = load_checkpoint("model.pth")

    testdata_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    images_list = []
    random.shuffle(images_list)
    index = 49

    for i in os.listdir("new_images"):
        if i != ".DS_Store":
            images_list.append("new_images/" + i)
            print(i)

    for img in images_list:
        infer_boundingbox(img, model, testdata_transform, index)
        index += 1


main()
