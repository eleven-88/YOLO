import json
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    cats = {"a":0, "c":1}
    print(cats)
    path = "/media/cobot/0000678400004823/orderrecheck/part2"
    out_path = "/media/cobot/0000678400004823/orderrecheck/orient"
    files = glob.glob(os.path.join(path, "*.json"))
    for file in tqdm(files):
        with open(file, "rb") as f:
            data = json.load(f)
            image_path = data["imagePath"]
            image_width = data["imageWidth"]
            image_height = data["imageHeight"]
            shapes = data["shapes"]
            image = cv2.imread(os.path.join(path, image_path))
            mask = np.zeros((image_height, image_width), np.uint8)

            txt_path = file.split("/")[-1]
            txt_path = os.path.join(out_path, txt_path)
            txt_path = txt_path.replace(".json", ".txt")
            with open(txt_path, "w+") as txt:
                for shape in shapes:
                    label = shape["label"]
                    if label not in cats.keys():
                        continue
                    label = cats[label]
                    points = shape["points"]
                    points = np.array(points, np.float32)

                    left = np.min(points[:, 0])
                    right = np.max(points[:, 0])
                    top = np.min(points[:, 1])
                    bot = np.max(points[:, 1])

                    width = right - left
                    height = bot - top
                    x = left + width / 2
                    y = top + height / 2

                    area_hb = width * height
                    sub_mask = np.zeros((image_height, image_width), np.uint8)
                    sub_mask = cv2.fillConvexPoly(sub_mask, points.astype(np.int), 255)
                    a = cv2.countNonZero(sub_mask)
                    r = a / area_hb
                    mask = cv2.fillConvexPoly(mask, np.array(points, np.int), 255)

                    a1 = np.argmin(points[:, 1])
                    a1 = (points[a1, 0] - left) / width
                    a2 = np.argmax(points[:, 0])
                    a2 = (points[a2, 1] - top) / height
                    a3 = np.argmax(points[:, 1])
                    a3 = (right - points[a3, 0]) / width
                    a4 = np.argmin(points[:, 0])
                    a4 = (bot - points[a4, 1]) / height

                    if a1 == 1.0 or a2 == 1.0 or a3 == 1.0 or a4 == 1.0 or \
                       a1 == 0.0 or a2 == 0.0 or a3 == 0.0 or a4 == 0.0:
                        rect = cv2.minAreaRect(points)
                        obox = cv2.boxPoints(rect)
                        a1 = np.argmin(obox[:, 1])
                        a1 = (obox[a1, 0] - left) / width
                        a2 = np.argmax(obox[:, 0])
                        a2 = (obox[a2, 1] - top) / height
                        a3 = np.argmax(obox[:, 1])
                        a3 = (right - obox[a3, 0]) / width
                        a4 = np.argmin(obox[:, 0])
                        a4 = (bot - obox[a4, 1]) / height
                        r = rect[1][0] * rect[1][1] / (width * height)
                        r = np.clip(r, 0.0, 1.0)

                        if rect[2] >= -0.0001:
                            a1 = 1.0
                            a2 = 1.0
                            a3 = 1.0
                            a4 = 1.0
                            r = 1.0
                        elif rect[2] <= -89.9999:
                            a1 = 0.0
                            a2 = 0.0
                            a3 = 0.0
                            a4 = 0.0
                            r = 1.0

                    if r >= 1:
                        r = 1.0
                        a1 = 1.0
                        a2 = 1.0
                        a3 = 1.0
                        a4 = 1.0
                    a1 = np.clip(a1, 0.0, 1.0)
                    a2 = np.clip(a2, 0.0, 1.0)
                    a3 = np.clip(a3, 0.0, 1.0)
                    a4 = np.clip(a4, 0.0, 1.0)
                    r = np.clip(r, 0.0, 1.0)
                    print(image_path, a1, a2, a3, a4, r)
                    txt.write(str(label) + " " + str(x/image_width) + " " + str(y/image_height) + " " + str(width/image_width) + " " + str(height/image_height) + " "
                              + str(a1) + " " + str(a2) + " " + str(a3) + " " + str(a4) + " " + str(r) + "\n")

                    hbox = cv2.boundingRect(points)
                    cv2.polylines(image, [points.astype(np.int)], True, (0, 0, 255), 2)
                    cv2.rectangle(image, (hbox[0], hbox[1]), (hbox[0] + hbox[2], hbox[1] + hbox[3]), (0, 255, 0), 2)
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
            cv2.imshow("image", image)
            cv2.imshow("mask", mask)
            cv2.waitKey(20)