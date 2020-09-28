from test_net.prediction import YOLODemo
import cv2
import glob
import os
from tqdm import tqdm
import numpy as np
import torch
import torchvision


def get_box(path, img_width, img_height):
    labels, boxes = [], []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(" ")
            label = int(line[0])
            if label <= 1:
                continue
            labels.append(label)
            x = float(line[1]) * img_width
            y = float(line[2]) * img_height
            width = float(line[3]) * img_width
            height = float(line[4]) * img_height

            left = x - width / 2
            top = y - height / 2
            right = x + width / 2
            bot = y + height / 2
            boxes.append(torch.tensor([[left, top, right, bot]]))
        f.close()
    if len(labels) > 1:
        return labels, torch.cat(boxes)
    elif len(labels) == 1:
        return labels, boxes[0]
    else:
        return labels, boxes


def predict_yolo_txt(path, res):
    txt_path = path.replace(".png", ".txt")
    height, width = img.shape[:2]
    with open(txt_path, "w+") as f:
        for bbox, label in zip(res.bbox, res.extra_fields['labels']):
            bbox = tuple(bbox.numpy())
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            x = bbox[0] + w / 2
            y = bbox[1] + h / 2
            f.write(
                str(label.numpy()) + " " + str(x / width) + " " + str(y / height) + " " + str(w / width) + " " + str(
                    h / height) + "\n")
        f.close()


if __name__=='__main__':
    md_yolo = YOLODemo("/media/cobot/00006784000048233/wuhanTobacco/csdarknet53-asff-mish.cfg",
                       "/media/cobot/00006784000048233/wuhanTobacco/weight_out/csdarknet53-asff-mish_last.weights",
                       show_img=False)
    imgs = glob.glob("/media/cobot/00006784000048233/wuhanTobacco/9月12号测试/all/*.png")
    out_path = "/media/cobot/00006784000048233/wuhanTobacco/9月12号测试/re1/"
    for path in tqdm(imgs):
        img = cv2.imread(path)
        try:
            res = md_yolo.predict(img)
        except:
            print(path)
            continue
        # predict_yolo_txt(path, res)

        # txt_path = path.replace(".jpg", ".txt")
        # if not os.path.exists(txt_path):
        #     continue
        # height, width = img.shape[:2]
        # # 功能一：分割出对应的目标
        # try:
        #     gt_labels, gt_boxes = get_box(txt_path, width, height)
        # except:
        #     print(txt_path)
        #     continue
        # if not len(gt_labels):
        #     continue
        # iou = torchvision.ops.box_iou(res.bbox, gt_boxes)
        # indices = torch.argmax(iou, 0)
        #
        # name = path.split("/")[-1].split(".jpg")[0]
        # for i, index in enumerate(indices):
        #     if index < res.bbox.shape[0]:
        #         box = gt_boxes[i].numpy()
        #         score = 1.0
        #     else:
        #         box = res.bbox[indices].numpy()
        #         score = res.extra_fields['scores'][indices]
        #     label = gt_labels[i]

        # #     print()
        # for bbox, index, score in zip(res.bbox, indices, res.extra_fields['scores']):
        #     box = tuple(bbox.numpy())
        #     label = gt_labels[index]
        #     if label <= 1:
        #         continue
        #
        #     w = box[2] - box[0]
        #     h = box[3] - box[1]
        #     x = box[0] + w / 2
        #     y = box[1] + h / 2
        #     w = w * 1.1
        #     h = h * 1.1
        #
        #     left = np.clip(x - w / 2, a_min=0, a_max=width).astype(np.int)
        #     right = np.clip(x + w / 2, a_min=0, a_max=width).astype(np.int)
        #     top = np.clip(y - h / 2, a_min=0, a_max=height).astype(np.int)
        #     bot = np.clip(y + h / 2, a_min=0, a_max=height).astype(np.int)
        #
        #     cig = img[top:bot, left:right]
        #
        #     dir = os.path.join(out_path, str(label))
        #     if not os.path.exists(dir):
        #         os.makedirs(dir)
        #     image_name = name + "_({}, {}, {}, {})_confident{:.4f}".format(left, right, top, bot, score) + ".png"
        #     cv2.imwrite(os.path.join(dir, image_name), cig)

        temp = img.copy()
        for bbox, label, score in zip(res.bbox, res.extra_fields['labels'], res.extra_fields['scores']):
            box = tuple(bbox.int().numpy())
            cv2.rectangle(temp, box[:2], box[2:], (0, 255, 0), 4)
            cv2.putText(temp, "%d, %.2f" % (label, score), (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

        name = path.split("/")[-1]
        name = os.path.join(out_path, name)
        cv2.imwrite(name, temp)
        cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
        cv2.imshow("Test", temp)
        cv2.waitKey(20)

