from test_net.prediction import YOLODemo
import cv2
import glob
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.jit
import torchvision


def get_rois(yolo, image, expand=1.0):
    result = yolo.predict(image)

    height, width = image.shape[:2]
    boxes, rois, labels, scores = [], [], [], []
    for bbox, label, score in zip(result.bbox, result.extra_fields['labels'], result.extra_fields['scores']):
        box = tuple(bbox.int().numpy())
        boxes.append(box)

        w = box[2] - box[0]
        h = box[3] - box[1]
        x = box[0] + w / 2
        y = box[1] + h / 2
        w = w * expand
        h = h * expand

        left = np.clip(x - w / 2, a_min=0, a_max=width).astype(np.int)
        right = np.clip(x + w / 2, a_min=0, a_max=width).astype(np.int)
        top = np.clip(y - h / 2, a_min=0, a_max=height).astype(np.int)
        bot = np.clip(y + h / 2, a_min=0, a_max=height).astype(np.int)

        roi = image[top:bot, left:right]
        rois.append(roi)
        labels.append(label.numpy())
        scores.append(score.numpy())
    return boxes, rois, labels, scores


def longest_resize(image, max_size):
    height, width = image.shape[:2]
    scale = max_size / float(max(width, height))
    return cv2.resize(image, (0,0), fx=scale, fy=scale)


def filling(image, size):
    h, w = image.shape[:2]
    size = max(size, h, w)
    if len(image.shape) == 2:
        out = np.zeros((size, size), np.uint8)
    elif len(image.shape) == 3:
        out = np.zeros((size, size, 3), np.uint8)
    else:
        return image
    y1 = (size - h) // 2
    y2 = y1 + h
    x1 = (size - w) // 2
    x2 = x1 + w

    out[y1:y2, x1:x2] = image
    return out.copy()


def center_crop(image, crop_height, crop_width):
    height, width = image[:2]
    crop_height = min(height, crop_height)
    crop_width = min(width, crop_width)
    y1 = (height - crop_height) // 2
    y2 = y1 + crop_height
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width

    return image[y1:y2, x1:x2]


def get_metric_inputs(images, size):
    c = images[0].shape[2]
    inputs = torch.zeros(1, c, size[1], size[0])
    for image in images:
        image = longest_resize(image, size[0])
        image = filling(image, size[0])
        input = torchvision.transforms.functional.to_tensor(image)
        inputs = torch.cat((inputs, input.unsqueeze(0)), 0)
    return inputs[1:]


def get_template_embeddings(path, device=torch.device('cpu')):
    cc = torch.load(path, map_location=device)
    embeddings = torch.zeros((1, model.embedding_num), device=device)
    names = []
    for key, value in cc.items():
        names.append(key.split("_")[0])
        embeddings = torch.cat((embeddings, value.unsqueeze(0)), 0)
    embeddings = embeddings[1:]
    return embeddings, names


if __name__=='__main__':
    md_metric = torch.jit.load("/media/cobot/00006784000048232/tobacco/0729test/metric_resnest50_102_0708.pt")
    md_metric.embedding_num = 256
    device = torch.device('cuda:0')
    md_metric.eval()
    model = md_metric.to(device)

    embedding_file = "/media/cobot/00006784000048232/tobacco/0729test/metric_resnest50_102_0708_features_template.pth"
    template_embeddings, embedding_names = get_template_embeddings(embedding_file, device)

    md_yolo = YOLODemo("/media/cobot/4e4a7518-760a-46bb-a065-cbdcce966213/changshaTobacco/yolo/side/csdarknet53-asff-mish.cfg",
                       "/media/cobot/4e4a7518-760a-46bb-a065-cbdcce966213/changshaTobacco/yolo/side/weight_out/csdarknet53-asff-mish_last.weights",
                       show_img=False)
    imgs = glob.glob("/media/cobot/00006784000048232/tobacco/0729test/AS/*.png")
    out_path = "/media/cobot/00006784000048232/tobacco/0729test/result/"
    for path in tqdm(imgs):
        img = cv2.imread(path)
        height, width = img.shape[:2]
        try:
            res = md_yolo.predict(img)
        except:
            print(path)
            continue

        boxes, rois, labels, scores = get_rois(md_yolo, img, expand=1.0)
        temp = img.copy()
        names = []
        if len(rois) > 0:
            inputs = get_metric_inputs(rois, (320, 320))
            with torch.no_grad():
                embeddings = model(inputs.to(device))
                simi = torch.matmul(embeddings, torch.t(template_embeddings))
                max_simi, hard_index = simi.max(1)
                max_simi = max_simi.cpu().numpy()
                names = [embedding_names[i] for i in hard_index.cpu().numpy()]

            for box, label, score in zip(boxes, names, max_simi):
                cv2.rectangle(temp, box[:2], box[2:], (0, 255, 0), 4)
                cv2.putText(temp, "%s, %.2f" % (label, score), (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        txt_path = path.replace(".png", ".txt")
        with open(txt_path, "w+") as f:
            for bbox, label in zip(boxes, names):
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                x = bbox[0] + w / 2
                y = bbox[1] + h / 2
                f.write(label + " " + str(x / width) + " " + str(y / height) + " " + str(w / width) + " " + str(h / height) + "\n")
            f.close()

        name = path.split("/")[-1]
        name = os.path.join(out_path, name)
        cv2.imwrite(name, temp)
        cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
        cv2.imshow("Test", temp)
        cv2.waitKey(20)
