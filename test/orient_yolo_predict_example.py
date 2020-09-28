from test_net.prediction import OrientYOLODemo
import cv2
import glob
import os
from tqdm import tqdm
import numpy as np
import time


if __name__=='__main__':
    md_yolo = OrientYOLODemo("/media/cobot/0000678400004823/orderrecheck/csdarknet53-orient.cfg",
                             "/media/cobot/0000678400004823/orderrecheck/csdarknet53-orient_last.weights")
    imgs = glob.glob("/media/cobot/0000678400004823/orderrecheck/json_seg/*.png")
    out_path = "/media/cobot/0000678400004823/orderrecheck/minRect/"
    for path in tqdm(imgs):
        img = cv2.imread(path)
        try:
            t0 = time.time()
            res = md_yolo.predict(img)
            t1 = time.time()
            print((t1 - t0) * 1000)
        except:
            print(path)
            continue

        if res.bbox.shape[0] ==0:
            print(path)

        temp = img.copy()
        for bbox, orient, label, score in zip(res.bbox, res.extra_fields["orient"], res.extra_fields['labels'], res.extra_fields['scores']):
            box = tuple(bbox.int().numpy())
            orient = tuple(orient.float().numpy())
            width = box[2] - box[0]
            height = box[3] - box[1]
            orient_points = np.array([[box[0] + orient[0] * width, box[1]],
                                      [box[2], box[1] + orient[1] * height],
                                      [box[2] - orient[2] * width, box[3]],
                                      [box[0], box[3] - orient[3] * height]])
            orient_points = orient_points.reshape((-1, 1, 2))

            rect = cv2.minAreaRect(np.array(orient_points, np.float32))
            obox = cv2.boxPoints(rect)
            cv2.drawContours(temp, [obox.astype(np.int0)], 0, (255, 255, 0), 3)

            cv2.polylines(temp, [orient_points.astype(np.int32)], True, (0, 0, 255), 3)

            cv2.rectangle(temp, box[:2], box[2:], (0, 255, 0), 4)
            cv2.putText(temp, "%d, %.2f, %.2f" % (label, score, orient[4]), (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

        name = path.split("/")[-1]
        name = os.path.join(out_path, name)
        # cv2.imwrite(name, temp)

        cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
        cv2.imshow("Test", temp)
        cv2.waitKey(20)

