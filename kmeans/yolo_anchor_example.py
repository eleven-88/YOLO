from kmeans.kmeans import YOLO_Kmeans
from kmeans.combine_bbox import parse_from_config
# yolo anchor box 聚类, 需要指定train_cfg文件和 anchor 个数

if __name__=='__main__':
    train_yolo_cfg = "/media/cobot/00006784000048231/yolov4/train/train_yolo.cfg"
    pfl = parse_from_config(train_yolo_cfg)
    cluster_number = 20
    kmeans = YOLO_Kmeans(cluster_number, pfl)
    kmeans.txt2clusters()
