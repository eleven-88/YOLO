//
// Created by cobot on 20-5-15.
//

#ifndef DARKNET_ORIENTED_YOLO_LAYER_H
#define DARKNET_ORIENTED_YOLO_LAYER_H

//#include "darknet.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_oriented_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes);
void forward_oriented_yolo_layer(const layer l, network_state state);
void backward_oriented_yolo_layer(const layer l, network_state state);
void resize_oriented_yolo_layer(layer *l, int w, int h);
// predict: not implementied
int oriented_yolo_num_detections(layer l, float thresh);
int oriented_yolo_num_detections_batch(layer l, float thresh, int batch);
int get_oriented_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter);
int get_oriented_yolo_detections_batch(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter, int batch);
void correct_oriented_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter);

#ifdef GPU
void forward_oriented_yolo_layer_gpu(const layer l, network_state state);
void backward_oriented_yolo_layer_gpu(layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif

#endif //DARKNET_ORIENTED_YOLO_LAYER_H
