#ifndef VIEW_LAYER_H
#define VIEW_LAYER_H

#include "image.h"
#include "dark_cuda.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_view_layer(int batch, int w, int h, int c, int out_w, int out_h, int out_c);
void resize_view_layer(layer *l, int w, int h);
void forward_view_layer(const layer l, network_state state);
void backward_view_layer(const layer l, network_state state);

void view_cpu(float *x, int w, int h, int c, int batch, float *out);

#ifdef GPU
void forward_view_layer_gpu(layer l, network_state state);
void backward_view_layer_gpu(layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif

#endif
