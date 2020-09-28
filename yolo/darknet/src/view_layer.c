#include "view_layer.h"
#include "dark_cuda.h"
#include "blas.h"
#include "utils.h"
#include <stdio.h>


layer make_view_layer(int batch, int w, int h, int c, int out_w, int out_h, int out_c)
{
    layer l = { (LAYER_TYPE)0 };
    l.type = VIEW;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = out_w;
    l.out_h = out_h;
    l.out_c = out_c;
    assert(h*w*c==out_w*out_h*out_c);
    fprintf(stderr, "view                    /%4d x%4d x%4d -> %4d x%4d x%4d\n", w, h, c, l.out_w, l.out_h, l.out_c);
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.output = (float*)xcalloc(output_size, sizeof(float));
    l.delta = (float*)xcalloc(output_size, sizeof(float));

    l.forward = forward_view_layer;
    l.backward = backward_view_layer;
#ifdef GPU
    l.forward_gpu = forward_view_layer_gpu;
    l.backward_gpu = backward_view_layer_gpu;

    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
#endif
    return l;
}

void resize_view_layer(layer *l, int w, int h)
{
    int c = l->c;
    l->h = h;
    l->w = w;

    l->inputs = h*w*c;
    l->outputs = l->inputs;
    int output_size = l->outputs * l->batch;

    l->output = (float*)xrealloc(l->output, output_size * sizeof(float));
    l->delta = (float*)xrealloc(l->delta, output_size * sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
#endif
}

void forward_view_layer(const layer l, network_state state)
{
    view_cpu(state.input, l.out_w, l.out_h, l.out_c, l.batch, l.output);
}

void backward_view_layer(const layer l, network_state state)
{
    view_cpu(l.delta, l.out_w, l.out_h, l.out_c, l.batch, state.delta);
}

#ifdef GPU
void forward_view_layer_gpu(layer l, network_state state)
{
    view_ongpu(state.input, l.out_w, l.out_h, l.out_c, l.batch, l.output_gpu);
}

void backward_view_layer_gpu(layer l, network_state state)
{
    view_ongpu(l.delta_gpu, l.out_w, l.out_h, l.out_c, l.batch, state.delta);
}
#endif


void view_cpu(float *x, int w, int h, int c, int batch, float *out)
{
    int b,i,j,k;

    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int in_index  = i + w*(j + h*(k + c*b));

                    out[in_index] = x[in_index];
                }
            }
        }
    }
}
