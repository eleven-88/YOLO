#include "activation_layer.h"
#include "utils.h"
#include "dark_cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    layer l = { (LAYER_TYPE)0 };
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    l.output = (float*)xcalloc(batch * inputs, sizeof(float));
    l.delta = (float*)xcalloc(batch * inputs, sizeof(float));

    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
#ifdef GPU
    l.forward_gpu = forward_activation_layer_gpu;
    l.backward_gpu = backward_activation_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
    l.activation = activation;
    fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
    return l;
}

void forward_activation_layer(layer l, network_state state)
{
    copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_activation_layer(layer l, network_state state)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    copy_cpu(l.outputs*l.batch, l.delta, 1, state.delta, 1);
}

#ifdef GPU

void forward_activation_layer_gpu(layer l, network_state state)
{
    copy_ongpu(l.outputs*l.batch, state.input, 1, l.output_gpu, 1);
    if (l.activation == NORM_CHAN_SOFTMAX || l.activation == NORM_CHAN_SOFTMAX_MAXVAL)
        activate_array_normalize_channels_softmax_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output_gpu, 1);
    else
        activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_activation_layer_gpu(layer l, network_state state)
{
    if (l.activation == NORM_CHAN_SOFTMAX || l.activation == NORM_CHAN_SOFTMAX_MAXVAL)
        gradient_array_normalize_channels_softmax_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);
    else
        gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    copy_ongpu(l.outputs*l.batch, l.delta_gpu, 1, state.delta, 1);
}
#endif
