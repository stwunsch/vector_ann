#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <volk/volk.h>

#define VECTOR_ANN_EULER 2.71828182845904523536

/*************************
 * ANN definition struct *
 *************************/
struct ann{
    // Number of layers
    size_t    num_layers;

    // Pointer to array with number of nodes per layer
    size_t*   num_nodes;

    // Pointer to weights matrix with indices [layer][node this layer][node next layer]
    float***  weights;

    // Beta value for activation function
    float beta;

    // Pointer to input values of each layer (sum of all node inputs of the previous layer, so-called net input), indices are [layer][node]
    float**   layer_input;

    // Pointer to output values of each layer (input_value of the node processed with activation function and always-on offset), indices are [layer][node]
    float**   layer_output;

    // Pointer to error values of each layer, input layer has no error and is an empty array but is kept cause of layer indices, indices are [layer][node]
    float**   layer_error;

    // Learn rate
    float learn_rate;
};

/*******************************
 * Get random number in [-1,1] *
 *******************************/
float rndm(){
    float val = 2.0*rand()/(float)RAND_MAX - 1.0;
    return val;
}

/***************************************************************************
 * Allocate memory for weights and init them with random numbers in [-1,1] *
 ***************************************************************************/
void init_weights(struct ann* net, uint32_t seed);

/*****************************************************
 * Allocate memory of input, output and error values *
 *****************************************************/
void init_io(struct ann* net);

/**************************************
 * Activation function and derivation *
 **************************************/
static inline float activ_func(float in, float beta){
    return 1.0/(1.0+powf(VECTOR_ANN_EULER, -beta*in));
}

static inline float activ_func_deriv(float in, float beta){
    return beta*powf(VECTOR_ANN_EULER, -beta*in)/powf(1.0+powf(VECTOR_ANN_EULER, -beta*in),2.0);
}

/**************************************
 * Foward propagation of input values *
 **************************************/
void forward_propagation(struct ann* net);

/****************************************************************************
 * Backward propagation of errors on output values from forward propagation *
 ****************************************************************************/
void backward_propagation(struct ann* net, float* known_output);

/**********************************************************
 * Perform one training cycle with given training samples *
 **********************************************************/
void training_cycle(struct ann* net, float** known_input, float** known_output, size_t num_samples);

/*****************************************************
 * Calculate mean error of network output with given *
 * training samples to estimate training progress    *
 *****************************************************/
void sample_error(struct ann* net, float** known_input, float** known_output, size_t num_samples);
