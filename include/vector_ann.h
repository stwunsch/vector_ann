#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <inttypes.h>
#include <volk/volk.h>

/*************************
 * ANN definition struct *
 *************************/
struct ann{
    // Number of layers
    size_t    num_layers;

    // Pointer to array with number of nodes per layer
    size_t*   num_nodes;

    // Pointer to weights matrix with indices [layer][neuron this layer][neuron next layer]
    float***  weights;
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
void init_weights(struct ann*);
