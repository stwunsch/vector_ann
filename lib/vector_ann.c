#include <vector_ann.h>

void init_weights(struct ann *net){
    const size_t num_layers = net->num_layers;
    const size_t* num_nodes = net->num_nodes;

    // Allocate memory for weights
    size_t alignment = volk_get_alignment();
    net->weights = (float***) volk_malloc(sizeof(float*)*(num_layers-1), alignment);

    for(size_t i=0; i<num_layers-1; i++){ // go through layers
        net->weights[i] = (float**) volk_malloc(sizeof(float*)*num_nodes[i], alignment);
        for(size_t j=0; j<num_nodes[i]; j++){ // go through this layer and allocate next layer
            net->weights[i][j] = (float*) volk_malloc(sizeof(float)*num_nodes[i+1], alignment);
        }
    }

    // Init weights with random values
    for(size_t i=0; i<num_layers-1; i++){ // go through layers
        for(size_t j=0; j<num_nodes[i]; j++){ // go through this layer
            for(size_t k=0; k<num_nodes[i+1]; k++){ // go through next layer
                net->weights[i][j][k] = rndm();
            }
        }
    }
}
