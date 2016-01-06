#include <vector_ann.h>

float my_rndm(){
    return rndm();
}

static inline float func(float* val, size_t num_val){
    // random function with output in [0,1]
    return 0.3*val[0] + 0.2*val[1];
}

int main(){
    // Init neural network struct
    struct ann net;
    net.num_layers = 4;
    net.num_nodes = (size_t*) malloc(sizeof(size_t)*net.num_layers);

    net.num_nodes[0] = 2;
    for(size_t i=1; i<net.num_layers-1; i++) net.num_nodes[i] = 5;
    net.num_nodes[net.num_layers-1] = 1;

    // Init weights
    uint32_t seed = 0;
    init_weights(&net, seed);

    // Setup
    init_io(&net);
    net.beta = 1.0;
    net.learn_rate = 1.0;

    // Get training samples
    // See func(...) for training scenario
    size_t num_samples_train_full = 100000;
    size_t num_samples_eval = 10000;

    size_t alignment = volk_get_alignment();
    float** train_input = (float**) volk_malloc(sizeof(float*)*num_samples_train_full, alignment);
    float** train_output = (float**) volk_malloc(sizeof(float*)*num_samples_train_full, alignment);
    float** eval_input = (float**) volk_malloc(sizeof(float*)*num_samples_eval, alignment);
    float** eval_output = (float**) volk_malloc(sizeof(float*)*num_samples_eval, alignment);

    for(size_t i=0; i<num_samples_train_full; i++){
        train_input[i] = (float*) volk_malloc(sizeof(float)*net.num_nodes[0], alignment);
        train_output[i] = (float*) volk_malloc(sizeof(float)*net.num_nodes[2], alignment);

        for(size_t j=0; j<net.num_nodes[0]; j++) train_input[i][j] = my_rndm();
        train_output[i][0] = func(train_input[i], net.num_nodes[0]);
    }

    for(size_t i=0; i<num_samples_eval; i++){
        eval_input[i] = (float*) volk_malloc(sizeof(float)*net.num_nodes[0], alignment);
        eval_output[i] = (float*) volk_malloc(sizeof(float)*net.num_nodes[2], alignment);

        for(size_t j=0; j<net.num_nodes[0]; j++) eval_input[i][j] = my_rndm();
        eval_output[i][0] = func(eval_input[i], net.num_nodes[0]);
    }

    // Perform full training
    size_t num_samples_train_cycle = 10000;

    train(&net, train_input, train_output, eval_input, eval_output, num_samples_train_full, num_samples_train_cycle, num_samples_eval);


    // Print some inputs
    for(size_t i=0; i<10; i++){
        net.layer_input[0][0] = i/10.0;
        net.layer_input[0][1] = i/20.0;
        forward_propagation(&net);
        printf("[TEST %lu] in\t->\t(network output)\t(true output)\n\t\t->\t%.2f\t\t\t%.2f\n", i, net.layer_output[net.num_layers-1][0], func(net.layer_input[0], 2));
    }

    return 0;
}
