#include <vector_ann.h>

int main(){
    // Init neural network struct
    struct ann net;
    net.num_layers = 3;
    net.num_nodes = (size_t*) malloc(sizeof(size_t)*net.num_layers);
    net.num_nodes[0] = 2;
    net.num_nodes[1] = 3;
    net.num_nodes[2] = 1;

    // Init weights
    uint32_t seed = 1234;
    init_weights(&net, seed);

    // Setup
    init_io(&net);
    net.beta = 1.0;
    net.learn_rate = 1.0;

    // Get training samples
    // Train sum(input)>0 -> 1, sum(input)<0 -> 0
    size_t num_samples = 10000;
    size_t alignment = volk_get_alignment();
    float** known_input = (float**) volk_malloc(sizeof(float*)*num_samples, alignment);
    float** known_output = (float**) volk_malloc(sizeof(float*)*num_samples, alignment);
    float val1, val2;
    for(size_t i=0; i<num_samples; i++){
        known_input[i] = (float*) volk_malloc(sizeof(float)*net.num_nodes[0], alignment);
        known_output[i] = (float*) volk_malloc(sizeof(float)*net.num_nodes[2], alignment);

        val1 = rndm();
        val2 = rndm();
        known_input[i][0] = val1;
        known_input[i][1] = val2;
        if(val1+val2>0) known_output[i][0] = 1;
        else known_output[i][0] = 0;
    }

    // Try some input values
    size_t num_test = 5;
    for(size_t i=0; i<num_test; i++){
        net.layer_input[0][0] = known_input[i][0];
        net.layer_input[0][1] = known_input[i][1];
        forward_propagation(&net);
        printf("[BEFORE]\t%i:\t%f\t->\t%f\n", i, net.layer_input[0][1]+net.layer_input[0][1], net.layer_output[2][0]);
    }

    // Run one training cycle
    printf("[TRAINING]\n");
    training_cycle(&net, known_input, known_output, num_samples);

    // Try some input values again
    for(size_t i=0; i<num_test; i++){
        net.layer_input[0][0] = known_input[i][0];
        net.layer_input[0][1] = known_input[i][1];
        forward_propagation(&net);
        printf("[AFTER]\t%i:\t%f\t->\t%f\n", i, net.layer_input[0][1]+net.layer_input[0][1], net.layer_output[2][0]);
    }

    return 0;
}
