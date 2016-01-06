#include <vector_ann.h>

int main(){
    // Init neural network struct
    struct ann net;
    net.num_layers = 3;
    net.num_nodes = (size_t*) malloc(sizeof(size_t)*net.num_layers);
    net.num_nodes[0] = 2;
    net.num_nodes[1] = 5;
    net.num_nodes[2] = 1;

    // Init weights
    uint32_t seed = 0;
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

    // Get mean output error on sample
    printf("[SAMPLE ERROR] Value: %f\n", sample_error(&net, known_input, known_output, num_samples));

    // Run one training cycle
    printf("[TRAINING]\n");
    training_cycle(&net, known_input, known_output, num_samples);

    // Get output error again
    printf("[SAMPLE ERROR] Value: %f\n", sample_error(&net, known_input, known_output, num_samples));

    return 0;
}
