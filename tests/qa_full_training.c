#include <vector_ann.h>

static inline float func(float val1, float val2){
    if(val1>val2) return 1;
    else return 0;
}

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
    // Train input_0*input*1 -> output_0
    size_t num_samples_train_full = 10000;
    size_t num_samples_eval = 10000;

    size_t alignment = volk_get_alignment();
    float** train_input = (float**) volk_malloc(sizeof(float*)*num_samples_train_full, alignment);
    float** train_output = (float**) volk_malloc(sizeof(float*)*num_samples_train_full, alignment);
    float** eval_input = (float**) volk_malloc(sizeof(float*)*num_samples_eval, alignment);
    float** eval_output = (float**) volk_malloc(sizeof(float*)*num_samples_eval, alignment);
    float val1, val2;
    for(size_t i=0; i<num_samples_train_full; i++){
        train_input[i] = (float*) volk_malloc(sizeof(float)*net.num_nodes[0], alignment);
        train_output[i] = (float*) volk_malloc(sizeof(float)*net.num_nodes[2], alignment);

        val1 = rndm();
        val2 = rndm();
        train_input[i][0] = val1;
        train_input[i][1] = val2;
        train_output[i][0] = func(val1, val2);
    }
    for(size_t i=0; i<num_samples_eval; i++){
        eval_input[i] = (float*) volk_malloc(sizeof(float)*net.num_nodes[0], alignment);
        eval_output[i] = (float*) volk_malloc(sizeof(float)*net.num_nodes[2], alignment);

        val1 = rndm();
        val2 = rndm();
        eval_input[i][0] = val1;
        eval_input[i][1] = val2;
        eval_output[i][0] = func(val1, val2);
    }

    // Print some inputs
    size_t num_print = 10;

    for(size_t i=0; i<num_print; i++){
        net.layer_input[0][0] = eval_input[i][0];
        net.layer_input[0][1] = eval_input[i][1];
        forward_propagation(&net);
        printf("[TEST] %.2f , %.2f -> %.2f\t(%.2f error)\n", net.layer_input[0][0], net.layer_input[0][1], net.layer_output[net.num_layers-1][0], net.layer_output[net.num_layers-1][0]-eval_output[i][0]);
    }

    // Perform full training
    size_t num_cycles = 10;

    size_t num_samples_train_cycle = num_samples_train_full/num_cycles;
    train(&net, train_input, train_output, eval_input, eval_output, num_samples_train_full, num_samples_train_cycle, num_samples_eval);


    // Print some inputs
    for(size_t i=0; i<num_print; i++){
        net.layer_input[0][0] = eval_input[i][0];
        net.layer_input[0][1] = eval_input[i][1];
        forward_propagation(&net);
        printf("[TEST] %.2f , %.2f -> %.2f\t(%.2f error)\n", net.layer_input[0][0], net.layer_input[0][1], net.layer_output[net.num_layers-1][0], net.layer_output[net.num_layers-1][0]-eval_output[i][0]);
    }

    return 0;
}
