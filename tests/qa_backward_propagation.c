#include <vector_ann.h>

int main(){
    // Init neural network struct
    struct ann net;
    net.num_layers = 3;
    net.num_nodes = (size_t*) malloc(sizeof(size_t)*net.num_layers);
    net.num_nodes[0] = 2;
    net.num_nodes[1] = 2;
    net.num_nodes[2] = 2;

    // Init weights
    uint32_t seed = 1234;
    init_weights(&net, seed);

    // Init input and output arrays
    init_io(&net);

    // Set input values
    net.layer_input[0][0] = 0.3;
    net.layer_input[0][1] = -0.7;

    // Set beta value for activation function
    net.beta = 1.0;

    // Do forward propagation
    forward_propagation(&net);

    // Set expected output
    size_t alignment = volk_get_alignment();
    float* expected_output = (float*) volk_malloc(sizeof(float)*net.num_nodes[2], alignment);
    expected_output[0] = 1.0;
    expected_output[1] = -1.0;

    // Set learn rate
    net.learn_rate = 1.0;

    // Do backward propagation
    backward_propagation(&net, expected_output);

    // Compare change of weights to expected values
    // TODO

    // Clean-up
    volk_free(expected_output);

    return 0;
}
