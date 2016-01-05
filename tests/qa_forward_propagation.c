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

    // Compare output to expected values
    float in_0 = activ_func(
                     net.layer_input[0][0],
                     net.beta);
    float in_1 = activ_func(
                     net.layer_input[0][1],
                     net.beta);
    float mid_0 = activ_func(
                     in_0*net.weights[0][0][0] +
                     in_1*net.weights[0][1][0],
                     net.beta);
    float mid_1 = activ_func(
                     in_0*net.weights[0][0][1] +
                     in_1*net.weights[0][1][1],
                     net.beta);
    float out_0 = activ_func(
                     mid_0*net.weights[1][0][0] +
                     mid_1*net.weights[1][1][0],
                     net.beta);
    float out_1 = activ_func(
                     mid_0*net.weights[1][0][1] +
                     mid_1*net.weights[1][1][1],
                     net.beta);

    if(out_0!=net.layer_output[2][0]) return 1;
    if(out_1!=net.layer_output[2][1]) return 2;

    return 0;
}
