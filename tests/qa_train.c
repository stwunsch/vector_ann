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

    return 0;
}
