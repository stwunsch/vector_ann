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
    init_weights(&net);

    // Test ends successfully
    return 0;
}
