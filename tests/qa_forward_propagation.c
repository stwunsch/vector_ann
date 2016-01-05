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

    // Set weights manually
    for(size_t i=0; i<net.num_layers-1; i++){
        for(size_t j=0; j<net.num_nodes[i]; j++){
            for(size_t k=0; k<net.num_nodes[i+1]; k++){
                net.weights[i][j][k] = 0.5;
            }
        }
    }

    // Init input and output arrays
    init_io(&net);

    // Set input values
    net.layer_input[0][0] = 0.1;
    net.layer_input[0][1] = 0.3;

    // Do forward propagation
    printf("Input: %f %f\n", net.layer_input[0][0], net.layer_input[0][1]);
    forward_propagation(&net);
    printf("Output: %f %f\n", net.layer_output[0][0], net.layer_output[0][1]);

    // Compare output to expected values

    return 0;
}
