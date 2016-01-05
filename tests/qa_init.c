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
    init_weights(&net, 0);

    // Check for values out of [-1,1] in weights
    float*** weights = net.weights;
    for(size_t i=0; i<net.num_layers-1; i++){
        for(size_t j=0; j<net.num_nodes[i]; j++){
            for(size_t k=0; k<net.num_nodes[i+1]; k++){
                if(weights[i][j][k]>1.0 || weights[i][j][k] < -1.0){
                    fprintf(stderr, "Init weights failed.\n");
                    return 1;
                }
            }
        }
    }

    return 0;
}
