#include <vector_ann.h>

void init_weights(struct ann *net, uint32_t seed){
    const size_t num_layers = net->num_layers;
    const size_t* num_nodes = net->num_nodes;

    // Set seed
    if(seed==0) srand(time(NULL));
    else srand(seed);

    // Allocate memory for weights
    size_t alignment = volk_get_alignment();
    net->weights = (float***) volk_malloc(sizeof(float*)*(num_layers-1), alignment);

    for(size_t i=0; i<num_layers-1; i++){ // go through layers
        net->weights[i] = (float**) volk_malloc(sizeof(float*)*num_nodes[i], alignment);
        for(size_t j=0; j<num_nodes[i]; j++){ // go through this layer and allocate next layer
            net->weights[i][j] = (float*) volk_malloc(sizeof(float)*num_nodes[i+1], alignment);
        }
    }

    // Init weights with random values
    for(size_t i=0; i<num_layers-1; i++){ // go through layers
        for(size_t j=0; j<num_nodes[i]; j++){ // go through this layer
            for(size_t k=0; k<num_nodes[i+1]; k++){ // go through next layer
                net->weights[i][j][k] = rndm();
            }
        }
    }
}

void init_io(struct ann* net){
    const size_t num_layers = net->num_layers;
    const size_t* num_nodes = net->num_nodes;

    // Allocate memory for input and output values
    size_t alignment = volk_get_alignment();
    net->layer_input = (float**) volk_malloc(sizeof(float*)*num_layers, alignment);
    net->layer_output = (float**) volk_malloc(sizeof(float*)*num_layers, alignment);
    net->layer_error = (float**) volk_malloc(sizeof(float*)*num_layers, alignment);

    for(size_t i=0; i<num_layers; i++){
        net->layer_input[i] = (float*) volk_malloc(sizeof(float)*num_nodes[i], alignment);
        net->layer_output[i] = (float*) volk_malloc(sizeof(float)*num_nodes[i], alignment);
        net->layer_error[i] = (float*) volk_malloc(sizeof(float)*num_nodes[i], alignment);
    }
}

void forward_propagation(struct ann* net){
    const size_t num_layers = net->num_layers;
    const size_t* num_nodes = net->num_nodes;
    const float beta = net->beta;
    size_t i, j, k;

    // Calculate output of input layer
    for(i=0; i<num_nodes[0]; i++){
        net->layer_output[0][i] = activ_func(net->layer_input[0][i], beta);
    }

    // Propagate forward through rest of the network
    float sum;
    for(i=1; i<num_layers; i++){ // go through layers
        for(j=0; j<num_nodes[i]; j++){ // go through nodes in this layer
            sum = 0;
            for(k=0; k<num_nodes[i-1]; k++){ // go through nodes in previous layer and sum input for this node
                sum += net->layer_output[i-1][k]*net->weights[i-1][k][j];
            }
            net->layer_input[i][j] = sum;
            net->layer_output[i][j] = activ_func(sum, beta);
        }
    }
}

void backward_propagation(struct ann* net, float* known_output){
    const size_t num_layers = net->num_layers;
    const size_t* num_nodes = net->num_nodes;
    const float beta = net->beta;
    int i;
    size_t j, k;

    // Propagate from output layer to previous layer
    for(i=0; i<num_nodes[num_layers-1]; i++){
        net->layer_error[num_layers-1][i] = activ_func_deriv(net->layer_input[num_layers-1][i], net->beta)*(known_output[i]-net->layer_output[num_layers-1][i]);
    }

    // Propagate backward through rest of the network
    float sum;
    for(i=num_layers-2; i>=0; i--){ // go through layers
        // Get error for this layer
        for(j=0; j<num_nodes[i]; j++){ // go through nodes in this layer
            sum = 0;
            for(k=0; k<num_nodes[i+1]; k++){
                sum += net->layer_error[i+1][k]*net->weights[i][j][k];
            }
            net->layer_error[i][j] = activ_func_deriv(net->layer_input[i][j], net->beta)*sum;
        }

        // Modify weights for this layer
        for(j=0; j<num_nodes[i]; j++){
            for(k=0; k<num_nodes[i+1]; k++){
                net->weights[i][j][k] += net->learn_rate*net->layer_output[i][j]*net->layer_error[i+1][k];
            }
        }
    }
}

void training_cycle(struct ann* net, float** known_input, float** known_output, size_t num_samples){
    const size_t num_layers = net->num_layers;
    const size_t* num_nodes = net->num_nodes;
    size_t i, j;

    for(i=0; i<num_samples; i++){
        for(j=0; j<num_nodes[0]; j++) net->layer_input[0][j] = known_input[i][j];
        forward_propagation(net);
        backward_propagation(net, known_output[i]);
    }
}

float sample_error(struct ann* net, float** known_input, float** known_output, size_t num_samples){
    const size_t num_layers = net->num_layers;
    const size_t* num_nodes = net->num_nodes;
    size_t i, j;
    float sum, error;

    error = 0;
    for(i=0; i<num_samples; i++){
        sum = 0;
        for(j=0; j<num_nodes[0]; j++) net->layer_input[0][j] = known_input[i][j];
        forward_propagation(net);
        for(j=0; j<num_nodes[num_layers-1]; j++) sum += powf(known_output[i][j]-net->layer_output[num_layers-1][j], 2.0);
        error += sum;
    }

    return error/(float)num_samples;
}


void train(struct ann* net, float** train_input, float** train_output, float** eval_input, float** eval_output, size_t num_samples_train_full, size_t num_samples_train_cycle, size_t num_samples_eval){
    size_t i;
    float error_initial, error_final;
    float** train_input_cycle = train_input;
    float** train_output_cycle = train_output;

    for(i=0; i<num_samples_train_full/num_samples_train_cycle; i++){
        error_initial = sample_error(net, eval_input, eval_output, num_samples_eval);

        training_cycle(net, train_input_cycle, train_output_cycle, num_samples_train_cycle);
        train_input_cycle += num_samples_train_cycle;
        train_output_cycle += num_samples_train_cycle;

        error_final = sample_error(net, eval_input, eval_output, num_samples_eval);
        printf("[TRAIN][CYCLE %lu]\tSample error:\t%f\t->\t%f\t(%f)\n", i, error_initial, error_final, error_final-error_initial);
    }
}
