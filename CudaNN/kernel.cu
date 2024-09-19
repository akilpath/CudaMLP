
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Tensor2D.cuh"
#include "VectorND.cuh"
#include "FCLayer.hpp"
#include "ReLu.cuh"
#include "MSE.cuh"
#include <iostream>
#include <math.h>

 //function to add the elements of two arrays
__global__ void add(int n, float* x, float* y)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    while (index < n) {
        y[index] = x[index] + y[index];
        index += stride;
    }
}

int main(void)
{
	double learning_rate = 0.03;
	FCLayer input_layer = FCLayer(2, 3, learning_rate);
    ReLu relu_1 = ReLu();
	FCLayer hidden_layer = FCLayer(3, 3, learning_rate);
	ReLu relu_2 = ReLu();
	FCLayer output_layer = FCLayer(3, 1, learning_rate);

	MSE mse = MSE();
	Tensor2D training_data = Tensor2D(2, 4);
	training_data.set_value(0, 0, 0);
	training_data.set_value(0, 1, 0);
	training_data.set_value(0, 2, 1);
	training_data.set_value(0, 3, 1);
	training_data.set_value(1, 0, 0);
	training_data.set_value(1, 1, 1);
	training_data.set_value(1, 2, 0);
	training_data.set_value(1, 3, 1);

	Tensor2D training_labels = Tensor2D(1, 4);
	training_labels.set_value(0, 0, 0);
	training_labels.set_value(0, 1, 1);
	training_labels.set_value(0, 2, 1);
	training_labels.set_value(0, 3, 0);

	printf("beginning trainig\n");
	const int epochs = 100;
	for (int i = 0; i < epochs; i++) {
		printf("=============================================\n");
		//forward pass
		Tensor2D layer_1_out = Tensor2D(3, 4);
		input_layer.forward(layer_1_out, training_data);
		//printf("output of input layer: \n");
		//layer_1_out.print_data();
		relu_1.forward(layer_1_out, layer_1_out);
		Tensor2D layer_2_out = Tensor2D(3, 4);
		
		hidden_layer.forward(layer_2_out, layer_1_out);
		relu_2.forward(layer_2_out, layer_2_out);

		Tensor2D pred = Tensor2D(1, 4);
		output_layer.forward(pred, layer_2_out);

		printf("Prediction: \n");
		pred.print_data();
		printf("Labels: \n");
		training_labels.print_data();
		float loss = mse.calculate_loss(training_labels, pred);
		

		Tensor2D loss_gradients = Tensor2D(1, 4);
		mse.calculate_gradients(training_labels, pred, loss_gradients);

		//printf("Gradients \n");
		//loss_gradients.print_data();

		//back propogate
		Tensor2D output_layer_error = Tensor2D(3, 4);
		output_layer.backward(output_layer_error, loss_gradients);
		relu_2.backward(output_layer_error, output_layer_error);
		//printf("Output Layer Error: \n");
		//output_layer_error.print_data();



		Tensor2D hidden_layer_error = Tensor2D(3, 4);
		hidden_layer.backward(hidden_layer_error, output_layer_error);
		relu_1.backward(hidden_layer_error, hidden_layer_error);
		//printf("Hidden Layer Error: \n");
		//hidden_layer_error.print_data();

		Tensor2D input_layer_error = Tensor2D(2, 4);
		input_layer.backward(input_layer_error, hidden_layer_error);

		//printf("input layer weights: \n");
		//input_layer.weights->print_data();
		//printf("hidden layer weights: \n");
		//hidden_layer.weights->print_data();
		//printf("output layer weights: \n");
		//output_layer.weights->print_data();


		//printf("Output Layer Weights: \n");
		//output_layer.weights -> print_data();
		//printf("Output Layer Biases: \n");
		//output_layer.bias -> print_data();
		//printf("\n");

		//printf("Hidden Layer Weights: \n");
		//hidden_layer.weights->print_data();
		//printf("Hidden Layer Biases: \n");
		//hidden_layer.bias->print_data();
		//printf("\n");

		//printf("Input Layer Weights: \n");
		//input_layer.weights->print_data();
		//printf("Input Layer Biases: \n\n");
		//input_layer.bias->print_data();
		//printf("\n");
		
		printf("Epoch: %i, Loss: %f ======================================================\n", i, loss);
	}



    

    return 0;
}
