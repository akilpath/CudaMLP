#include "XORModel.hpp"



XORModel::XORModel() {
	double learning_rate = 0.03;
	input_layer = new FCLayer(2, 3, learning_rate);
	relu_1 = new ReLu();
	hidden_layer = new FCLayer(3, 3, learning_rate);
	relu_2 = new ReLu();
	output_layer = new FCLayer(3, 1, learning_rate);

	mse = new MSE();
}

XORModel::~XORModel() {
	if (input_layer != nullptr) delete input_layer;
	if (relu_1 != nullptr) delete relu_1;
	if (hidden_layer != nullptr) delete hidden_layer;
	if (relu_2 != nullptr) delete relu_2;
	if (output_layer != nullptr) delete output_layer;
	if (mse != nullptr) delete mse;

}

void XORModel::forward(Tensor2D& out, Tensor2D& in) {
	Tensor2D layer_1_out = Tensor2D(3, 4);
	input_layer -> forward(layer_1_out, in);
	relu_1 -> forward(layer_1_out, layer_1_out);

	Tensor2D layer_2_out = Tensor2D(3, 4);
	hidden_layer -> forward(layer_2_out, layer_1_out);
	relu_2 -> forward(layer_2_out, layer_2_out);

	output_layer -> forward(out, layer_2_out);
}

void XORModel::train() {
	const int epochs = 1000;

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
	printf("Beginning Training\n");
	for (int i = 0; i < epochs; i++) {
		printf("=============================================\n");
		//forward pass
		Tensor2D pred = Tensor2D(1, 4);
		forward(pred, training_data);

		printf("Prediction: \n");
		pred.print_data();
		printf("Labels: \n");
		training_labels.print_data();
		float loss = mse -> calculate_loss(training_labels, pred);

		Tensor2D loss_gradients = Tensor2D(1, 4);
		mse -> calculate_gradients(training_labels, pred, loss_gradients);

		//back propogate
		Tensor2D output_layer_error = Tensor2D(3, 4);
		output_layer -> backward(output_layer_error, loss_gradients);
		relu_2 -> backward(output_layer_error, output_layer_error);

		Tensor2D hidden_layer_error = Tensor2D(3, 4);
		hidden_layer -> backward(hidden_layer_error, output_layer_error);
		relu_1 -> backward(hidden_layer_error, hidden_layer_error);

		Tensor2D input_layer_error = Tensor2D(2, 4);
		input_layer -> backward(input_layer_error, hidden_layer_error);
		printf("Epoch: %i, Loss: %f ======================================================\n", i, loss);
	}
}