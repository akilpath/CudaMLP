#include "MNISTModel.hpp"

MNISTModel::MNISTModel() {
	double learning_rate = 0.1;
	input_layer = new FCLayer(784, 10, learning_rate);
	relu_1 = new ReLu();
	hidden_layer = new FCLayer(10, 10, learning_rate);
	softmax = new SoftMax();
	mse = new MSE();
}

MNISTModel::~MNISTModel() {
	if (input_layer != nullptr) delete input_layer;
	if (relu_1 != nullptr) delete relu_1;
	if (hidden_layer != nullptr) delete hidden_layer;
	if (softmax != nullptr) delete softmax;
	if (mse != nullptr) delete mse;
}

void MNISTModel::forward(Tensor2D &pred, Tensor2D &data) {
	Tensor2D layer_1_out = Tensor2D(10, data.columns());

	input_layer->forward(layer_1_out, data);
	relu_1->forward(layer_1_out, layer_1_out);

	Tensor2D layer_2_out = Tensor2D(10, data.columns());
	hidden_layer->forward(layer_2_out, layer_1_out);

	softmax->forward(pred, layer_2_out);
}



void MNISTModel::train() {
	const int epochs = 500;
	const int batch_size = 32;
	MNISTDataset mnist = MNISTDataset();
	mnist.load_dataset("C:\\Users\\Admin\\Desktop\\mnist\\mnist_train.csv", "C:\\Users\\Admin\\Desktop\\mnist\\mnist_test.csv",32, 100);
	mnist.shuffle_dataset();

	//mnist.batchify(batch_size);

	for (int i = 0; i < epochs; i++) {
		printf("======= Epoch %d =======================================\n", i+1);
		float epoch_loss = 0;
		float epoch_accuracy = 0;
		for (int batch_index = 0; batch_index * batch_size < mnist.training_images.size(); batch_index++) {
			//forward pass
			//printf("Beginning batch %d\n", batch_index);s
			Tensor2D* data;
			Tensor2D* labels;
			
			//data = mnist.training_data[batch_index];
			//labels = mnist.training_labels[batch_index];

			mnist.get_training_batch(data, labels, batch_size, batch_index);
			int current_batch_size = data->columns();
			//printf("Loaded batch\n");

			
			Tensor2D pred = Tensor2D(10, current_batch_size);
			forward(pred, *data);
			epoch_accuracy += pred.tensor_accuracy(*labels);

			//printf("Prediction: \n");
			//pred.print_data();
			//printf("Labels: \n");
			//labels -> print_data();

			Tensor2D gradients = Tensor2D(10, current_batch_size);
			
			

			mse->calculate_gradients(*labels, pred, gradients);
			if (batch_index == 0) {
				printf("First batch\n");
				printf("Pred: \n");
				pred.print_data();
				printf("Labels: \n");
				labels->print_data();
				printf("Gradients: \n");
				gradients.print_data();
			}
			//printf("Calculated gradients\n");
			//gradients.print_data();

			float loss = mse->calculate_loss(*labels, pred);
			//printf("Batch: %d, Loss: %f\n", batch_index, loss);
			epoch_loss += loss;
			//backward pass

			//printf("backprop output layer\n");
			Tensor2D temploss_1 = Tensor2D(10, current_batch_size);
			hidden_layer->backward(temploss_1, gradients, false);
			//printf("Temploss 1\n");
			//temploss_1.print_data();

			//printf("backprop input layer\n");
			Tensor2D temploss_2 = Tensor2D(784, current_batch_size);
			relu_1->backward(temploss_1, temploss_1);
			//printf("Temploss 2\n");
			//temploss_1.print_data();
			input_layer->backward(temploss_2, temploss_1, tru908e);

			delete data;
			delete labels;
			//printf("Done looping\n");
		}
		epoch_loss /= std::ceil(mnist.training_images.size() / (double) batch_size);
		epoch_accuracy /= std::ceil(mnist.training_images.size() / (double) batch_size);
		printf("Epoch %d / %d: Average epoch loss: %f, Average accuracy: %f\n", i+1, epochs, epoch_loss, epoch_accuracy);
	
		//test
		float test_loss = 0;
		float test_accuracy = 0;
		Tensor2D* test_data;
		Tensor2D* test_labels;
		for (int batch_index = 0; batch_index * batch_size < mnist.test_images.size(); batch_index++) {
			mnist.get_test_batch(test_data, test_labels, batch_size, batch_index);
			int current_batch_size = test_data->columns();
			Tensor2D pred = Tensor2D(10, current_batch_size);
			forward(pred, *test_data);
			test_loss += mse->calculate_loss(*test_labels, pred);
			test_accuracy += pred.tensor_accuracy(*test_labels);
			delete test_data;
			delete test_labels;
		}
		test_loss /= std::ceil(mnist.test_images.size() / (double) batch_size);
		test_accuracy /= std::ceil(mnist.test_images.size() / (double) batch_size);
		printf("Test loss: %f, Test accuracy: %f\n", test_loss, test_accuracy);

		printf("=============================================\n");
	}
}
