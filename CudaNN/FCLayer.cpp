#include "FCLayer.hpp"

FCLayer::FCLayer() {
	learning_rate = 0.01;
	input_size = 0;
	output_size = 0;

	input_data_ = nullptr;
	output_data_ = nullptr;
	weights = nullptr;
	bias = nullptr;
}

FCLayer::FCLayer(int in_size, int out_size, double lr) {
	learning_rate = lr;
	input_size = in_size;
	output_size = out_size;

	input_data_ = nullptr;
	output_data_ = nullptr;

	weights = new Tensor2D(output_size, input_size);
	bias = new VectorND(output_size);
	printf("Initializing weights and biases\n");
	init_weights_biases();
	printf("Weights and biases initialized\n");
}

FCLayer::~FCLayer() {
	if (input_data_ != nullptr) {
		delete input_data_;
	}
	if (output_data_ != nullptr) {
		delete output_data_;
	}
	if (weights != nullptr) {
		delete weights;
	}
	if (bias != nullptr) {
		delete bias;
	}
}

int FCLayer::forward(Tensor2D& out, Tensor2D& in) {
	if (in.rows() != weights -> columns()) {
		std::cerr << "Input data size does not match layer input size" << std::endl;
		return -1;
	}
	if (out.rows() != weights -> rows() || out.columns() != in.columns()) {
		std::cerr << "Output data size does not match layer output size" << std::endl;
		return -1;
	}

	if (input_data_ != nullptr) {
		delete input_data_;
	}
	if (output_data_ != nullptr) {
		delete output_data_;
	}

	input_data_ = new Tensor2D(in.rows(), in.columns());
	output_data_ = new Tensor2D(out.rows(), out.columns());

	in.copy(*input_data_);
	//printf("weights\n");
	//weights->print_data();
	//printf("in\n");
	//in.print_data();
	weights->tensor_multiply(out, in);
	//printf("out\n");
	//out.print_data();

	//printf("bias\n");
	//bias->print_data();
	out.add_vector_to_columns(out, *bias);
	out.copy(*output_data_);
	return 0;
}

int FCLayer::backward(Tensor2D &output_error, Tensor2D& input_err) {
	if (input_err.rows() != weights->rows()) {
		std::cerr << "Input error size does not match layer input size for backwards pass" << std::endl;
		return -1;
	}
	if (input_data_ == nullptr || output_data_ == nullptr) {
		std::cerr << "No forward pass has been run before backward pass" << std::endl;
		return -1;
	}

	Tensor2D input_data_t = Tensor2D(input_data_ -> columns(), input_data_ -> rows());
	input_data_ -> transpose(input_data_t);

	Tensor2D weight_gradients = Tensor2D(weights -> rows(), weights -> columns());
	VectorND bias_gradients = VectorND(bias -> size());

	input_err.tensor_multiply(weight_gradients, input_data_t);

	input_err.mean_rows(bias_gradients);
;

	weight_gradients.scalar_multiply(weight_gradients, (-learning_rate) / (input_data_->columns()));

	bias_gradients.scalar_multiply(bias_gradients, -learning_rate);

	weights -> tensor_add(*weights, weight_gradients);
	bias -> vector_add(*bias, bias_gradients);

	Tensor2D weights_t = Tensor2D(weights -> columns(), weights -> rows());
	weights -> transpose(weights_t);
	weights_t.tensor_multiply(output_error, input_err);

	if (input_data_ != nullptr) {
		delete input_data_;
		input_data_ = nullptr;
	}
	if (output_data_ != nullptr) {
		delete output_data_;
		output_data_ = nullptr;
	}
	return 0;
}

void FCLayer::init_weights_biases() {
	unsigned seed = 32434;
	std::default_random_engine generator(seed);
	std::normal_distribution<float> distribution(0.0, 1.0);

	for (int i = 0; i < weights -> rows(); i++) {
		for (int j = 0; j < weights -> columns(); j++) {
			weights -> set_value(i, j, distribution(generator));
		}
	}

	seed = 32422;
	generator = std::default_random_engine(seed);
	for (int i = 0; i < bias -> size(); i++) {
		bias -> data_[i] = distribution(generator);
	}
	cudaDeviceSynchronize();
}

