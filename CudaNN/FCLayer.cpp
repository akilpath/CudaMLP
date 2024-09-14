#include "FCLayer.hpp"

FCLayer::FCLayer() {
	learning_rate = 0.01;
	input_size = 0;
	output_size = 0;
}

FCLayer::FCLayer(int in_size, int out_size, double lr) {
	learning_rate = lr;
	input_size = in_size;
	output_size = out_size;

	weights = Tensor2D::Tensor2D(input_size, output_size);
	bias = VectorND::VectorND(output_size);
}


Tensor2D* FCLayer::forward(Tensor2D& in) {
	if (in.rows() != weights.columns()) {
		std::cerr << "Input data size does not match layer input size" << std::endl;
		return nullptr;
	}
	Tensor2D* output = new Tensor2D(bias.size(), in.columns());
	input_data_ = new Tensor2D(in.rows(), in.columns());
	in.copy(*input_data_);
	Tensor2D intermediate_tensor = Tensor2D(output -> rows(), output -> columns());
	in.tensor_multiply(intermediate_tensor, weights);
	intermediate_tensor.add_vector_to_columns(*output, bias);
	output_data_ = output;
	return output;
}

Tensor2D* FCLayer::backward(Tensor2D& input_err) {

	Tensor2D input_data_t = Tensor2D(weights.columns(), weights.rows());
	input_data_ -> transpose(input_data_t);

	Tensor2D weight_gradients = Tensor2D(weights.rows(), weights.columns());
	VectorND bias_gradients = VectorND(bias.size());

	input_data_t.tensor_multiply(weight_gradients, input_err);
	input_err.mean_rows(bias_gradients);

	weight_gradients.scalar_multiply(weight_gradients, -learning_rate);
	bias_gradients.scalar_multiply(bias_gradients, -learning_rate);
	weights.tensor_add(weights, weight_gradients);
	bias.vector_add(bias_gradients, bias);

	Tensor2D* output_err = new Tensor2D(input_data_->rows(), input_data_->columns());
	Tensor2D weights_t = Tensor2D(weights.columns(), weights.rows());
	weights.transpose(weights_t);
	weights_t.tensor_multiply(*output_err, input_err);

	delete input_data_;
	delete output_data_;

	return output_err;
}

