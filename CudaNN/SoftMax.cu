#include "SoftMax.cuh"


SoftMax::SoftMax() {
	input_data_ = new Tensor2D();
	output_data_ = new Tensor2D();
}

SoftMax::~SoftMax() {
	delete input_data_;
	delete output_data_;
}

int SoftMax::forward(Tensor2D&out, Tensor2D&in) {
	return 0;
}

int SoftMax::backward(Tensor2D& out_err, Tensor2D& in_err) {
	return 0;
}