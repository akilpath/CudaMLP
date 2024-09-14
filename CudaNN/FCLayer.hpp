#ifndef FCLAYER_HPP
#define FCLAYER_HPP

#include "Tensor2D.cuh"
#include "VectorND.cuh"

class FCLayer {

	public:
		int input_size;
		int output_size;
		double learning_rate;
		FCLayer();
		FCLayer(int input_size, int output_size, double lr);
		Tensor2D weights;
		VectorND bias;
		Tensor2D* forward(Tensor2D &input_data);
		Tensor2D* backward(Tensor2D &input_err);
	private:
		Tensor2D* input_data_;
		Tensor2D* output_data_;
};

#endif