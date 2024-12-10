#ifndef FCLAYER_HPP
#define FCLAYER_HPP

#include "Tensor2D.cuh"
#include "VectorND.cuh"
#include <random>
#include <iostream>

class FCLayer {

	public:
		int input_size;
		int output_size;
		double learning_rate;
		~FCLayer();
		FCLayer();
		FCLayer(int input_size, int output_size, double lr);
		Tensor2D* weights;
		VectorND* bias;
		int forward(Tensor2D &out, Tensor2D &input_data);
		int backward(Tensor2D & output_err, Tensor2D &input_err, bool debug=false);

		void init_weights_biases();
	private:
		Tensor2D* input_data_;
		Tensor2D* output_data_;
};

#endif