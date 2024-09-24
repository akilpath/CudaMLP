#ifndef XOR_MODEL_HPP
#define XOR_MODEL_HPP

#include "Tensor2D.cuh"
#include "VectorND.cuh"
#include "FCLayer.cuh"
#include "ReLu.cuh"
#include "MSE.cuh"
#include <iostream>
#include <math.h>

class XORModel {
public:
	XORModel();
	~XORModel();
	void train();
	void forward(Tensor2D& out, Tensor2D& in);
private:
	FCLayer* input_layer;
	ReLu* relu_1;
	FCLayer* hidden_layer;
	ReLu* relu_2;
	FCLayer* output_layer;
	MSE* mse;
};





#endif