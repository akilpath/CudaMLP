#ifndef MNIST_MODEL_HPP
#define MNIST_MODEL_HPP

#include "Tensor2D.cuh"
#include "FCLayer.cuh"
#include "MSE.cuh"
#include "ReLu.cuh"
#include "SoftMax.cuh"
#include "MNISTDataset.hpp"

class MNISTModel {
public:
	MNISTModel();
	~MNISTModel();
	void train();
	void forward(Tensor2D &pred, Tensor2D &data);

private:
	FCLayer* input_layer;
	ReLu* relu_1;
	FCLayer* hidden_layer;
	SoftMax* softmax;

	MSE* mse;


};

#endif