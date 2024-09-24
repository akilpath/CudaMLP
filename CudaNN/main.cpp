#include "Tensor2D.cuh"
#include "VectorND.cuh"
#include "FCLayer.cuh"
#include "ReLu.cuh"
#include "MSE.cuh"
#include <iostream>
#include <math.h>
#include "XORModel.hpp"
#include "MNISTDataset.hpp"

#include "MNISTModel.hpp"

int main(void)
{
	//XORModel model = XORModel();
	//model.train();

	//MNISTDataset mnist = MNISTDataset();
	//mnist.load_dataset("C:\\Users\\Admin\\Desktop\\mnist\\mnist_train.csv", "C:\\Users\\Admin\\Desktop\\mnist\\mnist_test.csv" );
	//mnist.shuffle_dataset();

	MNISTModel model = MNISTModel();
	model.train();

	//Tensor2D data = Tensor2D(784, 32);
	//Tensor2D labels = Tensor2D(10, 32);
	//mnist.get_batch(data, labels, 32, 0);
	//labels.print_data();
    return 0;
}
