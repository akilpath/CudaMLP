#ifndef MNISTDATASET_CUH
#define MNISTDATASET_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include "Tensor2D.cuh"
#include "math.h"


class MNISTDataset {
	public:
		MNISTDataset();

		~MNISTDataset();

		std::vector<std::vector<float>> training_images;
		std::vector<std::vector<float>> test_images;

		std::vector<Tensor2D*> training_data;
		std::vector<Tensor2D*> training_labels;

		std::vector<Tensor2D*> test_data;
		std::vector<Tensor2D*> test_labels;


		void load_dataset(std::string training_fpath, std::string test_fpath, int training_count, int test_count);

		void batchify(int batch_size);
		void clear_old_data();
		void shuffle_dataset();

		void get_training_batch(Tensor2D* &data, Tensor2D* &labels, unsigned int batch_size, unsigned int batch_index);
		void get_test_batch(Tensor2D* &data, Tensor2D* &labels, unsigned int batch_size, unsigned int batch_index);
};

#endif