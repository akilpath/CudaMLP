#include "MNISTDataset.hpp"

MNISTDataset::MNISTDataset() {
}

MNISTDataset::~MNISTDataset() {
}


void MNISTDataset::load_dataset(std::string training_fpath, std::string test_fpath) {
	std::fstream train_csv = std::fstream(training_fpath, std::ios::in);
	std::string line, word, temp;

	if (!train_csv.is_open()) {
		printf("Failed to open file\n");
		return;
	}

	std::getline(train_csv, line); // skip the first line, it is columns
	
	int image_count = 0;
	printf("Loading Training Data from MNIST Dataset\n");
	while (std::getline(train_csv, line) && image_count < 60000) {
		//printf("Inner loop\n");
		std::vector<float> image_data; // first element is the label, rest are pixel values
		std::stringstream s(line);
		std::string data;
		while (std::getline(s, data, ',')) {
			if (data == ",") continue;
			image_data.push_back(std::stof(data));
		}
		//printf("Image data size: %d\n", image_data.size());
		training_images.push_back(image_data);
		image_count++;
	}
	printf("Training dataset size: %d images\n", (int) training_images.size());


	//load test
	std::fstream test_csv = std::fstream(test_fpath, std::ios::in);
	if (!test_csv.is_open()) {
		printf("Failed to open file\n");
		return;
	}

	std::getline(test_csv, line); // skip the first line, it is columns
	printf("Loading Test Data from MNIST Dataset\n");
	while (std::getline(test_csv, line)) {
		std::vector<float> image_data; // first element is the label, rest are pixel values
		std::stringstream s(line);
		std::string data;
		while (std::getline(s, data, ',')) {
			if (data == ",") continue;
			image_data.push_back(std::stof(data));
		}
		test_images.push_back(image_data);
	}
	printf("Test dataset size: %d images\n", (int)test_images.size());
}

void MNISTDataset::shuffle_dataset() {
	std::random_shuffle(training_images.begin(), training_images.end());
	std::random_shuffle(test_images.begin(), test_images.end());
}

void MNISTDataset::get_training_batch(Tensor2D* &data, Tensor2D* &labels, unsigned int batch_size, unsigned int batch_index) {
	if (batch_size * batch_index > training_images.size()) {
		printf("MNISTDataset::get_batch Batch index out of bounds\n");
		return;
	}

	unsigned int batch_index_start = batch_size * batch_index;
	unsigned int batch_index_end = batch_index_start + batch_size;
	if (batch_index_end > training_images.size()) {
		batch_index_end = training_images.size();
	}

	data = new Tensor2D(784, batch_index_end - batch_index_start);
	labels = new Tensor2D(10, batch_index_end - batch_index_start);

	for (unsigned int i = batch_index_start; i < batch_index_end && i < training_images.size(); i++) {
		std::vector<float> &image_data = training_images[i];
		labels -> set_value((int)image_data[0], i - batch_index_start, 1.0);
		for (int j = 0; j < 784; j++) {
			data -> set_value(j, i - batch_index_start, image_data[j+1]);
		}

	}
	data -> scalar_multiply(*data, 1.0 / 255.0);
}

void MNISTDataset::get_test_batch(Tensor2D* &data, Tensor2D* &labels, unsigned int batch_size, unsigned int batch_index) {
	if (batch_size * batch_index > test_images.size()) {
		printf("MNISTDataset::get_batch Batch index out of bounds\n");
		return;
	}

	unsigned int batch_index_start = batch_size * batch_index;
	unsigned int batch_index_end = batch_index_start + batch_size;
	if (batch_index_end > test_images.size()) {
		batch_index_end = test_images.size();
	}
	data = new Tensor2D(784, batch_index_end - batch_index_start);
	labels = new Tensor2D(10, batch_index_end - batch_index_start);

	for (unsigned int i = batch_index_start; i < batch_index_end && i < test_images.size(); i++) {
		std::vector<float>& image_data = test_images[i];
		labels -> set_value((int)image_data[0], i - batch_index_start, 1.0);
		for (int j = 0; j < 784; j++) {
			data -> set_value(j, i - batch_index_start, image_data[j + 1]);
		}

	}
	data -> scalar_multiply(*data, 1.0 / 255.0);
}