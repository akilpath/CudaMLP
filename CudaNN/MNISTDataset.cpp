#include "MNISTDataset.hpp"

MNISTDataset::MNISTDataset() {
}

MNISTDataset::~MNISTDataset() {
	clear_old_data();
}

void MNISTDataset::clear_old_data() {
	for (int i = 0; i < training_data.size(); i++) {
		if (training_data[i] != nullptr) delete training_data[i];
	}

	for (int i = 0; i < training_labels.size(); i++) {
		if (training_labels[i] != nullptr) delete training_labels[i];
	}

	for (int i = 0; i < test_data.size(); i++) {
		if (test_data[i] != nullptr) delete test_data[i];
	}
	for (int i = 0; i < test_labels.size(); i++) {
		if (test_labels[i] != nullptr) delete test_labels[i];
	}

	training_data.clear();
	training_labels.clear();
	test_data.clear();
	test_labels.clear();
}

void MNISTDataset::batchify(int batch_size) {
	clear_old_data();

	for (int i = 0; i < training_images.size(); i += batch_size) {
		int tensor_cols = std::min(batch_size, (int) (training_images.size() - i));
		Tensor2D* data = new Tensor2D(784, tensor_cols);
		Tensor2D* labels = new Tensor2D(10, tensor_cols);
		for (int j = 0; j < tensor_cols; j++) {
			std::vector<float>& image_data = training_images[i + j];
			labels->set_value((int)image_data[0], j, 1.0);
			for (int k = 0; k < 784; k++) {
				data->set_value(k, j, image_data[k + 1]);
			}
		}
		training_data.push_back(data);
		training_labels.push_back(labels);
	}

	for (int i = 0; i < test_images.size(); i += batch_size) {
		Tensor2D* data = new Tensor2D(784, batch_size);
		Tensor2D* labels = new Tensor2D(10, batch_size);
		int tensor_cols = std::min(batch_size, (int)(training_images.size() - i));
		for (int j = 0; j < tensor_cols; j++) {
			std::vector<float>& image_data = test_images[i + j];
			labels->set_value((int)image_data[0], j, 1.0);
			for (int k = 0; k < 784; k++) {
				data->set_value(k, j, image_data[k + 1]);
			}
		}
		test_data.push_back(data);
		test_labels.push_back(labels);
	}

	if (training_data.size() != training_labels.size() || test_data.size() != test_labels.size()) {
		printf("MNISTDataset::batchify Error: Data and labels size mismatch\n");
	}

	printf("Batchified Training Data: %d batches\n", (int)training_data.size());
}


void MNISTDataset::load_dataset(std::string training_fpath, std::string test_fpath, int training_count, int test_count) {
	std::fstream train_csv = std::fstream(training_fpath, std::ios::in);
	std::string line, word, temp;

	if (!train_csv.is_open()) {
		printf("Failed to open file\n");
		return;
	}

	std::getline(train_csv, line); // skip the first line, it is columns
	
	int image_count = 0;
	printf("Loading Training Data from MNIST Dataset\n");
	while (std::getline(train_csv, line) && image_count < training_count) {
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
	int test_image_count = 0;
	printf("Loading Test Data from MNIST Dataset\n");
	while (std::getline(test_csv, line) && test_image_count < test_count) {
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