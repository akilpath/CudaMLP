#include "Tensor2D.cuh"


__global__ void _tensor_scalar_multiply(float* out, float* data, int size, float scalar) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	if (index < size) {
		out[index] = data[index] * scalar;
	}
}

__global__ void _tensor_add(float* out, float* data_1, float* data_2, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		out[index] = data_1[index] + data_2[index];
	}
}

__global__ void _tensor_multiply(float* out, float* data_1, int r1, int c1, float* data_2, int r2, int c2) {
	__shared__ float data_s[18];

	float* data_1s = data_s;
	float* data_2s = data_s + 9;

	int output_rows = r1;
	int output_columns = c2;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float acc = 0;
	for (int k = 0; k < c1; k += blockDim.x) {
		if (threadIdx.x + k < c1 && y < r1) {
			data_1s[threadIdx.y*blockDim.x + threadIdx.x] = data_1[y * c1 + threadIdx.x + k];
		}
		if (x < c2 && threadIdx.y + k < r2) {
			data_2s[threadIdx.y * blockDim.x + threadIdx.x] = data_2[(threadIdx.y + k) * c2 + x];
		}
		__syncthreads();
		//if (x == 0 && y == 0) {
		//	printf("Data1s: %.2f, %.2f, %.2f\n", data_1s[threadIdx.y*blockDim.x], data_1s[threadIdx.y * blockDim.x + 1], data_1s[threadIdx.y * blockDim.x + 2]);
		//	printf("Data2s: %.2f, %.2f, %.2f\n", data_2s[threadIdx.x], data_2s[blockDim.x + threadIdx.x], data_2s[2 * blockDim.x + threadIdx.x]);
		//}
		if (x < output_columns && y < output_rows) {
			for (int i = 0; i < blockDim.x; i++) {
				if (i + k < c1) {
					//printf("deez");
					//if (x == 0 && y == 0) {
					//	printf("data1: %.2f, data2: %.2f\n", data_1s[threadIdx.y * blockDim.x + i], data_2s[(i * blockDim.x) + threadIdx.x]);
					//}
					//printf("data1: %.2f, data2: %.2f", data_1s[threadIdx.y * blockDim.x + i], data_2s[(i * blockDim.x) + threadIdx.x]);
					acc += data_1s[threadIdx.y*blockDim.x + i] * data_2s[(i*blockDim.x) + threadIdx.x];
				}
				//printf("acc y: %i, x: %i,  %i\n", y, x, acc);
			}
			out[y * output_columns + x] = acc;
		}
		//printf("acc y: %i, x: %i,  %i\n",y, x, acc);
		
	}
}

__global__ void _tensor_element_multiply(float* out, float* data_1, float* data_2, int size) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size) {
		out[x] = data_1[x] * data_2[x];
	}
}

__global__ void _add_vector_to_columns(float* out, float* data_1, int rows, int columns, float* vector) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;


	//if (x < columns && y < rows) {
	//	out[y * columns + x] = data_1[y * columns + x] + vector[y];

	//	printf("x,y : %i, %i; out %.2f, data: %.2f, vector: %.2f\n", x,y,out[y * columns + x], data_1[y * columns + x], vector[y]);
	//}
	__shared__ float vector_s[8];
	int local_vec_idx = threadIdx.y * blockDim.x + threadIdx.x;
	if (threadIdx.x < 1 && y < rows) {
		vector_s[threadIdx.y] = vector[y];
	}
	__syncthreads();
	if (x < columns && y < rows) {
		out[y * columns + x] = data_1[y * columns + x] + vector_s[threadIdx.y];
	}
}

__global__ void _transpose(float* out, float* data, int out_rows, int out_columns) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < out_columns && y < out_rows) {
		out[y * out_columns + x] = data[x * out_rows + y];
	}
}

__global__ void _copy(float* target, float* source, int rows, int columns) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < columns && y < rows) {
		target[y*columns + x] = source[y*columns + x];
	}
}

__global__ void _mean_rows(float* target, float* data, int rows, int columns) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < rows) {
		float sum = 0;
		for (int i = 0; i < columns; i++) {
			sum += data[x * columns + i];
		}
		target[x] = sum / columns;
	}
}



void Tensor2D::set_value(int r, int c, float val){
	data_[r * columns_ + c] = val;
}

float Tensor2D::get_value(int r, int c) {
	return data_[r * columns_ + c];
}

void Tensor2D::print_data() {
	printf("[");
	for (int i = 0; i < rows_; i++) {
		if (i == 0) {printf("[");}
		else {
			printf(" [");
		};

		
		for (int j = 0; j < columns_; j++) {
			printf("%.3f, ", data_[i * columns_ + j]);
		}
		if (i == rows_ - 1){
			printf("]");
		}
		else {
			printf("]\n");
		}
	}
	printf("]\n");
}

int Tensor2D::scalar_multiply(Tensor2D &out, float scalar) {
	if (out.rows() != rows_ || out.columns() != columns_) {
		std::cout << "Output tensor size does not match input tensor size" << std::endl;
		return -1;
	}

	int num_blocks = (int) ceil(total_elements_ / 128.0);
	if (num_blocks == 0) {
		num_blocks = 1;
	}
	_tensor_scalar_multiply<<<num_blocks, 128>>>(out.data_, data_, total_elements(), scalar);
	cudaDeviceSynchronize();
	return 0;
}

int Tensor2D::tensor_multiply(Tensor2D &out, Tensor2D &in) {
	if (this->columns_ != in.rows()) {
		return -1;
	}
	if (out.rows() != this->rows_ || out.columns() != in.columns()) {
		return -1;
	}
	dim3 block_size(3, 3);
	int num_blocks_y = (int) ceil(((float) out.rows()) / 3);
	int num_blocks_x = (int) ceil(((float) out.columns()) / 3);
	if (num_blocks_x == 0) {
		num_blocks_x = 1;
	}
	if (num_blocks_y == 0) {
		num_blocks_y = 1;
	}
	dim3 num_blocks(num_blocks_x, num_blocks_y);
	_tensor_multiply <<<num_blocks, block_size>>> (out.data_, this->data_, rows_, columns_, in.data_, in.rows(), in.columns());
	cudaDeviceSynchronize();
	return 0;
}

int Tensor2D::tensor_element_multiply(Tensor2D& out, Tensor2D& in) {
	if (columns_ != in.columns() || rows_ != in.rows()) {
		std::cout << "Input tensor sizes do not match" << std::endl;
		return -1;
	}
	if (out.columns() != columns_ || out.rows() != rows_) {
		std::cout << "Output tensor sizes do not match" << std::endl;
		return -1;
	}
	int num_blocks = (int)ceil(total_elements_ / 128.0);
	if (num_blocks == 0) {
		num_blocks = 1;
	}
	_tensor_element_multiply << <num_blocks, 128 >> > (out.data_, this->data_, in.data_, total_elements_);
	cudaDeviceSynchronize();
	return 0;
}

int Tensor2D::tensor_add(Tensor2D &out, Tensor2D& in) {
	if (columns_ != in.columns() || rows_ != in.rows()) {
		std::cout << "Input tensor sizes do not match" << std::endl;
		return -1;
	}
	if (out.columns() != columns_ || out.rows() != rows_) {
		std::cout << "Output tensor sizes do not match" << std::endl;
		return -1;
	}
	int num_blocks = (int) ceil(total_elements_ / 128.0);
	if (num_blocks == 0) {
		num_blocks = 1;
	}
	_tensor_add << <num_blocks, 128 >> > (out.data_, this->data_, in.data_, total_elements_);
	cudaDeviceSynchronize();
	return 0;
}


int Tensor2D::add_vector_to_columns(Tensor2D& out, VectorND& in)
{
	if (in.size() != this->rows_) {
		printf("Vector size does not match rows in tensor, Vec size %i, rows %i\n", in.size(), this->rows_);
		return -1;
	}
	if (out.rows() != this->rows_ || out.columns() != this->columns_) {
		printf("Output tensor size does not match input tensor size, out dims: %i, %i, this dims: %i, %i\n", out.rows(), out.columns(), rows_, columns_);
		return -1;
	}
	dim3 block_size(8, 8);

	int num_blocks_x = (int)ceil(((float)out.columns()) / block_size.x);
	if (num_blocks_x == 0) {
		num_blocks_x = 1;
	}
	int num_blocks_y = (int)ceil(((float)out.rows()) / block_size.y);
	if (num_blocks_y == 0) {
		num_blocks_y = 1;
	}
	dim3 num_blocks(num_blocks_x, num_blocks_y);

	_add_vector_to_columns << <num_blocks, block_size >> > (out.data_, this ->data_, rows_, columns_, in.data_);
	cudaDeviceSynchronize();
	return 0;
}

int Tensor2D::transpose(Tensor2D& out) {
	if (rows_ != out.columns() || columns_ != out.rows()) {
		std::cout << "Output does not have the correct dimensions" << std::endl;
		return -1;
	}

	dim3 block_size(16, 16);
	int num_blocks_x = (int)ceil(((float)out.columns()) / block_size.x);
	if (num_blocks_x == 0) {
		num_blocks_x = 1;
	}
	int num_blocks_y = (int)ceil(((float)out.rows()) / block_size.y);
	if (num_blocks_y == 0) {
		num_blocks_y = 1;
	}
	dim3 num_blocks(num_blocks_x, num_blocks_y);
	_transpose<< <num_blocks, block_size >> > (out.data_, data_, out.rows(), out.columns());
	cudaDeviceSynchronize();

	return 0;
}

int Tensor2D::copy(Tensor2D& target)
{
	if (target.rows() != rows_ || target.columns() != columns_) {
		return -1;
	}
	dim3 block_size = (16, 16);
	int num_blocks_x = (int)ceil(((float)columns_) / block_size.x);
	int num_blocks_y = (int)ceil(((float)rows_) / block_size.y);
	if (num_blocks_x == 0) {
		num_blocks_x = 1;
	}
	if (num_blocks_y == 0) {
		num_blocks_y = 1;
	}
	dim3 num_blocks(num_blocks_x, num_blocks_y);
	_copy << <num_blocks, block_size >> > (target.data_, data_, rows_, columns_);
	cudaDeviceSynchronize();
	return 0;
}

int Tensor2D::mean_rows(VectorND& target) {
	if (target.size() != rows_) {
		return -1;
	}
	int num_blocks = (int)ceil(rows_ / 128.0);
	if (num_blocks == 0) {
		num_blocks = 1;
	}
	_mean_rows << <num_blocks, 128 >> > (target.data_, data_, rows_, columns_);
	cudaDeviceSynchronize();
	return 0;
}

int Tensor2D::rows() const
{
	return rows_;
}

int Tensor2D::columns() const
{
	return columns_;
}

int Tensor2D::total_elements() const
{
	return total_elements_;
}

Tensor2D::Tensor2D(int rows, int columns) {
	this->rows_ = rows;
	this->columns_ = columns;
	total_elements_ = rows * columns;
	cudaMallocManaged(&data_, rows * columns * sizeof(float));
	for (int i = 0; i < rows * columns; i++) {
		data_[i] = 0;
	}
}

Tensor2D::Tensor2D() {
	rows_ = 0;
	columns_ = 0;
	total_elements_ = 0;
	data_ = nullptr;
}

Tensor2D::~Tensor2D() {
	if (data_ != nullptr) {
		cudaFree(data_);
	}
}
