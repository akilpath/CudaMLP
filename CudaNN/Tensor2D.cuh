#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

class Tensor2D {
	public:
		int rows;
		int columns;
		int total_elements;
		float* data;

		Tensor2D(int width, int height);
		~Tensor2D();
		void set_value(int r, int c, float val);
		float get_value(int r, int c);
		void print_data();
		void scalar_multiply(float scalar);
		int tensor_multiply(Tensor2D &out, Tensor2D &in);
		int tensor_add(Tensor2D& in);

};