#ifndef TENSOR2D_CUH
#define TENSOR2D_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "VectorND.cuh"
#include <iostream>




class Tensor2D {
	public:
		Tensor2D(int rows, int columns);
		Tensor2D();
		~Tensor2D();
		void set_value(int r, int c, float val);
		float get_value(int r, int c);
		void print_data();
		
		int scalar_multiply(Tensor2D &out, float scalar);
		int tensor_multiply(Tensor2D &out, Tensor2D &in);
		int tensor_add(Tensor2D &out, Tensor2D& in);
		int add_vector_to_columns(Tensor2D& out, VectorND& in);
		int transpose(Tensor2D& out);

		int tensor_element_multiply(Tensor2D& out, Tensor2D& in);

		int mean_rows(VectorND& target);

		int copy(Tensor2D& target);



		int rows() const;
		int columns() const;
		int total_elements() const;
		float* data_;

	private:
		int rows_;
		int columns_;
		int total_elements_;

};

#endif