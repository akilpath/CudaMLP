#ifndef VECTORND_CUH
#define VECTORND_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>




class VectorND {
	public:
		VectorND(int size);
		VectorND(int size, float *in_data);
		VectorND();
		~VectorND();
		int size();
		void print_data() const;

		int scalar_multiply(VectorND &out, float scalar);
		int vector_add(VectorND& out, VectorND& in);
		int vector_subtract(VectorND& out, VectorND& in);
		int vector_multiply(VectorND& out, VectorND& in);
		float* data_;

	private:
		int size_;

};

#endif