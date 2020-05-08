#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<iostream>

__global__ void matixMultiplication(int *gA, int*gB, int *gC, int row, int col) {
	unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int idy = blockDim.y*blockIdx.y + threadIdx.y;
	if (idx < row && idy < col){
		int cvalue = 0;
		for (int i = 0; i < col; i++){
			cvalue+=gA[idy*row+i] * gB[idx+(col*i)];
			//printf("The C index is %d and the A index is %d B is %d\n", idy*row + idx, idy*row + i, idx + (col*i));			
			//__syncthreads();
		}
		gC[idy*row + idx] = cvalue;
		//__syncthreads();
	}
}
int main()
{
	int row = 800;
	int col = 800;
	int *A,*B,*C, *devA, *devB, *devC,*hostC;

	// Allocating memory for host matrices A and B
	A = new int[row*col];
	B = new int[row*col];

	//allocating memory to store the output matrix C
	C = new int[row*col];
	hostC = new int[row*col];
	// initialising values for Matrix A
	for (int i = 0; i < row*col; i++) A[i] = i;
	// initialising values for Matrix B
	for (int i = 0; i < row*col; i++) B[i] = i;

	// Allocating memory for host matrices devA, devB and output matrix devC
	cudaMalloc((void**)&devA, row*col*sizeof(int));
	cudaMalloc((void**)&devB, row*col * sizeof(int));
	cudaMalloc((void**)&devC, row*col * sizeof(int));

	// copying the data from hos tot device
	

	//host multiplication
	for (int j = 0; j < col; j++)
	{
		
		for (int i = 0; i < row; i++)
		{
			int temp = 0;
			for (int k = 0; k < col; k++)
			{
				temp += A[col*j + k] * B[row*k + i];
				//std::cout << col * j + k << " " << row * k + i << std::endl;
			}
			//std::cout << std::endl << col * j + i << std::endl;
			C[col*j + i] = temp;
		}

	}
	//for (int i = 0; i < row*col; i++)std::cout << C[i] << " " << i << std::endl;
	//Threads per block is 32
	cudaMemcpy(devA, A, row*col * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devB, B, row*col * sizeof(int), cudaMemcpyHostToDevice);
	dim3 block(32,32);
	dim3 grid((row + block.x - 1) / block.x, (col + block.y - 1) / block.y);

	matixMultiplication << <block, grid >> > (devA, devB,devC,row,col);

	cudaMemcpy(hostC, devC, row*col * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < row*col; i++)
	{
		if (hostC[i] != C[i])
		{
			std::cout << hostC[i] << " " << C[i] << std::endl;
			std::cout << " You got some work to do chap !!" << std::endl;
			goto exit;
		}
	}
	std::cout << "You made it !!" << std::endl;
exit:
	delete(A);
	delete(B);
	delete(C);
	delete(hostC);
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);
	return 0;
}