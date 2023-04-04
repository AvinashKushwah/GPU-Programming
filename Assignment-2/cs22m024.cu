#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;

//Function to add the two matrices of size p*r:
__global__ void add_matrices(int* A, int* B, int* C, int p, int r) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < p && col < r) {
        int idx = row * r + col;
        C[idx] = A[idx] + B[idx];
    }
}


//Function to compute the transpose of a Matrix "B".
__global__ void transposeMatrix(int* inputMatrix, int* outputMatrix, int q, int r)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < q && j < r)
    {
        int index_in = i * r + j;
        int index_out = j * q + i;
        
        outputMatrix[index_out] = inputMatrix[index_in];
    }
}

//Function to compute multiplication of CDT and AB:
__global__ void matrixMultiply(int* C, int* D, int* T, int p, int q, int r) {
   
   __shared__ int sc[1024];
   if(threadIdx.x ==0)
   {
       for(int i=0;i<q;i++)
       {
            sc[i]=C[blockIdx.x *q + i];
       }
   }
     
   __syncthreads();

   int temp=0;
   for(int i=0;i<q;i++)
   {
       temp += sc[i]*D[threadIdx.x*q +i];
   }
  
  T[blockIdx.x*blockDim.x + threadIdx.x] = temp;
}




// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */
      int *b_transpose_host = (int*)malloc(r*q*sizeof(int)); 
      int *b_transpose_device;
      cudaMalloc(&b_transpose_device,r*q*sizeof(int));
      dim3 blockDim(16, 16);
      dim3 gridDim((q + blockDim.x - 1) / blockDim.x, (r + blockDim.y - 1) / blockDim.y);
      transposeMatrix<<<gridDim, blockDim>>>(d_matrixB,b_transpose_device, q, r);
      cudaMemcpy(b_transpose_host,b_transpose_device,r*q*sizeof(int),cudaMemcpyDeviceToHost);
      
   
   
      int *CD_host = (int*) malloc(p*r * sizeof(int));  
      int *CD_device;
      cudaMalloc(&CD_device,p*r*sizeof(int));
      matrixMultiply<<<p,r>>>(d_matrixC,d_matrixD,CD_device,p,q,r);
      cudaMemcpy(CD_host,CD_device,p*r*sizeof(int),cudaMemcpyDeviceToHost);
      
    

      int *AB_host = (int*) malloc(p*r * sizeof(int));  
      int *AB_device;
      cudaMalloc(&AB_device,p*r*sizeof(int));
      
      matrixMultiply<<<p,r>>>(d_matrixA,b_transpose_device,AB_device,p,q,r);
      cudaMemcpy(AB_host,AB_device,p*r*sizeof(int),cudaMemcpyDeviceToHost);
     
      
      
      dim3 threadsPerBlocks(16, 16);
      dim3 numBlocks((r + blockDim.x - 1) / blockDim.x, (p + blockDim.y - 1) / blockDim.y);
      add_matrices<<<numBlocks, threadsPerBlocks>>>(AB_device,CD_device,d_matrixE ,p, r);

	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
	
