#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;
#define BLOCK_SIZE 32

//Function to add the two matrices of size p*r:
__global__ void matrix_addition(int* A, int* B, int* C, int p, int r) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int colm = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < p && colm < r) 
    {
        int index = row * r + colm;
        C[index] = A[index] + B[index];
    }
}



__global__ void matrix_MultiplyAB(int *A, int *B, int *AB, int p, int q, int r) {
   

    __shared__ int s_A[1024];
    __shared__ int s_B[1024];

    int row = threadIdx.y + blockIdx.y *blockDim.y ;
    int colm = threadIdx.x + blockIdx.x *blockDim.x ;
    int temp= 0;

    for (int i = 0; i < (q - 1) / BLOCK_SIZE+1 ; i++) 
    {   
        int temp1 = i *BLOCK_SIZE + threadIdx.x;
        int temp2 = threadIdx.x + threadIdx.y * BLOCK_SIZE;
        if (row < p && temp1 < q) 
        {
            s_A[temp2] = A[row * q +temp1];
        }
       else 
        {
            s_A[temp2] = 0;
        }
       
        int temp3 = threadIdx.y + i * BLOCK_SIZE;
        int temp4 = threadIdx.x + threadIdx.y * BLOCK_SIZE;
        
        if (colm < r && temp3 < q)
         {
            s_B[temp4] = B[r* temp3 + colm];
        }
        else 
        {
            s_B[temp4] = 0;
        }
        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++) 
        {
            temp = temp +  s_A[threadIdx.y*BLOCK_SIZE + j] * s_B[j* BLOCK_SIZE +threadIdx.x];
        }

        __syncthreads();
    }

    if (row < p && colm < r) {
        AB[row * r + colm] = temp;
    }
}




//Function to compute multiplication of CD^T and AB:In this 
__global__ void matrix_MultiplyCD(int *C,int *D,int *CD,int p, int q, int r) {
   
/* Approach : 
I have declared a shared memory array sc with a size of 1024.
Each thread in a block accesses one element in Matrix A and one element in Matrix B, and multiplies them together.
Each thread stores the result of the multiplication in the shared memory array sc.
The __syncthreads() function ensures that all threads in the block have finished storing their multiplication results in sc before any thread continues.
Thread 0 in each block calculates the sum of all the multiplication results stored in sc.
Finally, the result of the block-wise matrix multiplication is stored in Matrix AB.
By utilizing shared memory, the code ensures that threads within a block are accessing contiguous memory locations in a coalesced manner. 
This reduces the number of memory transactions required and improves memory access efficiency.

*/
   int blockId = blockIdx.x* gridDim.y + blockIdx.y;

   __shared__ int sc[1024];
   int row_c = blockId / r;
   int row_d = blockId % r;

   sc[threadIdx.x] = C[row_c*q + threadIdx.x] * D[row_d * q + threadIdx.x] ;
   __syncthreads();

    if (threadIdx.x == 0) {
        
        for(int i=1;i<q;i++)
        {
            sc[0] +=  sc[i];
        }
       CD[blockId] = sc[0];
    }

}




// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, int *h_matrixC, int *h_matrixD, int *h_matrixE){
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

     /* int *b_transpose_host = (int*)malloc(r*q*sizeof(int)); 
      int *b_transpose_device;
      cudaMalloc(&b_transpose_device,r*q*sizeof(int));
      dim3 blockDim(16, 16);
      int numOfBlocks= (q + blockDim.x - 1) / blockDim.x;
      int numOfthreads_per_block = (r + blockDim.y - 1) / blockDim.y;
      dim3 gridDim(numOfBlocks,numOfthreads_per_block);
      transpose_Matrix<<<gridDim, blockDim>>>(d_matrixB,b_transpose_device, q, r);
      cudaMemcpy(b_transpose_host,b_transpose_device,r*q*sizeof(int),cudaMemcpyDeviceToHost);
      */
   
   
      int *CD_host = (int*) malloc(p*r * sizeof(int));  
      int *CD_device;
      cudaMalloc(&CD_device,p*r*sizeof(int));
      dim3 blocksize(p,r,1);
      matrix_MultiplyCD<<<blocksize,q>>>(d_matrixC,d_matrixD,CD_device,p,q,r);
      cudaMemcpy(CD_host,CD_device,p*r*sizeof(int),cudaMemcpyDeviceToHost);
      

      
      int *AB_host = (int*) malloc(p*r * sizeof(int));  
      int *AB_device;
      cudaMalloc(&AB_device,p*r*sizeof(int));
      dim3 blockDimfor_AB(32, 32);
      int GridDim_forAB_temp1 = 1+ (r - 1) / 32 ;
      int GridDim_forAB_temp2 = 1+ (p - 1) / 32;
      dim3 GridDim_forAB(GridDim_forAB_temp1 , GridDim_forAB_temp2); 
      matrix_MultiplyAB<<<GridDim_forAB, blockDimfor_AB>>>(d_matrixA,d_matrixB,AB_device,p,q,r);
      cudaMemcpy(AB_host,AB_device,p*r*sizeof(int),cudaMemcpyDeviceToHost);
     
     
      
      
      dim3 threadsPerBlocks(16, 16);
      dim3 numBlocks((r + blockDim.x - 1) / blockDim.x, (p + blockDim.y - 1) / blockDim.y);
      matrix_addition<<<numBlocks, threadsPerBlocks>>>(AB_device,CD_device,d_matrixE ,p, r);
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);
   

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
      cudaFree(CD_device);
      cudaFree(b_transpose_device);
      cudaFree(AB_device);
      
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
	
