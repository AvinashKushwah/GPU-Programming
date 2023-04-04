/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-1
 * Description: Computation of a matrix C = Kronecker_prod(A, B.T)
 *              where A and B are matrices of dimension (m, n) and
 *              the output is of the dimension (m * n, m * n). 
 * Note: All lines marked in --> should be replaced with code. 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
using namespace std;

ofstream outfile; // The handle for printing the output

__global__ void per_row_AB_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    // --> Complete the kernel ....



long int x = blockIdx.x* n;
 long int y = threadIdx.x *n;
   for(long int i=0;i<n;i++){
       
    for(long int j=0;j<n;j++)
    {

        long int Id = blockIdx.x *(m*n*n) + (m*n)*j + i*m + threadIdx.x;
        if(Id< m*n*m*n){

        C[Id] = A[x+i]*B[y+j];
      }
    }

   }

}

__global__ void per_column_AB_kernel(long int *A, long int *B, long int *C,long int m, long int n){
  
   /*
   long int blockId = blockIdx.x;          
  
   for(long int i=0;i<m;i++)
   {
       for(long int j=0;j<m;j++)
       {
            //long int id =  i*m*n*n + j + blockIdx.x*blockDim.x*blockDim.y;
           
            long int col = threadIdx.x + threadIdx.y * blockDim.x; 
           C[blockId *m + i*m*n*n + j + col*m*n] = A[blockId + i*n] * B[j*n +col];
           

       }
   }
   */

//


 long int blockId = blockIdx.x;
  
   for(long int i=0;i<m;i++)
   {
       for(long int j=0;j<m;j++)
       {
            //long int id =  i*m*n*n + j + blockIdx.x*blockDim.x*blockDim.y;
           
            int col = threadIdx.x + threadIdx.y * blockDim.x; // column number in a matrix B.
            long int Id = blockId *m + i*m*n*n + j + col*m*n; // Id is calculated to Find out where the multiplcation Of cij = aij * bij lies.
        
          if(( Id< m*n*m*n)&&(col<n))
          {
              C[Id] = A[blockId + i*n] * B[j*n +col];
          }
           
           

       }
   }


}

__global__ void per_element_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    // --> Complete the kernel ....


       int blockId = blockIdx.y*gridDim.x + blockIdx.x;
       int Id = blockId *(blockDim.x *blockDim.y) + threadIdx.y + threadIdx.x* blockDim.y;
       int temp1 = Id / (m*n);
       int temp2 = Id %(m*n);


       long int i1 = temp1/n;
       long int j1 = temp2/m;
       long int i2= temp2%m;
       long int j2 = temp1%n;
       
      int Id1 = i1*n + j1;
      int Id2= i2*n + j2;
    if(Id< m*n*n*m)
    {
      
        C[Id] = A[Id1] * B[Id2];
    }

}

/**
 * Prints any 1D array in the form of a matrix
 **/
void printMatrix(long int *arr, long int rows, long int cols, char* filename){
    outfile.open(filename);
    for(long int i = 0; i < rows; i++){
        for(long int j = 0; j < cols; j++){
            outfile<<arr[i * cols + j]<<" ";
        }
        outfile<<"\n";
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    long int m,n;	
    cin>>m>>n;	

    // Host_arrays 
    long int *h_a,*h_b,*h_c;

    // Device arrays 
    long int *d_a,*d_b,*d_c;
	
    // Allocating space for the host_arrays 
    h_a = (long int *) malloc(m * n * sizeof(long int));
    h_b = (long int *) malloc(m * n * sizeof(long int));	
    h_c = (long int *) malloc(m * m * n * n * sizeof(long int));	

    // Allocating memory for the device arrays 
    // --> Allocate memory for A on device 
    // --> Allocate memory for B on device 
    // --> Allocate memory for C on device 


   cudaMalloc(&d_a,m*n*sizeof(long int));
   cudaMalloc(&d_b,m*n*sizeof(long int));
   cudaMalloc(&d_c,m*n*m*n*sizeof(long int));



    // Read the input matrix A 
    for(long int i = 0; i < m * n; i++) {
        cin>>h_a[i];
    }

    //Read the input matrix B 
    for(long int i = 0; i < m * n; i++) {
        cin>>h_b[i];
    }

    // Transfer the input host arrays to the device 
    // --> Copy A from Host to Device
    // --> Copy B from Host to Device 

   cudaMemcpy(d_a,h_a,m*n*sizeof(long int),cudaMemcpyHostToDevice);
   cudaMemcpy(d_b,h_b,m*n*sizeof(long int),cudaMemcpyHostToDevice);


    long int gridDimx, gridDimy;
    
    // Launch the kernels
    /**
     * Kernel 1 - per_row_AB_kernel
     * To be launched with 1D grid, 1D block
     * Each thread should process a complete row of A, B
     **/

    // --> Set the launch configuration 

    double starttime = rtclock();  

    // --> Launch the kernel 




   per_row_AB_kernel<<<m,m>>>(d_a,d_b,d_c,m,n); //The assignment of the rows of matrix A to unique blocks is carried out, with each block having m threads. The rows of matrix B are each assigned to a unique thread within these blocks, and the index of C is computed to obtain the Kronecker matrix. 

    cudaDeviceSynchronize();                                                           

    double endtime = rtclock(); 
	printtime("GPU Kernel-1 time: ", starttime, endtime);  

    // --> Copy C from Device to Host 

   cudaMemcpy(h_c,d_c,m*n*m*n*sizeof(long int),cudaMemcpyDeviceToHost);

    printMatrix(h_c, m * n, m * n,"kernel1.txt");
    cudaMemset(d_c, 0, m * n * m * n * sizeof(long int));

    /**
     * Kernel 2 - per_column_AB_kernel
     * To be launched with 1D grid, 2D block
     * Each thread should process a complete column of  A, B
     **/
    
    // --> Set the launch configuration 

    starttime = rtclock(); 
    dim3 block2(64,16,1);
   per_column_AB_kernel<<<n,block2>>>(d_a,d_b,d_c,m,n); // The columns of matrix A are assigned to unique blocks. Given that the threads are organized in a 2D manner (64,16,1), each thread is allocated to a specific column of matrix B. The Kronecker matrix is then determined by computing the index of matrix C.
    
    // --> Launch the kernel 
    cudaDeviceSynchronize(); 
    
    endtime = rtclock(); 
  	printtime("GPU Kernel-2 time: ", starttime, endtime);  

    // --> Copy C from Device to Host
    cudaMemcpy(h_c,d_c,m*n*m*n*sizeof(long int),cudaMemcpyDeviceToHost);


    printMatrix(h_c, m * n, m * n,"kernel2.txt");
    cudaMemset(d_c, 0, m * n * m * n * sizeof(long int));

    /**
     * Kernel 3 - per_element_kernel
     * To be launched with 2D grid, 2D block
     * Each thread should process one element of the output 
     **/
     gridDimx = ceil(float(n * n) / 16);
     gridDimy = ceil(float(m * m) / 64);
    dim3 grid3(gridDimx,gridDimy,1);
    dim3 block3(64,16,1);

    starttime = rtclock();  

    // --> Launch the kernel

   per_element_kernel<<<grid3,block3>>>(d_a,d_b,d_c,m,n); 
    
 
    cudaDeviceSynchronize();                                                              

    endtime = rtclock();  
	printtime("GPU Kernel-3 time: ", starttime, endtime);  

    // --> Copy C from Device to Host
    cudaMemcpy(h_c,d_c,m*n*m*n*sizeof(long int),cudaMemcpyDeviceToHost);

    printMatrix(h_c, m * n, m * n,"kernel3.txt");

    return 0;
}
