#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <algorithm>
// #include <bits/stdc++.h>

#define max_N 100000
#define max_P 30
#define BLOCK_SIZE 1024
#define CAP 30*24+1

using namespace std;

//*******************************************

struct node{
  int id, cen, fac, start, slots;
};

bool comparator(struct node &n1, struct node &n2)
{
  return (n1.cen<n2.cen) || ((n1.cen == n2.cen) && (n1.id < n2.id));
}


/* 

initAOS: Initializes an array of structs in device memory (d_AOS) with data from several arrays (req_id, req_cen, req_fac, req_start, req_slots).
Each thread in the CUDA kernel is responsible for initializing one struct element of the array 

*/



 __global__ void initAOS(struct node *d_AOS, int *req_id, int *req_cen, int *req_fac, int *req_start, int *req_slots, int StructSize) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < StructSize)
    {
        d_AOS[idx].id = req_id[idx];
        d_AOS[idx].cen = req_cen[idx];
        d_AOS[idx].fac = req_fac[idx];
        d_AOS[idx].start = req_start[idx];
        d_AOS[idx].slots = req_slots[idx];
    }
}


// copyStructValues: This kernel function copies the values of a struct array (AOS - array of structures) to individual arrays.

 __global__ void copyStructValues(struct node *d_AOS, int *req_id, int *req_cen, int *req_fac, int *req_start, int *req_slots, int StructSize) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < StructSize) {
        req_slots[idx] = d_AOS[idx].slots;
        req_start[idx] = d_AOS[idx].start;
        req_fac[idx] = d_AOS[idx].fac;
        req_cen[idx] = d_AOS[idx].cen;
        req_id[idx] = d_AOS[idx].id;
    }
}


/* 

FacilityBooking:
->Implements parallel processing of user requests for computer facilities.
->The kernel receives several input arrays and outputs the total number of successful and failed requests and the number of successful and failed requests for each computer centre.
->The kernel uses a prefix sum algorithm to efficiently determine the requests' indices corresponding to each computer centre.
->For each request, the kernel checks the availability of the requested facility during the requested time slots and updates the availability schedule if the facility is available.
->If the facility is not available, the request is marked as a failure.

*/

__global__ void FacilityBooking(int *req_cen, int *req_fac, int *capacity,int *req_start, int *req_slots,int N, int *reqPrefixSum, int *totalRequestComWise, int *total, int *success, int *facPrefixArray){
  
  int time[CAP];
  int index = blockIdx.x * blockDim.x + threadIdx.x; 

  

  if(index<N){
  // printf("Inside Kernel");


    // Initialize the time array to 0
    int i=0;
    while(i<CAP){
      time[i]=0;
      i++;
    }

    // printf("Computer Center - %d, ID Range - from %d to %d places \n", id, reqPrefixSum[id], totalRequestComWise[id]);
    // printf("Comp %d :: Range -> %d to %d\n", id, reqPrefixSum[id], reqPrefixSum[id] + totalRequestComWise[id] - 1);

    // Process each request
    i=reqPrefixSum[index];
    while(i<=(reqPrefixSum[index] + totalRequestComWise[index] - 1)){
      total[index]++;

      // printf("R");
      int check = 0;

      int effFacilityID = facPrefixArray[req_cen[i]]+req_fac[i];
      int temp = capacity[effFacilityID]; 
      int cap = temp;

      
     // Check if the facility is available during the requested time slots
      int j=req_fac[i]*24+req_start[i]; 
      while(j<(req_fac[i]*24+req_start[i]+req_slots[i])){
        if(time[j]<cap){
          j++;
          continue;
        }else{
          check = 1;
          break;
        }
      }

      if(check == 1){
        check = 0;
        i++;
        continue;
      }
      

      // If the facility is not available, mark the request as a failure and move on to the next request
      int k = req_fac[i]*24+req_start[i];
      while(k<(req_fac[i]*24+req_start[i]+req_slots[i])){
        time[k]++;
        k++;
      }

      success[index]++;
      i++;
    }
  }
}

//   printf("%d ", totalRequestComWise[id]);


//***********************************************






int main(int argc,char **argv)
{
	// variable declarations...
    int N,*centre,*facility,*capacity,*fac_ids, *succ_reqs, *tot_reqs;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &N ); // N is number of centres
	
    // Allocate memory on cpu
    centre=(int*)malloc(N * sizeof (int));  // Computer  centre numbers
    facility=(int*)malloc(N * sizeof (int));  // Number of facilities in each computer centre
    fac_ids=(int*)malloc(max_P * N  * sizeof (int));  // Facility room numbers of each computer centre
    capacity=(int*)malloc(max_P * N * sizeof (int));  // stores capacities of each facility for every computer centre 


    int success=0;  // total successful requests
    int fail = 0;   // total failed requests
    tot_reqs = (int *)malloc(N*sizeof(int));   // total requests for each centre
    succ_reqs = (int *)malloc(N*sizeof(int)); // total successful requests for each centre

    // Input the computer centres data
    int k1=0 , k2 = 0;
    for(int i=0;i<N;i++)
    {
      fscanf( inputfilepointer, "%d", &centre[i] );
      fscanf( inputfilepointer, "%d", &facility[i] );
      
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &fac_ids[k1] );
        k1++;
      }
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &capacity[k2]);
        k2++;     
      }
    }

    // variable declarations
    int *req_id, *req_cen, *req_fac, *req_start, *req_slots;   // Number of slots requested for every request
    
    // Allocate memory on CPU 
	int R;
	fscanf( inputfilepointer, "%d", &R); // Total requests
    req_id = (int *) malloc ( (R) * sizeof (int) );  // Request ids
    req_cen = (int *) malloc ( (R) * sizeof (int) );  // Requested computer centre
    req_fac = (int *) malloc ( (R) * sizeof (int) );  // Requested facility
    req_start = (int *) malloc ( (R) * sizeof (int) );  // Start slot of every request
    req_slots = (int *) malloc ( (R) * sizeof (int) );   // Number of slots requested for every request
    
    // Input the user request data
    for(int j = 0; j < R; j++)
    {
       fscanf( inputfilepointer, "%d", &req_id[j]);
       fscanf( inputfilepointer, "%d", &req_cen[j]);
       fscanf( inputfilepointer, "%d", &req_fac[j]);
       fscanf( inputfilepointer, "%d", &req_start[j]);
       fscanf( inputfilepointer, "%d", &req_slots[j]);
       tot_reqs[req_cen[j]]+=1;  
    }
		
    //*********************************
    

   struct node *AOS;
   int StructSize = R; 
   AOS = (struct node *) malloc (StructSize * sizeof(struct node));
   struct node * d_AOS;
   cudaMalloc(&d_AOS, StructSize * sizeof(struct node));
    
    
   int *d_req_id, *d_req_cen, *d_req_fac, *d_req_start, *d_req_slots;
   cudaMalloc(&d_req_id, StructSize * sizeof(int));
   cudaMemcpy(d_req_id, req_id, StructSize * sizeof(int), cudaMemcpyHostToDevice);
   cudaMalloc(&d_req_cen, StructSize * sizeof(int));
   cudaMemcpy(d_req_cen, req_cen, StructSize * sizeof(int), cudaMemcpyHostToDevice);
   cudaMalloc(&d_req_start, StructSize * sizeof(int));
   cudaMemcpy(d_req_start, req_start, StructSize * sizeof(int), cudaMemcpyHostToDevice);
   cudaMalloc(&d_req_slots, StructSize * sizeof(int));
   cudaMemcpy(d_req_slots, req_slots, StructSize * sizeof(int), cudaMemcpyHostToDevice);  
   cudaMalloc(&d_req_fac, StructSize * sizeof(int));
   cudaMemcpy(d_req_fac, req_fac, StructSize * sizeof(int), cudaMemcpyHostToDevice);
   

  /*---------------------------------------------------kernel Launch-1-------------------------------------------------------------------- */

  
   int grid_size = (StructSize + BLOCK_SIZE - 1) / BLOCK_SIZE;    // StructSize = R = number of threads
   initAOS<<<grid_size, BLOCK_SIZE>>>(d_AOS, d_req_id, d_req_cen, d_req_fac, d_req_start, d_req_slots, StructSize); 
   cudaDeviceSynchronize();
   cudaMemcpy(AOS, d_AOS, StructSize * sizeof(struct node), cudaMemcpyDeviceToHost);

   // Sort function sorts the AOS array in ascending order based on the "cen" member variable of the node structure. 
   // If multiple node structures have the same "cen" value, then the sorting is based on the "id" member variable.
   
   sort(AOS, AOS+R, comparator);
   cudaMemcpy(d_AOS,AOS, StructSize * sizeof(struct node), cudaMemcpyHostToDevice);


  
   /*---------------------------------------------------kernel Launch-2-------------------------------------------------------------------- */

   copyStructValues<<<grid_size, BLOCK_SIZE>>>(d_AOS, d_req_id, d_req_cen, d_req_fac, d_req_start, d_req_slots, StructSize); 
   cudaDeviceSynchronize();
   cudaMemcpy(req_id,d_req_id, StructSize * sizeof(int), cudaMemcpyDeviceToHost);
   cudaMemcpy(req_slots,d_req_slots, StructSize * sizeof(int), cudaMemcpyDeviceToHost); 
   cudaMemcpy(req_cen,d_req_cen, StructSize * sizeof(int), cudaMemcpyDeviceToHost);
   cudaMemcpy(req_start,d_req_start, StructSize * sizeof(int), cudaMemcpyDeviceToHost);   
   cudaMemcpy(req_fac,d_req_fac, StructSize * sizeof(int), cudaMemcpyDeviceToHost);


    // The prefix sum of the facility array is computed here in order to determine the index of the corresponding facility room number and the index of its capacity array.
    int *facPrefixArray,*d_facPrefixArray;
    facPrefixArray = (int *) malloc(N*sizeof(int));
    cudaMalloc(&d_facPrefixArray, N*sizeof(int));

    int temp = 0;
    int i=0;
    while(i<N){
      facPrefixArray[i]=temp;
      temp+=facility[i];
      i++;
    }

    cudaMemcpy(d_facPrefixArray, facPrefixArray, N*sizeof(int), cudaMemcpyHostToDevice);

    // Count the Number of requests for each Computer Centre.
    int *totalRequestComWise,*d_totalRequestComWise;  
    totalRequestComWise = (int*)malloc(N * sizeof (int));
    memset(totalRequestComWise, 0, N*sizeof(int));

    i=0;
    while(i<R){
      totalRequestComWise[req_cen[i]]++;
      i++;
    }
    
    cudaMalloc(&d_totalRequestComWise,N*sizeof(int));
    cudaMemcpy(d_totalRequestComWise, totalRequestComWise, N*sizeof(int), cudaMemcpyHostToDevice);


    // prefix sum of the count of requests for each computer centre 

    int *reqPrefixSum;
    reqPrefixSum = (int*)malloc(N * sizeof (int));
    reqPrefixSum[0]=0;

    i=1;
    while(i<N){
      reqPrefixSum[i] = totalRequestComWise[i-1]+reqPrefixSum[i-1];
      i++;
    }

    int *d_reqPrefixSum,*d_capacity;
    cudaMalloc(&d_reqPrefixSum,N*sizeof(int));
    cudaMemcpy(d_reqPrefixSum, reqPrefixSum, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_capacity,max_P * N*sizeof(int));
    cudaMemcpy(d_capacity, capacity, max_P * N*sizeof(int), cudaMemcpyHostToDevice);
     
    // Initialize a device array named "total" to 0 and then declare another device array named "success" and also initialize it to 0.
    int *d_total, *d_success;
    cudaMalloc(&d_total,N*sizeof(int));
    cudaMalloc(&d_success,N*sizeof(int));
    cudaMemset(d_total, 0, N*sizeof(int));
    cudaMemset(d_success, 0, N*sizeof(int));
     
   /*--------------------------------------------------------kernel Launch-3---------------------------------------------------------------- */
    
   // The FacilityBooking Kernel processes each booking request and checks if the requested facility is available during the requested time slots.
   // If the facility is available, the function marks the request as successful and updates the time array to reflect the booked slots.
   // If the facility is not available, the function marks the request as a failure and moves on to the next request.    


    int blocks = (N + BLOCK_SIZE -1)/BLOCK_SIZE;          // N is number of threads
    FacilityBooking<<<blocks,BLOCK_SIZE>>>(d_req_cen, d_req_fac, d_capacity, d_req_start, d_req_slots,N, d_reqPrefixSum, d_totalRequestComWise, d_total, d_success, d_facPrefixArray);
    cudaDeviceSynchronize();
    cudaMemcpy(tot_reqs, d_total, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(succ_reqs, d_success, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    i=0,fail= R;
    while(i<N){
      success = success + succ_reqs[i];
      fail = fail-succ_reqs[i];
      i++;
    }


   //**********************************************************************************************************************************************

    // Output
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    fprintf( outputfilepointer, "%d %d\n", success, fail);
    for(int j = 0; j < N; j++)
    {
        fprintf( outputfilepointer, "%d %d\n", succ_reqs[j], tot_reqs[j]-succ_reqs[j]);
    }
    fclose( inputfilepointer );
    fclose( outputfilepointer );
    cudaDeviceSynchronize();
	return 0;
}