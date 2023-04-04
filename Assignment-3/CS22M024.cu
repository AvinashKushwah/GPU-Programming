/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
 
#define BLOCK_SIZE 1024
#define max_value INT_MAX-1
using namespace std;


ofstream outfile; // The handle for printing the output

/******************************Write your kernels here ************************************/

/*

We are creating several arrays to store information about the nodes in a graph. Specifically, we have the following arrays: 
d_aid: an array of size V that stores the in-degree of each node in the graph. 
d_active: an array of size L that stores the number of active nodes at each level in the graph. 
d_act_deact: an array that stores 0 if a node is deactivated and 1 if it is activated. 
d_level: an array that stores the level of each node in the graph. 
d_lastNode: an array that stores the last node of each level in the graph. 
active: an array that stores 0 if a node is deactivated and 1 if it is activated. 
act an array that stores the last node of each layer in the graph. 
count a variable that stores the number of nodes at level 0 in the graph.
first node: a variable that stores the starting nodes of a particular layer in the graph. 
lastNode: a variable that stores the ending node of a particular layer in the graph.


*/







__global__ void Indegree_levelZero(int *d_csr, int *d_offset, int *d_apr, int L, int V, int M, int *d_aid, int *d_active,int *active,int *act,int count,int firstNode,int lastNode){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id <= lastNode){
		    active[id]=1;
        int s=d_offset[id];
        int e=d_offset[id+1];
        for(int k=s;k<e;k++)
        {
        	atomicInc((unsigned int *)&d_aid[d_csr[k]], max_value);
                atomicMax(&act[1],d_csr[k]);
        }
	}
}

__global__ void Indegree_level1toL(int *d_csr, int *d_offset, int *d_apr, int L, int V, int M, int *d_aid, int *d_active,int *active,int *act,int count,int firstNode,int lastNode,int lid){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	id+=firstNode;
	if(id<=lastNode){
		if(d_aid[id] >= d_apr[id])
		{
			active[id] =1;
		          
		}
		else
		{
			active[id]=0;
		}
		
      /*If node 'v' is not located at a corner and both its neighboring nodes, 'v-1' and 'v+1', are deactivated, then node 'v' will also be deactivated */

		if(id>firstNode && id<lastNode && d_aid[id-1] < d_apr[id-1] && d_aid[id+1] < d_apr[id+1] )
		{
		           
		   active[id]=0;
		           
		}  
		      
		int s1 = d_offset[id];
		int e1 = d_offset[id+1];
		for(int p =s1;p<e1;p++)
		{
			if(active[id])
		    {
		    	atomicInc((unsigned int *)&d_aid[d_csr[p]], max_value);
		    }
                    atomicMax(&act[lid+1],d_csr[p]);
		           
		}
	}
}


__global__ void find_lastNode_of_each_level(int *d_csr, int *d_offset, int *d_apr, int L, int V, int M, int *d_aid, int *d_active,int *active,int *act,int count) {
   


// this is to store the last nodes of each level
   
   act[0] = count-1;
   int st = 0;   
   for(int i=1;i<L-1;i++)
   {
       int end = act[i-1];
	 int first=d_offset[st],last=d_offset[end+1],lnode=-1;
	 for(int j=first;j<last;j++)
	{
		lnode=max(lnode,d_csr[j]);
	}
       //int temp = d_csr[d_offset[z+1]] -1;
       act[i] = lnode;
	 st=end+1;
   }
   act[L-1] = V-1;



}

 __global__ void activationAtEachLevel(int *d_csr, int *d_offset, int *d_apr, int L, int V, int M, int *d_aid, int *d_active,int *active,int *act,int count)
{
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0)
    {
        d_active[0] = count;
    }
    else if (tid < L)
    {
        int end = act[tid];
        int start = act[tid - 1] + 1;
        int count = 0;
        for (int i = start; i <= end; i++)
        {
            if (d_aid[i] >= d_apr[i] && active[i] == 1)
            {
                count++;
            }
        }

        d_active[tid] = count;
    }

}      



    
    
    
    
/**************************************END*************************************************/



//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
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
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // acive in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    

    // variable for result, storing number of active vertices at each level, on device
     int *d_activeVertex;
     cudaMalloc(&d_activeVertex, L*sizeof(int));
     cudaMemset(d_activeVertex, 0, V * sizeof(int)); 


/***Important***/

// Initialize d_aid array to zero for each vertex
// Make sure to use comments

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/
  memset(h_activeVertex, 0, L*sizeof(int));
  cudaMemset(d_aid, 0, V * sizeof(int));
 // int * h_act_deact;                                // store 0/1 for each vertex whether Node is activated or Deactivated, Size=V
 // h_act_deact = (int*)malloc(V*sizeof(int));
  int *d_act_deact;                          
  cudaMalloc(&d_act_deact,V*sizeof(int));
  cudaMemset(d_act_deact, 0, V * sizeof(int)); 


 
 
  int *h_lastNode=(int*)malloc(L*sizeof(int));        // store lastNode of each level, Size=L
  memset(h_lastNode,0,L*sizeof(int));
  int *d_lastNode;                          
  cudaMalloc(&d_lastNode,L*sizeof(int));
  cudaMemset(d_lastNode, 0, L * sizeof(int)); 
  
  
/*By iterating through the APR array, this loop is identifying the count of zeros present in it, which corresponds to the number of nodes located at level 0*/

   int count =0;
   for(int i=0;i<V;i++)
   {
       if(h_apr[i] != 0)
       {
           break;
       }
    count++;
   }
   
   h_lastNode[0]=count-1;
  cudaMemcpy(d_lastNode,h_lastNode,L*sizeof(int),cudaMemcpyHostToDevice);

  /*The find_lastNode_of_each_levelfunction is identifying the final node that belongs to each level*/

  int num_blocks = (L + BLOCK_SIZE - 1) / BLOCK_SIZE;
  // find_lastNode_of_each_level<<<1,1>>>(d_csrList,d_offset,d_apr,L,V,E,d_aid,d_activeVertex,d_act_deact,d_lastNode,count);
  // cudaMemcpy(h_lastNode,d_lastNode,L*sizeof(int),cudaMemcpyDeviceToHost);


/*
Running a loop that iterates from 0 to L. During each iteration, I am calculating the in-degree of every node at that level.
Additionally, I am verifying two conditions related to a game that has two rules. 
The first rule pertains to the activation of a vertex v: if AID(v) is greater than or equal to APR(v), then v will become activated. 
The second rule pertains to the deactivation of a vertex v: if vertices (v-1) and (v+1) are inactive and all three vertices (v-1), v, and (v+1) are on the same level, then v will become inactive. However, this rule only applies if both v-1 and v+1 exist.

*/
 
   for(int i=0;i<L;i++)
   {
      
      if(i==0)
      {
       
      int firstNode=0;
      int lastNode=h_lastNode[i];
      int numThread = h_lastNode[i] - 0 + 1;
      int numBlock = (numThread + BLOCK_SIZE - 1)/BLOCK_SIZE;
      Indegree_levelZero<<<numBlock, BLOCK_SIZE>>>(d_csrList,d_offset,d_apr,L,V,E,d_aid,d_activeVertex,d_act_deact,d_lastNode,count,firstNode,lastNode);
      cudaDeviceSynchronize();       
          

      }
      
      else
    {
      int lastNode=h_lastNode[i]; 
      int firstNode=h_lastNode[i-1]+1;
      int numThread = (lastNode - firstNode + 1);
      int numBlock = (numThread+BLOCK_SIZE-1)/BLOCK_SIZE;
      Indegree_level1toL<<<numBlock, BLOCK_SIZE>>>(d_csrList,d_offset,d_apr,L,V,E,d_aid,d_activeVertex,d_act_deact,d_lastNode,count,firstNode,lastNode,i);
      cudaDeviceSynchronize();


    }
    cudaMemcpy(h_lastNode,d_lastNode,L*sizeof(int),cudaMemcpyDeviceToHost);
      
   }

/*
This activationAtEachLevel is  determining the number of activated nodes at each level, To count the number of activated nodes at each level, the condition (d_aid[i] >= d_apr[i] && active[i] == 1) is checked

*/

   activationAtEachLevel<<<num_blocks,BLOCK_SIZE>>>(d_csrList,d_offset,d_apr,L,V,E,d_aid,d_activeVertex,d_act_deact,d_lastNode,count);
  // cudaMemcpy(h_level,d_level,V*sizeof(int),cudaMemcpyDeviceToHost);
   cudaMemcpy(h_activeVertex,d_activeVertex,L*sizeof(int),cudaMemcpyDeviceToHost);
   //cudaMemcpy(h_aid,d_aid,V*sizeof(int),cudaMemcpyDeviceToHost);
  // cudaMemcpy(h_act_deact,d_act_deact,V*sizeof(int),cudaMemcpyDeviceToHost);
  // cudaMemcpy(h_lastNode,d_lastNode,L*sizeof(int),cudaMemcpyDeviceToHost);

     

/********************************END OF CODE AREA**********************************/
double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);
if(argc>2)
{
    for(int i=0; i<L; i++)
    {
        printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
    }
}

    return 0;
}
