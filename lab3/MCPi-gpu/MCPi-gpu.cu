#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define Threads 768
//#define Threads 5 //Threads per block in 1D

__global__ void init (unsigned int, curandState_t*);
__global__ void randoms (curandState_t*, int*, int*);

int main (int argc, char *argv[]) {

	if (argc != 2) { 
		printf( "\nError: Number of arguments incorrect.\n"
				"There can only be 1 additional argument, which is the number of random points.\n"
				"Ex: ./MCPi-cpu 100\n"
				"Program gracefully terminated.\n"); 
		exit(0);
	}

	FILE *f = fopen("freq.dat", "w+");
	if (f == NULL) {
		printf("File could not be created/opened!\n"
				"Program gracefully terminated.\n");
		exit(1);
	}

	int points = atoi(argv[1]);
	if (points < 0) {
		printf("Input must be a positive number!\n"
			   "Program gracefully terminated.\n");
		exit(0);
	}
	
	//------------------------------------------------------------------------------------------
	
	//Starting curand_unit since it's slower
	curandState_t* dev_states; //keep track of seed value for every thread
	cudaMalloc((void**) &dev_states, points * sizeof(curandState_t)); //N
	//initialize all of the random states on the GPU
	init<<<(int)ceil(points/Threads) + 1, Threads>>>(time(NULL), dev_states);
	
	//------------------------------------------------------------------------------------------
	
	//Store in heap cause number of tries can be huge
	int *count = (int *)malloc(points * sizeof(int)); 
	int *normX = (int *)malloc(points * sizeof(int)); 
	int *normY = (int *)malloc(points * sizeof(int)); 
	int *dev_count, *dev_points;
	cudaMalloc((void**) &dev_count, points * sizeof(int));
	cudaMalloc((void**) &dev_points, sizeof(int));
	
	cudaMemcpy(dev_points, &points, sizeof(int), cudaMemcpyHostToDevice);
	
	randoms<<<(int)ceil(points/Threads) + 1, Threads>>>(dev_states, dev_points, dev_count); //Ceil does not round up, +1
	
	cudaThreadSynchronize();
	
	cudaMemcpy(count, dev_count, points * sizeof(int), cudaMemcpyDeviceToHost);
	
	//------------------------------------------------------------------------------------------
	
	cudaFree(dev_states);
	cudaFree(dev_count);
	cudaFree(dev_points);

	/*
		Issues to take note:
		1. Result tends to be a little smaller than 3.14, ie 3.8, 3.12
		2. Over certain attemps size (ie 30,000), [index] starts getting HUGE (non 1 or 0) numbers
	
	*/
	
	printf("\n");
	int i, hits = 0;
	for (i = 0; i < points; i++) {
		//printf("[%i] = %i\n", i, count[i]);
		if (count[i] > 1)
			printf("[%i] = %i\n", i, count[i]);
		hits = hits + count[i];
		//fprintf(f, "%0.1f\t%i\n", i * 0.1, histogram[i]);
	}
	
	printf("Hits: %i\n", hits);
	printf("Pi: %0.2f\n", 4.0 * ((float)hits/(float)points));
	
	fclose(f);
	
}

__global__ void randoms (curandState_t* states, int* totalPoints, int* count) { 
	int globalId = blockDim.x * blockIdx.x + threadIdx.x;
	if (globalId > *totalPoints) { //Return if extra threads in end of operation
		//printf("%i\t\t\t", globalId);
		//printf("out\t\t\tout\n");
		return;
	}
	printf("%i, %i\n", globalId, *totalPoints);
	
	float x = curand_uniform(&states[globalId]); //*2
	float y = curand_uniform(&states[globalId]);
	int normX = (int)floor(x*10);
	int normY = (int)floor(y*10);
	printf("%0.5f(%0.1f), %0.5f(%0.1f)\n", x, normX*0.1, y, normY*0.1);
	
	//Math explanation in C code section
	float t = (x - 1.0)*(x - 1.0) + (y - 1.0)*(y - 1.0);
	if (1.0 >= t)  //if radius > t
		count[globalId] = 1;
	else 
		count[globalId] = 0; //Easier to add up in main(), and to avoid memCpy before kernel call
	
}

// kernel to initialize the random states
__global__ void init (unsigned int seed, curandState_t* states) {
	int globalId = blockDim.x * blockIdx.x + threadIdx.x;
	//globaId -> threadIdx.x
	curand_init(seed, // same seed for each core send from host
	globalId, // sequence number; different for each core
	0, // offset
	&states[globalId]);
} 




