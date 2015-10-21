#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define Threads 768

__global__ void init (unsigned int, curandState_t*);
__global__ void randoms (curandState_t*, int*, int*, int*, int*);

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
	int *hits = (int *)malloc(points * sizeof(int)); 
	int *normX = (int *)malloc(points * sizeof(int)); 
	int *normY = (int *)malloc(points * sizeof(int)); 
	int *dev_hits, *dev_attempts, *dev_normX, *dev_normY; //dev_attempts = total number of attempts (aka var points)
	cudaMalloc((void**) &dev_hits, points * sizeof(int)); //dev_hits = array to store hits/misses
	cudaMalloc((void**) &dev_attempts, sizeof(int));
	cudaMalloc((void**) &dev_normX, points * sizeof(int));
	cudaMalloc((void**) &dev_normY, points * sizeof(int));
	
	cudaMemcpy(dev_attempts, &points, sizeof(int), cudaMemcpyHostToDevice);
	
	randoms<<<(int)ceil(points/Threads) + 1, Threads>>>(dev_states, dev_attempts, dev_hits, dev_normX, dev_normY); //Ceil does not round up, +1
	
	cudaThreadSynchronize();
	
	cudaMemcpy(hits, dev_hits, points * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(normX, dev_normX, points * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(normY, dev_normY, points * sizeof(int), cudaMemcpyDeviceToHost);
	
	//------------------------------------------------------------------------------------------
	
	cudaFree(dev_states);
	cudaFree(dev_hits);
	cudaFree(dev_attempts);
	cudaFree(dev_normX);
	cudaFree(dev_normY);
	
	//Parsing histogram data
	int i, histogram[10] = {0};
	for (i = 0; i < points; i++) {
		histogram[normX[i]]++;
		histogram[normY[i]]++;
	}
	for (i = 0; i < 10; i++) {
		//printf("%0.1f\t%i\n", i * 0.1, histogram[i]);
		fprintf(f, "%0.1f\t%i\n", i * 0.1, histogram[i]);
	}
	
	//Print values
	int sumHits = 0;
	for (i = 0; i < points; i++) {
		//printf("[%i] = %i\n", i, hits[i]);
		if (hits[i] > 1) //Error check. Fixed.
			printf("[%i] = %i\n", i, hits[i]);
		sumHits = sumHits + hits[i];
	}
	
	//printf("Hits: %i\n", sumHits);
	printf("Pi: %0.3f\n", 4.0 * ((float)sumHits/(float)points));
	
	fclose(f);
	
}

__global__ void randoms (curandState_t* states, int* totalPoints, int* hits, int* normX, int* normY) { 
	int globalId = blockDim.x * blockIdx.x + threadIdx.x;
	if (globalId > *totalPoints) { //Return if extra threads in end of operation
		//printf("%i\t\t\t", globalId);
		//printf("out\t\t\tout\n");
		return;
	}
	//printf("%i, %i\n", globalId, *totalPoints);
	
	float x = curand_uniform(&states[globalId]); //*2
	float y = curand_uniform(&states[globalId]);
	normX[globalId] = (int)floor(x*10);
	normY[globalId] = (int)floor(y*10);	
	//printf("%0.5f(%0.1f), %0.5f(%0.1f)\n", x, normX*0.1, y, normY*0.1);
	
	//Math explanation in C code section
	float t = (x - 1.0)*(x - 1.0) + (y - 1.0)*(y - 1.0);
	if (1.0 >= t)  //if radius > t
		hits[globalId] = 1;
	else 
		hits[globalId] = 0; //Easier to add up in main(), and to avoid memCpy before kernel call
	
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




