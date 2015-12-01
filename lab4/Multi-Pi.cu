#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

//#define Threads 768
#define Threads 768

#define readBlockSize 7

__global__ void computeHistogram(char*, int*);
__global__ void blankCall() {int i = 0; if (i == 0) {} };

int main(int argc, char *argv[]) {

	if (argc != 2) {
		printf("\nError: Number of arguments incorrect.\n"
			"There can only be 1 additional argument, which is the name of the file.\n"
			"Ex: ./Multi-Pi pi.txt\n"
			"Program gracefully terminated.\n");
		exit(0);
	}

	FILE *input = fopen(argv[1], "r+");
	FILE *output = fopen("freq.dat", "w+");
	if (input == NULL) {
		printf("Error: %s could not be read!\n"
			"Program gracefully terminated.\n", argv[1]);
		exit(1);
	}
	if (output == NULL) {
		printf("Error: freq.dat could not be created!\n"
				"Program gracefully terminated.\n");
		exit(1);
	}

	//------------------------------------------------------------------------------------//

	//Timer and Dummy Call
	blankCall<<<1, 1>>>();

	cudaStream_t stream1;
	cudaStreamCreate(&stream1);
	cudaStream_t stream2;
	cudaStreamCreate(&stream2); 

	clock_t start, end;
	double exec_time;
	start = clock();

	//===================================================

	char inputString[readBlockSize];
	int histogram[10] = {0}, count = 1;

	char *dev_inputString; 
	int *dev_histogram;
	cudaMalloc((void**)&dev_inputString, sizeof(char) * readBlockSize);
	cudaMalloc((void**)&dev_histogram, sizeof(int) * 10);
	//cudaHostAlloc((void**) &inputString, sizeof(char) * readBlockSize, cudaHostAllocDefault);
	//cudaHostAlloc((void**) &histogram, sizeof(int) * 10, cudaHostAllocDefault);

	cudaSetDevice(0);
	cudaMemcpy(dev_histogram, &histogram, 10 * sizeof(int), cudaMemcpyHostToDevice);
	cudaSetDevice(1);
	cudaMemcpy(dev_histogram, &histogram, 10 * sizeof(int), cudaMemcpyHostToDevice);

	while(fgets(inputString, readBlockSize, input) != NULL) {
		printf("\t%s\n", inputString);
		cudaSetDevice(0);
		cudaMemcpy(dev_inputString, &inputString, readBlockSize * sizeof(char), cudaMemcpyHostToDevice);
		computeHistogram<<<(int)ceil(readBlockSize / Threads) + 1, Threads>>>(dev_inputString, dev_histogram);
		// cudaMemcpyAsync(dev_inputString, &inputString, readBlockSize * sizeof(char), cudaMemcpyHostToDevice, stream1);
		// computeHistogram<<<(int)ceil(readBlockSize / Threads) + 1, Threads, stream1>>>(dev_inputString, dev_histogram);

		if (fgets(inputString, readBlockSize, input) != NULL) {
			printf("Inside Nest\n");
			cudaSetDevice(1);
			cudaMemcpy(dev_inputString, &inputString, readBlockSize * sizeof(char), cudaMemcpyHostToDevice);
			computeHistogram<<<(int)ceil(readBlockSize / Threads) + 1, Threads>>>(dev_inputString, dev_histogram);
			// cudaMemcpyAsync(dev_inputString, &inputString, readBlockSize * sizeof(char), cudaMemcpyHostToDevice, stream1);
			// computeHistogram<<<(int)ceil(readBlockSize / Threads) + 1, Threads, stream1>>>(dev_inputString, dev_histogram);
		
		}

		cudaDeviceSynchronize();

		printf("GPUs Synchronized (%i)\n", count);
		count++;

	}
	int histogram2[10] = {0};
	cudaSetDevice(0);
	cudaMemcpy(histogram, dev_histogram, 10 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaSetDevice(1);
	cudaMemcpy(histogram2, dev_histogram, 10 * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 10; i++) {
		printf("[%i]: %i\n", i, histogram[i] + histogram2[i]);
	}



	//===================================================

	//Stopping the timer
	end = clock();
	exec_time = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("\nExecution Time: %f\n", exec_time);

	fclose(input);
	fclose(output);

}

__global__ void computeHistogram(char* inputArr, int* histArr) {

	int globalID = blockDim.x * blockIdx.x + threadIdx.x;
	if (globalID >= readBlockSize)
		return;

	//Return if '.', or EOL
	if (inputArr[globalID] == '.' || inputArr[globalID] == '\0')
		return;
	printf("Reading: %c\n", inputArr[globalID]);
	atomicAdd(&histArr[inputArr[globalID] - '0'], 1);
	// printf("Current [%i] = %i\n", inputArr[globalID] - '0', histArr[inputArr[globalID] - '0']);

	// for (int i = 0; i < sizeof(inputArr)/sizeof('c') - 1; i++) {
	// 		if (inputArr[i] == '.') continue;
	// 		else if (inputArr[i] == '\0') { 
	// 			*exitFlag = 1; 
	// 			return; 
	// 		}
	// 		printf("Reading: %c, %i\n", inputArr[i], i);
	// 		histArr[inputArr[i] - '0']++;
	// 	}
}

