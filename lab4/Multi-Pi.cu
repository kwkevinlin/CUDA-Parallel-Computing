#include <stdio.h>
#include <stdlib.h>

#define Threads 1000

#define readBlockSize 10000000

/*
	A little hackish once digits of pi exceed 67~ million. See lines 80-84.
	For digits of pi lower than that, commenting out lines 81-84 should do. Leave
	line 80 uncommented so printf and fprintf from lines 135-136 will remain functional.
*/

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

	cudaSetDevice(0);
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);
	cudaSetDevice(1);
	cudaStream_t stream2;
	cudaStreamCreate(&stream2); 

	clock_t start, end;
	double exec_time;
	start = clock();

	//===================================================

	char* inputString1; char* inputString2; 
	int* histogram;
	int histogram2[10] = {0}, count = 0, count2 = 1, firstDigit; 
	char *dev_inputString1;
	char *dev_inputString2; 
	int *dev_histogram1;
	int *dev_histogram2;

	cudaHostAlloc((void**) &inputString1, sizeof(char) * readBlockSize, cudaHostAllocDefault);
	cudaHostAlloc((void**) &inputString2, sizeof(char) * readBlockSize, cudaHostAllocDefault);
	cudaHostAlloc((void**) &histogram, sizeof(int) * 10, cudaHostAllocDefault);
	for (int i = 0; i < 10; i++) {
		histogram[i] = 0;
		histogram2[i] = 0;
	}

	cudaSetDevice(0);
		blankCall<<<1, 1>>>();
		cudaMalloc((void**)&dev_inputString1, sizeof(char) * readBlockSize);
		cudaMalloc((void**)&dev_histogram1, sizeof(int) * 10);
		cudaMemcpy(dev_histogram1, histogram, 10 * sizeof(int), cudaMemcpyHostToDevice);
	cudaSetDevice(1);
		blankCall<<<1, 1>>>();
		cudaMalloc((void**)&dev_inputString2, sizeof(char) * readBlockSize);
		cudaMalloc((void**)&dev_histogram2, sizeof(int) * 10);
		cudaMemcpy(dev_histogram2, histogram, 10 * sizeof(int), cudaMemcpyHostToDevice);

	int tmp[12] = {0};
	fgets(inputString1, 12, input); //12
	firstDigit = inputString1[0] - '0';
	for (int i = 0; i < 11; i++) {
		tmp[inputString1[i] - '0']++;
		printf("%i ", inputString1[i] - '0');
	}

	// for (int i = 0; i < 5; i++) {
	// 	printf("Run %i\n", i+1);
	// 	fgets(inputString1, readBlockSize, input);
	// 	cudaSetDevice(0);
	// 	cudaMemcpyAsync(dev_inputString1, inputString1, readBlockSize * sizeof(char), cudaMemcpyHostToDevice, stream1);
	// 	computeHistogram<<<(int)ceil(readBlockSize / Threads) + 1, Threads, 0, stream1>>>(dev_inputString1, dev_histogram1);
		
	// 	fgets(inputString2, readBlockSize, input);
	// 	cudaSetDevice(1);
	// 	cudaMemcpyAsync(dev_inputString2, inputString2, readBlockSize * sizeof(char), cudaMemcpyHostToDevice, stream2);
	// 	computeHistogram<<<(int)ceil(readBlockSize / Threads) + 1, Threads, 0, stream2>>>(dev_inputString2, dev_histogram2);
	// }


	while(fgets(inputString1, readBlockSize, input) != NULL) {
		cudaSetDevice(0);
		// cudaMemcpy(dev_inputString1, inputString1, readBlockSize * sizeof(char), cudaMemcpyHostToDevice);
		// computeHistogram<<<(int)ceil(readBlockSize / Threads) + 1, Threads>>>(dev_inputString1, dev_histogram1);
		cudaMemcpyAsync(dev_inputString1, inputString1, readBlockSize * sizeof(char), cudaMemcpyHostToDevice, stream1);
		computeHistogram<<<(int)ceil(readBlockSize / Threads) + 1, Threads, 0, stream1>>>(dev_inputString1, dev_histogram1);
		printf("GPU-0 (%i)\n", count); count = count + readBlockSize;

		if (fgets(inputString2, readBlockSize, input) != NULL) {
			cudaSetDevice(1);
			// cudaMemcpy(dev_inputString2, inputString2, readBlockSize * sizeof(char), cudaMemcpyHostToDevice);
			// computeHistogram<<<(int)ceil(readBlockSize / Threads) + 1, Threads>>>(dev_inputString2, dev_histogram2);
			cudaMemcpyAsync(dev_inputString2, inputString2, readBlockSize * sizeof(char), cudaMemcpyHostToDevice, stream2);
			computeHistogram<<<(int)ceil(readBlockSize / Threads) + 1, Threads, 0, stream2>>>(dev_inputString2, dev_histogram2);
			printf("GPU-1 (%i)\n", count); count = count + readBlockSize;
		}
		// printf("GPUs Synchronized (%i)\n", count);
		// count++;
		
		cudaDeviceSynchronize();

	}


	//This can be made async too
	cudaSetDevice(0);
	cudaMemcpy(histogram, dev_histogram1, 10 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaSetDevice(1);
	cudaMemcpy(histogram2, dev_histogram2, 10 * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 10; i++) {
		printf("[%i]: %i + %i + %i = %i\n", i, histogram[i], histogram2[i], tmp[i], histogram[i] + histogram2[i] + tmp[i]);
		fprintf(output, "%i\t%i\t%i\n", histogram[i], histogram2[i], histogram[i] + histogram2[i] + tmp[i]);
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
	if (globalID >= readBlockSize) {
		//printf("Returning for ID: %i", globalID);
		return;
	}

	//Return if '.', or EOL
	if (inputArr[globalID] == '\0')
		return;

	atomicAdd(&histArr[inputArr[globalID] - '0'], 1);
}

