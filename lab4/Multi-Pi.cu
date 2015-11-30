#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define Threads 768

// __global__ void update(float, float*, float*, float*, float*);
__global__ void computeHistogram(int);

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
	clock_t start, end;
	double exec_time;
	start = clock();

	//===================================================



	char inputString[5];
	int histogram[10] = {0};
	int exitFlag = 0;

	while(fgets(inputString, 5, input) != NULL) {
		printf("%s\n", inputString);
		for (int i = 0; i < sizeof(inputString)/sizeof('c') - 1; i++) {
			if (inputString[i] == '.') continue;
			else if (inputString[i] == '\0') { 
				exitFlag = 1; 
				break; 
			}
			printf("Reading: %c\n", inputString[i]);
			histogram[inputString[i] - '0']++;
		}
		if (exitFlag == 1)
			break;
	}

	for (int i = 0; i < 10; i++) {
		printf("[%i]: %i\n", i, histogram[i]);
	}



	//===================================================

	//Stopping the timer
	end = clock();
	exec_time = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("\nExecution Time: %f\n", exec_time);


	fclose(input);
	fclose(output);

}

__global__ void computeHistogram(int histArr[10]) {
	
}

