#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*
	Slide 10: Each SM can only take up to 768 threads. Ex: 256 threads/block x 3 block, or 128 x 6 blocks etc.
*/
#define TILE_WIDTH 2 //CHANGE THIS TO ARBITRARY

__global__ void matrixMult (float *a, float *b, float *c, int width);
__global__ void blankCall() {int i = 0; if (i == 0) {} }; //Dummy calls does nothing. Setting int i so it actually does something (simple). Rest because don't like warnings when compiling!

int main (int argc, char *argv[]) {
	
	/*
		Notes: floating values are set to output with 2 decimals only. Without limitation, product.dat ended up being humongous in size!
	*/

	if (argc != 2) { 
		printf( "\nError: Number of arguments incorrect.\n"
				"There can only be 1 additional argument, which is the row/column length.\n"
				"Ex: ./MatrixMult 512\n"
				"Program gracefully terminated.\n"); 
		exit(0);
	}

	FILE *f = fopen("product.dat", "w+");
	if (f == NULL) {
		printf("File could not be created/opened!\n"
				"Program gracefully terminated.\n");
		exit(1);
	}

	int i, j;
	int dim = atoi(argv[1]);
	if (dim < 0) {
		printf("Input must be a positive number!\n"
			   "Program gracefully terminated.\n");
		exit(0);
	}
	
	//CPU pointers to 2d array in heap
	float *matrix1 = (float *)malloc(dim * dim * sizeof(float));
	float *matrix2 = (float *)malloc(dim * dim * sizeof(float));
	float *result = (float *)malloc(dim * dim * sizeof(float));

	//Populating the two matrices (and printing matrix 1)
	//srand(time(NULL));
	srand48(time(NULL));
	for (i = 0; i < dim; i++) {
		for (j = 0; j < dim; j++) {
			//*(matrix1 + i*dim + j) = ((float)rand()/(float)(RAND_MAX)) * 4; //Mod for smaller, easier to hand check
			//*(matrix2 + i*dim + j) = ((float)rand()/(float)(RAND_MAX)) * 4; 
			*(matrix1 + i*dim + j) = drand48() * 2;
			*(matrix2 + i*dim + j) = drand48() * 2;
			/*
			
			2) Yes. For the random numbers, I suggest that you use either srand48()/drand48(), which is a random number 
			generator for floats and floats or cuRand() but the latter is only available for GPUs. Both are significantly 
			faster than what you’re doing.
			
			*/
			//printf("%.2f ",  *(matrix1 + i*dim + j)); //Print Matrix1
			//fprintf(f, "%.2f\t", *(matrix1 + i*dim + j));
		}
		//printf("\n");
		//fprintf(f, "\n");
	}
	

	//Print matrix2
	/*
	//printf("\n");
	fprintf(f, "\n");
	for (i = 0; i < dim; i++) {
		for (j = 0; j < dim; j++) {
			//printf("%.2f ",  *(matrix2 + i*dim + j));
			//fprintf(f, "%.2f\t", *(matrix2 + i*dim + j));
		}
		//printf("\n");
		//fprintf(f, "\n");
	}
	//printf("\n");
	//fprintf(f, "\n");
	*/
	
	
	//GPU pointers for 2d array
	float *dev_a, *dev_b, *dev_c;

	int size = dim * dim * sizeof(float);
	cudaMalloc((void **) &dev_a, size);
	cudaMalloc((void **) &dev_b, size);
	cudaMalloc((void **) &dev_c, size);

	cudaMemcpy(dev_a, matrix1, size, cudaMemcpyHostToDevice); //Ptr or dereference?
	cudaMemcpy(dev_b, matrix2, size, cudaMemcpyHostToDevice);
	
	/*
		If tile = 151, then 18 blocks. 
	*/

	//Preparing GPU call
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH); //Threads in block?
	//Blocks in grid
	//dim/threads in dim, returns number of blocks
	dim3 dimGrid((int)ceil(dim/dimBlock.x) + 1, (int)ceil(dim/dimBlock.y) + 1); //pick a block size so that threads < 768

	//Dummy call
	blankCall<<<1, 1>>>();
	
	//Setting up timer start
	clock_t start, end;
	double cpu_time;
	start = clock();
	
	//Calling GPU
	matrixMult<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, dim); //Ex: 256 threads per block. Anything less than 27 * 27
	
	//Synchronize before end
	cudaThreadSynchronize();

	//Stopping the timer
	end = clock();
	cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;
	
	//Retrieving computed data back
	cudaMemcpy(result, dev_c, size, cudaMemcpyDeviceToHost);

	//Saving the world
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	//Printing Result 3
	for (i = 0; i < dim; i++) {
		for (j = 0; j < dim; j++) { 
			//printf("%.2f ",  *(result + i*dim + j));
			fprintf(f, "%.2f\t", *(result + i*dim + j));
		}
		//printf("\n");
		fprintf(f, "\n");
	}
	fclose(f); 
	
	printf("Time: %lf\n", cpu_time);
}

__global__ void matrixMult(float* A, float* B, float* C, int width) {
	int k;
	float sum = 0;
	int col = blockIdx.x*TILE_WIDTH + threadIdx.x;
	int row = blockIdx.y*TILE_WIDTH + threadIdx.y;
	if(col < width && row < width) {
		for (k = 0; k < width; k++)
			sum += A[row * width + k] * B[k * width + col];
		C[row * width + col] = sum;
	}
}
