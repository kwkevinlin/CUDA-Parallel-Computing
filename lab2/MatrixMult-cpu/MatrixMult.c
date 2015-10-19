#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void matrixMult(float*, float*, float*, int);

int main (int argc, char *argv[]) {
	
	/*
		Notes: Floating values are set to output with 2 decimals only, to synchronize with the GPU function
		call in MatrixMult.cu. Without limitations in GPU, product.dat ended up being humongous in size!
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

	int dim = atoi(argv[1]);
	if (dim < 0) {
		printf("Input must be a positive number!\n"
			   "Program gracefully terminated.\n");
		exit(0);
	}
	int i, j, k;

	float *matrix1 = (float *)malloc(dim * dim * sizeof(float));
	float *matrix2 = (float *)malloc(dim * dim * sizeof(float));
	float *result = (float *)malloc(dim * dim * sizeof(float));

	//srand48(time(NULL));
	srand(time(NULL));

	//Populating the two matrices (and printing matrix 1)
	for (i = 0; i < dim; i++) {
		for (j = 0; j < dim; j++) {
			//*(matrix1 + i*dim + j) = ((float)rand()/(float)(RAND_MAX)) * 4; //Mod for smaller, easier to hand check
			//*(matrix2 + i*dim + j) = ((float)rand()/(float)(RAND_MAX)) * 4; 
			*(matrix1 + i*dim + j) = drand48() * 4;
			*(matrix2 + i*dim + j) = drand48() * 4;
			//printf("%.2f ",  *(matrix1 + i*dim + j));
			//fprintf(f, "%.2f ",  *(matrix1 + i*dim + j));
		}
		//printf("\n");
		//fprintf(f, "\n");
	}

	//Print matrix2
	//printf("\n");
	/*
	fprintf(f, "\n");
	for (i = 0; i < dim; i++) {
		for (j = 0; j < dim; j++) {
			//printf("%.2f ",  *(matrix2 + i*dim + j));
			fprintf(f, "%.2f ",  *(matrix2 + i*dim + j));
		}
		//printf("\n");
		fprintf(f, "\n");
	}
	//printf("\n");
	fprintf(f, "\n");
	*/
	
	//Timer start
	clock_t start, end;
	double cpu_time;
	start = clock();

	//Sequential Matrix Multiplication O(N^3)
	matrixMult(matrix1, matrix2, result, dim);
	
	//Timer end
	end = clock();
	cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;


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

void matrixMult(float* matrix1, float* matrix2, float* result, int dim) {
	float sum, m, n;
	int i, j, k;
	for (i = 0; i < dim; i++) {
		for (j = 0; j < dim; j++) {
			sum = 0;
			for (k = 0; k < dim; k++) {
				m =  *(matrix1 + i*dim + k);
				n =  *(matrix2 + k*dim + j);
				sum += m * n;
			}
			 *(result + i*dim + j) = sum;
		}
	}
}
