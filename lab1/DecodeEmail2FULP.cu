#include <stdio.h>

__global__ void decode(char*);

int main (int argc, char *argv[]) {
	if (argc != 2) {
		printf( "\nError: Number of arguments incorrect.\n"
				"There can only be 1 additional argument, which is the filename.\n"
				"Ex: ./DecodeEmail2FULP encoded07.dat\n"
				"Program gracefully terminated.\n");
		exit(0);
	}
	
	char *dev_c;
	char string[120]; //120 for buffer/leeway
	
	FILE *fp = fopen(argv[1], "r");
	if (fp == NULL) {
		printf("\nFile cannot be opened.\nProgram gracefully terminated.\n");
		exit(0);
	}
	printf("File opened!\n");
	
	for (int i = 0; i < 120; i++) {
		if ( feof(fp) ) {//If 0, not EOF
			//printf("\nBreak at [i] = %i. [102] = %i, [103] = %i, [104] = %i, %i\n", i, int(string[102]), int(string[103]), int(string[104]), int(string[111]));
			//Last char 'E' is at [101]. [102] = new line, [103] = -2 in ASCII (?), [104] = 0 (null)
			//Few weird ones are because extended ASCII, cannot be represented with signed, must go unsigned
			string[102] = '\0';
			break;
		}
			
		string[i] = fgetc(fp);
		//string[i] = fgetc(fp)-1; //Decoding ASCII. Now done in kernel.
		//printf("%c", string[i]);
	}
	
	fclose(fp);
	
	printf("\nRead In:\n%s\n", string);
	
	//Allocate space in GPU for dev_c
	cudaMalloc((void**) &dev_c, sizeof(char) * 120);
	
	//Copy string (undecoded) to dev_c inside GPU
	cudaMemcpy(dev_c, string, sizeof(char) * 120, cudaMemcpyHostToDevice);
	
	//Running same instruction with X (length of string) threads
	decode<<<1,sizeof(string)/sizeof(string[0])>>>(dev_c);
	
	//Copy result from dev_c (GPU) back to string (CPU)
	cudaMemcpy(&string, dev_c, sizeof(char) * 120, cudaMemcpyDeviceToHost);
	
	printf("\nDecoded:\n%s\n\n", string);
	
	//Reduce greenhouse emissions :/
	cudaFree(dev_c);
	
	exit(0);
}

__global__ void decode(char * dev_c) {
	int i = threadIdx.x;
	if (dev_c[i] == '\0') //So null terminator still present, knows when to stop
		return;
	
	dev_c[i] = dev_c[i] - 1;
}