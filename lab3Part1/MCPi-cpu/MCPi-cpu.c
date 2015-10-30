#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*
	Note: Assumes square width as 2 units wide; area = 4
	Compile: gcc MCPi-cpu.c -o MCPi-cpu -lm
*/

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
	
	srand48(time(NULL));

	int i, count = 0, normX, normY, histogram[10] = {0};
	double x, y, t;
	for (i = 0; i < points; i++) {
		x = drand48(); //*2
		y = drand48(); //0.8. * 10 = 8. norm[8]++. 0.8 = norm[8]
		normX = (int)floor(x*10);
		normY = (int)floor(y*10);
		histogram[normX]++;
		histogram[normY]++;
		//printf("%f, %f, %f, %i\n", x, y, normX, (int)floor(normX*10) );
		//printf("%f(%0.1f), %f(%0.1f), %i, %i\n", x, normX*0.1, y, normY*0.1, histogram[normX], histogram[normY]);

		/*
			Since square width hardcoded = 2, radius must be 1. Using:
			radiu2^2 > (x - center_x)^2 + (y - center_y)^2
		*/

		t = pow((x - 1.0), 2.0) + pow((y - 1.0), 2.0); //2.0 to return t as double
				
		if (1.0 >= t) //if radius >= t
			count++;
		//printf("%f, %f, %f, %i\n", x, y, t, count);

	}

	/*	
		Area(circle)/Area(square) = pi / 4
		pi = Area(circle)/Area(square) * 4
	*/

	printf("Pi: %0.3f\n", 4.0 * (double)count/(double)points); 

	for (i = 0; i < 10; i++) {
		//printf("%0.1f, %i\n", i * 0.1, histogram[i]);
		fprintf(f, "%0.1f\t%i\n", i * 0.1, histogram[i]);
	}
	fclose(f);

}