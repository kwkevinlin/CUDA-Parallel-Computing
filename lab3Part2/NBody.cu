#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define MASS 0     // row in array for mass
#define X_POS 1    // row in array for x position
#define Y_POS 2    // row in array for y position
#define Z_POS 3    // row in array for z position
#define X_VEL 4    // row in array for x velocity
#define Y_VEL 5    // row in array for y velocity
#define Z_VEL 6    // row in array for z velocity

#define N 9999     // number of bodies
#define G 10       // "gravitational constant" (not really)
#define MU 0.001   // "frictional coefficient" 
#define BOXL 100.0 // periodic boundary box length

float dt = 0.05; // time interval


__global__ void init (unsigned int, curandState_t*);
__global__ void nbody (curandState_t*, float**, float*, float*, float*);

int main (int argc, char *argv[]) {

	if (argc != 2) { 
		printf( "\nError: Number of arguments incorrect.\n"
				"There can only be 1 additional argument, which is the number of timesteps.\n"
				"Ex: ./NBody 100\n"
				"Program gracefully terminated.\n"); 
		exit(0);
	}

	FILE *f = fopen("NBody.pdb", "w+");
	if (f == NULL) {
		printf("File could not be created/opened!\n"
				"Program gracefully terminated.\n");
		exit(1);
	}

	int timesteps = atoi(argv[1]);
	if (timesteps < 0) {
		printf("Input must be a positive number!\n"
			   "Program gracefully terminated.\n");
		exit(0);
	}
	
	//------------------------------------------------------------------------------------------
	curandState_t* dev_states; //keep track of seed value for every thread
	cudaMalloc((void**) &dev_states, N * sizeof(curandState_t)); //N
	//initialize all of the random states on the GPU
	init<<<(int)ceil(N/1) + 1, N>>>(time(NULL), dev_states); //N
	//------------------------------------------------------------------------------------------

	/* Following section CANNOT be PARALLELIZED yet */

	/*
		tmax = timesteps
	*/

	float **body = (float **)malloc(10000 * sizeof(float *));
    for (int i = 0; i < 10000; i++)
         body[i] = (float *)malloc(7 * sizeof(float));
	float *Fx_dir = (float *)malloc(N * sizeof(float)); //Probably don't need to put these on heap
	float *Fy_dir = (float *)malloc(N * sizeof(float)); 
	float *Fz_dir = (float *)malloc(N * sizeof(float)); 
	float **dev_body, *dev_fx, *dev_fy, *dev_fz;

	srand48(time(NULL));

	// Assign each body a random initial positions and velocities
	for (int i = 0; i < N; i++) {
		body[i][MASS] = 0.001;

		body[i][X_VEL] = drand48();
		body[i][Y_VEL] = drand48();
		body[i][Z_VEL] = drand48();

		body[i][X_POS] = drand48();
		body[i][Y_POS] = drand48();
		body[i][Z_POS] = drand48();
	}	

	// Print out initial positions in PDB format
	printf("MODEL %8d\n", 0);
	fprintf(f, "MODEL %8d\n", 0);
	for (int i = 0; i < N; i++) {
		printf("%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
				"ATOM", i+1, "CA ", "GLY", "A", i+1, body[i][X_POS], body[i][Y_POS], body[i][Z_POS], 1.00, 0.00);
		fprintf(f, "%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
				"ATOM", i+1, "CA ", "GLY", "A", i+1, body[i][X_POS], body[i][Y_POS], body[i][Z_POS], 1.00, 0.00);
	}
	printf("TER\nENDMDL\n");
	fprintf(f, "TER\nENDMDL\n");

	// Step through each time step
	for (int t = 0; t < timesteps; t++) { 

		// TODO: initialize forces to zero
		for (int i = 0; i < N; i++) {
			Fx_dir[i] = 0.0; 
			Fy_dir[i] = 0.0; 
			Fz_dir[i] = 0.0;
		}

		/* 
			PARALLELIZATION STARTS HERE

				Initiate CUDA call
		*/

		cudaMalloc((void**) &dev_body, 10000 * 7 * sizeof(float));
		cudaMalloc((void**) &dev_fx, N * sizeof(float));
		cudaMalloc((void**) &dev_fy, N * sizeof(float));
		cudaMalloc((void**) &dev_fz, N * sizeof(float));
		
		cudaMemcpy(dev_body, &body, 10000 * 7 * sizeof(float), cudaMemcpyHostToDevice); 
		cudaMemcpy(dev_fx, &Fx_dir, N * sizeof(float), cudaMemcpyHostToDevice); //Check this, could be faulty
		cudaMemcpy(dev_fy, &Fy_dir, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_fz, &Fz_dir, N * sizeof(float), cudaMemcpyHostToDevice);
		
		//<<<(int)ceil(points/Threads) + 1, Threads>>>
		nbody<<<N, 1>>>(dev_states, dev_body, dev_fx, dev_fy, dev_fz);
		
		cudaThreadSynchronize();
		
		cudaMemcpy(body, dev_body, 10000 * 7 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(Fx_dir, dev_fx, N * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(Fy_dir, dev_fy, N * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(Fz_dir, dev_fz, N * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(dev_body);
		cudaFree(dev_fx);
		cudaFree(dev_fy);
		cudaFree(dev_fz);

//------------------------------------------------------------------------------------------

		// update postions and velocity in array
		for (int i = 0; i < N; i++) {

			// Update velocities
			body[i][X_VEL] += Fx_dir[i] * dt / body[i][MASS];
			body[i][Y_VEL] += Fy_dir[i] * dt / body[i][MASS];
			body[i][Z_VEL] += Fz_dir[i] * dt / body[i][MASS];

			// periodic boundary conditions
			if (body[i][X_VEL] <  -BOXL * 0.5) body[i][X_VEL] += BOXL;
			if (body[i][X_VEL] >=  BOXL * 0.5) body[i][X_VEL] -= BOXL;
			if (body[i][Y_VEL] <  -BOXL * 0.5) body[i][Y_VEL] += BOXL;
			if (body[i][Y_VEL] >=  BOXL * 0.5) body[i][Y_VEL] -= BOXL;
			if (body[i][Z_VEL] <  -BOXL * 0.5) body[i][Z_VEL] += BOXL;
			if (body[i][Z_VEL] >=  BOXL * 0.5) body[i][Z_VEL] -= BOXL;

			// Update positions
			body[i][X_POS] += body[i][X_VEL] * dt;
			body[i][Y_POS] += body[i][Y_VEL] * dt;
			body[i][Z_POS] += body[i][Z_VEL] * dt;

			// Periodic boundary conditions
			if (body[i][X_POS] <  -BOXL * 0.5) body[i][X_POS] += BOXL;
			if (body[i][X_POS] >=  BOXL * 0.5) body[i][X_POS] -= BOXL;
			if (body[i][Y_POS] <  -BOXL * 0.5) body[i][Y_POS] += BOXL;
			if (body[i][Y_POS] >=  BOXL * 0.5) body[i][Y_POS] -= BOXL;
			if (body[i][Z_POS] <  -BOXL * 0.5) body[i][Z_POS] += BOXL;
			if (body[i][Z_POS] >=  BOXL * 0.5) body[i][Z_POS] -= BOXL;

		}

		// Print out positions in PDB format
		printf("MODEL %8d\n", t+1);
		fprintf(f, "MODEL %8d\n", t+1);
		for (int i = 0; i < N; i++) {
			printf("%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
					"ATOM", i+1, "CA ", "GLY", "A", i+1, body[i][X_POS], body[i][Y_POS], body[i][Z_POS], 1.00, 0.00);
			fprintf(f, "%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
					"ATOM", i+1, "CA ", "GLY", "A", i+1, body[i][X_POS], body[i][Y_POS], body[i][Z_POS], 1.00, 0.00);
		}
		printf("TER\nENDMDL\n");
		fprintf(f, "TER\nENDMDL\n");
	}  // end of time period loop

	
	
	
	//------------------------------------------------------------------------------------------
	
	
	
	fclose(f);
	
}

__global__ void nbody (curandState_t* states, float** body, float* Fx_dir, float* Fy_dir, float* Fz_dir) {  //**

	//This loop should run N times in total (aka, kernel should be called N times)
	int currentBodyID = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i < N; i++) { // All other bodies 

		/* Each body interacting with every other pair */
		// position differences in x-, y-, and z-directions
		float x_diff, y_diff, z_diff;

		if (i != currentBodyID) {
			// TODO: calculate position difference between body i and x in x-,y-, and z-directions
			x_diff = body[i][X_POS] - body[currentBodyID][X_POS];
			y_diff = body[i][Y_POS] - body[currentBodyID][Y_POS];
			z_diff = body[i][Z_POS] - body[currentBodyID][Z_POS];


			// periodic boundary conditions
			if (x_diff <  -BOXL * 0.5) x_diff += BOXL;
			if (x_diff >=  BOXL * 0.5) x_diff -= BOXL;
			if (y_diff <  -BOXL * 0.5) y_diff += BOXL;
			if (y_diff >=  BOXL * 0.5) y_diff -= BOXL;
			if (z_diff <  -BOXL * 0.5) z_diff += BOXL;
			if (z_diff >=  BOXL * 0.5) z_diff -= BOXL;

			// calculate distance (r)
			float rr = (x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);
			float r = sqrt(rr);

			// force between bodies i and x
			float F = 0.0, Fg = 0.0, Fr = 0.0; /* Total force between i and x */

			// if sufficiently far away, gravitation force

			/*
				Alternative to sqrt, if rr > 2.0 * 2.0 (for 2.0^2)
			*/
			if (r > 2.0) {
				// Compute gravitational force between body i and x
				//F = G * m1 * m2 / rr
				Fg = (G * body[i][MASS] * body[currentBodyID][MASS]) / rr; /* Added. Check this, something might not be right - Cho */

				// Compute frictional force
				//Fr = MU * (drand48() - 0.5); // Added // Bug fix: range [0.5, 0.5]. Revert just take out -0.5
				int globalId = blockDim.x * blockIdx.x + threadIdx.x;
				Fr = MU * curand_uniform(&states[globalId]);
				
				F = Fg + Fr; // Added. Get total force

				Fx_dir[currentBodyID] += F * x_diff / r;  // resolve forces in x and y directions
				Fy_dir[currentBodyID] += F * y_diff / r;  // and accumulate forces
				Fz_dir[currentBodyID] += F * z_diff / r;  // 
			} else {
				// If too close, weak anti-gravitational force
				float F = G * 0.01 * 0.01 / r;
				Fx_dir[currentBodyID] -= F * x_diff / r;  // resolve forces in x and y directions
				Fy_dir[currentBodyID] -= F * y_diff / r;  // and accumulate forces
				Fz_dir[currentBodyID] -= F * z_diff / r;  // 
			}
		}
	}
	
}

// Initialize the random states
__global__ void init (unsigned int seed, curandState_t* states) {
	int globalId = blockDim.x * blockIdx.x + threadIdx.x;
	//globaId -> threadIdx.x
	curand_init(seed, // same seed for each core send from host
	globalId, // sequence number; different for each core
	0, // offset
	&states[globalId]);
} 