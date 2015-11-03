#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define Threads 768

#define MASS 0     // row in array for mass
#define X_POS 1    // row in array for x position
#define Y_POS 2    // row in array for y position
#define Z_POS 3    // row in array for z position
#define X_VEL 4    // row in array for x velocity
#define Y_VEL 5    // row in array for y velocity
#define Z_VEL 6    // row in array for z velocity

#define N 10000    // number of bodies // 9999
#define COL 7	   //Easy for traversal
#define G 10       // "gravitational constant" (not really)
#define MU 0.001   // "frictional coefficient" 
#define BOXL 100.0 // periodic boundary box length

float dt = 0.05; // time interval


__global__ void init (unsigned int, curandState_t*);
__global__ void initAssign(curandState_t*, float*);
__global__ void nbody (curandState_t*, float*, float*, float*, float*);

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
	cudaSetDevice(1);
	printf("Before initrand\n");
	//------------------------------------------------------------------------------------------
	curandState_t* dev_states; //keep track of seed value for every thread
	cudaMalloc((void**) &dev_states, N * sizeof(curandState_t)); //N
	//initialize all of the random states on the GPU
	init<<<(int)ceil(N/Threads) + 1, Threads>>>(time(NULL), dev_states); //N
	//------------------------------------------------------------------------------------------

	/* Following section CANNOT be PARALLELIZED yet */

	/*
		tmax = timesteps
	*/

	float body[N * 7];
    //float *body = (float *)malloc(N * 7 * sizeof(float));

	float *Fx_dir = (float *)malloc(N * sizeof(float)); //Probably don't need to put these on heap
	float *Fy_dir = (float *)malloc(N * sizeof(float)); 
	float *Fz_dir = (float *)malloc(N * sizeof(float));
	float *dev_body, *dev_fx, *dev_fy, *dev_fz;

	cudaMalloc((void**) &dev_body, N * 7 * sizeof(float));
	cudaMalloc((void**) &dev_fx, N * sizeof(float));
	cudaMalloc((void**) &dev_fy, N * sizeof(float));
	cudaMalloc((void**) &dev_fz, N * sizeof(float));

	//------------------------------------------------------------------------------------------
	// Assign each body a random initial positions and velocities
	cudaMemcpy(dev_body, &body, N * 7 * sizeof(float), cudaMemcpyHostToDevice); 
	initAssign<<<(int)ceil(N/Threads) + 1, Threads>>>(dev_states, dev_body);
	cudaThreadSynchronize();
	//Do I need this?	
	cudaMemcpy(body, dev_body, N * 7 * sizeof(float), cudaMemcpyDeviceToHost);
	//------------------------------------------------------------------------------------------

	// Print out initial positions in PDB format
	/*
	printf("MODEL %8d\n", 0);
	fprintf(f, "MODEL %8d\n", 0);
	for (int i = 0; i < N; i++) {
		printf("%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
				"ATOM", i+1, "CA ", "GLY", "A", i+1, body[i * COL + X_POS], body[i * COL + Y_POS], body[i * COL + Z_POS], 1.00, 0.00);
		fprintf(f, "%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
				"ATOM", i+1, "CA ", "GLY", "A", i+1, body[i * COL + X_POS], body[i * COL + Y_POS], body[i * COL + Z_POS], 1.00, 0.00);
	}
	printf("TER\nENDMDL\n");
	fprintf(f, "TER\nENDMDL\n");*/

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
 
		cudaMemcpy(dev_body, &body, N * 7 * sizeof(float), cudaMemcpyHostToDevice); 
		cudaMemcpy(dev_fx, &Fx_dir, N * sizeof(float), cudaMemcpyHostToDevice); //Check this, could be faulty
		cudaMemcpy(dev_fy, &Fy_dir, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_fz, &Fz_dir, N * sizeof(float), cudaMemcpyHostToDevice);
		
		nbody<<<(int)ceil(N/Threads) + 1, Threads>>>(dev_states, dev_body, dev_fx, dev_fy, dev_fz);
		
		cudaThreadSynchronize();
		
		cudaMemcpy(body, dev_body, N * 7 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(Fx_dir, dev_fx, N * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(Fy_dir, dev_fy, N * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(Fz_dir, dev_fz, N * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(dev_states);
		cudaFree(dev_body);
		cudaFree(dev_fx);
		cudaFree(dev_fy);
		cudaFree(dev_fz);

//------------------------------------------------------------------------------------------

		/* PARALLELIZE THIS TOO - Can be inside above, or new kernel */
		// update postions and velocity in array
		for (int i = 0; i < N; i++) {

			// Update velocities
			body[i * COL + X_VEL] += Fx_dir[i] * dt / body[i * COL + MASS];
			body[i * COL + Y_VEL] += Fy_dir[i] * dt / body[i * COL + MASS];
			body[i * COL + Z_VEL] += Fz_dir[i] * dt / body[i * COL + MASS];

			// periodic boundary conditions
			if (body[i * COL + X_VEL] <  -BOXL * 0.5) body[i * COL + X_VEL] += BOXL;
			if (body[i * COL + X_VEL] >=  BOXL * 0.5) body[i * COL + X_VEL] -= BOXL;
			if (body[i * COL + Y_VEL] <  -BOXL * 0.5) body[i * COL + Y_VEL] += BOXL;
			if (body[i * COL + Y_VEL] >=  BOXL * 0.5) body[i * COL + Y_VEL] -= BOXL;
			if (body[i * COL + Z_VEL] <  -BOXL * 0.5) body[i * COL + Z_VEL] += BOXL;
			if (body[i * COL + Z_VEL] >=  BOXL * 0.5) body[i * COL + Z_VEL] -= BOXL;

			// Update positions
			body[i * COL + X_POS] += body[i * COL + X_VEL] * dt;
			body[i * COL + Y_POS] += body[i * COL + Y_VEL] * dt;
			body[i * COL + Z_POS] += body[i * COL + Z_VEL] * dt;

			// Periodic boundary conditions
			if (body[i * COL + X_POS] <  -BOXL * 0.5) body[i * COL + X_POS] += BOXL;
			if (body[i * COL + X_POS] >=  BOXL * 0.5) body[i * COL + X_POS] -= BOXL;
			if (body[i * COL + Y_POS] <  -BOXL * 0.5) body[i * COL + Y_POS] += BOXL;
			if (body[i * COL + Y_POS] >=  BOXL * 0.5) body[i * COL + Y_POS] -= BOXL;
			if (body[i * COL + Z_POS] <  -BOXL * 0.5) body[i * COL + Z_POS] += BOXL;
			if (body[i * COL + Z_POS] >=  BOXL * 0.5) body[i * COL + Z_POS] -= BOXL;

		}
/*
		// Print out positions in PDB format
		printf("MODEL %8d\n", t+1);
		fprintf(f, "MODEL %8d\n", t+1);
		for (int i = 0; i < N; i++) {
			printf("%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
					"ATOM", i+1, "CA ", "GLY", "A", i+1, body[i * COL + X_POS], body[i * COL + Y_POS], body[i * COL + Z_POS], 1.00, 0.00);
			fprintf(f, "%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
					"ATOM", i+1, "CA ", "GLY", "A", i+1, body[i * COL + X_POS], body[i * COL + Y_POS], body[i * COL + Z_POS], 1.00, 0.00);
		}
		printf("TER\nENDMDL\n");
		fprintf(f, "TER\nENDMDL\n");
		*/
	}  // end of time period loop
	
	//------------------------------------------------------------------------------------------
	
	fclose(f);
	
}

__global__ void nbody (curandState_t* states, float* body, float* Fx_dir, float* Fy_dir, float* Fz_dir) {  //**

	//This loop should run N times in total (aka, kernel should be called N times)
	int currentBodyID = blockDim.x * blockIdx.x + threadIdx.x;

	if (currentBodyID >= N) //Since if N = 10000, body[10000] shouldn't work either
		return;

	for (int i = 0; i < N; i++) { // All other bodies 

		/* Each body interacting with every other pair */
		// position differences in x-, y-, and z-directions
		float x_diff, y_diff, z_diff;

		if (i != currentBodyID) {
			// TODO: calculate position difference between body i and x in x-,y-, and z-directions

			x_diff = body[i * COL + X_POS] - body[currentBodyID * COL + X_POS];
			y_diff = body[i * COL + Y_POS] - body[currentBodyID * COL + Y_POS];
			z_diff = body[i * COL + Z_POS] - body[currentBodyID * COL + Z_POS];

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
				Fg = (G * body[i * COL + MASS] * body[currentBodyID * COL + MASS]) / rr; /* Added. Check this, something might not be right - Cho */

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

__global__ void initAssign(curandState_t* states, float* body) {
	int currentBodyID = blockDim.x * blockIdx.x + threadIdx.x;
	if (currentBodyID >= N)
		return;

	body[currentBodyID * COL + MASS] = 0.001;

	body[currentBodyID * COL + X_VEL] = curand_uniform(&states[currentBodyID]);
	body[currentBodyID * COL + Y_VEL] = curand_uniform(&states[currentBodyID]);
	body[currentBodyID * COL + Z_VEL] = curand_uniform(&states[currentBodyID]);

	body[currentBodyID * COL + X_POS] = curand_uniform(&states[currentBodyID]);
	body[currentBodyID * COL + Y_POS] = curand_uniform(&states[currentBodyID]);
	body[currentBodyID * COL + Z_POS] = curand_uniform(&states[currentBodyID]);

	printf("%f, %f, %f, %f, %f, %f, %f", body[currentBodyID * COL + 0], body[currentBodyID * COL + 1], 
		body[currentBodyID * COL + 2], body[currentBodyID * COL + 3], body[currentBodyID * COL + 4], 
		body[currentBodyID * COL + 5], body[currentBodyID * COL + 6]);
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