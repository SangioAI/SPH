/****************************************************************************
 *
 * sph.c -- Smoothed Particle Hydrodynamics
 *
 * https://github.com/cerrno/mueller-sph
 *
 * Copyright (C) 2016 Lucas V. Schuermann
 * Copyright (C) 2022 Moreno Marzolla
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * --------------------------------------------------------------------------
 * 
 * Data: 2023-04-21
 * Author: Sangiorgi Marco (marco.sangiorgi24@studio.unibo.it)
 * Matr: 0000971272
 *
 ****************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include "hpc.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* "Particle-Based Fluid Simulation for Interactive Applications" by
   Müller et al. solver parameters */
__constant__ const float Gx = 0.0, Gy = -10.0;   // external (gravitational) forces
__constant__ const float REST_DENS = 300;    // rest density
__constant__ const float GAS_CONST = 2000;   // const for equation of state
__constant__ const float H = 16;             // kernel radius
__constant__ const float EPS = 16;           // equal to H
__constant__ const float MASS = 2.5;         // assume all particles have the same mass
__constant__ const float VISC = 200;         // viscosity constant
__constant__ const float DT = 0.0007;        // integration timestep
__constant__ const float BOUND_DAMPING = -0.5;

// rendering projection parameters
// (the following ought to be "const float", but then the compiler
// would give an error because VIEW_WIDTH and VIEW_HEIGHT are
// initialized with non-literal expressions)


const int MAX_PARTICLES = 20000;
// Larger window size to accommodate more particles
#define WINDOW_WIDTH 3000
#define WINDOW_HEIGHT 2000


const int DAM_PARTICLES = 500;

const float VIEW_WIDTH = 1.5 * WINDOW_WIDTH;
const float VIEW_HEIGHT = 1.5 * WINDOW_HEIGHT;

/* Particle data structure; stores position, velocity, and force for
   integration stores density (rho) and pressure values for SPH.

   You may choose a different layout of the particles[] data structure
   to suit your needs. */

#define BLKDIM 1024

// data layout : SoA
typedef struct {
    float *x, *y;         // position
    float *vx, *vy;       // velocity
    float *fx, *fy;       // force
    float *rho, *p;       // density, pressure
} particles_t;

particles_t particles;
particles_t *d_particles; // device copy of particles


int n_particles = 0;    // number of currently active particles

/**
 * Return a random value in [a, b]
 */
float randab(float a, float b)
{
    return a + (b-a)*rand() / (float)(RAND_MAX);
}

/**
 * Return max between a and b
 */
float maxab(float a, float b)
{
    return (a > b) ? a : b;
}

/**
 * Return min between a and b
 */
float minab(float a, float b)
{
    return (a < b) ? a : b;
}

/**
 * Set initial position of particle `index` in particles system 'p' to (x, y);
 * initialize all other attributes to default values (zeros).
 */
void init_particle( particles_t p, int index, float x, float y )
{
    p.x[index] = x;
    p.y[index] = y;
    p.vx[index] = 0.0;
    p.vy[index] = 0.0;
    p.fx[index] = 0.0;
    p.fy[index] = 0.0;
    p.rho[index] = 0.0;
    p.p[index] = 0.0;
}



/**
 * Return nonzero iff (x, y) is within the frame
 */
int is_in_domain( float x, float y )
{
    return ((x < VIEW_WIDTH - EPS) &&
            (x > EPS) &&
            (y < VIEW_HEIGHT - EPS) &&
            (y > EPS));
}

/**
 * Initialize the SPH model with `n` particles. The caller is
 * responsible for allocating the `particles[]` array of size
 * `MAX_PARTICLES`.
 *
 * DO NOT parallelize this function, since it calls rand() which is
 * not thread-safe.
 *
 * For MPI and OpenMP: only the master must initialize the domain;
 *
 * For CUDA: the CPU must initialize the domain.
 */
void init_sph( int n )
{
    n_particles = 0;
    printf("Initializing with %d particles\n", n);

    for (float y = EPS; y < VIEW_HEIGHT - EPS; y += H) {
        for (float x = EPS; x <= VIEW_WIDTH * 0.8f; x += H) {
            if (n_particles < n) {
                float jitter = rand() / (float)RAND_MAX;
                init_particle(particles, n_particles, x+jitter, y);
                n_particles++;
            } else {
                return;
            }
        }
    }
    assert(n_particles == n);
}

__global__ void init_sph_device(particles_t *particles, float *d_x, float *d_y, float *d_fx, float *d_fy, float *d_vx, float *d_vy, float *d_rho, float *d_p)
{
    particles->x = d_x;
    particles->y = d_y;
    particles->fx = d_fx;
    particles->fy = d_fy;
    particles->vx = d_vx;
    particles->vy = d_vy;
    particles->rho = d_rho;
    particles->p = d_p;
}

/**
 ** You may parallelize the following four functions
 **/

__global__ void compute_density_pressure( particles_t *particles, int N )
{
    const float HSQ = H * H;    // radius^2 for optimization

    /* Smoothing kernels defined in Muller and their gradients adapted
       to 2D per "SPH Based Shallow Water Simulation" by Solenthaler
       et al. */
    const float POLY6 = 4.0 / (M_PI * pow(H, 8));

    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    if(gindex >= N) {
        return;
    }

    float rho = 0.0;
    for (int j=0; j<N; j++) {

        const float dx = particles->x[j] - particles->x[gindex];
        const float dy = particles->y[j] - particles->y[gindex];
        const float d2 = dx*dx + dy*dy;
        const int mask = (d2 < HSQ);
        rho += mask *(MASS * POLY6 * powf(HSQ - d2, 3.0));
    }
    particles->rho[gindex] = rho;
    particles->p[gindex] = GAS_CONST * (rho - REST_DENS);
}

__global__ void compute_forces( particles_t *particles, int N )
{
    /* Smoothing kernels defined in Muller and their gradients adapted
       to 2D per "SPH Based Shallow Water Simulation" by Solenthaler
       et al. */
    const float SPIKY_GRAD = -10.0 / (M_PI * powf(H, 5));
    const float VISC_LAP = 40.0 / (M_PI * powf(H, 5));
    const float EPS = 1e-6;

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index >= N) {
        return;
    }
    
    float fpress_x = 0.0, fpress_y = 0.0;
    float fvisc_x = 0.0, fvisc_y = 0.0;

    for (int j=0; j<N; j++) {

        const float dx = particles->x[j] - particles->x[index];
        const float dy = particles->y[j] - particles->y[index];
        const float dist = hypotf(dx, dy) + EPS; // avoids division by zero later on
        const int mask = (dist < H) && (index != j);
            const float norm_dx = dx / dist;
            const float norm_dy = dy / dist;
            // compute pressure force contribution
            fpress_x += mask * (-norm_dx * MASS * (particles->p[index] + particles->p[j]) / (2 * particles->rho[j]) * SPIKY_GRAD * powf(H - dist, 3));
            fpress_y += mask * (-norm_dy * MASS * (particles->p[index] + particles->p[j]) / (2 * particles->rho[j]) * SPIKY_GRAD * powf(H - dist, 3));
            // compute viscosity force contribution
            fvisc_x += mask * (VISC * MASS * (particles->vx[j] - particles->vx[index]) / particles->rho[j] * VISC_LAP * (H - dist));
            fvisc_y += mask * (VISC * MASS * (particles->vy[j] - particles->vy[index]) / particles->rho[j] * VISC_LAP * (H - dist));
    }
    const float fgrav_x = Gx * MASS / particles->rho[index];
    const float fgrav_y = Gy * MASS / particles->rho[index];
    particles->fx[index] = fpress_x + fvisc_x + fgrav_x;
    particles->fy[index] = fpress_y + fvisc_y + fgrav_y;
}

__global__ void integrate( particles_t *particles, int N )
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index >= N) {
        return;
    }
    
    // forward Euler integration
    particles->vx[index] += DT *particles->fx[index] /particles->rho[index];
    particles->vy[index] += DT *particles->fy[index] /particles->rho[index];
    particles->x[index] += DT *particles->vx[index];
    particles->y[index] += DT *particles->vy[index];

    // enforce boundary conditions
    if (particles->x[index] - EPS < 0.0) {
        particles->vx[index] *= BOUND_DAMPING;
        particles->x[index] = EPS;
    }
    if (particles->x[index] + EPS > VIEW_WIDTH) {
        particles->vx[index] *= BOUND_DAMPING;
        particles->x[index] = VIEW_WIDTH - EPS;
    }
    if (particles->y[index] - EPS < 0.0) {
        particles->vy[index] *= BOUND_DAMPING;
        particles->y[index] = EPS;
    }
    if (particles->y[index] + EPS > VIEW_HEIGHT) {
        particles->vy[index] *= BOUND_DAMPING;
        particles->y[index] = VIEW_HEIGHT - EPS;
    }
}

/* Note: *result must be initially zero for this kernel to work! */
__global__ void avg_velocities( particles_t *particles, int n, float *result)
{
    __shared__ float temp[BLKDIM];
    int lindex = threadIdx.x;
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int bsize = blockDim.x / 2;

    if ( gindex < n ) {
        temp[lindex] = hypotf(particles->vx[gindex], particles->vy[gindex]) / n;
    } else {
        temp[lindex] = 0;
    }

    /* wait for all threads to finish the copy operation */
    __syncthreads();

    /* All threads within the block cooperate to compute the local sum */
    while ( bsize > 0 ) {
        if ( lindex < bsize ) {
            temp[lindex] += temp[lindex + bsize];
        }
        bsize = bsize / 2;
        /* threads must synchronize before performing the next
           reduction step */
        __syncthreads();
    }

    if ( 0 == lindex ) {
        atomicAdd(result, temp[0]);
    }
}

void update()
{
    compute_density_pressure<<<(n_particles + BLKDIM-1)/BLKDIM, BLKDIM>>>(d_particles, n_particles);
    cudaDeviceSynchronize();// needed -  wait for threads in grid (all blocks)

    compute_forces<<<(n_particles + BLKDIM-1)/BLKDIM, BLKDIM>>>(d_particles, n_particles);
    cudaDeviceSynchronize();// needed -  wait for threads in grid (all blocks)

    integrate<<<(n_particles + BLKDIM-1)/BLKDIM, BLKDIM>>>(d_particles, n_particles);
    cudaDeviceSynchronize();// needed -  wait for threads in grid (all blocks)
}

int main(int argc, char **argv)
{
    srand(1234);

    particles.x = (float*)malloc(MAX_PARTICLES * sizeof(*particles.x));
    assert( particles.x != NULL );
    particles.y = (float*)malloc(MAX_PARTICLES * sizeof(*particles.y));
    assert( particles.y != NULL );

    particles.vx = (float*)malloc(MAX_PARTICLES * sizeof(*particles.vx));
    assert( particles.vx != NULL );
    particles.vy = (float*)malloc(MAX_PARTICLES * sizeof(*particles.vy));
    assert( particles.vy != NULL );

    particles.fx = (float*)malloc(MAX_PARTICLES * sizeof(*particles.fx));
    assert( particles.fx != NULL );
    particles.fy = (float*)malloc(MAX_PARTICLES * sizeof(*particles.fy));
    assert( particles.fy != NULL );

    particles.rho = (float*)malloc(MAX_PARTICLES * sizeof(*particles.rho));
    assert( particles.rho != NULL );
    particles.p = (float*)malloc(MAX_PARTICLES * sizeof(*particles.p));
    assert( particles.p != NULL );

    int n = DAM_PARTICLES;
    int nsteps = 50;

    if (argc > 4) {
        fprintf(stderr, "Usage: %s [nparticles [nsteps]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (argc > 2) {
        nsteps = atoi(argv[2]);
    }

    if (n > MAX_PARTICLES) {
        fprintf(stderr, "FATAL: the maximum number of particles is %d\n", MAX_PARTICLES);
        return EXIT_FAILURE;
    }


    const size_t size = n*sizeof (float);
    float *d_x, *d_y, *d_fx, *d_fy, *d_vx, *d_vy, *d_rho, *d_p; //device arrays of values


    //CUDA mallocs
    /* Allocate space for device copy of particles structure */
    cudaSafeCall( cudaMalloc((void **)&d_particles, sizeof(particles_t)) );
    cudaSafeCall( cudaMalloc((void **)&d_x, size) );
    cudaSafeCall( cudaMalloc((void **)&d_y, size) );
    cudaSafeCall( cudaMalloc((void **)&d_fx, size) );
    cudaSafeCall( cudaMalloc((void **)&d_fy, size) );
    cudaSafeCall( cudaMalloc((void **)&d_vx, size) );
    cudaSafeCall( cudaMalloc((void **)&d_vy, size) );
    cudaSafeCall( cudaMalloc((void **)&d_rho, size) );
    cudaSafeCall( cudaMalloc((void **)&d_p, size) );
    // init SoA device structure
    init_sph_device<<<1,1>>>(d_particles, d_x, d_y, d_fx, d_fy, d_vx, d_vy, d_rho, d_p);



    float *d_avg; // device copy of avg
    float avg = 0;
    cudaMalloc((void **)&d_avg, sizeof(*d_avg));

    printf("----start-----\n");

    init_sph(n);

    // copy data to cuda memory
    cudaSafeCall( cudaMemcpy(d_x, particles.x, size, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_y, particles.y, size, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_fx, particles.fx, size, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_fy, particles.fy, size, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_vx, particles.vx, size, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_vy, particles.vy, size, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_rho, particles.rho, size, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_p, particles.p, size, cudaMemcpyHostToDevice) );

    for (int s=0; s<nsteps; s++) {
        update();
        cudaDeviceSynchronize();
        /* the average velocities MUST be computed at each step, even
        if it is not shown (to ensure constant workload per
        iteration) */
        avg=0;
        cudaMemcpy(d_avg, &avg, sizeof (float), cudaMemcpyHostToDevice); // init d_avg
        avg_velocities<<<(n + BLKDIM-1)/BLKDIM, BLKDIM>>>(d_particles, n, d_avg); // calculate avg_velocities
        cudaMemcpy(&avg, d_avg, sizeof (float), cudaMemcpyDeviceToHost); // copy into avg
        if (s % 10 == 0)
            printf("step %5d, avgV=%f\n", s, avg);
    }

    //free host memory
    free(particles.x);
    free(particles.y);
    free(particles.vx);
    free(particles.vy);
    free(particles.fx);
    free(particles.fy);
    free(particles.rho);
    free(particles.p);

    //free cuda memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_rho);
    cudaFree(d_p);

    cudaFree(d_particles);

    cudaFree(&d_avg);
    return EXIT_SUCCESS;
}
