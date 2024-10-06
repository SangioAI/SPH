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
#include <math.h>
#include <assert.h>
#include <limits.h>
#include "hpc.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* "Particle-Based Fluid Simulation for Interactive Applications" by
   Müller et al. solver parameters */

const float Gx = 0.0, Gy = -10.0;   // external (gravitational) forces
const float REST_DENS = 300;    // rest density
const float GAS_CONST = 2000;   // const for equation of state
const float H = 16;             // kernel radius
const float EPS = 16;           // equal to H
const float MASS = 2.5;         // assume all particles have the same mass
const float VISC = 200;         // viscosity constant
const float DT = 0.0007;        // integration timestep
const float BOUND_DAMPING = -0.5;

// rendering projection parameters
// (the following ought to be "const float", but then the compiler
// would give an error because VIEW_WIDTH and VIEW_HEIGHT are
// initialized with non-literal expressions)
#ifdef GUI

const int MAX_PARTICLES = 5000;
#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 768

#else

const int MAX_PARTICLES = 20000;
// Larger window size to accommodate more particles
#define WINDOW_WIDTH 3000
#define WINDOW_HEIGHT 2000

#endif

const int DAM_PARTICLES = 500;

const float VIEW_WIDTH = 1.5 * WINDOW_WIDTH;
const float VIEW_HEIGHT = 1.5 * WINDOW_HEIGHT;


/* Particle data structure; stores position, velocity, and force for
   integration stores density (rho) and pressure values for SPH.

   You may cho ose a different layout of the particles[] data structure
   to suit your needs. */

// data layout : SoA
typedef struct {
    float *x, *y;         // position
    float *vx, *vy;       // velocity
    float *fx, *fy;       // force
    float *rho, *p;       // density, pressure
} particles_t;

particles_t particles;

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

/**
 ** You may parallelize the following four functions
 **/

void compute_density_pressure( void )
{
    const float HSQ = H * H;    // radius^2 for optimization

    /* Smoothing kernels defined in Muller and their gradients adapted
       to 2D per "SPH Based Shallow Water Simulation" by Solenthaler
       et al. */
    const float POLY6 = 4.0 / (M_PI * pow(H, 8));
    
    // #pragma omp parallel for schedule(dynamic,1)
    for (int i=0; i<n_particles; i++) {
        float rho = 0.0;

        #pragma omp parallel for reduction(+:rho)
        for (int j=i; j<n_particles; j++) {

            if(i==0) {
                particles.rho[j] = 0.0;
            }

            const float dx = particles.x[j] - particles.x[i];
            const float dy = particles.y[j] - particles.y[i];
            const float d2 = dx*dx + dy*dy;

            if (d2 < HSQ) {
                rho += MASS * POLY6 * pow(HSQ - d2, 3.0);
                if(i!=j)
                    particles.rho[j] += MASS * POLY6 * pow(HSQ - d2, 3.0);
            }
        }

        particles.rho[i] += rho;
        particles.p[i] = GAS_CONST * (particles.rho[i] - REST_DENS);
    }
}

void compute_forces( void )
{
    /* Smoothing kernels defined in Muller and their gradients adapted
       to 2D per "SPH Based Shallow Water Simulation" by Solenthaler
       et al. */
    const float SPIKY_GRAD = -10.0 / (M_PI * pow(H, 5));
    const float VISC_LAP = 40.0 / (M_PI * pow(H, 5));
    const float EPS = 1e-6;
    // const float HSQ = H * H;    // radius^2 for optimization


    //--------NOTE: dynamic funziona meglio di static --> alcuni thread possono non dover entrare nell'if del for interno
    // #pragma omp parallel for schedule(dynamic,1) // spedup 3.3x
    for (int i=0; i<n_particles; i++) {
        float fpress_x = 0.0, fpress_y = 0.0;
        float fvisc_x = 0.0, fvisc_y = 0.0;

        if(i==0) {
            particles.fx[i] = 0.0;
            particles.fy[i] = 0.0;
        }

        #pragma omp parallel for reduction(+:fpress_x) reduction(+:fpress_y) reduction(+:fvisc_x) reduction(+:fvisc_y) // from 3.3x to 3.9x
        for (int j=i+1; j<n_particles; j++) {

            // resettare fx e fy al primo turno
            if(i==0) {
                particles.fx[j] = 0.0;
                particles.fy[j] = 0.0;
            }

            const float dx = particles.x[j] - particles.x[i];
            const float dy = particles.y[j] - particles.y[i];
            const float dist = hypotf(dx, dy) + EPS; // avoids division by zero later on

            if (dist < H) {
                const float norm_dx = dx / dist;
                const float norm_dy = dy / dist;
                // compute pressure force contribution
                fpress_x += -norm_dx * MASS * (particles.p[i] + particles.p[j]) / (2 * particles.rho[j]) * SPIKY_GRAD * pow(H - dist, 3); // h-dist = h-dist^3 = h-dist ^ 3 * h-dist
                fpress_y += -norm_dy * MASS * (particles.p[i] + particles.p[j]) / (2 * particles.rho[j]) * SPIKY_GRAD * pow(H - dist, 3);
                // compute viscosity force contribution
                fvisc_x += VISC * MASS * (particles.vx[j] - particles.vx[i]) / particles.rho[j] * VISC_LAP * (H - dist);
                fvisc_y += VISC * MASS * (particles.vy[j] - particles.vy[i]) / particles.rho[j] * VISC_LAP * (H - dist);

                // compute pressure force contribution for particle j
                // -------NOTE: opposite sign
                particles.fx[j] += norm_dx * MASS * (particles.p[j] + particles.p[i]) / (2 * particles.rho[i]) * SPIKY_GRAD * pow(H - dist, 3);
                particles.fy[j] += norm_dy * MASS * (particles.p[j] + particles.p[i]) / (2 * particles.rho[i]) * SPIKY_GRAD * pow(H - dist, 3);
                // compute viscosity force contribution for particle j
                particles.fx[j] += VISC * MASS * (particles.vx[i] - particles.vx[j]) / particles.rho[i] * VISC_LAP * (H - dist);
                particles.fy[j] += VISC * MASS * (particles.vy[i] - particles.vy[j]) / particles.rho[i] * VISC_LAP * (H - dist);
            }
        }
        const float fgrav_x = Gx * MASS / particles.rho[i];
        const float fgrav_y = Gy * MASS / particles.rho[i];
        particles.fx[i] += fpress_x + fvisc_x + fgrav_x;
        particles.fy[i] += fpress_y + fvisc_y + fgrav_y;
    }   
}

void integrate( void )
{
    #pragma omp parallel for
    for (int i=0; i<n_particles; i++) {
        // forward Euler integration
       particles.vx[i] += DT *particles.fx[i] /particles.rho[i];
       particles.vy[i] += DT *particles.fy[i] /particles.rho[i];
       particles.x[i] += DT *particles.vx[i];
       particles.y[i] += DT *particles.vy[i];

        // enforce boundary conditions
        if (particles.x[i] - EPS < 0.0) {
           particles.vx[i] *= BOUND_DAMPING;
           particles.x[i] = EPS;
        }
        if (particles.x[i] + EPS > VIEW_WIDTH) {
           particles.vx[i] *= BOUND_DAMPING;
           particles.x[i] = VIEW_WIDTH - EPS;
        }
        if (particles.y[i] - EPS < 0.0) {
           particles.vy[i] *= BOUND_DAMPING;
           particles.y[i] = EPS;
        }
        if (particles.y[i] + EPS > VIEW_HEIGHT) {
           particles.vy[i] *= BOUND_DAMPING;
           particles.y[i] = VIEW_HEIGHT - EPS;
        }
    }
}

float avg_velocities( void )
{
    double result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (int i=0; i<n_particles; i++) {
        /* the hypot(x,y) function is equivalent to sqrt(x*x +
           y*y); */
        result += hypot(particles.vx[i], particles.vy[i]) / n_particles;
    }
    return result;
}

void update( void )
{
    compute_density_pressure();
    compute_forces();
    integrate();
}

int main(int argc, char **argv)
{
    srand(1234);

    // allocating SoA
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
    int iters = 10;

    if (argc > 4) {
        fprintf(stderr, "Usage: %s [nparticles [nsteps] [iters]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (argc > 2) {
        nsteps = atoi(argv[2]);
    }

    if (argc > 3) {
        iters = atoi(argv[3]);
    }

    if (n > MAX_PARTICLES) {
        fprintf(stderr, "FATAL: the maximum number of particles is %d\n", MAX_PARTICLES);
        return EXIT_FAILURE;
    }

    printf("it can use %d threads\n",omp_get_max_threads());
    printf("%d particles\n",n);
    printf("%d step\n",nsteps);
    printf("%d iters\n",iters);

    //time info vars
    double *timings = (double*)malloc(sizeof(*timings)*iters);
    double max_time = INT_MIN;
    double min_time = INT_MAX;
    double total_time = 0.0;
    // value info vars
    double *results = (double*)malloc(sizeof(*results)*iters);
    double max_result = INT_MIN;
    double min_result = INT_MAX;
    double total_result = 0.0;
    // inters to watch
    for(int i = 0; i<iters;i++) {
        printf("----start-----\n");

        // reinit random seed
        srand(1234);
        init_sph(n);
        // get start time
        double start = hpc_gettime();
        float avg = 0;
        // start calculating
        for (int s=0; s<nsteps; s++) {
            update();
            /* the average velocities MUST be computed at each step, even
            if it is not shown (to ensure constant workload per
            iteration) */
            avg = avg_velocities();
            if (s % 10 == 0)
                printf("step %5d, avgV=%f\n", s, avg);
        }
        // get end time
        double elapsed = hpc_gettime()-start;
        printf("time elapsed: %f s\n",elapsed);

        // save timing info
        timings[i] = elapsed;
        total_time += elapsed;
        max_time = maxab(max_time, elapsed);
        min_time = minab(min_time, elapsed);
        // save results info
        results[i] = avg;
        total_result += avg;
        max_result = maxab(max_result, avg);
        min_result = minab(min_result, avg);
    }
    // calculate timing info
    double avg_time = total_time/iters;
    double var_time = 0.0;
    for (int i = 0; i < iters; i++)
    {
        var_time += pow(timings[i] - avg_time, 2);
    }
    var_time = var_time/iters;

    // calculate value info
    double avg_result = total_result/iters;
    double var_result = 0.0;
    for (int i = 0; i < iters; i++)
    {
        var_result += pow(results[i] - avg_result, 2);
    }
    var_result = var_result/iters;

    // print timing info
    printf("--- OVERALL TIME info ---\n");
    printf("min time: %f s\n",min_time);
    printf("max time: %f s\n",max_time);
    printf("average time: %f s\n",avg_time);
    printf("variance time: %f (std_dev: %f) s\n",var_time, sqrt(var_time));
    printf("--- OVERALL RESULTS info ---\n");
    printf("min result: %f\n",min_result);
    printf("max result: %f\n",max_result);
    printf("average result: %f\n",avg_result);
    printf("variance result: %f (std_dev: %f)\n",var_result, sqrt(var_result));
    free(timings);
    free(results);

    // free SoA
    free(particles.x);
    free(particles.y);
    free(particles.vx);
    free(particles.vy);
    free(particles.fx);
    free(particles.fy);
    free(particles.rho);
    free(particles.p);

    return EXIT_SUCCESS;
}
