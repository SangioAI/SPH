# SPH

CUDA and OpenMP versions of [SPH](https://github.com/cerrno/mueller-sph) (Smoothed Particle Hydrodynamics) serial algorithm.

# Dependencies

The GCC compiler is used to build the code.

# Build
For this project are available: 
- 4 OpenMP parallelized versions
- 2 CUDA parallelized versions.

> [!Note]
> The `make` command must be run in the *src* folder.

## OpenMP versions
To build all OpenMP versions: `make openmp`
    
To build them indipendently:
<br/>
```
    gcc -std=c99 -fopenmp -Wall -Wpedantic omp-sph.c -o omp-sph -lm
    gcc -std=c99 -fopenmp -Wall -Wpedantic omp-sphv2.c -o omp-sph -lm
    gcc -std=c99 -fopenmp -Wall -Wpedantic omp-sphv3.c -o omp-sph -lm
    gcc -std=c99 -fopenmp -Wall -Wpedantic omp-sphv4.c -o omp-sph -lm
```

## CUDA versions
To build all CUDA versions: `make cuda`
    
In alternativa:
```
    nvcc cuda-sph.cu -o cuda-sph
    nvcc cuda-sphv2.cu -o cuda-sphv2
```
# Results

See the [Report](Report.pdf).
