## Makefile per la compilazione dei programmi di esempio del corso di
## High Performance Computing 2022/2023, corso di laurea in Ingegneria
## e Scienze Informatiche, Universita' di Bologna.
##
## Ultima modifica: 2022-09-12, Moreno Marzolla <moreno.marzolla(at)unibo.it>
##
## `make` senza parametri genera tutti i target
## `make help` per mostrare i target disponibili

EXE_OMP:=$(basename $(wildcard omp-*.c))
EXE_CUDA:=$(basename $(wildcard cuda-*.cu))
EXE:=$(EXE_OMP) $(EXE_MPI) $(EXE_SERIAL) $(EXE_SIMD) $(EXE_OPENCL) $(EXE_CUDA)
CFLAGS+=-std=c99 -Wall -Wpedantic
NVCC?=nvcc
NVCFLAGS+=

all: openmp cuda

help:
	@echo
	@echo "Target disponibili:"
	@echo
	@echo "     help    stampa questo messaggio di aiuto"
	@echo "      all    compila tutto"
	@echo "   openmp    compila tutti i programim OpenMP"
	@echo "     cuda    compila tutti i programmi CUDA"
	@echo "    clean    rimuove gli eseguibili e alcuni file temporanei"
	@echo
	@echo "Per scaricare CUDA SDK per la propria piattaforma:"
	@echo
	@echo "    https://developer.nvidia.com/cuda-downloads"
	@echo

$(EXE_OMP): CFLAGS+=-fopenmp
$(EXE_OMP): LDLIBS+=-lm
openmp: $(EXE_OMP)

cuda: $(EXE_CUDA)

% : %.cu
	$(NVCC) $(NVCFLAGS) $< -o $@

clean:
	\rm -f $(EXE) *.o *~ *.pbm *.s *.lst

