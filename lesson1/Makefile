CC=gcc
OBLAS_LIB_HOME=/opt/shares/openfoam/software/OpenBLAS/0.3.23-GCC-12.3.0

LIBS=-L$(OBLAS_LIB_HOME)/lib
INCLUDE=-I$(OBLAS_LIB_HOME)/include/ -lopenblas


program: program.c
	$(CC) $^ -o $@ $(LIBS) $(INCLUDE)
