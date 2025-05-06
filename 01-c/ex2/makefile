# create a makefile variable named CC for your C/C++ compiler (es gcc, c++, ... )

# create a makefile variable named OPT with your favorite C flags (at least with -std=c99 -O3)


BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
BATCH_OUT_FOLDER := outputs

MAIN    =  get_time_test.c
OBJECTS = $(OBJ_FOLDER)/my_time_lib.o

all: $(BIN_FOLDER)/get_time_test

#create the target 'bin/get_time_test' which needs 'get_time_test.c' and 'my_time_lib.o', generate the folder 'bin' and compile get_time_test.c using CC and OPT




$(OBJ_FOLDER)/my_time_lib.o: $(SRC_FOLDER)/my_time_lib.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/my_time_lib.c -o $@ $(OPT)


#create the target 'clean' which recursively remove the folders 'bin' and 'obj'


clean_batch_outputs:
	rm $(BATCH_OUT_FOLDER)/*

clean:
	rm -rf ${BIN_FOLDER}
	rm -rf ${OBJ_FOLDER}
