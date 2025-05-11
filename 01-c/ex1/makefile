CC = c++
OPT = -std=c++14

# set main here
MAIN = stack_n_heap.c

##################################################################################################################
TARGETDIR   := bin

all: $(TARGETDIR)/stack_n_heap

debug: OPT += -O0 -DDEBUG -g
debug: all

$(TARGETDIR)/stack_n_heap: ${MAIN}
	@mkdir -p $(@D)
	$(CC) $^ -o $@ $(INCLUDE) $(OPT)

clean:
	rm -r $(TARGETDIR)/
