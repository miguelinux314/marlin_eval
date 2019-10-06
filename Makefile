CFLAGS += -std=gnu++11 -Wall -Wextra -Wcast-qual -Wcast-align -Wstrict-aliasing=1 -Wswitch-enum -Wundef -pedantic  -Wfatal-errors

CFLAGS += -I./src
CFLAGS += -I./fapec/fapec_lib
CFLAGS += -I./fapec/fapec_api

CFLAGS += `pkg-config opencv --cflags`
LFLAGS += `pkg-config opencv --libs`

LFLAGS += -lboost_system -lboost_program_options -lboost_serialization
LFLAGS += -lz -lrt

LCODECS += -lsnappy -lCharLS -lzstd -llz4 -llzo2 -lpng -lturbojpeg -lwebp
LCODECS += -L . -l:libFapecCompress.so -l:libFapecDecompress.so


CFLAGS += -fopenmp
LFLAGS += -lgomp


#CFLAGS += -Ofast

#CFLAGS += -g
#CFLAGS += -g
#CFLAGS += -g -Ofast -march=native
#CFLAGS += -g -fno-unroll-loops
CFLAGS += -g -Ofast

CFLAGS += -I./ext
LFLAGS += $(wildcard ./ext/*.a)




#CFLAGS += -DNDEBUG
CFLAGS += -fno-inline-functions
#CFLAGS += -frename-registers -fopenmp
#CFLAGS += -fno-unroll-loops
#CFLAGS += -funroll-all-loops
#CFLAGS += -fno-align-loops
#CFLAGS += -fno-align-labels
#CFLAGS += -fno-tree-vectorize
#CFLAGS += -falign-functions -falign-labels -falign-jumps -falign-loops -frename-registers -finline-functions
#CFLAGS += -fomit-frame-pointer
#CFLAGS += -fmerge-all-constants -fmodulo-sched -fmodulo-sched-allow-regmoves -funsafe-loop-optimizations -floop-unroll-and-jam

CODECS :=  $(patsubst %.cc,%.o,$(wildcard ./src/codecs/*.cc))


CXX = g++
#CXX = clang++-3.3 -D__extern_always_inline=inline -fslp-vectorize
#CXX = icpc -fast -auto-ilp32 -xHost -fopenmp

all: data ./bin/benchmark ./bin/analyzeMarlin

.PHONY: data ext show prof clean realclean

.SECONDARY: $(CODECS)

data:
	@$(MAKE) -C rawzor --no-print-directory
	
ext:
	@$(MAKE) -C ext --no-print-directory
	
./src/codecs/marlin2018.o: ./src/codecs/marlin2018.cc ./src/codecs/marlin2018.hpp ./src/util/*.hpp  ./src/marlinlib/marlin.hpp
	@echo "CREATING $@"
	@$(CXX) -c -o $@ $< $(CFLAGS)

./src/codecs/%.o: ./src/codecs/%.cc ./src/codecs/%.hpp ./src/util/*.hpp
	@echo "CREATING $@"
	@$(CXX) -c -o $@ $< $(CFLAGS)

./bin/%: ./src/%.cc $(CODECS) ext
	@echo "CREATING $@" $(CODECS) ext
	@$(CXX) -o $@ $< $(CODECS) $(LCODECS) $(CFLAGS) $(LFLAGS) -l:libFapecCompress.so -l:libFapecDecompress.so


prof: ./bin/dcc2017
	 valgrind --dsymutil=yes --cache-sim=yes --branch-sim=yes --dump-instr=yes --trace-jump=no --tool=callgrind --callgrind-out-file=callgrind.out ./eval 
	 kcachegrind callgrind.out

clean:
	rm -f $(CODECS) eval out.tex out.aux out.log out.pdf callgrind.out bin/*

realclean:
	@make -C rawzor clean
	@make -C ext clean
	rm -f $(CODECS) eval out.tex out.aux out.log out.pdf callgrind.out bin/*
