CFLAGS += -std=gnu++11 -Wall -Wextra -Wcast-qual -Wcast-align -Wstrict-aliasing=1 -Wswitch-enum -Wundef -pedantic  -Wfatal-errors

CFLAGS += -I./src

CFLAGS += `pkg-config opencv --cflags`
LFLAGS += `pkg-config opencv --libs`

LFLAGS += -lz -lrt

LCODECS += -lsnappy -lzstd -llz4 -llzo2
CFLAGS += -g -Ofast

CFLAGS += -I./ext
LFLAGS += $(wildcard ./ext/*.a)

CODECS :=  $(patsubst %.cc,%.o,$(wildcard ./src/codecs/*.cc))


CXX = g++

all: ./bin/benchmark

.PHONY: data ext show prof clean realclean

.SECONDARY: $(CODECS)
	
	
./src/codecs/marlin2018.o: ./src/codecs/marlin2018.cc ./src/codecs/marlin2018.hpp ./src/util/*.hpp  ./src/marlinlib/marlin.hpp
	@echo "CREATING $@"
	@$(CXX) -c -o $@ $< $(CFLAGS)

./src/codecs/%.o: ./src/codecs/%.cc ./src/codecs/%.hpp ./src/util/*.hpp
	@echo "CREATING $@"
	@$(CXX) -c -o $@ $< $(CFLAGS)

./bin/%: ./src/%.cc $(CODECS) ext
	@echo "CREATING $@" $(CODECS) ext
	@$(CXX) -o $@ $< $(CODECS) $(LCODECS) $(CFLAGS) $(LFLAGS)

ext:
	@cd ext && make && cd ..

clean:
	rm -rf $(CODECS) eval out.tex out.aux out.log out.pdf callgrind.out bin/* cmake-build-debug
	find -name '*.a' -exec rm -v {} \;
