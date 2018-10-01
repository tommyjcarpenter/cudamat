#Arch
#CUDAPATH=/opt/cuda/lib64
#Ubuntu
CUDAPATH=/usr/local/cuda/lib64/
GPPINCLUDES=-I /usr/local/cuda/include/

GPPOPTIONS += -fexceptions

#Linting style flags from https://stackoverflow.com/questions/5088460/flags-to-enable-thorough-and-verbose-g-warnings
GPPOPTIONS += -pedantic -W -Wall -Werror -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wundef -Wno-unused -Wconversion -Wpadded  -Wunsafe-loop-optimizations

NVCCOPTIONS += -Xlinker -rpath,$(CUDAPATH)

LIBRARIES += -L$(CUDAPATH) -lcudart -lcuda

# Debug/release configuration
ifeq ($(dbg),1)
	GPPOPTIONS  += -g -D_DEBUG
	NVCCOPTIONS += -D_DEBUG -G -g
else 
	GPPOPTIONS  += -O3 -fno-strict-aliasing
	NVCCOPTIONS += --compiler-options -fno-strict-aliasing 
endif

#Used to need this...
#DYLD_LIBRARY_PATH=$(CUDAPATH)
all:
	g++ -c *.cpp $(GPPOPTIONS) $(GPPINCLUDES)
	nvcc -c *.cu $(NVCCOPTIONS)
	g++ -o runme *.o $(LIBRARIES)
clean:
	rm *.o
	rm runme
