#Copyright (c) 2018 Tommy Carpenter

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.

# Last tested on Arch Linux using cuda9.2 in Sept 2018

#Arch
#CUDAPATH=/opt/cuda/lib64
#Ubuntu
CUDAPATH=/usr/local/cuda/lib64

GPPOPTIONS += -O -fexceptions

GPPOPTIONS += -pedantic

NVCCOPTIONS += -Xlinker -rpath,$(CUDAPATH)

LIBRARIES += -L$(CUDAPATH) -lcudart -lcuda

# Debug/release configuration
ifeq ($(dbg),1)
	GPPOPTIONS  += -g -D_DEBUG
	NVCCOPTIONS += -D_DEBUG -G -g
else 
	GPPOPTIONS  += -O2 -fno-strict-aliasing
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
