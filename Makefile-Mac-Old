#Copyright (c) 2013 Tommy Carpenter

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


#NOTE THIS IS FOR MAC ONLY.... You will need to write your own makefile if using another OS

GPPOPTIONS += -m32 -O -fPIC -fexceptions -DNDEBUG -DIL_STD 

NVCCOPTIONS += -Xlinker -rpath,/usr/local/cuda/lib #-arch sm_13

LIBRARIES += -L/usr/local/cuda/lib -lcuda -lcudart -L/Applications/CPLEX/cplex/lib/x86_darwin9_gcc4.0/static_pic  -lilocplex -lcplex  -L/Applications/CPLEX/concert/lib/x86_darwin9_gcc4.0/static_pic  -lconcert  -m32 -lm -lpthread -framework CoreFoundation -framework IOKit
	
# Debug/release configuration
ifeq ($(dbg),1)
	GPPOPTIONS  += -g -D_DEBUG
	NVCCOPTIONS += -D_DEBUG -G -g
else 
	GPPOPTIONS  += -O2 -fno-strict-aliasing
	NVCCOPTIONS += --compiler-options -fno-strict-aliasing 
endif

all:
	DYLD_LIBRARY_PATH=/usr/local/cuda/lib
	g++ -c *.cpp $(GPPOPTIONS) $(GPPINCLUDES)
	/usr/local/cuda/bin/nvcc -c *.cu $(NVCCOPTIONS)
	g++ -o runme *.o $(LIBRARIES)			
clean:
	rm *.o
