#include <stdio.h>
#include <string>
#include <iostream>
#include <time.h>
#include "MyMatrix.h"

using namespace std;

// *** note I am currently not doing any input checking, thus make sure you follow the example usage =)***

int main(int argc, char** argv)
{

   string FUNC = argv[1];

   // set random to read based on time
   srand((unsigned)time(0));

   if (FUNC == "--help")
   {
      cout << endl << "EXAMPLE USAGE:" << endl;
      cout << "1) 'runme -- help' -> prints this" << endl << endl;
      cout << "2) 'runme --mult 0' -> use the matrices in 'mult_input1.txt' and 'mult_input2.txt' and multiply them." << endl << endl;
      cout << "3) 'runme --mult 1 n p m' ->  generate two new random matrices A which is nxp, and B which is pxm\n, \
                           (blows away 'mult_input1.txt' and 'mult_input2.txt') and multiply them. " << endl << endl;
      cout <<  "4) 'runme --raise2 0 T' -> raise the square martix in 'raise_input.txt' to the T power. Currently, T must be a power of 2." << endl << endl;
      cout << "5) 'runme --raise2 1 T n' -> generate a new random square matrix with dimension nxn (blows away 'input.txt') and raise it to T.\n \
                           Currently, T must be a power of 2." << endl << endl  << flush;
   }
   else if (FUNC == "--mult")
   {
      bool genNew = atoi(argv[2]);
      if (!genNew)
         MyMatrix::multMats("mult_input1.txt", "mult_input2.txt", "GPUOutput.txt",genNew,0,0,0);
      else
         MyMatrix::multMats("mult_input1.txt", "mult_input2.txt", "GPUOutput.txt",genNew,atoi(argv[3]),atoi(argv[4]),atoi(argv[5]));
   }
   else if(FUNC == "--raise2")
   {
      bool genNew = atoi(argv[2]);
      if (!genNew)
         MyMatrix::raisePowerOf2("raise_input.txt", "GPUOutput.txt",genNew,atoi(argv[3]),0);
      else
         MyMatrix::raisePowerOf2("raise_input.txt", "GPUOutput.txt",genNew,atoi(argv[3]), atoi(argv[4]));
   }
     else
      { cout << "Function not valid: please see --help" << endl << flush;
        exit(1);
      }

   return 0;
}

