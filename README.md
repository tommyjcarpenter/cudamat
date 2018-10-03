# cudamat

This is a C++/CUDA (C) *demonstration/learning* library. It is not intended to be used as a real deal linear algebra library. I wrote it only to expiriment around with writing CUDA, and to some extent, C++.

# Compiling

    make

# Running
E.g.,

    ./runme --mult 1 31 323439 19;
    ./runme --raise2 1 4 555

Cleanup:

    make clean

# Testing

    make tester
    ./tester
    make tester-clean
