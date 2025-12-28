The standalone CUDA program collatz_cudav7.cu verifies convergence of the Collatz function up to 2^N for some N.
The program collatz_openclv2.cpp, together with compute_kernel2.cl, does the same using OpenCL.
You should adjust some constants to your liking before compiling.

With a Nvidia RTX 3060 verification up to 2^60 takes about 24 hours.
