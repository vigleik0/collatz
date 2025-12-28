/*
 OpenCL rewrite of CUDA program. Compile with
  g++ -std=c++17 -O3 -DNDEBUG collatz_openclv2.cpp -lOpenCL -o collatz_openclv2
 The compilation relies on having the file compute_kernel2.cl in the same directory.
 Note that some constants are duplicated in the two files, so take care when changing
 them. But MAXBITS and CUTOFFPOWER are safe to change.

 Also note that changing any of the constants will change the hash.
*/

#include <CL/cl.h>
// #include <cuda_runtime.h>
#include <cstdint>
#include <array>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>
#include <sstream>
#include <cstring>
#include <fstream>   // for std::ifstream
#include <chrono>
// #include <time.h>
#include <unistd.h>
#include <random>

// This is purely to make the Rust names for integer types work:
using u8 = std::uint8_t;
using i16 = std::int16_t;
using u16 = std::uint16_t;
using i32 = std::int32_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using isize = std::size_t;
using u128 = __uint128_t;

// Adjust MAXBITS and CUTOFFPOWER to your liking before compiling.
// Experiment with the remaining constants at your own risk.
const u32 MAXBITS = 45;
const u32 CUTOFFPOWER = 100; // Must be at least 96.
const u128 LARGENUMBERCUTOFF = (u128)1 << CUTOFFPOWER;
const u32 U32LARGENUMBERCUTOFF = (u32)1 << (CUTOFFPOWER - 96);
const u32 STEPSIZEINITIAL = 24; // Must be identical to STEPSIZEINITIAL in compute_kernel.cl
const u32 LASTBITS = 10; // Do not change: compute_kernel.cl depends on LASTBITS = 10
const u32 BATCHSIZE = (1 << 16); // Safe to change
const u32 NUMERATOR = 306; // NUMERATOR / DENOMINATOR is a rational
                           // approximation to log(2)/log(3)
const u32 DENOMINATOR = 485;
const u32 LASTBITSTEPS = ((1 << LASTBITS) + 31)/32;
const u32 HASHTABLESIZE = 256;
const u32 SPOTCHECKFREQUENCY = 1000;

constexpr u32 THREEPOWERSU32[17] = {1, 3, 9, 27, 81, 243, 729, 2187, 6561,
                                    19683, 59049, 177147, 531441, 1594323,
                                    4782969, 14348907, 43046721};

u128 THREEPOWERS[80]; // Initialised in main()

#define MAXBITSINITIAL (MAXBITS - LASTBITS)
#define ONETHIRD 2863311531 // Inverse to 3 in the units of Z/(2^32)
#define THREEPOWEROFFSET (MAXBITSINITIAL * NUMERATOR / DENOMINATOR + 1)
#define INITIALBITARRAYSIZE ((1 << (STEPSIZEINITIAL - 5)) + ((1 << LASTBITS) + 31)/32)
#define ctzu32 __builtin_ctz
#define ctzu64 __builtin_ctzll

// Setup the random engine once at the start of your program
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(1, SPOTCHECKFREQUENCY);

struct PrecomputedData {
    std::array<std::vector<u32>, 16> initialbitarray;
    std::array<u32, 16> magicnumbers; // Inverses of certain powers of 3
    std::array<std::vector<u32>, 9> mod9sieve;
    std::array<std::vector<u32>, 16> bigpairvector2;
    u32 complete_hash;
    u32 spot_check_count;
};

// Function declarations
void precompute(PrecomputedData* data);
u64 search1(u128 base, u32 sigfig, u32 threepower, u128 number, u32 doubleeven, u32 numevens,
            PrecomputedData* data, cl_mem d_flat, cl_mem d_flatmod9sieve, cl_context ctx,
            cl_command_queue queue, cl_kernel compute_kernel_cl);
u32 search2(u128 base, u32 threepower, u128 number, PrecomputedData* data, u32 bigindex);
u32 search3(u128 base, u128 number, PrecomputedData* data);
u64 launchkernel(PrecomputedData* data, u32 bigindex, cl_mem d_flat, cl_mem d_flatmod9sieve,
                 u32 threepower, cl_context ctx, cl_command_queue queue, cl_kernel compute_kernel_cl);

int main() {
    std::string line;
    std::getline(std::cin, line);

    std::istringstream iss(line);
    u64 arg1, arg2, arg3, arg4, arg5, arg6;
    iss >> arg1 >> arg2 >> arg3 >> arg4 >> arg5 >> arg6;

    // We initialise THREEPOWERS. There is probably a way to do it at compile time,
    // but it's cheap so we don't care.
    u128 a = 1; THREEPOWERS[0] = a;
    for (isize i = 1; i < 80; i++) { a *= 3; THREEPOWERS[i] = a; }

    PrecomputedData data {
        std::array<std::vector<u32>, 16>{},
        std::array<u32, 16>{},
        std::array<std::vector<u32>, 9>{},
        std::array<std::vector<u32>, 16>{},
        0,
        0,
    };

    for (auto& v : data.initialbitarray)
        v.resize(INITIALBITARRAYSIZE);

    for (auto& v : data.mod9sieve)
        v.resize(LASTBITSTEPS);

    u32 magicnumber = 1;
    for (isize i = 1; i < 128; i++) {
        magicnumber *= ONETHIRD;
        if (i >= THREEPOWEROFFSET && i - THREEPOWEROFFSET < 16) {
            data.magicnumbers[i - THREEPOWEROFFSET] = magicnumber;
        }
    }

    precompute(&data);
    fprintf(stderr, "Finished precompute\n");

    cl_int err;

    // 1. Platform
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS)
        throw std::runtime_error("clGetPlatformIDs failed");

    // 2. Device
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS)
        throw std::runtime_error("clGetDeviceIDs failed");

    // 3. Context
    cl_context ctx = clCreateContext(
        nullptr,
        1,
        &device,
        nullptr,
        nullptr,
        &err
    );
    if (err != CL_SUCCESS)
        throw std::runtime_error("clCreateContext failed");

    // 4. Command queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(
        ctx,
        device,
        nullptr,  // no special properties
        &err
    );

    if (err != CL_SUCCESS)
        throw std::runtime_error("clCreateCommandQueue failed");

    std::ifstream file("compute_kernel2.cl");
    std::string src((std::istreambuf_iterator<char>(file)),
                    std::istreambuf_iterator<char>());
    const char* src_cstr = src.c_str();
    cl_program program = clCreateProgramWithSource(ctx, 1, &src_cstr, nullptr, &err);
    err = clBuildProgram(program, 1, &device, "", nullptr, nullptr);

    if (err != CL_SUCCESS) {
        // Get build log size
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);

        std::cerr << "OpenCL build log:\n" << log << std::endl;
        throw std::runtime_error("clBuildProgram failed");
    }

    fprintf(stderr, "Finished clBuildProgram\n");

    cl_kernel compute_kernel_cl = clCreateKernel(program, "compute_kernel_cl", &err);

    if (err != CL_SUCCESS) {
        fprintf(stderr, "compute_kernel_cl failed\n");
    }

    // Put initialbitarray on the GPU:
    isize rows = data.initialbitarray.size();
    isize cols = data.initialbitarray[0].size();

    // Flatten
    std::vector<u32> flat(rows * cols);
    for (isize r = 0; r < rows; ++r) {
        std::memcpy(&flat[r * cols],
                    data.initialbitarray[r].data(),
                    cols * sizeof(u32));
    }

    cl_mem d_flat = clCreateBuffer(
        ctx,
        CL_MEM_READ_ONLY,
        flat.size() * sizeof(u32),
        nullptr,
        &err
    );

    if (err != CL_SUCCESS)
        throw std::runtime_error("clCreateBuffer(d_flat) failed");

    // Copy host data to device
    err = clEnqueueWriteBuffer(
        queue,          // command queue
        d_flat,         // buffer
        CL_FALSE,        // blocking write
        0,              // offset
        flat.size() * sizeof(u32),
        flat.data(),    // host pointer
        0, nullptr, nullptr
    );
    if (err != CL_SUCCESS)
        throw std::runtime_error("clEnqueueWriteBuffer(d_flat) failed");

    // Put mod9sieve on the GPU:
    rows = 9;
    cols = data.mod9sieve[0].size();

    // Flatten
    std::vector<u32> flatmod9sieve(rows * cols);
    for (isize r = 0; r < rows; ++r) {
        std::memcpy(&flatmod9sieve[r * cols],
                    data.mod9sieve[r].data(),
                    cols * sizeof(u32));
    }

    cl_mem d_flatmod9sieve = clCreateBuffer(
        ctx,
        CL_MEM_READ_ONLY,
        flatmod9sieve.size() * sizeof(u32),
        nullptr,
        &err
    );

    if (err != CL_SUCCESS)
        throw std::runtime_error("clCreateBuffer(d_flatmod9sieve) failed");

    // Copy host data to device
    err = clEnqueueWriteBuffer(
        queue,          // command queue
        d_flatmod9sieve,         // buffer
        CL_FALSE,        // blocking write
        0,              // offset
        flatmod9sieve.size() * sizeof(u32),
        flatmod9sieve.data(),    // host pointer
        0, nullptr, nullptr
    );
    if (err != CL_SUCCESS)
        throw std::runtime_error("clEnqueueWriteBuffer(d_flatmod9sieve) failed");

    char name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    std::cerr << "OpenCL device: " << name << std::endl;

    // Do search1(3, 2, 2, 9, 0, 0) for initial one.
    u64 numcases = search1((u128)arg1, (u32)arg2, (u32)arg3, (u128)arg4, (u32)arg5, (u32)arg6, &data, d_flat, d_flatmod9sieve, ctx, queue, compute_kernel_cl);

    // Finally we do the remaining cases.
    for (isize a = 0; a < 16; a++) {
        numcases += launchkernel(&data, a, d_flat, d_flatmod9sieve, THREEPOWEROFFSET + a, ctx, queue, compute_kernel_cl);
    }
    clFinish(queue);

    fprintf(stderr, "Number of kernel launches = %ld \n", numcases);
    fprintf(stderr, "Spot checks = %u. Complete hash = %X.\n", data.spot_check_count, data.complete_hash);
}


// Check if there is a string of 1's followed by two 0's, starting from position i of upordown.
// This is for the Odd-Even-Even sieve.
bool doubleeven(u32 upordown, u64 i) {
    u32 a = ctzu64((upordown >> i) + 1);
    if (i + a + 2 >= STEPSIZEINITIAL) { return false; }
    if ((upordown >> (i + a)) & 0b11 == 0) { return true; }

    return false;
}

void precompute(PrecomputedData* data) {
    i32 firstcutoff = (i16)(DENOMINATOR * THREEPOWEROFFSET - NUMERATOR * MAXBITSINITIAL);
    i32 cutoffs[16];
    for (isize i = 0; i < 16; i++) { cutoffs[i] = firstcutoff + i * DENOMINATOR; }

    u64 wraparoundcutoff = ((1 << LASTBITS) + 31)/32;
    u64 wraparoundadd    = 1 << (STEPSIZEINITIAL - 5);

    for (u32 n0 = 0; n0 < (1 << STEPSIZEINITIAL); n0++) {
        u32 upordown = 0;
        u32 n = n0;
        for (isize i = 0; i < STEPSIZEINITIAL; i++) {
            if (n % 2 == 0) {
                n = n + n/2;
                upordown |= 1 << i;
            } else {
                n = n/2 + 1;
            }
        }
        i32 curval = 0;
        i32 maxval = 0;
        u32 numevens = 0;
        for (u32 i = 0; i < STEPSIZEINITIAL; i++) {

            if ((upordown & (1 << i)) != 0) {
                // Check if the Odd-Even-Even Sieve might be in effect.
                if (curval + NUMERATOR > maxval) {
                    if (doubleeven(upordown, i) == true) {
                        maxval = curval + NUMERATOR;
                    }
                }
                curval -= (DENOMINATOR - NUMERATOR);
                numevens = 0;
            } else {
                curval += NUMERATOR;
                numevens += 1;
                // Check if hte Path-Mergin Sieve might be in effect.
                if (numevens % 2 == 0) {
                    if (curval + (DENOMINATOR - NUMERATOR) > maxval) {
                        maxval = curval + (DENOMINATOR - NUMERATOR);
                    }
                }
            }
            if (curval > maxval) { maxval = curval; }
        }
        // Now we record the result in each of the 16 bitvectors BV_i
        for (isize a = 0; a < 16; a++) {
            if (maxval <= cutoffs[a]) {
                u64 index = (u64)(n0 * data->magicnumbers[a]) & ((1 << STEPSIZEINITIAL) - 1);

                data->initialbitarray[a][index/32] |= (u64)1 << (index % 32);

                if (index/32 < wraparoundcutoff) {
                    data->initialbitarray[a][index/32 + wraparoundadd] |= (u32)1 << (index % 32);
                }
            }
        }
    }

    u64 redmod9step = u64(((u128)1 << MAXBITSINITIAL) % 9);

    // Prepare the Mod 9 Preimage Sieve
    for (isize a = 0; a < 9; a++) {
        isize redmod9 = a;
        for (isize i = 0; i < ((isize)1 << LASTBITS); i++) {
            if (redmod9 == 0 || redmod9 == 1 || redmod9 == 3 || redmod9 == 6 || redmod9 == 7) {
                data->mod9sieve[a][i/32] |= (u64)1 << (i % 32);
            }
            redmod9 += redmod9step;
            if (redmod9 >= 9) { redmod9 -= 9; }
        }
    }
}

u64 search1(u128 base, //n0
            u32 sigfig, //k, with n0 < 2^k
            u32 threepower, //f_k(n_0)
            u128 number, // T^k(base)
            u32 doubleeven, // For Odd-Even-Even Sieve
            u32 numevens, // For mod 3 calculation needed for Path-Merging Sieve.
            PrecomputedData* data, // Needed later
            cl_mem d_flat, // Reference to the BV_i bitvectors on the GPU
            cl_mem d_flatmod9sieve, // Reference to the Mod 9 Preimage Sieve on the GPU
            cl_context ctx, // The last 3 arguments are needed for the kernel only
            cl_command_queue queue,
            cl_kernel compute_kernel_cl) {
    // Induction sieve:
    if (DENOMINATOR * threepower <= NUMERATOR * sigfig) { return 0; }
    // Mod 3 Preimage Sieve:
    if (numevens >= 2 && numevens % 2 == 0) {
        if (NUMERATOR * (sigfig - 1) >= DENOMINATOR * (threepower - 1)) { return 0; }
    }

    // If we've reached k == N - A we're done. We either store the result for batch
    // processing on the GPU or, in rare cases, process the case on the CPU.
    if (sigfig == MAXBITSINITIAL) {
        u32 bigindex = threepower - THREEPOWEROFFSET;
        if (bigindex >= 16) {
            data->complete_hash ^= search2(base, threepower, number, data, bigindex);
            return 0;
        }
        data->bigpairvector2[bigindex].push_back((u32)(base));
        data->bigpairvector2[bigindex].push_back((u32)(base >> 32));
        data->bigpairvector2[bigindex].push_back((u32)(base >> 64));
        data->bigpairvector2[bigindex].push_back((u32)(base >> 96));
        data->bigpairvector2[bigindex].push_back((u32)(number));
        data->bigpairvector2[bigindex].push_back((u32)(number >> 32));
        data->bigpairvector2[bigindex].push_back((u32)(number >> 64));
        data->bigpairvector2[bigindex].push_back((u32)(number >> 96));

        if (data->bigpairvector2[bigindex].size() == 8 * BATCHSIZE) {
            // return 1;
            // return launchkernel(data, bigindex, d_flat, d_flatmod9sieve, threepower, ctx, queue, compute_kernel_cl);
            return launchkernel(data, bigindex, d_flat,
                 d_flatmod9sieve, threepower, ctx, queue, compute_kernel_cl);
        }
        return 0;
    }

    // Check if the Odd-Even-Even Sieve is in play. If so, pass doubleeven = 1 to
    // the odd branch. If doubleeven is already 1, we also pass doubleeven = 1 to
    // the odd branch and pass doubleeven = 2 to the even branch.
    // If doubleeven = 2 we don't take the even branch.
    u32 doubleinplay = 0;
    if (NUMERATOR * (sigfig + 1) >= DENOMINATOR * threepower) { doubleinplay = 1; }

    u128 numberA = number;
    u128 numberB = number + THREEPOWERS[threepower];

    if (number % 2 == 0) {
        u64 resL = search1(base,                       sigfig + 1, threepower + 1,
                           numberA + numberA/2, (doubleeven | doubleinplay) & (~2), 0, data, d_flat, d_flatmod9sieve, ctx, queue, compute_kernel_cl);
        u64 resR = 0;
        if (doubleeven < 2) { resR = search1(base | ((u128)1 << sigfig), sigfig + 1, threepower,
                           numberB/2 + 1,  doubleeven << 1, numevens + 1,      data, d_flat, d_flatmod9sieve, ctx, queue, compute_kernel_cl); }
        return resL + resR;
    } else {
        u64 resL = 0;
        if (doubleeven < 2) { resL = search1(base,                       sigfig + 1, threepower,
                           numberA/2 + 1, doubleeven << 1, numevens + 1,       data, d_flat, d_flatmod9sieve, ctx, queue, compute_kernel_cl); }
        u64 resR = search1(base | ((u128)1 << sigfig), sigfig + 1, threepower + 1,
                           numberB + numberB/2 ,(doubleeven | doubleinplay) & (~2), 0, data, d_flat, d_flatmod9sieve, ctx, queue, compute_kernel_cl);
        return resL + resR;
    }
}

u64 launchkernel(PrecomputedData* data, u32 bigindex, cl_mem d_flat,
                 cl_mem d_flatmod9sieve, u32 threepower, cl_context ctx, cl_command_queue queue, cl_kernel compute_kernel_cl) {
    cl_uint magic = data->magicnumbers[bigindex];
    cl_uint bigindex_h = static_cast<cl_uint>(bigindex);
    u128 threepowervalue = THREEPOWERS[threepower];

    isize n = data->bigpairvector2[bigindex].size();

    if (n == 0) { return 0; }
    // data->bigpairvector2[bigindex].clear(); return 1;

    cl_int err;

    cl_mem d_input = clCreateBuffer(
        ctx,
        CL_MEM_READ_ONLY,
        n * sizeof(u32),
        nullptr,
        &err
    );
    if (err != CL_SUCCESS)
        throw std::runtime_error("clCreateBuffer(d_input) failed");

    err = clEnqueueWriteBuffer(
        queue,
        d_input,
        CL_FALSE,
        0,
        n * sizeof(u32),
        data->bigpairvector2[bigindex].data(),
        0,
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS) {
        std::cerr << "clEnqueueWriteBuffer failed: " << err << std::endl;
        throw std::runtime_error("clEnqueueWriteBuffer failed");
    }

    cl_uint h_hash[HASHTABLESIZE] = {};

    // 1. Allocate the buffer on the GPU (Device)
    cl_mem d_hash = clCreateBuffer(ctx,
                                   CL_MEM_READ_WRITE,
                                   HASHTABLESIZE * sizeof(cl_uint),
                                   NULL, &err);

    // 2. Copy from Host to Device
    err = clEnqueueWriteBuffer(queue,
                              d_hash,
                              CL_FALSE,
                              0,
                              HASHTABLESIZE * sizeof(cl_uint),
                              h_hash,
                              0, NULL, NULL);

    if (err != CL_SUCCESS) {
        std::cerr << "clEnqueueWriteBuffer failed: " << err << std::endl;
        throw std::runtime_error("clEnqueueWriteBuffer failed");
    }

    cl_uint threads_needed = n / 8;

    size_t local_size  = 256;
    size_t global_size = ((size_t)threads_needed + local_size - 1)
                        / local_size * local_size;

    int arg = 0;

    cl_uint tp_0 = (u32)(threepowervalue);
    cl_uint tp_1 = (u32)(threepowervalue >> 32);
    cl_uint tp_2 = (u32)(threepowervalue >> 64);
    cl_uint tp_3 = (u32)(threepowervalue >> 96);

    cl_uint mbi = MAXBITSINITIAL;
    cl_uint lnc = U32LARGENUMBERCUTOFF;

    clSetKernelArg(compute_kernel_cl, arg++, sizeof(cl_mem), &d_flat);
    clSetKernelArg(compute_kernel_cl, arg++, sizeof(cl_mem), &d_flatmod9sieve);
    clSetKernelArg(compute_kernel_cl, arg++, sizeof(cl_mem), &d_input);
    clSetKernelArg(compute_kernel_cl, arg++, sizeof(cl_uint), &bigindex_h);
    clSetKernelArg(compute_kernel_cl, arg++, sizeof(cl_uint), &threads_needed);
    clSetKernelArg(compute_kernel_cl, arg++, sizeof(cl_uint), &magic);
    clSetKernelArg(compute_kernel_cl, arg++, sizeof(cl_uint), &tp_0);
    clSetKernelArg(compute_kernel_cl, arg++, sizeof(cl_uint), &tp_1);
    clSetKernelArg(compute_kernel_cl, arg++, sizeof(cl_uint), &tp_2);
    clSetKernelArg(compute_kernel_cl, arg++, sizeof(cl_uint), &tp_3);
    clSetKernelArg(compute_kernel_cl, arg++, sizeof(cl_uint), &mbi);
    clSetKernelArg(compute_kernel_cl, arg++, sizeof(cl_uint), &lnc);
    clSetKernelArg(compute_kernel_cl, arg++, sizeof(cl_mem), &d_hash);


    // On my computer, with a Nvidia GPU, this call is blocking even though the documentation
    // says it's not. I've been unable to avoid 100% utilization.
    err = clEnqueueNDRangeKernel(
        queue,
        compute_kernel_cl,
        1,              // 1D kernel
        nullptr,
        &global_size,
        &local_size,
        0,
        nullptr,
        nullptr
    );

    clFinish(queue);

    err = clEnqueueReadBuffer(queue,
                          d_hash,
                          CL_TRUE, // CL_TRUE makes this a blocking call
                          0,
                          HASHTABLESIZE * sizeof(cl_uint),
                          h_hash,
                          0, NULL, NULL);

    if (err != CL_SUCCESS) {
        std::cerr << "clEnqueueReadBuffer failed: " << err << std::endl;
        throw std::runtime_error("clEnqueueReadBuffer failed");
    }

    u32 hash = 0;
    for (int i = 0; i < HASHTABLESIZE; i++) {
        hash ^= h_hash[i];
    }
    data->complete_hash ^= hash;

    if (dis(gen) == 1) {
        u32 hash2 = 0;
        for (int i = 0; i < threads_needed; i++) {
            u128 base =  (u128)(data->bigpairvector2[bigindex][8 * i])
                    | ((u128)(data->bigpairvector2[bigindex][8 * i + 1]) << 32)
                    | ((u128)(data->bigpairvector2[bigindex][8 * i + 2]) << 64);
            u128 number =  (u128)(data->bigpairvector2[bigindex][8 * i + 4])
                        | ((u128)(data->bigpairvector2[bigindex][8 * i + 5]) << 32)
                        | ((u128)(data->bigpairvector2[bigindex][8 * i + 6]) << 64)
                        | ((u128)(data->bigpairvector2[bigindex][8 * i + 7]) << 96);
            hash2 ^= search2(base, threepower, number, data, bigindex);
        }
        // fprintf(stderr, "Spot check: hash = %X, hash2 = %X\n", hash, hash2);
        if (hash != hash2) {
            fprintf(stderr, "Spot check failed\n");
        }
        data->spot_check_count += 1;
    }

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_hash);

    data->bigpairvector2[bigindex].clear();

    return 1;
}

// search2 and search3 do the same as compute_kernel in the case where
// threepower is too big for our 16 precomputed bitvectors. This case
// is rare.
u32 search2(u128 base, u32 threepower, u128 number, PrecomputedData* data, u32 bigindex) {
    u32 hash = 0;
    isize redmod9 = u64(base % 9);


    for (isize a = 0; a < LASTBITSTEPS; a++) {
        u32 temp = data->mod9sieve[redmod9][a];
        if (bigindex < 16) {
            u32 magic = data->magicnumbers[bigindex];
            u32 index = (magic * (u32)number) & ((1 << STEPSIZEINITIAL) - 1);
            u32 offset = index % 32;
                // Mod 9 Preimage Sieve:
            if (offset == 0) {
                temp &= data->initialbitarray[bigindex][index/32 + a];
            } else {
                temp &= (data->initialbitarray[bigindex][index/32 + a] >> offset)
                      | (data->initialbitarray[bigindex][index/32 + a + 1] << (32 - offset));
            }
        }
        while (temp != 0) {
            u128 highbits = 32 * a + ctzu32(temp);
            temp &= temp-1;
            // Pair p = { base + (highbits << MAXBITSINITIAL), number + (highbits * THREEPOWERS[threepower]), };
            hash ^= search3(base + (highbits << MAXBITSINITIAL), number + (highbits * THREEPOWERS[threepower]), data);
        }
    }

    return hash;
}

u32 search3(u128 base, u128 number, PrecomputedData* data) {
    u32 hash = 0;
    while (true) {
        u32 a = ctzu32((u32)number | (1u << 8));
        number = THREEPOWERSU32[a] * (number >> a);

        if (number >= LARGENUMBERCUTOFF) {
            printf("%lX%016lX\n", (u64)(base >> 64), (u64)base);
            number = 2;
        }

        number -= 1;
        u32 b = ctzu32((u32)number | (1ul << 31));
        number = (number >> b) + 1;

        a = ctzu32((u32)number | (1u << 8));
        number = THREEPOWERSU32[a] * (number >> a);

        if (number > LARGENUMBERCUTOFF) {
            printf("%lX%016lX\n", (u64)(base >> 64), (u64)base);
            number = 2;
        }

        number -= 1;
        b = ctzu32((u32)number | (1ul << 31));
        number = (number >> b) + 1;

        hash ^= (u32)number;

        if (number < base) { return hash; }
    }
}

