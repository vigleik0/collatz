/*
 Compile with nvcc -std=c++14 -O3 -arch=sm_70 -o collatz_cudav3 collatz_cudav3.cu
 Adjust MAXBITS and LARGENUMBERCUTOFF as appropriate.
 Run with echo "3 2 2 9" | ./collatz_cudav3 for a complete calculation, or
 with "echo $base $sigfig $threepower $number" | ./collatz_cudav3 to get only results whose starting number
 is $base modulo 2^$sigfig. This relies on having computed the number of odd applications to get to $base. $number is $T^$sigfig($base)+1.
 For example, you can set MAXBITS = 72, LARGENUMBERCUTOFF = (u128)1 << 114, and (base, sigfig, threepower, number) = (7340059 24 16 18833016).
 Pipe to  | { echo "ibase=16"; cat; } | bc to get decimal
*/

#include <cuda_runtime.h>
#include <cstdint>
#include <array>
#include <iostream>
#include <vector>
#include <cstdint>
#include <string>
#include <algorithm>
#include <cstdio>
#include <sstream>
#include <string>
#include <cstring>

// This is purely to make Rust like syntax work in some places.
using u8 = std::uint8_t;
using i16 = std::int16_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using isize = std::size_t;
using u128 = __uint128_t;

const u32 MAXBITS = 60;
const isize STEPSIZEINITIAL = 24;
const u32 LASTBITS = 10;
const u32 LASTBITSTEPS = ((1 << LASTBITS) + 63)/64;
const u128 LARGENUMBERCUTOFF = (u128)1 << 100;
const isize BATCHSIZE = (1 << 16);

constexpr u32 THREEPOWERSU32[17] = {1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441, 1594323, 4782969, 14348907, 43046721};

u128 THREEPOWERS[72];

#define MAXBITSINITIAL (MAXBITS - LASTBITS)
#define ONETHIRD 2863311531
#define THREEPOWEROFFSET (MAXBITSINITIAL * 41 / 65 + 1)
#define INITIALBITARRAYSIZE ((1 << (STEPSIZEINITIAL - 6)) + ((1 << LASTBITS) + 63)/64)
#define ctzu32 __builtin_ctz
#define ctzu64 __builtin_ctzll

struct Pair {
    u128 base;
    u128 number;
};

struct PrecomputedData {
    std::vector<std::vector<u64>> initialbitarray;
    u32 magicnumbers[16];
    std::vector<u64> mod9sieve[9];
    std::vector<std::vector<Pair>> bigpairvector;
};

// Function declaration
void precompute(PrecomputedData* data);
u64 search1(u128 base, u32 sigfig, u32 threepower, u128 number, PrecomputedData* data, isize* d_flat, isize* d_flatmod9sieve);
u64 search2(u128 base, u32 threepower, u128 number, PrecomputedData* data);
u64 search3(u128 base, u128 number, PrecomputedData* data);
u64 launchkernel(PrecomputedData* data, u32 bigindex, isize* d_flat, isize* d_flatmod9sieve, u32 threepower);

int main() {
    std::string line;
    std::getline(std::cin, line);         // read entire line

    std::istringstream iss(line);
    u64 arg1, arg2, arg3, arg4;
    iss >> arg1 >> arg2 >> arg3 >> arg4;

    // This is a little bit stupid, but defining THREEPOWERS at compile time was somehow complicated.
    u128 a = 1; THREEPOWERS[0] = a;
    for (isize i = 1; i < 72; i++) { a *= 3; THREEPOWERS[i] = a; }

    // This is a little bit stupid, but an idiomatic way to initialise data seems more complicated.
    PrecomputedData data = {
        {std::vector<u64>(INITIALBITARRAYSIZE), std::vector<u64>(INITIALBITARRAYSIZE), std::vector<u64>(INITIALBITARRAYSIZE), std::vector<u64>(INITIALBITARRAYSIZE), std::vector<u64>(INITIALBITARRAYSIZE), std::vector<u64>(INITIALBITARRAYSIZE), std::vector<u64>(INITIALBITARRAYSIZE), std::vector<u64>(INITIALBITARRAYSIZE), std::vector<u64>(INITIALBITARRAYSIZE), std::vector<u64>(INITIALBITARRAYSIZE), std::vector<u64>(INITIALBITARRAYSIZE), std::vector<u64>(INITIALBITARRAYSIZE), std::vector<u64>(INITIALBITARRAYSIZE), std::vector<u64>(INITIALBITARRAYSIZE), std::vector<u64>(INITIALBITARRAYSIZE), std::vector<u64>(INITIALBITARRAYSIZE)},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {std::vector<u64>(LASTBITSTEPS), std::vector<u64>(LASTBITSTEPS), std::vector<u64>(LASTBITSTEPS), std::vector<u64>(LASTBITSTEPS), std::vector<u64>(LASTBITSTEPS), std::vector<u64>(LASTBITSTEPS), std::vector<u64>(LASTBITSTEPS), std::vector<u64>(LASTBITSTEPS), std::vector<u64>(LASTBITSTEPS) },
        {std::vector<Pair>(), std::vector<Pair>(), std::vector<Pair>(), std::vector<Pair>(), std::vector<Pair>(), std::vector<Pair>(), std::vector<Pair>(), std::vector<Pair>(), std::vector<Pair>(), std::vector<Pair>(), std::vector<Pair>(), std::vector<Pair>(), std::vector<Pair>(), std::vector<Pair>(), std::vector<Pair>(), std::vector<Pair>() },
    };

    u32 magicnumber = 1;
    for (isize i = 1; i < 64; i++) {
        magicnumber *= ONETHIRD;
        if (i >= THREEPOWEROFFSET && i - THREEPOWEROFFSET < 16) {
            data.magicnumbers[i - THREEPOWEROFFSET] = magicnumber;
        }
    }

    precompute(&data);

    // Put initialbitarray on the GPU:
    isize rows = data.initialbitarray.size();
    isize cols = data.initialbitarray[0].size();

    // Flatten
    std::vector<u64> flat(rows * cols);
    for (isize r = 0; r < rows; ++r) {
        std::memcpy(&flat[r * cols],
                    data.initialbitarray[r].data(),
                    cols * sizeof(u64));
    }

    // Allocate device memory
    isize* d_flat;
    cudaMalloc(&d_flat, flat.size() * sizeof(isize));

    // Copy ONCE
    cudaMemcpy(d_flat, flat.data(), flat.size() * sizeof(u64), cudaMemcpyHostToDevice);

    // Put mod9sieve on the GPU:
    rows = 9;
    cols = data.mod9sieve[0].size();

    // Flatten
    std::vector<u64> flatmod9sieve(rows * cols);
    for (isize r = 0; r < rows; ++r) {
        std::memcpy(&flatmod9sieve[r * cols],
                    data.mod9sieve[r].data(),
                    cols * sizeof(u64));
    }

    // Allocate device memory
    u64* d_flatmod9sieve;
    cudaMalloc(&d_flatmod9sieve, flatmod9sieve.size() * sizeof(u64));

    // Copy ONCE
    cudaMemcpy(d_flatmod9sieve, flatmod9sieve.data(), flatmod9sieve.size() * sizeof(u64), cudaMemcpyHostToDevice);

    // Do search1(3, 2, 2, 9) for initial one.
    u64 numcases = search1((u128)arg1, (u32)arg2, (u32)arg3, (u128)arg4, &data, d_flat, d_flatmod9sieve);

    // Finally we do the remaining cases.
    for (isize a = 0; a < 16; a++) {
        numcases += launchkernel(&data, a, d_flat, d_flatmod9sieve, THREEPOWEROFFSET + a);
    }

    fprintf(stderr, "numcases = %ld \n", numcases);
}

__global__ void compute_kernel(u64* flat, u64* flatmod9sieve, Pair *d_pairs, isize bigindex, isize n, u32 magic, u128 threepowervalue) {
    const u8 THREEPOWERSU8[6] = {1, 3, 9, 27, 81, 243};

    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    Pair p = d_pairs[idx];
    u128 base0 = p.base;
    u128 number0 = p.number;

    u32 redmod9 = u32(base0 % 9);
    u32 index = 0;
    u16 avalues[(1 << LASTBITS) * 5 / 9 + 2];
    u32 alength = 0;

    u32 n0 = (u32)number0;
    index = (magic * n0) & ((1 << STEPSIZEINITIAL) - 1);

    u16 offset = index % 64;

    for (u32 a = 0; a < LASTBITSTEPS; a++) {
        u64 temp = flatmod9sieve[(1 << (LASTBITS - 6)) * redmod9 + a];
        if (offset == 0) {
            temp &= flat[INITIALBITARRAYSIZE * bigindex + index/64 + a];
        } else {
            temp &= (flat[INITIALBITARRAYSIZE * bigindex + index/64 + a] >> offset)
                    | (flat[INITIALBITARRAYSIZE * bigindex + index/64 + a + 1] << (64 - offset));
        }
        while (temp != 0) {
            u16 b = 64 * a + __ffsll(temp) - 1;
            temp &= temp-1;
            avalues[alength] = b;
            alength += 1;
        }
    }

    u32 pos = 0;
    if (alength >= 1) {
        u16 aval = avalues[pos];
        u128 base = base0 | ((u128)aval << MAXBITSINITIAL);
        u128 number = number0 + aval * threepowervalue;
        while (true) {
            u32 a = __ffs((u32)number | 0b100000) - 1; //We don't have a ctz, so use ffs - 1 instead.
            number = THREEPOWERSU8[a] * (number >> a);

            if (number > LARGENUMBERCUTOFF) {
                printf("%llX%016llX\n", (u64)(base >> 64), (u64)base);
                number = 2;
            }

            number -= 1;
            u32 b = __ffs((u32)number | (1ul << 31)) - 1;
            number = (number >> b) + 1;

            if (number <= base) {
                pos += 1;
                if (pos == alength) { break; }
                aval = avalues[pos];
                base = base0 | ((u128)aval << MAXBITSINITIAL);
                number = number0 + aval * threepowervalue;
            }
        }
    }
}

void precompute(PrecomputedData* data) {
    i16 firstcutoff = (i16)(65 * THREEPOWEROFFSET - 41 * MAXBITSINITIAL);
    i16 cutoffs[16];
    for (isize i = 0; i < 16; i++) { cutoffs[i] = firstcutoff + i * 65; }

    u64 wraparoundcutoff = ((1 << LASTBITS) + 63)/64;
    u64 wraparoundadd    = 1 << (STEPSIZEINITIAL - 6);

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
        i16 curval = 0;
        i16 maxval = 0;
        for (u32 i = 0; i < STEPSIZEINITIAL; i++) {

            if ((upordown & (1 << i)) != 0) {
                curval -= 24;
            } else {
                curval += 41;
            }
            if (curval > maxval) { maxval = curval; }
        }
        for (isize a = 0; a < 16; a++) {
            if (maxval <= cutoffs[a]) {
                u64 index = (u64)(n0 * data->magicnumbers[a]) & ((1 << STEPSIZEINITIAL) - 1);

                data->initialbitarray[a][index/64] |= (u64)1 << (index % 64);

                if (index/64 < wraparoundcutoff) {
                    data->initialbitarray[a][index/64 + wraparoundadd] |= (u64)1 << (index % 64);
                }
            }
        }
    }

    u64 redmod9step = u64(((u128)1 << MAXBITSINITIAL) % 9);

    for (isize a = 0; a < 9; a++) {
        isize redmod9 = a;
        for (isize i = 0; i < ((isize)1 << LASTBITS); i++) {
            if (redmod9 == 0 || redmod9 == 1 || redmod9 == 3 || redmod9 == 6 || redmod9 == 7) {
                data->mod9sieve[a][i/64] |= (u64)1 << (i % 64);
            }
            redmod9 += redmod9step;
            if (redmod9 >= 9) { redmod9 -= 9; }
        }
    }
}

u64 search1(u128 base, u32 sigfig, u32 threepower, u128 number, PrecomputedData* data, u64* d_flat, u64* d_flatmod9sieve) {
    if (number <= base) { return 0; }
    if (sigfig == MAXBITSINITIAL) {

        u32 bigindex = threepower - THREEPOWEROFFSET;
        if (bigindex >= 16) {
            return search2(base, threepower, number, data);
        }
        Pair p = { base, number };
        data->bigpairvector[bigindex].push_back(p);

        if (data->bigpairvector[bigindex].size() == BATCHSIZE) {
            return launchkernel(data, bigindex, d_flat, d_flatmod9sieve, threepower);
        }
        return 0;
    }

    u128 numberA = number;
    u128 numberB = number + THREEPOWERS[threepower];

    if (number % 2 == 0) {
        u64 resL = search1(base,                       sigfig + 1, threepower + 1,
                           numberA + numberA/2, data, d_flat, d_flatmod9sieve);
        u64 resR = search1(base | ((u128)1 << sigfig), sigfig + 1, threepower,
                           numberB/2 + 1,       data, d_flat, d_flatmod9sieve);
        return resL + resR;
    } else {
        u64 resL = search1(base,                       sigfig + 1, threepower,
                           numberA/2 + 1,       data, d_flat, d_flatmod9sieve);
        u64 resR = search1(base | ((u128)1 << sigfig), sigfig + 1, threepower + 1,
                           numberB + numberB/2, data, d_flat, d_flatmod9sieve);
        return resL + resR;
    }
}

u64 launchkernel(PrecomputedData* data, u32 bigindex, u64* d_flat, u64* d_flatmod9sieve, u32 threepower) {
    u32 magic = 0;
    if (bigindex <= 7) { magic = data->magicnumbers[bigindex]; }
    u128 threepowervalue = THREEPOWERS[threepower];

    Pair* d_input = nullptr;
    isize n = data->bigpairvector[bigindex].size();

    if (n == 0) { return 0; }

    cudaMalloc(&d_input, n * sizeof(Pair));

    // Copy vector contents directly to device
    cudaMemcpy(d_input, data->bigpairvector[bigindex].data(), n * sizeof(Pair), cudaMemcpyHostToDevice);

    isize threads_needed = n;

    u32 threads_per_block = 256;
    u32 blocks = (threads_needed + threads_per_block - 1) / threads_per_block;

    compute_kernel<<<blocks, threads_per_block>>>(d_flat, d_flatmod9sieve, d_input, bigindex, n, magic, threepowervalue);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "Kernel launch error: " << cudaGetErrorString(err) << "\n";

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "Kernel runtime error: " << cudaGetErrorString(err) << "\n";

    cudaFree(d_input);

    data->bigpairvector[bigindex].clear();
    return (u64)n;
}

u64 search2(u128 base, u32 threepower, u128 number, PrecomputedData* data) {
    isize redmod9 = u64(base % 9);
    isize index = 0;

    if (threepower < THREEPOWEROFFSET + 8) {
        u32 n0 = (u32)number;
        index = (data->magicnumbers[threepower - THREEPOWEROFFSET] * n0) & ((1 << STEPSIZEINITIAL) - 1);
    }

    u64 offset = index % 64;

    for (isize a = 0; a < LASTBITSTEPS; a++) {
        u64 temp = data->mod9sieve[redmod9][a];
        if (threepower < THREEPOWEROFFSET + 8) {
            if (offset == 0) {
                temp &= data->initialbitarray[threepower - THREEPOWEROFFSET][index/64 + a];
            }
            else {
                temp &= (data->initialbitarray[threepower - THREEPOWEROFFSET][index/64 + a] >> offset)
                    | (data->initialbitarray[threepower - THREEPOWEROFFSET][index/64 + a + 1] << (64 - offset));
            }
        }
        while (temp != 0) {
            u128 highbits = 64 * a + ctzu64(temp);
            temp &= temp-1;
            Pair p = { base + (highbits << MAXBITSINITIAL), number + (highbits * THREEPOWERS[threepower]), };
            search3(p.base, p.number, data);
        }
    }

    return 1;
}

u64 search3(u128 base, u128 number, PrecomputedData* data) {
    u64 tally = 1;
    while (true) {
        u32 a = ctzu32((u32)number | 0b100000);
        number = THREEPOWERSU32[a] * (number >> a);

        if (number > LARGENUMBERCUTOFF) {
            printf("%lX%016lX\n", (u64)(base >> 64), (u64)base);
            number = 2;
        }

        number -= 1;
        u32 b = ctzu32((u32)number | (1ul << 31));
        number = (number >> b) + 1;

        if (number < base) { return tally; }
    }
}
