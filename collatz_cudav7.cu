/*
 Compile with something like
 "nvcc -std=c++17 -O3 -arch=sm_86 -maxrregcount=34 -o collatz_cudav7 collatz_cudav7.cu".
 Adjust -arch according to your GPU.
 Adjust MAXBITS and CUTOFFPOWER as appropriate. As written, we need CUTOFFPOWER >= 96.
 Run with "echo "3 2 2 9 1 0" | ./collatz_cudav7" for a complete calculation, or
 use one of the files cases_tiny, cases_small or cases_big. Simply run
 "head -n $i cases_small | tail -n 1 | ./collatz_cudav7" to run the i'th case.
 It will output all starting numbers which go above 2^CUTOFFPOWER.
 Pipe to  "{ echo "ibase=16"; cat; } | bc" to get decimal output.

 Note that changing any of the constants will change the hash.
*/

#include <cuda_runtime.h>
#include <cstdint>
#include <array>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>
#include <sstream>
#include <cstring>
#include <unistd.h>
#include <chrono>
#include <stdlib.h>
#include <fstream>
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
const u32 CUTOFFPOWER = 100;
const u128 LARGENUMBERCUTOFF = (u128)1 << CUTOFFPOWER;
const u32 U32LARGENUMBERCUTOFF = (u32)1 << (CUTOFFPOWER - 96);
const u32 STEPSIZEINITIAL = 24;
const u32 LASTBITS = 10;
const u32 BATCHSIZE = (1 << 16);
const u32 NUMERATOR = 306; // NUMERATOR / DENOMINATOR is a rational
                           // approximation to log(2)/log(3)
const u32 DENOMINATOR = 485;
const u32 LASTBITSTEPS = ((1 << LASTBITS) + 31)/32;
const u32 HASHTABLESIZE = 256;
const u32 SPOTCHECKFREQUENCY = 10000; // How often we run a check on the CPU

constexpr u32 THREEPOWERSU32[17] = {1, 3, 9, 27, 81, 243, 729, 2187, 6561,
                                    19683, 59049, 177147, 531441, 1594323,
                                    4782969, 14348907, 43046721};

__constant__ uint16_t THREEPOWERSU16[9] = {1, 3, 9, 27, 81, 243, 729, 2187, 6561};

u128 THREEPOWERS[80]; // Initialised in main()

#define MAXBITSINITIAL (MAXBITS - LASTBITS)
#define ONETHIRD 2863311531 // Inverse to 3 in the units of Z/(2^32)
#define THREEPOWEROFFSET (MAXBITSINITIAL * NUMERATOR / DENOMINATOR + 1)
#define INITIALBITARRAYSIZE ((1 << (STEPSIZEINITIAL - 5)) + ((1 << LASTBITS) + 31)/32)
#define ctzu32 __builtin_ctz

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
u64 search1(u128 base, u32 sigfig, u32 threepower, u128 number,
            u32 doubleeven, u32 numevens, PrecomputedData* data,
            u32* d_flat, u32* d_flatmod9sieve);
u32 search2(u128 base, u32 threepower, u128 number, PrecomputedData* data, u32 bigindex);
u32 search3(u128 base, u128 number, PrecomputedData* data);
u64 launchkernel(PrecomputedData* data, u32 bigindex, u32* d_flat,
                 u32* d_flatmod9sieve, u32 threepower);

int main() {
    std::string line;
    std::getline(std::cin, line);

    std::istringstream iss(line);
    u64 arg1, arg2, arg3, arg4, arg5, arg6;
    iss >> arg1 >> arg2 >> arg3 >> arg4 >> arg5 >> arg6;

    // We initialise THREEPOWERS. There is probably a way to do it at compile time.
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
    // fprintf(stderr, "Finished precompute\n");

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

    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    // Allocate device memory
    u32* d_flat;
    cudaMalloc(&d_flat, flat.size() * sizeof(u32));

    // Copy ONCE
    cudaMemcpy(d_flat, flat.data(), flat.size() * sizeof(u32), cudaMemcpyHostToDevice);

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

    // Allocate device memory
    u32* d_flatmod9sieve;
    cudaMalloc(&d_flatmod9sieve, flatmod9sieve.size() * sizeof(u32));

    // Copy ONCE
    cudaMemcpy(d_flatmod9sieve, flatmod9sieve.data(), flatmod9sieve.size() * sizeof(u32), cudaMemcpyHostToDevice);

    // Do search1(3, 2, 2, 9, 1, 0) for initial one.
    u64 numcases = search1((u128)arg1, (u32)arg2, (u32)arg3, (u128)arg4, (u32)arg5, (u32)arg6, &data, d_flat, d_flatmod9sieve);

    // Finally we do the remaining cases.
    for (isize a = 0; a < 16; a++) {
        numcases += launchkernel(&data, a, d_flat, d_flatmod9sieve, THREEPOWEROFFSET + a);
    }

    fprintf(stderr, "Number of kernel launches = %ld. ", numcases);
    fprintf(stderr, "Spot checks = %u. Complete hash = %X.\n", data.spot_check_count, data.complete_hash);
}

__global__ void compute_kernel(u32* flat,
                               u32* flatmod9sieve,
                               u32* d_pairs,
                               u32 bigindex,
                               u32 threads_launched,
                               u32 magic,
                               u32 tp_0, u32 tp_1, u32 tp_2, u32 tp_3,
                               u32* hash) {

    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= threads_launched) return;

    u32 base0_0 = d_pairs[8 * idx];
    u32 base0_1 = d_pairs[8 * idx + 1];
    u32 base0_2 = d_pairs[8 * idx + 2];
    // u32 base0_3 = d_pairs[8 * idx + 3]; // We assume base fits in 96 bits.
    u32 number0_0 = d_pairs[8 * idx + 4];
    u32 number0_1 = d_pairs[8 * idx + 5];
    u32 number0_2 = d_pairs[8 * idx + 6];
    u32 number0_3 = d_pairs[8 * idx + 7];

    u32 rolling_hash = 0;

    u32 redmod9 = (base0_0 & 0x3FFFF) + (base0_0 >> 18)
              + 4 * (base0_1 & 0x3FFFF) + 4 * (base0_1 >> 18)
              + 7 * (base0_2 & 0x3FFFF) + 7 * (base0_2 >> 18);
              // We assume base0 fits in 96 bits. Otherwise add
              // (base0_3 & 0x3FFFF) + (base0_3 >> 18) as well
    redmod9 = redmod9 % 9;

    u16 avalues[(1 << LASTBITS) * 5 / 9 + 2]; //Worst case scenario we need this many a's.
    u32 alength = 0;

    u32 index = (magic * number0_0) & ((1 << STEPSIZEINITIAL) - 1);
    u32 offset = index % 32;

    for (u32 a = 0; a < LASTBITSTEPS; a++) {
        // Mod 9 Preimage Sieve:
        u32 temp = flatmod9sieve[(1 << (LASTBITS - 5)) * redmod9 + a]
                 & __funnelshift_r(flat[INITIALBITARRAYSIZE * bigindex + index/32 + a],
                                   flat[INITIALBITARRAYSIZE * bigindex + index/32 + a + 1], offset);

        u32 revtemp = __brev(temp);
        while (revtemp != 0) {
            u32 lz = __clz(revtemp);
            revtemp &= ~(0x80000000u >> lz);
            avalues[alength] = (u16)(32 * a + lz);
            alength += 1;
        }
    }

    if (alength == 0) { return; }
    u32 pos = 0;
    u16 aval = avalues[pos];

    u32 base_0 = base0_0;
    u32 base_1 = base0_1;
    u32 base_2 = base0_2;
    // We're assuming base, represented by base_0, base_1, base_2 fits in 96 bits.

    // NVCC warns about shift count >= 32, but MAXBITSINITIAL is guaranteed 1..31
    if constexpr (MAXBITSINITIAL > 0 && MAXBITSINITIAL < 32) {
        base_0 |= (u32)aval << (MAXBITSINITIAL & 31);
        base_1 |= (u32)aval >> ((32 - MAXBITSINITIAL) & 31);
    }
    if constexpr (MAXBITSINITIAL == 32) {
        base_1 = (u32) aval;
    }
    if constexpr (MAXBITSINITIAL > 32 && MAXBITSINITIAL < 64) {
        base_1 |= ((u32)aval << (MAXBITSINITIAL - 32));
        base_2 = (u32)aval >> (64 - MAXBITSINITIAL);
    }
    if constexpr (MAXBITSINITIAL >= 64 && MAXBITSINITIAL < 96) {
        base_2 |= (u32)aval << ((MAXBITSINITIAL - 64) & 31);
    }

    // We compute the limbs of num = number0 + aval * threepowervalue
    u32 carry;

    u64 p0 = (u64)aval * tp_0 + number0_0;
    u64 p1 = (u64)aval * tp_1 + number0_1;
    u64 p2 = (u64)aval * tp_2 + number0_2;
    u32 p3 = (u32)aval * tp_3 + number0_3;

    u32 num_0 = (u32)p0; carry = p0 >> 32;

    u64 t = p1 + carry;
    u32 num_1 = (u32)t; carry = t >> 32;

    t = p2 + carry;
    u32 num_2 = (u32)t; carry = t >> 32;

    u32 num_3 = p3 + carry;

    while (true) {
        // First odd iterations:
        u32 a = __ffs(num_0 | (1u << 8)) - 1; //We don't have a ctz, so use ffs - 1 instead.
        u32 m = THREEPOWERSU16[a];
        u32 t0 = __funnelshift_r(num_0, num_1, a);
        u64 t = (u64)t0 * m;
        num_0 = (u32)t; carry = t >> 32;
        t0 = __funnelshift_r(num_1, num_2, a);
        t = (u64)t0 * m + carry;
        num_1 = (u32)t; carry = t >> 32;
        t0 = __funnelshift_r(num_2, num_3, a);
        t = (u64)t0 * m + carry;
        num_2 = (u32)t; carry = t >> 32;
        num_3 = (num_3 >> a) * m + carry;

        bool overflow = (num_3 >= U32LARGENUMBERCUTOFF);
        if (__any_sync(0xFFFFFFFF, overflow)) {
            if (overflow) {
                printf("%X%08X%08X\n", base_2, base_1, base_0);
                num_0 = 2; num_1 = 0; num_2 = 0; num_3 = 0;
            }
        }

        // First even iterations:
        num_0 -= 1;
        u32 b = __ffs(num_0 | (1u << 31)) - 1;
        num_0 = __funnelshift_r(num_0, num_1, b) + 1;
        num_1 = __funnelshift_r(num_1, num_2, b);
        num_2 = __funnelshift_r(num_2, num_3, b);
        num_3 = num_3 >> b;
        if (num_0 == 0 && b > 0) {
            num_1 += 1;
            if (num_1 == 0) {
                num_2 += 1;
                if (num_2 == 0) {
                    num_3 += 1;
                }
            }
        }

        // Second odd iteration:
        a = __ffs(num_0 | (1u << 8)) - 1; //We don't have a ctz, so use ffs - 1 instead.
        m = THREEPOWERSU16[a];
        t0 = __funnelshift_r(num_0, num_1, a);
        t = (u64)t0 * m;
        num_0 = (u32)t; carry = t >> 32;
        t0 = __funnelshift_r(num_1, num_2, a);
        t = (u64)t0 * m + carry;
        num_1 = (u32)t; carry = t >> 32;
        t0 = __funnelshift_r(num_2, num_3, a);
        t = (u64)t0 * m + carry;
        num_2 = (u32)t; carry = t >> 32;
        num_3 = (num_3 >> a) * m + carry;

        overflow = (num_3 >= U32LARGENUMBERCUTOFF);
        if (__any_sync(0xFFFFFFFF, overflow)) {
            if (overflow) {
                printf("%X%08X%08X\n", base_2, base_1, base_0);
                num_0 = 2; num_1 = 0; num_2 = 0; num_3 = 0;
            }
        }

        // Second even iteration:
        num_0 -= 1;
        b = __ffs(num_0 | (1u << 31)) - 1;
        num_0 = __funnelshift_r(num_0, num_1, b) + 1;
        num_1 = __funnelshift_r(num_1, num_2, b);
        num_2 = __funnelshift_r(num_2, num_3, b);
        num_3 = num_3 >> b;
        if (num_0 == 0 && b > 0) {
            num_1 += 1;
            if (num_1 == 0) {
                num_2 += 1;
                if (num_2 == 0) {
                    num_3 += 1;
                }
            }
        }

        rolling_hash ^= num_0;

        // Check if num < base (Descent Sieve):
        bool number_lt_base =
        (num_3 == 0 && num_2 < base_2)
        | (num_3 == 0 && num_2 == base_2 && num_1 < base_1)
        | (num_3 == 0 && num_2 == base_2 && num_1 == base_1 && num_0 < base_0);
        if (number_lt_base) {
            pos += 1;
            if (pos == alength) { goto exit_sequence; }
            aval = avalues[pos];

            // NVCC warns about shift count >= 32, but MAXBITSINITIAL is guaranteed 1..31
            // Some code repetition here. There's probably a way to avoid it.
            if constexpr (MAXBITSINITIAL > 0 && MAXBITSINITIAL < 32) {
                base_0 = base0_0 | (u32)aval << (MAXBITSINITIAL & 31);
                base_1 = (u32)aval >> ((32 - MAXBITSINITIAL) & 31);
            }
            if constexpr (MAXBITSINITIAL == 32) {
                base_1 = (u32) aval;
            }
            if constexpr (MAXBITSINITIAL > 32 && MAXBITSINITIAL < 64) {
                base_1 = base0_1 | ((u32)aval << (MAXBITSINITIAL - 32));
                base_2 = (u32)aval >> (64 - MAXBITSINITIAL);
            }
            if constexpr (MAXBITSINITIAL >= 64 && MAXBITSINITIAL < 96) {
                base_2 = base0_2 | (u32)aval << ((MAXBITSINITIAL - 64) & 31);
            }

            u32 carry;

            p0 = (u64)aval * tp_0 + number0_0;
            p1 = (u64)aval * tp_1 + number0_1;
            p2 = (u64)aval * tp_2 + number0_2;
            p3 = (u32)aval * tp_3 + number0_3;

            num_0 = (u32)p0; carry = p0 >> 32;

            u64 t = p1 + carry;
            num_1 = (u32)t; carry = t >> 32;

            t = p2 + carry;
            num_2 = (u32)t; carry = t >> 32;

            num_3 = p3 + carry;
        }
    }

    exit_sequence:
    u32 slot = idx % HASHTABLESIZE;
    atomicXor(&hash[slot], rolling_hash);

}

// Check if there is a string of 1's followed by two 0's, starting from position i of upordown.
// This is for the Odd-Even-Even sieve.
bool doubleeven(u32 upordown, u64 i) {
    u32 a = ctzu32((upordown >> i) + 1);
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
                if (curval + (i32)NUMERATOR > maxval) {
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
                    if (curval + ((i32)DENOMINATOR - (i32)NUMERATOR) > maxval) {
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

                data->initialbitarray[a][index/32] |= (u32)1 << (index % 32);

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
                data->mod9sieve[a][i/32] |= (u32)1 << (i % 32);
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
            u32* d_flat, // Reference to the BV_i bitvectors on the GPU
            u32* d_flatmod9sieve) { // Reference to the Mod 9 Preimage Sieve on the GPU
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
        // Pair p = { base, number };
        // data->bigpairvector[bigindex].push_back(p);

        if (data->bigpairvector2[bigindex].size() == 8 * BATCHSIZE) {
            return launchkernel(data, bigindex, d_flat, d_flatmod9sieve, threepower);
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
                           numberA + numberA/2, (doubleeven | doubleinplay) & (~2), 0, data, d_flat, d_flatmod9sieve);
        u64 resR = 0;
        if (doubleeven < 2) { resR = search1(base | ((u128)1 << sigfig), sigfig + 1, threepower,
                           numberB/2 + 1,  doubleeven << 1, numevens + 1,      data, d_flat, d_flatmod9sieve); }
        return resL + resR;
    } else {
        u64 resL = 0;
        if (doubleeven < 2) { resL = search1(base,                       sigfig + 1, threepower,
                           numberA/2 + 1, doubleeven << 1, numevens + 1,       data, d_flat, d_flatmod9sieve); }
        u64 resR = search1(base | ((u128)1 << sigfig), sigfig + 1, threepower + 1,
                           numberB + numberB/2 ,(doubleeven | doubleinplay) & (~2), 0, data, d_flat, d_flatmod9sieve);
        return resL + resR;
    }
}

u64 launchkernel(PrecomputedData* data,
                 u32 bigindex,
                 u32* d_flat,
                 u32* d_flatmod9sieve,
                 u32 threepower) {
    u32 magic = data->magicnumbers[bigindex];
    u128 threepowervalue = THREEPOWERS[threepower];

    u32* d_input = nullptr;
    u32 n = data->bigpairvector2[bigindex].size();

    if (n == 0) { return 0; }

    cudaMalloc(&d_input, n * sizeof(u32));

    // Copy vector contents directly to device
    cudaMemcpy(d_input, data->bigpairvector2[bigindex].data(), n * sizeof(u32), cudaMemcpyHostToDevice);

    u32* d_hash = nullptr;
    u32 h_hash[HASHTABLESIZE] = {};
    cudaMalloc(&d_hash, HASHTABLESIZE * sizeof(u32));
    cudaMemcpy(d_hash, h_hash, HASHTABLESIZE * sizeof(u32), cudaMemcpyHostToDevice);

    u32 threads_needed = n / 8;

    u32 threads_per_block = 256;
    u32 blocks = (threads_needed + threads_per_block - 1) / threads_per_block;

    u32 tp_0 = (u32) threepowervalue;
    u32 tp_1 = (u32) (threepowervalue >> 32);
    u32 tp_2 = (u32) (threepowervalue >> 64);
    u32 tp_3 = (u32) (threepowervalue >> 96);

    compute_kernel<<<blocks, threads_per_block>>>(d_flat,
                                                  d_flatmod9sieve,
                                                  d_input,
                                                  bigindex,
                                                  threads_needed,
                                                  magic,
                                                  tp_0, tp_1, tp_2, tp_3,
                                                  d_hash);

    // Do some error checking, just in case:
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "Kernel launch error: " << cudaGetErrorString(err) << "\n";

    u32 hash = 0;
    cudaMemcpy(h_hash, d_hash, HASHTABLESIZE * sizeof(u32), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 256; i++) {
        hash ^= h_hash[i];
    }
    data->complete_hash ^= hash;

    if (dis(gen) == 1) {
        u32 hash2 = 0;
        for (uint i = 0; i < threads_needed; i++) {
            u128 base =  (u128)(data->bigpairvector2[bigindex][8 * i])
                    | ((u128)(data->bigpairvector2[bigindex][8 * i + 1]) << 32)
                    | ((u128)(data->bigpairvector2[bigindex][8 * i + 2]) << 64);
            u128 number =  (u128)(data->bigpairvector2[bigindex][8 * i + 4])
                        | ((u128)(data->bigpairvector2[bigindex][8 * i + 5]) << 32)
                        | ((u128)(data->bigpairvector2[bigindex][8 * i + 6]) << 64)
                        | ((u128)(data->bigpairvector2[bigindex][8 * i + 7]) << 96);
            hash2 ^= search2(base, threepower, number, data, bigindex);
        }
        if (hash != hash2) {
            fprintf(stderr, "Spot check failed\n");
        }
        data->spot_check_count += 1;
    }

    cudaDeviceSynchronize();

    // More error checking, just in case:
    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "Kernel runtime error: " << cudaGetErrorString(err) << "\n";

    cudaFree(d_input);
    cudaFree(d_hash);

    data->bigpairvector2[bigindex].clear();
    return 1;
}

// search2 and search3 do the same as compute_kernel in the case where
// threepower is too big for our 16 precomputed bitvectors. This case
// is rare. We also use these for spot checking the hash.
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

