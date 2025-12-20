/*
 Compile with something like
 "nvcc -std=c++17 -O3 -arch=sm_86 -o collatz_cudav5 collatz_cudav5.cu".
 Adjust -arch according to your GPU.
 Ignore compiler warnings about "shift count is too large".
 Adjust MAXBITS and CUTOFFPOWER as appropriate. As written, we need CUTOFFPOWER >= 97.
 Run with "echo "3 2 2 9 0 0" | ./collatz_cudav5" for a complete calculation, or
 use one of the files cases_small or cases_big. Simply run
 "head -n $i cases_small | tail -n 1 | ./collatz_cudav5" to run the i'th case.
 It will output all starting numbers which go above 2^CUTOFFPOWER. (But it will
 only look at the most significant bits, so it might miss some that go over by a
 small margin.)
 Pipe to  "{ echo "ibase=16"; cat; } | bc" to get decimal output.
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
const u32 MAXBITS = 60;
const u32 CUTOFFPOWER = 100;
const u128 LARGENUMBERCUTOFF = (u128)1 << CUTOFFPOWER;
const u32 U32LARGENUMBERCUTOFF = (u32)1 << (CUTOFFPOWER - 96);
const isize STEPSIZEINITIAL = 24;
const u32 LASTBITS = 10;
const isize BATCHSIZE = (1 << 16);
const u32 NUMERATOR = 306; // NUMERATOR / DENOMINATOR is a rational
                           // approximation to log(2)/log(3)
const u32 DENOMINATOR = 485;
const u32 LASTBITSTEPS = ((1 << LASTBITS) + 63)/64;

constexpr u32 THREEPOWERSU32[17] = {1, 3, 9, 27, 81, 243, 729, 2187, 6561,
                                    19683, 59049, 177147, 531441, 1594323,
                                    4782969, 14348907, 43046721};

u128 THREEPOWERS[80]; // Initialised in main()

#define MAXBITSINITIAL (MAXBITS - LASTBITS)
#define ONETHIRD 2863311531 // Inverse to 3 in the units of Z/(2^32)
#define THREEPOWEROFFSET (MAXBITSINITIAL * NUMERATOR / DENOMINATOR + 1)
#define INITIALBITARRAYSIZE ((1 << (STEPSIZEINITIAL - 6)) + ((1 << LASTBITS) + 63)/64)
#define ctzu32 __builtin_ctz
#define ctzu64 __builtin_ctzll

struct Pair {
    u128 base;
    u128 number;
};

struct PrecomputedData {
    std::array<std::vector<u64>, 16> initialbitarray;
    std::array<u32, 16> magicnumbers; // Inverses of certain powers of 3
    std::array<std::vector<u64>, 9> mod9sieve;
    std::array<std::vector<Pair>, 16> bigpairvector;
};

// Function declarations
void precompute(PrecomputedData* data);
u64 search1(u128 base, u32 sigfig, u32 threepower, u128 number,
            u32 doubleeven, u32 numevens, PrecomputedData* data,
            isize* d_flat, isize* d_flatmod9sieve);
u64 search2(u128 base, u32 threepower, u128 number, PrecomputedData* data);
u64 search3(u128 base, u128 number, PrecomputedData* data);
u64 launchkernel(PrecomputedData* data, u32 bigindex, isize* d_flat,
                 isize* d_flatmod9sieve, u32 threepower);

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
        std::array<std::vector<u64>, 16>{},
        std::array<u32, 16>{},
        std::array<std::vector<u64>, 9>{},
        std::array<std::vector<Pair>, 16>{},
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

    // Do search1(3, 2, 2, 9, 0, 0) for initial one.
    u64 numcases = search1((u128)arg1, (u32)arg2, (u32)arg3, (u128)arg4, (u32)arg5, (u32)arg6, &data, d_flat, d_flatmod9sieve);

    // Finally we do the remaining cases.
    for (isize a = 0; a < 16; a++) {
        numcases += launchkernel(&data, a, d_flat, d_flatmod9sieve, THREEPOWEROFFSET + a);
    }

    fprintf(stderr, "numcases = %ld \n", numcases);
}

__global__ void compute_kernel(u64* flat, u64* flatmod9sieve, Pair *d_pairs, isize bigindex, isize n, u32 magic, u128 threepowervalue) {
    const u16 THREEPOWERSU16[9] = {1, 3, 9, 27, 81, 243, 729, 2187, 6561};

    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    Pair p = d_pairs[idx];
    u128 base0 = p.base;
    u128 number0 = p.number;

    u32 redmod9 = u32(base0 % 9);
    u32 index = 0;
    u16 avalues[(1 << LASTBITS) * 5 / 9 + 2]; //Worst case scenario we need this many a's.
    u32 alength = 0;

    u32 n0 = (u32)number0;
    index = (magic * n0) & ((1 << STEPSIZEINITIAL) - 1);

    u16 offset = index % 64;

    for (u32 a = 0; a < LASTBITSTEPS; a++) {
        // Mod 9 Preimage Sieve:
        u64 temp = flatmod9sieve[(1 << (LASTBITS - 6)) * redmod9 + a];
        if (offset == 0) {
            // The other 3 sieves combined:
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

    if (alength == 0) { return; }

    // We break up the various 128-bit integers into 32 bit limbs.
    u32 base0_0 = (u32) base0;
    u32 base0_1 = (u32) (base0 >> 32);
    u32 base0_2 = (u32) (base0 >> 64);
    // We're assuming base0 fits in 96 bits

    u32 number0_0 = (u32) number0;
    u32 number0_1 = (u32) (number0 >> 32);
    u32 number0_2 = (u32) (number0 >> 64);
    u32 number0_3 = (u32) (number0 >> 96);

    u32 tp_0 = (u32) threepowervalue;
    u32 tp_1 = (u32) (threepowervalue >> 32);
    u32 tp_2 = (u32) (threepowervalue >> 64);
    u32 tp_3 = (u32) (threepowervalue >> 96);

    u32 pos = 0;
    u16 aval = avalues[pos];

    u32 base_0 = base0_0;
    u32 base_1 = base0_1;
    u32 base_2 = base0_2;
    // We're assuming base, represented by base_0, base_1, base_2 fits in 96 bits.

    // NVCC warns about shift count >= 32, but MAXBITSINITIAL is guaranteed 1..31
    if constexpr (MAXBITSINITIAL > 0 && MAXBITSINITIAL < 32) {
        base_0 |= (u32)aval << MAXBITSINITIAL;
        base_1 |= (u32)aval >> (32 - MAXBITSINITIAL);
    }
    if constexpr (MAXBITSINITIAL == 32) {
        base_1 = (u32) aval;
    }
    if constexpr (MAXBITSINITIAL > 32 && MAXBITSINITIAL < 64) {
        base_1 |= ((u32)aval << (MAXBITSINITIAL - 32));
        base_2 = (u32)aval >> (64 - MAXBITSINITIAL);
    }
    if constexpr (MAXBITSINITIAL >= 64 && MAXBITSINITIAL < 96) {
        base_2 |= (u32)aval << (MAXBITSINITIAL - 64);
    }

    // We compute the limbs of num = number0 + aval * threepowervalue
    u32 carry;

    u64 p0 = (u64)aval * tp_0;
    u64 p1 = (u64)aval * tp_1;
    u64 p2 = (u64)aval * tp_2;
    u64 p3 = (u64)aval * tp_3;

    u64 t = p0 + number0_0;
    u32 num_0 = (u32)t; carry = t >> 32;

    t = p1 + number0_1 + carry;
    u32 num_1 = (u32)t; carry = t >> 32;

    t = p2 + number0_2 + carry;
    u32 num_2 = (u32)t; carry = t >> 32;

    t = p3 + number0_3 + carry;
    u32 num_3 = (u32)t;

    while (true) {
        // Do odd iterations, if there are any. (There usually are.)
        if ((num_0 & 1) == 0) {
            u32 a = __ffs(num_0 | (1ul << 8)) - 1; //We don't have a ctz, so use ffs - 1 instead.

            // u32 t0 = __funnelshift_r(num_0, num_1, a);
            u32 t0 = (num_0 >> a) | (num_1 << (32 - a));
            u32 t1 = (num_1 >> a) | (num_2 << (32 - a));
            u32 t2 = (num_2 >> a) | (num_3 << (32 - a));
            u32 t3 = num_3 >> a;

            u32 m = THREEPOWERSU16[a];
            u64 t = (u64)t0 * m;
            num_0 = (u32)t; carry = t >> 32;

            t = (u64)t1 * m + carry;
            num_1 = (u32)t; carry = t >> 32;

            t = (u64)t2 * m + carry;
            num_2 = (u32)t; carry = t >> 32;

            num_3 = t3 * m + carry;

            if (num_3 >= U32LARGENUMBERCUTOFF) {
                printf("%X%08X%08X\n", base_2, base_1, base_0);
                num_0 = 2; num_1 = 0; num_2 = 0; num_3 = 0;
            }
        }

        // Do even iterations, if there are any. (There usually are.)
        if ((num_0 & 1) == 1) {
            num_0 -= 1;

            u32 b = __ffs(num_0 | (1ul << 31)) - 1;
            u32 t0 = (num_0 >> b) | (num_1 << (32 - b));
            u32 t1 = (num_1 >> b) | (num_2 << (32 - b));
            u32 t2 = (num_2 >> b) | (num_3 << (32 - b));
            u32 t3 = num_3 >> b;
            num_0 = t0 + 1; num_1 = t1; num_2 = t2; num_3 = t3;
            if (num_0 == 0) {
                num_1 += 1;
                if (num_1 == 0) {
                    num_2 += 1;
                    if (num_2 == 0) {
                        num_3 += 1;
                    }
                }
            }
        }

        // Check if num < base (Descent Sieve):
        bool number_lt_base =
          (num_3 == 0 && num_2 < base_2)
        | (num_3 == 0 && num_2 == base_2 && num_1 < base_1)
        | (num_3 == 0 && num_2 == base_2 && num_1 == base_1 && num_0 < base_0);
        if (number_lt_base) {
            pos += 1;
            if (pos == alength) { return; }
            aval = avalues[pos];

            // NVCC warns about shift count >= 32, but MAXBITSINITIAL is guaranteed 1..31
            // Some code repetition here. There's probably a way to avoid it.
            if constexpr (MAXBITSINITIAL > 0 && MAXBITSINITIAL < 32) {
                base_0 = base0_0 | (u32)aval << MAXBITSINITIAL;
                base_1 = (u32)aval >> (32 - MAXBITSINITIAL);
            }
            if constexpr (MAXBITSINITIAL == 32) {
                base_1 = (u32) aval;
            }
            if constexpr (MAXBITSINITIAL > 32 && MAXBITSINITIAL < 64) {
                base_1 = base0_1 | ((u32)aval << (MAXBITSINITIAL - 32));
                base_2 = (u32)aval >> (64 - MAXBITSINITIAL);
            }
            if constexpr (MAXBITSINITIAL >= 64 && MAXBITSINITIAL < 96) {
                base_2 = base0_2 | (u32)aval << (MAXBITSINITIAL - 64);
            }

            u32 carry;

            u64 p0 = (u64)aval * tp_0;
            u64 p1 = (u64)aval * tp_1;
            u64 p2 = (u64)aval * tp_2;
            u64 p3 = (u64)aval * tp_3;

            u64 t = p0 + number0_0;
            num_0 = (u32)t; carry = t >> 32;

            t = p1 + number0_1 + carry;
            num_1 = (u32)t; carry = t >> 32;

            t = p2 + number0_2 + carry;
            num_2 = (u32)t; carry = t >> 32;

            t = p3 + number0_3 + carry;
            num_3 = (u32)t;
        }
    }
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

                data->initialbitarray[a][index/64] |= (u64)1 << (index % 64);

                if (index/64 < wraparoundcutoff) {
                    data->initialbitarray[a][index/64 + wraparoundadd] |= (u64)1 << (index % 64);
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
                data->mod9sieve[a][i/64] |= (u64)1 << (i % 64);
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
            u64* d_flat, // Reference to the BV_i bitvectors on the GPU
            u64* d_flatmod9sieve) { // Reference to the Mod 9 Preimage Sieve on the GPU
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
            return search2(base, threepower, number, data);
        }
        Pair p = { base, number };
        data->bigpairvector[bigindex].push_back(p);

        if (data->bigpairvector[bigindex].size() == BATCHSIZE) {
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
                 u64* d_flat,
                 u64* d_flatmod9sieve,
                 u32 threepower) {
    u32 magic = data->magicnumbers[bigindex];
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

    // fprintf(stderr, "Launching kernel\n");
    compute_kernel<<<blocks, threads_per_block>>>(d_flat,
                                                  d_flatmod9sieve,
                                                  d_input,
                                                  bigindex,
                                                  n,
                                                  magic,
                                                  threepowervalue);

    // Do some error checking, just in case:
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "Kernel launch error: " << cudaGetErrorString(err) << "\n";

    cudaDeviceSynchronize();

    // More error checking, just in case:
    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "Kernel runtime error: " << cudaGetErrorString(err) << "\n";

    cudaFree(d_input);

    data->bigpairvector[bigindex].clear();
    return (u64)n;
}

// search2 and search3 do the same as compute_kernel in the case where
// threepower is too big for our 16 precomputed bitvectors. This case
// is rare.
u64 search2(u128 base, u32 threepower, u128 number, PrecomputedData* data) {
    isize redmod9 = u64(base % 9);

    for (isize a = 0; a < LASTBITSTEPS; a++) {
        u64 temp = data->mod9sieve[redmod9][a];
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

