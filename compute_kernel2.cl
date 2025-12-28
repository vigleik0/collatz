// #include <stddef.h>

const uint STEPSIZEINITIAL = 24;
const uint LASTBITSTEPS = 32;
const uint INITIALBITARRAYSIZE = (1 << (STEPSIZEINITIAL - 5)) + LASTBITSTEPS;
const uint HASHTABLESIZE = 256;

// OpenCL Portable Funnel Shift Right
// Emulates __funnelshift_r(lo, hi, shift)
inline uint funnelshift_r(uint lo, uint hi, uint shift) {
    return (shift == 0) ? lo : (lo >> shift) | (hi << (32 - shift));
}

inline uint portable_bitrev(uint x) {
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return ((x >> 16) | (x << 16));
}

__kernel void compute_kernel_cl(__global const uint *flat,
                               __global const uint *flatmod9sieve,
                               __global const uint *d_pairs,
                               uint bigindex, // offset into flat
                               uint threads_launched, // usually BATCH_SIZE
                               uint magic, // needed for computing additional offset into flat
                               uint tp_0, // limb 0 of some large power of 3
                               uint tp_1, // limb 1 of some large power of 3
                               uint tp_2, // limb 2 of some large power of 3
                               uint tp_3, // limb 3 of some large power of 3
                               uint mbi, // MAXBITSINITIAL: the number of significant bits of base0
                               uint lnc,// LARGENUMBERCUTOFF: cutoff for when we print.
                                        // Print based on limb 3 of num
                               __global uint *d_hash) {


    const unsigned short THREEPOWERSU16[9] = {1, 3, 9, 27, 81, 243, 729, 2187, 6561};

    uint idx = get_global_id(0);
    uint slot = idx % HASHTABLESIZE;

    if (idx >= threads_launched) return;

    uint base0_0 = d_pairs[8 * idx];
    uint base0_1 = d_pairs[8 * idx + 1];
    uint base0_2 = d_pairs[8 * idx + 2];
//     uint base0_3 = d_pairs[8 * idx + 3]; // We assume base fits in 96 bits.
    uint number0_0 = d_pairs[8 * idx + 4];
    uint number0_1 = d_pairs[8 * idx + 5];
    uint number0_2 = d_pairs[8 * idx + 6];
    uint number0_3 = d_pairs[8 * idx + 7];

    uint rolling_hash = 0;

    uint temp = (base0_0 & 0x3FFFF) + (base0_0 >> 18)
              + 4 * (base0_1 & 0x3FFFF) + 4 * (base0_1 >> 18)
              + 7 * (base0_2 & 0x3FFFF) + 7 * (base0_2 >> 18);
              // We assume base0 fits in 96 bits.
    uint redmod9 = temp % 9;
    uint index = 0;
    unsigned short avalues[570]; //Worst case scenario we need this many a's. Based on A = 10.
    uint alength = 0;

    index = (magic * number0_0) & ((1 << STEPSIZEINITIAL) - 1);

    unsigned short offset = index % 32;

    for (uint a = 0; a < LASTBITSTEPS; a++) {
        // Mod 9 Preimage Sieve:
        uint temp = flatmod9sieve[LASTBITSTEPS * redmod9 + a]
                 & funnelshift_r(flat[INITIALBITARRAYSIZE * bigindex + index/32 + a],
                                 flat[INITIALBITARRAYSIZE * bigindex + index/32 + a + 1], offset);

        uint revtemp = portable_bitrev(temp);
        while (revtemp != 0) {
            uint lz = clz(revtemp);
            revtemp &= ~(0x80000000u >> lz);
            avalues[alength] = (unsigned short)(32 * a + lz);
            alength += 1;
        }
    }

    if (alength == 0) { return; }

    uint pos = 0;
    unsigned short aval = avalues[pos];

    uint base_0 = base0_0;
    uint base_1 = base0_1;
    uint base_2 = base0_2;

    if (mbi > 0 && mbi < 32) {
        base_0 |= (uint)aval << mbi;
        base_1 |= (uint)aval >> (32 - mbi);
    }
    if (mbi == 32) {
        base_1 = (uint) aval;
    }
    if (mbi > 32 && mbi < 64) {
        base_1 |= ((uint)aval << (mbi - 32));
        base_2 = (uint)aval >> (64 - mbi);
    }
    if (mbi >= 64 && mbi < 96) {
        base_2 |= (uint)aval << (mbi - 64);
    }

    // We compute the limbs of num = number0 + aval * threepowervalue
    uint carry;

    unsigned long p0 = (unsigned long)aval * tp_0;
    unsigned long p1 = (unsigned long)aval * tp_1;
    unsigned long p2 = (unsigned long)aval * tp_2;
    unsigned long p3 = (unsigned long)aval * tp_3;

    unsigned long t = p0 + number0_0;
    uint num_0 = (uint)t; carry = t >> 32;

    t = p1 + number0_1 + carry;
    uint num_1 = (uint)t; carry = t >> 32;

    t = p2 + number0_2 + carry;
    uint num_2 = (uint)t; carry = t >> 32;

    t = p3 + number0_3 + carry;
    uint num_3 = (uint)t;

    while (true) {
        // First odd iteration
        uint a = ctz(num_0 | (1ul << 8));
        uint t0 = funnelshift_r(num_0, num_1, a);
        uint t1 = funnelshift_r(num_1, num_2, a);
        uint t2 = funnelshift_r(num_2, num_3, a);
        uint t3 = num_3 >> a;
        uint m = THREEPOWERSU16[a];
        unsigned long t = (unsigned long)t0 * m;
        num_0 = (uint)t; carry = t >> 32;
        t = (unsigned long)t1 * m + carry;
        num_1 = (uint)t; carry = t >> 32;
        t = (unsigned long)t2 * m + carry;
        num_2 = (uint)t; carry = t >> 32;
        num_3 = t3 * m + carry;

        if (num_3 >= lnc) {
            printf("%X%08X%08X\n", base_2, base_1, base_0);
            num_0 = 2; num_1 = 0; num_2 = 0; num_3 = 0;
        }

        // First even iteration
        num_0 -= 1;
        uint b = ctz(num_0 | (1ul << 31));
        num_0 = funnelshift_r(num_0, num_1, b) + 1;
        num_1 = funnelshift_r(num_1, num_2, b);
        num_2 = funnelshift_r(num_2, num_3, b);
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

        // Second odd iteration
        a = ctz(num_0 | (1ul << 8));
        t0 = funnelshift_r(num_0, num_1, a);
        t1 = funnelshift_r(num_1, num_2, a);
        t2 = funnelshift_r(num_2, num_3, a);
        t3 = num_3 >> a;
        m = THREEPOWERSU16[a];
        t = (unsigned long)t0 * m;
        num_0 = (uint)t; carry = t >> 32;
        t = (unsigned long)t1 * m + carry;
        num_1 = (uint)t; carry = t >> 32;
        t = (unsigned long)t2 * m + carry;
        num_2 = (uint)t; carry = t >> 32;
        num_3 = t3 * m + carry;

        if (num_3 >= lnc) {
            printf("%X%08X%08X\n", base_2, base_1, base_0);
            num_0 = 2; num_1 = 0; num_2 = 0; num_3 = 0;
        }

        // Second even iteration
        num_0 -= 1;
        b = ctz(num_0 | (1ul << 31));
        num_0 = funnelshift_r(num_0, num_1, b) + 1;
        num_1 = funnelshift_r(num_1, num_2, b);
        num_2 = funnelshift_r(num_2, num_3, b);
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

            // Some code repetition here. There's probably a way to avoid it.
            if (mbi > 0 && mbi < 32) {
                base_0 = base0_0 | (uint)aval << mbi;
                base_1 = (uint)aval >> (32 - mbi);
            }
            if (mbi == 32) {
                base_1 = (uint) aval;
            }
            if (mbi > 32 && mbi < 64) {
                base_1 = base0_1 | ((uint)aval << (mbi - 32));
                base_2 = (uint)aval >> (64 - mbi);
            }
            if (mbi >= 64 && mbi < 96) {
                base_2 = base0_2 | (uint)aval << (mbi - 64);
            }

            uint carry;

            unsigned long p0 = (unsigned long)aval * tp_0;
            unsigned long p1 = (unsigned long)aval * tp_1;
            unsigned long p2 = (unsigned long)aval * tp_2;
            unsigned long p3 = (unsigned long)aval * tp_3;

            unsigned long t = p0 + number0_0;
            num_0 = (uint)t; carry = t >> 32;

            t = p1 + number0_1 + carry;
            num_1 = (uint)t; carry = t >> 32;

            t = p2 + number0_2 + carry;
            num_2 = (uint)t; carry = t >> 32;

            t = p3 + number0_3 + carry;
            num_3 = (uint)t;
        }
    }

    exit_sequence:
//     uint slot = idx % HASHTABLESIZE;
    atomic_xor(&d_hash[slot], rolling_hash);
}
