/*
This program verifies the Collatz conjecture for n < 2^N. See the paper that is also posted https://github.com/vigleik0/collatz.

It incorporates several new ideas:
1. The first part of the program, the function search1(), does binary search on the bits of the starting number n. We start with the least significant bit, and abort if we reach a value < n or if the path of n joins the path of some n' < n. This works because the first k division by 2 operations depend only of the first k bits of n, and if this results in a number < n then n + a2^k will result in a number < n + a2^k.

2. The second part of the program, the function search2(), uses bitwise operations to check which ways of adding the most significant bits of n are allowed. This incorporates both mod 9 sieving, and looking ahead up to 24 iterations to see if the result ever dips below the initial value.

The relevant bit arrays are precomputed in the function precompute(), and the relevant data is accessible through the struct PrecomputedData.

3. The third part of the program, the function search3(), performs 16 iterations at a time using a lookup table.

Note that we work with T^k(n) + 1 internally instead of T^k(n).
*/
#![allow(non_snake_case)]
use std::io::{self, Read};

const MAXBITS: u32 = 72;
const STEPSIZE: usize = 16;
const LASTBITS: u32 = 6; //tuned.
const LASTBITSTEPS: usize = ((1 << LASTBITS) + 63)/64;
const MAXBITSINITIAL: u32 = MAXBITS - LASTBITS;
const STEPSIZEINITIAL: usize = 24;
const NUMERATOR: u32 = 306; // NUMERATOR/DENOMINATOR is a rational approximation to log(2)/log(3)
const DENOMINATOR: u32 = 485;
const THREEPOWEROFFSET: u32 = MAXBITSINITIAL * NUMERATOR / DENOMINATOR + 1;
const ONETHIRD: u32 = 2863311531; // Multiplicative inverse in the units in Z/2^32.
const PRINTINGCASES: bool = false;
const TRACKINGPROGRESS: bool = false;
const PRINTINGSTATS: bool = false;
const NUMARRAYS: usize = 8;

const fn pow3(mut i: usize) -> u128 {
    let mut x: u128 = 1;
    while i > 0 {
        x *= 3;
        i -= 1;
    }
    x
}

const fn make_array() -> [u128; MAXBITS as usize] {
    let mut arr = [0u128; MAXBITS as usize];
    let mut i = 0;
    while i < MAXBITS as usize {
        arr[i] = pow3(i);
        i += 1;
    }
    arr
}

// This makes THREEPOWERS as a constant array at compile time.
const THREEPOWERS: [u128; MAXBITS as usize] = make_array();

//It's (marginally) better to use this one in search3, because even though we'll promote it to a u128 the compiler can prove that not all the u128 * u128
//multiplications are necessary.
const THREEPOWERSU32: [u32; 17] = [1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441, 1594323, 4782969, 14348907, 43046721];

struct PrecomputedData {
    multisteparray: Vec<u32>,
    initialbitarray: Vec<Vec<u64>>,
    magicnumbers: [u32; NUMARRAYS],
    mod9sieve: [Vec<u64>; 9],
}


fn main() {
    // We make the array a little bit longer than necessary to avoid having to wrap around.
    let lengthofinitialbitarray = (1 << (STEPSIZEINITIAL - 6)) + ((1 << LASTBITS) + 63)/64;
    let lengthofmod9sieve = ((1 << LASTBITS) + 63)/64;
    let mut data = PrecomputedData {
        multisteparray: vec![0u32; 1 << STEPSIZE],
        initialbitarray: vec![vec![0u64; lengthofinitialbitarray]; NUMARRAYS],
        magicnumbers: [0u32; NUMARRAYS],
        mod9sieve: std::array::from_fn(|_| vec![0u64; lengthofmod9sieve]),
    };
    let mut magicnumber = 1u32;
    for i in 1..80 {
        magicnumber *= ONETHIRD;
        if i >= THREEPOWEROFFSET && i - THREEPOWEROFFSET < NUMARRAYS as u32 {
            data.magicnumbers[i as usize - THREEPOWEROFFSET as usize] = magicnumber;
        }
    }

    precompute(&mut data);
    eprintln!("Done with precompute");

    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();

    let nums: Vec<u64> = input
        .split_whitespace()
        .map(|s| s.parse::<u64>().unwrap())
        .collect();

    // Do search1(3, 2, 2, 9) for initial one.
    let numcases = search1(nums[0] as u128, nums[1] as u32, nums[2] as u32, nums[3] as u128, 0, 0, &data);
    eprintln!("numcases = {}", numcases);
}

fn doubleeven(upordown: u32, i: usize) -> bool {
    let a = ((upordown >> i) + 1).trailing_zeros() as usize;
    if i + a + 2 >= STEPSIZEINITIAL { return false; }
    if (upordown >> (i + a)) & 0b11 == 0 { return true; }

    return false;
}

fn precompute(data: &mut PrecomputedData) {
    // First the data for doing 16 steps at a time in search3:
    for n0 in 0..(1u32 << STEPSIZE) {
        let mut threepower = 0u32;
        let mut n = n0;
        for _ in 0..STEPSIZE {
            if n & 1 == 0 { //Have 3n+1 branch
                n = n + n/2;
                threepower += 1;
            } else { //have n/2 branch
                n = n/2 + 1;
            }
        }
        data.multisteparray[n0 as usize] = threepower | (n << 5);
    }

    let firstcutoff = (DENOMINATOR * THREEPOWEROFFSET - NUMERATOR * MAXBITSINITIAL) as i32;
    let cutoffs: [i32; NUMARRAYS] = std::array::from_fn(|i| firstcutoff + i as i32 * DENOMINATOR as i32);

    let wraparoundcutoff = ((1 << LASTBITS) + 63)/64;
    let wraparoundadd = 1 << (STEPSIZEINITIAL - 6);

    // Now compute all the initialbitarray[a] at once.
    for n0 in 0..(1u64 << STEPSIZEINITIAL) {
        let mut upordown = 0u32;
        let mut n = n0;
        // First compute the sequence of even and odd steps.
        for i in 0..STEPSIZEINITIAL {
            if n & 1 == 0 {
                n = n + n/2;
                upordown |= 1 << i;
            } else {
                n = n/2 + 1;
            }
        }
        let mut curval = 0i32;
        let mut maxval = 0i32;
        let mut numevens = 0u32;
        // Then compute the maximal dip.
        for i in 0..STEPSIZEINITIAL {
            if upordown & (1 << i) != 0 {
                if curval + NUMERATOR as i32 > maxval {
                    // This is the two even steps sieve from the paper.
                    if doubleeven(upordown, i) == true {
                        maxval = curval + NUMERATOR as i32;
                    }
                }
                curval -= (DENOMINATOR - NUMERATOR) as i32;
                numevens = 0;
            } else {
                curval += NUMERATOR as i32;
                numevens += 1;
                // If numevens % 2 == 0 then we can "go backwards". This is the mod 3
                // sieve from the paper.
                if numevens % 2 == 0 {
                    if curval + (DENOMINATOR - NUMERATOR) as i32 > maxval {
                        maxval = curval + (DENOMINATOR - NUMERATOR) as i32;
                    }
                 }
            }
            if curval > maxval { maxval = curval; }
        }
        for a in 0..NUMARRAYS {
            if maxval <= cutoffs[a] {
                let index = n0 as usize * data.magicnumbers[a] as usize & ((1 << STEPSIZEINITIAL) - 1);
                data.initialbitarray[a][index/64] |= 1u64 << (index % 64);

                if index/64 < wraparoundcutoff {
                    data.initialbitarray[a][index/64 + wraparoundadd] |= 1 << (index % 64);
                }
            }
        }
    }

    if PRINTINGSTATS == true {
        for a in 0..NUMARRAYS {
            let mut tally = 0;
            for i in 0..(1 << (STEPSIZEINITIAL - 6)) {
                tally += data.initialbitarray[a][i].count_ones() as u64;
            }
            println!("Percentage for {} = {}", a, tally as f64 / (1 << STEPSIZEINITIAL) as f64);
        }
    }

    let redmod9step = ((1u128 << MAXBITSINITIAL) % 9) as usize;
    for a in 0..9 {
        let mut redmod9 = a;
        for i in 0..(64 * data.mod9sieve[a].len()) {
            if redmod9 == 0 || redmod9 == 1 || redmod9 == 3 || redmod9 == 6 || redmod9 == 7 {
                data.mod9sieve[a][i/64] |= 1 << (i % 64);
            }
            redmod9 += redmod9step;
            if redmod9 >= 9 { redmod9 -= 9; }
        }
    }
}

fn search1(base: u128, // starting number, accurate mod 2^sigfig
           sigfig: u32, // number of iterations so far
           threepower: u32, // number of odd iterations so far
           number: u128, // number - 1 = T^sigfig(base)
           doubleeven: u32, // if doubleevens == 1 that means we can abort after 2 even iterations in a row.
                            // if doubleevens == 2 that means we can abort after 1 even iteration, so we don't take the even branch.
           numevens: u32, // the number of even iterations in a row. If numevens % 2 == 0 then m = number - 1 = 2 mod 3 and the path joins that
                          // of (2m-1)/3.
           data: &PrecomputedData // Contains data needed for search2. We're just keeping it safe.
           ) -> u64 {
    if number <= base { return 0; }
    if NUMERATOR * sigfig >= DENOMINATOR * threepower { return 0; }

    // This checks if m = number - 1 is congruent to 2 mod 3. If it is then its path joins
    // with (2m-1)/3, and if (2m-1)/3 is smaller than base we can abort.
    // By doing it this way we avoid checking the number we just came from.
    // When we go backwards that decreases sigfig by 1 and threepower by 1. Saves ~7%
    if numevens >= 2 && numevens % 2 == 0 {
        if NUMERATOR * (sigfig - 1) >= DENOMINATOR * (threepower - 1) { return 0; }
    }

    if sigfig == MAXBITSINITIAL {
        return search2(base, threepower, number, doubleeven, data);
    }

    let doubleinplay: u32 = { if NUMERATOR * (sigfig + 1) >= DENOMINATOR * threepower { 1 } else { 0 } };

    // Special code for splitting up into cases.
    // cases_small contains 1390 cases and cases_big contains 328718 cases. (One per line.)
    if PRINTINGCASES == true {
        if sigfig == 14 && threepower <= 9 { //14 and 9 for small case, and 23 and 15 for big case.
            println!("{} {} {} {} {} {}", base, sigfig, threepower, number, doubleeven, numevens);
            return 1;
        }
        if sigfig == 15 && threepower <= 11 {
            println!("{} {} {} {} {} {}", base, sigfig, threepower, number, doubleeven, numevens);
            return 1;
        }
        if sigfig == 16 && threepower <= 13 {
            println!("{} {} {} {} {} {}", base, sigfig, threepower, number, doubleeven, numevens);
            return 1;
        }
        if sigfig == 17 && threepower <= 15 {
            println!("{} {} {} {} {} {}", base, sigfig, threepower, number, doubleeven, numevens);
            return 1;
        }
        if sigfig == 18 {
            println!("{} {} {} {} {} {}", base, sigfig, threepower, number, doubleeven, numevens);
            return 1;
        }
    }

    // This is for tracking progress only.
    if TRACKINGPROGRESS == true {
        if sigfig == 36 {
            println!("{:#038b}", base);
        }
    }

    let numberA = number;
    let numberB = number + THREEPOWERS[threepower as usize];

    if number & 1 == 0 { //Case A is 3n+1 branch and case B is n/2 branch
        let resL = search1(base,                 sigfig + 1, threepower + 1, numberA + numberA/2, (doubleeven | doubleinplay) & !2, 0, data);
        let resR = { if doubleeven >= 2 { 0 } else { search1(base | (1 << sigfig), sigfig + 1, threepower    , numberB/2 + 1      , doubleeven << 1, numevens + 1, data) } };
        return resL + resR;

    } else { // Case A is n/2 branch and case B is 3n+1 branch
        let resL = { if doubleeven >= 2 { 0 } else { search1(base,                 sigfig + 1, threepower    , numberA/2 + 1      , doubleeven << 1, numevens + 1, data) } };
        let resR = search1(base | (1 << sigfig), sigfig + 1, threepower + 1, numberB + numberB/2, (doubleeven | doubleinplay) & !2, 0, data);
        return resL + resR;
    }
}

fn search2(base: u128,
           threepower: u32,
           number: u128,
           doubleeven: u32,
           data: &PrecomputedData
           ) -> u64 {
    let mut tally = 0u64;
    let redmod9 = (base % 9) as usize;
    let mut index = 0;
    if threepower < THREEPOWEROFFSET + NUMARRAYS as u32 {
        let n0 = number as u32;
        index = ((data.magicnumbers[(threepower - THREEPOWEROFFSET) as usize] * n0) & ((1 << STEPSIZEINITIAL) - 1)) as usize;
    }
    let offset = index % 64;

    for a in 0..LASTBITSTEPS {
        let mut temp = data.mod9sieve[redmod9][a];
        if threepower < THREEPOWEROFFSET + 8 {
            if offset == 0 {
                temp &= data.initialbitarray[(threepower - THREEPOWEROFFSET) as usize][index/64 + a];
            } else {
                temp &= (data.initialbitarray[(threepower - THREEPOWEROFFSET) as usize][index/64 + a] >> offset)
                      | (data.initialbitarray[(threepower - THREEPOWEROFFSET) as usize][index/64 + a + 1] << (64 - offset))
            }
        }
        while temp != 0 {
            let highbits = 64 * a as u64 + temp.trailing_zeros() as u64;
            temp &= temp-1;
            // These checks only shave off ~0.3% of the number of cases, so might not be worth it.
            if doubleeven == 2 && (number as u64 + highbits) % 2 == 1 { continue; }
            if doubleeven == 1 {
                if threepower % 2 == 0 && (number as u64 + highbits) % 4 == 1 { continue; }
                if threepower % 2 == 1 && (number as u64 + 3 * highbits) % 4 == 1 { continue; }
            }
            tally += search3(base + ((highbits as u128) << MAXBITSINITIAL), number + (highbits as u128) * THREEPOWERS[threepower as usize], data);
        }
    }

    return tally;
}

fn search3(base: u128,
           mut number: u128,
           data: &PrecomputedData
           ) -> u64 {

    let mut tally = 0;
    loop {
        let lookup = data.multisteparray[(number as usize) & ((1 << STEPSIZE) - 1)];
        let factorterm = THREEPOWERSU32[(lookup & 0x1f) as usize];
        let additionterm = lookup >> 5;
        number = (number >> STEPSIZE) * (factorterm as u128) + additionterm as u128;

        tally += 1;
        if number < base { return tally; }

        if (number >> 64) >= (1 << 56) {
            println!("{}", base);
            return tally;
        }
    }
}
