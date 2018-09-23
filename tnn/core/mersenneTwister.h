#if !defined(_TNN_UTILS_MERSENNE_TWISTER_H)
#define _TNN_UTILS_MERSENNE_TWISTER_H

#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>

// https://en.wikipedia.org/wiki/Mersenne_Twister#Pseudocode

#define TNN_GENERATE_MT(T,struct_name,func_prefix, w,n,m,r, a, u,d, s,b, t,c, l,f) \
typedef struct struct_name\
{\
	T MT[n];\
	uint16_t index;\
} struct_name;\
\
void func_prefix##_seed(struct_name *mt, T seed)\
{\
	mt->index = n;\
	mt->MT[0] = seed;\
	for (uint16_t i = 1; i < n; ++i)\
		mt->MT[i] = (f * (mt->MT[i-1] ^ (mt->MT[i-1] >> (w-2)))) & 0xffffffffffffffff;\
}\
\
static void func_prefix##_twist (struct_name *mt)\
{\
	const T LOWER_MASK = (1 << r) - 1;\
	const T UPPER_MASK = (1 << w) & (~LOWER_MASK);\
	for (uint16_t i = 1; i < n; ++i)\
	{\
		T x = (mt->MT[i] & (UPPER_MASK)) + (mt->MT[(i+1) % n] & LOWER_MASK);\
		T xA = x >> 1;\
\
		if ((x % 2) != 0)\
			xA ^= a;\
\
		mt->MT[i] = mt->MT[(i + m) % n] ^ xA;\
	}\
	mt->index = 0;\
}\
\
T func_prefix##_next(struct_name *mt)\
{\
	if (mt->index >= n)\
	{\
		if(mt->index > n)\
		{\
			fprintf(stderr, #struct_name #func_prefix "Twist was called before" #func_prefix "Seed"); return 0;\
		}\
		func_prefix##_twist(mt);\
	}\
\
	T y = mt->MT[mt->index];\
	y ^= ((y >> u) & d);\
	y ^= ((y << s) & b);\
	y ^= ((y << t) & c);\
	y ^= (y >> 1);\
\
	++mt->index;\
	return y & 0xffffffffffffffff;\
}

TNN_GENERATE_MT(uint32_t, tnn_mersenneTwister, tnn_mt,
			32, 624, 397, 31,
				0x9908b0df,
			11, 0xffffffff,
			7,  0x9d2c5680,
			15, 0xefc60000,
			18, 1812433253)

TNN_GENERATE_MT(uint64_t, tnn_mersenneTwister_64, tnn_mt64,
			64, 312, 156, 31,
				0xb5026f5aa96619e9,
			29, 0x5555555555555555,
			17, 0x71d67fffeda60000,
			37, 0xfff7eee000000000,
			43, 6364136223846793005)

#endif // _TNN_UTILS_MERSENNE_TWISTER_H