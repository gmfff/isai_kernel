#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>


#include<iostream>
#include<sys/time.h>
//#include<stdlib.h>
//#include<stdio.h>
#include<cuda.h>
#include "config.h"

// common.h
#ifdef USE_DOUBLE
    #define VALUE_TYPE double
#else
    #define VALUE_TYPE float
#endif


#ifndef BENCH_REPEAT
#define BENCH_REPEAT BENCH_TIMING_ITERS
#endif

#ifndef WARP_SIZE
#define WARP_SIZE   32
#endif

#ifndef WARP_PER_BLOCK
#define WARP_PER_BLOCK  32
#endif

#ifndef MAX_BLOCK
#define MAX_BLOCK 65000
#endif

#ifndef CACHE_N
#define CACHE_N 16
#endif
//
//#ifndef CPU_CU_NUM
//#define CPU_CU_NUM  4
//#endif
