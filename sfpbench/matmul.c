//
// Copyright (C) 2014  Clifford Wolf <clifford@clifford.at>
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// ----
//
// Dense Matrix Multiplication (10x10)
//
// multiply two 10x10 matrices. this is large enough to allow for vectorization
// and loop unrolling to make a difference and small enough so that the whole
// problem fits into the cpu cache.
//
// The test matrices are orthogonal and the intermediate matrices are re-orthogonalized
// every 30 multiplications to keep the matrix determinants near 1. The output is not
// stable: Small variations in intermediate floating point values have a large impact
// on the overall result!
//

#define _GNU_SOURCE

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// matrix multiply: O(n^3)
void mul_mat_10_10(const float * restrict A, const float * restrict B, float * restrict Y)
{
    int i, j, k;
    for (i = 0; i < 10; i++)
    for (j = 0; j < 10; j++) {
        float s = 0;
        for (k = 0; k < 10; k++)
            s += A[i*10+k] * B[k*10+j];
        Y[i*10+j] = s;
    }
}

// gram-schmidt process: O(n^3)
void make_orthogonal(float *M)
{
    int i, j, k;

    for (i = 0; i < 10; i++)
    {
        for (j = 0; j < i; j++) {
            float s = 0;
            for (k = 0; k < 10; k++)
                s += M[i*10 + k] * M[j*10 + k];
            for (k = 0; k < 10; k++)
                M[i*10 + k] -= s * M[j*10 + k];
        }

        float s = 0;
        for (k = 0; k < 10; k++)
            s += M[i*10 + k] * M[i*10 + k];
        for (k = 0; k < 10; k++)
            M[i*10 + k] /= sqrtf(s);
    }
}

int main(int argc, char **argv)
{
    int iter = 1000000;

    if (argc >= 2)
        iter = atoi(argv[1]);

    // create test data

    float A[100], B[100], Y[100];

    int i, j;
    for (int i = 0; i < 10; i++)
    for (int j = 0; j < 10; j++) {
        A[i*10+j] = drand48();
        B[i*10+j] = drand48();
    }

    make_orthogonal(A);
    make_orthogonal(B);

    // run test

    while (iter-- > 0) {
        if (iter % 10 == 0) {
            make_orthogonal(A);
            make_orthogonal(B);
        }
        mul_mat_10_10(A, B, Y);
        mul_mat_10_10(B, Y, A);
        mul_mat_10_10(Y, A, B);
    }

    // print results

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++)
            printf("%10.2e", B[i*10+j]);
        printf("\n");
    }

    return 0;
}

