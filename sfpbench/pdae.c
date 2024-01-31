//
// Copyright (C) 2014  Clifford Wolf <clifford@clifford.at>
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// ----
//
// PDAE Algorithm
//

#define _GNU_SOURCE

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

float fingerprint_lp[128][128];
float fingerprint_hp[128][128];
float fingerprint_shp[128];

float weightmap_lp[128][128];
float weightmap_hp[128][128];
float weightmap_shp[128];

static inline float lookup2d(float fp[128][128], float ux, float vy, float *dx, float *dy)
{
    int u = ux, v = vy;
    float x = ux-u, y = vy - v;

    float p1 = fp[u+0][v+0];
    float p3 = fp[u+1][v+0];
    float p7 = fp[u+0][v+1];
    float p9 = fp[u+1][v+1];

    if (dx == NULL) {
        float p2 = p1*(1.0f - x) + p3*x;
        float p8 = p7*(1.0f - x) + p9*x;
        if (dy != NULL)
            *dy = p8 - p2;
        return p2*(1.0f - y) + p8*y;
    }

    if (dy == NULL) {
        float p4 = p1*(1.0f - y) + p7*y;
        float p6 = p3*(1.0f - y) + p9*y;
        if (dx != NULL)
            *dx = p6 - p4;
        return p4*(1.0f - x) + p6*x;
    }

    float p2 = p1*(1.0f - x) + p3*x;
    float p8 = p7*(1.0f - x) + p9*x;
    *dy = p8 - p2;

    float p4 = p1*(1.0f - y) + p7*y;
    float p6 = p3*(1.0f - y) + p9*y;
    *dx = p6 - p4;

    return p2*(1.0f - y) + p8*y;
}

static inline float lookup1d(float fp[128], float ux, float *dx)
{
    int u = ux;
    float x = ux-u;

    float p1 = fp[u+0];
    float p3 = fp[u+1];

    if (dx != NULL)
      *dx = p3 - p1;

    return p1*(1.0f - x) + p3*x;
}

void pdae(float *ux_p, float *vy_p, float samples[11])
{
    float ux = *ux_p, vy = *vy_p;
    float sum_dx2 = 0, sum_dy2 = 0, sum_dxy = 0, sum_dxr = 0, sum_dyr = 0;
    float dy, dx, val, wght, diff;

    for (int i = 0; i < 5; i++)
    {
        // LP fingerprint

        val = lookup2d(fingerprint_lp, ux, vy + 16*i, &dx, &dy);
        wght = lookup2d(weightmap_lp, ux, vy + 16*i, NULL, NULL);
        diff = samples[i] - val;

        sum_dx2 += wght * wght * dx * dx;
        sum_dy2 += wght * wght * dy * dy;
        sum_dxy += wght * wght * dx * dy;
        sum_dxr += wght * wght * dx * diff;
        sum_dyr += wght * wght * dy * diff;

        // HP fingerprint

        val = lookup2d(fingerprint_hp, ux, vy + 16*i, &dx, &dy);
        wght = lookup2d(weightmap_hp, ux, vy + 16*i, NULL, NULL);
        diff = samples[5+i] - val;

        sum_dx2 += wght * wght * dx * dx;
        sum_dy2 += wght * wght * dy * dy;
        sum_dxy += wght * wght * dx * dy;
        sum_dxr += wght * wght * dx * diff;
        sum_dyr += wght * wght * dy * diff;
    }

    // SHP fingerprint

    val = lookup1d(fingerprint_shp, ux, &dx);
    wght = lookup1d(weightmap_shp, ux, NULL);
    diff = samples[10] - val;

    sum_dx2 += wght * wght * dx * dx;
    sum_dxr += wght * wght * dx * diff;

    // solve linear lsq problem

    double det_base = sum_dx2 * sum_dy2 - sum_dxy * sum_dxy;
    double det_x = sum_dxr * sum_dy2 - sum_dyr * sum_dxy;
    double det_y = sum_dx2 * sum_dyr - sum_dxy * sum_dxr;
    double delta_x = det_x / det_base;
    double delta_y = det_y / det_base;

    *ux_p = ux + delta_x;
    *vy_p = vy + delta_y;
}

int main(int argc, char **argv)
{
    int iter = 1000000;

    if (argc >= 2)
        iter = atoi(argv[1]);

    // init fp data

    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j++) {
            fingerprint_lp[i][j] = drand48();
            fingerprint_hp[i][j] = drand48();
            weightmap_lp[i][j] = drand48();
            weightmap_hp[i][j] = drand48();
        }
        fingerprint_shp[i] = drand48();
        weightmap_shp[i] = drand48();
    }

    // run test

    float ux = 64, vy = 20;
    float samples[11] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    while (iter-- > 0) {
        if (ux < 1 || ux > 127 || vy < 1 || vy > 40)
            ux = 64, vy = 20;
        pdae(&ux, &vy, samples);
    }

    // print results
    printf("%f %f\n", ux, vy);

    return 0;
}

