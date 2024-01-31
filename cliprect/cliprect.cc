/*
 *  Copyright (C) 2011  Clifford Wolf <clifford@clifford.at>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *
 *  Inspired by:
 *  http://freespace.virgin.net/hugo.elias/graphics/x_persp.htm
 */

#include <Eigen/Core>
#include <Eigen/LU>
#include <assert.h>
#include <stdio.h>

extern "C" {
#include <ppm.h>
}

// import most common Eigen types 
USING_PART_OF_NAMESPACE_EIGEN

//
// The math behind it in a nutshell:
//
// Let (x, y) be the coordinates in the original flat image and (x', y') the
// coordinates in the image with perspective distortion, then the convertion
// from (x, y) to (x', y') can be described (in the style of perspective
// OpenGL rendering) as:
//
//      [  d  e  ?  f  ]  [  x  ]     [  a  ]
//      [  g  h  ?  i  ]  [  y  ]  =  [  b  ]
//      [  ?  ?  ?  ?  ]  [  0  ]     [  ?  ]
//      [  k  l  ?  1  ]  [  1  ]     [  c  ]
//
//      x' = a / c,  y' = b / c
//
// with '?' as placeholder for ``don't care'' values. This can be rewritten
// as the linear system
//
//      x'x k + x'y l - x d - y e - f  =  - x'
//
//      y'x k + y'y l - x g - y h - i  =  - y'
//
// with the 8 unkowns d, e, f, g, h, i, k and l. This system can be solved
// using the 4 observations of the (x, y, x', y') tuples of the projected
// images corners.
//

void calcParameters(double p[8], const double x[4], const double y[4], const double xtic[4], const double ytic[4])
{
	MatrixXd M(8, 8);
	VectorXd rhs(8), para(8);

	for (int i = 0; i < 4; i++) {
		M.row(i*2 + 0) << xtic[i]*x[i], xtic[i]*y[i], -x[i], -y[i], -1,     0,    0,  0;
		M.row(i*2 + 1) << ytic[i]*x[i], ytic[i]*y[i],     0,    0,  0, -x[i], -y[i], -1;
		rhs(i*2 + 0) = -xtic[i];
		rhs(i*2 + 1) = -ytic[i];
	}

	bool ok = M.lu().solve(rhs, &para);
	assert(ok);

	for (int i = 0; i < 8; i++)
		p[i] = para[i];
}

void fwdMapXY(const double p[8], double &xtic, double &ytic, double x, double y)
{
	double k = p[0], l = p[1], d = p[2], e = p[3], f = p[4], g = p[5], h = p[6], i = p[7];
	double a = d*x + e*y + f, b = g*x + h*y + i, c = k*x + l*y + 1;
	xtic = a / c, ytic = b / c;
}

void invMapXY(const double p[8], double xt, double yt, double &x, double &y)
{
	double k = p[0], l = p[1], d = p[2], e = p[3], f = p[4], g = p[5], h = p[6], i = p[7];
	// Maxima: string(solve([xt*x*k + xt*y*l - x*d - y*e - f = -xt, yt*x*k + yt*y*l - x*g - y*h - i = -yt], [x,y]));
	x = -(e*(yt-i)-f*l*yt+(i*l-h)*xt+f*h)/(-d*l*yt+e*k*yt+(g*l-h*k)*xt+d*h-e*g);
	y = (d*(yt-i)-f*k*yt+(i*k-g)*xt+f*g)/(-d*l*yt+e*k*yt+(g*l-h*k)*xt+d*h-e*g);
}

void imageMapFwd(const double p[8], const char *src1, const char *src2, const char *trg)
{
	FILE *f;

	int cols1, rows1, cols2, rows2;
	pixval coldepth1, coldepth2;
	pixel **pixel1, **pixel2;

	f = fopen(src1, "r");
	assert(f != NULL);
	pixel1 = ppm_readppm(f, &cols1, &rows1, &coldepth1);
	fclose(f);

	f = fopen(src2, "r");
	assert(f != NULL);
	pixel2 = ppm_readppm(f, &cols2, &rows2, &coldepth2);
	fclose(f);

	assert(coldepth1 == coldepth2);

	for (int i = 0; i < cols2; i++)
	for (int j = 0; j < rows2; j++) {
		double xt, yt;
		fwdMapXY(p, xt, yt, i, j);
		int k = round(xt), l = round(yt);
		pixel1[l][k] = pixel2[j][i];
	}

	f = fopen(trg, "w");
	ppm_writeppm(f, pixel1, cols1, rows1, coldepth1, 0);
	fclose(f);

	ppm_freearray(pixel1, rows1);
	ppm_freearray(pixel2, rows2);
}

void imageMapInv(const double p[8], const char *src1, const char *src2, const char *trg)
{
	FILE *f;

	int cols1, rows1, cols2, rows2;
	pixval coldepth1, coldepth2;
	pixel **pixel1, **pixel2;

	f = fopen(src1, "r");
	assert(f != NULL);
	pixel1 = ppm_readppm(f, &cols1, &rows1, &coldepth1);
	fclose(f);

	f = fopen(src2, "r");
	assert(f != NULL);
	pixel2 = ppm_readppm(f, &cols2, &rows2, &coldepth2);
	fclose(f);

	assert(coldepth1 == coldepth2);

	for (int i = 0; i < cols1; i++)
	for (int j = 0; j < rows1; j++) {
		double x, y;
		invMapXY(p, i, j, x, y);
		int k = round(x), l = round(y);
		if (l >= 0 && l < rows2 && k >= 0 && k < cols2)
			pixel1[j][i] = pixel2[l][k];
	}

	f = fopen(trg, "w");
	ppm_writeppm(f, pixel1, cols1, rows1, coldepth1, 0);
	fclose(f);

	ppm_freearray(pixel1, rows1);
	ppm_freearray(pixel2, rows2);
}

int main()
{
	double x[4]    = {   0,   0, 255, 255 };
	double y[4]    = {   0, 172, 172,   0 };
	double xtic[4] = {  56,  53, 215, 216 };
	double ytic[4] = {  26, 230, 238, 121 };
	double para[8];

	calcParameters(para, x, y, xtic, ytic);

	printf("\n== Forward Mapping ==\n\n");

	for (int i = 0; i < 4; i++) {
		double xt, yt;
		fwdMapXY(para, xt, yt, x[i], y[i]);
		printf("x'[%d]: %8.2f %8.2f\n", i, xt, xtic[i]);
		printf("y'[%d]: %8.2f %8.2f\n\n", i, yt, ytic[i]);
	}

	imageMapFwd(para, "image1.ppm", "image2.ppm", "image3.ppm");

	printf("\n== Inverse Mapping ==\n\n");

	for (int i = 0; i < 4; i++) {
		double xo, yo;
		invMapXY(para, xtic[i], ytic[i], xo, yo);
		printf("x'[%d]: %8.2f %8.2f\n", i, xo, x[i]);
		printf("y'[%d]: %8.2f %8.2f\n\n", i, yo, y[i]);
	}

	imageMapInv(para, "image1.ppm", "image2.ppm", "image4.ppm");

	printf("\n== Identity Test ==\n\n");

	for (int i = 0; i < 10; i++) {
		double x = drand48() * 100 - 50;
		double y = drand48() * 100 - 50;
		double xtic, ytic, x2, y2;
		fwdMapXY(para, xtic, ytic, x, y);
		invMapXY(para, xtic, ytic, x2, y2);
		printf("x[%d]: %8.2f %8.2f %8.2f [%g]\n", i, x, xtic, x2, x-x2);
		printf("y[%d]: %8.2f %8.2f %8.2f [%g]\n", i, y, ytic, y2, y-y2);
	}

	printf("\n");

	return 0;
}

