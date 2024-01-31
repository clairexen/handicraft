/*
 *  Trace PCB bitmaps and create gcode for CNC manufacturing
 *
 *  Copyright (C) 2010  Clifford Wolf <clifford@clifford.at>
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
 */

#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <pbm.h>
#include <ppm.h>

// command line settings
float offset_x, offset_y;
float safe_z = 1, cut_z;
float feed_rate = 100;
float drill_diameter = 0.8;
int resolution_dpi = 500;
int cnc_iterations = 1;
int mode_alt_png_output;
int mode_use_rect_fill;
int mode_use_oct_fill;
int mode_drill;
int mode_outline;
int mode_dont_mirror;
int print_timing;
int tolerance_divider;
char *gcode_preamble;
char *gcode_postamble;

struct node_s {
	int x, y;
	float level;
	int edges_nr;
	int edges[4];
};

struct pixel_s {
	float level;
	int area_nr;
	// the node sits on the upper left corner of this pixel
	int node;
};

struct drill_s {
	int pixelcount;
	int sum_x, sum_y;
	float x, y, r;
};

// this bit is stored in pixel_s.area_nr
#define PIXEL_QUEUED_FLAG (1<<30)

int area_count;
int width, height;
struct pixel_s *pixmap;
struct drill_s *drillmap;
#define P(_x, _y) pixmap[(_y)*width + (_x)]

int *queue;
int queue_begin, queue_end;
float ntab_distance[8];
int ntab[8];

int nodes_nr, nodes_top;
struct node_s *nodes;

int sum_time;
struct timeval sum_start;

static void sum_time_start()
{
	gettimeofday(&sum_start, NULL);
}

static void sum_time_stop()
{
	struct timeval sum_stop;
	gettimeofday(&sum_stop, NULL);
	int udiff = 1000000L*(sum_stop.tv_sec - sum_start.tv_sec);
	udiff += ((int)sum_stop.tv_usec - (int)sum_start.tv_usec);
	sum_time += udiff;
}

static void print_time(const char *msg)
{
	sum_time_stop();
	if (print_timing)
		fprintf(stderr, "[%9d] %s\n", sum_time, msg);
	sum_time_start();
	sum_time = 0;
}

static inline void queue_push(int idx)
{
	queue[queue_end] = idx;
	queue_end = (queue_end+1) % (width*height);

	if (queue_begin == queue_end) {
		fprintf(stderr, "This is very unlikely but possible.. oops!\n");
		abort();
	}
}

static inline int queue_shift()
{
	if (queue_begin == queue_end)
		return -1;

	int val = queue[queue_begin];
	queue_begin = (queue_begin+1) % (width*height);
	return val;
}

static inline int is_on_border(int idx)
{
	int x = idx % width, y = idx / width;
	return x <= 0 || x >= width-1 || y <= 0 || y >= height-1;
}

static inline int node(int x, int y)
{
	if (P(x,y).node == -1) {
		if (nodes_nr == nodes_top) {
			if (nodes_top)
				fprintf(stderr, "INFO: Doubling nodes space.\n");
			nodes_top = nodes_top ? nodes_top*2 : (width * height) / 10;
			nodes = realloc(nodes, nodes_top * sizeof(struct node_s));
		}
		nodes[nodes_nr].x = x;
		nodes[nodes_nr].y = y;
		if (tolerance_divider <= 0)
			nodes[nodes_nr].level = sqrt(P(x,y).level);
		else
			nodes[nodes_nr].level = P(x,y).level / tolerance_divider;
		nodes[nodes_nr].edges_nr = 0;
		P(x,y).node = nodes_nr++;
	}
	return P(x,y).node;
}

static void add_edge(int x1, int y1, int x2, int y2)
{
	int n1 = node(x1, y1);
	int n2 = node(x2, y2);

	for (int i=0; i<nodes[n1].edges_nr; i++)
		if (nodes[n1].edges[i] == n2)
			return;

	if (nodes[n1].edges_nr >= 4 || nodes[n2].edges_nr >= 4) {
		fprintf(stderr, "This acutally is impossible.\n");
		abort();
	}

	nodes[n1].edges[nodes[n1].edges_nr++] = n2;
	nodes[n2].edges[nodes[n2].edges_nr++] = n1;
}

static void remove_node(int idx)
{
	if (nodes[idx].edges_nr != 2) {
		fprintf(stderr, "Can't remove a node with more or less than 2 edges!\n");
		abort();
	}

	int n1 = nodes[idx].edges[0];
	int n2 = nodes[idx].edges[1];

	if (nodes[n1].edges_nr == 2) {
		if (nodes[idx].level < nodes[n1].level)
			nodes[n1].level = nodes[idx].level;
		queue_push(n1);
	}

	if (nodes[n2].edges_nr == 2) {
		if (nodes[idx].level < nodes[n2].level)
			nodes[n2].level = nodes[idx].level;
		queue_push(n2);
	}

	for (int i=0; i<nodes[n1].edges_nr; i++)
		if (nodes[n1].edges[i] == idx)
			nodes[n1].edges[i] = n2;

	for (int i=0; i<nodes[n2].edges_nr; i++)
		if (nodes[n2].edges[i] == idx)
			nodes[n2].edges[i] = n1;

	nodes[idx].edges_nr = 0;
}

static float level_line(int x1, int y1, int x2, int y2)
{
	float level = 0;
	int steps = abs(y2-y1) + abs (x2-x1) + 10;
	for (int i=0; i<=steps; i++) {
		int x = x1 + (x2-x1)*i/steps;
		int y = y1 + (y2-y1)*i/steps;
		if (level < P(x,y).level)
			level = P(x,y).level;
	}
	return level;
}

static void ppm_line(pixel **ppmdata, int x1, int y1, int x2, int y2)
{
	int steps = abs(y2-y1) + abs (x2-x1) + 10;
	int last_x = -1, last_y = -1;
	for (int i=0; i<=steps; i++) {
		int x = x1 + (x2-x1)*i/steps;
		int y = y1 + (y2-y1)*i/steps;
		if (last_x != x || last_y != y) {
			ppmdata[y][x].r = ppmdata[y][x].r + 20 > 200 ? 200 : ppmdata[y][x].r + 20;
			ppmdata[y][x].g = ppmdata[y][x].g + 20 > 200 ? 200 : ppmdata[y][x].g + 20;
			ppmdata[y][x].b = ppmdata[y][x].b + 20 > 200 ? 200 : ppmdata[y][x].b + 20;
			last_x = x, last_y = y;
		}
	}
}

static int compar_node_queue(const void *vp1, const void *vp2)
{
	int idx1 = *(int*)vp1;
	int idx2 = *(int*)vp2;
	if (nodes[idx1].level < nodes[idx2].level)
		return +1;
	if (nodes[idx1].level > nodes[idx2].level)
		return -1;
	return 0;
}

static int compar_drillmap(const void *vp1, const void *vp2)
{
	// simply sort drills by xy morton number
	struct drill_s *d1 = (struct drill_s *)vp1;
	struct drill_s *d2 = (struct drill_s *)vp2;
	int32_t x1 = d1->x, y1 = d1->y;
	int32_t x2 = d2->x, y2 = d2->y;
	int64_t v1 = 0, v2 = 0;
	for (int i = 0; i < 32; i++) {
		v1 |= ((x1 & (1<<i)) != 0 ? 1 : 0) << (i*2);
		v1 |= ((y1 & (1<<i)) != 0 ? 1 : 0) << (i*2+1);
		v2 |= ((x2 & (1<<i)) != 0 ? 1 : 0) << (i*2);
		v2 |= ((y2 & (1<<i)) != 0 ? 1 : 0) << (i*2+1);
	}
	if (v1 < v2)
		return -1;
	if (v1 > v2)
		return +1;
	return 0;
}

static float angle_two_vectors(float x1, float y1, float x2, float y2)
{
	float dotprod = x1*x2 + y1*y2;
	float absprod = sqrt(x1*x1 + y1*y1) * sqrt(x2*x2 + y2*y2);
	return absprod <= 0 ? 5 : acos(dotprod/absprod);
}

static int ym(int y)
{
	if (mode_dont_mirror)
		return (height-1) - y;
	return y;
}

int main(int argc, char **argv)
{
	int opt;
	const char *input_filename = NULL;
	const char *output_filename = NULL;
	const char *pixmap_filename = NULL;

	FILE *fp;
	bit **pbmdata;
	pixel **ppmdata;

	while ((opt = getopt(argc, argv, "i:o:p:r:AROtT:x:y:z:c:I:f:d:DXmE:S:")) != -1) {
		switch (opt) {
			case 'i':
				input_filename = optarg;
				break;
			case 'o':
				output_filename = optarg;
				break;
			case 'p':
				pixmap_filename = optarg;
				break;
			case 'r':
				resolution_dpi = atof(optarg);
				break;
			case 'A':
				mode_alt_png_output = 1;
				break;
			case 'R':
				mode_use_rect_fill = 1;
				break;
			case 'O':
				mode_use_oct_fill = 1;
				break;
			case 't':
				print_timing = 1;
				break;
			case 'T':
				tolerance_divider = atof(optarg);
				break;
			case 'x':
				offset_x = atof(optarg);
				break;
			case 'y':
				offset_y = atof(optarg);
				break;
			case 'z':
				safe_z = atof(optarg);
				break;
			case 'c':
				cut_z = atof(optarg);
				break;
			case 'I':
				cnc_iterations = atoi(optarg);
				break;
			case 'f':
				feed_rate = atof(optarg);
				break;
			case 'd':
				drill_diameter = atof(optarg);
				break;
			case 'D':
				mode_drill = 1;
				break;
			case 'X':
				mode_outline = 1;
				break;
			case 'm':
				mode_dont_mirror = 1;
				break;
			case 'E':
			case 'S':
				for (int i=0; optarg[i]; i++)
					if (optarg[i] == '|')
						optarg[i] = '\n';
				if (opt == 'E')
					gcode_preamble = optarg;
				else
					gcode_postamble = optarg;
				break;
			default:
help_and_exit:
				fprintf(stderr, "\n");
				fprintf(stderr, "Usage: %s -i filename.pbm [ -o filename.ngc ] [ -p filename.ppm ] [-r DPI] \\\n", argv[0]);
				fprintf(stderr, "       %*s                [-A] [-R] [-O] [-t] [-T divider] [-x X] [-y Y] \\\n", (int)strlen(argv[0])+1, "");
				fprintf(stderr, "       %*s                [-z Z] [-c Z] [-I N] [-f F] [-d d] [-D] [-X] [-m] \\\n", (int)strlen(argv[0])+1, "");
				fprintf(stderr, "       %*s                [-E preamble_str] [-S posamble_str]\n", (int)strlen(argv[0])+1, "");
				fprintf(stderr, "\n");
				fprintf(stderr, "  -i filename.pbm\n");
				fprintf(stderr, "        Input filename ('-' for stdin)\n");
				fprintf(stderr, "        This should be a netpbm image of the board. Please not that\n");
				fprintf(stderr, "        all border pixels are expected to be cleared.\n");
				fprintf(stderr, "\n");
				fprintf(stderr, "  -o filename.ngc\n");
				fprintf(stderr, "        Output filename ('-' for stdout)\n");
				fprintf(stderr, "        This is the gcode file that can be used to mill the board.\n");
				fprintf(stderr, "\n");
				fprintf(stderr, "  -p filename.ppm\n");
				fprintf(stderr, "        Optional debug image filename ('-' for stdout)\n");
				fprintf(stderr, "        This image can be used to inspect the results of the algorithm.\n");
				fprintf(stderr, "\n");
				fprintf(stderr, "  -r DPI\n");
				fprintf(stderr, "        The optional resolution of the input file in dots per inch.\n");
				fprintf(stderr, "        The default setting is 500 dpi.\n");
				fprintf(stderr, "\n");
				fprintf(stderr, "  -A\n");
				fprintf(stderr, "        Use the alternate format for the debug image (see -p above).\n");
				fprintf(stderr, "\n");
				fprintf(stderr, "  -R, -O\n");
				fprintf(stderr, "        Use rectangular or octagonal fill metrics.\n");
				fprintf(stderr, "        The default is a close-to-radial fill metric.\n");
				fprintf(stderr, "\n");
				fprintf(stderr, "  -t\n");
				fprintf(stderr, "        Print timing profile while processing\n");
				fprintf(stderr, "\n");
				fprintf(stderr, "  -T divider\n");
				fprintf(stderr, "        Use the specified fraction of the clearing area\n");
				fprintf(stderr, "        as room for optimizing the generated paths.\n");
				fprintf(stderr, "\n");
				fprintf(stderr, "  -x X, -y Y\n");
				fprintf(stderr, "        Add this x/y offset (in mm) to all gcode coordinates\n");
				fprintf(stderr, "        (default: 0)\n");
				fprintf(stderr, "\n");
				fprintf(stderr, "  -z Z\n");
				fprintf(stderr, "        The z coordinate for safe movements in mm.\n");
				fprintf(stderr, "        (default: 1)\n");
				fprintf(stderr, "\n");
				fprintf(stderr, "  -c Z\n");
				fprintf(stderr, "        Depth of cut in mm.\n");
				fprintf(stderr, "        (default: 0)\n");
				fprintf(stderr, "\n");
				fprintf(stderr, "  -I N\n");
				fprintf(stderr, "        Number of iterations for cutting/milling.\n");
				fprintf(stderr, "        (default: 1)\n");
				fprintf(stderr, "\n");
				fprintf(stderr, "  -f F\n");
				fprintf(stderr, "        The feed rate for cutting in mm/min.\n");
				fprintf(stderr, "        (default: 100)\n");
				fprintf(stderr, "\n");
				fprintf(stderr, "  -d d\n");
				fprintf(stderr, "        Diameter of the drill in mm (drilling mode only)\n");
				fprintf(stderr, "        (default: 0.8)\n");
				fprintf(stderr, "\n");
				fprintf(stderr, "  -D\n");
				fprintf(stderr, "        Run in drilling mode\n");
				fprintf(stderr, "\n");
				fprintf(stderr, "  -X\n");
				fprintf(stderr, "        Trace board outlines (cut the traces)\n");
				fprintf(stderr, "\n");
				fprintf(stderr, "  -m\n");
				fprintf(stderr, "        Do not mirror ngc output (frontside milling)\n");
				fprintf(stderr, "\n");
				fprintf(stderr, "  -E preamble_str\n");
				fprintf(stderr, "  -S posamble_str\n");
				fprintf(stderr, "        Pre- and postamble for gcode output (use '|' for newlines)\n");
				fprintf(stderr, "\n");
				return 1;
		}
	}

	if (!input_filename)
		goto help_and_exit;

	sum_time_start();

	// read input bitmap
	fp = pm_openr(input_filename);
	pbmdata = pbm_readpbm(fp, &width, &height);
	pm_close(fp);
	print_time("PBM read");

	// allocate working memory for pixel-oriented steps
	pixmap = calloc(width*height, sizeof(struct pixel_s));
	queue = calloc(width*height, sizeof(int));
	print_time("Pixmap and queue alloc");

	// import image data to pixmap
	if (mode_outline) {
		for (int x=0; x<width; x++)
		for (int y=0; y<height; y++) {
			P(x,y).level = 0;
			P(x,y).area_nr = 1;
			P(x,y).node = -1;
		}
		for (int x=1; x<width-1; x++)
		for (int y=1; y<height-1; y++) {
			if (pbmdata[y][x] != pbmdata[0][0]) {
				for (int u=-1; u<=1; u++)
				for (int v=-1; v<=1; v++) {
					P(x+u,y+v).level = width + height;
					P(x+u,y+v).area_nr = 0;
				}
			}
		}
	} else {
		for (int x=0; x<width; x++)
		for (int y=0; y<height; y++) {
			if (pbmdata[y][x] != pbmdata[0][0]) {
				P(x,y).level = 0;
				P(x,y).area_nr = 1;
			} else {
				P(x,y).level = width + height;
				P(x,y).area_nr = 0;
			}
			P(x,y).node = -1;
		}
	}
	pbm_freearray(pbmdata, height);
	print_time("import bitmap");

	// fill neighborhood table
	// (start left, continue clockwise)
	ntab_distance[0] = sqrt(1); ntab[0] = -1;
	ntab_distance[1] = sqrt(2); ntab[1] = -width-1;
	ntab_distance[2] = sqrt(1); ntab[2] = -width;
	ntab_distance[3] = sqrt(2); ntab[3] = -width+1;
	ntab_distance[4] = sqrt(1); ntab[4] = +1;
	ntab_distance[5] = sqrt(2); ntab[5] = +width+1;
	ntab_distance[6] = sqrt(1); ntab[6] = +width;
	ntab_distance[7] = sqrt(2); ntab[7] = +width-1;

	// mangle neighborhood table for alt. filling styles
	if (mode_use_rect_fill) {
		for (int i=0; i<8; i++)
			ntab_distance[i] = 1;
	}
	if (mode_use_oct_fill) {
		for (int i=0; i<8; i++)
			if (ntab_distance[i] > 1)
				ntab_distance[i] = 2;
	}
	print_time("init ntab");

	// fill with area codes
	area_count = 2;
	for (int x=1; x<width-1; x++)
	for (int y=1; y<height-1; y++) {
		if (P(x,y).area_nr == 1) {
			int idx = y*width + x;
			pixmap[idx].area_nr = area_count++;
			queue_push(idx);
			while (queue_begin != queue_end) {
				idx = queue_shift();
				for (int i=0; i<8; i++) {
					int idx2 = idx + ntab[i];
					if (pixmap[idx2].area_nr == 1 && !is_on_border(idx2)) {
						pixmap[idx2].area_nr = pixmap[idx].area_nr;
						queue_push(idx2);
					}
				}
			}
		}
	}
	print_time("init fill");

	// in drilling mode we are almost done
	if (mode_drill)
	{
		// analyse drills and create drillmap
		drillmap = calloc(sizeof(struct drill_s), area_count);
		float dr = resolution_dpi * (drill_diameter/2)/25.4;
		for (int x=0; x<width; x++)
		for (int y=0; y<height; y++) {
			int nr = P(x,y).area_nr;
			drillmap[nr].pixelcount++;
			drillmap[nr].sum_x += x;
			drillmap[nr].sum_y += y;
		}
		for (int i=0; i<area_count; i++) {
			drillmap[i].x = (float)drillmap[i].sum_x / (float)drillmap[i].pixelcount;
			drillmap[i].y = (float)drillmap[i].sum_y / (float)drillmap[i].pixelcount;
			drillmap[i].r = sqrt(drillmap[i].pixelcount / M_PI);
		}
		qsort(drillmap+2, area_count-2, sizeof(struct drill_s), compar_drillmap);
		print_time("drillmap prep");

		if (pixmap_filename)
		{
			// write output pixmap
			fp = pm_openw(pixmap_filename);
			ppmdata = ppm_allocarray(width, height);
			for (int x=0; x<width; x++)
			for (int y=0; y<height; y++) {
				ppmdata[y][x].r = 0;
				ppmdata[y][x].g = 0;
				ppmdata[y][x].b = 0;
			}
			for (int i=2; i<area_count; i++) {
				float x = drillmap[i].x;
				float y = drillmap[i].y;
				float r = drillmap[i].r;
				float mr = fmax(r, dr);
				for (int u = x-(mr+1); u <= x+(mr+1); u++)
				for (int v = y-(mr+1); v <= y+(mr+1); v++) {
					if (u < 0 || u >= width || v < 0 || v >= height)
						continue;
					float dx = u-x, dy = v-y;
					if (v < y) {
						if (sqrt(dx*dx + dy*dy) <= dr)
							ppmdata[v][u].b = 255;
					} else {
						if (sqrt(dx*dx + dy*dy) <= r) {
							if (r < dr-1) {
								ppmdata[v][u].r = 255;
							} else if (r > dr+1) {
								ppmdata[v][u].g = 255;
							} else {
								ppmdata[v][u].r = 255;
								ppmdata[v][u].g = 255;
							}
						}
					}
				}
			}
			ppm_writeppm(fp, ppmdata, width, height, 255, 0);
			ppm_freearray(ppmdata, width);
			pm_close(fp);
			print_time("pixmap output");
		}

		if (output_filename)
		{
			// write output gcode
			FILE *out = !strcmp(output_filename, "-") ? stdin : fopen(output_filename, "w");
			fprintf(out, "#1 = %10.5f  (x offset)\n", offset_x);
			fprintf(out, "#2 = %10.5f  (y offset)\n", offset_y);
			fprintf(out, "#3 = %10.5f  (z for safe height)\n", safe_z);
			fprintf(out, "#4 = %10.5f  (z for cutting per iteration)\n", -cut_z/cnc_iterations);
			fprintf(out, "#5 = %10.5f  (feed rate)\n", feed_rate);
			fprintf(out, "G90 (abosulte positioning)\n");
			fprintf(out, "G21 (use mm as unit of length)\n");
			if (gcode_preamble)
				fprintf(out, "%s\n", gcode_preamble);
			fprintf(out, "G00 Z#3\n");
			float scale = 25.4 / resolution_dpi;
			for (int i=2; i<area_count; i++) {
				float x = drillmap[i].x;
				float y = drillmap[i].y;
				float r = drillmap[i].r;
				if (r <= dr + 0.1) {
					fprintf(out, "G00 X[%f+#1] Y[%f+#2]\n", x*scale, ym(y)*scale);
					fprintf(out, "G01 Z[#4*%d] F#5\n", cnc_iterations);
				} else {
					fprintf(out, "G00 X[%f+#1] Y[%f+#2]\n", x*scale, ym(y)*scale);
					for (int j=0; j < cnc_iterations; j++) {
						fprintf(out, "G01 Z[#4*%d] F#5\n", j+1);
						for (float rr = 2*dr; rr < r; rr += dr) {
							fprintf(out, "G01 X[%f+#1] Y[%f+#2]\n", (x-(rr-dr))*scale, ym(y)*scale);
							fprintf(out, "G03 X[%f+#1] Y[%f+#2] I%f\n", (x-(rr-dr))*scale, ym(y)*scale, (rr-dr)*scale);
						}
						fprintf(out, "G01 X[%f+#1] Y[%f+#2]\n", (x-(r-dr))*scale, ym(y)*scale);
						fprintf(out, "G03 X[%f+#1] Y[%f+#2] I%f\n", (x-(r-dr))*scale, ym(y)*scale, (r-dr)*scale);
						fprintf(out, "G01 X[%f+#1] Y[%f+#2]\n", x*scale, ym(y)*scale);
					}
				}
				fprintf(out, "G00 Z#3\n");
			}
			if (gcode_postamble)
				fprintf(out, "%s\n", gcode_postamble);
			fprintf(out, "M2\n");
			if (strcmp(output_filename, "-"))
				fclose(out);
			print_time("gcode output");
		}
		goto free_and_exit;
	}

	// queue all trace pixels
	for (int x=1; x<width-1; x++)
	for (int y=1; y<height-1; y++) {
		if (P(x,y).area_nr > 1) {
			P(x,y).area_nr |= PIXEL_QUEUED_FLAG;
			queue_push(y*width + x);
		}
	}
	print_time("init queue for flood fill");

	// flood fill the gaps
	while (queue_begin != queue_end)
	{
		int idx = queue_shift();
		int level = pixmap[idx].level;
		pixmap[idx].area_nr &= ~PIXEL_QUEUED_FLAG;
		for (int i=0; i<8; i++) {
			float distance = ntab_distance[i];
			int idx2 = idx+ntab[i];
			if (is_on_border(idx2))
				continue;
			if (pixmap[idx2].level > level+distance) {
				pixmap[idx2].level = level+distance;
				pixmap[idx2].area_nr = pixmap[idx].area_nr | (pixmap[idx2].area_nr & PIXEL_QUEUED_FLAG);
				if ((pixmap[idx2].area_nr & PIXEL_QUEUED_FLAG) == 0) {
					pixmap[idx2].area_nr |= PIXEL_QUEUED_FLAG;
					queue_push(idx2);
				}
			}
		}
	}
	print_time("flood fill");

	// create initial nodes and eges
	for (int x=2; x<width-2; x++)
	for (int y=2; y<height-2; y++) {
		if (P(x,y).area_nr != P(x+1,y).area_nr)
			add_edge(x+1, y, x+1, y+1);
		if (P(x,y).area_nr != P(x,y+1).area_nr)
			add_edge(x, y+1, x+1, y+1);
	}
	print_time("create initial nodes");

	// reset levels for reverse leveling
	for (int x=0; x<width; x++)
	for (int y=0; y<height; y++) {
		if (P(x,y).level == 0)
			P(x,y).level = width + height;
		else
			P(x,y).level = width + height - 1;
		if (P(x,y).node != -1) {
			int idx = y*width + x;
			pixmap[idx].level = 0;
			pixmap[idx].area_nr |= PIXEL_QUEUED_FLAG;
			queue_push(idx);
			for (int i=0; i<3; i++) {
				pixmap[idx+ntab[i]].level = 0;
				pixmap[idx+ntab[i]].area_nr |= PIXEL_QUEUED_FLAG;
				queue_push(idx+ntab[i]);
			}
		}
		if (x == 0 || y == 0 || x == width-1 || y == height-1) {
			int idx = y*width + x;
			pixmap[idx].level = 0;
			pixmap[idx].area_nr |= PIXEL_QUEUED_FLAG;
			queue_push(idx);
		}
	}
	print_time("prep for reverse fill");

	// reverse fill
	while (queue_begin != queue_end)
	{
		int idx = queue_shift();
		pixmap[idx].area_nr &= ~PIXEL_QUEUED_FLAG;
		int level = pixmap[idx].level;
		for (int i=0; i<8; i++) {
			int idx2 = idx+ntab[i];
			if (is_on_border(idx2))
				continue;
			if (pixmap[idx2].level == width + height)
				continue;
			if (pixmap[idx2].level > level+ntab_distance[i]) {
				pixmap[idx2].level = level+ntab_distance[i];
				if ((pixmap[idx2].area_nr & PIXEL_QUEUED_FLAG) == 0) {
					pixmap[idx2].area_nr |= PIXEL_QUEUED_FLAG;
					queue_push(idx2);
				}
			}
		}
	}
	print_time("reverse fill");

	// queue all nodes with two edges
	queue_begin = queue_end = 0;
	for (int x=2; x<width-2; x++)
	for (int y=2; y<height-2; y++) {
		if (P(x,y).node != -1 && nodes[P(x,y).node].edges_nr == 2)
			queue_push(P(x,y).node);
	}
	print_time("queue nodes with two edges");

	// sort queue so node with much freedom are processed first
	qsort(queue, queue_end, sizeof(int), compar_node_queue);
	print_time("sorting all nodes");

	// remove nodes where possible
	while (queue_begin != queue_end)
	{
		int idx = queue_shift();
		if (nodes[idx].edges_nr == 0)
			continue;
		if (nodes[idx].edges_nr != 2) {
			fprintf(stderr, "How can a node with != 2 edges end up queued?\n");
			abort();
		}
		// don't touch the nodes on the border
		int x = nodes[idx].x;
		int y = nodes[idx].y;
		if (x <= 1 || x >= width-1)
			continue;
		if (y <= 1 || y >= height-1)
			continue;
		// remove unless if makes things bad
		int n1 = nodes[idx].edges[0];
		int n2 = nodes[idx].edges[1];
		int x1 = nodes[n1].x;
		int y1 = nodes[n1].y;
		int x2 = nodes[n2].x;
		int y2 = nodes[n2].y;
		// FIXME: We should check the triangle defined by all three nodes instead..
		if (level_line(x1, y1, x2, y2) < nodes[idx].level)
			remove_node(idx);
	}
	print_time("removing nodes");

	if (pixmap_filename)
	{
		// write output pixmap
		fp = pm_openw(pixmap_filename);
		ppmdata = ppm_allocarray(width, height);
		for (int x=0; x<width; x++)
		for (int y=0; y<height; y++) {
			int area_nr = P(x,y).area_nr;
			if (area_nr != 0 && P(x,y).level != width+height) {
				if (!mode_alt_png_output) {
					ppmdata[y][x].r = 32 + ((area_nr) % 5) * 20;
					ppmdata[y][x].g = 32 + ((area_nr/5) % 5) * 20;
					ppmdata[y][x].b = 32 + ((area_nr/25) % 5) * 20;
				} else {
					int gray = P(x,y).level > 100 ? 200 : 2*P(x,y).level;
					ppmdata[y][x].r = gray < 10 ? 100 : gray;
					ppmdata[y][x].g = gray;
					ppmdata[y][x].b = gray;
				}
			} else {
				if (!mode_alt_png_output) {
					ppmdata[y][x].r = 0;
					ppmdata[y][x].g = 0;
					ppmdata[y][x].b = 0;
				} else {
					ppmdata[y][x].r = 110;
					ppmdata[y][x].g = 90;
					ppmdata[y][x].b = 30;
				}
			}
		}
		for (int x=0; x<width; x++)
		for (int y=0; y<height; y++) {
			int n = P(x,y).node;
			if (n != -1) {
				for (int i=0; i<nodes[n].edges_nr; i++) {
					int e = nodes[n].edges[i];
					int x1 = nodes[n].x, y1 = nodes[n].y;
					int x2 = nodes[e].x, y2 = nodes[e].y;
					ppm_line(ppmdata, x1, y1, x2, y2);
					ppmdata[y1][x1].r = 255;
					ppmdata[y1][x1].g = 255;
					ppmdata[y1][x1].b = 255;
					ppmdata[y2][x2].r = 255;
					ppmdata[y2][x2].g = 255;
					ppmdata[y2][x2].b = 255;
				}
			}
		}
		ppm_writeppm(fp, ppmdata, width, height, 255, 0);
		ppm_freearray(ppmdata, width);
		pm_close(fp);
		print_time("pixmap output");
	}

	if (output_filename)
	{
		// write output gcode
		FILE *out = !strcmp(output_filename, "-") ? stdin : fopen(output_filename, "w");
		fprintf(out, "#1 = %10.5f  (x offset)\n", offset_x);
		fprintf(out, "#2 = %10.5f  (y offset)\n", offset_y);
		fprintf(out, "#3 = %10.5f  (z for safe height)\n", safe_z);
		fprintf(out, "#4 = %10.5f  (z for cutting for 1st iteration)\n", -cut_z / cnc_iterations);
		fprintf(out, "#5 = %10.5f  (feed rate)\n", feed_rate);
		fprintf(out, "G90 (abosulte positioning)\n");
		fprintf(out, "G21 (use mm as unit of length)\n");
		if (gcode_preamble)
			fprintf(out, "%s\n", gcode_preamble);
		fprintf(out, "G00 Z#3\n");
		if (cnc_iterations > 1)
			fprintf(out, "O100 repeat [%d]\n", cnc_iterations);
		int cnc_x = 0, cnc_y = 0;
		float last_vec_x = 0, last_vec_y = 0;
		float scale = 25.4 / resolution_dpi;
		while (1) {
			int n1 = -1;
			float n1_distance = 0;
			for (int i=0; i<nodes_nr; i++) {
				if (nodes[i].edges_nr != 0) {
					float this_distance = sqrt(pow(cnc_x-nodes[i].x, 2) + pow(cnc_y-nodes[i].y, 2));
					if (n1 == -1)
						goto this_node_is_better;
					switch (nodes[n1].edges_nr) {
					case 2:
						if (nodes[i].edges_nr != 2)
							goto this_node_is_better;
						break;
					case 4:
						if (nodes[i].edges_nr == 2)
							goto this_node_is_not_better;
						if (nodes[i].edges_nr != 4)
							goto this_node_is_better;
						break;
					case 3:
						if (cnc_x == 0 && cnc_y == 0 && nodes[i].edges_nr == 1)
							goto this_node_is_better;
						if (0) {
					case 1:
							if (cnc_x == 0 && cnc_y == 0 && nodes[i].edges_nr == 3)
								goto this_node_is_not_better;
						}
						if (nodes[i].edges_nr == 2)
							goto this_node_is_not_better;
						if (nodes[i].edges_nr == 4)
							goto this_node_is_not_better;
						break;
					}
					if (this_distance < n1_distance)
						goto this_node_is_better;
this_node_is_not_better:
					if (0) {
this_node_is_better:
						n1_distance = this_distance;
						n1 = i;
					}
				}
			}
			if (n1 == -1)
				break;
			fprintf(out, "G00 X[%f+#1] Y[%f+#2]\n", (cnc_x = nodes[n1].x)*scale, ym(cnc_y = nodes[n1].y)*scale);
			fprintf(out, "G01 Z#4 F#5\n");
			while (nodes[n1].edges_nr > 0) {
				int n2 = -1;
				float best_angle = 10;
				for (int i=0; i<nodes[n1].edges_nr; i++) {
					int this_n2 = nodes[n1].edges[i];
					float this_angle = angle_two_vectors(last_vec_x, last_vec_y,
							nodes[this_n2].x - nodes[n1].x, nodes[this_n2].y - nodes[n1].y);
					if (this_angle < best_angle) {
						best_angle = this_angle;
						n2 = this_n2;
					}
				}
				if (n2 < 0) {
					fprintf(stderr, "Only found edges with impossible angles!\n");
					abort();
				}
				for (int i=0; i<nodes[n1].edges_nr-1; i++) {
					if (n2 == nodes[n1].edges[i]) {
						nodes[n1].edges[i] = nodes[n1].edges[i+1];
						nodes[n1].edges[i+1] = n2;
					}
				}
				for (int i=0; i<nodes[n2].edges_nr-1; i++) {
					if (n1 == nodes[n2].edges[i]) {
						nodes[n2].edges[i] = nodes[n2].edges[i+1];
						nodes[n2].edges[i+1] = n1;
					}
				}
				nodes[n1].edges_nr--;
				nodes[n2].edges_nr--;
				last_vec_x = nodes[n2].x - nodes[n1].x;
				last_vec_y = nodes[n2].y - nodes[n1].y;
				fprintf(out, "G01 X[%f+#1] Y[%f+#2]\n", (cnc_x = nodes[n2].x)*scale, ym(cnc_y = nodes[n2].y)*scale);
				n1 = n2;
			}
			fprintf(out, "G00 Z#3\n");
		}
		if (cnc_iterations > 1) {
			fprintf(out, "#4 = [#4 - %.5f]  (increment cutting depth)\n", cut_z / cnc_iterations);
			fprintf(out, "O100 endrepeat\n");
		}
		if (gcode_postamble)
			fprintf(out, "%s\n", gcode_postamble);
		fprintf(out, "M2\n");
		if (strcmp(output_filename, "-"))
			fclose(out);
		print_time("gcode output");
	}

	// free memory
free_and_exit:
	free(pixmap);
	free(queue);
	free(nodes);
	print_time("free memory");

	return 0;
}

