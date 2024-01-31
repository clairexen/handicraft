/*
 *  Simple OpenGL demo program with shaders
 *
 *  Copyright (C) 2009  Clifford Wolf <clifford@clifford.at>
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
 *  Compile:
 *  g++ -std=gnu++0x -o mytetview -lSDL -lGLEW -lm mytetview.cc
 */

// this must be included before any OpenGL headers
#include <GL/glew.h>

#include <SDL/SDL.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include <map>
#include <list>

/**** BEGIN: http://svn.clifford.at/tools/trunk/examples/check.h ****/

// This is to not confuse the VIM syntax highlighting
#define CHECK_VAL_OPEN (
#define CHECK_VAL_CLOSE )

#define CHECK(result, check)                                           \
   CHECK_VAL_OPEN{                                                     \
     typeof(result) _R = (result);                                     \
     if (!(_R check)) {                                                \
       fprintf(stderr, "Error from '%s' (%ld %s) in %s:%d.\n",         \
                  #result, (long int)_R, #check, __FILE__, __LINE__);  \
       fprintf(stderr, "ERRNO(%d): %s\n", errno, strerror(errno));     \
       abort();                                                        \
     }                                                                 \
     _R;                                                               \
   }CHECK_VAL_CLOSE

/**** END: http://svn.clifford.at/tools/trunk/examples/check.h ****/

SDL_Surface *screen;
double theta = 1, phi = 1;

void checkGlError(const char *file, int line)
{
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		fprintf(stderr, "OpenGL Error at %s:%d: %s\n", file, line, gluErrorString(err));
	}
}

#define CHECKGL(...)                             \
   do {                                          \
     checkGlError("(pre) " __FILE__, __LINE__);  \
     __VA_ARGS__;                                \
     checkGlError(__FILE__, __LINE__);           \
   } while (0)

#define CHECKGL_V(result)                        \
   CHECK_VAL_OPEN{                               \
     checkGlError("(pre) " __FILE__, __LINE__);  \
     typeof(result) _R = result;                 \
     checkGlError(__FILE__, __LINE__);           \
     _R;                                         \
   }CHECK_VAL_CLOSE

struct Point {
	double x, y, z;
	Point() : x(0), y(0), z(0) { }
	Point(double x, double y, double z) : x(x), y(y), z(z) { }
};
struct Face {
	int p1, p2, p3;
	Face(int p1, int p2, int p3) : p1(p1), p2(p2), p3(p3) { }
};

std::map<int, Point> points;
std::list<Face> faces;

void draw()
{
	glUseProgram(0);

	glEnable(GL_DEPTH_TEST);
	glDepthRange(-100, 100);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-1.0, +1.0, -1.0, +1.0, +1.5, +20.0);
	gluLookAt(5.0 * cos(theta) * cos(phi), 5.0 * sin(theta), 5.0 * cos(theta) * sin(phi), 0.0, 0.0, 0.0,
			5.0 * cos(theta+1) * cos(phi), 5.0 * sin(theta+1), 5.0 * cos(theta+1) * sin(phi));

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glColor3f(1.0, 1.0, 1.0);
	for (auto i=faces.begin(); i != faces.end(); i++)
	{
		glBegin(GL_LINE_LOOP);
		glVertex3d(points[i->p1].x, points[i->p1].y, points[i->p1].z);
		glVertex3d(points[i->p2].x, points[i->p2].y, points[i->p2].z);
		glVertex3d(points[i->p3].x, points[i->p3].y, points[i->p3].z);
		glEnd();
	}

	CHECKGL();
	SDL_GL_SwapBuffers();
}

void read_node_file(const char *fn)
{
	int count, dim, attrcount, boundmark;

	FILE *f = CHECK(fopen(fn, "r"), != NULL);

	CHECK(fscanf(f, "%d %d %d %d", &count, &dim, &attrcount, &boundmark), == 4);
	CHECK(dim, == 3);
	CHECK(attrcount, == 0);
	CHECK(boundmark, == 0);

	for (int i=0; i<count; i++) {
		int idx;
		double x, y, z;
		CHECK(fscanf(f, "%d %lf %lf %lf", &idx, &x, &y, &z), == 4);
		points[idx] = Point(x, y, z);
	}
	
	fclose(f);

	printf("Total number of points: %d\n", count);
}

void read_face_file(const char *fn)
{
	int count, boundmark;

	FILE *f = CHECK(fopen(fn, "r"), != NULL);

	CHECK(fscanf(f, "%d %d", &count, &boundmark), == 2);

	for (int i=0; i<count; i++) {
		int idx, p1, p2, p3;
		CHECK(fscanf(f, "%d %d %d %d", &idx, &p1, &p2, &p3), == 4);
		for (int j=0; j<boundmark; j++)
			 fscanf(f, "%*f");
		faces.push_back(Face(p1, p2, p3));
	}
	
	fclose(f);
}

void read_ele_file(const char *fn)
{
	int count, nodespertet, regattr;

	FILE *f = CHECK(fopen(fn, "r"), != NULL);

	CHECK(fscanf(f, "%d %d %d", &count, &nodespertet, &regattr), == 3);
	CHECK(nodespertet, == 4);
	CHECK(regattr, == 0);

	int edgecount;
	std::map<std::pair<int,int>, int> edgemap;

	for (int i=0; i<count; i++)
	{
		int idx, p1, p2, p3, p4;
		CHECK(fscanf(f, "%d %d %d %d %d", &idx, &p1, &p2, &p3, &p4), == 5);

		faces.push_back(Face(p2, p3, p4));
		faces.push_back(Face(p1, p3, p4));
		faces.push_back(Face(p1, p2, p4));
		faces.push_back(Face(p1, p2, p3));

		// keep it simple and stupid - count unique edges
		if (edgemap.count(std::pair<int,int>(p1, p2)) == 0) edgecount++;
		if (edgemap.count(std::pair<int,int>(p1, p3)) == 0) edgecount++;
		if (edgemap.count(std::pair<int,int>(p1, p4)) == 0) edgecount++;
		if (edgemap.count(std::pair<int,int>(p2, p3)) == 0) edgecount++;
		if (edgemap.count(std::pair<int,int>(p2, p4)) == 0) edgecount++;
		if (edgemap.count(std::pair<int,int>(p3, p4)) == 0) edgecount++;

		// mark the edges
		edgemap[std::pair<int,int>(p1, p2)] = 1;
		edgemap[std::pair<int,int>(p1, p3)] = 1;
		edgemap[std::pair<int,int>(p1, p4)] = 1;
		edgemap[std::pair<int,int>(p2, p3)] = 1;
		edgemap[std::pair<int,int>(p2, p4)] = 1;
		edgemap[std::pair<int,int>(p3, p4)] = 1;

		// also mark the counter-edges
		edgemap[std::pair<int,int>(p2, p1)] = 1;
		edgemap[std::pair<int,int>(p3, p1)] = 1;
		edgemap[std::pair<int,int>(p4, p1)] = 1;
		edgemap[std::pair<int,int>(p3, p2)] = 1;
		edgemap[std::pair<int,int>(p4, p2)] = 1;
		edgemap[std::pair<int,int>(p4, p3)] = 1;
	}
	
	fclose(f);

	printf("Total number of edges: %d\n", edgecount);
}

int main(int argc, char **argv)
{
	CHECK(SDL_Init(SDL_INIT_VIDEO), >= 0);
	SDL_WM_SetCaption("MyTetView", "MyTetView");
	atexit(SDL_Quit);

	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

	screen = CHECK(SDL_SetVideoMode(640, 480, 0, SDL_OPENGL), != NULL);

	GLenum glew_err = glewInit();
	if (glew_err != GLEW_OK) {
		fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(glew_err));
		return 1;
	}

	if (!glewIsSupported("GL_VERSION_2_0")) {
		fprintf(stderr, "Accoring to GLEW there is no OpenGL 2.0 support!\n");
		return 1;
	}

	if (argc == 4 && !strcmp(argv[1], "F"))
	{
	
		read_node_file(argv[2]);
		read_face_file(argv[3]);
	}
	else
	if (argc == 4 && !strcmp(argv[1], "E"))
	{
	
		read_node_file(argv[2]);
		read_ele_file(argv[3]);
	}
	else
	{
		points[0] = Point(0, 0, 0);
		points[1] = Point(1, 0, 0);
		points[2] = Point(0, 1, 0);
		points[3] = Point(0, 0, 1);

		faces.push_back(Face(0, 1, 2));
		faces.push_back(Face(1, 2, 3));
	}


	draw();

	while (1)
	{
		SDL_Event event;

		if (!SDL_PollEvent(&event)) {
			draw();
			CHECK(SDL_WaitEvent(&event), != 0);
		}

		if (event.type == SDL_QUIT)
			break;

		if (event.type == SDL_MOUSEMOTION && (event.motion.state & SDL_BUTTON(1)) != 0) {
			theta += event.motion.yrel / 100.0;
			phi += event.motion.xrel / 100.0;
		}
	}

	return 0;
}

