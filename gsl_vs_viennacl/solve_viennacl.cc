#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/direct_solve.hpp"

#include "profiler.h"
#include "challenge.h"

struct MatrixWrapper
{
	double (&mat)[DIM][DIM];
	MatrixWrapper(double (&mat)[DIM][DIM]) : mat(mat) { }
	double &operator()(int i, int j) { return mat[i][j]; }
	const double &operator()(int i, int j) const { return mat[i][j]; }
	size_t size1() const { return DIM; }
	size_t size2() const { return DIM; }
};

typedef double ScalarType;

void info_viennacl()
{
	Profiler("viennacl_info") {
		std::vector<viennacl::ocl::device> devices = viennacl::ocl::current_context().devices();
		for (size_t i = 0; i < devices.size(); i++) {
			viennacl::ocl::current_context().switch_device(devices[i]);
			printf("ViennaCL device #%zd: %s\n", i, viennacl::ocl::current_device().name().c_str());
		}
	}
}

void solve_viennacl(challenge_s &cl)
{
	Profiler::begin("viennacl_ctor");
	viennacl::matrix<ScalarType> vcl_A[NUM];
	viennacl::vector<ScalarType> vcl_bx[NUM];
	Profiler::end("viennacl_ctor");

	Profiler("viennacl_copy_matrix") {
		for (int n = 0; n < NUM; n++) {
			vcl_A[n] = viennacl::matrix<ScalarType>(DIM, DIM);
			viennacl::copy(MatrixWrapper(cl.A[n]), vcl_A[n]);
		}
	}

	Profiler("viennacl_copy_vect1") {
		std::vector<double> cpu_vect(DIM);
		for (int n = 0; n < NUM; n++) {
			for (int i = 0; i < DIM; i++)
				cpu_vect[i] = cl.b[n][i];
			vcl_bx[n] = viennacl::vector<ScalarType>(DIM);
			viennacl::fast_copy(cpu_vect, vcl_bx[n]);
		}
	}

	Profiler("viennacl_solve") {
		for (int n = 0; n < NUM; n++) {
			viennacl::linalg::lu_factorize(vcl_A[n]);
			viennacl::linalg::lu_substitute(vcl_A[n], vcl_bx[n]);
		}
	}

	Profiler("viennacl_copy_vect2") {
		std::vector<double> cpu_vect(DIM);
		for (int n = 0; n < NUM; n++) {
			// this call blocks using sched_yield() until results are available..
			viennacl::fast_copy(vcl_bx[n], cpu_vect);
			for (int i = 0; i < DIM; i++)
				cl.x_viennacl[n][i] = cpu_vect[i];
		}
	}
}

