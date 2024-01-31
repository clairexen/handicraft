#include <emscripten/val.h>
#include <emscripten/bind.h>

#include <stdio.h>
#include <string>

using emscripten::val;
using emscripten::vecFromJSArray;
using std::string;

// Reference for emscripten::val class:
// https://kripken.github.io/emscripten-site/docs/porting/connecting_cpp_and_javascript/embind.html#embind-val-guide
// https://kripken.github.io/emscripten-site/docs/api_reference/val.h.html

extern "C"
void example_cfun(val request)
{
	// fetch function name from request object
	auto funname = request["function"].as<string>();

	printf("calling example_cfun '%s'\n", funname.c_str());

	if (funname == "sum_prod")
	{
		// read arguments
		auto numbers = vecFromJSArray<int>(request["numbers"]);

		// caclulation
		int sum = 0, prod = 1;
		for (int v : numbers) {
			sum += v;
			prod *= v;
		}

		// create results array
		auto results = val::array();
		results.set(0, sum);
		results.set(1, prod);

		// add results to request object
		request.set("results", results);
	}
}

EMSCRIPTEN_BINDINGS(my_module) {
	function("example_cfun", &example_cfun);
}

int main()
{
	printf("running main\n");
	return 0;
}

