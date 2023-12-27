#include <vector>
#include <iostream>
extern std::vector<int> extern_int_vec;
int len_of_extern_int_vec = extern_int_vec.size();
int main() {
	std::cout << "len=" << len_of_extern_int_vec << std::endl;
	return 0;
}
