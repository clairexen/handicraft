#include <vector>
extern std::vector<int> extern_int_vec;

#ifdef ASAN_INIT_ORDER_EXPECT_FAIL
std::vector<int> extern_int_vec{1,2,3};
#endif

#ifdef ASAN_INIT_ORDER_EXPECT_PASS
std::vector<int> extern_int_vec;
#endif

#ifdef ASAN_INIT_ORDER_CONSTINIT
constinit std::vector<int> extern_int_vec;
#endif
