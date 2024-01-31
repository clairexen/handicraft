#include <stdio.h>
#include <list>

template<typename T> typename T::value_type pop(T &stack)
{
	auto value = stack.back();
	stack.pop_back();
	return value;
}

int main()
{
	std::list<int> stack;
	stack.push_back(1);
	stack.push_back(2);
	stack.push_back(3);
	printf("%d\n", pop(stack));
	printf("%d\n", pop(stack));
	printf("%d\n", pop(stack));
	return 0;
}

