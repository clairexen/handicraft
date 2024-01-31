// see http://www.youtube.com/watch?v=d8TRcZklX_Q
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <assert.h>

int sortNumAsc(int n)
{
	int v = 0;
	std::vector<int> digits;
	for (size_t i = 0; i < 4; i++)
		digits.push_back(n % 10), n /= 10;
	std::sort(digits.begin(), digits.end());
	for (size_t i = 0; i < digits.size(); i++)
		v = v * 10 + digits[i];
	return v;
}

int sortNumDesc(int n)
{
	int v = 0;
	std::vector<int> digits;
	for (size_t i = 0; i < 4; i++)
		digits.push_back(n % 10), n /= 10;
	std::sort(digits.begin(), digits.end());
	for (size_t i = 1; i <= digits.size(); i++)
		v = v * 10 + digits[digits.size()-i];
	return v;
}

int try6174(int n, bool verbose)
{
	int count = 0;
	while (n != 6174) {
		int a = sortNumAsc(n);
		int b = sortNumDesc(n);
		if (verbose)
			printf("[%2d] %4d => %4d - %4d = %4d\n",
					count + 1, n, b, a, b - a);
		n = b - a;
		if (n < 100) {
			assert(count == 0 && n == 0);
			return 0;
		}
		count++;
	}
	return count;
}

int main()
{
	int max_n = 1000, max_it = 0, j = 0;
	for (int i = 0; i <= 9999; i++) {
		int it = try6174(i, false);
		if (it > max_it)
			max_n = i, max_it = it;
		if (it == 7)
			printf("%s%4d", j == 0 ? "" : j % 10 == 0 ? "\n" : " ", i), j++;
	}
	assert(max_it == 7);
	printf("\ntotal number of start values that result in 7 iterations: %d\n", j);
	try6174(max_n, true);
	return 0;
}

