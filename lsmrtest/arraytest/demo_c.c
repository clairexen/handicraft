#include <stdio.h>

#define func0 func0_
#define func1 func1_
#define func2 func2_

extern void func0();
extern void func1(int *n, void (*CallBack)(int *n, double *v));
extern void func2(int *n, double *v);

void func3(int *n, double *v)
{
	int i;
	for (i = 0; i < *n; i++)
		printf("%d: %f\n", i, v[i]);
}

int main()
{
	int n = 5;
	func0();
	func1(&n, &func2);
	func1(&n, &func3);
	return 0;
}

