#include <stdio.h>

typedef struct _POINT
{
	float x;
	float y;
	float z;

}POINT;

// range of z should be (p1.z to p2.z)
POINT GetPointOnLine(POINT p1, POINT p2, float z);

int main()
{
	float increment = 0.0f;
	POINT p1 = { 1, 2, 3 };
	POINT p2 = { 7, 8, -9 };

	POINT p = GetPointOnLine(p1, p2, -8);

	printf("x: %f\t, y: %f\t, z: %f\n", p.x, p.y, p.z);

	return 0;
}

POINT GetPointOnLine(POINT p1, POINT p2, float z)
{
	POINT p = { 0 };

	p.x = (((p2.x - p1.x) / (p2.z - p1.z)) * (z - p1.z)) + p1.x;
	p.y = (((p2.y - p1.y) / (p2.z - p1.z)) * (z - p1.z)) + p1.y;
	p.z = z;

	return p;
}