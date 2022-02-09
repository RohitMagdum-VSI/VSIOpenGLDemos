#include <stdio.h>
#include <stdlib.h>

struct rgba
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
	unsigned char a;
};

union COLOR_VAL  
{
	unsigned int val;
	struct rgba components;
	
};

//
//	Here I can Pass RGBA values to any function as a int
//	and later typecast to rgba and use. 
//
int main()
{
	void PrintPresentColors(unsigned int i);
	
	COLOR_VAL ColorVal;
	
	ColorVal.components.r = 1;
	ColorVal.components.g = 1;	
	ColorVal.components.b = 0;	
	ColorVal.components.a = 1;
	
	PrintPresentColors(ColorVal.val);
}

void PrintPresentColors(unsigned int i)
{
	COLOR_VAL ColorVal;
	
	ColorVal.val = i;
	
	printf("\n Red - %d \n Green - %d \n Black - %d \n Alpha - %d", ColorVal.components.r, ColorVal.components.g, ColorVal.components.b, 			ColorVal.components.a);
}

