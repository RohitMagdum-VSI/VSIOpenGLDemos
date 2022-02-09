#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef int (*PFN_ADD)(int, int);
int main()
{
	PFN_ADD pfnAdd = NULL;
	void *module;
	
	//	Load module.
	module = dlopen("libmy_math.so", RTLD_LAZY);
	if (NULL == module)
	{
		printf("\n module load failed.");
		return 0;
	}
	
	//	Get function address.
	pfnAdd = dlsym(module, "add");
	if (NULL == pfnAdd)
	{
		printf("\n Function not found.");
		return 0;
	}
	
	//	Call function.
	printf("\n Addition is : %d ", pfnAdd(10, 15));
	
	//	Unload module.
	dlclose(module);
	
	return 0;
}
