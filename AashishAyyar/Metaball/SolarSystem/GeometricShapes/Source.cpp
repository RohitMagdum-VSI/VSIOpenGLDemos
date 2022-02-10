#include "Common.h"

FILE *gpFile = NULL;

BOOL WINAPI DllMain(HINSTANCE hInstance, DWORD dwReason, LPVOID Reserved)	 
{
	switch (dwReason) 
	{
	case DLL_PROCESS_ATTACH:

		if (gpFile == NULL)
		{
			if (fopen_s(&gpFile, "Log1.txt", "w") != 0)
			{
				MessageBox(NULL, TEXT("Log File can not be Created\nExitting...."), TEXT("Error"), MB_OK | MB_TOPMOST | MB_ICONSTOP);
				exit(0);
			}
			fprintf(gpFile, "Log File is successfully opened. \n");
		}
		break;
	case DLL_THREAD_ATTACH:
		break;
	case DLL_THREAD_DETACH:
		break;
	case DLL_PROCESS_DETACH:
		if (gpFile) 
		{
			fclose(gpFile);
		}
		break;
	}

	return TRUE;
}

