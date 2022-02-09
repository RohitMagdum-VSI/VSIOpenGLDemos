#include<windows.h>
#include<stdio.h>

#pragma comment(lib,"User32.lib")
#pragma comment(lib,"GDI32.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

FILE *gpFile;
DEVMODE gDevMode = {sizeof(DEVMODE)};
HWND ghwnd;
LONG glResult;
DWORD dwStyle;
bool gbFullscreen_CDS = false;
bool gbFullscreen_Flag = false;

struct Resolution
{
	DWORD MyWidth;
	DWORD MyHeight;
}MaxRes;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	void uninitialize(void);
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("My App");

	if (fopen_s(&gpFile, "Log.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("CANNOT CREATE FILE"), TEXT("ERROR"), MB_OK);
		exit(0);
	}
	else
		fprintf(gpFile, "Log File Is Created...\n");

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_VREDRAW | CS_HREDRAW;
	wndclass.cbClsExtra = NULL;
	wndclass.cbWndExtra = NULL;
	wndclass.lpszClassName = szClassName;
	wndclass.lpszMenuName = NULL;
	wndclass.hInstance = hInstance;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);

	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, TEXT("Window With Change Display Settings"), WS_OVERLAPPEDWINDOW,CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,  NULL, NULL, hInstance, NULL);
	if (hwnd == NULL)
	{
		fprintf(gpFile, "Cannot Create Window");
		exit(0);
	}
	ghwnd = hwnd;

	ShowWindow(hwnd, iCmdShow);
	UpdateWindow(hwnd);

	while (GetMessage(&msg, NULL, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	fprintf(gpFile, "Log File Is Successfully Closed");
	fclose(gpFile);
	uninitialize();
	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	void ToggleFullscreen(void);
	int iCnt;
	switch (iMsg)
	{
	case WM_CREATE:
		iCnt = 0;
		while (EnumDisplaySettings(NULL, iCnt, &gDevMode))
		{
			fprintf(gpFile, "WIDTH:%d HEIGHT:%d\n", gDevMode.dmPelsWidth, gDevMode.dmPelsHeight);
			if (iCnt == 0)
			{
				MaxRes.MyWidth = GetSystemMetrics(SM_CXSCREEN);
				MaxRes.MyHeight = GetSystemMetrics(SM_CYSCREEN);;
			}
			else
			{
				if (gDevMode.dmPelsWidth >= MaxRes.MyWidth)
				{
					MaxRes.MyWidth = gDevMode.dmPelsWidth;
					if (gDevMode.dmPelsHeight > MaxRes.MyHeight)
					{
						MaxRes.MyHeight = gDevMode.dmPelsHeight;
					}
				}
			}
			//iCnt += 3;
		}
		break;
	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;
		case 0x46:
			if (gbFullscreen_CDS == false)
			{
				ToggleFullscreen();
				gbFullscreen_CDS = true;
			}
			else
			{
				ToggleFullscreen();
				gbFullscreen_CDS = false;
			}
			break;
		/*case 0x51:
			if (gbFullscreen_Flag == false)
			{
				SetWindowLong(ghwnd, GWL_STYLE, WS_OVERLAPPEDWINDOW);
				gbFullscreen_Flag = true;
			}
			else
			{
				dwStyle=GetWindowLong(ghwnd, GWL_STYLE);
				SetWindowLong(ghwnd, GWL_STYLE, dwStyle | ~WS_OVERLAPPEDWINDOW | WS_POPUP);
				gbFullscreen_Flag = false;
			}
			break;*/
		}
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullscreen(void)
{
	dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
	if (gbFullscreen_CDS == false)
	{
		EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &gDevMode);
		fprintf(gpFile, "Resolution in else ChangeDisplaySettings() Width: %d, Height %d\n", gDevMode.dmPelsWidth, gDevMode.dmPelsHeight);
		gDevMode.dmPelsWidth = MaxRes.MyWidth;
		gDevMode.dmPelsHeight = MaxRes.MyHeight;
		gDevMode.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT;
		ChangeDisplaySettings(&gDevMode, CDS_RESET | CDS_FULLSCREEN);
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW|WS_POPUP);
		ShowWindow(ghwnd, SW_SHOWMAXIMIZED);
	}
	else
	{
		EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &gDevMode);
		fprintf(gpFile, "Resolution in if ChangeDisplaySettings() Width: %d, Height %d\n", gDevMode.dmPelsWidth, gDevMode.dmPelsHeight);
		gDevMode.dmSize = sizeof(DEVMODE);
		gDevMode.dmPelsWidth = 800;
		gDevMode.dmPelsHeight = 600;
		gDevMode.dmFields = DM_PELSHEIGHT | DM_PELSWIDTH;
		glResult = ChangeDisplaySettings(&gDevMode, CDS_RESET | CDS_FULLSCREEN);
		if (glResult == DISP_CHANGE_SUCCESSFUL)
		{
			fprintf(gpFile, "Change Display Setting Successful\n");
		}
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW|WS_POPUP);
		ShowWindow(ghwnd, SW_SHOWMAXIMIZED);
	}
	UpdateWindow(ghwnd);
}

void uninitialize(void)
{
	if (gbFullscreen_CDS == false)
	{
		EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &gDevMode);
		fprintf(gpFile, "Resolution in else ChangeDisplaySettings() Width: %d, Height %d\n", gDevMode.dmPelsWidth, gDevMode.dmPelsHeight);
		gDevMode.dmPelsWidth = MaxRes.MyWidth;
		gDevMode.dmPelsHeight = MaxRes.MyHeight;
		gDevMode.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT;
		ChangeDisplaySettings(&gDevMode, CDS_RESET | CDS_FULLSCREEN);
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW | WS_POPUP);
		ShowWindow(ghwnd, SW_SHOWMAXIMIZED);
	}
}