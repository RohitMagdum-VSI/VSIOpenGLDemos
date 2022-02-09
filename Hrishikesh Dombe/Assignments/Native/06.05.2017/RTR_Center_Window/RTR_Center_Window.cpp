#include<Windows.h>
#include<stdio.h>

#pragma comment(lib,"User32.lib")
#pragma comment(lib,"GDI32.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

FILE *gpFile;
HWND ghwnd;
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
int  i_X_Coordinate, i_Y_Coordinate;
bool gbFullscreen = false;
bool gbActiveWindow = false;
bool gbIsEscapeKeyPressed = false;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	void display(void);
	WNDCLASSEX wndclass;
	HWND hwnd=NULL;
	MSG msg;
	TCHAR szClassName[] = TEXT("My App");
	int iSM_CXFULLSCREEN, iSM_CYFULLSCREEN, iSM_CXMAXIMIZED, iSM_CYMAXIMIZED, i_Horizontal_Centre_Display, i_Vertical_Centre_Display, i_Horizontal_Centre_Window, i_Vertical_Centre_Window;
	int iSM_CXSCREEN, iSM_CYSCREEN;


	bool bDone = false;

	if (fopen_s(&gpFile, "Log.txt", "w") != NULL)
	{
		MessageBox(NULL, TEXT("Cannot Create Log File"), TEXT("ERROR"), MB_OK);
		exit(EXIT_FAILURE);
	}
	else
		fprintf(gpFile, "Log File is Created Successfilly...\n");

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpfnWndProc = WndProc;
	wndclass.lpszClassName = szClassName;
	wndclass.lpszMenuName = NULL;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);

	RegisterClassEx(&wndclass);

	iSM_CXSCREEN = GetSystemMetrics(SM_CXSCREEN);
	iSM_CYSCREEN = GetSystemMetrics(SM_CYSCREEN);
	iSM_CXFULLSCREEN = GetSystemMetrics(SM_CXFULLSCREEN);
	iSM_CYFULLSCREEN = GetSystemMetrics(SM_CYFULLSCREEN);
	iSM_CXMAXIMIZED = GetSystemMetrics(SM_CXMAXIMIZED);
	iSM_CYMAXIMIZED = GetSystemMetrics(SM_CYMAXIMIZED);

	i_Horizontal_Centre_Display = iSM_CXMAXIMIZED / 2;
	i_Vertical_Centre_Display = iSM_CYMAXIMIZED / 2;

	i_Horizontal_Centre_Window = WIN_WIDTH / 2;
	i_Vertical_Centre_Window = WIN_HEIGHT / 2;

	i_X_Coordinate = i_Horizontal_Centre_Display - i_Horizontal_Centre_Window;
	i_Y_Coordinate = i_Vertical_Centre_Display - i_Vertical_Centre_Window;

	fprintf(gpFile, "SM_CXSCREEN:%d,SM_CYSCREEN:%d\nSM_CXFULLSCREEN:%d,SM_CYFULLSCREEN:%d\nSM_CXMAXIMIZED:%d,SM_CYMAXIMIZED:%d\n", iSM_CXSCREEN, iSM_CYSCREEN, iSM_CXFULLSCREEN, iSM_CYFULLSCREEN, iSM_CXMAXIMIZED, iSM_CYMAXIMIZED);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, TEXT("Center Window Application"), WS_OVERLAPPEDWINDOW, i_X_Coordinate, i_Y_Coordinate, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);
	if (hwnd == NULL)
	{
		MessageBox(NULL, TEXT("Cannot Create Window"), TEXT("ERROR"), MB_OK);
		fclose(gpFile);
		exit(EXIT_FAILURE);
	}

	ghwnd = hwnd;

	ShowWindow(hwnd, iCmdShow);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
				bDone = true;
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			if (gbActiveWindow == true)
			{
				if (gbIsEscapeKeyPressed == true)
					bDone = true;
				display();
			}
		}
	}

	fprintf(gpFile, "Log File is Closed Successfully...");
	fclose(gpFile);
	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	void ToggleFullscreen(void);
	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			gbActiveWindow = true;
		else
			gbActiveWindow = false;
		break;
	case WM_CREATE:
		break;
	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			gbIsEscapeKeyPressed = true;
			break;
		case 0x46:
			if (gbFullscreen == false)
			{
				ToggleFullscreen();
				gbFullscreen = true;
			}
			else
			{
				ToggleFullscreen();
				gbFullscreen = false;
			}
			break;
		}
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullscreen(void)
{
	MONITORINFO mi = { sizeof(MONITORINFO) };
	if (gbFullscreen == false)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi));
			{
				SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
	}
	else
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(FALSE);
	}
}

void display(void)
{

}