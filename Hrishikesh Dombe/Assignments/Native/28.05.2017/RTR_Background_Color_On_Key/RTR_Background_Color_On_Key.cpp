#include<windows.h>
#include<stdio.h>

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
DWORD dwStyle;
FILE *gpFile;
HWND ghwnd;
COLORREF gColorRef;
int giPaintFlag=0;
bool gbFullscreen = false;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("MyApp");

	if (fopen_s(&gpFile, "Log.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("CANNOT CREATE LOG FILE"), TEXT("ERROR"), MB_OK);
		exit(0);
	}
	else
		fprintf(gpFile, "Log File Is Created Successfully...\n");

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW;
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

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, TEXT("BACKGROUND COLOR CHANGING WINDOW"), WS_OVERLAPPEDWINDOW, 0, 0, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);
	if (hwnd == NULL)
	{
		fprintf(gpFile, "FAILED TO OPEN WINDOW");
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

	fprintf(gpFile, "Log File Closed Successfully!!!");
	fclose(gpFile);
	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	void ToggleFullscreen(void);
	HDC hdc;
	PAINTSTRUCT ps;
	static RECT rc;
	HBRUSH hBrush;

	switch (iMsg)
	{
	case WM_CREATE:
		break;
	case WM_PAINT:
		GetClientRect(hwnd, &rc);
		hdc = BeginPaint(hwnd, &ps);
		hBrush = CreateSolidBrush(gColorRef);
		if (giPaintFlag == 1)
		{
			FillRect(hdc,&rc,hBrush);
		}
		else if (giPaintFlag == 2)
		{
			FillRect(hdc, &rc, hBrush);
		}
		else if (giPaintFlag == 3)
		{
			FillRect(hdc, &rc, hBrush);
		}
		else if (giPaintFlag == 4)
		{
			FillRect(hdc, &rc, hBrush);
		}
		else if (giPaintFlag == 5)
		{
			FillRect(hdc, &rc, hBrush);
		}
		else if (giPaintFlag == 6)
		{
			FillRect(hdc, &rc, hBrush);
		}
		else if (giPaintFlag == 7)
		{
			FillRect(hdc, &rc, hBrush);
		}
		else if (giPaintFlag == 8)
		{
			FillRect(hdc, &rc, hBrush);
		}
		EndPaint(hwnd, &ps);
		break;
	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			DestroyWindow(ghwnd);
		case 0x52://R
			giPaintFlag = 1;
			gColorRef = RGB(255, 0, 0);
			InvalidateRect(hwnd, &rc, TRUE);
			break;
		case 0X47://G
			giPaintFlag = 2;
			gColorRef = RGB(0, 255, 0);
			InvalidateRect(NULL, &rc, TRUE);
			break;
		case 0x42://B
			giPaintFlag = 3;
			gColorRef = RGB(0, 0, 255);
			InvalidateRect(NULL, &rc, TRUE);
			break;
		case 0x43://C
			giPaintFlag = 4;
			gColorRef = RGB(0, 255, 255);
			InvalidateRect(NULL, &rc, TRUE);
			break;
		case 0x4D://M
			giPaintFlag = 5;
			gColorRef = RGB(255, 0, 255);
			InvalidateRect(NULL, &rc, TRUE);
			break;
		case 0x59://Y
			giPaintFlag = 6;
			gColorRef = RGB(255, 255, 0);
			InvalidateRect(NULL, &rc, TRUE);
			break;
		case 0x57://W
			giPaintFlag = 7;
			gColorRef = RGB(255, 255, 255);
			InvalidateRect(NULL, &rc, TRUE);
			break;
		case 0x4B://K(for Black)
			giPaintFlag = 8;
			gColorRef = RGB(0, 0, 0);
			InvalidateRect(NULL, &rc, TRUE);
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
		default:
			MessageBox(hwnd, TEXT("WRONG KEY IS PRESSED"), TEXT("ERROR"), MB_OKCANCEL);
			break;
		}
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
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
			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghwnd,GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER| SWP_FRAMECHANGED);
			}
			ShowCursor(FALSE);
		}
	}
	else
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}
}