#include<windows.h>
#include<stdio.h>
#define WIN_WIDTH 800
#define WIN_HEIGHT 600

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

FILE *gpFile;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("MyApp");

	if (fopen_s(&gpFile, "Log.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("CANNOT OPEN FILE"), TEXT("ERROR"), MB_OK);
		exit(0);
	}
	else
		fprintf(gpFile, "Log File is created successfully...\n");

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

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, TEXT("MESSAGE BOX"), WS_OVERLAPPEDWINDOW, 100, 100, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);
	if (hwnd == NULL)
	{
		fprintf(gpFile, "Cannot Create Window");
		exit(0);
	}

	ShowWindow(hwnd, iCmdShow);
	UpdateWindow(hwnd);

	while (GetMessage(&msg, NULL, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	fprintf(gpFile, "Log File is Closed!!!");
	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	TCHAR coordinates[256];
	switch (iMsg)
	{
	case WM_CREATE:
		MessageBox(hwnd, TEXT("IN WM_CREATE"), TEXT("MSG OF WM_CREATE"), MB_OK);
		break;
	case WM_LBUTTONDOWN:
		sprintf(coordinates, "IN WM_LBUTTONDOWN : CO-ORDINATES ARE X:%ld AND Y:%ld", LOWORD(lParam), HIWORD(lParam));
		MessageBox(hwnd, coordinates, TEXT("MSG OF WM_LBUTTONDOWN"), MB_OK);
		break;
	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			MessageBox(hwnd, TEXT("ESCAPE KEY IS PRESSED"), TEXT("MESSAGE OF WM_KEYDOWN"), MB_OK);
			DestroyWindow(hwnd);
			break;
		case 0x42:
			MessageBox(hwnd, TEXT("B KEY IS PRESSED"), TEXT("MESSAGE OF WM_KEYDOWN"), MB_OK);
			break;
		case 0x46:
			MessageBox(hwnd, TEXT("F KEY IS PRESSED"), TEXT("MESSAGE OF WM_KEYDOWN"), MB_OK);
			break;
		case 0x4C:
			MessageBox(hwnd, TEXT("L KEY IS PRESSED"), TEXT("MESSAGE OF WM_KEYDOWN"), MB_OK);	
			break;
		case 0x54:
			MessageBox(hwnd, TEXT("T KEY IS PRESSED"), TEXT("MESSAGE OF WM_KEYDOWN"), MB_OK);
			break;
		case 0x51:
			MessageBox(hwnd, TEXT("Q KEY IS PRESSED"), TEXT("MESSAGE OF WM_KEYDOWN"), MB_OK);
			break;
		default:
			MessageBox(hwnd, TEXT("WRONG KEY IS PRESSED"), TEXT("MESSAGE OF WM_KEYDOWN"), MB_OK);
			break;
		}
		break;
	case WM_DESTROY:
		MessageBox(hwnd, TEXT("IN WM_DESTROY"), TEXT("MESSAGE OF WM_DESTROY"), MB_OK);
		PostQuitMessage(0);
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}