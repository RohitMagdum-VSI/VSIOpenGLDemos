#include<windows.h>

LRESULT CALLBACK WndProc(HWND,UINT,WPARAM,LPARAM);

int WINAPI WinMain(HINSTANCE hInstance,HINSTANCE hPrevInstance,LPSTR lpszCmdLine,int iCmdShow)
{
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[]={"MS_Window"};

	wndclass.cbSize=sizeof(WNDCLASSEX);
	wndclass.style=CS_HREDRAW|CS_VREDRAW;
	wndclass.cbClsExtra=0;
	wndclass.cbWndExtra=0;
	wndclass.lpszClassName=szClassName;
	wndclass.lpszMenuName=NULL;
	wndclass.lpfnWndProc=WndProc;
	wndclass.hInstance=hInstance;
	wndclass.hbrBackground=(HBRUSH)CreateSolidBrush(RGB(0,0,255));
	wndclass.hIcon=LoadIcon(NULL,IDI_APPLICATION);
	wndclass.hIconSm=LoadIcon(NULL,IDI_APPLICATION);
	wndclass.hCursor=LoadCursor(NULL,IDC_ARROW);

	RegisterClassEx(&wndclass);

	hwnd=CreateWindow(szClassName,TEXT("HAD : HRISHIKESH ANIL DOMBE"),WS_OVERLAPPEDWINDOW,CW_USEDEFAULT,CW_USEDEFAULT,CW_USEDEFAULT,CW_USEDEFAULT,NULL,NULL,hInstance,NULL);

	ShowWindow(hwnd,iCmdShow);
	UpdateWindow(hwnd);

	while(GetMessage(&msg,NULL,0,0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
	
	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd,UINT iMsg,WPARAM wParam,LPARAM lParam)
{
	HDC hdc;
	PAINTSTRUCT ps;
	switch(iMsg)
	{
		/*case WM_PAINT:
			hdc=BeginPaint(hwnd,&ps);
			SetDCBrushColor(hdc,RGB(0,0,255));
			SelectObject(hdc,GetStockObject(DC_BRUSH));
			EndPaint(hwnd,&ps);
			break;*/
		case WM_DESTROY:
			PostQuitMessage(0);
			break;
	}
	return(DefWindowProc(hwnd,iMsg,wParam,lParam));
}