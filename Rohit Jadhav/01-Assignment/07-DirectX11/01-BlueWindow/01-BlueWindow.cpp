#include<windows.h>
#include<stdio.h>

#include<d3d11.h>

#pragma comment(lib, "d3d11.lib")


#define WIN_WIDTH 800
#define WIN_HEIGHT 600

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

//For Error
FILE *gbFile_RRJ = NULL;
char szLogFileName_RRJ[] = "Log.txt";

//For FullScreen
bool bIsFullScreen_RRJ = false;
HWND ghwnd_RRJ = NULL;
WINDOWPLACEMENT wpPrev_RRJ = { sizeof(WINDOWPLACEMENT) };
DWORD dwStyle_RRJ;
bool bActiveWindow_RRJ = false;
float gClearColor_RRJ[4];
bool bDone_RRJ = false;


//For DirectX
IDXGISwapChain *gpIDXGISwapChain_RRJ = NULL;
ID3D11Device *gpID3D11Device_RRJ = NULL;
ID3D11DeviceContext *gpID3D11DeviceContext_RRJ = NULL;
ID3D11RenderTargetView *gpID3D11RenderTargetView_RRJ = NULL;


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow) {

	HRESULT initialize(void);
	void display(void);

	WNDCLASSEX wndclass_RRJ;
	HWND hwnd_RRJ;
	MSG msg_RRJ;
	TCHAR szName_RRJ[] = TEXT("Rohit_R_Jadhav-D3D-BlueWindow");

	

	if (fopen_s(&gbFile_RRJ, szLogFileName_RRJ, "w") != 0) {
		MessageBox(NULL, TEXT("Log File Creation Failed!!\n"), TEXT("Error"), MB_OK);
		DestroyWindow(NULL);
		exit(0);
	}
	else {
		fprintf_s(gbFile_RRJ, "Log Created!!\n\n");
		fclose(gbFile_RRJ);
	}


	wndclass_RRJ.lpszClassName = szName_RRJ;
	wndclass_RRJ.lpszMenuName = NULL;
	wndclass_RRJ.lpfnWndProc = WndProc;

	wndclass_RRJ.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass_RRJ.cbSize = sizeof(WNDCLASSEX);
	wndclass_RRJ.cbClsExtra = 0;
	wndclass_RRJ.cbWndExtra = 0;

	wndclass_RRJ.hInstance = hInstance;
	wndclass_RRJ.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass_RRJ.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass_RRJ.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass_RRJ.hCursor = LoadCursor(NULL, IDC_ARROW);

	RegisterClassEx(&wndclass_RRJ);

	hwnd_RRJ = CreateWindowEx(WS_EX_APPWINDOW,
		szName_RRJ,
		TEXT("Rohit_R_Jadhav-D3D-BlueWindow"),
		WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_VISIBLE,
		100, 100,
		WIN_WIDTH, WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL
	);

	ghwnd_RRJ = hwnd_RRJ;

	ShowWindow(hwnd_RRJ, iCmdShow);
	SetForegroundWindow(hwnd_RRJ);
	SetFocus(hwnd_RRJ);

	HRESULT hr;
	hr = initialize();
	if (FAILED(hr)) {
		fopen_s(&gbFile_RRJ, szLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "Initialize Failed!!\n\n");
		fclose(gbFile_RRJ);
		DestroyWindow(hwnd_RRJ);
	}
	else {
		fopen_s(&gbFile_RRJ, szLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "Initialize done!!\n\n");
		fclose(gbFile_RRJ);
	}

	while (bDone_RRJ != true) {
		
		if (PeekMessage(&msg_RRJ, NULL, 0, 0, PM_REMOVE)) {
			if (msg_RRJ.message == WM_QUIT) {
				bDone_RRJ = true;
			}
			else {
				TranslateMessage(&msg_RRJ);
				DispatchMessage(&msg_RRJ);
			}
		}
		else {
			
			if (bActiveWindow_RRJ == true) {
				//update
			}
			display();
		}
	}
	return((int)msg_RRJ.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam) {
	
	void uninitialize(void);
	HRESULT resize(int, int);
	void ToggleFullScreen(void);

	switch (iMsg) {
		
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			bActiveWindow_RRJ = true;
		else
			bActiveWindow_RRJ = false;
		break;



	case WM_ERASEBKGND:
		return(0);



	case WM_SIZE:
		if (gpID3D11DeviceContext_RRJ) {
			HRESULT hr;
			hr = resize(LOWORD(lParam), HIWORD(lParam));
			if (FAILED(hr)) {
				fopen_s(&gbFile_RRJ, szLogFileName_RRJ, "a+");
				fprintf_s(gbFile_RRJ, "WM_SIZE: Resize Failed!!\n\n");
				fclose(gbFile_RRJ);
				return(hr);
			}
			else {
				fopen_s(&gbFile_RRJ, szLogFileName_RRJ, "a+");
				fprintf_s(gbFile_RRJ, "WM_SIZE: Resize() Done!!\n\n");
				fclose(gbFile_RRJ);
			}
		}
		break;

	case WM_KEYDOWN:
		switch (wParam) {
		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;

		case 0X46:
			ToggleFullScreen();
			break;
		}
		break;


	case WM_CLOSE:
		uninitialize();
		break;

	case WM_DESTROY:
		uninitialize();
		PostQuitMessage(0);
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullScreen(void) {
	
	MONITORINFO mi;
	mi = { sizeof(MONITORINFO) };

	if (bIsFullScreen_RRJ == false) {
		dwStyle_RRJ = GetWindowLong(ghwnd_RRJ, GWL_STYLE);
		
		if (dwStyle_RRJ & WS_OVERLAPPEDWINDOW) {
			if (GetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ) && GetMonitorInfo(MonitorFromWindow(ghwnd_RRJ, MONITORINFOF_PRIMARY), &mi)) {
				SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd_RRJ, HWND_TOP,
					mi.rcMonitor.left,
					mi.rcMonitor.top,
					(mi.rcMonitor.right - mi.rcMonitor.left),
					(mi.rcMonitor.bottom - mi.rcMonitor.top),
					SWP_NOZORDER | SWP_FRAMECHANGED);
			
				ShowCursor(FALSE);
				bIsFullScreen_RRJ = true;
			}
		}
	}
	else {
		SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ);
		SetWindowPos(ghwnd_RRJ, HWND_TOP,
			0, 0, 0, 0,
			SWP_NOZORDER | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
		bIsFullScreen_RRJ = false;
	}
}


HRESULT initialize(void) {
	
	void uninitialize(void);
	HRESULT resize(int, int);

	HRESULT hr;

	D3D_DRIVER_TYPE d3dDriverType_RRJ;
	D3D_DRIVER_TYPE d3dDriverTypes_RRJ[] = { D3D_DRIVER_TYPE_HARDWARE,
		D3D_DRIVER_TYPE_WARP,
		D3D_DRIVER_TYPE_REFERENCE };

	D3D_FEATURE_LEVEL d3dFeatureLevel_required_RRJ = D3D_FEATURE_LEVEL_11_0;
	D3D_FEATURE_LEVEL d3dFeatureLevel_acquired_RRJ = D3D_FEATURE_LEVEL_10_0;

	UINT createDeviceFlags_RRJ = 0;
	UINT numDriverTypes_RRJ = 0;
	UINT numFeatureLevels_RRJ = 1;

	numDriverTypes_RRJ = sizeof(d3dDriverTypes_RRJ) / sizeof(d3dDriverTypes_RRJ[0]);



	DXGI_SWAP_CHAIN_DESC dxgiSwapChainDesc_RRJ;
	ZeroMemory((void*)&dxgiSwapChainDesc_RRJ, sizeof(DXGI_SWAP_CHAIN_DESC));

	dxgiSwapChainDesc_RRJ.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;

	dxgiSwapChainDesc_RRJ.BufferCount = 1;
	dxgiSwapChainDesc_RRJ.OutputWindow = ghwnd_RRJ;
	dxgiSwapChainDesc_RRJ.Windowed = TRUE;

	dxgiSwapChainDesc_RRJ.BufferDesc.Width = WIN_WIDTH;
	dxgiSwapChainDesc_RRJ.BufferDesc.Height = WIN_HEIGHT;
	dxgiSwapChainDesc_RRJ.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	dxgiSwapChainDesc_RRJ.BufferDesc.RefreshRate.Numerator = 60;
	dxgiSwapChainDesc_RRJ.BufferDesc.RefreshRate.Denominator = 1;
	
	dxgiSwapChainDesc_RRJ.SampleDesc.Count = 1;
	dxgiSwapChainDesc_RRJ.SampleDesc.Quality = 0;


	for (UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes_RRJ; driverTypeIndex++) {
	
		d3dDriverType_RRJ = d3dDriverTypes_RRJ[driverTypeIndex];
		hr = D3D11CreateDeviceAndSwapChain(
			NULL,
			d3dDriverType_RRJ,
			NULL,
			createDeviceFlags_RRJ,
			&d3dFeatureLevel_required_RRJ,
			numFeatureLevels_RRJ,
			D3D11_SDK_VERSION,
			&dxgiSwapChainDesc_RRJ,
			&gpIDXGISwapChain_RRJ,
			&gpID3D11Device_RRJ,
			&d3dFeatureLevel_acquired_RRJ,
			&gpID3D11DeviceContext_RRJ
		);

		if (SUCCEEDED(hr))
			break;
	}

	if (FAILED(hr)) {
		fopen_s(&gbFile_RRJ, szLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "D3D11CreateDeviceAndSwapChain() Failed!!\n\n");
		fclose(gbFile_RRJ);
		return(hr);
	}
	else {
		
		fopen_s(&gbFile_RRJ, szLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "D3D11CreateDeviceAndSwapChain() Done!!\n\n");
		fprintf_s(gbFile_RRJ, "The Chose Driver is Of : ");
		if (d3dDriverType_RRJ == D3D_DRIVER_TYPE_HARDWARE)
			fprintf_s(gbFile_RRJ, "Hardware Rendering!!\n\n");
		else if (d3dDriverType_RRJ == D3D_DRIVER_TYPE_WARP)
			fprintf_s(gbFile_RRJ, "Warp Type!!\n\n");
		else if (d3dDriverType_RRJ == D3D_DRIVER_TYPE_REFERENCE)
			fprintf_s(gbFile_RRJ, "Reference Type/ Sotfware Rendering !!\n\n");
		else
			fprintf_s(gbFile_RRJ, "Unknown Type!!\n\n");

		fprintf_s(gbFile_RRJ, "The Supported Highest Feature Level is: ");
		if (d3dFeatureLevel_acquired_RRJ == D3D_FEATURE_LEVEL_11_0)
			fprintf_s(gbFile_RRJ, "D3D_FEATURE_LEVEL_11_0\n\n");
		else if (d3dFeatureLevel_acquired_RRJ == D3D_FEATURE_LEVEL_10_1)
			fprintf_s(gbFile_RRJ, "D3D_FEATURE_LEVEL_10_1\n\n");
		else if (d3dFeatureLevel_acquired_RRJ == D3D_FEATURE_LEVEL_10_0)
			fprintf_s(gbFile_RRJ, "D3D_FEATURE_LEVEL_10_0\n\n");
		else
			fprintf_s(gbFile_RRJ, "Unknown!!\n\n");
	
		fclose(gbFile_RRJ);
	}

	gClearColor_RRJ[0] = 0.0f;
	gClearColor_RRJ[1] = 0.0f;
	gClearColor_RRJ[2] = 1.0f;
	gClearColor_RRJ[3] = 1.0f;


	hr = resize(WIN_WIDTH, WIN_HEIGHT);
	if (FAILED(hr)) {
		fopen_s(&gbFile_RRJ, szLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "resize() Failed!!\n\n");
		fclose(gbFile_RRJ);
		return(hr);
	}
	else {
		fopen_s(&gbFile_RRJ, szLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "resize() done!!\n\n");
		fclose(gbFile_RRJ);
	}
	return(S_OK);
}


void uninitialize(void) {
	
		
	if (gpID3D11RenderTargetView_RRJ) {
		gpID3D11RenderTargetView_RRJ->Release();
		gpID3D11RenderTargetView_RRJ = NULL;
	}

	if (gpIDXGISwapChain_RRJ) {
		gpIDXGISwapChain_RRJ->Release();
		gpIDXGISwapChain_RRJ = NULL;
	}

	if (gpID3D11DeviceContext_RRJ) {
		gpID3D11DeviceContext_RRJ->Release();
		gpID3D11DeviceContext_RRJ = NULL;
	}

	if (gpID3D11Device_RRJ) {
		gpID3D11Device_RRJ->Release();
		gpID3D11Device_RRJ = NULL;
	}

	if (gbFile_RRJ) {
		fopen_s(&gbFile_RRJ, szLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "Log Close!!n");
		fclose(gbFile_RRJ);
		gbFile_RRJ = NULL;
	}

}



HRESULT resize(int width, int height) {

	HRESULT hr = S_OK;

	if (gpID3D11RenderTargetView_RRJ) {
		gpID3D11RenderTargetView_RRJ->Release();
		gpID3D11RenderTargetView_RRJ = NULL;
	}

	gpIDXGISwapChain_RRJ->ResizeBuffers(1, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, 0);


	ID3D11Texture2D *pID3D11Texture2D_BackBuffer_RRJ = NULL;
	gpIDXGISwapChain_RRJ->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pID3D11Texture2D_BackBuffer_RRJ);

	hr = gpID3D11Device_RRJ->CreateRenderTargetView(pID3D11Texture2D_BackBuffer_RRJ, NULL,
		&gpID3D11RenderTargetView_RRJ);

	if (FAILED(hr)) {
		fopen_s(&gbFile_RRJ, szLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ID3D11Device::CreateRenderTargetView() Failed!!\n\n");
		fclose(gbFile_RRJ);
		return(hr);
	}
	else {
		fopen_s(&gbFile_RRJ, szLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ID3D11Device::CreateRenderTargetView() Done!!\n\n");
		fclose(gbFile_RRJ);
	}

	pID3D11Texture2D_BackBuffer_RRJ->Release();
	pID3D11Texture2D_BackBuffer_RRJ = NULL;

	gpID3D11DeviceContext_RRJ->OMSetRenderTargets(1, &gpID3D11RenderTargetView_RRJ, NULL);


	D3D11_VIEWPORT d3dViewPort;
	d3dViewPort.TopLeftX = 0;
	d3dViewPort.TopLeftY = 0;
	d3dViewPort.Width = (float)width;
	d3dViewPort.Height = (float)height;
	d3dViewPort.MinDepth = 0.0f;
	d3dViewPort.MaxDepth = 1.0f;
	gpID3D11DeviceContext_RRJ->RSSetViewports(1, &d3dViewPort);

	return(hr);
}

void display(void) {
	
	gpID3D11DeviceContext_RRJ->ClearRenderTargetView(gpID3D11RenderTargetView_RRJ, gClearColor_RRJ);

	gpIDXGISwapChain_RRJ->Present(0, 0);
}



