#include<windows.h>
#include<stdio.h>

#include<d3d11.h>
#include<d3dcompiler.h>


#pragma warning(disable: 4838)
#include"XNAMath/xnamath.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "D3dcompiler.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

//For Error
FILE *gbFile_RRJ = NULL;
char gszLogFileName_RRJ[] = "Log.txt";

//For FullScreen
bool bIsFullScreen_RRJ = false;
WINDOWPLACEMENT wpPrev_RRJ = { sizeof(WINDOWPLACEMENT) };
DWORD dwStyle_RRJ;
HWND ghwnd_RRJ = NULL;

bool bActiveWindow_RRJ = false;
bool gbEscapeKeyIsPressed_RRJ = false;

//For DirectX
float gClearColor_RRJ[4];

IDXGISwapChain *gpIDXGISwapChain_RRJ = NULL;
ID3D11Device *gpID3D11Device_RRJ = NULL;
ID3D11DeviceContext *gpID3D11DeviceContext_RRJ = NULL;
ID3D11RenderTargetView *gpID3D11RenderTargetView_RRJ = NULL;

//For DirectX Shaders
ID3D11VertexShader *gpID3D11VertexShader_RRJ = NULL;
ID3D11PixelShader *gpID3D11PixelShader_RRJ = NULL;

ID3D11Buffer *gpID3D11Buffer_VertexBuffer_RRJ = NULL;
ID3D11Buffer *gpID3D11Buffer_ConstantBuffer_RRJ = NULL;

ID3D11InputLayout *gpID3D11InputLayout_RRJ = NULL;


//For Uniform
struct CBUFFER {
	XMMATRIX WorldViewProjectionMatrix_RRJ;
};

XMMATRIX gOrthographicProjectionMatrix_RRJ;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow) {

	HRESULT initialize(void);
	void uninitialize(void);
	void display(void);

	WNDCLASSEX wndclass_RRJ;
	HWND hwnd_RRJ;
	MSG msg_RRJ;
	TCHAR szName_RRJ[] = TEXT("Rohit_R_Jadhav-D3D-02-Ortho");
	bool bDone_RRJ = false;

	fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "w");
	if (gbFile_RRJ == NULL) {
		MessageBox(NULL, TEXT("ERROR: Log Creation Failed!!\n"), TEXT("ERROR"), MB_OK);
		uninitialize();
		DestroyWindow(NULL);
	}
	else
		fprintf_s(gbFile_RRJ, "SUCCESS: Log Created!!\n");

	fclose(gbFile_RRJ);


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

	hwnd_RRJ = CreateWindow(szName_RRJ,
		TEXT("Rohit_R_Jadhav-D3D-02-Ortho"),
		WS_OVERLAPPEDWINDOW,
		100, 100,
		WIN_WIDTH, WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghwnd_RRJ = hwnd_RRJ;

	ShowWindow(hwnd_RRJ, iCmdShow);
	SetForegroundWindow(hwnd_RRJ);
	SetFocus(hwnd_RRJ);

	HRESULT hr_RRJ;
	hr_RRJ = initialize();
	if (FAILED(hr_RRJ)) {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize Failed!!\n");
		fclose(gbFile_RRJ);
		DestroyWindow(hwnd_RRJ);
		hwnd_RRJ = NULL;
	}
	else {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize() Done!!\n");
		fclose(gbFile_RRJ);
	}


	while (bDone_RRJ == false) {
		if (PeekMessage(&msg_RRJ, NULL, 0, 0, PM_REMOVE)){
			if (msg_RRJ.message == WM_QUIT)
				bDone_RRJ = true;
			else {
				TranslateMessage(&msg_RRJ);
				DispatchMessage(&msg_RRJ);
			}

		}
		else {
			display();

			if (bActiveWindow_RRJ == true) {
				if (gbEscapeKeyIsPressed_RRJ == true)
					bDone_RRJ = true;
			}
		}
	}

	uninitialize();
	return((int)msg_RRJ.wParam);
}


LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam) {
	
	HRESULT resize(int, int);
	void ToggleFullScreen(void);
	void uninitialize(void);

	HRESULT hr_RRJ;

	switch (iMsg) {
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			bActiveWindow_RRJ = true;
		else
			bActiveWindow_RRJ = false;
		break;

	case WM_ERASEBKGND:
		break;

	case WM_SIZE:
		if (gpID3D11DeviceContext_RRJ) {
			hr_RRJ = resize(LOWORD(lParam), HIWORD(lParam));
			if (FAILED(hr_RRJ)) {
				fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
				fprintf_s(gbFile_RRJ, "ERROR: Resize() In WM_SIZE Failed!!\n");
				fclose(gbFile_RRJ);
				return(hr_RRJ);
			}
			else {
				fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
				fprintf_s(gbFile_RRJ, "SUCCESS: Resize() In WM_SIZE Done !!\n");
				fclose(gbFile_RRJ);
			}
		}
		break;

	case WM_KEYDOWN:
		switch (wParam) {
		case VK_ESCAPE:
			if (gbEscapeKeyIsPressed_RRJ == false)
				gbEscapeKeyIsPressed_RRJ = true;
			else
				gbEscapeKeyIsPressed_RRJ = false;
			break;

		case 0x46:
			if (bIsFullScreen_RRJ == false) {
				ToggleFullScreen();
				bIsFullScreen_RRJ = true;
			}
			else {
				ToggleFullScreen();
				bIsFullScreen_RRJ = false;
			}
			break;

		default:
			break;
		}
		break;

	case WM_LBUTTONDOWN:
		break;

	case WM_CLOSE:
		uninitialize();
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullScreen(void) {
	
	MONITORINFO mi_RRJ = {sizeof(MONITORINFO)};

	if (bIsFullScreen_RRJ == false) {
		dwStyle_RRJ = GetWindowLong(ghwnd_RRJ, GWL_STYLE);
		if (dwStyle_RRJ & WS_OVERLAPPEDWINDOW) {
			if (GetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ) && GetMonitorInfo(MonitorFromWindow(ghwnd_RRJ, MONITORINFOF_PRIMARY), &mi_RRJ)) {
				SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd_RRJ, HWND_TOP,
					mi_RRJ.rcMonitor.left,
					mi_RRJ.rcMonitor.top,
					(mi_RRJ.rcMonitor.right - mi_RRJ.rcMonitor.left),
					(mi_RRJ.rcMonitor.bottom - mi_RRJ.rcMonitor.top),
					SWP_NOZORDER | SWP_FRAMECHANGED);
				ShowCursor(FALSE);
			}
		}
	}
	else {
		SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ);
		SetWindowPos(ghwnd_RRJ, HWND_TOP,
			0, 0, 0, 0,
			SWP_NOZORDER | SWP_NOSIZE | SWP_NOMOVE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}
}


HRESULT initialize(void) {

	void uninitialize(void);
	HRESULT resize(int, int);


	HRESULT hr_RRJ;

	D3D_DRIVER_TYPE d3dDriverType_RRJ;
	D3D_DRIVER_TYPE d3dDriverTypes_RRJ[] = {
		D3D_DRIVER_TYPE_HARDWARE,
		D3D_DRIVER_TYPE_WARP,
		D3D_DRIVER_TYPE_REFERENCE,
	};


	D3D_FEATURE_LEVEL d3dFeatureLevel_required_RRJ = D3D_FEATURE_LEVEL_11_0;
	D3D_FEATURE_LEVEL d3dFeatureLevel_acquired_RRJ = D3D_FEATURE_LEVEL_10_0;

	UINT createDeviceFlags_RRJ = 0;
	UINT numDriverTypes_RRJ = 0;
	UINT numFeatureLevels_RRJ = 1;

	numDriverTypes_RRJ = sizeof(d3dDriverTypes_RRJ) / sizeof(d3dDriverTypes_RRJ[0]);


	DXGI_SWAP_CHAIN_DESC dxgiSwapChainDesc_RRJ;

	ZeroMemory((void*)&dxgiSwapChainDesc_RRJ, sizeof(DXGI_SWAP_CHAIN_DESC));

	dxgiSwapChainDesc_RRJ.BufferCount = 1;

	dxgiSwapChainDesc_RRJ.BufferDesc.Width = WIN_WIDTH;
	dxgiSwapChainDesc_RRJ.BufferDesc.Height = WIN_HEIGHT;
	dxgiSwapChainDesc_RRJ.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	dxgiSwapChainDesc_RRJ.BufferDesc.RefreshRate.Numerator = 60;
	dxgiSwapChainDesc_RRJ.BufferDesc.RefreshRate.Denominator = 1;

	dxgiSwapChainDesc_RRJ.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	dxgiSwapChainDesc_RRJ.OutputWindow = ghwnd_RRJ;

	dxgiSwapChainDesc_RRJ.SampleDesc.Count = 1;
	dxgiSwapChainDesc_RRJ.SampleDesc.Quality = 0;

	dxgiSwapChainDesc_RRJ.Windowed = TRUE;


	for (UINT driverTypeIndex_RRJ = 0; driverTypeIndex_RRJ < numDriverTypes_RRJ; driverTypeIndex_RRJ++) {


		d3dDriverType_RRJ = d3dDriverTypes_RRJ[driverTypeIndex_RRJ];

		hr_RRJ = D3D11CreateDeviceAndSwapChain(
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

		if (SUCCEEDED(hr_RRJ))
			break;
	}

	if (FAILED(hr_RRJ)) {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: D3D11CreateDeviceAndSwapChain() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else {

		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: D3D11CreateDeviceAndSwapChain() Done!!!\n");

		fprintf_s(gbFile_RRJ, "SUCCESS: Selected Driver is: ");
		if (d3dDriverType_RRJ == D3D_DRIVER_TYPE_HARDWARE)
			fprintf_s(gbFile_RRJ, "D3D_DRIVER_TYPE_HARDWARE !!\n");
		else if (d3dDriverType_RRJ == D3D_DRIVER_TYPE_WARP)
			fprintf_s(gbFile_RRJ, "D3D_DRIVER_TYPE_WARP !!\n");
		else if (d3dDriverType_RRJ == D3D_DRIVER_TYPE_REFERENCE)
			fprintf_s(gbFile_RRJ, "D3D_DRIVER_TYPE_REFERENCE !!\n");
		else
			fprintf_s(gbFile_RRJ, "Unknow!!\n");


		fprintf_s(gbFile_RRJ, "SUCCESS: Feature Level Get: ");
		if (d3dFeatureLevel_acquired_RRJ == D3D_FEATURE_LEVEL_11_0)
			fprintf_s(gbFile_RRJ, "11.0 !!\n");
		else if (d3dFeatureLevel_acquired_RRJ == D3D_FEATURE_LEVEL_10_1)
			fprintf_s(gbFile_RRJ, "10.1 !!\n");
		else if (d3dFeatureLevel_acquired_RRJ == D3D_FEATURE_LEVEL_10_0)
			fprintf_s(gbFile_RRJ, "10.0 !!\n");
		else
			fprintf_s(gbFile_RRJ, "Unknown !!\n");

		fclose(gbFile_RRJ);
	}



	/********** VERTEX SHADER **********/
	const char *vertexShaderSourceCode_RRJ =
		"cbuffer ConstantBuffer" \
		"{" \
		"float4x4 worldViewProjectionMatrix;" \
		"}" \

		"float4 main(float4 pos: POSITION) : SV_POSITION" \
		"{" \
		"float4 position = mul(worldViewProjectionMatrix, pos);" \
		"return(position);" \
		"}";

	ID3DBlob *pID3DBlob_VertexShaderCode_RRJ = NULL;
	ID3DBlob *pID3DBlob_Error_RRJ = NULL;

	hr_RRJ = D3DCompile(
		vertexShaderSourceCode_RRJ,
		lstrlenA(vertexShaderSourceCode_RRJ) + 1,
		"VS",
		NULL,
		D3D_COMPILE_STANDARD_FILE_INCLUDE,
		"main",
		"vs_5_0",
		0,
		0,
		&pID3DBlob_VertexShaderCode_RRJ,
		&pID3DBlob_Error_RRJ);

	if (FAILED(hr_RRJ)) {
		
		if (pID3DBlob_Error_RRJ != NULL) {
			fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "ERROR: Vertex Shader Compilation Error: \n%s\n",
				(char*)pID3DBlob_Error_RRJ->GetBufferPointer());

			fclose(gbFile_RRJ);
			pID3DBlob_Error_RRJ->Release();
			pID3DBlob_Error_RRJ = NULL;

			return(hr_RRJ);
		}
	}
	else {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Vertex Shader Compilation Done!!\n");
		fclose(gbFile_RRJ);
	}


	hr_RRJ = gpID3D11Device_RRJ->CreateVertexShader(
		pID3DBlob_VertexShaderCode_RRJ->GetBufferPointer(),
		pID3DBlob_VertexShaderCode_RRJ->GetBufferSize(),
		NULL,
		&gpID3D11VertexShader_RRJ);


	if (FAILED(hr_RRJ)) {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: ID3D11Device::CreateVertex() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: ID3D11Device::CreateVertex() Done!!\n");
		fclose(gbFile_RRJ);
	}

	gpID3D11DeviceContext_RRJ->VSSetShader(gpID3D11VertexShader_RRJ, NULL, NULL);



	/********** PIXEL SHADER **********/
	const char *pixelShaderSourceCode_RRJ =
		"float4 main(void) : SV_TARGET " \
		"{" \
		"return(float4(1.0f, 1.0f, 1.0f, 1.0f));" \
		"}";

	ID3DBlob *pID3DBlob_PixelShaderCode_RRJ = NULL;
	pID3DBlob_Error_RRJ = NULL;

	hr_RRJ = D3DCompile(
		pixelShaderSourceCode_RRJ,
		lstrlenA(pixelShaderSourceCode_RRJ) + 1,
		"PS",
		NULL,
		D3D_COMPILE_STANDARD_FILE_INCLUDE,
		"main",
		"ps_5_0",
		0, 0,
		&pID3DBlob_PixelShaderCode_RRJ,
		&pID3DBlob_Error_RRJ);

	if (FAILED(hr_RRJ)) {
		if (pID3DBlob_Error_RRJ != NULL) {
			fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "ERROR: Pixel Shader Compilation Error: %s\n",
				(char*)pID3DBlob_Error_RRJ->GetBufferPointer());
			fclose(gbFile_RRJ);

			pID3DBlob_Error_RRJ->Release();
			pID3DBlob_Error_RRJ = NULL;

			return(hr_RRJ);
		}
	}
	else {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Pixel Shader Compilation Done!!\n");
		fclose(gbFile_RRJ);
	}


	hr_RRJ = gpID3D11Device_RRJ->CreatePixelShader(
		pID3DBlob_PixelShaderCode_RRJ->GetBufferPointer(),
		pID3DBlob_PixelShaderCode_RRJ->GetBufferSize(),
		NULL, &gpID3D11PixelShader_RRJ
	);

	if (FAILED(hr_RRJ)) {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: ID3D11Device::CreatePixelShader() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: ID3D11Device::CreatePixelShader() Done!!\n");
		fclose(gbFile_RRJ);
	}

	gpID3D11DeviceContext_RRJ->PSSetShader(gpID3D11PixelShader_RRJ, NULL, NULL);




	/********** INPUT LAYOUT **********/
	D3D11_INPUT_ELEMENT_DESC inputElementDesc_RRJ;
	ZeroMemory((void*)&inputElementDesc_RRJ, sizeof(D3D11_INPUT_ELEMENT_DESC));

	inputElementDesc_RRJ.SemanticName = "POSITION";
	inputElementDesc_RRJ.SemanticIndex = 0;
	inputElementDesc_RRJ.Format = DXGI_FORMAT_R32G32B32_FLOAT;
	inputElementDesc_RRJ.InputSlot = 0;
	inputElementDesc_RRJ.AlignedByteOffset = 0;
	inputElementDesc_RRJ.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	inputElementDesc_RRJ.InstanceDataStepRate = 0;

	hr_RRJ = gpID3D11Device_RRJ->CreateInputLayout(
		&inputElementDesc_RRJ, 1,
		pID3DBlob_VertexShaderCode_RRJ->GetBufferPointer(),
		pID3DBlob_VertexShaderCode_RRJ->GetBufferSize(),
		&gpID3D11InputLayout_RRJ);

	if (FAILED(hr_RRJ)) {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: ID3D11Device::CreateInputLayout() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: ID3D11Device::CreateInputLayout() Done!!\n");
		fclose(gbFile_RRJ);
	}

	gpID3D11DeviceContext_RRJ->IASetInputLayout(gpID3D11InputLayout_RRJ);

	pID3DBlob_PixelShaderCode_RRJ->Release();
	pID3DBlob_PixelShaderCode_RRJ = NULL;
	pID3DBlob_VertexShaderCode_RRJ->Release();
	pID3DBlob_VertexShaderCode_RRJ = NULL;


	/********** POSITION and COLORS **********/
	float tri_Position_RRJ[] = {
		0.0f, 50.0f, 0.0f,
		50.0f, -50.0f, 0.0f,
		-50.0f, -50.0f, 0.0f,
	};


	/********** VERTEX BUFFER **********/
	D3D11_BUFFER_DESC bufferDesc_VertexBuffer_RRJ;
	ZeroMemory((void*)&bufferDesc_VertexBuffer_RRJ, sizeof(D3D11_BUFFER_DESC));
	
	bufferDesc_VertexBuffer_RRJ.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc_VertexBuffer_RRJ.ByteWidth = sizeof(float) * _ARRAYSIZE(tri_Position_RRJ);
	bufferDesc_VertexBuffer_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc_VertexBuffer_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_VertexBuffer_RRJ, NULL,
		&gpID3D11Buffer_VertexBuffer_RRJ);

	if (FAILED(hr_RRJ)) {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: ID3D11Device::CreateBuffer() For VertexBuffer Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: ID3D11Device:CreateBuffer() For VertexBuffer For Position Done!!\n");
		fclose(gbFile_RRJ);
	}



	/********** For MEMORY MAPPED IO **********/
	D3D11_MAPPED_SUBRESOURCE mappedSubresource_RRJ;
	ZeroMemory((void*)&mappedSubresource_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));
	
	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_RRJ, 0,
		D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource_RRJ);

	memcpy(mappedSubresource_RRJ.pData, tri_Position_RRJ, sizeof(tri_Position_RRJ));
	
	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_RRJ, 0);


	/********** For Constant Buffer **********/
	D3D11_BUFFER_DESC bufferDesc_ConstantBuffer_RRJ;
	ZeroMemory((void*)&bufferDesc_ConstantBuffer_RRJ, sizeof(D3D11_BUFFER_DESC));

	bufferDesc_ConstantBuffer_RRJ.Usage = D3D11_USAGE_DEFAULT;
	bufferDesc_ConstantBuffer_RRJ.ByteWidth = sizeof(CBUFFER);
	bufferDesc_ConstantBuffer_RRJ.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_ConstantBuffer_RRJ, 0, 
		&gpID3D11Buffer_ConstantBuffer_RRJ);

	if (FAILED(hr_RRJ)) {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: ID3D11Device::CreateBuffer() For Constant Buffer Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: ID3D11Device:CreateBuffer() For Constant Buffer Done!!\n");
		fclose(gbFile_RRJ);
	}

	gpID3D11DeviceContext_RRJ->VSSetConstantBuffers(0, 1, 
		&gpID3D11Buffer_ConstantBuffer_RRJ);


	/********** Clear Color **********/
	gClearColor_RRJ[0] = 0.0f;
	gClearColor_RRJ[1] = 0.0f;
	gClearColor_RRJ[2] = 1.0f;
	gClearColor_RRJ[3] = 1.0f;


	gOrthographicProjectionMatrix_RRJ = XMMatrixIdentity();

	hr_RRJ = resize(WIN_WIDTH, WIN_HEIGHT);
	if (FAILED(hr_RRJ)) {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Resize() in Initialize() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Resize() in Initialize() Done!!\n");
		fclose(gbFile_RRJ);
	}

	return(S_OK);
}

void uninitialize(void) {

	if (gpID3D11Buffer_ConstantBuffer_RRJ) {
		gpID3D11Buffer_ConstantBuffer_RRJ->Release();
		gpID3D11Buffer_ConstantBuffer_RRJ = NULL;
	}


	if (gpID3D11InputLayout_RRJ) {
		gpID3D11InputLayout_RRJ->Release();
		gpID3D11InputLayout_RRJ = NULL;
	}

	if (gpID3D11Buffer_VertexBuffer_RRJ) {
		gpID3D11Buffer_VertexBuffer_RRJ->Release();
		gpID3D11Buffer_VertexBuffer_RRJ = NULL;
	}

	if (gpID3D11PixelShader_RRJ) {
		gpID3D11PixelShader_RRJ->Release();
		gpID3D11PixelShader_RRJ = NULL;
	}

	if (gpID3D11VertexShader_RRJ) {
		gpID3D11VertexShader_RRJ->Release();
		gpID3D11VertexShader_RRJ = NULL;
	}

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
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Log File Close!!\n");
		fclose(gbFile_RRJ);
		gbFile_RRJ = NULL;
	}

}


HRESULT resize(int width, int height) {

	HRESULT hr_RRJ;


	if (gpID3D11RenderTargetView_RRJ) {
		gpID3D11RenderTargetView_RRJ->Release();
		gpID3D11RenderTargetView_RRJ = NULL;
	}

	gpIDXGISwapChain_RRJ->ResizeBuffers(1, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, 0);

	ID3D11Texture2D *pID3D11Texture2D_BackBuffer_RRJ = NULL;
	gpIDXGISwapChain_RRJ->GetBuffer(0, __uuidof(ID3D11Texture2D),
		(LPVOID*)&pID3D11Texture2D_BackBuffer_RRJ);


	hr_RRJ = gpID3D11Device_RRJ->CreateRenderTargetView(pID3D11Texture2D_BackBuffer_RRJ,
		NULL,
		&gpID3D11RenderTargetView_RRJ);
	
	
	if (FAILED(hr_RRJ)) {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: ID3D11Device::CreateRenderTargetView() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: ID3D11Device::CreateRenderTargetView() Done!!\n");
		fclose(gbFile_RRJ);
	}

	pID3D11Texture2D_BackBuffer_RRJ->Release();
	pID3D11Texture2D_BackBuffer_RRJ = NULL;

	gpID3D11DeviceContext_RRJ->OMSetRenderTargets(1, &gpID3D11RenderTargetView_RRJ, NULL);


	/********** ViewPort **********/
	D3D11_VIEWPORT d3dViewPort_RRJ;

	d3dViewPort_RRJ.TopLeftX = 0;
	d3dViewPort_RRJ.TopLeftY = 0;
	d3dViewPort_RRJ.Width = (float)width;
	d3dViewPort_RRJ.Height = (float)height;
	d3dViewPort_RRJ.MinDepth = 0.0f;
	d3dViewPort_RRJ.MaxDepth = 1.0f;

	gpID3D11DeviceContext_RRJ->RSSetViewports(1, &d3dViewPort_RRJ);


	gOrthographicProjectionMatrix_RRJ = XMMatrixIdentity();

	if (width <= height) {
		gOrthographicProjectionMatrix_RRJ = XMMatrixOrthographicOffCenterLH(
			-100.0f, 100.0f,
			-100.0f * ((float)height / (float)width), 100.0f * ((float)height / (float)width),
			-100.0f, 100.0f);
	}
	else {
		gOrthographicProjectionMatrix_RRJ = XMMatrixOrthographicOffCenterLH(
			-100.0f * ((float)width / float(height)), 100.0f * ((float)width / (float)height),
			-100.0f, 100.0f,
			-100.0f, 100.0f);
	}

	return(hr_RRJ);
}

void display(void) {

	gpID3D11DeviceContext_RRJ->ClearRenderTargetView(gpID3D11RenderTargetView_RRJ, gClearColor_RRJ);


	//Select Which  Vertex Buffer To Display
	UINT  stride = sizeof(float) * 3;
	UINT offset = 0;
	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(0, 1, &gpID3D11Buffer_VertexBuffer_RRJ, &stride, &offset);


	//Select Geometry Primitive
	gpID3D11DeviceContext_RRJ->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);


	//Translation
	XMMATRIX worldMatrix_RRJ = XMMatrixIdentity();
	XMMATRIX viewMatrix_RRJ = XMMatrixIdentity();

	XMMATRIX wvpMatrix_RRJ = worldMatrix_RRJ * viewMatrix_RRJ * gOrthographicProjectionMatrix_RRJ;
	//XMMATRIX wvpMatrix_RRJ = gOrthographicProjectionMatrix_RRJ * viewMatrix_RRJ * worldMatrix_RRJ;


	//Load the data into Constant Buffer
	CBUFFER constantBuffer_RRJ;
	constantBuffer_RRJ.WorldViewProjectionMatrix_RRJ = wvpMatrix_RRJ;
	gpID3D11DeviceContext_RRJ->UpdateSubresource(gpID3D11Buffer_ConstantBuffer_RRJ, 0, 
		NULL, &constantBuffer_RRJ, 0, 0);

	//Draw
	gpID3D11DeviceContext_RRJ->Draw(3, 0);

	gpIDXGISwapChain_RRJ->Present(0, 0);
}

