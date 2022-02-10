#include<windows.h>
#include<stdio.h>

#include<d3d11.h>
#include<D3dcompiler.h>

#pragma warning(disable : 4838)
#include"XNAMATH/xnamath.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "D3dcompiler.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);


//For Fullscreen
bool bIsFullScreen_RRJ = false;
WINDOWPLACEMENT wpPrev_RRJ = {sizeof(WINDOWPLACEMENT)};
DWORD dwStyle_RRJ;
HWND ghwnd_RRJ = NULL;
bool bActiveWindow_RRJ = false;

//For ERROR
FILE *gbFile_RRJ = NULL;
char *gszLogFileName_RRJ = "Log.txt";

//For DirectX
IDXGISwapChain *gpIDXGISwapChain_RRJ = NULL;
ID3D11Device *gpID3D11Device_RRJ = NULL;
ID3D11DeviceContext *gpID3D11DeviceContext_RRJ = NULL;
ID3D11RenderTargetView *gpID3D11RenderTargetView_RRJ = NULL;

//For Shader
ID3D11VertexShader *gpID3D11VertexShader_RRJ = NULL;
ID3D11PixelShader *gpID3D11PixelShader_RRJ = NULL;

//For Attributes
ID3D11InputLayout *gpID3D11InputLayout_RRJ = NULL;

//For Tri
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Tri_Position_RRJ = NULL;
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Tri_Color_RRJ = NULL;

//For Rect
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Rect_Position_RRJ = NULL;
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Rect_Color_RRJ = NULL;

//For Uniform
ID3D11Buffer *gpID3D11Buffer_ConstantBuffer_RRJ = NULL;

struct CBUFFER{
	XMMATRIX WorldViewProjectionMatrix;
};

//For Projection
XMMATRIX gPerspectiveProjectionMatrix_RRJ;


//For ClearColor;
float gClearColor[4];


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow){


	HRESULT initialize(void);
	void display(void);


	fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "w");
	if(gbFile_RRJ == NULL){
		MessageBox(NULL, TEXT("Log Creation Failed!!"), TEXT("ERROR"), MB_OK);
		exit(0);
	}
	else{
		fprintf_s(gbFile_RRJ, "SUCCESS: Log Created!!\n");
		fclose(gbFile_RRJ);
	}



	HRESULT hr_RRJ;
	bool bDone_RRJ = false;

	WNDCLASSEX wndclass_RRJ;
	HWND hwnd_RRJ;
	MSG msg_RRJ;
	TCHAR szName_RRJ[] = TEXT("Rohit_R_Jadhav-D3D-09-Color-Tri-Rect");




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
		TEXT("Rohit_R_Jadhav-D3D-09-Color-Tri-Rect"),
		WS_OVERLAPPEDWINDOW,
		100, 100,
		WIN_WIDTH, WIN_HEIGHT,
		NULL, 
		NULL,
		hInstance,
		NULL);

	ghwnd_RRJ = hwnd_RRJ;



	hr_RRJ = initialize();
	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize() Failed!!\n");
		fclose(gbFile_RRJ);

		DestroyWindow(hwnd_RRJ);
		hwnd_RRJ = NULL;
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize() Done!!\n");
		fclose(gbFile_RRJ);
	}



	ShowWindow(hwnd_RRJ, iCmdShow);
	SetForegroundWindow(hwnd_RRJ);
	SetFocus(hwnd_RRJ);


	while(bDone_RRJ == false){
		if(PeekMessage(&msg_RRJ, NULL, 0, 0, PM_REMOVE)){
			
			if(msg_RRJ.message == WM_QUIT)
				bDone_RRJ = true;
			else{
				TranslateMessage(&msg_RRJ);
				DispatchMessage(&msg_RRJ);
			}

		}
		else{
			if(bActiveWindow_RRJ == true){
				//UPdate();
			}

			display();
		}
	}

	return((int)msg_RRJ.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM  lParam){

	void ToggleFullScreen(void);
	HRESULT resize(int, int);
	void uninitialize(void);

	HRESULT hr_RRJ;

	switch(iMsg){

		case WM_ACTIVATE:
			if(HIWORD(wParam) == 0)
				bActiveWindow_RRJ = true;
			else
				bActiveWindow_RRJ = false;
			break;


		case WM_ERASEBKGND:
			return(0);

		case WM_SIZE:
			if(gpID3D11DeviceContext_RRJ){
				hr_RRJ = resize(LOWORD(lParam), HIWORD(lParam));
				if(FAILED(hr_RRJ)){
					fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
					fprintf_s(gbFile_RRJ, "ERROR: Resize() in WM_SIZE Failed!!\n");
					fclose(gbFile_RRJ);

					return(hr_RRJ);
				}
				else{
					fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
					fprintf_s(gbFile_RRJ, "SUCCESS: Resize() in WM_SIZE Done!!\n");
					fclose(gbFile_RRJ);
				}
			}
			break;

		case WM_KEYDOWN:
			switch(wParam){
				case VK_ESCAPE:
					DestroyWindow(hwnd);
					break;

				case 'F':
				case 'f':
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


void ToggleFullScreen(void){


	MONITORINFO mi = {sizeof(MONITORINFO)};

	if(bIsFullScreen_RRJ == false){
		dwStyle_RRJ = GetWindowLong(ghwnd_RRJ, GWL_STYLE);
		if(dwStyle_RRJ & WS_OVERLAPPEDWINDOW){

			if(GetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ) && GetMonitorInfo(MonitorFromWindow(ghwnd_RRJ, MONITORINFOF_PRIMARY), &mi)){

				SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd_RRJ, HWND_TOP,
					mi.rcMonitor.left,
					mi.rcMonitor.top,
					(mi.rcMonitor.right - mi.rcMonitor.left),
					(mi.rcMonitor.bottom - mi.rcMonitor.top),
					SWP_NOZORDER | SWP_FRAMECHANGED);

				bIsFullScreen_RRJ = true;
				ShowCursor(FALSE);
			}
		}
	}
	else{
		SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ);
		SetWindowPos(ghwnd_RRJ, HWND_TOP,
			0, 0, 0 ,0,
			SWP_NOZORDER | SWP_NOSIZE | SWP_NOMOVE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);

		bIsFullScreen_RRJ = false;
		ShowCursor(TRUE);
	}
}


HRESULT initialize(void){

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
	UINT numOfDrivers_RRJ = 0;
	UINT numOfFeatureLevels = 1;

	numOfDrivers_RRJ = sizeof(d3dDriverTypes_RRJ) / sizeof(d3dDriverTypes_RRJ[0]);


	DXGI_SWAP_CHAIN_DESC dxgiSwapChainDesc_RRJ;
	ZeroMemory((void*)&dxgiSwapChainDesc_RRJ, sizeof(DXGI_SWAP_CHAIN_DESC));

	dxgiSwapChainDesc_RRJ.BufferCount = 1;

	dxgiSwapChainDesc_RRJ.BufferDesc.Width = WIN_WIDTH;
	dxgiSwapChainDesc_RRJ.BufferDesc.Height = WIN_HEIGHT;
	dxgiSwapChainDesc_RRJ.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	dxgiSwapChainDesc_RRJ.BufferDesc.RefreshRate.Numerator = 60;
	dxgiSwapChainDesc_RRJ.BufferDesc.RefreshRate.Denominator = 1;
	dxgiSwapChainDesc_RRJ.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;

	dxgiSwapChainDesc_RRJ.Windowed = TRUE;
	dxgiSwapChainDesc_RRJ.OutputWindow = ghwnd_RRJ;

	dxgiSwapChainDesc_RRJ.SampleDesc.Count = 1;
	dxgiSwapChainDesc_RRJ.SampleDesc.Quality = 0;


	for(UINT driverIndex_RRJ = 0; driverIndex_RRJ < numOfDrivers_RRJ; driverIndex_RRJ++){

		d3dDriverType_RRJ = d3dDriverTypes_RRJ[driverIndex_RRJ];
		hr_RRJ = D3D11CreateDeviceAndSwapChain(
			NULL,
			d3dDriverType_RRJ,
			NULL,
			createDeviceFlags_RRJ,
			&d3dFeatureLevel_required_RRJ,
			numOfFeatureLevels,
			D3D11_SDK_VERSION,
			&dxgiSwapChainDesc_RRJ,
			&gpIDXGISwapChain_RRJ,
			&gpID3D11Device_RRJ,
			&d3dFeatureLevel_acquired_RRJ,
			&gpID3D11DeviceContext_RRJ);

		if(SUCCEEDED(hr_RRJ))
			break;
	}

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: D3D11CreateDeviceAndSwapChain() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{

		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");	
		fprintf_s(gbFile_RRJ, "SUCCESS: D3D11CreateDeviceAndSwapChain() Failed!!\n");

		fprintf_s(gbFile_RRJ, "SUCCESS: Driver Type: ");
		if(d3dDriverType_RRJ == D3D_DRIVER_TYPE_HARDWARE)
			fprintf_s(gbFile_RRJ, "Hardware !!\n");
		else if(d3dDriverType_RRJ == D3D_DRIVER_TYPE_WARP)
			fprintf_s(gbFile_RRJ, "WARP !!\n");
		else if(d3dDriverType_RRJ == D3D_DRIVER_TYPE_REFERENCE)
			fprintf_s(gbFile_RRJ, "Reference !!\n");
		else
			fprintf_s(gbFile_RRJ, "Unknown !!\n");

		fprintf_s(gbFile_RRJ, "\n");

		fprintf_s(gbFile_RRJ, "SUCCESS: Feature Level: ");
		if(d3dFeatureLevel_acquired_RRJ == D3D_FEATURE_LEVEL_11_0)
			fprintf_s(gbFile_RRJ, "11.0 !!\n");
		else if(d3dFeatureLevel_acquired_RRJ == D3D_FEATURE_LEVEL_10_1)
			fprintf_s(gbFile_RRJ, "10.1 !!\n");
		else if(d3dFeatureLevel_acquired_RRJ == D3D_FEATURE_LEVEL_10_0)
			fprintf_s(gbFile_RRJ, "10.0 !!\n");
		else
			fprintf_s(gbFile_RRJ, "Unknown !!\n");

		fprintf_s(gbFile_RRJ, "\n");
		fclose(gbFile_RRJ);
	}


	/********** VERTEX SHADER **********/
	const char *vertexShaderSourceCode_RRJ = 
		"cbuffer ConstantBuffer {" \
			"float4x4 worldViewProjectionMatrix;" \
		"};" \

		"struct Vertex_Output {"\
			"float4 position : SV_POSITION;" \
			"float4 color : COLOR;" \
		"};" \

		"Vertex_Output main(float4 pos : POSITION, float4 col : COLOR) {" \
			"Vertex_Output v;" \
			"v.position = mul(worldViewProjectionMatrix, pos);" \
			"v.color = col;" \
			"return(v);" \

		"}";

	ID3DBlob *pID3DBlob_VertexShaderCode_RRJ = NULL;	 
	ID3DBlob *pID3DBlob_Error_RRJ = NULL;

	hr_RRJ = D3DCompile(vertexShaderSourceCode_RRJ,
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

	if(FAILED(hr_RRJ)){

		if(pID3DBlob_Error_RRJ != NULL){
			fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "ERROR: Vertex Shader: D3DCompile() Failed!!\n");
			fprintf_s(gbFile_RRJ, "%s\n", (char*)pID3DBlob_Error_RRJ->GetBufferPointer());
			fclose(gbFile_RRJ);

			return(hr_RRJ);
		}
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Vertex Shader: D3DCompile() Done!!\n");
		fclose(gbFile_RRJ);
	}


	hr_RRJ = gpID3D11Device_RRJ->CreateVertexShader(
		pID3DBlob_VertexShaderCode_RRJ->GetBufferPointer(),
		pID3DBlob_VertexShaderCode_RRJ->GetBufferSize(),
		NULL,
		&gpID3D11VertexShader_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Vertex Shader: ID3D11Device::CreateVertexShader() Failed !!\n");
		fclose(gbFile_RRJ);

		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Vertex Shader: ID3D11Device::CreateVertexShader() Done!!\n");
		fclose(gbFile_RRJ);
	}


	gpID3D11DeviceContext_RRJ->VSSetShader(gpID3D11VertexShader_RRJ, NULL, NULL);



	/********** PIXEL SHADER **********/
	const char *pixelShaderSourceCode_RRJ = 
		"float4 main(float4 pos : SV_POSITION, float4 col : COLOR) : SV_TARGET { " \
			"return(col); "\
		"}";

	ID3DBlob *pID3DBlob_PixelShaderCode_RRJ = NULL;
	pID3DBlob_Error_RRJ = NULL;

	hr_RRJ = D3DCompile(pixelShaderSourceCode_RRJ,
		lstrlenA(pixelShaderSourceCode_RRJ) + 1,
		"PS",
		NULL,
		D3D_COMPILE_STANDARD_FILE_INCLUDE,
		"main",
		"ps_5_0",
		0, 0,
		&pID3DBlob_PixelShaderCode_RRJ,
		&pID3DBlob_Error_RRJ);

	if(FAILED(hr_RRJ)){

		if(pID3DBlob_Error_RRJ != NULL){
			fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "ERROR: Pixel Shader: D3DCompile() Failed!!\n");
			fprintf_s(gbFile_RRJ, "%s\n", (char*)pID3DBlob_Error_RRJ->GetBufferPointer());
			fclose(gbFile_RRJ);

			return(hr_RRJ);
		}
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Pixel Shader: D3DCompile() Done!!\n");
		fclose(gbFile_RRJ);
	}


	hr_RRJ = gpID3D11Device_RRJ->CreatePixelShader(
		pID3DBlob_PixelShaderCode_RRJ->GetBufferPointer(),
		pID3DBlob_PixelShaderCode_RRJ->GetBufferSize(),
		NULL,
		&gpID3D11PixelShader_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Pixel Shader: ID3D11Device::CreatePixelShader() Failed !!\n");
		fclose(gbFile_RRJ);

		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Pixel Shader: ID3D11Device::CreatePixelShader() Done!!\n");
		fclose(gbFile_RRJ);
	}


	gpID3D11DeviceContext_RRJ->PSSetShader(gpID3D11PixelShader_RRJ, NULL, NULL);


	/********** Input Layout **********/
	D3D11_INPUT_ELEMENT_DESC inputElementDesc_RRJ[2];
	ZeroMemory((void*)inputElementDesc_RRJ, sizeof(D3D11_INPUT_ELEMENT_DESC));


	//Position
	inputElementDesc_RRJ[0].SemanticName = "POSITION";
	inputElementDesc_RRJ[0].SemanticIndex = 0;
	inputElementDesc_RRJ[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	inputElementDesc_RRJ[0].InputSlot = 0;
	inputElementDesc_RRJ[0].AlignedByteOffset = 0;
	inputElementDesc_RRJ[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	inputElementDesc_RRJ[0].InstanceDataStepRate = 0;

	//Color
	inputElementDesc_RRJ[1].SemanticName = "COLOR";
	inputElementDesc_RRJ[1].SemanticIndex = 0;
	inputElementDesc_RRJ[1].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	inputElementDesc_RRJ[1].InputSlot = 1;
	inputElementDesc_RRJ[1].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	inputElementDesc_RRJ[1].AlignedByteOffset = 0;
	inputElementDesc_RRJ[1].InstanceDataStepRate = 0;


	hr_RRJ = gpID3D11Device_RRJ->CreateInputLayout(inputElementDesc_RRJ, 2,
		pID3DBlob_VertexShaderCode_RRJ->GetBufferPointer(),
		pID3DBlob_VertexShaderCode_RRJ->GetBufferSize(),
		&gpID3D11InputLayout_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "\n");
		fprintf_s(gbFile_RRJ, "ERROR: Input Layout: ID3D11Device::CreateInputLayout() Failed!!\n");
		fclose(gbFile_RRJ);

		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "\n");
		fprintf_s(gbFile_RRJ, "SUCCESS: Input Layout: ID3D11Device::CreateInputLayout() Done!!\n");
		fclose(gbFile_RRJ);
	}


	gpID3D11DeviceContext_RRJ->IASetInputLayout(gpID3D11InputLayout_RRJ);

	pID3DBlob_PixelShaderCode_RRJ->Release();
	pID3DBlob_PixelShaderCode_RRJ = NULL;

	pID3DBlob_VertexShaderCode_RRJ->Release();
	pID3DBlob_VertexShaderCode_RRJ = NULL;




	/********** Position and Color **********/
	float tri_Position[] = {
		0.0f, 1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f, 
	};

	float tri_Color[] = {
		1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 1.0f, 0.0f,
	};

	float rect_Position[] = {
		-1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
	};

	float rect_Color[] = {
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
	};



	/********** Vertex Buffers **********/

	/********** TRIANGLE POSITION BUFFER **********/
	D3D11_BUFFER_DESC bufferDesc_VertexBuffer_Tri_Position_RRJ;
	ZeroMemory((void*)&bufferDesc_VertexBuffer_Tri_Position_RRJ, sizeof(D3D11_BUFFER_DESC));

	bufferDesc_VertexBuffer_Tri_Position_RRJ.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc_VertexBuffer_Tri_Position_RRJ.ByteWidth = sizeof(float) * _ARRAYSIZE(tri_Position);
	bufferDesc_VertexBuffer_Tri_Position_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc_VertexBuffer_Tri_Position_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_VertexBuffer_Tri_Position_RRJ, NULL,
		&gpID3D11Buffer_VertexBuffer_Tri_Position_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "\n");
		fprintf_s(gbFile_RRJ, "ERROR: Triangle Position: ID3D11Device::CreateBuffer() Failed!!\n");
		fclose(gbFile_RRJ);

		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "\n");
		fprintf_s(gbFile_RRJ, "SUCCESS: Triangle Position: ID3D11Device::CreateBuffer() Done!!\n");
		fclose(gbFile_RRJ);
	}


	/********** TRIANGLE POSITION MAPPING **********/
	D3D11_MAPPED_SUBRESOURCE mappedSubresource_Tri_Position_RRJ;
	ZeroMemory((void*)&mappedSubresource_Tri_Position_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));

	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_Tri_Position_RRJ, 
		0, D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource_Tri_Position_RRJ);
	memcpy(mappedSubresource_Tri_Position_RRJ.pData, tri_Position, sizeof(tri_Position));
	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_Tri_Position_RRJ, 0);



	/********** TRIANGLE COLOR BUFFER **********/
	D3D11_BUFFER_DESC bufferDesc_VertexBuffer_Tri_Color_RRJ;
	ZeroMemory((void*)&bufferDesc_VertexBuffer_Tri_Color_RRJ, sizeof(D3D11_BUFFER_DESC));

	bufferDesc_VertexBuffer_Tri_Color_RRJ.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc_VertexBuffer_Tri_Color_RRJ.ByteWidth = sizeof(float) * _ARRAYSIZE(tri_Color);
	bufferDesc_VertexBuffer_Tri_Color_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc_VertexBuffer_Tri_Color_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_VertexBuffer_Tri_Color_RRJ, NULL,
		&gpID3D11Buffer_VertexBuffer_Tri_Color_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "\n");
		fprintf_s(gbFile_RRJ, "ERROR: Triangle Color: ID3D11Device::CreateBuffer() Failed!!\n");
		fclose(gbFile_RRJ);

		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "\n");
		fprintf_s(gbFile_RRJ, "SUCCESS: Triangle Color: ID3D11Device::CreateBuffer() Done!!\n");
		fclose(gbFile_RRJ);
	}


	/********** TRIANGLE COLOR MAPPING **********/
	D3D11_MAPPED_SUBRESOURCE mappedSubresource_Tri_Color_RRJ;
	ZeroMemory((void*)&mappedSubresource_Tri_Color_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));

	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_Tri_Color_RRJ, 0,
		D3D11_MAP_WRITE_DISCARD, 0,  &mappedSubresource_Tri_Color_RRJ);
	memcpy(mappedSubresource_Tri_Color_RRJ.pData, tri_Color, sizeof(tri_Color));
	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_Tri_Color_RRJ, 0);





	/********** RECTANGLE POSITION BUFFER **********/
	D3D11_BUFFER_DESC bufferDesc_VertexBuffer_Rect_Position_RRJ;
	ZeroMemory((void*)&bufferDesc_VertexBuffer_Rect_Position_RRJ, sizeof(D3D11_BUFFER_DESC));

	bufferDesc_VertexBuffer_Rect_Position_RRJ.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc_VertexBuffer_Rect_Position_RRJ.ByteWidth = sizeof(float) * _ARRAYSIZE(rect_Position);
	bufferDesc_VertexBuffer_Rect_Position_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc_VertexBuffer_Rect_Position_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_VertexBuffer_Rect_Position_RRJ, NULL,
		&gpID3D11Buffer_VertexBuffer_Rect_Position_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "\n");
		fprintf_s(gbFile_RRJ, "ERROR: Rectangle Position: ID3D11Device::CreateBuffer() Failed!!\n");
		fclose(gbFile_RRJ);

		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "\n");
		fprintf_s(gbFile_RRJ, "SUCCESS: Rectangle Position: ID3D11Device::CreateBuffer() Done!!\n");
		fclose(gbFile_RRJ);
	}


	/********** RECTANGLE POSITION MAPPING **********/
	D3D11_MAPPED_SUBRESOURCE mappedSubresource_Rect_Postion_RRJ;
	ZeroMemory((void*)&mappedSubresource_Rect_Postion_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));

	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_Rect_Position_RRJ, 0,
		D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource_Rect_Postion_RRJ);
	memcpy(mappedSubresource_Rect_Postion_RRJ.pData, rect_Position, sizeof(rect_Position));
	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_Rect_Position_RRJ, 0);



	/********** RECTANGLE COLOR BUFFER **********/
	D3D11_BUFFER_DESC bufferDesc_VertexBuffer_Rect_Color_RRJ;
	ZeroMemory((void*)&bufferDesc_VertexBuffer_Rect_Color_RRJ, sizeof(D3D11_BUFFER_DESC));

	bufferDesc_VertexBuffer_Rect_Color_RRJ.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc_VertexBuffer_Rect_Color_RRJ.ByteWidth = sizeof(float) * _ARRAYSIZE(rect_Color);
	bufferDesc_VertexBuffer_Rect_Color_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc_VertexBuffer_Rect_Color_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_VertexBuffer_Rect_Color_RRJ, NULL,
		&gpID3D11Buffer_VertexBuffer_Rect_Color_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "\n");
		fprintf_s(gbFile_RRJ, "ERROR: Rectangle Color: ID3D11Device::CreateBuffer() Failed!!\n");
		fclose(gbFile_RRJ);

		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "\n");
		fprintf_s(gbFile_RRJ, "SUCCESS: Rectangle Color: ID3D11Device::CreateBuffer() Done!!\n");
		fclose(gbFile_RRJ);
	}


	/********** RECTANGLE COLOR MAPPING **********/
	D3D11_MAPPED_SUBRESOURCE mappedSubresource_Rect_Color_RRJ;
	ZeroMemory((void*)&mappedSubresource_Rect_Color_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));

	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_Rect_Color_RRJ, 0,
		D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource_Rect_Color_RRJ);
	memcpy(mappedSubresource_Rect_Color_RRJ.pData, rect_Color, sizeof(rect_Color));
	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_Rect_Color_RRJ, 0);




	/********** CONSTANT BUFFER **********/
	D3D11_BUFFER_DESC bufferDesc_ConstantBuffer_RRJ;
	ZeroMemory((void*)&bufferDesc_ConstantBuffer_RRJ, sizeof(D3D11_BUFFER_DESC));

	bufferDesc_ConstantBuffer_RRJ.Usage = D3D11_USAGE_DEFAULT;
	bufferDesc_ConstantBuffer_RRJ.ByteWidth = sizeof(CBUFFER);
	bufferDesc_ConstantBuffer_RRJ.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	bufferDesc_ConstantBuffer_RRJ.CPUAccessFlags = 0;

	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_ConstantBuffer_RRJ, NULL,
		&gpID3D11Buffer_ConstantBuffer_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "\n");
		fprintf_s(gbFile_RRJ, "ERROR: Constant Buffer: ID3D11Device::CreateBuffer() Failed!!\n");
		fclose(gbFile_RRJ);

		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "\n");
		fprintf_s(gbFile_RRJ, "SUCCESS: Constant Buffer: ID3D11Device::CreateBuffer() Done!!\n");
		fclose(gbFile_RRJ);
	}	


	gpID3D11DeviceContext_RRJ->VSSetConstantBuffers(0, 1, &gpID3D11Buffer_ConstantBuffer_RRJ);


	/********** For ClearColor **********/
	gClearColor[0] = 0.0f;
	gClearColor[1] = 0.0f;
	gClearColor[2] = 0.0f;
	gClearColor[3] = 1.0f;

	hr_RRJ =resize(WIN_WIDTH, WIN_HEIGHT);
	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "\n");
		fprintf_s(gbFile_RRJ, "ERROR: Warmup Resize() Failed!!\n");
		fclose(gbFile_RRJ);

		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "\n");
		fprintf_s(gbFile_RRJ, "SUCCESS:  Warmup Resize() Done!!\n");
		fclose(gbFile_RRJ);
	}

	gPerspectiveProjectionMatrix_RRJ = XMMatrixIdentity();
	return(hr_RRJ);
}

void uninitialize(void){


	if(gpID3D11Buffer_ConstantBuffer_RRJ){
		gpID3D11Buffer_ConstantBuffer_RRJ->Release();
		gpID3D11Buffer_ConstantBuffer_RRJ = NULL;
	}

	if(gpID3D11Buffer_VertexBuffer_Rect_Color_RRJ){
		gpID3D11Buffer_VertexBuffer_Rect_Color_RRJ->Release();
		gpID3D11Buffer_VertexBuffer_Rect_Color_RRJ = NULL;
	}

	if(gpID3D11Buffer_VertexBuffer_Rect_Position_RRJ){
		gpID3D11Buffer_VertexBuffer_Rect_Position_RRJ->Release();
		gpID3D11Buffer_VertexBuffer_Rect_Position_RRJ =NULL;
	}

	if(gpID3D11Buffer_VertexBuffer_Tri_Color_RRJ){
		gpID3D11Buffer_VertexBuffer_Tri_Color_RRJ->Release();
		gpID3D11Buffer_VertexBuffer_Tri_Color_RRJ = NULL;
	}

	if(gpID3D11Buffer_VertexBuffer_Tri_Position_RRJ){
		gpID3D11Buffer_VertexBuffer_Tri_Position_RRJ->Release();
		gpID3D11Buffer_VertexBuffer_Tri_Position_RRJ = NULL;
	}

	if(gpID3D11InputLayout_RRJ){
		gpID3D11InputLayout_RRJ->Release();
		gpID3D11InputLayout_RRJ = NULL;
	}

	if(gpID3D11RenderTargetView_RRJ){
		gpID3D11RenderTargetView_RRJ->Release();
		gpID3D11RenderTargetView_RRJ = NULL;
	}

	if(gpIDXGISwapChain_RRJ){
		gpIDXGISwapChain_RRJ->Release();
		gpIDXGISwapChain_RRJ = NULL;
	}

	if(gpID3D11DeviceContext_RRJ){
		gpID3D11DeviceContext_RRJ->Release();
		gpID3D11DeviceContext_RRJ = NULL;
	}

	if(gpID3D11Device_RRJ){
		gpID3D11Device_RRJ->Release();
		gpID3D11Device_RRJ = NULL;
	}

	if(gbFile_RRJ){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Log Close!!\n");
		fprintf_s(gbFile_RRJ, "SUCCESS: END!!\n");
		fclose(gbFile_RRJ);
		gbFile_RRJ = NULL;
	}
}

HRESULT resize(int width, int height){


	HRESULT hr_RRJ;

	if(gpID3D11RenderTargetView_RRJ){
		gpID3D11RenderTargetView_RRJ->Release();
		gpID3D11RenderTargetView_RRJ = NULL;
	}


	gpIDXGISwapChain_RRJ->ResizeBuffers(1, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, 0);

	ID3D11Texture2D *pID3D11Texture2D_BackBuffer_RRJ = NULL;
	gpIDXGISwapChain_RRJ->GetBuffer(0, __uuidof(ID3D11Texture2D), 
		(LPVOID*)&pID3D11Texture2D_BackBuffer_RRJ);


	hr_RRJ = gpID3D11Device_RRJ->CreateRenderTargetView(pID3D11Texture2D_BackBuffer_RRJ, NULL,
		&gpID3D11RenderTargetView_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "\n");
		fprintf_s(gbFile_RRJ, "ERROR: ID3D11RenderTargetView::CreateRenderTargetView() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "\n");
		fprintf_s(gbFile_RRJ, "SUCCESS: ID3D11RenderTargetView::CreateRenderTargetView() Done!!\n");
		fclose(gbFile_RRJ);
	}


	pID3D11Texture2D_BackBuffer_RRJ->Release();
	pID3D11Texture2D_BackBuffer_RRJ = NULL;

	gpID3D11DeviceContext_RRJ->OMSetRenderTargets(1, &gpID3D11RenderTargetView_RRJ, NULL);


	//ViewPort
	D3D11_VIEWPORT d3dViewPort;
	ZeroMemory((void*)&d3dViewPort, sizeof(D3D11_VIEWPORT));

	d3dViewPort.TopLeftX = 0;
	d3dViewPort.TopLeftY = 0;
	d3dViewPort.Width = (float)width;
	d3dViewPort.Height = (float)height;
	d3dViewPort.MinDepth = 0.0f;
	d3dViewPort.MaxDepth = 1.0f;

	gpID3D11DeviceContext_RRJ->RSSetViewports(1, &d3dViewPort);

	gPerspectiveProjectionMatrix_RRJ = XMMatrixPerspectiveFovLH(
		XMConvertToRadians(45.0f), (float)width / (float)height, 0.1f, 100.0f);

	return(S_OK);
}


void display(void){


	static float angle_Tri_RRJ = 0.0f;
	static float angle_Rect_RRJ = 360.0f;

	gpID3D11DeviceContext_RRJ->ClearRenderTargetView(gpID3D11RenderTargetView_RRJ, gClearColor);


	XMMATRIX translateMatrix_RRJ;
	XMMATRIX rotateMatrix_RRJ;
	XMMATRIX scaleMatrix_RRJ;
	XMMATRIX worldViewMatrix_RRJ;
	XMMATRIX worldViewProjectionMatrix_RRJ;


	/********** TRIANGLE **********/
	translateMatrix_RRJ = XMMatrixIdentity();
	rotateMatrix_RRJ = XMMatrixIdentity();
	scaleMatrix_RRJ = XMMatrixIdentity();
	worldViewMatrix_RRJ = XMMatrixIdentity();
	worldViewProjectionMatrix_RRJ = XMMatrixIdentity();

	UINT stride = sizeof(float) * 3;
	UINT offset = 0;

	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(0, 1, &gpID3D11Buffer_VertexBuffer_Tri_Position_RRJ, &stride, &offset);

	stride = sizeof(float) * 3;
	offset = 0;

	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(1, 1, &gpID3D11Buffer_VertexBuffer_Tri_Color_RRJ, &stride, &offset);


	gpID3D11DeviceContext_RRJ->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);


	translateMatrix_RRJ = XMMatrixTranslation(-2.0f, 0.0f, 6.0f);
	rotateMatrix_RRJ = XMMatrixRotationY(-angle_Tri_RRJ);
	

	worldViewMatrix_RRJ = rotateMatrix_RRJ * translateMatrix_RRJ;
	worldViewProjectionMatrix_RRJ = worldViewMatrix_RRJ * gPerspectiveProjectionMatrix_RRJ;

	CBUFFER constantBuffer;
	constantBuffer.WorldViewProjectionMatrix = worldViewProjectionMatrix_RRJ;
	gpID3D11DeviceContext_RRJ->UpdateSubresource(gpID3D11Buffer_ConstantBuffer_RRJ, 0,
		NULL, &constantBuffer, 0, 0);


	gpID3D11DeviceContext_RRJ->Draw(3, 0);



	/********** RECTANGLE **********/
	translateMatrix_RRJ = XMMatrixIdentity();
	rotateMatrix_RRJ = XMMatrixIdentity();
	scaleMatrix_RRJ = XMMatrixIdentity();
	worldViewMatrix_RRJ = XMMatrixIdentity();
	worldViewProjectionMatrix_RRJ = XMMatrixIdentity();

	 stride = sizeof(float) * 3;
	 offset = 0;

	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(0, 1, &gpID3D11Buffer_VertexBuffer_Rect_Position_RRJ, &stride, &offset);

	stride = sizeof(float) * 3;
	offset = 0;

	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(1, 1, &gpID3D11Buffer_VertexBuffer_Rect_Color_RRJ, &stride, &offset);


	gpID3D11DeviceContext_RRJ->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);


	translateMatrix_RRJ = XMMatrixTranslation(2.0f, 0.0f, 6.0f);
	rotateMatrix_RRJ = XMMatrixRotationX(-angle_Rect_RRJ);
	scaleMatrix_RRJ = XMMatrixScaling(0.9f, 0.9f, 0.9f);

	worldViewMatrix_RRJ = rotateMatrix_RRJ * scaleMatrix_RRJ * translateMatrix_RRJ ;
	worldViewProjectionMatrix_RRJ = worldViewMatrix_RRJ * gPerspectiveProjectionMatrix_RRJ;


	constantBuffer.WorldViewProjectionMatrix = worldViewProjectionMatrix_RRJ;
	gpID3D11DeviceContext_RRJ->UpdateSubresource(gpID3D11Buffer_ConstantBuffer_RRJ, 0,
		NULL, &constantBuffer, 0, 0);


	gpID3D11DeviceContext_RRJ->Draw(4, 0);

	gpIDXGISwapChain_RRJ->Present(0, 0);


	angle_Rect_RRJ = angle_Rect_RRJ - 0.005f;
	angle_Tri_RRJ = angle_Tri_RRJ + 0.005f;

	if(angle_Tri_RRJ > 360.0f)
		angle_Tri_RRJ = 0.0f;

	if(angle_Rect_RRJ < 0.0f)
		angle_Rect_RRJ = 360.0f;

}