#define UNICODE
#include<windows.h>
#include<stdio.h>
#include<d3d11.h>
#include<d3dcompiler.h>

#include"WICTextureLoader.h"

#pragma warning(disable: 4838)
#include"XNAMATH/xnamath.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "D3dcompiler.lib")
#pragma comment(lib, "DirectXTK.lib")


#define WIN_WIDTH 800
#define WIN_HEIGHT 600


using namespace DirectX;

//For FullScreen
bool bIsFullScreen_RRJ = false;
WINDOWPLACEMENT wpPrev_RRJ = {sizeof(WINDOWPLACEMENT)};
DWORD dwStyle_RRJ;
HWND ghwnd_RRJ = NULL;
bool bActiveWindow_RRJ = false;

//For Error
FILE *gbFile_RRJ = NULL;
const char *gszLogFileName = "Log.txt";


//For DirectX
IDXGISwapChain *gpIDXGISwapChain_RRJ = NULL;
ID3D11Device *gpID3D11Device_RRJ = NULL;
ID3D11DeviceContext *gpID3D11DeviceContext_RRJ = NULL;
ID3D11RenderTargetView *gpID3D11RenderTargetView_RRJ = NULL;
ID3D11DepthStencilView *gpID3D11DepthStencilView_RRJ = NULL;

//For Pyramid Buffers
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Pyramid_Position_RRJ = NULL;
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Pyramid_Texcoord_RRJ = NULL;
float angle_Pyramid_RRJ = 0.0f;


//For Shader
ID3D11VertexShader *gpID3D11VertexShader_RRJ = NULL;
ID3D11PixelShader *gpID3D11PixelShader_RRJ = NULL;


//For Culling
ID3D11RasterizerState *gpID3D11RasterizerState_RRJ = NULL;

//For Uniform
ID3D11Buffer *gpID3D11Buffer_ConstantBuffer_RRJ = NULL;

//For Input Layout
ID3D11InputLayout *gpID3D11InputLayout_RRJ = NULL;

//For Texture
ID3D11SamplerState *gpID3D11SamplerState_Pyramid_Texture_RRJ = NULL;
ID3D11ShaderResourceView *gpID3D11ShaderResourceView_Pyramid_Texture_RRJ = NULL;


struct CBUFFER{
	XMMATRIX worldViewProjectionMatrix;
};


float gClearColor[4];


XMMATRIX gPerspectiveProjectionMatrix_RRJ;

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow){


	HRESULT initialize(void);
	void display(void);
	void update(void);

	HRESULT hr_RRJ = S_OK;
	bool bDone_RRJ = false;


	fopen_s(&gbFile_RRJ, gszLogFileName, "w");
	if(gbFile_RRJ == NULL){
		MessageBox(NULL, TEXT("Log Creation Failed!!"), TEXT("Error"), MB_OK);
		exit(0);
	}
	else{
		fprintf_s(gbFile_RRJ, "SUCCESS: Log Created!!\n");
		fclose(gbFile_RRJ);
	}




	WNDCLASSEX wndclass_RRJ;
	HWND hwnd_RRJ;
	MSG msg_RRJ;
	TCHAR szName_RRJ[] = TEXT("Rohit_R_Jadhav-D3D-14-Texture-Pyramid");



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
		TEXT("Rohit_R_Jadhav-D3D-14-Texture-Pyramid"),
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
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: WinMain: Initialize() Failed!!\n");
		fclose(gbFile_RRJ);

		DestroyWindow(hwnd_RRJ);
		hwnd_RRJ = NULL;
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: WinMain: Initialize() Done!!\n");
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
			if(bActiveWindow_RRJ == true)
				update();
			display();
		}
	}

	return((int)msg_RRJ.wParam); 			
}


LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam){

	void uninitialize(void);
	void ToggleFullScreen(void);
	HRESULT resize(int, int);

	HRESULT hr_RRJ = S_OK;

	switch(iMsg){
		case WM_ACTIVATE:
			if(HIWORD(lParam) == 0)
				bActiveWindow_RRJ = true;
			else
				bActiveWindow_RRJ = false;
			break;

		case WM_ERASEBKGND:
			return(0);

		case WM_KEYDOWN:
			switch(wParam){
				case VK_ESCAPE:
					uninitialize();
					break;

				case 'F':
				case 'f':
					ToggleFullScreen();
					break;

				default:
					break;
			}
			break;


		case WM_SIZE:
			if(gpID3D11DeviceContext_RRJ){
				hr_RRJ = resize(LOWORD(lParam), HIWORD(lParam));

				if(FAILED(hr_RRJ)){
					fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
					fprintf_s(gbFile_RRJ, "ERROR: WndProc: Resize() Failed!!\n");
					fclose(gbFile_RRJ);

					DestroyWindow(hwnd);
					hwnd = NULL;
				}
				else{
					fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
					fprintf_s(gbFile_RRJ, "SUCCESS: WndProc: Resize() Done!!\n");
					fclose(gbFile_RRJ);
				}		
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

	MONITORINFO mi_RRJ = {sizeof(MONITORINFO)};

	if(bIsFullScreen_RRJ == false){
		dwStyle_RRJ = GetWindowLong(ghwnd_RRJ, GWL_STYLE);
		if(dwStyle_RRJ & WS_OVERLAPPEDWINDOW){
			if(GetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ)  && GetMonitorInfo(MonitorFromWindow(ghwnd_RRJ, MONITORINFOF_PRIMARY), &mi_RRJ)){
				SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd_RRJ, HWND_TOP,
					mi_RRJ.rcMonitor.left,
					mi_RRJ.rcMonitor.top,
					(mi_RRJ.rcMonitor.right - mi_RRJ.rcMonitor.left),
					(mi_RRJ.rcMonitor.bottom - mi_RRJ.rcMonitor.top),
					SWP_NOZORDER | SWP_FRAMECHANGED);

				ShowCursor(FALSE);
				bIsFullScreen_RRJ = true;
			}
		}
	}
	else{
		SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ);
		SetWindowPos(ghwnd_RRJ, HWND_TOP,
			0, 0, 0, 0,
			SWP_NOZORDER | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);

		ShowCursor(TRUE);
		bIsFullScreen_RRJ = false;
	}
}


HRESULT initialize(void){

	void uninitialize(void);
	HRESULT resize(int, int);
	HRESULT LoadD3DTexture(const wchar_t*, ID3D11ShaderResourceView**);

	HRESULT hr_RRJ = S_OK;

	D3D_DRIVER_TYPE d3dDriverType_RRJ;
	D3D_DRIVER_TYPE d3dDriverTypes_RRJ[] = {
		D3D_DRIVER_TYPE_HARDWARE,
		D3D_DRIVER_TYPE_WARP,
		D3D_DRIVER_TYPE_REFERENCE,
	};

	D3D_FEATURE_LEVEL d3dFeatureLevel_required_RRJ = D3D_FEATURE_LEVEL_11_0;
	D3D_FEATURE_LEVEL d3dFeatureLevel_acquired_RRJ = D3D_FEATURE_LEVEL_10_0;

	UINT createDeviceFlags_RRJ = 0;
	UINT numOfDevices_RRJ = 0;
	UINT numOfFeatureLevels_RRJ = 1;

	numOfDevices_RRJ = sizeof(d3dDriverTypes_RRJ) / sizeof(d3dDriverTypes_RRJ[0]);

	DXGI_SWAP_CHAIN_DESC dxgiSwapChainDesc_RRJ;
	ZeroMemory((void*)&dxgiSwapChainDesc_RRJ, sizeof(DXGI_SWAP_CHAIN_DESC));

	dxgiSwapChainDesc_RRJ.BufferCount = 1;
	dxgiSwapChainDesc_RRJ.BufferDesc.Width = WIN_WIDTH;
	dxgiSwapChainDesc_RRJ.BufferDesc.Height = WIN_HEIGHT;
	dxgiSwapChainDesc_RRJ.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	dxgiSwapChainDesc_RRJ.BufferDesc.RefreshRate.Numerator = 60;
	dxgiSwapChainDesc_RRJ.BufferDesc.RefreshRate.Denominator = 1;
	dxgiSwapChainDesc_RRJ.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;

	dxgiSwapChainDesc_RRJ.SampleDesc.Count = 1;
	dxgiSwapChainDesc_RRJ.SampleDesc.Quality = 0;

	dxgiSwapChainDesc_RRJ.Windowed = TRUE;
	dxgiSwapChainDesc_RRJ.OutputWindow = ghwnd_RRJ;


	for(int indexDriverType_RRJ = 0; indexDriverType_RRJ < numOfDevices_RRJ; indexDriverType_RRJ++){

		d3dDriverType_RRJ = d3dDriverTypes_RRJ[indexDriverType_RRJ];

		hr_RRJ = D3D11CreateDeviceAndSwapChain(
			NULL,
			d3dDriverType_RRJ,
			NULL,
			createDeviceFlags_RRJ,
			&d3dFeatureLevel_required_RRJ,
			numOfFeatureLevels_RRJ,
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
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Inintialize: D3D11CreateDeviceAndSwapChain() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{

		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Inintialize: D3D11CreateDeviceAndSwapChain() Done!!\n");
		
		fprintf_s(gbFile_RRJ, "SUCCESS: Driver Type: ");
		if(d3dDriverType_RRJ == D3D_DRIVER_TYPE_HARDWARE)
			fprintf_s(gbFile_RRJ, "Hardware!!\n");
		else if(d3dDriverType_RRJ == D3D_DRIVER_TYPE_WARP)
			fprintf_s(gbFile_RRJ, "Warp!!\n");
		else if(d3dDriverType_RRJ == D3D_DRIVER_TYPE_REFERENCE)
			fprintf_s(gbFile_RRJ, "Reference!!\n");
		else
			fprintf_s(gbFile_RRJ, "Unknown!!\n");

		fprintf_s(gbFile_RRJ, "SUCCESS: Feature Level: ");
		if(d3dFeatureLevel_acquired_RRJ == D3D_FEATURE_LEVEL_11_0)
			fprintf_s(gbFile_RRJ, "11.0\n");
		else if(d3dFeatureLevel_acquired_RRJ == D3D_FEATURE_LEVEL_10_1)
			fprintf_s(gbFile_RRJ, "10.1\n");
		else if(d3dFeatureLevel_acquired_RRJ == D3D_FEATURE_LEVEL_10_0)
			fprintf_s(gbFile_RRJ, "10.0\n");
		else
			fprintf_s(gbFile_RRJ, "Unknown!!\n");

		if(gpID3D11DeviceContext_RRJ)
			fprintf_s(gbFile_RRJ, "Device Got!!\n");

		fclose(gbFile_RRJ);
	}




	/********** VERTEX SHADER **********/
	const char *vertexShaderSourceCode_RRJ = 
		"cbuffer ConstantBuffer { " \
			"float4x4 worldViewProjection;" \
		"};" \

		"struct vertex_output {" \
			"float4 position : SV_POSITION;" \
			"float2 texcoord : TEXCOORD;" \
		"};" \

		"vertex_output main(float4 pos : POSITION, float2 tex : TEXCOORD){ " \
			"vertex_output v;" \
			"v.position = mul(worldViewProjection, pos); " \
			"v.texcoord = tex;" \
			"return(v); " \
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
		0, 0,
		&pID3DBlob_VertexShaderCode_RRJ,
		&pID3DBlob_Error_RRJ);


	if(FAILED(hr_RRJ)){
		if(pID3DBlob_Error_RRJ != NULL){
			fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
			fprintf_s(gbFile_RRJ, "ERROR: Initialize: Vertex Shader Compilation Error: \n");
			fprintf_s(gbFile_RRJ, "%s\n", (char*)pID3DBlob_Error_RRJ->GetBufferPointer());
			fclose(gbFile_RRJ);
			return(hr_RRJ);
		}
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize: Vertex Shader Compilation Done!!\n");
		fclose(gbFile_RRJ);
	}

	hr_RRJ = gpID3D11Device_RRJ->CreateVertexShader(
		pID3DBlob_VertexShaderCode_RRJ->GetBufferPointer(),
		pID3DBlob_VertexShaderCode_RRJ->GetBufferSize(),
		NULL,
		&gpID3D11VertexShader_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize: Vertex Shader Creation Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize: Vertex Shader Created!!\n");
		fclose(gbFile_RRJ);
	}


	gpID3D11DeviceContext_RRJ->VSSetShader(gpID3D11VertexShader_RRJ, NULL, NULL);


	/********** PIXEL SHADER **********/
	const char *pixelShaderSourceCode_RRJ = 
		"Texture2D myTexture2D;" \
		"SamplerState samplerState;" \

		"float4 main(float4 pos : SV_POSITION, float2 tex: TEXCOORD) : SV_TARGET {" \
			"float4 color = myTexture2D.Sample(samplerState, tex);" \
			"return(color);" \
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

	if(FAILED(hr_RRJ)){
		if(pID3DBlob_Error_RRJ != NULL){
			fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
			fprintf_s(gbFile_RRJ, "ERROR: Initialize: Pixel Shader Compilation Error: \n");
			fprintf_s(gbFile_RRJ, "%s\n", (char*)pID3DBlob_Error_RRJ->GetBufferPointer());
			fclose(gbFile_RRJ);
			return(hr_RRJ);
		}
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize: Pixel Shader Compilation Done!!\n");
		fclose(gbFile_RRJ);
	}


	hr_RRJ = gpID3D11Device_RRJ->CreatePixelShader(
		pID3DBlob_PixelShaderCode_RRJ->GetBufferPointer(),
		pID3DBlob_PixelShaderCode_RRJ->GetBufferSize(),
		NULL,
		&gpID3D11PixelShader_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize: Pixel Shader Creation Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize: Pixel Shader Created!!\n");
		fclose(gbFile_RRJ);
	}


	gpID3D11DeviceContext_RRJ->PSSetShader(gpID3D11PixelShader_RRJ, NULL, NULL);


	


	/********** INPUT LAYOUT **********/
	D3D11_INPUT_ELEMENT_DESC inputElementDesc_RRJ[2];
	ZeroMemory(inputElementDesc_RRJ, sizeof(D3D11_INPUT_ELEMENT_DESC));

	inputElementDesc_RRJ[0].SemanticName = "POSITION";
	inputElementDesc_RRJ[0].SemanticIndex = 0;
	inputElementDesc_RRJ[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	inputElementDesc_RRJ[0].InputSlot = 0;
	inputElementDesc_RRJ[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	inputElementDesc_RRJ[0].AlignedByteOffset = 0;
	inputElementDesc_RRJ[0].InstanceDataStepRate = 0;


	inputElementDesc_RRJ[1].SemanticName = "TEXCOORD";
	inputElementDesc_RRJ[1].SemanticIndex = 0;
	inputElementDesc_RRJ[1].Format = DXGI_FORMAT_R32G32_FLOAT;
	inputElementDesc_RRJ[1].InputSlot = 1;
	inputElementDesc_RRJ[1].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	inputElementDesc_RRJ[1].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
	inputElementDesc_RRJ[1].InstanceDataStepRate = 0;


	hr_RRJ = gpID3D11Device_RRJ->CreateInputLayout(inputElementDesc_RRJ, _ARRAYSIZE(inputElementDesc_RRJ),
		pID3DBlob_VertexShaderCode_RRJ->GetBufferPointer(),
		pID3DBlob_VertexShaderCode_RRJ->GetBufferSize(),
		&gpID3D11InputLayout_RRJ);


	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize: ID3D11Device::CreateInputLayout() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize: ID3D11Device::CreateInputLayout() Done!!\n");
		fclose(gbFile_RRJ);
	}


	gpID3D11DeviceContext_RRJ->IASetInputLayout(gpID3D11InputLayout_RRJ);


	pID3DBlob_PixelShaderCode_RRJ->Release();
	pID3DBlob_PixelShaderCode_RRJ = NULL;
	pID3DBlob_VertexShaderCode_RRJ->Release();
	pID3DBlob_VertexShaderCode_RRJ = NULL;


	/********** POSITIOn  And TEXCOORD **********/
	float pyramid_Pos_RRJ[] = {

		//Front
 		0.0f, 1.0f, 0.0f,
 		1.0f, -1.0f, -1.0f,
 		-1.0f, -1.0f, -1.0f,

 		//Right
 		0.0f, 1.0f, 0.0f,
 		1.0f, -1.0f, 1.0f,
 		1.0f, -1.0f, -1.0f,

 		//Back
 		0.0f, 1.0f, 0.0f,
 		-1.0f, -1.0f, 1.0f, 
 		1.0f, -1.0f, 1.0f,

 		//Left
 		0.0f, 1.0f, 0.0f,
 		-1.0f, -1.0f, -1.0f,
 		-1.0f, -1.0f, 1.0f,

	};

	float pyramid_Tex_RRJ[] = {

		//Front
		0.5f, 1.0f,
		0.0f, 0.0f,
		-1.0f, 0.0f,

		//Right
		0.5f, 0.0f,
		1.0f, 0.0f,
		0.0f, 0.0f,

		//Back
		0.5f, 0.0f,
		1.0f, 0.0f,
		0.0f, 0.0f,

		//Left
		0.5f, 1.0f,
		0.0f, 0.0f,
		-1.0f, 0.0f,
	};




	/********** PYRAMID POSITION BUFFER **********/
	D3D11_BUFFER_DESC bufferDesc_RRJ;
	ZeroMemory((void*)&bufferDesc_RRJ, sizeof(D3D11_BUFFER_DESC));

	bufferDesc_RRJ.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc_RRJ.ByteWidth = sizeof(float) * _ARRAYSIZE(pyramid_Pos_RRJ);
	bufferDesc_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_RRJ, NULL,
		&gpID3D11Buffer_VertexBuffer_Pyramid_Position_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize: ID3D11Device::CreateBuffer() Position Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize: ID3D11Device::CreateBuffer() Position Done!!\n");
		fclose(gbFile_RRJ);
	}


	/********** MEMORY MAPPING *********/
	D3D11_MAPPED_SUBRESOURCE mappedSubresouce_RRJ;
	ZeroMemory((void*)&mappedSubresouce_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));

	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_Pyramid_Position_RRJ, NULL,
		D3D11_MAP_WRITE_DISCARD, NULL, &mappedSubresouce_RRJ);
	memcpy(mappedSubresouce_RRJ.pData, pyramid_Pos_RRJ, sizeof(pyramid_Pos_RRJ));
	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_Pyramid_Position_RRJ, NULL);



	/********** PYRAMID TEXCOORD BUFFER **********/
	ZeroMemory((void*)&bufferDesc_RRJ, sizeof(D3D11_BUFFER_DESC));

	bufferDesc_RRJ.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc_RRJ.ByteWidth = sizeof(float) * _ARRAYSIZE(pyramid_Tex_RRJ);
	bufferDesc_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_RRJ, NULL, 
		&gpID3D11Buffer_VertexBuffer_Pyramid_Texcoord_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize: ID3D11Device::CreateBuffer() Texcoord Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize: ID3D11Device::CreateBuffer() Texcoord Done!!\n");
		fclose(gbFile_RRJ);
	}


	/********** MAPPING **********/
	ZeroMemory((void*)&mappedSubresouce_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));
	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_Pyramid_Texcoord_RRJ, NULL,
		D3D11_MAP_WRITE_DISCARD, NULL, &mappedSubresouce_RRJ);
	memcpy(mappedSubresouce_RRJ.pData, pyramid_Tex_RRJ, sizeof(pyramid_Tex_RRJ));
	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_Pyramid_Texcoord_RRJ, NULL);



	/********** CONSTANT BUFFER **********/
	D3D11_BUFFER_DESC bufferDesc_ConstantBuffer_RRJ;
	ZeroMemory((void*)&bufferDesc_ConstantBuffer_RRJ, sizeof(D3D11_BUFFER_DESC));
	bufferDesc_ConstantBuffer_RRJ.Usage = D3D11_USAGE_DEFAULT;
	bufferDesc_ConstantBuffer_RRJ.ByteWidth = sizeof(CBUFFER);
	bufferDesc_ConstantBuffer_RRJ.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_ConstantBuffer_RRJ, NULL,
		&gpID3D11Buffer_ConstantBuffer_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize: ID3D11Device::CreateBuffer() Constant Buffer Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize: ID3D11Device::CreateBuffer() Constant Buffer Done!!\n");
		fclose(gbFile_RRJ);
	}


	gpID3D11DeviceContext_RRJ->VSSetConstantBuffers(0, 1, &gpID3D11Buffer_ConstantBuffer_RRJ);



	/********** FOR CULLING **********/
	D3D11_RASTERIZER_DESC rasterizerDesc_RRJ;
	ZeroMemory((void*)&rasterizerDesc_RRJ, sizeof(D3D11_RASTERIZER_DESC));

	rasterizerDesc_RRJ.AntialiasedLineEnable = FALSE;
	rasterizerDesc_RRJ.CullMode = D3D11_CULL_NONE;
	rasterizerDesc_RRJ.DepthBias = 0;
	rasterizerDesc_RRJ.DepthBiasClamp = 0.0f;
	rasterizerDesc_RRJ.DepthClipEnable = TRUE;
	rasterizerDesc_RRJ.FillMode = D3D11_FILL_SOLID;
	rasterizerDesc_RRJ.FrontCounterClockwise = FALSE;
	rasterizerDesc_RRJ.MultisampleEnable = FALSE;
	rasterizerDesc_RRJ.ScissorEnable = FALSE;
	rasterizerDesc_RRJ.SlopeScaledDepthBias = 0.0f;

	hr_RRJ = gpID3D11Device_RRJ->CreateRasterizerState(&rasterizerDesc_RRJ,
		&gpID3D11RasterizerState_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize: ID3D11Device::CreateRasterizerState() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize: ID3D11Device::CreateRasterizerState() Done!!\n");
		fclose(gbFile_RRJ);
	}

	gpID3D11DeviceContext_RRJ->RSSetState(gpID3D11RasterizerState_RRJ);


	/********** LOAD TEXTURE **********/
	hr_RRJ = LoadD3DTexture(L"Stone.bmp", &gpID3D11ShaderResourceView_Pyramid_Texture_RRJ);
	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize: LoadD3DTexture() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize: LoadD3DTexture() Done!!\n");
		fclose(gbFile_RRJ);
	}


	/********** SAMPLER **********/
	D3D11_SAMPLER_DESC samplerDesc_RRJ;
	ZeroMemory((void*)&samplerDesc_RRJ, sizeof(D3D11_SAMPLER_DESC));

	samplerDesc_RRJ.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	samplerDesc_RRJ.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc_RRJ.AddressV	 = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc_RRJ.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;

	hr_RRJ = gpID3D11Device_RRJ->CreateSamplerState(&samplerDesc_RRJ, 
		&gpID3D11SamplerState_Pyramid_Texture_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize: CreateSamplerState() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize: CreateSamplerState() Done!!\n");
		fclose(gbFile_RRJ);
	}



	gClearColor[0] = 0.0f;
	gClearColor[1] = 0.0f;
	gClearColor[2] = 1.0f;
	gClearColor[3] = 1.0f;

	gPerspectiveProjectionMatrix_RRJ = XMMatrixIdentity();

	hr_RRJ = resize(WIN_WIDTH, WIN_HEIGHT);
	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize: warmup Resize() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize: warmup Resize() Done!!\n");
		fclose(gbFile_RRJ);
	}

	return(S_OK);
}


HRESULT LoadD3DTexture(const wchar_t *fileName, ID3D11ShaderResourceView **ppShaderResourceView){

	HRESULT hr_RRJ = S_OK;


	hr_RRJ = CreateWICTextureFromFile(
		gpID3D11Device_RRJ,
		gpID3D11DeviceContext_RRJ,
		fileName, NULL, ppShaderResourceView);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: %ws\n", fileName);
		fprintf_s(gbFile_RRJ, "ERROR: LoadD3DTexture: DirectX::CreateWICTextureFromFile() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: LoadD3DTexture: DirectX::CreateWICTextureFromFile() Done!!\n");
		fclose(gbFile_RRJ);
	}

	return(S_OK);
}


void uninitialize(void){


	if(gpID3D11SamplerState_Pyramid_Texture_RRJ){
		gpID3D11SamplerState_Pyramid_Texture_RRJ->Release();
		gpID3D11SamplerState_Pyramid_Texture_RRJ = NULL;
	}

	if(gpID3D11ShaderResourceView_Pyramid_Texture_RRJ){
		gpID3D11ShaderResourceView_Pyramid_Texture_RRJ->Release();
		gpID3D11ShaderResourceView_Pyramid_Texture_RRJ = NULL;
	}

	if(gpID3D11RasterizerState_RRJ){
		gpID3D11RasterizerState_RRJ->Release();
		gpID3D11RasterizerState_RRJ = NULL;
	}

	if(gpID3D11Buffer_ConstantBuffer_RRJ){
		gpID3D11Buffer_ConstantBuffer_RRJ->Release();
		gpID3D11Buffer_ConstantBuffer_RRJ = NULL;
	}

	if(gpID3D11Buffer_VertexBuffer_Pyramid_Texcoord_RRJ){
		gpID3D11Buffer_VertexBuffer_Pyramid_Texcoord_RRJ->Release();
		gpID3D11Buffer_VertexBuffer_Pyramid_Texcoord_RRJ = NULL;
	}

	if(gpID3D11Buffer_VertexBuffer_Pyramid_Position_RRJ){
		gpID3D11Buffer_VertexBuffer_Pyramid_Position_RRJ->Release();
		gpID3D11Buffer_VertexBuffer_Pyramid_Position_RRJ = NULL;
	}

	if(gpID3D11InputLayout_RRJ){
		gpID3D11InputLayout_RRJ->Release();
		gpID3D11InputLayout_RRJ = NULL;
	}

	if(gpID3D11PixelShader_RRJ){
		gpID3D11PixelShader_RRJ->Release();
		gpID3D11PixelShader_RRJ = NULL;
	}

	if(gpID3D11VertexShader_RRJ){
		gpID3D11VertexShader_RRJ->Release();
		gpID3D11VertexShader_RRJ = NULL;
	}

	if(gpID3D11DepthStencilView_RRJ){
		gpID3D11DepthStencilView_RRJ->Release();
		gpID3D11DepthStencilView_RRJ = NULL;
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
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Log Close!!\n");
		fprintf_s(gbFile_RRJ, "SUCCESS: Program is Terminated!!\n");
		fclose(gbFile_RRJ);
		gbFile_RRJ = NULL;
	}

}

HRESULT resize(int width, int height){

	HRESULT hr_RRJ = S_OK;


	if(gpID3D11DepthStencilView_RRJ){
		gpID3D11DepthStencilView_RRJ->Release();
		gpID3D11DepthStencilView_RRJ = NULL;
	}


	if(gpID3D11RenderTargetView_RRJ){
		gpID3D11RenderTargetView_RRJ->Release();
		gpID3D11RenderTargetView_RRJ = NULL;
	}


	gpIDXGISwapChain_RRJ->ResizeBuffers(1, width, height,DXGI_FORMAT_R8G8B8A8_UNORM, 0);


	ID3D11Texture2D *pID3D11Texture2D_BackBuffer_RRJ = NULL;
	gpIDXGISwapChain_RRJ->GetBuffer(0, __uuidof(ID3D11Texture2D), 
		(LPVOID*)&pID3D11Texture2D_BackBuffer_RRJ);

	hr_RRJ = gpID3D11Device_RRJ->CreateRenderTargetView(pID3D11Texture2D_BackBuffer_RRJ,
		NULL, &gpID3D11RenderTargetView_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: resize: ID3D11Device::CreateRenderTargetView() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: resize: ID3D11Device::CreateRenderTargetView() Done!!\n");
		fclose(gbFile_RRJ);
	}


	pID3D11Texture2D_BackBuffer_RRJ->Release();
	pID3D11Texture2D_BackBuffer_RRJ = NULL;



	/********** DEPTH BUFFER *********/
	D3D11_TEXTURE2D_DESC textureDesc_RRJ;
	ZeroMemory((void*)&textureDesc_RRJ, sizeof(D3D11_TEXTURE2D_DESC));
	textureDesc_RRJ.Width = (UINT)width;
	textureDesc_RRJ.Height = (UINT)height;
	textureDesc_RRJ.Format = DXGI_FORMAT_D32_FLOAT;
	textureDesc_RRJ.ArraySize = 1;
	textureDesc_RRJ.MipLevels = 1;
	textureDesc_RRJ.SampleDesc.Count = 1;
	textureDesc_RRJ.SampleDesc.Quality = 0;
	textureDesc_RRJ.Usage = D3D11_USAGE_DEFAULT;
	textureDesc_RRJ.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	textureDesc_RRJ.CPUAccessFlags = 0;
	textureDesc_RRJ.MiscFlags = 0;

	ID3D11Texture2D *pID3D11Texture2D_DepthBuffer_RRJ = NULL;
	hr_RRJ = gpID3D11Device_RRJ->CreateTexture2D(&textureDesc_RRJ, NULL, 
		&pID3D11Texture2D_DepthBuffer_RRJ);


	/********** DEPTH STENCIL VIEW **********/
	D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilView_RRJ;
	ZeroMemory((void*)&depthStencilView_RRJ, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));
	depthStencilView_RRJ.Format = DXGI_FORMAT_D32_FLOAT;
	depthStencilView_RRJ.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;

	hr_RRJ = gpID3D11Device_RRJ->CreateDepthStencilView(pID3D11Texture2D_DepthBuffer_RRJ,
		&depthStencilView_RRJ, &gpID3D11DepthStencilView_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: resize: ID3D11Device::CreateDepthStencilView() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: resize: ID3D11Device::CreateDepthStencilView() Done!!\n");
		fclose(gbFile_RRJ);
	}


	gpID3D11DeviceContext_RRJ->OMSetRenderTargets(1, &gpID3D11RenderTargetView_RRJ,
		gpID3D11DepthStencilView_RRJ);



	/********** VIEW PORT **********/
	D3D11_VIEWPORT d3dViewport_RRJ;
	ZeroMemory((void*)&d3dViewport_RRJ, sizeof(D3D11_VIEWPORT));
	d3dViewport_RRJ.TopLeftX = 0;
	d3dViewport_RRJ.TopLeftY	= 0;
	d3dViewport_RRJ.Width = (float)width;
	d3dViewport_RRJ.Height = (float)height;
	d3dViewport_RRJ.MinDepth = 0.0f;
	d3dViewport_RRJ.MaxDepth = 1.0f;

	gpID3D11DeviceContext_RRJ->RSSetViewports(1, &d3dViewport_RRJ);

	gPerspectiveProjectionMatrix_RRJ = XMMatrixPerspectiveFovLH(
		XMConvertToRadians(45.0f), (float)width / (float)height, 0.1f, 100.0f);

	return(S_OK);
}


void display(void){


	gpID3D11DeviceContext_RRJ->ClearRenderTargetView(gpID3D11RenderTargetView_RRJ, gClearColor);

	gpID3D11DeviceContext_RRJ->ClearDepthStencilView(gpID3D11DepthStencilView_RRJ, 
		D3D11_CLEAR_DEPTH, 1.0f, 0);


	UINT stride_RRJ = sizeof(float) * 3;
	UINT offset_RRJ = 0;

	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(0, 1,
		&gpID3D11Buffer_VertexBuffer_Pyramid_Position_RRJ, &stride_RRJ, &offset_RRJ);


	stride_RRJ = sizeof(float) * 2;
	offset_RRJ = 0;
	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(1, 1,
		&gpID3D11Buffer_VertexBuffer_Pyramid_Texcoord_RRJ, &stride_RRJ, &offset_RRJ);


	/********** Sampler and Shader Resource View **********/
	gpID3D11DeviceContext_RRJ->PSSetShaderResources(0, 1, &gpID3D11ShaderResourceView_Pyramid_Texture_RRJ);
	gpID3D11DeviceContext_RRJ->PSSetSamplers(0, 1, &gpID3D11SamplerState_Pyramid_Texture_RRJ);


	gpID3D11DeviceContext_RRJ->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);


	XMMATRIX translateMatrix_RRJ = XMMatrixIdentity();
	XMMATRIX rotateMatrix_RRJ = XMMatrixIdentity();
	XMMATRIX worldViewMatrix_RRJ = XMMatrixIdentity();
	XMMATRIX worldViewProjectionMatrix_RRJ = XMMatrixIdentity();


	translateMatrix_RRJ = XMMatrixTranslation(0.0f, 0.0f, 5.0f);
	rotateMatrix_RRJ = XMMatrixRotationY(-angle_Pyramid_RRJ);
	worldViewMatrix_RRJ = rotateMatrix_RRJ * translateMatrix_RRJ;
	worldViewProjectionMatrix_RRJ = worldViewMatrix_RRJ * gPerspectiveProjectionMatrix_RRJ;

	CBUFFER constantBuffer;
	constantBuffer.worldViewProjectionMatrix = worldViewProjectionMatrix_RRJ;
	gpID3D11DeviceContext_RRJ->UpdateSubresource(gpID3D11Buffer_ConstantBuffer_RRJ, 0,
		NULL, &constantBuffer, 0, 0);

	gpID3D11DeviceContext_RRJ->Draw(3 * 4, 0);

	gpIDXGISwapChain_RRJ->Present(0, 0);
}

void update(void){
	angle_Pyramid_RRJ = angle_Pyramid_RRJ + 0.005f;
	if(angle_Pyramid_RRJ > 360.0f)
		angle_Pyramid_RRJ = 0.0f;
}
