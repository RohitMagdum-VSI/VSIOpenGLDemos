#include<windows.h>
#include<stdio.h>

#include<d3d11.h>
#include<D3dcompiler.h>

#pragma warning(disable : 4838)
#include"XNAMath/xnamath.h"

#include"WICTextureLoader.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "D3dcompiler.lib")
#pragma comment(lib, "DirectXTK.lib")


#define WIN_WIDTH 800
#define WIN_HEIGHT 600



//For FullScreen
bool bIsFullScreen_RRJ = false;
WINDOWPLACEMENT wpPrev_RRJ = {sizeof(WINDOWPLACEMENT)};
DWORD dwStyle_RRJ;
HWND ghwnd_RRJ = NULL;
bool bActiveWindow_RRJ = false;


//For DirectX
IDXGISwapChain *gpIDXGISwapChain_RRJ = NULL;
ID3D11Device *gpID3D11Device_RRJ = NULL;
ID3D11DeviceContext *gpID3D11DeviceContext_RRJ = NULL;

//For Render and Depth
ID3D11RenderTargetView *gpID3D11RenderTargetView_RRJ = NULL;
ID3D11DepthStencilView *gpID3D11DepthStencilView_RRJ = NULL;

//For Shader
ID3D11VertexShader *gpID3D11VertexShader_RRJ = NULL;
ID3D11PixelShader *gpID3D11PixelShader_RRJ = NULL;

//For InputLayout
ID3D11InputLayout *gpID3D11InputLayout_RRJ = NULL;

//For Uniform
ID3D11Buffer *gpID3D11Buffer_ConstantBuffer_RRJ = NULL;

struct CBUFFER{
	XMMATRIX WorldViewProjectionMatrix;
};


//For Pyramid
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Pyramid_Position_RRJ = NULL;
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Pyramid_Texcoord_RRJ = NULL;
float angle_Pyramid_RRJ = 360.0f;


//For Cube
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Cube_Position_RRJ = NULL;
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Cube_Texcoord_RRJ = NULL;
float angle_Cube_RRJ = 0.0f;

//For Culling
ID3D11RasterizerState *gpID3D11RasterizerState_RRJ = NULL;

//For Error
FILE *gbFile_RRJ = NULL;
const char *gszLogFileName_RRJ = "Log.txt";


//For Texture
ID3D11SamplerState *gpID3D11SamplerState_Pyramid_RRJ;
ID3D11ShaderResourceView *gpID3D11ShaderResourceView_Pyramid_RRJ;

ID3D11SamplerState *gpID3D11SamplerState_Cube_RRJ;
ID3D11ShaderResourceView *gpID3D11ShaderResourceView_Cube_RRJ;

//For Projection
XMMATRIX gPerspectiveProjectionMatrix_RRJ;


//For ClearTexcoord
float gClearColor_RRJ[4];

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow){

	HRESULT initialize(void);
	void display(void);
	void update(void);



	fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "w");
	if(gbFile_RRJ == NULL){
		MessageBox(NULL, TEXT("ERROR: Log Creation Failed!!\n"), TEXT("ERROR"), MB_OK);
		exit(0);
	}
	else{
		fprintf_s(gbFile_RRJ, "SUCCESS: Log Created!!\n");
		fclose(gbFile_RRJ);
	}



	WNDCLASSEX wndclass_RRJ;
	HWND hwnd_RRJ;
	MSG msg_RRJ;
	TCHAR szName_RRJ[] = TEXT("Rohit_R_Jadhav-D3D-20-Texture_Pyramid-Cube");

	bool bDone_RRJ = false;
	HRESULT hr_RRJ = S_OK;


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
		TEXT("Rohit_R_Jadhav-D3D-20-Texture_Pyramid-Cube"),
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


	hr_RRJ = initialize();
	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: WinMain->Initialize() Failed!!\n");
		fclose(gbFile_RRJ);

		DestroyWindow(hwnd_RRJ);
		hwnd_RRJ = NULL;
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: WinMain->Initialize() Done!!\n");
		fclose(gbFile_RRJ);		
	}


	
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
					fprintf_s(gbFile_RRJ, "ERROR: WndProc-> Resize() Failed!!\n");
					fclose(gbFile_RRJ);
					return(hr_RRJ);
				}
				else{
					fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
					fprintf_s(gbFile_RRJ, "SUCCESS: WndProc-> Resize() Done!!\n");
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

	MONITORINFO mi_RRJ = {sizeof(MONITORINFO)};

	if(bIsFullScreen_RRJ == false){

		dwStyle_RRJ = GetWindowLong(ghwnd_RRJ, GWL_STYLE);
		if(dwStyle_RRJ & WS_OVERLAPPEDWINDOW){

			if(GetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ) && GetMonitorInfo(MonitorFromWindow(ghwnd_RRJ, MONITORINFOF_PRIMARY), &mi_RRJ)){

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
			SWP_NOZORDER | SWP_NOSIZE | SWP_NOMOVE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
		bIsFullScreen_RRJ = false;
	}
}


HRESULT initialize(void){

	HRESULT resize(int, int);
	void uninitialize(void);
	HRESULT LoadD3DTexture(const wchar_t*, ID3D11ShaderResourceView**);

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


	for(UINT driverIndex_RRJ = 0; driverIndex_RRJ < numOfDevices_RRJ; driverIndex_RRJ++){

		d3dDriverType_RRJ = d3dDriverTypes_RRJ[driverIndex_RRJ];

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
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize-> D3D11CreateDeviceAndSwapChain() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{

		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize-> D3D11CreateDeviceAndSwapChain() Done!!\n");
		
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize-> Driver Type: ");
		if(d3dDriverType_RRJ == D3D_DRIVER_TYPE_HARDWARE)
			fprintf_s(gbFile_RRJ, "Hardware\n");
		else if(d3dDriverType_RRJ == D3D_DRIVER_TYPE_WARP)
			fprintf_s(gbFile_RRJ, "Warp\n");
		else if(d3dDriverType_RRJ == D3D_DRIVER_TYPE_REFERENCE)
			fprintf_s(gbFile_RRJ, "Reference\n");
		else
			fprintf_s(gbFile_RRJ, "Unknown!!\n");

		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize-> Feature Level: ");
		if(d3dFeatureLevel_acquired_RRJ == D3D_FEATURE_LEVEL_11_0)
			fprintf_s(gbFile_RRJ, "11.0\n");
		else if(d3dFeatureLevel_acquired_RRJ == D3D_FEATURE_LEVEL_10_1)
			fprintf_s(gbFile_RRJ, "10.1\n");
		else if(d3dFeatureLevel_acquired_RRJ == D3D_FEATURE_LEVEL_10_0)
			fprintf_s(gbFile_RRJ, "10.0\n");
		else
			fprintf_s(gbFile_RRJ, "Unknown!!\n");

		fclose(gbFile_RRJ);
	}



	/********** VERTEX SHADER **********/
	const char *vertexShaderSourceCode_RRJ = 
		"cbuffer ConstantBuffer { " \
			"float4x4 worldViewProjectionMatrix;" \
		"};" \

		"struct Vertex_Output { " \
			"float4 position : SV_POSITION;" \
			"float2 texcoord: TEXCOORD;" \
		"};" \

		"Vertex_Output main(float4 pos: POSITION, float2 texcoord : TEXCOORD) {" \
			"Vertex_Output v;" \
			"v.position = mul(worldViewProjectionMatrix, pos);" \
			"v.texcoord = texcoord;" \
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
		0, 0,
		&pID3DBlob_VertexShaderCode_RRJ,
		&pID3DBlob_Error_RRJ);

	if(FAILED(hr_RRJ)){
		if(pID3DBlob_Error_RRJ != NULL){
			fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "ERROR: Initialize-> Vertex Shader Compilation Error!!\n");
			fprintf_s(gbFile_RRJ, "%s\n", (char*)pID3DBlob_Error_RRJ->GetBufferPointer());
			fclose(gbFile_RRJ);
			return(hr_RRJ);
		}
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initiaize-> Vertex Shader Compilation Done!!\n");
		fclose(gbFile_RRJ);
	}


	hr_RRJ = gpID3D11Device_RRJ->CreateVertexShader(
		pID3DBlob_VertexShaderCode_RRJ->GetBufferPointer(),
		pID3DBlob_VertexShaderCode_RRJ->GetBufferSize(),
		NULL,
		&gpID3D11VertexShader_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize-> ID3D11Device:CreateVertexShader() !!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize-> ID3D11Device:CreateVertexShader() !!\n");
		fclose(gbFile_RRJ);
	}


	gpID3D11DeviceContext_RRJ->VSSetShader(gpID3D11VertexShader_RRJ, NULL, NULL);



	/********** PIXEL SHADER **********/
	const char *pixelShaderSourceCode_RRJ = 
		"Texture2D myTexture;" \
		"SamplerState mySamplerState;" \

		"float4 main(float4 pos : SV_POSITION, float2 texcoord : TEXCOORD) : SV_TARGET { " \
			"float4 tex = myTexture.Sample(mySamplerState, texcoord);" \
			"return(tex);" \
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
			fprintf_s(gbFile_RRJ, "ERROR: Initialize-> Pixel Shader Compilation Error!!\n");
			fprintf_s(gbFile_RRJ, "%s\n", (char*)pID3DBlob_Error_RRJ->GetBufferPointer());
			fclose(gbFile_RRJ);
			return(hr_RRJ);
		}
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initiaize-> Pixel Shader Compilation Done!!\n");
		fclose(gbFile_RRJ);
	}


	hr_RRJ = gpID3D11Device_RRJ->CreatePixelShader(
		pID3DBlob_PixelShaderCode_RRJ->GetBufferPointer(),
		pID3DBlob_PixelShaderCode_RRJ->GetBufferSize(),
		NULL,
		&gpID3D11PixelShader_RRJ);


	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize-> ID3D11Device:CreatePixelShader() !!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize-> ID3D11Device:CreatePixelShader() !!\n");
		fclose(gbFile_RRJ);
	}


	gpID3D11DeviceContext_RRJ->PSSetShader(gpID3D11PixelShader_RRJ, NULL, NULL);




	/********** INPUT LAYOUT **********/
	D3D11_INPUT_ELEMENT_DESC inputElementDesc_RRJ[2];
	ZeroMemory((void*)&inputElementDesc_RRJ, sizeof(D3D11_INPUT_ELEMENT_DESC));

	/********** vPosition **********/
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
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize-> ID3D11Device::CreateInputLayout()\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	} 
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize-> ID3D11Device::CreateInputLayout()\n");
		fclose(gbFile_RRJ);
	}


	gpID3D11DeviceContext_RRJ->IASetInputLayout(gpID3D11InputLayout_RRJ);

	pID3DBlob_PixelShaderCode_RRJ->Release();
	pID3DBlob_PixelShaderCode_RRJ = NULL;

	pID3DBlob_VertexShaderCode_RRJ->Release();
	pID3DBlob_VertexShaderCode_RRJ = NULL;





	/********** POSITION AND COLOR **********/
	float pyramid_Position[] = {

 		//Front
 		0.0f, 1.0f, 0.0f,
 		-1.0f, -1.0f, -1.0f,
 		1.0f, -1.0f, -1.0f,

 		//Right
 		0.0f, 1.0f, 0.0f,
 		1.0f, -1.0f, -1.0f,
 		1.0f, -1.0f, 1.0f,

 		//Back
 		0.0f, 1.0f, 0.0f,
 		1.0f, -1.0f, 1.0f, 
 		-1.0f, -1.0f, 1.0f,

 		//Left
 		0.0f, 1.0f, 0.0f,
 		-1.0f, -1.0f, 1.0f,
 		-1.0f, -1.0f, -1.0f,

 	};

 	float pyramid_Texcoord[] = {

 		//Front
		0.5f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,

		//Right
		0.5f, 1.0f,
		1.0f, 0.0f,
		0.0f, 0.0f,

		//Back
		0.5f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,

		//Left
		0.5f, 1.0f,
		1.0f, 0.0f,
		0.0f, 0.0f,

 	};

	float cube_Position[] = {

 		//Front
 		-1.0f, 1.0f, -1.0f,
 		1.0f, 1.0f, -1.0f,
 		-1.0f, -1.0f, -1.0f,

		-1.0f, -1.0f, -1.0f,
		1.0f, 1.0f, -1.0f,
 		1.0f, -1.0f, -1.0f,

 		//Right
 		1.0f, 1.0f, -1.0f,
 		1.0f, 1.0f, 1.0f, 
 		1.0f, -1.0f, -1.0f,

 		1.0f, -1.0f, -1.0f,
 		1.0f, 1.0f, 1.0f, 
 		1.0f, -1.0f, 1.0f,


 		//Back
		1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,

		1.0f, -1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,


		//Left
		-1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f,

		-1.0f, -1.0f, 1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,

		//Top
		-1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, -1.0f,

		-1.0f, 1.0f, -1.0f,
		1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, -1.0f,

		//Bottom
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		-1.0f, -1.0f, -1.0f,

		-1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f,
 		

 	};

 	float cube_Texcoord[] = {

 		//Front
		0.0f, 0.0f,
		1.0f, 0.0f,
		0.0f, 1.0f,

		0.0f, 1.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,

		//Right
		0.0f, 0.0f,
		1.0f, 0.0f,
		0.0f, 1.0f,

		0.0f, 1.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,

		//Back
		0.0f, 0.0f,
		1.0f, 0.0f,
		0.0f, 1.0f,

		0.0f, 1.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,

		//Left
		0.0f, 0.0f,
		1.0f, 0.0f,
		0.0f, 1.0f,

		0.0f, 1.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,

		//Top
		0.0f, 0.0f,
		1.0f, 0.0f,
		0.0f, 1.0f,

		0.0f, 1.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,

		//Bottom
		0.0f, 0.0f,
		1.0f, 0.0f,
		0.0f, 1.0f,

		0.0f, 1.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
 	};




	/********** VERTEX BUFFER **********/

	/********** PYRAMID POSITION BUFFER **********/
	D3D11_BUFFER_DESC bufferDesc_VertexBuffer_Pyramid_Position_RRJ;
	ZeroMemory((void*)&bufferDesc_VertexBuffer_Pyramid_Position_RRJ, sizeof(D3D11_BUFFER_DESC));

	bufferDesc_VertexBuffer_Pyramid_Position_RRJ.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc_VertexBuffer_Pyramid_Position_RRJ.ByteWidth = sizeof(float) * _ARRAYSIZE(pyramid_Position);	
	bufferDesc_VertexBuffer_Pyramid_Position_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc_VertexBuffer_Pyramid_Position_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_VertexBuffer_Pyramid_Position_RRJ, NULL,
		&gpID3D11Buffer_VertexBuffer_Pyramid_Position_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize-> Pyramid Position-> ID3D11Device::CreateBuffer()\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	} 
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize-> Pyramid Position-> ID3D11Device::CreateBuffer()\n");
		fclose(gbFile_RRJ);
	}


	/********** PYRAMID POSITION MAPPING **********/
	D3D11_MAPPED_SUBRESOURCE mappedSubresource_Pyramid_Position_RRJ;
	ZeroMemory((void*)&mappedSubresource_Pyramid_Position_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));

	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_Pyramid_Position_RRJ, 0,
		D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource_Pyramid_Position_RRJ);
	memcpy(mappedSubresource_Pyramid_Position_RRJ.pData, pyramid_Position, sizeof(pyramid_Position));
	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_Pyramid_Position_RRJ, 0);


	/********** PYRAMID COLOR BUFFER **********/
	D3D11_BUFFER_DESC bufferDesc_VertexBuffer_Pyramid_Texcoord_RRJ;
	ZeroMemory((void*)&bufferDesc_VertexBuffer_Pyramid_Texcoord_RRJ, sizeof(D3D11_BUFFER_DESC));

	bufferDesc_VertexBuffer_Pyramid_Texcoord_RRJ.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc_VertexBuffer_Pyramid_Texcoord_RRJ.ByteWidth = sizeof(float) * _ARRAYSIZE(pyramid_Texcoord);
	bufferDesc_VertexBuffer_Pyramid_Texcoord_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc_VertexBuffer_Pyramid_Texcoord_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	 hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_VertexBuffer_Pyramid_Texcoord_RRJ, NULL,
	 	&gpID3D11Buffer_VertexBuffer_Pyramid_Texcoord_RRJ);

	 if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize-> Pyramid Texcoord-> ID3D11Device::CreateBuffer()\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	} 
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize-> Pyramid Texcoord-> ID3D11Device::CreateBuffer()\n");
		fclose(gbFile_RRJ);
	}



	/********** PYRAMID COLOR MAPPING **********/
	D3D11_MAPPED_SUBRESOURCE mappedSubresource_Pyramid_Texcoord_RRJ;
	ZeroMemory((void*)&mappedSubresource_Pyramid_Texcoord_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));

	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_Pyramid_Texcoord_RRJ, 0,
		D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource_Pyramid_Texcoord_RRJ);
	memcpy(mappedSubresource_Pyramid_Texcoord_RRJ.pData, pyramid_Texcoord, sizeof(pyramid_Texcoord));
	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_Pyramid_Texcoord_RRJ, 0);









	/********** CUBE POSITION BUFFER **********/
	D3D11_BUFFER_DESC bufferDesc_VertexBuffer_Cube_Position_RRJ;
	ZeroMemory((void*)&bufferDesc_VertexBuffer_Cube_Position_RRJ, sizeof(D3D11_BUFFER_DESC));

	bufferDesc_VertexBuffer_Cube_Position_RRJ.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc_VertexBuffer_Cube_Position_RRJ.ByteWidth = sizeof(float) * _ARRAYSIZE(cube_Position);
	bufferDesc_VertexBuffer_Cube_Position_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc_VertexBuffer_Cube_Position_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_VertexBuffer_Cube_Position_RRJ, NULL,
		&gpID3D11Buffer_VertexBuffer_Cube_Position_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize-> Cube Position-> ID3D11Device::CreateBuffer()\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	} 
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize-> Cube Position-> ID3D11Device::CreateBuffer()\n");
		fclose(gbFile_RRJ);
	}


	/********** CUBE POSITION MAPPING **********/
	D3D11_MAPPED_SUBRESOURCE mappedSubresource_Cube_Position_RRJ;
	ZeroMemory((void*)&mappedSubresource_Cube_Position_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));

	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_Cube_Position_RRJ, 0,
		D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource_Cube_Position_RRJ);
	memcpy(mappedSubresource_Cube_Position_RRJ.pData, cube_Position, sizeof(cube_Position));
	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_Cube_Position_RRJ, 0);


	/********** CUBE COLOR BUFFER **********/
	D3D11_BUFFER_DESC bufferDesc_VertexBuffer_Cube_Texcoord_RRJ;
	ZeroMemory((void*)&bufferDesc_VertexBuffer_Cube_Texcoord_RRJ, sizeof(D3D11_BUFFER_DESC));

	bufferDesc_VertexBuffer_Cube_Texcoord_RRJ.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc_VertexBuffer_Cube_Texcoord_RRJ.ByteWidth = sizeof(float) * _ARRAYSIZE(cube_Texcoord);
	bufferDesc_VertexBuffer_Cube_Texcoord_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc_VertexBuffer_Cube_Texcoord_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_VertexBuffer_Cube_Texcoord_RRJ, NULL,
		&gpID3D11Buffer_VertexBuffer_Cube_Texcoord_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize-> Cube Texcoord-> ID3D11Device::CreateBuffer()\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	} 
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize-> Cube Texcoord-> ID3D11Device::CreateBuffer()\n");
		fclose(gbFile_RRJ);
	}


	/********** CUBE COLOR MAPPING **********/
	D3D11_MAPPED_SUBRESOURCE mappedSubresource_Cube_Texcoord_RRJ;
	ZeroMemory((void*)&mappedSubresource_Cube_Texcoord_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));

	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_Cube_Texcoord_RRJ, 0,
		D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource_Cube_Texcoord_RRJ);
	memcpy(mappedSubresource_Cube_Texcoord_RRJ.pData, cube_Texcoord, sizeof(cube_Texcoord));
	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_Cube_Texcoord_RRJ, 0);






	/********** CONSTANT BUFFER **********/
	D3D11_BUFFER_DESC bufferDesc_ConstantBuffer_RRJ;
	ZeroMemory((void*)&bufferDesc_ConstantBuffer_RRJ, sizeof(D3D11_BUFFER_DESC));

	bufferDesc_ConstantBuffer_RRJ.Usage = D3D11_USAGE_DEFAULT;
	bufferDesc_ConstantBuffer_RRJ.ByteWidth = sizeof(CBUFFER);
	bufferDesc_ConstantBuffer_RRJ.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	

	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_ConstantBuffer_RRJ, NULL, 
		&gpID3D11Buffer_ConstantBuffer_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize-> Constant Buffer-> ID3D11Device::CreateBuffer()\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	} 
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize-> Constant Buffer-> ID3D11Device::CreateBuffer()\n");
		fclose(gbFile_RRJ);
	}


	gpID3D11DeviceContext_RRJ->VSSetConstantBuffers(0, 1, &gpID3D11Buffer_ConstantBuffer_RRJ);



	/********** Rasterizer State For CULLING **********/
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
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize-> Rasterizer State -> ID3D11Device::CreateRasterizerState()\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	} 
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize-> Rasterizer State -> ID3D11Device::CreateRasterizerState()\n");
		fclose(gbFile_RRJ);
	}


	gpID3D11DeviceContext_RRJ->RSSetState(gpID3D11RasterizerState_RRJ);


	/********** Clear Texcoord **********/
	gClearColor_RRJ[0] = 0.0f;
	gClearColor_RRJ[1] = 0.0f;
	gClearColor_RRJ[2] = 0.0f;
	gClearColor_RRJ[3] = 1.0f;





	/********** LOAD TEXTURE **********/
	hr_RRJ = LoadD3DTexture(L"Stone.bmp", &gpID3D11ShaderResourceView_Pyramid_RRJ);
	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize: LoadD3DTexture() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
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
		&gpID3D11SamplerState_Pyramid_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize: CreateSamplerState() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize: CreateSamplerState() Done!!\n");
		fclose(gbFile_RRJ);
	}




	/********** LOAD TEXTURE **********/
	hr_RRJ = LoadD3DTexture(L"Kundali.bmp", &gpID3D11ShaderResourceView_Cube_RRJ);
	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize: LoadD3DTexture() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize: LoadD3DTexture() Done!!\n");
		fclose(gbFile_RRJ);
	}




 	/********** SAMPLER **********/
	ZeroMemory((void*)&samplerDesc_RRJ, sizeof(D3D11_SAMPLER_DESC));

	samplerDesc_RRJ.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	samplerDesc_RRJ.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc_RRJ.AddressV	 = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc_RRJ.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;

	hr_RRJ = gpID3D11Device_RRJ->CreateSamplerState(&samplerDesc_RRJ, 
		&gpID3D11SamplerState_Cube_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize: CreateSamplerState() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize: CreateSamplerState() Done!!\n");
		fclose(gbFile_RRJ);
	}








	gPerspectiveProjectionMatrix_RRJ = XMMatrixIdentity();

	hr_RRJ = resize(WIN_WIDTH, WIN_HEIGHT);
	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize-> Resize()\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	} 
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Initialize-> Resize()\n");
		fclose(gbFile_RRJ);
	}

	return(S_OK);
}




HRESULT LoadD3DTexture(const wchar_t *fileName, ID3D11ShaderResourceView **ppShaderResourceView){

	HRESULT hr_RRJ = S_OK;


	hr_RRJ = DirectX::CreateWICTextureFromFile(
		gpID3D11Device_RRJ,
		gpID3D11DeviceContext_RRJ,
		fileName, NULL, ppShaderResourceView);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: %ws\n", fileName);
		fprintf_s(gbFile_RRJ, "ERROR: LoadD3DTexture: DirectX::CreateWICTextureFromFile() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: LoadD3DTexture: DirectX::CreateWICTextureFromFile() Done!!\n");
		fclose(gbFile_RRJ);
	}

	return(S_OK);
}



void uninitialize(void){

	if(gpID3D11SamplerState_Cube_RRJ){
		gpID3D11SamplerState_Cube_RRJ->Release();
		gpID3D11SamplerState_Cube_RRJ = NULL;
	}

	if(gpID3D11ShaderResourceView_Cube_RRJ){
		gpID3D11ShaderResourceView_Cube_RRJ->Release();
		gpID3D11ShaderResourceView_Cube_RRJ = NULL;
	}

	if(gpID3D11SamplerState_Pyramid_RRJ){
		gpID3D11SamplerState_Pyramid_RRJ->Release();
		gpID3D11SamplerState_Pyramid_RRJ = NULL;
	}

	if(gpID3D11ShaderResourceView_Pyramid_RRJ){
		gpID3D11ShaderResourceView_Pyramid_RRJ->Release();
		gpID3D11ShaderResourceView_Pyramid_RRJ = NULL;
	}


	if(gpID3D11RasterizerState_RRJ){
		gpID3D11RasterizerState_RRJ->Release();
		gpID3D11RasterizerState_RRJ = NULL;
	}

	if(gpID3D11Buffer_ConstantBuffer_RRJ){
		gpID3D11Buffer_ConstantBuffer_RRJ->Release();
		gpID3D11Buffer_ConstantBuffer_RRJ = NULL;
	}

	if(gpID3D11Buffer_VertexBuffer_Cube_Texcoord_RRJ){
		gpID3D11Buffer_VertexBuffer_Cube_Texcoord_RRJ->Release();
		gpID3D11Buffer_VertexBuffer_Cube_Texcoord_RRJ = NULL;
	}

	if(gpID3D11Buffer_VertexBuffer_Cube_Position_RRJ){
		gpID3D11Buffer_VertexBuffer_Cube_Position_RRJ->Release();
		gpID3D11Buffer_VertexBuffer_Cube_Position_RRJ = NULL;
	}

	if(gpID3D11Buffer_VertexBuffer_Pyramid_Texcoord_RRJ){
		gpID3D11Buffer_VertexBuffer_Pyramid_Texcoord_RRJ->Release();
		gpID3D11Buffer_VertexBuffer_Pyramid_Texcoord_RRJ = NULL;
	}

	if(gpID3D11Buffer_VertexBuffer_Pyramid_Position_RRJ){
		gpID3D11Buffer_VertexBuffer_Pyramid_Position_RRJ->Release();
		gpID3D11Buffer_VertexBuffer_Pyramid_Position_RRJ = NULL;
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
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Uninitialize -> Log Close!!\n");
		fprintf_s(gbFile_RRJ, "SUCCESS: End!!\n");
		fclose(gbFile_RRJ);
		gbFile_RRJ = NULL;
	}
}


HRESULT resize(int width, int height){

	HRESULT hr_RRJ;


	if(gpID3D11DepthStencilView_RRJ){
		gpID3D11DepthStencilView_RRJ->Release();
		gpID3D11DepthStencilView_RRJ = NULL;
	}

	if(gpID3D11RenderTargetView_RRJ){
		gpID3D11RenderTargetView_RRJ->Release();
		gpID3D11RenderTargetView_RRJ = NULL;
	}


	/********** Render Target View **********/
	gpIDXGISwapChain_RRJ->ResizeBuffers(1, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, 0);

	ID3D11Texture2D *pID3D11Texture2D_BackBuffer_RRJ = NULL;
	gpIDXGISwapChain_RRJ->GetBuffer(0, __uuidof(ID3D11Texture2D),
		(LPVOID*)&pID3D11Texture2D_BackBuffer_RRJ);

	hr_RRJ = gpID3D11Device_RRJ->CreateRenderTargetView(pID3D11Texture2D_BackBuffer_RRJ, NULL,
		&gpID3D11RenderTargetView_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Resize-> Render Target View-> ID3D11Device::CreateRenderTargetView()\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	} 
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Resize-> Render Target View-> ID3D11Device::CreateRenderTargetView()\n");
		fclose(gbFile_RRJ);
	}


	pID3D11Texture2D_BackBuffer_RRJ->Release();
	pID3D11Texture2D_BackBuffer_RRJ = NULL;



	/********** Depth Buffer **********/
	D3D11_TEXTURE2D_DESC textureDesc_RRJ;
	ZeroMemory((void*)&textureDesc_RRJ, sizeof(D3D11_TEXTURE2D_DESC));

	textureDesc_RRJ.Width = (UINT)width;
	textureDesc_RRJ.Height = (UINT)height;
	textureDesc_RRJ.ArraySize = 1;
	textureDesc_RRJ.MipLevels = 1;
	textureDesc_RRJ.SampleDesc.Count = 1;
	textureDesc_RRJ.SampleDesc.Quality = 0;
	textureDesc_RRJ.Format = DXGI_FORMAT_D32_FLOAT;
	textureDesc_RRJ.Usage = D3D11_USAGE_DEFAULT;
	textureDesc_RRJ.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	textureDesc_RRJ.CPUAccessFlags = 0;
	textureDesc_RRJ.MiscFlags = 0;

	ID3D11Texture2D *pID3D11Texture2D_DepthBuffer_RRJ = NULL;
	hr_RRJ = gpID3D11Device_RRJ->CreateTexture2D(&textureDesc_RRJ, NULL,
		&pID3D11Texture2D_DepthBuffer_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Resize-> Depth Stencil View-> Depth Buffer-> ID3D11Device::CreateTexture2D()\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	} 
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Resize-> Depth Stencil View-> Depth Buffer-> ID3D11Device::CreateTexture2D()\n");
		fclose(gbFile_RRJ);
	}


	/********** Depth Stencil View **********/
	D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc_RRJ;
	ZeroMemory((void*)&depthStencilViewDesc_RRJ, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));

	depthStencilViewDesc_RRJ.Format = DXGI_FORMAT_D32_FLOAT;
	depthStencilViewDesc_RRJ.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;

	hr_RRJ = gpID3D11Device_RRJ->CreateDepthStencilView(pID3D11Texture2D_DepthBuffer_RRJ, 
		&depthStencilViewDesc_RRJ, &gpID3D11DepthStencilView_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Resize-> Depth Stencil View-> ID3D11Device::CreateDepthStencilView()\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	} 
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Resize-> Depth Stencil View-> ID3D11Device::CreateDepthStencilView()\n");
		fclose(gbFile_RRJ);
	}	 



	gpID3D11DeviceContext_RRJ->OMSetRenderTargets(1, &gpID3D11RenderTargetView_RRJ, gpID3D11DepthStencilView_RRJ);


	/********** ViewPort **********/
	D3D11_VIEWPORT d3dViewport_RRJ;
	ZeroMemory((void*)&d3dViewport_RRJ, sizeof(D3D11_VIEWPORT));

	d3dViewport_RRJ.TopLeftX = 0;
	d3dViewport_RRJ.TopLeftY	= 0;
	d3dViewport_RRJ.Width = (float)width;
	d3dViewport_RRJ.Height = (float)height;
	d3dViewport_RRJ.MinDepth = 0.0f;
	d3dViewport_RRJ.MaxDepth =1.0f;

	gpID3D11DeviceContext_RRJ->RSSetViewports(1, &d3dViewport_RRJ);

	gPerspectiveProjectionMatrix_RRJ = XMMatrixPerspectiveFovLH(
		XMConvertToRadians(45.0f), (float)width / (float) height, 0.1f, 100.0f);

	return(S_OK);
}


void display(void){

	XMMATRIX translateMatrix_RRJ;
	XMMATRIX rotateMatrix_RRJ;
	XMMATRIX scaleMatrix_RRJ;
	XMMATRIX worldViewMatrix_RRJ;
	XMMATRIX worldViewProjectionMatrix_RRJ;

	gpID3D11DeviceContext_RRJ->ClearRenderTargetView(gpID3D11RenderTargetView_RRJ, gClearColor_RRJ);

	gpID3D11DeviceContext_RRJ->ClearDepthStencilView(gpID3D11DepthStencilView_RRJ, 
		D3D11_CLEAR_DEPTH, 1.0f, 0);


	
	/********** PYRAMID **********/
	translateMatrix_RRJ = XMMatrixIdentity();
	rotateMatrix_RRJ = XMMatrixIdentity();
	worldViewMatrix_RRJ = XMMatrixIdentity();
	worldViewProjectionMatrix_RRJ = XMMatrixIdentity();


	translateMatrix_RRJ = XMMatrixTranslation(-2.0f, 0.0f, 6.0f);
	rotateMatrix_RRJ = XMMatrixRotationY(-angle_Pyramid_RRJ);

	worldViewMatrix_RRJ = rotateMatrix_RRJ * translateMatrix_RRJ;
	worldViewProjectionMatrix_RRJ = worldViewMatrix_RRJ * gPerspectiveProjectionMatrix_RRJ;

	
	UINT stride = sizeof(float) * 3;
	UINT offset = 0;
	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(0, 1, &gpID3D11Buffer_VertexBuffer_Pyramid_Position_RRJ, &stride, &offset);

	stride = sizeof(float) * 2;
	offset = 0;
	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(1, 1, &gpID3D11Buffer_VertexBuffer_Pyramid_Texcoord_RRJ, &stride, &offset); 	

	
	gpID3D11DeviceContext_RRJ->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);


	gpID3D11DeviceContext_RRJ->PSSetSamplers(0, 1, &gpID3D11SamplerState_Pyramid_RRJ);
	gpID3D11DeviceContext_RRJ->PSSetShaderResources(0, 1, &gpID3D11ShaderResourceView_Pyramid_RRJ);


	CBUFFER constantBuffer;
	constantBuffer.WorldViewProjectionMatrix = worldViewProjectionMatrix_RRJ;
	gpID3D11DeviceContext_RRJ->UpdateSubresource(gpID3D11Buffer_ConstantBuffer_RRJ, 0,
		NULL, &constantBuffer, 0, 0);	


	gpID3D11DeviceContext_RRJ->Draw(3 * 4, 0);



	/********** CUBE **********/
	translateMatrix_RRJ = XMMatrixIdentity();
	rotateMatrix_RRJ = XMMatrixIdentity();
	scaleMatrix_RRJ = XMMatrixIdentity();
	worldViewMatrix_RRJ = XMMatrixIdentity();
	worldViewProjectionMatrix_RRJ = XMMatrixIdentity();


	translateMatrix_RRJ = XMMatrixTranslation(2.0f, 0.0f, 6.0f);
	rotateMatrix_RRJ = XMMatrixRotationX(-angle_Cube_RRJ) * XMMatrixRotationY(-angle_Cube_RRJ) * XMMatrixRotationZ(-angle_Cube_RRJ);
	scaleMatrix_RRJ = XMMatrixScaling(0.9f, 0.9f, 0.9f);

	worldViewMatrix_RRJ = rotateMatrix_RRJ * scaleMatrix_RRJ * translateMatrix_RRJ;
	worldViewProjectionMatrix_RRJ = worldViewMatrix_RRJ * gPerspectiveProjectionMatrix_RRJ;


	stride = sizeof(float) * 3;
	offset = 0;
	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(0, 1, &gpID3D11Buffer_VertexBuffer_Cube_Position_RRJ, &stride, &offset);

	stride = sizeof(float) * 2;
	offset = 0;
	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(1, 1, &gpID3D11Buffer_VertexBuffer_Cube_Texcoord_RRJ, &stride, &offset);


	gpID3D11DeviceContext_RRJ->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);


	gpID3D11DeviceContext_RRJ->PSSetSamplers(0, 1, &gpID3D11SamplerState_Cube_RRJ);
	gpID3D11DeviceContext_RRJ->PSSetShaderResources(0, 1, &gpID3D11ShaderResourceView_Cube_RRJ);

	constantBuffer.WorldViewProjectionMatrix = worldViewProjectionMatrix_RRJ;
	gpID3D11DeviceContext_RRJ->UpdateSubresource(gpID3D11Buffer_ConstantBuffer_RRJ, 0,
		NULL, &constantBuffer, 0, 0);


	gpID3D11DeviceContext_RRJ->Draw(6 * 6, 0);


	gpIDXGISwapChain_RRJ->Present(0, 0);
}

void update(void){

	angle_Pyramid_RRJ = angle_Pyramid_RRJ - 0.005f;
	angle_Cube_RRJ = angle_Cube_RRJ + 0.005;

	if(angle_Pyramid_RRJ < 0.0f)
		angle_Pyramid_RRJ = 360.0f;

	if(angle_Cube_RRJ > 360.0f)
		angle_Cube_RRJ = 0.0f;
}

