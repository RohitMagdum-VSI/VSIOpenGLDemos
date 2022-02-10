#include<windows.h>
#include<stdio.h>

#include<d3d11.h>
#include<D3dcompiler.h>

#pragma warning(disable: 4838)

#include"XNAMath/xnamath.h"
#include"Sphere.h"



#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "D3dcompiler.lib")
#pragma comment(lib, "Sphere.lib")


#define WIN_WIDTH 800
#define WIN_HEIGHT 600

//For FullScreen
bool bIsFullScreen_RRJ = false;
WINDOWPLACEMENT wpPrev_RRJ = {sizeof(WINDOWPLACEMENT)};
DWORD dwStyle_RRJ;
HWND ghwnd_RRJ = NULL;

bool bActiveWindow_RRJ = false;


//For Error
FILE *gbFile_RRJ = NULL;
const char *gszLogFileName_RRJ = "Log.txt";



//For DirectX
IDXGISwapChain *gpIDXGISwapChain_RRJ = NULL;
ID3D11Device *gpID3D11Device_RRJ = NULL;
ID3D11DeviceContext *gpID3D11DeviceContext_RRJ = NULL;
ID3D11RenderTargetView *gpID3D11RenderTargetView_RRJ = NULL;
ID3D11DepthStencilView *gpID3D11DepthStencilView_RRJ = NULL;

//For Shader
ID3D11VertexShader *gpID3D11VertexShader_RRJ = NULL;
ID3D11PixelShader *gpID3D11PixelShader_RRJ = NULL;


//For Sphere
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Sphere_Position_RRJ = NULL;
ID3D11Buffer *gpID3D11Buffer_IndexBuffer_Sphere_Index_RRJ = NULL;

float sphere_Vertices_RRJ[1146];
float sphere_Normal_RRJ[1146];
float sphere_Texcoord_RRJ[764];
unsigned short sphere_Elements_RRJ[2280];
unsigned int gNumOfElements_RRJ;
unsigned int gNumOfVertices_RRJ;

//For Layout/ Attributes
ID3D11InputLayout *gpID3D11InputLayout_RRJ = NULL;

//For Uniform
ID3D11Buffer *gpID3D11Buffer_ConstantBuffer_RRJ = NULL;

//For ClearColor
float gClearColor_RRJ[4];

struct CBUFFER{
	XMMATRIX WorldViewProjectionMatrix;
	XMVECTOR Color;
};


XMMATRIX gPerspectiveProjectionMatrix_RRJ;


//For Culling
ID3D11RasterizerState *gpID3D11RasterizerState_RRJ = NULL;



//For Stack
struct STACK {
	XMMATRIX WorldViewMatrix;
	struct STACK *next;
	struct STACK *prev;
};

typedef struct STACK ModelViewStack;
ModelViewStack *TopNode_RRJ = NULL;
int MaxTop_RRJ = 32;
int iTop_RRJ = -1;


//For Planet
float year_RRJ;
float day_RRJ;
float moon_RRJ;

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow){

	void uninitialize(void);
	void ToggleFullScreen(void);
	HRESULT initialize(void);
	void display(void);

	HRESULT hr_RRJ;
	bool bDone_RRJ = false;


	fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "w");
	if(gbFile_RRJ == NULL){
		MessageBox(NULL, TEXT("ERROR: Log Creation Failed!!"), TEXT("ERROR"), MB_OK);
		exit(0);
	}
	else{
		fprintf_s(gbFile_RRJ, "SUCCESS: Log Created!!\n");
		fclose(gbFile_RRJ);
	}



	WNDCLASSEX wndclass_RRJ;
	HWND hwnd_RRJ;
	MSG msg_RRJ;
	TCHAR szName_RRJ[] = TEXT("Rohit_R_Jadhav-D3D-32-SolarSystem");

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
			TEXT("Rohit_R_Jadhav-D3D-32-SolarSystem"),
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
				//For Update();
			}
			display();
		}
	}

	return((int)msg_RRJ.wParam);
}



LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam){

	void uninitialize(void);
	void ToggleFullScreen(void);
	HRESULT resize(int, int);

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
					fprintf_s(gbFile_RRJ, "ERROR: In WM_SIZE -> Resize() Failed!!\n");
					fclose(gbFile_RRJ);
					return(hr_RRJ);
				}
				else{
					fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
					fprintf_s(gbFile_RRJ, "SUCCESS: In WM_SIZE -> Resize() Done!!\n");
					fclose(gbFile_RRJ);
				}
			}


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


		case WM_CHAR:
			switch(wParam){
				case  'Y':
					year_RRJ = year_RRJ + (30.0f / 360.0f);
					break;

				case  'y':
					year_RRJ = year_RRJ - (30.0f / 360.0f);
					break;

				case 'D':
					day_RRJ = day_RRJ + (30.0f / 360.0f);
					break;

				case 'd':
					day_RRJ = day_RRJ - (30.0f / 360.0f);
					break;

				case 'M':
					moon_RRJ = moon_RRJ + (30.0f / 360.0f);
					break;

				case 'm':
					moon_RRJ = moon_RRJ - (30.0f / 360.0f);
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
			SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED | SWP_NOSIZE | SWP_NOMOVE);
		ShowCursor(TRUE);
		bIsFullScreen_RRJ = false;
	}
}


HRESULT initialize(void){


	HRESULT resize(int, int);
	void uninitialize(void);

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

	dxgiSwapChainDesc_RRJ.SampleDesc.Count = 1;
	dxgiSwapChainDesc_RRJ.SampleDesc.Quality = 0;

	dxgiSwapChainDesc_RRJ.Windowed = TRUE;
	dxgiSwapChainDesc_RRJ.OutputWindow = ghwnd_RRJ;


	for(UINT driverTypeIndex_RRJ = 0; driverTypeIndex_RRJ < numDriverTypes_RRJ; driverTypeIndex_RRJ++){

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
		fprintf_s(gbFile_RRJ, "SUCCESS: D3D11CreateDeviceAndSwapChain() done!!\n");

		fprintf_s(gbFile_RRJ, "SUCCESS: Driver Type: ");
		if(d3dDriverType_RRJ == D3D_DRIVER_TYPE_HARDWARE)
			fprintf_s(gbFile_RRJ, "Hardware !!\n");
		else if(d3dDriverType_RRJ == D3D_DRIVER_TYPE_WARP)
			fprintf_s(gbFile_RRJ, "Warp !!\n");
		else if(d3dDriverType_RRJ == D3D_DRIVER_TYPE_REFERENCE)
			fprintf_s(gbFile_RRJ, "Reference/ Software !!\n");
		else
			fprintf_s(gbFile_RRJ, "Unknown !!\n");


		fprintf(gbFile_RRJ, "SUCCESS: Feature Level: ");
		if(d3dFeatureLevel_acquired_RRJ == D3D_FEATURE_LEVEL_11_0)
			fprintf_s(gbFile_RRJ, "v11.0 \n");
		else if(d3dFeatureLevel_acquired_RRJ == D3D_FEATURE_LEVEL_10_1)
			fprintf_s(gbFile_RRJ, "v10.1 \n");
		else if(d3dFeatureLevel_acquired_RRJ == D3D_FEATURE_LEVEL_10_0)
			fprintf_s(gbFile_RRJ, "v10.0 \n");
		else
			fprintf_s(gbFile_RRJ, "Unknown !!\n");

		fclose(gbFile_RRJ);
	}




	/********** VERTEX SHADER **********/
	const char *vertexShaderSourceCode_RRJ = 
		"cbuffer ConstantBuffer {" \
			"float4x4 worldViewProjectionMatrix;" \
			"float4 color;" \
		"};" \


		"struct Vertex_Output {" \
			"float4 position : SV_POSITION;" \
		"};" \

		"Vertex_Output main(float4 pos : POSITION){" \
			"Vertex_Output v;" \

			"v.position = mul(worldViewProjectionMatrix, pos);" \
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
 			fprintf_s(gbFile_RRJ, "VERTEX SHADER ERROR: \n %s \n", (char*)pID3DBlob_Error_RRJ->GetBufferPointer());
 			fclose(gbFile_RRJ);

 			pID3DBlob_Error_RRJ->Release();
 			pID3DBlob_Error_RRJ = NULL;

 			return(hr_RRJ);
 		}
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: Vertex Shader Compilation Done!!\n");
 		fclose(gbFile_RRJ);
 	}



 	hr_RRJ = gpID3D11Device_RRJ->CreateVertexShader(
 		pID3DBlob_VertexShaderCode_RRJ->GetBufferPointer(),
 		pID3DBlob_VertexShaderCode_RRJ->GetBufferSize(),
 		NULL,
 		&gpID3D11VertexShader_RRJ);


 	if(FAILED(hr_RRJ)){
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "ERROR: Vertex Shader Creation Failed!!\n");
 		fclose(gbFile_RRJ);

 		return(hr_RRJ);
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: Vertex Shader Created!!\n");
 		fclose(gbFile_RRJ);
 	}



 	gpID3D11DeviceContext_RRJ->VSSetShader(gpID3D11VertexShader_RRJ, NULL, NULL);





 	/********** PIXEL SHADER **********/
 	const char *pixelShaderSourceCode_RRJ = 
 		"cbuffer ConstantBuffer {" \
 			"float4x4 worldViewProjectionMatrix;" \
 			"float4 color;" \
 		"};" \

 		"float4 main(float4 pos : SV_POSITION) : SV_TARGET { " \
 			"return(color);" \
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
 			fprintf_s(gbFile_RRJ, "PIXEL SHADER ERROR: \n %s \n",
 				(char*)pID3DBlob_Error_RRJ->GetBufferPointer());
 			fclose(gbFile_RRJ);

 			return(hr_RRJ);
 		}
 	}
 	else{

 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: Pixel Shader Compilation Done!!\n");
 		fclose(gbFile_RRJ);
 	}



 	hr_RRJ = gpID3D11Device_RRJ->CreatePixelShader(
 		pID3DBlob_PixelShaderCode_RRJ->GetBufferPointer(),
 		pID3DBlob_PixelShaderCode_RRJ->GetBufferSize(),
 		NULL,
 		&gpID3D11PixelShader_RRJ);


 	if(FAILED(hr_RRJ)){
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "ERROR: Pixel Shader Creation Failed!!\n");
 		fclose(gbFile_RRJ);

 		return(hr_RRJ);
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: Pixel Shader Created!!\n");
 		fclose(gbFile_RRJ);
 	}


 	gpID3D11DeviceContext_RRJ->PSSetShader(gpID3D11PixelShader_RRJ, NULL, NULL);




 	/********* INPUT LAYOUT **********/
 	D3D11_INPUT_ELEMENT_DESC inputElementDesc_RRJ;
 	ZeroMemory((void*)&inputElementDesc_RRJ, sizeof(D3D11_INPUT_ELEMENT_DESC));

 	//For vPosition
 	inputElementDesc_RRJ.SemanticName = "POSITION";
 	inputElementDesc_RRJ.SemanticIndex = 0;
 	inputElementDesc_RRJ.Format = DXGI_FORMAT_R32G32B32_FLOAT;
 	inputElementDesc_RRJ.InputSlot = 0;
 	inputElementDesc_RRJ.AlignedByteOffset = 0;
 	inputElementDesc_RRJ.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
 	inputElementDesc_RRJ.InstanceDataStepRate = 0;



 	hr_RRJ = gpID3D11Device_RRJ->CreateInputLayout(&inputElementDesc_RRJ, 1,
 		pID3DBlob_VertexShaderCode_RRJ->GetBufferPointer(),
 		pID3DBlob_VertexShaderCode_RRJ->GetBufferSize(),
 		&gpID3D11InputLayout_RRJ);

 	if(FAILED(hr_RRJ)){
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "ERROR: ID3D11Device::CreateInputLayout() Failed!!\n");
 		fclose(gbFile_RRJ);

 		return(hr_RRJ);
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: ID3D11Device::CreateInputDevice() Done!!\n");
 		fclose(gbFile_RRJ);
 	}

 	gpID3D11DeviceContext_RRJ->IASetInputLayout(gpID3D11InputLayout_RRJ);


 	pID3DBlob_PixelShaderCode_RRJ->Release();
 	pID3DBlob_PixelShaderCode_RRJ = NULL;

 	pID3DBlob_VertexShaderCode_RRJ->Release();
 	pID3DBlob_VertexShaderCode_RRJ = NULL;




 	/********** POSITION **********/
 	getSphereVertexData(sphere_Vertices_RRJ, sphere_Normal_RRJ, sphere_Texcoord_RRJ, sphere_Elements_RRJ);
	gNumOfVertices_RRJ = getNumberOfSphereVertices();
	gNumOfElements_RRJ = getNumberOfSphereElements();



 	/********** Vertex Buffer Sphere_Position **********/
 	D3D11_BUFFER_DESC bufferDesc_Sphere_Pos_RRJ;
 	ZeroMemory((void*)&bufferDesc_Sphere_Pos_RRJ, sizeof(D3D11_BUFFER_DESC));

 	bufferDesc_Sphere_Pos_RRJ.Usage = D3D11_USAGE_DYNAMIC;
 	bufferDesc_Sphere_Pos_RRJ.ByteWidth = sizeof(float) * _ARRAYSIZE(sphere_Vertices_RRJ);
 	bufferDesc_Sphere_Pos_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
 	bufferDesc_Sphere_Pos_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

 	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_Sphere_Pos_RRJ, NULL, &gpID3D11Buffer_VertexBuffer_Sphere_Position_RRJ);

 	if(FAILED(hr_RRJ)){
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "ERROR: ID3D11Device::CreateBuffer() For Sphere_Position Failed!!\n");
 		fclose(gbFile_RRJ);

 		return(hr_RRJ);
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: ID3D11Device::CreateBuffer() For Sphere_Position Done!!\n");
 		fclose(gbFile_RRJ);
 	}




 	/********** Memory Mapped I/O **********/
 	D3D11_MAPPED_SUBRESOURCE mappedSubresource_Sphere_Position_RRJ;
 	ZeroMemory((void*)&mappedSubresource_Sphere_Position_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));

 	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_Sphere_Position_RRJ, 0, 
 		D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource_Sphere_Position_RRJ);
 	memcpy(mappedSubresource_Sphere_Position_RRJ.pData, sphere_Vertices_RRJ, sizeof(sphere_Vertices_RRJ));
 	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_Sphere_Position_RRJ, NULL);





 	/********** Vertex Buffer For Sphere Elements **********/
 	D3D11_BUFFER_DESC bufferDesc_Sphere_Elements_RRJ;
 	ZeroMemory((void*)&bufferDesc_Sphere_Elements_RRJ, sizeof(D3D11_BUFFER_DESC));

 	bufferDesc_Sphere_Elements_RRJ.Usage = D3D11_USAGE_DYNAMIC;
 	bufferDesc_Sphere_Elements_RRJ.ByteWidth = sizeof(float) * _ARRAYSIZE(sphere_Elements_RRJ);
 	bufferDesc_Sphere_Elements_RRJ.BindFlags = D3D11_BIND_INDEX_BUFFER;
 	bufferDesc_Sphere_Elements_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

 	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_Sphere_Elements_RRJ, NULL,
 		&gpID3D11Buffer_IndexBuffer_Sphere_Index_RRJ);

 	if(FAILED(hr_RRJ)){
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "ERROR: ID3D11::CreateBuffer() for Sphere_Elements Failed!!\n");
 		fclose(gbFile_RRJ);
 		return(hr_RRJ);
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: ID3D11::CreateBuffer() for Sphere_Elements Done!!\n");
 		fclose(gbFile_RRJ);	
 	}



 	/********** Memory Map I/O For Sphere_Elements **********/
 	D3D11_MAPPED_SUBRESOURCE mappedSubresource_Sphere_Elements_RRJ;
 	ZeroMemory((void*)&mappedSubresource_Sphere_Elements_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));

 	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_IndexBuffer_Sphere_Index_RRJ, 0, 
 		D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource_Sphere_Elements_RRJ);

 	memcpy(mappedSubresource_Sphere_Elements_RRJ.pData, sphere_Elements_RRJ, sizeof(sphere_Elements_RRJ));
 	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_IndexBuffer_Sphere_Index_RRJ, 0);








 	/********** Constant Buffer **********/
 	D3D11_BUFFER_DESC bufferDesc_ConstantBuffer_RRJ;
 	ZeroMemory((void*)&bufferDesc_ConstantBuffer_RRJ, sizeof(D3D11_BUFFER_DESC));

 	bufferDesc_ConstantBuffer_RRJ.Usage = D3D11_USAGE_DEFAULT;
 	bufferDesc_ConstantBuffer_RRJ.ByteWidth = sizeof(CBUFFER);
 	bufferDesc_ConstantBuffer_RRJ.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

 	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_ConstantBuffer_RRJ, NULL,
 		&gpID3D11Buffer_ConstantBuffer_RRJ);

 	if(FAILED(hr_RRJ)){
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "ERROR: ID3D11Device::CreateBuffer() For Constant Buffer Failed!!\n");
 		fclose(gbFile_RRJ);

 		return(hr_RRJ);
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: ID3D11Device::CreateBuffer() For ConstantBuffer Done!!\n");
 		fclose(gbFile_RRJ);
 	}


 	gpID3D11DeviceContext_RRJ->VSSetConstantBuffers(0, 1, &gpID3D11Buffer_ConstantBuffer_RRJ);
 	gpID3D11DeviceContext_RRJ->PSSetConstantBuffers(0, 1, &gpID3D11Buffer_ConstantBuffer_RRJ);




 	/********** SETTING CULLING OFF **********/
 	D3D11_RASTERIZER_DESC rasterizerDesc_RRJ;
 	ZeroMemory((void*)&rasterizerDesc_RRJ, sizeof(D3D11_RASTERIZER_DESC));

 	rasterizerDesc_RRJ.AntialiasedLineEnable = FALSE;
 	rasterizerDesc_RRJ.CullMode = D3D11_CULL_NONE;
 	rasterizerDesc_RRJ.DepthBias = 0;
 	rasterizerDesc_RRJ.DepthBiasClamp = 0.0f;
 	rasterizerDesc_RRJ.DepthClipEnable = FALSE;
 	rasterizerDesc_RRJ.FillMode = D3D11_FILL_SOLID;
 	rasterizerDesc_RRJ.FrontCounterClockwise = FALSE;
 	rasterizerDesc_RRJ.MultisampleEnable = FALSE;
 	rasterizerDesc_RRJ.ScissorEnable = FALSE;
 	rasterizerDesc_RRJ.SlopeScaledDepthBias = 0.0f;

 	hr_RRJ = gpID3D11Device_RRJ->CreateRasterizerState(&rasterizerDesc_RRJ, &gpID3D11RasterizerState_RRJ);
 	if(FAILED(hr_RRJ)){
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "ERROR: ID3D11Device:CreateRasterizerState() Failed!!\n");
 		fclose(gbFile_RRJ);
 		return(hr_RRJ);
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: ID3D11Device:CreateRasterizerState() Done!!\n");
 		fclose(gbFile_RRJ);
 	}


 	gpID3D11DeviceContext_RRJ->RSSetState(gpID3D11RasterizerState_RRJ);


 	gClearColor_RRJ[0] = 0.0f;
 	gClearColor_RRJ[1] = 0.0f;
 	gClearColor_RRJ[2] = 0.0f;
 	gClearColor_RRJ[3] = 1.0f;

 	gPerspectiveProjectionMatrix_RRJ = XMMatrixIdentity();

 	hr_RRJ =resize(WIN_WIDTH, WIN_HEIGHT);
 	if(FAILED(hr_RRJ)){
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "ERROR: Warmup Resize() Failed!!\n");
 		fclose(gbFile_RRJ);

 		return(hr_RRJ);
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: Warmup Resize() Done!!\n");
 		fclose(gbFile_RRJ);
 	}

 	return(S_OK);
}


void uninitialize(void){

		if(gpID3D11RasterizerState_RRJ){
			gpID3D11RasterizerState_RRJ->Release();
			gpID3D11RasterizerState_RRJ = NULL;
		}


		if(gpID3D11Buffer_ConstantBuffer_RRJ){
			gpID3D11Buffer_ConstantBuffer_RRJ->Release();
			gpID3D11Buffer_ConstantBuffer_RRJ = NULL;
		}

		if(gpID3D11InputLayout_RRJ){
			gpID3D11InputLayout_RRJ->Release();
			gpID3D11InputLayout_RRJ = NULL;
		}

		if(gpID3D11Buffer_IndexBuffer_Sphere_Index_RRJ){
			gpID3D11Buffer_IndexBuffer_Sphere_Index_RRJ->Release();
			gpID3D11Buffer_IndexBuffer_Sphere_Index_RRJ = NULL;
		}


		if(gpID3D11Buffer_VertexBuffer_Sphere_Position_RRJ){
			gpID3D11Buffer_VertexBuffer_Sphere_Position_RRJ->Release();
			gpID3D11Buffer_VertexBuffer_Sphere_Position_RRJ = NULL;
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
			fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "SUCCESS: Log Close !!\n");
			fprintf_s(gbFile_RRJ, "SUCCESS: END!!\n");
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



	/********** Render Target View **********/

	gpIDXGISwapChain_RRJ->ResizeBuffers(1, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, 0);


	ID3D11Texture2D *pID3D11Texture2D_BackBuffer_RRJ = NULL;

	gpIDXGISwapChain_RRJ->GetBuffer(0, __uuidof(ID3D11Texture2D),
		(LPVOID*)&pID3D11Texture2D_BackBuffer_RRJ);

	
	hr_RRJ = gpID3D11Device_RRJ->CreateRenderTargetView(pID3D11Texture2D_BackBuffer_RRJ,
		NULL, &gpID3D11RenderTargetView_RRJ);

	if(FAILED(hr_RRJ)){
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "ERROR: ID3D11Device::CreateRenderTargetView() Failed!!\n");
 		fclose(gbFile_RRJ);

 		return(hr_RRJ);
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: ID3D11Device::CreateRenderTargetView() Done!!\n");
 		fclose(gbFile_RRJ);
 	}


 	pID3D11Texture2D_BackBuffer_RRJ->Release();
 	pID3D11Texture2D_BackBuffer_RRJ = NULL;




 	/********** Depth Stencil View **********/
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

 	ID3D11Texture2D *pID3D11Texture2D_DepthBuffer_RRJ;
 	gpID3D11Device_RRJ->CreateTexture2D(&textureDesc_RRJ, NULL, &pID3D11Texture2D_DepthBuffer_RRJ);


 	//Create DSV From the Buffer Created Above
 	D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc_RRJ;
 	ZeroMemory((void*)&depthStencilViewDesc_RRJ, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));

 	depthStencilViewDesc_RRJ.Format = DXGI_FORMAT_D32_FLOAT;
 	depthStencilViewDesc_RRJ.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;

 	hr_RRJ = gpID3D11Device_RRJ->CreateDepthStencilView(pID3D11Texture2D_DepthBuffer_RRJ, 
 		&depthStencilViewDesc_RRJ, &gpID3D11DepthStencilView_RRJ);

 	if(FAILED(hr_RRJ)){
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "ERROR: ID3D11Device::CreateDepthStencilView() Failed!!\n");
 		fclose(gbFile_RRJ);

 		return(hr_RRJ);
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: ID3D11Device::CreateDepthStencilView() Done!!\n");
 		fclose(gbFile_RRJ);
 	}


 	gpID3D11DeviceContext_RRJ->OMSetRenderTargets(1, &gpID3D11RenderTargetView_RRJ, gpID3D11DepthStencilView_RRJ);


 	/********** ViewPort **********/
 	D3D11_VIEWPORT d3dViewPort_RRJ;
 	ZeroMemory((void*)&d3dViewPort_RRJ, sizeof(D3D11_VIEWPORT));

 	d3dViewPort_RRJ.TopLeftX = 0;
 	d3dViewPort_RRJ.TopLeftY = 0;
 	d3dViewPort_RRJ.Width = (float)width;
 	d3dViewPort_RRJ.Height = (float)height;
 	d3dViewPort_RRJ.MinDepth = 0.0f;
 	d3dViewPort_RRJ.MaxDepth = 1.0f;

 	gpID3D11DeviceContext_RRJ->RSSetViewports(1, &d3dViewPort_RRJ);


 	gPerspectiveProjectionMatrix_RRJ = XMMatrixPerspectiveFovLH(
 		XMConvertToRadians(45.0f), (float)width / (float)height, 0.1f, 100.0f);

 	return(S_OK);

}

void display(void){

	void my_glPushMatrix(XMMATRIX);
	XMMATRIX my_glPopMatrix(void);


	gpID3D11DeviceContext_RRJ->ClearRenderTargetView(gpID3D11RenderTargetView_RRJ, gClearColor_RRJ);

	gpID3D11DeviceContext_RRJ->ClearDepthStencilView(gpID3D11DepthStencilView_RRJ, D3D11_CLEAR_DEPTH, 1.0f, 0.0f);


	UINT stride = sizeof(float) * 3;
	UINT offset = 0;
	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(0, 1, &gpID3D11Buffer_VertexBuffer_Sphere_Position_RRJ, &stride, &offset);


	gpID3D11DeviceContext_RRJ->IASetIndexBuffer(gpID3D11Buffer_IndexBuffer_Sphere_Index_RRJ, DXGI_FORMAT_R16_UINT, 0);



	//Sun
	gpID3D11DeviceContext_RRJ->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

	XMMATRIX worldViewMatrix_RRJ = XMMatrixIdentity();
	XMMATRIX worldViewProjectionMatrix_RRJ = XMMatrixIdentity();


	worldViewMatrix_RRJ = XMMatrixTranslation(0.0f, 0.0f, 6.0);

	my_glPushMatrix(worldViewMatrix_RRJ);


	worldViewProjectionMatrix_RRJ = worldViewMatrix_RRJ * gPerspectiveProjectionMatrix_RRJ;
	CBUFFER constantBuffer_RRJ;

	constantBuffer_RRJ.WorldViewProjectionMatrix = worldViewProjectionMatrix_RRJ;
	constantBuffer_RRJ.Color = XMVectorSet(1.0f, 1.0f, 0.0f, 1.0f);

	gpID3D11DeviceContext_RRJ->UpdateSubresource(gpID3D11Buffer_ConstantBuffer_RRJ, 0,
		NULL, &constantBuffer_RRJ, 0, 0);


	gpID3D11DeviceContext_RRJ->DrawIndexed(gNumOfElements_RRJ, 0, 0);



	//Earth
	gpID3D11DeviceContext_RRJ->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP);
	worldViewMatrix_RRJ = my_glPopMatrix();


	worldViewMatrix_RRJ = XMMatrixRotationY(year_RRJ) * worldViewMatrix_RRJ;
	worldViewMatrix_RRJ = XMMatrixTranslation(2.0, 0.0, 0.0f) * worldViewMatrix_RRJ;

	my_glPushMatrix(worldViewMatrix_RRJ);

	worldViewMatrix_RRJ = XMMatrixScaling(0.5f, 0.5f, 0.5f) * worldViewMatrix_RRJ;
	worldViewMatrix_RRJ = XMMatrixRotationY(day_RRJ) * worldViewMatrix_RRJ;

	worldViewProjectionMatrix_RRJ = worldViewMatrix_RRJ * gPerspectiveProjectionMatrix_RRJ;
	constantBuffer_RRJ;

	constantBuffer_RRJ.WorldViewProjectionMatrix = worldViewProjectionMatrix_RRJ;
	constantBuffer_RRJ.Color = XMVectorSet(0.0f, 0.0f, 0.50f, 1.0f);

	gpID3D11DeviceContext_RRJ->UpdateSubresource(gpID3D11Buffer_ConstantBuffer_RRJ, 0,
		NULL, &constantBuffer_RRJ, 0, 0);


	gpID3D11DeviceContext_RRJ->DrawIndexed(gNumOfElements_RRJ, 0, 0);



	//Moon
	gpID3D11DeviceContext_RRJ->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	worldViewMatrix_RRJ = my_glPopMatrix();


	worldViewMatrix_RRJ = XMMatrixRotationY(moon_RRJ) * worldViewMatrix_RRJ;
	worldViewMatrix_RRJ = XMMatrixTranslation(1.0, 0.0, 0.0f) * worldViewMatrix_RRJ;

	my_glPushMatrix(worldViewMatrix_RRJ);

	worldViewMatrix_RRJ = XMMatrixScaling(0.2f, 0.2f, 0.2f) * worldViewMatrix_RRJ;
	
	worldViewProjectionMatrix_RRJ = worldViewMatrix_RRJ * gPerspectiveProjectionMatrix_RRJ;
	constantBuffer_RRJ;

	constantBuffer_RRJ.WorldViewProjectionMatrix = worldViewProjectionMatrix_RRJ;
	constantBuffer_RRJ.Color = XMVectorSet(0.500f, 0.500f, 0.50f, 1.0f);

	gpID3D11DeviceContext_RRJ->UpdateSubresource(gpID3D11Buffer_ConstantBuffer_RRJ, 0,
		NULL, &constantBuffer_RRJ, 0, 0);


	my_glPopMatrix();


	gpID3D11DeviceContext_RRJ->DrawIndexed(gNumOfElements_RRJ, 0, 0);



	gpIDXGISwapChain_RRJ->Present(0, 0);

}

void my_glPushMatrix(XMMATRIX matrix) {

	void uninitialize(void);

	ModelViewStack *temp_RRJ = (ModelViewStack*)malloc(sizeof(ModelViewStack));
	if (temp_RRJ == NULL) {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Malloc Failed!!\n");
		fclose(gbFile_RRJ);
		uninitialize();
		DestroyWindow(ghwnd_RRJ);
	}
	else {

		temp_RRJ->WorldViewMatrix = matrix;
		temp_RRJ->next = NULL;

		if (TopNode_RRJ == NULL) {
			TopNode_RRJ = temp_RRJ;
			TopNode_RRJ->prev = NULL;
			/*fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "Node Added!!\n");
			fclose(gbFile_RRJ);*/

		}
		else {
			TopNode_RRJ->next = temp_RRJ;
			temp_RRJ->prev = TopNode_RRJ;
			TopNode_RRJ = temp_RRJ;
			/*fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "Node Added!!\n");
			fclose(gbFile_RRJ);*/
		}
	}

	if (iTop_RRJ > MaxTop_RRJ) {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Stack Overflow!!\n");
		fclose(gbFile_RRJ);
		DestroyWindow(ghwnd_RRJ);
	}



}

XMMATRIX my_glPopMatrix(void) {

	void uninitialize(void);

	ModelViewStack *temp_RRJ = TopNode_RRJ;
	XMMATRIX matrix_RRJ;
	if (temp_RRJ->prev != NULL) {
		TopNode_RRJ = temp_RRJ->prev;
		temp_RRJ->next = NULL;
		temp_RRJ->prev = NULL;
		matrix_RRJ = temp_RRJ->WorldViewMatrix;
		/*fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "Node Delete!!\n");
		fclose(gbFile_RRJ);*/
		free(temp_RRJ);
	}
	else {
		temp_RRJ->next = NULL;
		temp_RRJ->prev = NULL;
		matrix_RRJ = temp_RRJ->WorldViewMatrix;
		/*fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "Node Delete!!\n");
		fclose(gbFile_RRJ);*/
		free(temp_RRJ);
		TopNode_RRJ = NULL;
	}
	return(matrix_RRJ);

}




