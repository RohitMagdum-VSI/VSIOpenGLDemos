#include<windows.h>
#include<stdio.h>

#include<d3d11.h>
#include<D3dcompiler.h>

#pragma warning(disable: 4838)

#include"XNAMath/xnamath.h"

#include"WICTextureLoader.h"


#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "D3dcompiler.lib")
#pragma comment(lib, "DirectXTK.lib")

using namespace DirectX;


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


//For Shader
ID3D11VertexShader *gpID3D11VertexShader_RRJ = NULL;
ID3D11PixelShader *gpID3D11PixelShader_RRJ = NULL;


//For Rectangle
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Rect_Position_RRJ = NULL;
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Rect_Texcoord_RRJ = NULL;

//For Layout/ Attributes
ID3D11InputLayout *gpID3D11InputLayout_RRJ = NULL;

//For Uniform
ID3D11Buffer *gpID3D11Buffer_ConstantBuffer_RRJ = NULL;


//For Texture
ID3D11SamplerState *gpID3D11SamplerState_Rect_Texture_RRJ = NULL;
ID3D11ShaderResourceView *gpID3D11ShaderResourceView_Rect_Texture_RRJ = NULL;

//For ClearTexcoord
float gClearTexcoord_RRJ[4];

struct CBUFFER{
	XMMATRIX WorldViewProjectionMatrix;
};


//For Texture
int iTexcoord_RRJ = 1;

XMMATRIX gPerspectiveProjectionMatrix_RRJ;

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
	TCHAR szName_RRJ[] = TEXT("Rohit_R_Jadhav-D3D-17-TwickedSmiley");

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
			TEXT("Rohit_R_Jadhav-D3D-17-TwickedSmiley"),
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

				case '1':
					iTexcoord_RRJ = 1;
					break;

				case '2':
					iTexcoord_RRJ = 2;
					break;

				case '3':
					iTexcoord_RRJ = 3;
					break;

				case '4':
					iTexcoord_RRJ = 4;
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
 		"Texture2D myTexture2D;" \
		"SamplerState samplerState;" \

		"float4 main(float4 pos : SV_POSITION, float2 tex: TEXCOORD) : SV_TARGET {" \
			"float4 color = myTexture2D.Sample(samplerState, tex);" \
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
 	D3D11_INPUT_ELEMENT_DESC inputElementDesc_RRJ[2];
 	ZeroMemory((void*)&inputElementDesc_RRJ, sizeof(D3D11_INPUT_ELEMENT_DESC));

 	//For vPosition
 	inputElementDesc_RRJ[0].SemanticName = "POSITION";
 	inputElementDesc_RRJ[0].SemanticIndex = 0;
 	inputElementDesc_RRJ[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
 	inputElementDesc_RRJ[0].InputSlot = 0;
 	inputElementDesc_RRJ[0].AlignedByteOffset = 0;
 	inputElementDesc_RRJ[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
 	inputElementDesc_RRJ[0].InstanceDataStepRate = 0;


 	//For vTexcoord
 	inputElementDesc_RRJ[1].SemanticName = "TEXCOORD";
 	inputElementDesc_RRJ[1].SemanticIndex = 0;
 	inputElementDesc_RRJ[1].Format = DXGI_FORMAT_R32G32_FLOAT;
 	inputElementDesc_RRJ[1].InputSlot = 1;
 	inputElementDesc_RRJ[1].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
 	inputElementDesc_RRJ[1].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
 	inputElementDesc_RRJ[1].InstanceDataStepRate = 0;



 	hr_RRJ = gpID3D11Device_RRJ->CreateInputLayout(inputElementDesc_RRJ, _ARRAYSIZE(inputElementDesc_RRJ),
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
 	float rect_Position[] = {
 		-1.0f, 1.0f, 0.0f,
 		1.0f, 1.0f, 0.0f,
 		-1.0f, -1.0f, 0.0f,

 		-1.0f, -1.0f, 0.0f,
 		1.0f, 1.0f, 0.0f,
 		1.0f, -1.0f, 0.0f,
 	};

 	




 	/********** Vertex Buffer Rect_Position **********/
 	D3D11_BUFFER_DESC bufferDesc_Rect_Pos_RRJ;
 	ZeroMemory((void*)&bufferDesc_Rect_Pos_RRJ, sizeof(D3D11_BUFFER_DESC));

 	bufferDesc_Rect_Pos_RRJ.Usage = D3D11_USAGE_DYNAMIC;
 	bufferDesc_Rect_Pos_RRJ.ByteWidth = sizeof(float) * _ARRAYSIZE(rect_Position);
 	bufferDesc_Rect_Pos_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
 	bufferDesc_Rect_Pos_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

 	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_Rect_Pos_RRJ, NULL, &gpID3D11Buffer_VertexBuffer_Rect_Position_RRJ);

 	if(FAILED(hr_RRJ)){
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "ERROR: ID3D11Device::CreateBuffer() For Rect_Position Failed!!\n");
 		fclose(gbFile_RRJ);

 		return(hr_RRJ);
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: ID3D11Device::CreateBuffer() For Rect_Position Done!!\n");
 		fclose(gbFile_RRJ);
 	}




 	/********** Memory Mapped I/O **********/
 	D3D11_MAPPED_SUBRESOURCE mappedSubresource_Rect_Position_RRJ;
 	ZeroMemory((void*)&mappedSubresource_Rect_Position_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));

 	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_Rect_Position_RRJ, 0, 
 		D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource_Rect_Position_RRJ);
 	memcpy(mappedSubresource_Rect_Position_RRJ.pData, rect_Position, sizeof(rect_Position));
 	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_Rect_Position_RRJ, NULL);






 	/********** Vertex Buffer For Rect Texcoord **********/
 	D3D11_BUFFER_DESC bufferDesc_Rect_Texcoord_RRJ;
 	ZeroMemory((void*)&bufferDesc_Rect_Texcoord_RRJ, sizeof(D3D11_BUFFER_DESC));

 	bufferDesc_Rect_Texcoord_RRJ.Usage = D3D11_USAGE_DYNAMIC;
 	bufferDesc_Rect_Texcoord_RRJ.ByteWidth = sizeof(float) * 2 * 3 * 2;
 	bufferDesc_Rect_Texcoord_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
 	bufferDesc_Rect_Texcoord_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

 	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_Rect_Texcoord_RRJ, NULL,
 		&gpID3D11Buffer_VertexBuffer_Rect_Texcoord_RRJ);

 	if(FAILED(hr_RRJ)){
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "ERROR: ID3D11::CreateBuffer() for Rect_Texcoord Failed!!\n");
 		fclose(gbFile_RRJ);
 		return(hr_RRJ);
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: ID3D11::CreateBuffer() for Rect_Texcoord Done!!\n");
 		fclose(gbFile_RRJ);	
 	}



 	



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




 	gClearTexcoord_RRJ[0] = 0.0f;
 	gClearTexcoord_RRJ[1] = 0.0f;
 	gClearTexcoord_RRJ[2] = 0.0f;
 	gClearTexcoord_RRJ[3] = 1.0f;





 	/********** LOAD TEXTURE **********/
	hr_RRJ = LoadD3DTexture(L"Smiley.bmp", &gpID3D11ShaderResourceView_Rect_Texture_RRJ);
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
		&gpID3D11SamplerState_Rect_Texture_RRJ);

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


HRESULT LoadD3DTexture(const wchar_t *fileName, ID3D11ShaderResourceView **ppShaderResourceView){

	HRESULT hr_RRJ = S_OK;


	hr_RRJ = CreateWICTextureFromFile(
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


		if(gpID3D11SamplerState_Rect_Texture_RRJ){
			gpID3D11SamplerState_Rect_Texture_RRJ->Release();
			gpID3D11SamplerState_Rect_Texture_RRJ = NULL;
		}

		if(gpID3D11ShaderResourceView_Rect_Texture_RRJ){
			gpID3D11ShaderResourceView_Rect_Texture_RRJ->Release();
			gpID3D11ShaderResourceView_Rect_Texture_RRJ = NULL;
		}


		if(gpID3D11Buffer_ConstantBuffer_RRJ){
			gpID3D11Buffer_ConstantBuffer_RRJ->Release();
			gpID3D11Buffer_ConstantBuffer_RRJ = NULL;
		}

		if(gpID3D11InputLayout_RRJ){
			gpID3D11InputLayout_RRJ->Release();
			gpID3D11InputLayout_RRJ = NULL;
		}

		if(gpID3D11Buffer_VertexBuffer_Rect_Texcoord_RRJ){
			gpID3D11Buffer_VertexBuffer_Rect_Texcoord_RRJ->Release();
			gpID3D11Buffer_VertexBuffer_Rect_Texcoord_RRJ = NULL;
		}


		if(gpID3D11Buffer_VertexBuffer_Rect_Position_RRJ){
			gpID3D11Buffer_VertexBuffer_Rect_Position_RRJ->Release();
			gpID3D11Buffer_VertexBuffer_Rect_Position_RRJ = NULL;
		}

		if(gpID3D11PixelShader_RRJ){
			gpID3D11PixelShader_RRJ->Release();
			gpID3D11PixelShader_RRJ = NULL;
		}

		if(gpID3D11VertexShader_RRJ){
			gpID3D11VertexShader_RRJ->Release();
			gpID3D11VertexShader_RRJ = NULL;
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

	if(gpID3D11RenderTargetView_RRJ){
		gpID3D11RenderTargetView_RRJ->Release();
		gpID3D11RenderTargetView_RRJ = NULL;
	}


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


 	gpID3D11DeviceContext_RRJ->OMSetRenderTargets(1, &gpID3D11RenderTargetView_RRJ, NULL);


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

	gpID3D11DeviceContext_RRJ->ClearRenderTargetView(gpID3D11RenderTargetView_RRJ, gClearTexcoord_RRJ);

	UINT stride = sizeof(float) * 3;
	UINT offset = 0;
	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(0, 1, &gpID3D11Buffer_VertexBuffer_Rect_Position_RRJ, &stride, &offset);

	float rect_Texcoord[12] = {
 		0.0f, 0.0f,
 		1.0f, 0.0f,
 		0.0f, 1.0f,

 		0.0f, 1.0f,
 		1.0f, 0.0f,
 		1.0f, 1.0f,
 	};

	if(iTexcoord_RRJ == 1){
		rect_Texcoord[0] = 0.0f;
		rect_Texcoord[1] = 0.50f;
		rect_Texcoord[2] = 0.50f;
		rect_Texcoord[3] = 0.50f;
		rect_Texcoord[4] = 0.0f;
		rect_Texcoord[5] = 1.0f;

		rect_Texcoord[6] = 0.0f;
		rect_Texcoord[7] = 1.0f;
		rect_Texcoord[8] = 0.50f;
		rect_Texcoord[9] = 0.50f;
		rect_Texcoord[10] = 0.50f;
		rect_Texcoord[11] = 1.0f;

	}
	else if(iTexcoord_RRJ == 2){
		rect_Texcoord[0] = 0.0f;
		rect_Texcoord[1] = 0.0f;
		rect_Texcoord[2] = 1.0f;
		rect_Texcoord[3] = 0.0f;
		rect_Texcoord[4] = 0.0f;
		rect_Texcoord[5] = 1.0f;

		rect_Texcoord[6] = 0.0f;
		rect_Texcoord[7] = 1.0f;
		rect_Texcoord[8] = 1.0f;
		rect_Texcoord[9] = 0.0f;
		rect_Texcoord[10] = 1.0f;
		rect_Texcoord[11] = 1.0f;
	}
	else if(iTexcoord_RRJ == 3){
		rect_Texcoord[0] = 0.0f;
		rect_Texcoord[1] = 0.0f;
		rect_Texcoord[2] = 2.0f;
		rect_Texcoord[3] = 0.0f;
		rect_Texcoord[4] = 0.0f;
		rect_Texcoord[5] = 2.0f;

		rect_Texcoord[6] = 0.0f;
		rect_Texcoord[7] = 2.0f;
		rect_Texcoord[8] = 2.0f;
		rect_Texcoord[9] = 0.0f;
		rect_Texcoord[10] = 2.0f;
		rect_Texcoord[11] = 2.0f;
	}
	else if(iTexcoord_RRJ == 4){
		rect_Texcoord[0] = 0.50f;
		rect_Texcoord[1] = 0.50f;
		rect_Texcoord[2] = 0.50f;
		rect_Texcoord[3] = 0.50f;
		rect_Texcoord[4] = 0.50f;
		rect_Texcoord[5] = 0.50f;

		rect_Texcoord[6] = 0.50f;
		rect_Texcoord[7] = 0.50f;
		rect_Texcoord[8] = 0.50f;
		rect_Texcoord[9] = 0.50f;
		rect_Texcoord[10] = 0.50f;
		rect_Texcoord[11] = 0.50f;
	}



	/********** Memory Map I/O For Rect_Texcoord **********/
 	D3D11_MAPPED_SUBRESOURCE mappedSubresource_Rect_Texcoord_RRJ;
 	ZeroMemory((void*)&mappedSubresource_Rect_Texcoord_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));

 	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_Rect_Texcoord_RRJ, 0, 
 		D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource_Rect_Texcoord_RRJ);

 	memcpy(mappedSubresource_Rect_Texcoord_RRJ.pData, rect_Texcoord, sizeof(rect_Texcoord));
 	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_Rect_Texcoord_RRJ, 0);


	UINT stride_Texcoord = sizeof(float) * 2;
	UINT offset_Texcoord = 0;
	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(1, 1, &gpID3D11Buffer_VertexBuffer_Rect_Texcoord_RRJ, &stride_Texcoord, &offset_Texcoord);


	gpID3D11DeviceContext_RRJ->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);


	/********** Sampler and Shader Resource View **********/
	gpID3D11DeviceContext_RRJ->PSSetShaderResources(0, 1, &gpID3D11ShaderResourceView_Rect_Texture_RRJ);
	gpID3D11DeviceContext_RRJ->PSSetSamplers(0, 1, &gpID3D11SamplerState_Rect_Texture_RRJ);


	XMMATRIX translateMatrix_RRJ = XMMatrixIdentity();
	XMMATRIX worldViewMatrix_RRJ = XMMatrixIdentity();
	XMMATRIX worldViewProjectionMatrix_RRJ	 = XMMatrixIdentity();

	translateMatrix_RRJ = XMMatrixTranslation(0.0f, 0.0f, 3.0f);
	worldViewMatrix_RRJ =  translateMatrix_RRJ;
	worldViewProjectionMatrix_RRJ = worldViewMatrix_RRJ * gPerspectiveProjectionMatrix_RRJ;

	CBUFFER constantBuffer;
	constantBuffer.WorldViewProjectionMatrix = worldViewProjectionMatrix_RRJ;
	gpID3D11DeviceContext_RRJ->UpdateSubresource(gpID3D11Buffer_ConstantBuffer_RRJ, 0,
		NULL, &constantBuffer, 0, 0);

	gpID3D11DeviceContext_RRJ->Draw(6, 0);

	gpIDXGISwapChain_RRJ->Present(0, 0);
}

