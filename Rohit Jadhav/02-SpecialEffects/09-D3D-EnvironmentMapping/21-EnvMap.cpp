#include<windows.h>
#include<stdio.h>

#include<d3d11.h>
#include<D3dcompiler.h>

#pragma warning(disable: 4838)

#include"XNAMath/xnamath.h"

#include"DDSTextureLoader.h"

#include<math.h>
#include<assert.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "D3dcompiler.lib")
#pragma comment(lib, "DirectXTK.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

#define CUBEMAP_WIDTH 2048
#define CUBEMAP_HEIGHT 2048

using namespace DirectX;



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


//For Cubeangle
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Cube_Position_RRJ = NULL;
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Cube_Texcoord_RRJ = NULL;

//For Layout/ Attributes
ID3D11InputLayout *gpID3D11InputLayout_RRJ = NULL;

//For Uniform
ID3D11Buffer *gpID3D11Buffer_ConstantBuffer_RRJ = NULL;

//For Texture
ID3D11SamplerState *gpID3D11SamplerState_Cube_Texture_RRJ = NULL;
ID3D11ShaderResourceView *gpID3D11ShaderResourceView_Cube_Texture_RRJ = NULL;


//For ClearColor
float gClearColor_RRJ[4];

struct CBUFFER{
	XMMATRIX WorldMatrix;
	XMMATRIX ViewMatrix;
	XMMATRIX ProjectionMatrix;
	XMMATRIX RotateMatrix;

	int CubeMapToggle;
	
	XMVECTOR La;
	XMVECTOR Ld;
	XMVECTOR Ls;
	XMVECTOR LightPosition;

	XMVECTOR Ka;
	XMVECTOR Kd;
	XMVECTOR Ks;
	float Shininess;

	unsigned int LKeyPress;
};


XMMATRIX gPerspectiveProjectionMatrix_RRJ;


//For Culling
ID3D11RasterizerState *gpID3D11RasterizerState_RRJ = NULL;


//For EnvMap
wchar_t *gszCubeMapName = L"sky.dds";


//For Model
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Model_Pos_RRJ = NULL;
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Model_Normal_RRJ = NULL;
ID3D11Buffer *gpID3D11Buffer_IndexBuffer_Model_Index_RRJ = NULL;


//For Lights and Material
bool bLights_RRJ = false;

float lightAmbient_RRJ[] = {0.0f, 0.0f, 0.0f, 1.0f};
float lightDiffuse_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
float lightSpecular_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
float lightPosition_RRJ[] = {100.0f, 100.0f, -100.0f, 1.0f};

float materialAmbient_RRJ[] = {0.0f, 0.0f, 0.0f, 1.0f};
float materialDiffuse_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
float materialSpecular_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
float materialShininess_RRJ = 128.0f;

//For Model Loading
struct VecFloat {
	float *pData;
	int iSize;
};


#define RRJ_SUCCESS 1
#define RRJ_ERROR 0


struct VecFloat *pVecFloat_Model_Vertices = NULL;
struct VecFloat *pVecFloat_Model_Normals = NULL;
struct VecFloat *pVecFloat_Model_Texcoord = NULL;
struct VecFloat *pVecFloat_Model_FaceIndices = NULL;
struct VecFloat *pVecFloat_Model_Sorted_Vertices = NULL;
struct VecFloat *pVecFloat_Model_Sorted_Normals = NULL;
struct VecFloat *pVecFloat_Model_Sorted_Texcoord = NULL;
struct VecFloat *pVecFloat_Model_Elements = NULL;

int PushBackVecFloat(struct VecFloat*, float);
void ShowVecFloat(struct VecFloat*);
struct VecFloat* CreateVecFloat(void);
int DestroyVecFloat(struct VecFloat*);


FILE *gbFile_Model = NULL;




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

	fopen_s(&gbFile_Model, "head.txt", "r");
	if (gbFile_Model == NULL) {
		MessageBox(NULL, TEXT("ERROR: Main(): Model File fopen() Failed!!\n"), TEXT("ERROR"), MB_OK);
		exit(0);
	}


	WNDCLASSEX wndclass_RRJ;
	HWND hwnd_RRJ;
	MSG msg_RRJ;
	TCHAR szName_RRJ[] = TEXT("Rohit_R_Jadhav-D3D-21-Environmental_Map");

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
			TEXT("Rohit_R_Jadhav-D3D-21-Environmental_Map"),
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

				case 'L':
				case 'l':
					if(bLights_RRJ == false)
						bLights_RRJ = true;
					else
						bLights_RRJ = false;
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
	void LoadModel(void);


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
			"float4x4 worldMatrix;" \
			"float4x4 viewMatrix;" \
			"float4x4 projectionMatrix;" \
			"float4x4 rotateMatrix;" \

			"int cubeMapToggle;" \

			"float4 la;" \
			"float4 ld;" \
			"float4 ls;" \
			"float4 lightPosition;" \

			"float4 ka;" \
			"float4 kd;" \
			"float4 ks;" \
			"float shininess;" \

			"uint keyPress;" \

		"};" \


		"struct vertex_output {" \
			"float4 position : SV_POSITION;" \
			"float3 texcoord : TEXCOORD;" \
			"float3 lightDirection : NORMAL0;" \
			"float3 transformedNormal : NORMAL1;" \
			"float3 viewerVec : NORMAL2;" \

		"};" \

		"vertex_output main(float4 pos : POSITION, float2 tex : TEXCOORD, float4 normal : NORMAL){ " \

			"if(cubeMapToggle == 1){" \

				"vertex_output v;" \

				"v.texcoord = pos.xyz;" \

				"v.position = mul(worldMatrix, pos);" \
				"v.position = mul(viewMatrix, v.position);" \
				"v.position = mul(projectionMatrix, v.position);" \
				
				"return(v); " \

			"}" \

			"else{" \

				"vertex_output v;" \
				
				"v.texcoord = mul(rotateMatrix, pos);" \


				"float4 eyeCoord = mul(worldMatrix, pos);" \
				"eyeCoord = mul(viewMatrix, eyeCoord);" \

				"float3x3 normalMatrix = (float3x3)mul(worldMatrix, viewMatrix);" \
				"v.transformedNormal = (float3)mul(normalMatrix, (float3)normal);" \

				"v.viewerVec = (-eyeCoord.xyz);" \

				"v.lightDirection = (float3)(lightPosition - eyeCoord);" \

				"v.position = mul(worldMatrix, pos);" \
				"v.position = mul(viewMatrix, v.position);" \
				"v.position = mul(projectionMatrix, v.position);" \

				// "v.texcoord = v.position.xyz;" \
				// "v.texcoord.z = -v.texcoord.z;" \

				"return(v);" \

			"}" \

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
			"float4x4 worldMatrix;" \
			"float4x4 viewMatrix;" \
			"float4x4 projectionMatrix;" \
			"float4x4 rotateMatrix;" \

			"int cubeMapToggle;" \

			"float4 la;" \
			"float4 ld;" \
			"float4 ls;" \
			"float4 lightPosition;" \

			"float4 ka;" \
			"float4 kd;" \
			"float4 ks;" \
			"float shininess;" \

			"uint keyPress;" \

		"};" \


		"struct vertex_output {" \
			"float4 position : SV_POSITION;" \
			"float3 texcoord : TEXCOORD;" \
			"float3 lightDirection : NORMAL0;" \
			"float3 transformedNormal : NORMAL1;" \
			"float3 viewerVec : NORMAL2;" \

		"};" \



 		"TextureCube myTextureCube;" \
		"SamplerState samplerState;" \


		"float4 main(float4 pos : SV_POSITION, vertex_output inVertex) : SV_TARGET {" \

			"if(cubeMapToggle == 1){" \

				"float4 color = myTextureCube.Sample(samplerState, inVertex.texcoord);" \
				"return(color);" \

			"}" \

			"else{" \


				"float4 phong_ads_color;" \
				"if(keyPress == 1) {" \

					"float3 normalizeLightDirection = normalize(inVertex.lightDirection);" \
					"float3 normalizeTransformedNormal = normalize(inVertex.transformedNormal);" \
					"float S_Dot_N = max(dot(normalizeLightDirection, normalizeTransformedNormal), 0.0);" \

					"float3 normalizeViewerVec = normalize(inVertex.viewerVec);" \
					"float3 reflectionVec = reflect(-normalizeLightDirection, normalizeTransformedNormal);" \
					"float R_Dot_V = max(dot(reflectionVec, normalizeViewerVec), 0.0);" \

					"float4 ambient = la * ka;" \
					"float4 diffuse = ld * kd * S_Dot_N;" \
					"float4 specular = ls * ks * max(pow(R_Dot_V, shininess), 0.0);" \

					"phong_ads_color = ambient + diffuse + specular;" \

				"}" \
				"else{" \
					"phong_ads_color = float4(1.0f, 1.0f, 1.0f, 1.0f);" \
				"}" \

				"float4 envCol = myTextureCube.Sample(samplerState, inVertex.texcoord);" \
				"float4 color = phong_ads_color;" \

				"float4 col = lerp(envCol, color, 0.25);" \
				"return(col);" \

			"}" \

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
 	D3D11_INPUT_ELEMENT_DESC inputElementDesc_RRJ[3];
 	ZeroMemory((void*)&inputElementDesc_RRJ, sizeof(D3D11_INPUT_ELEMENT_DESC));

 	//For vPosition
 	inputElementDesc_RRJ[0].SemanticName = "POSITION";
 	inputElementDesc_RRJ[0].SemanticIndex = 0;
 	inputElementDesc_RRJ[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
 	inputElementDesc_RRJ[0].InputSlot = 0;
 	inputElementDesc_RRJ[0].AlignedByteOffset = 0;
 	inputElementDesc_RRJ[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
 	inputElementDesc_RRJ[0].InstanceDataStepRate = 0;


 	//For vColor
 	inputElementDesc_RRJ[1].SemanticName = "TEXCOORD";
 	inputElementDesc_RRJ[1].SemanticIndex = 0;
 	inputElementDesc_RRJ[1].Format = DXGI_FORMAT_R32G32_FLOAT;
 	inputElementDesc_RRJ[1].InputSlot = 1;
 	inputElementDesc_RRJ[1].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
 	inputElementDesc_RRJ[1].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
 	inputElementDesc_RRJ[1].InstanceDataStepRate = 0;


 	// For Normals
 	inputElementDesc_RRJ[2].SemanticName = "NORMAL";
 	inputElementDesc_RRJ[2].SemanticIndex = 0;
 	inputElementDesc_RRJ[2].Format = DXGI_FORMAT_R32G32B32_FLOAT;
 	inputElementDesc_RRJ[2].InputSlot = 2;
 	inputElementDesc_RRJ[2].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
 	inputElementDesc_RRJ[2].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
 	inputElementDesc_RRJ[2].InstanceDataStepRate = 0;




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




 	/********** Vertex Buffer Cube_Position **********/
 	D3D11_BUFFER_DESC bufferDesc_Cube_Pos_RRJ;
 	ZeroMemory((void*)&bufferDesc_Cube_Pos_RRJ, sizeof(D3D11_BUFFER_DESC));

 	bufferDesc_Cube_Pos_RRJ.Usage = D3D11_USAGE_DYNAMIC;
 	bufferDesc_Cube_Pos_RRJ.ByteWidth = sizeof(float) * _ARRAYSIZE(cube_Position);
 	bufferDesc_Cube_Pos_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
 	bufferDesc_Cube_Pos_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

 	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_Cube_Pos_RRJ, NULL, &gpID3D11Buffer_VertexBuffer_Cube_Position_RRJ);

 	if(FAILED(hr_RRJ)){
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "ERROR: ID3D11Device::CreateBuffer() For Cube_Position Failed!!\n");
 		fclose(gbFile_RRJ);

 		return(hr_RRJ);
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: ID3D11Device::CreateBuffer() For Cube_Position Done!!\n");
 		fclose(gbFile_RRJ);
 	}




 	/********** Memory Mapped I/O **********/
 	D3D11_MAPPED_SUBRESOURCE mappedSubresource_Cube_Position_RRJ;
 	ZeroMemory((void*)&mappedSubresource_Cube_Position_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));

 	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_Cube_Position_RRJ, 0, 
 		D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource_Cube_Position_RRJ);
 	memcpy(mappedSubresource_Cube_Position_RRJ.pData, cube_Position, sizeof(cube_Position));
 	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_Cube_Position_RRJ, NULL);






 	/********** Vertex Buffer For Cube TEXCOORD **********/
 	D3D11_BUFFER_DESC bufferDesc_Cube_Texcoord_RRJ;
 	ZeroMemory((void*)&bufferDesc_Cube_Texcoord_RRJ, sizeof(D3D11_BUFFER_DESC));

 	bufferDesc_Cube_Texcoord_RRJ.Usage = D3D11_USAGE_DYNAMIC;
 	bufferDesc_Cube_Texcoord_RRJ.ByteWidth = sizeof(float) * _ARRAYSIZE(cube_Texcoord);
 	bufferDesc_Cube_Texcoord_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
 	bufferDesc_Cube_Texcoord_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

 	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_Cube_Texcoord_RRJ, NULL,
 		&gpID3D11Buffer_VertexBuffer_Cube_Texcoord_RRJ);

 	if(FAILED(hr_RRJ)){
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "ERROR: ID3D11::CreateBuffer() for Cube_Texcoord Failed!!\n");
 		fclose(gbFile_RRJ);
 		return(hr_RRJ);
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: ID3D11::CreateBuffer() for Cube_Color Done!!\n");
 		fclose(gbFile_RRJ);	
 	}



 	/********** Memory Map I/O For Cube_Color **********/
 	D3D11_MAPPED_SUBRESOURCE mappedSubresource_Cube_Texcoord_RRJ;
 	ZeroMemory((void*)&mappedSubresource_Cube_Texcoord_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));

 	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_Cube_Texcoord_RRJ, 0, 
 		D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource_Cube_Texcoord_RRJ);

 	memcpy(mappedSubresource_Cube_Texcoord_RRJ.pData, cube_Texcoord, sizeof(cube_Texcoord));
 	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_Cube_Texcoord_RRJ, 0);



 	





	/********** Position, Normal and Elements **********/
	pVecFloat_Model_Vertices = CreateVecFloat();
	pVecFloat_Model_Normals = CreateVecFloat();
	pVecFloat_Model_Texcoord = CreateVecFloat();

	pVecFloat_Model_Elements = CreateVecFloat();

	pVecFloat_Model_Sorted_Vertices = CreateVecFloat();
	pVecFloat_Model_Sorted_Normals = CreateVecFloat();
	pVecFloat_Model_Sorted_Texcoord = CreateVecFloat();


	LoadModel();
		



		/********** SPHERE POSITION **********/
	D3D11_BUFFER_DESC d3dBufferDesc_Model_Pos_RRJ;
	ZeroMemory((void*)&d3dBufferDesc_Model_Pos_RRJ, sizeof(D3D11_BUFFER_DESC));

	d3dBufferDesc_Model_Pos_RRJ.Usage = D3D11_USAGE_DYNAMIC;
	d3dBufferDesc_Model_Pos_RRJ.ByteWidth = sizeof(float) * pVecFloat_Model_Sorted_Vertices->iSize;
	d3dBufferDesc_Model_Pos_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	d3dBufferDesc_Model_Pos_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(
		&d3dBufferDesc_Model_Pos_RRJ,
		NULL,
		&gpID3D11Buffer_VertexBuffer_Model_Pos_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Model Position CreateBuffer()\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Model Position CreateBuffer()\n");
		fclose(gbFile_RRJ);	
	}



	/********** SPHERE POSITION MAPPING **********/
	D3D11_MAPPED_SUBRESOURCE d3dMappedSubresource_Model_Pos_RRJ;
	ZeroMemory((void*)&d3dMappedSubresource_Model_Pos_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));

	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_Model_Pos_RRJ, 0,
		D3D11_MAP_WRITE_DISCARD, 0, &d3dMappedSubresource_Model_Pos_RRJ);
	memcpy(d3dMappedSubresource_Model_Pos_RRJ.pData, pVecFloat_Model_Sorted_Vertices->pData, sizeof(float) * pVecFloat_Model_Sorted_Vertices->iSize);
	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_Model_Pos_RRJ, 0);




	/********** SPHERE NORMAL **********/
	D3D11_BUFFER_DESC bufferDesc_Model_Normal_RRJ;
	ZeroMemory((void*)&bufferDesc_Model_Normal_RRJ, sizeof(D3D11_BUFFER_DESC));

	bufferDesc_Model_Normal_RRJ.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc_Model_Normal_RRJ.ByteWidth = sizeof(float) * pVecFloat_Model_Sorted_Normals->iSize;
	bufferDesc_Model_Normal_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc_Model_Normal_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;


	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(
		&bufferDesc_Model_Normal_RRJ,
		NULL,
		&gpID3D11Buffer_VertexBuffer_Model_Normal_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Model Normal CreateBuffer()\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Model Normal CreateBuffer()\n");
		fclose(gbFile_RRJ);	
	}


	/********** SPHERE NORMAL MAPPING **********/
	D3D11_MAPPED_SUBRESOURCE mappedSubresource_Model_Normal_RRJ;
	ZeroMemory((void*)&mappedSubresource_Model_Normal_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));

	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_Model_Normal_RRJ, 0,
		D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource_Model_Normal_RRJ);

	memcpy(mappedSubresource_Model_Normal_RRJ.pData, pVecFloat_Model_Sorted_Normals->pData, sizeof(float) * pVecFloat_Model_Sorted_Normals->iSize);

	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_Model_Normal_RRJ, 0);









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


 

 	/********** LOAD TEXTURE **********/
	hr_RRJ = LoadD3DTexture(gszCubeMapName, &gpID3D11ShaderResourceView_Cube_Texture_RRJ);
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
		&gpID3D11SamplerState_Cube_Texture_RRJ);

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


	hr_RRJ = CreateDDSTextureFromFile(
		gpID3D11Device_RRJ,
		gpID3D11DeviceContext_RRJ,
		fileName, NULL, ppShaderResourceView);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: %ws\n", fileName);
		fprintf_s(gbFile_RRJ, "ERROR: LoadD3DTexture: DirectX::CreateDDSTextureFromFile() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: LoadD3DTexture: DirectX::CreateDDSTextureFromFile() Done!!\n");
		fclose(gbFile_RRJ);
	}

	return(S_OK);
}



void uninitialize(void){

		if(gpID3D11SamplerState_Cube_Texture_RRJ){
			gpID3D11SamplerState_Cube_Texture_RRJ->Release();
			gpID3D11SamplerState_Cube_Texture_RRJ = NULL;
		}

		if(gpID3D11ShaderResourceView_Cube_Texture_RRJ){
			gpID3D11ShaderResourceView_Cube_Texture_RRJ->Release();
			gpID3D11ShaderResourceView_Cube_Texture_RRJ = NULL;
		}



		if (pVecFloat_Model_Sorted_Texcoord) {
			DestroyVecFloat(pVecFloat_Model_Sorted_Texcoord);
			pVecFloat_Model_Sorted_Texcoord = NULL;
		}

		if (pVecFloat_Model_Sorted_Normals) {
			DestroyVecFloat(pVecFloat_Model_Sorted_Normals);
			pVecFloat_Model_Sorted_Normals = NULL;
		}


		if (pVecFloat_Model_Sorted_Vertices) {
			DestroyVecFloat(pVecFloat_Model_Sorted_Vertices);
			pVecFloat_Model_Sorted_Vertices = NULL;
		}


		if (pVecFloat_Model_Normals) {
			DestroyVecFloat(pVecFloat_Model_Normals);
			pVecFloat_Model_Normals = NULL;
		}

		if (pVecFloat_Model_Texcoord) {
			DestroyVecFloat(pVecFloat_Model_Texcoord);
			pVecFloat_Model_Texcoord = NULL;
		}

		if (pVecFloat_Model_Vertices) {
			DestroyVecFloat(pVecFloat_Model_Vertices);
			pVecFloat_Model_Vertices = NULL;
		}


		if (pVecFloat_Model_Elements) {
			DestroyVecFloat(pVecFloat_Model_Elements);
			pVecFloat_Model_Elements = NULL;
		}


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

		if(gpID3D11Buffer_VertexBuffer_Cube_Texcoord_RRJ){
			gpID3D11Buffer_VertexBuffer_Cube_Texcoord_RRJ->Release();
			gpID3D11Buffer_VertexBuffer_Cube_Texcoord_RRJ = NULL;
		}


		if(gpID3D11Buffer_VertexBuffer_Cube_Position_RRJ){
			gpID3D11Buffer_VertexBuffer_Cube_Position_RRJ->Release();
			gpID3D11Buffer_VertexBuffer_Cube_Position_RRJ = NULL;
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


	static float angle_Cube = 0.0f;

	gpID3D11DeviceContext_RRJ->ClearRenderTargetView(gpID3D11RenderTargetView_RRJ, gClearColor_RRJ);

	gpID3D11DeviceContext_RRJ->ClearDepthStencilView(gpID3D11DepthStencilView_RRJ,
		D3D11_CLEAR_DEPTH, 1.0f, 0.0f);

	UINT stride = sizeof(float) * 3;
	UINT offset = 0;
	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(0, 1, &gpID3D11Buffer_VertexBuffer_Cube_Position_RRJ, &stride, &offset);


	UINT stride_Texcoord = sizeof(float) * 2;
	UINT offset_Texcoord = 0;
	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(1, 1, &gpID3D11Buffer_VertexBuffer_Cube_Texcoord_RRJ, &stride_Texcoord, &offset_Texcoord);



	/********** Sampler and Shader Resource View **********/
	gpID3D11DeviceContext_RRJ->PSSetShaderResources(0, 1, &gpID3D11ShaderResourceView_Cube_Texture_RRJ);
	gpID3D11DeviceContext_RRJ->PSSetSamplers(0, 1, &gpID3D11SamplerState_Cube_Texture_RRJ);


	gpID3D11DeviceContext_RRJ->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	XMMATRIX r1 = XMMatrixIdentity();
	XMMATRIX r2 = XMMatrixIdentity();
	XMMATRIX r3 = XMMatrixIdentity();
	XMMATRIX rotateMatrix_RRJ = XMMatrixIdentity();
	XMMATRIX rotateMatrixCubeMap_RRJ = XMMatrixIdentity();
	XMMATRIX translateMatrix_RRJ = XMMatrixIdentity();
	XMMATRIX scaleMatrix_RRJ = XMMatrixIdentity();
	XMMATRIX worldMatrix_RRJ = XMMatrixIdentity();
	XMMATRIX viewMatrix_RRJ = XMMatrixIdentity();

	translateMatrix_RRJ = XMMatrixTranslation(0.0f, 0.0f, 5.0f);
	
	r1 = XMMatrixRotationX(-angle_Cube);
	r2 = XMMatrixRotationY(-angle_Cube);
	r3 = XMMatrixRotationZ(-angle_Cube);

	scaleMatrix_RRJ = XMMatrixScaling(50.9f, 50.9f, 50.9f);

	//rotateMatrix_RRJ = r2;
	worldMatrix_RRJ =  rotateMatrix_RRJ * scaleMatrix_RRJ * translateMatrix_RRJ;
	
	
	CBUFFER constantBuffer;
	ZeroMemory(&constantBuffer, sizeof(CBUFFER));
	constantBuffer.WorldMatrix = worldMatrix_RRJ;
	constantBuffer.ViewMatrix = viewMatrix_RRJ;
	constantBuffer.ProjectionMatrix = gPerspectiveProjectionMatrix_RRJ;
	constantBuffer.CubeMapToggle = 1;
	constantBuffer.LKeyPress = 0;


	gpID3D11DeviceContext_RRJ->UpdateSubresource(gpID3D11Buffer_ConstantBuffer_RRJ, 0,
		NULL, &constantBuffer, 0, 0);

	gpID3D11DeviceContext_RRJ->Draw(2* 4 * 6, 0);
	





	// ***** Model *****
	UINT stride_RRJ = sizeof(float) * 3;
	UINT offset_RRJ = 0;
	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(0, 1, &gpID3D11Buffer_VertexBuffer_Model_Pos_RRJ, &stride_RRJ, &offset_RRJ);

	stride_RRJ = sizeof(float) * 3;
	offset_RRJ = 0;
	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(2, 1, &gpID3D11Buffer_VertexBuffer_Model_Normal_RRJ, &stride_RRJ, &offset_RRJ);


	gpID3D11DeviceContext_RRJ->IASetIndexBuffer(gpID3D11Buffer_IndexBuffer_Model_Index_RRJ, DXGI_FORMAT_R16_UINT, 0);


	gpID3D11DeviceContext_RRJ->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	
	/********** Sampler and Shader Resource View **********/
	gpID3D11DeviceContext_RRJ->PSSetShaderResources(0, 1, &gpID3D11ShaderResourceView_Cube_Texture_RRJ);
	gpID3D11DeviceContext_RRJ->PSSetSamplers(0, 1, &gpID3D11SamplerState_Cube_Texture_RRJ);


	r1 = XMMatrixIdentity();
	r2 = XMMatrixIdentity();
	r3 = XMMatrixIdentity();
	worldMatrix_RRJ = XMMatrixIdentity();
	viewMatrix_RRJ = XMMatrixIdentity();
	translateMatrix_RRJ = XMMatrixIdentity();
	rotateMatrix_RRJ = XMMatrixIdentity();

	translateMatrix_RRJ = XMMatrixTranslation(0.0f, 0.0f, 5.0f);

	r1 = XMMatrixRotationX(-angle_Cube);
	r2 = XMMatrixRotationY(-angle_Cube);
	r3 = XMMatrixRotationZ(-angle_Cube);

	rotateMatrix_RRJ = r2;

	worldMatrix_RRJ = rotateMatrix_RRJ * translateMatrix_RRJ;

	CBUFFER constantBuffer_RRJ;

	if(bLights_RRJ == true){
		constantBuffer_RRJ.LKeyPress = 1;

		constantBuffer_RRJ.La = XMVectorSet(lightAmbient_RRJ[0], lightAmbient_RRJ[1], lightAmbient_RRJ[2], lightAmbient_RRJ[3]);
		constantBuffer_RRJ.Ld = XMVectorSet(lightDiffuse_RRJ[0], lightDiffuse_RRJ[1], lightDiffuse_RRJ[2], lightDiffuse_RRJ[3]);
		constantBuffer_RRJ.Ls = XMVectorSet(lightSpecular_RRJ[0], lightSpecular_RRJ[1], lightSpecular_RRJ[2], lightSpecular_RRJ[3]);
		constantBuffer_RRJ.LightPosition = XMVectorSet(lightPosition_RRJ[0], lightPosition_RRJ[1], lightPosition_RRJ[2], lightPosition_RRJ[3]);

		constantBuffer_RRJ.Ka = XMVectorSet(materialAmbient_RRJ[0], materialAmbient_RRJ[1], materialAmbient_RRJ[2], materialAmbient_RRJ[3]);
		constantBuffer_RRJ.Kd = XMVectorSet(materialDiffuse_RRJ[0], materialDiffuse_RRJ[1], materialDiffuse_RRJ[2], materialDiffuse_RRJ[3]);
		constantBuffer_RRJ.Ks = XMVectorSet(materialSpecular_RRJ[0], materialSpecular_RRJ[1], materialSpecular_RRJ[2], materialSpecular_RRJ[3]);
		constantBuffer_RRJ.Shininess = materialShininess_RRJ;

	}
	else
		constantBuffer_RRJ.LKeyPress = 0;


	constantBuffer_RRJ.WorldMatrix = worldMatrix_RRJ;
	constantBuffer_RRJ.ViewMatrix = viewMatrix_RRJ;
	constantBuffer_RRJ.ProjectionMatrix = gPerspectiveProjectionMatrix_RRJ;
	constantBuffer_RRJ.CubeMapToggle = 0;
	constantBuffer_RRJ.RotateMatrix = rotateMatrix_RRJ;

	gpID3D11DeviceContext_RRJ->UpdateSubresource(gpID3D11Buffer_ConstantBuffer_RRJ,
		0, NULL, &constantBuffer_RRJ, 0, 0);


	gpID3D11DeviceContext_RRJ->Draw(pVecFloat_Model_Sorted_Vertices->iSize / 3, 0);



	gpIDXGISwapChain_RRJ->Present(0, 0);


	angle_Cube = angle_Cube - 0.001f;
	if(angle_Cube < 00.0f)
		angle_Cube = 360.0f;
}




struct VecFloat* CreateVecFloat(void) {

	struct VecFloat *pTemp = NULL;

	pTemp = (struct VecFloat*)malloc(sizeof(struct VecFloat));
	if (pTemp == NULL) {
		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: CreateVecInt(): Malloc() Failed!\n");
		fclose(gbFile_RRJ);
		exit(0);
	}

	memset((void*)pTemp, 0, sizeof(struct VecFloat));

	return(pTemp);
}


int PushBackVecFloat(struct VecFloat *pVec, float data) {

	pVec->pData = (float*)realloc(pVec->pData, sizeof(struct VecFloat) * (pVec->iSize + 1));

	assert(pVec->pData);

	pVec->iSize = pVec->iSize + 1;
	pVec->pData[pVec->iSize - 1] = data;
	//fprintf(gbFile_RRJ, "iSize: %d   iData: %f\n", pVec->iSize, pVec->pData[pVec->iSize - 1]);

	return(RRJ_SUCCESS);
}


void ShowVecFloat(struct VecFloat *pVec) {

	fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
	for (int i = 0; i < pVec->iSize; i++)
		fprintf_s(gbFile_RRJ, "P[%d]: %f\t", i, pVec->pData[i]);

	fclose(gbFile_RRJ);
}


int DestroyVecFloat(struct VecFloat *pVec) {


	free(pVec->pData);
	pVec->pData = NULL;
	pVec->iSize = 0;
	free(pVec);
	pVec = NULL;

	return(RRJ_SUCCESS);
}



void LoadModel(void) {

	char buffer[1024];
	char *firstToken = NULL;
	char *My_Strtok(char*, char);
	const char *space = " ";
	char *cContext = NULL;


	while (fgets(buffer, 1024, gbFile_Model) != NULL) {

		firstToken = strtok_s(buffer, space, &cContext);

		if (strcmp(firstToken, "v") == 0) {
			//Vertices
			float x, y, z;
			x = (float)atof(strtok_s(NULL, space, &cContext));
			y = (float)atof(strtok_s(NULL, space, &cContext));
			z = (float)atof(strtok_s(NULL, space, &cContext));

			//fprintf_s(gbFile_Vertices, "%f/%f/%f\n", x, y, z);

			PushBackVecFloat(pVecFloat_Model_Vertices, x);
			//fprintf(gbFile_Vertices, "\n\nSrt: %f\n", pVecFloat_Model_Vertices->pData[0]);
			PushBackVecFloat(pVecFloat_Model_Vertices, y);
			PushBackVecFloat(pVecFloat_Model_Vertices, z);

		}
		else if (strcmp(firstToken, "vt") == 0) {
			//Texture

			float u, v;
			u = (float)atof(strtok_s(NULL, space, &cContext));
			v = (float)atof(strtok_s(NULL, space, &cContext));

			//fprintf_s(gbFile_TexCoord, "%f/%f\n", u, v);
			PushBackVecFloat(pVecFloat_Model_Texcoord, u);
			PushBackVecFloat(pVecFloat_Model_Texcoord, v);
		}
		else if (strcmp(firstToken, "vn") == 0) {
			//Normals

			float x, y, z;
			x = (float)atof(strtok_s(NULL, space, &cContext));
			y = (float)atof(strtok_s(NULL, space, &cContext));
			z = (float)atof(strtok_s(NULL, space, &cContext));

			//fprintf_s(gbFile_Normals, "%f/%f/%f\n", x, y, z);
			PushBackVecFloat(pVecFloat_Model_Normals, x);
			PushBackVecFloat(pVecFloat_Model_Normals, y);
			PushBackVecFloat(pVecFloat_Model_Normals, z);

		}
		else if (strcmp(firstToken, "f") == 0) {
			//Faces


			for (int i = 0; i < 3; i++) {

				char *faces = strtok_s(NULL, space, &cContext);
				int v, vt, vn;
				v = atoi(My_Strtok(faces, '/')) - 1;
				vt = atoi(My_Strtok(faces, '/')) - 1;
				vn = atoi(My_Strtok(faces, '/')) - 1;

				float x, y, z;

				//Sorted Vertices
				x = pVecFloat_Model_Vertices->pData[(v * 3) + 0];
				y = pVecFloat_Model_Vertices->pData[(v * 3) + 1];
				z = pVecFloat_Model_Vertices->pData[(v * 3) + 2];

				PushBackVecFloat(pVecFloat_Model_Sorted_Vertices, x);
				PushBackVecFloat(pVecFloat_Model_Sorted_Vertices, y);
				PushBackVecFloat(pVecFloat_Model_Sorted_Vertices, z);


				//Sorted Normals
				x = pVecFloat_Model_Normals->pData[(vn * 3) + 0];
				y = pVecFloat_Model_Normals->pData[(vn * 3) + 1];
				z = pVecFloat_Model_Normals->pData[(vn * 3) + 2];

				PushBackVecFloat(pVecFloat_Model_Sorted_Normals, x);
				PushBackVecFloat(pVecFloat_Model_Sorted_Normals, y);
				PushBackVecFloat(pVecFloat_Model_Sorted_Normals, z);


				//Sorted Texcoord;
				x = pVecFloat_Model_Texcoord->pData[(vt * 2) + 0];
				y = pVecFloat_Model_Texcoord->pData[(vt * 2) + 1];

				PushBackVecFloat(pVecFloat_Model_Sorted_Texcoord, x);
				PushBackVecFloat(pVecFloat_Model_Sorted_Texcoord, y);



				//Face Elements
				PushBackVecFloat(pVecFloat_Model_Elements, v);

				//fprintf(gbFile_FaceIndices, "%d/ %d/ %d     ", v, vt, vn);
			}
			//fprintf(gbFile_FaceIndices, "\n");


		}


	}


}

char gBuffer[128];

char* My_Strtok(char* str, char delimiter) {

	static int  i = 0;
	int  j = 0;
	char c;


	while ((c = str[i]) != delimiter && c != '\0') {
		gBuffer[j] = c;
		j = j + 1;
		i = i + 1;
	}

	gBuffer[j] = '\0';


	if (c == '\0') {
		i = 0;
	}
	else
		i = i + 1;


	return(gBuffer);
}
