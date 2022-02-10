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

//For Per Vertex Light Shader
ID3D11VertexShader *gpID3D11VertexShader_PV_RRJ = NULL;
ID3D11PixelShader *gpID3D11PixelShader_PV_RRJ = NULL;

//For Layout/ Attributes
ID3D11InputLayout *gpID3D11InputLayout_PV_RRJ = NULL;


//For Per Pixel Lights Vertex and Pixel Shader
ID3D11VertexShader *gpID3D11VertexShader_PP_RRJ = NULL;
ID3D11PixelShader *gpID3D11PixelShader_PP_RRJ = NULL;

//For Attribute
ID3D11InputLayout *gpID3D11InputLayout_PP_RRJ = NULL;




//For Sphere
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Sphere_Position_RRJ = NULL;
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Sphere_Normal_RRJ = NULL;
ID3D11Buffer *gpID3D11Buffer_IndexBuffer_Sphere_Index_RRJ = NULL;

float sphere_Vertices_RRJ[1146];
float sphere_Normal_RRJ[1146];
float sphere_Texcoord_RRJ[764];
unsigned short sphere_Elements_RRJ[2280];
unsigned int gNumOfElements_RRJ;
unsigned int gNumOfVertices_RRJ;



//For Uniform
ID3D11Buffer *gpID3D11Buffer_ConstantBuffer_RRJ = NULL;

//For ClearNormal
float gClearNormal_RRJ[4];

struct CBUFFER{
	XMMATRIX WorldMatrix;
	XMMATRIX ViewMatrix;
	XMMATRIX ProjectionMatrix;

	XMVECTOR La_Red;
	XMVECTOR Ld_Red;
	XMVECTOR Ls_Red;
	XMVECTOR LightPosition_Red;


	XMVECTOR La_Green;
	XMVECTOR Ld_Green;
	XMVECTOR Ls_Green;
	XMVECTOR LightPosition_Green;

	
	XMVECTOR La_Blue;
	XMVECTOR Ld_Blue;
	XMVECTOR Ls_Blue;
	XMVECTOR LightPosition_Blue;

	XMVECTOR Ka;
	XMVECTOR Kd;
	XMVECTOR Ks;	
	float Shininess;


	unsigned int LKeyPress;
};


//For Lights
bool bLights_RRJ = false;


XMMATRIX gPerspectiveProjectionMatrix_RRJ;


//For Lights
float lightAmibient_Red_RRJ[] = {0.0f, 0.0f, 0.0f, 1.0f};
float lightDiffuse_Red_RRJ[] = {1.0f, 0.0f, 0.0f, 1.0f};
float lightSpecular_Red_RRJ[] = {1.0f, 0.0f, 0.0f, 1.0f};
float lightPosition_Red_RRJ[] = {0.0f, 0.0f, 0.0f, 1.0f};

float lightAmbient_Green_RRJ[] = {0.0f, 0.0f, 0.0f, 1.0f};
float lightDiffuse_Green_RRJ[] = {0.0f, 1.0f, 0.0f, 1.0f};
float lightSpecular_Green_RRJ[] = {0.0f, 1.0f, 0.0f, 1.0f};
float lightPosition_Green_RRJ[] = {0.0f, 0.0f, 0.0f, 1.0};

float lightAmbient_Blue_RRJ[] = {0.0f, 0.0f, 0.0f, 1.0f};
float lightDiffuse_Blue_RRJ[] = {0.0f, 0.0f, 1.0f, 1.0f};
float lightSpecular_Blue_RRJ[] = {0.0f, 0.0f, 1.0f, 1.0f};
float lightPosition_Blue_RRJ[] = {0.0f, 0.0f, 0.0f, 1.0f};

//For Material
float materialAmbient_RRJ[] = {0.0f, 0.0f, 0.0f, 1.0f};
float materialDiffuse_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
float materialSpecular_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
float materialShininess_RRJ = 128.0f;


//For Culling
ID3D11RasterizerState *gpID3D11RasterizerState_RRJ = NULL;

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
	TCHAR szName_RRJ[] = TEXT("Rohit_R_Jadhav-D3D-27-3Lights");

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
			TEXT("Rohit_R_Jadhav-D3D-27-3Lights"),
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
					ToggleFullScreen();
					break;

				case 'Q':
				case 'q':
					DestroyWindow(hwnd);
					break;


				case 'P':
				case 'p':
					gpID3D11DeviceContext_RRJ->VSSetShader(gpID3D11VertexShader_PP_RRJ, NULL, 0);
					gpID3D11DeviceContext_RRJ->PSSetShader(gpID3D11PixelShader_PP_RRJ, NULL, 0);
					gpID3D11DeviceContext_RRJ->IASetInputLayout(gpID3D11InputLayout_PP_RRJ);
					gpID3D11DeviceContext_RRJ->VSSetConstantBuffers(0, 1, &gpID3D11Buffer_ConstantBuffer_RRJ);
					gpID3D11DeviceContext_RRJ->PSSetConstantBuffers(0, 1, &gpID3D11Buffer_ConstantBuffer_RRJ);
					break;


				case 'V':
				case 'v':
					gpID3D11DeviceContext_RRJ->VSSetShader(gpID3D11VertexShader_PV_RRJ, NULL, 0);
					gpID3D11DeviceContext_RRJ->PSSetShader(gpID3D11PixelShader_PV_RRJ, NULL, 0);
					gpID3D11DeviceContext_RRJ->IASetInputLayout(gpID3D11InputLayout_PV_RRJ);
					gpID3D11DeviceContext_RRJ->VSSetConstantBuffers(0, 1, &gpID3D11Buffer_ConstantBuffer_RRJ);
					break;




				case 'L':
				case 'l':
					if(bLights_RRJ == false)
						bLights_RRJ = true;
					else
						bLights_RRJ = false;
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




	/********** PER VERTEX SHADER **********/
	const char *vertexShaderSourceCode_PV_RRJ = 
		"cbuffer ConstantBuffer {" \
			"float4x4 worldMatrix;" \
			"float4x4 viewMatrix;" \
			"float4x4 projectionMatrix;" \

			"float4 la_red;" \
			"float4 ld_red;" \
			"float4 ls_red;" \
			"float4 lightPosition_red;" \

			"float4 la_green;" \
			"float4 ld_green;" \
			"float4 ls_green;" \
			"float4 lightPosition_green;" \

			"float4 la_blue;" \
			"float4 ld_blue;" \
			"float4 ls_blue;" \
			"float4 lightPosition_blue;" \

			"float4 ka;" \
			"float4 kd;" \
			"float4 ks;" \
			"float shininess;" \

			"uint keyPress;" \
		"};" \


		"struct Vertex_Output {" \
			"float4 position : SV_POSITION;" \
			"float4 phong_ads_color : COLOR;" \
		"};" \

		"Vertex_Output main(float4 pos : POSITION, float4 normal : NORMAL){" \
				
			"Vertex_Output v;" \
			"if(keyPress == 1) {" \

				"float4 eyeCoord = mul(worldMatrix, pos);" \
				"eyeCoord = mul(viewMatrix, eyeCoord);" \

				"float3x3 normalMatrix = (float3x3)mul(worldMatrix, viewMatrix);" \
				"float3 transformedNormal = normalize(mul(normalMatrix, (float3)normal));" \

				"float3 lightDirection_red = (float3)normalize(lightPosition_red - eyeCoord);" \
				"float SRed_Dot_N = max(dot(lightDirection_red, transformedNormal), 0.0);" \

				"float3 lightDirection_green = (float3)normalize(lightPosition_green - eyeCoord);" \
				"float SGreen_Dot_N = max(dot(lightDirection_green, transformedNormal), 0.0);" \

				"float3 lightDirection_blue = (float3)normalize(lightPosition_blue - eyeCoord);" \
				"float SBlue_Dot_N = max(dot(lightDirection_blue, transformedNormal), 0.0);" \


				"float3 viewerVec = normalize(-eyeCoord.xyz);" \
				
				"float3 reflectionVec_red = reflect(-lightDirection_red, transformedNormal);" \
				"float RRed_Dot_V = max(dot(reflectionVec_red, viewerVec), 0.0);" \

				"float3 reflectionVec_green = reflect(-lightDirection_green, transformedNormal);" \
				"float RGreen_Dot_V = max(dot(reflectionVec_green, viewerVec), 0.0);" \

				"float3 reflectionVec_blue = reflect(-lightDirection_blue, transformedNormal);" \
				"float RBlue_Dot_V = max(dot(reflectionVec_blue, viewerVec), 0.0);" \


				"float4 ambient_red = la_red * ka;" \
				"float4 diffuse_red = ld_red * kd * SRed_Dot_N;" \
				"float4 specular_red = ls_red * ks * max(pow(RRed_Dot_V, shininess), 0.0);" \
				"float4 red = ambient_red + diffuse_red + specular_red;" \

				"float4 ambient_green = la_green * ka;" \
				"float4 diffuse_green = ld_green * kd * SGreen_Dot_N;" \
				"float4 specular_green = ls_green * ks * max(pow(RGreen_Dot_V, shininess), 0.0);" \
				"float4 green = ambient_green + diffuse_green + specular_green;" \

				"float4 ambient_blue = la_blue * ka;" \
				"float4 diffuse_blue = ld_blue * kd * SBlue_Dot_N;" \
				"float4 specular_blue = ls_blue * ks * max(pow(RBlue_Dot_V, shininess), 0.0);" \
				"float4 blue = ambient_blue + diffuse_blue + specular_blue;" \


				"v.phong_ads_color = red + green + blue;" \
			"}" \
			"else {" \
				"v.phong_ads_color = float4(1.0f, 1.0f, 1.0f, 1.0f);" \
			"}" \

			"v.position = mul(worldMatrix, pos);" \
			"v.position = mul(viewMatrix, v.position);" \
			"v.position = mul(projectionMatrix, v.position);" \

			"return(v);" \
 		"}";


 	ID3DBlob *pID3DBlob_VertexShaderCode_PV_RRJ = NULL;
 	ID3DBlob *pID3DBlob_Error_RRJ = NULL;


 	hr_RRJ = D3DCompile(vertexShaderSourceCode_PV_RRJ,
 		lstrlenA(vertexShaderSourceCode_PV_RRJ) + 1,
 		"VS",
 		NULL,
 		D3D_COMPILE_STANDARD_FILE_INCLUDE,
 		"main",
 		"vs_5_0",
 		0,
 		0,
 		&pID3DBlob_VertexShaderCode_PV_RRJ,
 		&pID3DBlob_Error_RRJ);


 	if(FAILED(hr_RRJ)){
 		if(pID3DBlob_Error_RRJ != NULL){

 			fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 			fprintf_s(gbFile_RRJ, "ERROR: Per Vertex Lighting!!\n");
 			fprintf_s(gbFile_RRJ, "\tVERTEX SHADER ERROR: \n %s \n", (char*)pID3DBlob_Error_RRJ->GetBufferPointer());
 			fclose(gbFile_RRJ);

 			pID3DBlob_Error_RRJ->Release();
 			pID3DBlob_Error_RRJ = NULL;

 			return(hr_RRJ);
 		}
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: Per Vertex Lighting!!\n");
 		fprintf_s(gbFile_RRJ, "\tSUCCESS: Vertex Shader Compilation Done!!\n");
 		fclose(gbFile_RRJ);
 	}



 	hr_RRJ = gpID3D11Device_RRJ->CreateVertexShader(
 		pID3DBlob_VertexShaderCode_PV_RRJ->GetBufferPointer(),
 		pID3DBlob_VertexShaderCode_PV_RRJ->GetBufferSize(),
 		NULL,
 		&gpID3D11VertexShader_PV_RRJ);


 	if(FAILED(hr_RRJ)){
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "ERROR: Per Vertex Lighting!!\n");
 		fprintf_s(gbFile_RRJ, "\tERROR: Vertex Shader Creation Failed!!\n");
 		fclose(gbFile_RRJ);

 		return(hr_RRJ);
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: Per Vertex Lighting!!\n");
 		fprintf_s(gbFile_RRJ, "\tSUCCESS: Vertex Shader Created!!\n");
 		fclose(gbFile_RRJ);
 	}



 	gpID3D11DeviceContext_RRJ->VSSetShader(gpID3D11VertexShader_PV_RRJ, NULL, 0);





 	/********** PIXEL SHADER **********/
 	const char *pixelShaderSourceCode_PV_RRJ = 
 		"float4 main(float4 pos : SV_POSITION, float4 phong_ads_color : COLOR) : SV_TARGET { " \
 			"return(phong_ads_color);" \
 		"}";


 	ID3DBlob *pID3DBlob_PixelShaderCode_PV_RRJ = NULL;
 	pID3DBlob_Error_RRJ = NULL;

 	hr_RRJ = D3DCompile(pixelShaderSourceCode_PV_RRJ, 
 		lstrlenA(pixelShaderSourceCode_PV_RRJ) + 1,
 		"PS",
 		NULL,
 		D3D_COMPILE_STANDARD_FILE_INCLUDE,
 		"main",
 		"ps_5_0",
 		0, 0,
 		&pID3DBlob_PixelShaderCode_PV_RRJ,
 		&pID3DBlob_Error_RRJ);

 	if(FAILED(hr_RRJ)){
 		if(pID3DBlob_Error_RRJ != NULL){
 			fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 			fprintf_s(gbFile_RRJ, "ERROR: Per Vertex Lighting!!\n");
 			fprintf_s(gbFile_RRJ, "\tPIXEL SHADER ERROR: \n %s \n",
 				(char*)pID3DBlob_Error_RRJ->GetBufferPointer());
 			fclose(gbFile_RRJ);

 			return(hr_RRJ);
 		}
 	}
 	else{

 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: Per Vertex Lighting!!\n");
 		fprintf_s(gbFile_RRJ, "\tSUCCESS: Pixel Shader Compilation Done!!\n");
 		fclose(gbFile_RRJ);
 	}



 	hr_RRJ = gpID3D11Device_RRJ->CreatePixelShader(
 		pID3DBlob_PixelShaderCode_PV_RRJ->GetBufferPointer(),
 		pID3DBlob_PixelShaderCode_PV_RRJ->GetBufferSize(),
 		NULL,
 		&gpID3D11PixelShader_PV_RRJ);


 	if(FAILED(hr_RRJ)){
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "ERROR: Per Vertex Lighting!!\n");
 		fprintf_s(gbFile_RRJ, "\tERROR: Pixel Shader Creation Failed!!\n");
 		fclose(gbFile_RRJ);

 		return(hr_RRJ);
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: Per Vertex Lighting!!\n");
 		fprintf_s(gbFile_RRJ, "\tSUCCESS: Pixel Shader Created!!\n");
 		fclose(gbFile_RRJ);
 	}


 	gpID3D11DeviceContext_RRJ->PSSetShader(gpID3D11PixelShader_PV_RRJ, NULL, 0);




 	/********* INPUT LAYOUT **********/
 	D3D11_INPUT_ELEMENT_DESC inputElementDesc_PV_RRJ[2];
 	ZeroMemory((void*)&inputElementDesc_PV_RRJ, sizeof(D3D11_INPUT_ELEMENT_DESC));

 	//For vPosition
 	inputElementDesc_PV_RRJ[0].SemanticName = "POSITION";
 	inputElementDesc_PV_RRJ[0].SemanticIndex = 0;
 	inputElementDesc_PV_RRJ[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
 	inputElementDesc_PV_RRJ[0].InputSlot = 0;
 	inputElementDesc_PV_RRJ[0].AlignedByteOffset = 0;
 	inputElementDesc_PV_RRJ[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
 	inputElementDesc_PV_RRJ[0].InstanceDataStepRate = 0;


 	//For vNormal
 	inputElementDesc_PV_RRJ[1].SemanticName = "NORMAL";
 	inputElementDesc_PV_RRJ[1].SemanticIndex = 0;
 	inputElementDesc_PV_RRJ[1].Format = DXGI_FORMAT_R32G32B32_FLOAT;
 	inputElementDesc_PV_RRJ[1].InputSlot = 1;
 	inputElementDesc_PV_RRJ[1].AlignedByteOffset = 0;
 	inputElementDesc_PV_RRJ[1].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
 	inputElementDesc_PV_RRJ[1].InstanceDataStepRate = 0;



 	hr_RRJ = gpID3D11Device_RRJ->CreateInputLayout(inputElementDesc_PV_RRJ, _ARRAYSIZE(inputElementDesc_PV_RRJ),
 		pID3DBlob_VertexShaderCode_PV_RRJ->GetBufferPointer(),
 		pID3DBlob_VertexShaderCode_PV_RRJ->GetBufferSize(),
 		&gpID3D11InputLayout_PV_RRJ);

 	if(FAILED(hr_RRJ)){
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "ERROR: Per Vertex Lighting!!\n");
 		fprintf_s(gbFile_RRJ, "\tERROR: ID3D11Device::CreateInputLayout() Failed!!\n");
 		fclose(gbFile_RRJ);

 		return(hr_RRJ);
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: Per Vertex Lighting!!\n");
 		fprintf_s(gbFile_RRJ, "\tSUCCESS: ID3D11Device::CreateInputDevice() Done!!\n");
 		fclose(gbFile_RRJ);
 	}

 	gpID3D11DeviceContext_RRJ->IASetInputLayout(gpID3D11InputLayout_PV_RRJ);


 	pID3DBlob_PixelShaderCode_PV_RRJ->Release();
 	pID3DBlob_PixelShaderCode_PV_RRJ = NULL;

 	pID3DBlob_VertexShaderCode_PV_RRJ->Release();
 	pID3DBlob_VertexShaderCode_PV_RRJ = NULL;








 	/********** PER PIXEL LIGHT **********/
	const char *vertexShaderSourceCode_PP_RRJ = 
		"cbuffer ConstantBuffer {" \
			"float4x4 worldMatrix;" \
			"float4x4 viewMatrix;" \
			"float4x4 projectionMatrix;" \

			"float4 la_red;" \
			"float4 ld_red;" \
			"float4 ls_red;" \
			"float4 lightPosition_red;" \

			"float4 la_green;" \
			"float4 ld_green;" \
			"float4 ls_green;" \
			"float4 lightPosition_green;" \

			"float4 la_blue;" \
			"float4 ld_blue;" \
			"float4 ls_blue;" \
			"float4 lightPosition_blue;" \

			"float4 ka;" \
			"float4 kd;" \
			"float4 ks;" \
			"float shininess;" \

			"uint keyPress;" \

		"};" \


		"struct Vertex_Output {" \
			"float4 position : SV_POSITION;" \
			"float3 lightDirection_red : NORMAL0;" \
			"float3 lightDirection_green : NORMAL1;" \
			"float3 lightDirection_blue : NORMAL2;" \
			"float3 transformedNormal : NORMAL3;" \
			"float3 viewerVec : NORMAL4;" \

		"};" \

		"Vertex_Output main(float4 pos : POSITION, float4 normal : NORMAL) { " \

			"Vertex_Output v;" \

			"float4 eyeCoord = mul(worldMatrix, pos);" \
			"eyeCoord = mul(viewMatrix, eyeCoord);" \

			"float3x3 normalMatrix = (float3x3)mul(worldMatrix, viewMatrix);" \
			"v.transformedNormal = (float3)mul(normalMatrix, (float3)normal);" \

			"v.viewerVec = (-eyeCoord.xyz);" \

			"v.lightDirection_red = (float3)(lightPosition_red - eyeCoord);" \
			"v.lightDirection_green = (float3)(lightPosition_green - eyeCoord);" \
			"v.lightDirection_blue = (float3)(lightPosition_blue - eyeCoord);" \

			"v.position = mul(worldMatrix, pos);" \
			"v.position = mul(viewMatrix, v.position);" \
			"v.position = mul(projectionMatrix, v.position);" \

			"return(v);" \

		"}";


		ID3DBlob *pID3DBlob_VertexShaderCode_PP_RRJ = NULL;
		pID3DBlob_Error_RRJ = NULL;


		hr_RRJ = D3DCompile(
			vertexShaderSourceCode_PP_RRJ,
			lstrlenA(vertexShaderSourceCode_PP_RRJ) + 1,
			"VS",
			NULL,
			D3D_COMPILE_STANDARD_FILE_INCLUDE,
			"main",
			"vs_5_0",
			0,
			0,
			&pID3DBlob_VertexShaderCode_PP_RRJ,
			&pID3DBlob_Error_RRJ
			);

		if(FAILED(hr_RRJ)){

			if(pID3DBlob_Error_RRJ != NULL){
				fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
				fprintf_s(gbFile_RRJ, "ERROR: \n");
				fprintf_s(gbFile_RRJ, "ERROR: Per Pixel Lighting\n");
				fprintf_s(gbFile_RRJ, "\tVERTEX SHADER COMPILATION ERROR: %s\n", 
					(char*)pID3DBlob_Error_RRJ->GetBufferPointer());
				fclose(gbFile_RRJ);
				return(hr_RRJ);
			}
		}
		else{
			fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "SUCCESS: Per Pixel Lighting\n");
			fprintf_s(gbFile_RRJ, "\tSUCCESS: Vertex Shader Compilation Done!!\n");
			fclose(gbFile_RRJ);	
		}


		hr_RRJ = gpID3D11Device_RRJ->CreateVertexShader(
				pID3DBlob_VertexShaderCode_PP_RRJ->GetBufferPointer(),
				pID3DBlob_VertexShaderCode_PP_RRJ->GetBufferSize(),
				NULL, 
				&gpID3D11VertexShader_PP_RRJ);

		if(FAILED(hr_RRJ)){
			fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "ERROR: Per Pixel Lighting\n");
			fprintf_s(gbFile_RRJ, "\tERROR: Vertex Shader Creation Failed!!\n");
			fclose(gbFile_RRJ);
			return(hr_RRJ);
		}
		else{
			fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "SUCCESS: Per Pixel Lighting\n");
			fprintf_s(gbFile_RRJ, "\tSUCCESS: Vertex Shader Creation Done!!\n");
			fclose(gbFile_RRJ);	
		}


		//gpID3D11DeviceContext_RRJ->VSSetShader(gpID3D11VertexShader_PP_RRJ, NULL, 0);





		/********** PIXEL SHADER **********/
		const char *pixelShaderSourceCode_PP_RRJ = 
			"cbuffer ConstantBuffer {" \
				"float4x4 worldMatrix;" \
				"float4x4 viewMatrix;" \
				"float4x4 projectionMatrix;" \

				"float4 la_red;" \
				"float4 ld_red;" \
				"float4 ls_red;" \
				"float4 lightPosition_red;" \

				"float4 la_green;" \
				"float4 ld_green;" \
				"float4 ls_green;" \
				"float4 lightPosition_green;" \

				"float4 la_blue;" \
				"float4 ld_blue;" \
				"float4 ls_blue;" \
				"float4 lightPosition_blue;" \

				"float4 ka;" \
				"float4 kd;" \
				"float4 ks;" \
				"float shininess;" \

				"uint keyPress;" \

			"};" \

			"struct Vertex_Output {" \
				"float4 position : SV_POSITION;" \
				"float3 lightDirection_red : NORMAL0;" \
				"float3 lightDirection_green : NORMAL1;" \
				"float3 lightDirection_blue : NORMAL2;" \
				"float3 transformedNormal: NORMAL3;" \
				"float3 viewerVec : NORMAL4;" \
			"};" \


			"float4 main(float4 pos: SV_POSITION, Vertex_Output inVertex) : SV_TARGET {" \
				"float4 phong_ads_color;" \
				"if(keyPress == 1) {" \

					"float3 normalizeLightDirection_red = normalize(inVertex.lightDirection_red);" \
					"float3 normalizeLightDirection_green = normalize(inVertex.lightDirection_green);" \
					"float3 normalizeLightDirection_blue = normalize(inVertex.lightDirection_blue);" \


					"float3 normalizeTransformedNormal = normalize(inVertex.transformedNormal);" \


					"float SRed_Dot_N = max(dot(normalizeLightDirection_red, normalizeTransformedNormal), 0.0);" \
					"float SGreen_Dot_N = max(dot(normalizeLightDirection_green, normalizeTransformedNormal), 0.0);" \
					"float SBlue_Dot_N = max(dot(normalizeLightDirection_blue, normalizeTransformedNormal), 0.0);" \


					"float3 normalizeViewerVec = normalize(inVertex.viewerVec);" \

					"float3 reflectionVec_red = reflect(-normalizeLightDirection_red, normalizeTransformedNormal);" \
					"float RRed_Dot_V = max(dot(reflectionVec_red, normalizeViewerVec), 0.0);" \

					"float3 reflectionVec_green = reflect(-normalizeLightDirection_green, normalizeTransformedNormal);" \
					"float RGreen_Dot_V = max(dot(reflectionVec_green, normalizeViewerVec), 0.0);" \

					"float3 reflectionVec_blue = reflect(-normalizeLightDirection_blue, normalizeTransformedNormal);" \
					"float RBlue_Dot_V = max(dot(reflectionVec_blue, normalizeViewerVec), 0.0);" \

					
					"float4 ambient_red = la_red * ka;" \
					"float4 diffuse_red = ld_red * kd * SRed_Dot_N;" \
					"float4 specular_red = ls_red * ks * max(pow(RRed_Dot_V, shininess), 0.0);" \
					"float4 red = ambient_red + diffuse_red + specular_red;" \

					"float4 ambient_green = la_green * ka;" \
					"float4 diffuse_green = ld_green * kd * SGreen_Dot_N;" \
					"float4 specular_green = ls_green * ks * max(pow(RGreen_Dot_V, shininess), 0.0);" \
					"float4 green = ambient_green + diffuse_green + specular_green;" \

					"float4 ambient_blue = la_blue * ka;" \
					"float4 diffuse_blue = ld_blue * kd * SBlue_Dot_N;" \
					"float4 specular_blue = ls_blue * ks * max(pow(RBlue_Dot_V, shininess), 0.0);" \
					"float4 blue = ambient_blue + diffuse_blue + specular_blue;" \


					"phong_ads_color =  red + green + blue;" \

				"}" \
				"else{" \
					"phong_ads_color = float4(1.0f, 1.0f, 1.0f, 1.0f);" \
				"}" \

				"return(phong_ads_color);" \
			"}";



		ID3DBlob *pID3DBlob_PixelShaderCode_PP_RRJ = NULL;
		pID3DBlob_Error_RRJ = NULL;

		hr_RRJ = D3DCompile(
			pixelShaderSourceCode_PP_RRJ,
			lstrlenA(pixelShaderSourceCode_PP_RRJ) + 1,
			"PS",
			NULL,
			D3D_COMPILE_STANDARD_FILE_INCLUDE,
			"main",
			"ps_5_0",
			0,
			0,
			&pID3DBlob_PixelShaderCode_PP_RRJ,
			&pID3DBlob_Error_RRJ);


		if(FAILED(hr_RRJ)){

			if(pID3DBlob_Error_RRJ != NULL){
				fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
				fprintf_s(gbFile_RRJ, "ERROR: \n");
				fprintf_s(gbFile_RRJ, "ERROR: Per Pixel Lighting!!\n");
				fprintf_s(gbFile_RRJ, "\tPIXEL SHADER COMPILATION ERROR: %s\n", 
					(char*)pID3DBlob_Error_RRJ->GetBufferPointer());
				fclose(gbFile_RRJ);
				return(hr_RRJ);
			}
		}
		else{
			fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "SUCCESS: Per Pixel Lighting!!\n");
			fprintf_s(gbFile_RRJ, "\tSUCCESS: Pixel Shader Compilation Done!!\n");
			fclose(gbFile_RRJ);	
		}


		hr_RRJ = gpID3D11Device_RRJ->CreatePixelShader(
			pID3DBlob_PixelShaderCode_PP_RRJ->GetBufferPointer(),
			pID3DBlob_PixelShaderCode_PP_RRJ->GetBufferSize(),
			NULL,
			&gpID3D11PixelShader_PP_RRJ
			);


		if(FAILED(hr_RRJ)){
			fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "ERROR: Per Pixel Lighting!!\n");
			fprintf_s(gbFile_RRJ, "\tERROR: Pixel Shader Creation Failed!!\n");
			fclose(gbFile_RRJ);
			return(hr_RRJ);
		}
		else{
			fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "SUCCESS: Per Pixel Lighting!!\n");
			fprintf_s(gbFile_RRJ, "\tSUCCESS: Pixel Shader Creation Done!!\n");
			fclose(gbFile_RRJ);	
		}


		//gpID3D11DeviceContext_RRJ->PSSetShader(gpID3D11PixelShader_PP_RRJ, NULL, 0);




		
		/********** INPUT LAYOUT **********/
		D3D11_INPUT_ELEMENT_DESC d3d11InputElementDesc_PP_RRJ[2];
		ZeroMemory((void*)d3d11InputElementDesc_PP_RRJ, sizeof(D3D11_INPUT_ELEMENT_DESC) * 2);


		d3d11InputElementDesc_PP_RRJ[0].SemanticName = "POSITION";
		d3d11InputElementDesc_PP_RRJ[0].SemanticIndex = 0;
		d3d11InputElementDesc_PP_RRJ[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
		d3d11InputElementDesc_PP_RRJ[0].InputSlot = 0;
		d3d11InputElementDesc_PP_RRJ[0].AlignedByteOffset = 0;
		d3d11InputElementDesc_PP_RRJ[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
		d3d11InputElementDesc_PP_RRJ[0].InstanceDataStepRate = 0;



		d3d11InputElementDesc_PP_RRJ[1].SemanticName = "NORMAL"; 
		d3d11InputElementDesc_PP_RRJ[1].SemanticIndex = 0;
		d3d11InputElementDesc_PP_RRJ[1].Format = DXGI_FORMAT_R32G32B32_FLOAT;
		d3d11InputElementDesc_PP_RRJ[1].InputSlot = 1;
		d3d11InputElementDesc_PP_RRJ[1].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
		d3d11InputElementDesc_PP_RRJ[1].AlignedByteOffset = 0;
		d3d11InputElementDesc_PP_RRJ[1].InstanceDataStepRate = 0;


		hr_RRJ = gpID3D11Device_RRJ->CreateInputLayout(
			d3d11InputElementDesc_PP_RRJ,
			_ARRAYSIZE(d3d11InputElementDesc_PP_RRJ),
			pID3DBlob_VertexShaderCode_PP_RRJ->GetBufferPointer(),
			pID3DBlob_VertexShaderCode_PP_RRJ->GetBufferSize(),
			&gpID3D11InputLayout_PP_RRJ);


		if(FAILED(hr_RRJ)){
			fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "ERROR: Per Pixel Lighting!!\n");
			fprintf_s(gbFile_RRJ, "\tERROR: CreateInputLayout() Failed!!\n");
			fclose(gbFile_RRJ);
			return(hr_RRJ);
		}
		else{
			fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "SUCCESS: Per Pixel Lighting!!\n");
			fprintf_s(gbFile_RRJ, "\tSUCCESS: Input Layout Created!!\n");
			fclose(gbFile_RRJ);
		}


		//gpID3D11DeviceContext_RRJ->IASetInputLayout(gpID3D11InputLayout_PP_RRJ);


		pID3DBlob_VertexShaderCode_PP_RRJ->Release();
		pID3DBlob_VertexShaderCode_PP_RRJ = NULL;

		pID3DBlob_PixelShaderCode_PP_RRJ->Release();
		pID3DBlob_PixelShaderCode_PP_RRJ = NULL;

		pID3DBlob_Error_RRJ = NULL;












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






 	/********** Vertex Buffer For Sphere Normal **********/
 	D3D11_BUFFER_DESC bufferDesc_Sphere_Normal_RRJ;
 	ZeroMemory((void*)&bufferDesc_Sphere_Normal_RRJ, sizeof(D3D11_BUFFER_DESC));

 	bufferDesc_Sphere_Normal_RRJ.Usage = D3D11_USAGE_DYNAMIC;
 	bufferDesc_Sphere_Normal_RRJ.ByteWidth = sizeof(float) * _ARRAYSIZE(sphere_Normal_RRJ);
 	bufferDesc_Sphere_Normal_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
 	bufferDesc_Sphere_Normal_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

 	hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(&bufferDesc_Sphere_Normal_RRJ, NULL,
 		&gpID3D11Buffer_VertexBuffer_Sphere_Normal_RRJ);

 	if(FAILED(hr_RRJ)){
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "ERROR: ID3D11::CreateBuffer() for Sphere_Normal Failed!!\n");
 		fclose(gbFile_RRJ);
 		return(hr_RRJ);
 	}
 	else{
 		fopen_s(&gbFile_RRJ, gszLogFileName_RRJ, "a+");
 		fprintf_s(gbFile_RRJ, "SUCCESS: ID3D11::CreateBuffer() for Sphere_Normal Done!!\n");
 		fclose(gbFile_RRJ);	
 	}



 	/********** Memory Map I/O For Sphere_Normal **********/
 	D3D11_MAPPED_SUBRESOURCE mappedSubresource_Sphere_Normal_RRJ;
 	ZeroMemory((void*)&mappedSubresource_Sphere_Normal_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));

 	gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_Sphere_Normal_RRJ, 0, 
 		D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource_Sphere_Normal_RRJ);

 	memcpy(mappedSubresource_Sphere_Normal_RRJ.pData, sphere_Normal_RRJ, sizeof(sphere_Normal_RRJ));
 	gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_Sphere_Normal_RRJ, 0);




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


 	gClearNormal_RRJ[0] = 0.0f;
 	gClearNormal_RRJ[1] = 0.0f;
 	gClearNormal_RRJ[2] = 0.0f;
 	gClearNormal_RRJ[3] = 1.0f;

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


		if(gpID3D11InputLayout_PP_RRJ){
			gpID3D11InputLayout_PP_RRJ->Release();
			gpID3D11InputLayout_PP_RRJ = NULL;
		}


		if(gpID3D11InputLayout_PV_RRJ){
			gpID3D11InputLayout_PV_RRJ->Release();
			gpID3D11InputLayout_PV_RRJ = NULL;
		}

		if(gpID3D11Buffer_IndexBuffer_Sphere_Index_RRJ){
			gpID3D11Buffer_IndexBuffer_Sphere_Index_RRJ->Release();
			gpID3D11Buffer_IndexBuffer_Sphere_Index_RRJ = NULL;
		}

		if(gpID3D11Buffer_VertexBuffer_Sphere_Normal_RRJ){
			gpID3D11Buffer_VertexBuffer_Sphere_Normal_RRJ->Release();
			gpID3D11Buffer_VertexBuffer_Sphere_Normal_RRJ = NULL;
		}


		if(gpID3D11Buffer_VertexBuffer_Sphere_Position_RRJ){
			gpID3D11Buffer_VertexBuffer_Sphere_Position_RRJ->Release();
			gpID3D11Buffer_VertexBuffer_Sphere_Position_RRJ = NULL;
		}

		if(gpID3D11PixelShader_PV_RRJ){
			gpID3D11PixelShader_PV_RRJ->Release();
			gpID3D11PixelShader_PV_RRJ = NULL;
		}

		if(gpID3D11VertexShader_PV_RRJ){
			gpID3D11VertexShader_PV_RRJ->Release();
			gpID3D11VertexShader_PV_RRJ = NULL;
		}


		if(gpID3D11PixelShader_PP_RRJ){
			gpID3D11PixelShader_PP_RRJ->Release();
			gpID3D11PixelShader_PP_RRJ = NULL;
		}

		if(gpID3D11VertexShader_PP_RRJ){
			gpID3D11VertexShader_PP_RRJ->Release();
			gpID3D11VertexShader_PP_RRJ = NULL;
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


	void rotateRedLight(float);
	void rotateGreenLight(float);
	void rotateBlueLight(float);

	static float angle_red_RRJ = 0.0f;
	static float angle_green_RRJ = 0.0f;
	static float angle_blue_RRJ = 0.0f;


	gpID3D11DeviceContext_RRJ->ClearRenderTargetView(gpID3D11RenderTargetView_RRJ, gClearNormal_RRJ);

	gpID3D11DeviceContext_RRJ->ClearDepthStencilView(gpID3D11DepthStencilView_RRJ,
		D3D11_CLEAR_DEPTH, 1.0f, 0.0f);

	UINT stride = sizeof(float) * 3;
	UINT offset = 0;
	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(0, 1, &gpID3D11Buffer_VertexBuffer_Sphere_Position_RRJ, &stride, &offset);


	UINT stride_Normal = sizeof(float) * 3;
	UINT offset_Normal = 0;
	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(1, 1, &gpID3D11Buffer_VertexBuffer_Sphere_Normal_RRJ, &stride_Normal, &offset_Normal);


	gpID3D11DeviceContext_RRJ->IASetIndexBuffer(gpID3D11Buffer_IndexBuffer_Sphere_Index_RRJ, DXGI_FORMAT_R16_UINT, 0);



	gpID3D11DeviceContext_RRJ->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
;
	XMMATRIX translateMatrix_RRJ = XMMatrixIdentity();
	XMMATRIX worldMatrix_RRJ = XMMatrixIdentity();
	XMMATRIX viewMatrix_RRJ = XMMatrixIdentity();

	translateMatrix_RRJ = XMMatrixTranslation(0.0f, 0.0f, 2.0f);

	worldMatrix_RRJ =   translateMatrix_RRJ;

	CBUFFER constantBuffer_RRJ;

	if(bLights_RRJ == true){

		rotateRedLight(angle_red_RRJ);
		rotateGreenLight(angle_green_RRJ);
		rotateBlueLight(angle_blue_RRJ);

		angle_red_RRJ += 0.005f;
		angle_green_RRJ += 0.005f;
		angle_blue_RRJ += 0.005f;
		if(angle_red_RRJ > 360.0f)
			angle_red_RRJ = 0.0f;
		if(angle_green_RRJ > 360.0f)
			angle_green_RRJ = 0.0f;
		if(angle_blue_RRJ > 360.0f)
			angle_blue_RRJ = 0.0f;


		constantBuffer_RRJ.LKeyPress = 1;
		
		constantBuffer_RRJ.La_Red = XMVectorSet(lightAmibient_Red_RRJ[0], lightAmibient_Red_RRJ[1], lightAmibient_Red_RRJ[2], lightAmibient_Red_RRJ[3]);
		constantBuffer_RRJ.Ld_Red = XMVectorSet(lightDiffuse_Red_RRJ[0], lightDiffuse_Red_RRJ[1], lightDiffuse_Red_RRJ[2], lightDiffuse_Red_RRJ[3]);
		constantBuffer_RRJ.Ls_Red = XMVectorSet(lightSpecular_Red_RRJ[0], lightSpecular_Red_RRJ[1], lightSpecular_Red_RRJ[2], lightSpecular_Red_RRJ[3]);
		constantBuffer_RRJ.LightPosition_Red = XMVectorSet(lightPosition_Red_RRJ[0], lightPosition_Red_RRJ[1], lightPosition_Red_RRJ[2], lightPosition_Red_RRJ[3]);


		constantBuffer_RRJ.La_Green = XMVectorSet(lightAmbient_Green_RRJ[0], lightAmbient_Green_RRJ[1], lightAmbient_Green_RRJ[2], lightAmbient_Green_RRJ[3]);
		constantBuffer_RRJ.Ld_Green = XMVectorSet(lightDiffuse_Green_RRJ[0], lightDiffuse_Green_RRJ[1], lightDiffuse_Green_RRJ[2], lightDiffuse_Green_RRJ[3]);
		constantBuffer_RRJ.Ls_Green = XMVectorSet(lightSpecular_Green_RRJ[0], lightSpecular_Green_RRJ[1], lightSpecular_Green_RRJ[2], lightSpecular_Green_RRJ[3]);
		constantBuffer_RRJ.LightPosition_Green = XMVectorSet(lightPosition_Green_RRJ[0], lightPosition_Green_RRJ[1], lightPosition_Green_RRJ[2], lightPosition_Green_RRJ[3]);

		constantBuffer_RRJ.La_Blue = XMVectorSet(lightAmbient_Blue_RRJ[0], lightAmbient_Blue_RRJ[1], lightAmbient_Blue_RRJ[2], lightAmbient_Blue_RRJ[3]);
		constantBuffer_RRJ.Ld_Blue = XMVectorSet(lightDiffuse_Blue_RRJ[0], lightDiffuse_Blue_RRJ[1], lightDiffuse_Blue_RRJ[2], lightDiffuse_Blue_RRJ[3]);
		constantBuffer_RRJ.Ls_Blue = XMVectorSet(lightSpecular_Blue_RRJ[0], lightSpecular_Blue_RRJ[1], lightSpecular_Blue_RRJ[2], lightSpecular_Blue_RRJ[3]);
		constantBuffer_RRJ.LightPosition_Blue = XMVectorSet(lightPosition_Blue_RRJ[0], lightPosition_Blue_RRJ[1], lightPosition_Blue_RRJ[2], lightPosition_Blue_RRJ[3]);


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

	gpID3D11DeviceContext_RRJ->UpdateSubresource(gpID3D11Buffer_ConstantBuffer_RRJ, 0,
		NULL, &constantBuffer_RRJ, 0, 0);


	gpID3D11DeviceContext_RRJ->DrawIndexed(gNumOfElements_RRJ, 0, 0);

	gpIDXGISwapChain_RRJ->Present(0, 0);

}

void rotateRedLight(float angle) {
	lightPosition_Red_RRJ[0] = 0.0f;
	lightPosition_Red_RRJ[1] = (float)(5.0f * sin(angle));
	lightPosition_Red_RRJ[2] = (float)(5.0f * cos(angle));
}

void rotateGreenLight(float angle) {
	lightPosition_Green_RRJ[0] = (float)(5.0f * sin(angle));
	lightPosition_Green_RRJ[1] = 0.0f;
	lightPosition_Green_RRJ[2] = (float)(5.0f * cos(angle));
}

void rotateBlueLight(float angle) {
	lightPosition_Blue_RRJ[0] = (float)(5.0f * cos(angle));
	lightPosition_Blue_RRJ[1] = (float)(5.0f * sin(angle));
	lightPosition_Blue_RRJ[2] = 0.0f;
}
