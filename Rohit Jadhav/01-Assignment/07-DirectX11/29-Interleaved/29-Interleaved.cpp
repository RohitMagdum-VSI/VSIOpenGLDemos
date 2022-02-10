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


using namespace DirectX;


#define WIN_WIDTH 800
#define WIN_HEIGHT 600


//For FullScreen
bool bIsFullScreen_RRJ = false;
WINDOWPLACEMENT wpPrev_RRJ = {sizeof(WINDOWPLACEMENT)};
DWORD dwStyle_RRJ;
bool bActiveWindow_RRJ = false;
HWND ghwnd_RRJ = NULL;


//For DirectX
IDXGISwapChain *gpIDXGISwapChain_RRJ = NULL;
ID3D11Device *gpID3D11Device_RRJ = NULL;
ID3D11DeviceContext *gpID3D11DeviceContext_RRJ = NULL;

//For Render and Depth Buffer
ID3D11RenderTargetView *gpID3D11RenderTargetView_RRJ = NULL;
ID3D11DepthStencilView *gpID3D11DepthStencilView_RRJ = NULL;

//For Vertex and Pixel Shader
ID3D11VertexShader *gpID3D11VertexShader_RRJ = NULL;
ID3D11PixelShader *gpID3D11PixelShader_RRJ = NULL;

//For Attribute
ID3D11InputLayout *gpID3D11InputLayout_RRJ = NULL;

//For Uniform
ID3D11Buffer *gpID3D11Buffer_ConstantBuffer_RRJ = NULL;

struct CBUFFER{
	XMMATRIX WorldMatrix;
	XMMATRIX ViewMatrix;
	XMMATRIX ProjectionMatrix;

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


//For Cube
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Cube_PCNT_RRJ = NULL;


//For Culling
ID3D11RasterizerState *gpID3D11RasterizerState_RRJ = NULL;


//For Error
FILE *gbFile_RRJ = NULL;
const char *gLogFileName_RRJ = "Log.txt";


//For ClearColor
float gClearColor_RRJ[4];


//For Lights and Material
bool bLights_RRJ = false;

float lightAmbient_RRJ[] = {0.50f, 0.50f, 0.50f, 1.0f};
float lightDiffuse_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
float lightSpecular_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
float lightPosition_RRJ[] = {100.0f, 100.0f, -100.0f, 1.0f};

float materialAmbient_RRJ[] = {0.50f, 0.50f, 0.50f, 1.0f};
float materialDiffuse_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
float materialSpecular_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
float materialShininess_RRJ = 128.0f;


//For Texture
ID3D11SamplerState *gpID3D11SamplerState_RRJ = NULL;
ID3D11ShaderResourceView *gpID3D11ShaderResourceView_RRJ = NULL;


LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow){

	HRESULT initialize(void);
	void ToggleFullScreen(void);
	void display(void);

	HRESULT hr_RRJ;
	bool bDone_RRJ = false;



	fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "w");
	if(gbFile_RRJ == NULL){
		MessageBox(NULL, TEXT("ERROR: Log Creation Failed"), TEXT("ERROR"), MB_OK);
		exit(0);
	}
	else{
		fprintf_s(gbFile_RRJ, "SUCCESS: Log Created!!\n");
		fclose(gbFile_RRJ);
	}



	WNDCLASSEX wndclass_RRJ;
	MSG msg_RRJ;
	HWND hwnd_RRJ;
	TCHAR szName_RRJ[] = TEXT("Rohit_R_Jadhav-D3D-29-Interleaved");

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
		TEXT("Rohit_R_Jadhav-D3D-29-Interleaved"),
		WS_OVERLAPPEDWINDOW,
		100, 100, 
		WIN_WIDTH, WIN_HEIGHT,
		NULL, 
		NULL, 
		hInstance,
		NULL);


	ghwnd_RRJ = hwnd_RRJ;


	SetForegroundWindow(hwnd_RRJ);
	SetFocus(hwnd_RRJ);
	ShowWindow(hwnd_RRJ, iCmdShow);

	
	hr_RRJ = initialize();
	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: Initialize() Failed!!\n");
		fclose(gbFile_RRJ);
		DestroyWindow(hwnd_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
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
				//For Update
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
					break;
			}
			break;

		
		case WM_SIZE:
			if(gpID3D11DeviceContext_RRJ){

				hr_RRJ = resize(LOWORD(lParam), HIWORD(lParam));
				if(FAILED(hr_RRJ)){
					fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
					fprintf_s(gbFile_RRJ, "ERROR: WM_SIZE: Resize() Failed!!\n");
					fclose(gbFile_RRJ);
					DestroyWindow(hwnd);
				}
				else{
					fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
					fprintf_s(gbFile_RRJ, "SUCCESS: WM_SIZE: Resize() Done!!\n");
					fclose(gbFile_RRJ);
				}
			}
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
					(mi.rcMonitor.bottom - mi.rcMonitor.left),
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

	HRESULT resize(int, int);
	void uninitialize(void);
	HRESULT LoadD3DTexture(const wchar_t*, ID3D11ShaderResourceView**);

	HRESULT hr_RRJ;


	D3D_DRIVER_TYPE d3dDriverType_RRJ;
	D3D_DRIVER_TYPE d3dDriverTypes_RRJ[] = {
		D3D_DRIVER_TYPE_HARDWARE,
		D3D_DRIVER_TYPE_WARP,
		D3D_DRIVER_TYPE_REFERENCE
	};


	D3D_FEATURE_LEVEL d3dFeatureLvl_req = D3D_FEATURE_LEVEL_11_0;
	D3D_FEATURE_LEVEL d3dFeatureLvl_acq = D3D_FEATURE_LEVEL_10_0;


	UINT createDeviceFlags_RRJ = 0;
	UINT numOfDrivers_RRJ = sizeof(d3dDriverTypes_RRJ) / sizeof(d3dDriverTypes_RRJ[0]);
	UINT numOfFeatureLvls_RRJ = 1;


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
	dxgiSwapChainDesc_RRJ.Windowed = TRUE;	

	dxgiSwapChainDesc_RRJ.SampleDesc.Count = 1;
	dxgiSwapChainDesc_RRJ.SampleDesc.Quality = 0;


	for(int indexDriverTypes_RRJ = 0; indexDriverTypes_RRJ < numOfDrivers_RRJ; indexDriverTypes_RRJ++){

		d3dDriverType_RRJ = d3dDriverTypes_RRJ[indexDriverTypes_RRJ];

		hr_RRJ = D3D11CreateDeviceAndSwapChain(
			NULL,
			d3dDriverType_RRJ,
			NULL,
			createDeviceFlags_RRJ,
			&d3dFeatureLvl_req,
			numOfFeatureLvls_RRJ,
			D3D11_SDK_VERSION,
			&dxgiSwapChainDesc_RRJ,
			&gpIDXGISwapChain_RRJ,
			&gpID3D11Device_RRJ,
			&d3dFeatureLvl_acq,
			&gpID3D11DeviceContext_RRJ
			);


		if(SUCCEEDED(hr_RRJ))
			break;

	}

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: D3D11CreateDeviceAndSwapChian() Failed!!\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");

		fprintf_s(gbFile_RRJ, "SUCCESS: D3D11CreateDeviceAndSwapChain() Done!!\n");

		fprintf_s(gbFile_RRJ, "SUCCESS: DirectX Feature Level : ");
		if(d3dFeatureLvl_acq == D3D_FEATURE_LEVEL_11_0)
			fprintf_s(gbFile_RRJ, "11.0\n");
		else if(d3dFeatureLvl_acq == D3D_FEATURE_LEVEL_10_1)
			fprintf_s(gbFile_RRJ, "10.1\n");
		else if(d3dFeatureLvl_acq == D3D_FEATURE_LEVEL_10_0)
			fprintf_s(gbFile_RRJ, "10.0\n");
		else
			fprintf_s(gbFile_RRJ, "Unknown\n");


		fprintf_s(gbFile_RRJ, "SUCCESS: Driver Type: ");
		if(d3dDriverType_RRJ == D3D_DRIVER_TYPE_HARDWARE)
			fprintf_s(gbFile_RRJ, "Hardware\n");
		else if(d3dDriverType_RRJ == D3D_DRIVER_TYPE_WARP)
			fprintf_s(gbFile_RRJ, "WARP\n");
		else if(d3dDriverType_RRJ == D3D_DRIVER_TYPE_REFERENCE)
			fprintf_s(gbFile_RRJ, "Reference\n");
		else
			fprintf_s(gbFile_RRJ, "Unknown\n");

		fclose(gbFile_RRJ);
	}





	/********** VERTEX SHADER **********/
	const char *vertexShaderSourceCode_RRJ = 
		"cbuffer ConstantBuffer {" \
			"float4x4 worldMatrix;" \
			"float4x4 viewMatrix;" \
			"float4x4 projectionMatrix;" \

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


		"struct Vertex_Output {" \
			"float4 position : SV_POSITION;" \
			"float3 lightDirection : NORMAL0;" \
			"float3 transformedNormal : NORMAL1;" \
			"float3 viewerVec : NORMAL2;" \
			"float4 color : COLOR;" \
			"float2 texcoord : TEXCOORD;" \

		"};" \

		"Vertex_Output main(float4 pos : POSITION, float4 col : COLOR, float4 normal : NORMAL, float2 tex : TEXCOORD) { " \

			"Vertex_Output v;" \

			"float4 eyeCoord = mul(worldMatrix, pos);" \
			"eyeCoord = mul(viewMatrix, eyeCoord);" \

			"float3x3 normalMatrix = (float3x3)mul(worldMatrix, viewMatrix);" \
			"v.transformedNormal = (float3)mul(normalMatrix, (float3)normal);" \

			"v.viewerVec = (-eyeCoord.xyz);" \

			"v.lightDirection = (float3)(lightPosition - eyeCoord);" \

			"v.color = col;" \
			"v.texcoord = tex;" \

			"v.position = mul(worldMatrix, pos);" \
			"v.position = mul(viewMatrix, v.position);" \
			"v.position = mul(projectionMatrix, v.position);" \

			"return(v);" \

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
			&pID3DBlob_Error_RRJ
			);

		if(FAILED(hr_RRJ)){

			if(pID3DBlob_Error_RRJ != NULL){
				fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
				fprintf_s(gbFile_RRJ, "ERROR: \n");
				fprintf_s(gbFile_RRJ, "VERTEX SHADER COMPILATION ERROR: %s\n", 
					(char*)pID3DBlob_Error_RRJ->GetBufferPointer());
				fclose(gbFile_RRJ);
				return(hr_RRJ);
			}
		}
		else{
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "SUCCESS: Vertex Shader Compilation Done!!\n");
			fclose(gbFile_RRJ);	
		}


		hr_RRJ = gpID3D11Device_RRJ->CreateVertexShader(
				pID3DBlob_VertexShaderCode_RRJ->GetBufferPointer(),
				pID3DBlob_VertexShaderCode_RRJ->GetBufferSize(),
				NULL, 
				&gpID3D11VertexShader_RRJ);

		if(FAILED(hr_RRJ)){
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "ERROR: Vertex Shader Creation Failed!!\n");
			fclose(gbFile_RRJ);
			return(hr_RRJ);
		}
		else{
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "SUCCESS: Vertex Shader Creation Done!!\n");
			fclose(gbFile_RRJ);	
		}


		gpID3D11DeviceContext_RRJ->VSSetShader(gpID3D11VertexShader_RRJ, NULL, 0);





		/********** PIXEL SHADER **********/
		const char *pixelShaderSourceCode_RRJ = 
			
			"Texture2D myTexture2D;" \
			"SamplerState mySamplerState;" \


			"cbuffer ConstantBuffer {" \
				"float4x4 worldMatrix;" \
				"float4x4 viewMatrix;" \
				"float4x4 projectionMatrix;" \

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

			"struct Vertex_Output {" \
				"float4 position : SV_POSITION;" \
				"float3 lightDirection : NORMAL0;" \
				"float3 transformedNormal: NORMAL1;" \
				"float3 viewerVec : NORMAL2;" \
				"float4 color : COLOR;" \
				"float3 texcoord: TEXCOORD;" \
			"};" \


			"float4 main(float4 pos: SV_POSITION, Vertex_Output inVertex) : SV_TARGET {" \
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

				"float4 tex = myTexture2D.Sample(mySamplerState, inVertex.texcoord);" \

				"return(tex * phong_ads_color * inVertex.color);" \
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
			0,
			0,
			&pID3DBlob_PixelShaderCode_RRJ,
			&pID3DBlob_Error_RRJ);


		if(FAILED(hr_RRJ)){

			if(pID3DBlob_Error_RRJ != NULL){
				fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
				fprintf_s(gbFile_RRJ, "ERROR: \n");
				fprintf_s(gbFile_RRJ, "PIXEL SHADER COMPILATION ERROR: %s\n", 
					(char*)pID3DBlob_Error_RRJ->GetBufferPointer());
				fclose(gbFile_RRJ);
				return(hr_RRJ);
			}
		}
		else{
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "SUCCESS: Pixel Shader Compilation Done!!\n");
			fclose(gbFile_RRJ);	
		}


		hr_RRJ = gpID3D11Device_RRJ->CreatePixelShader(
			pID3DBlob_PixelShaderCode_RRJ->GetBufferPointer(),
			pID3DBlob_PixelShaderCode_RRJ->GetBufferSize(),
			NULL,
			&gpID3D11PixelShader_RRJ
			);


		if(FAILED(hr_RRJ)){
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "ERROR: Pixel Shader Creation Failed!!\n");
			fclose(gbFile_RRJ);
			return(hr_RRJ);
		}
		else{
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "SUCCESS: Pixel Shader Creation Done!!\n");
			fclose(gbFile_RRJ);	
		}


		gpID3D11DeviceContext_RRJ->PSSetShader(gpID3D11PixelShader_RRJ, NULL, 0);




		
		/********** INPUT LAYOUT **********/
		D3D11_INPUT_ELEMENT_DESC d3d11InputElementDesc_RRJ[4];
		ZeroMemory((void*)d3d11InputElementDesc_RRJ, sizeof(D3D11_INPUT_ELEMENT_DESC) * 4);


		d3d11InputElementDesc_RRJ[0].SemanticName = "POSITION";
		d3d11InputElementDesc_RRJ[0].SemanticIndex = 0;
		d3d11InputElementDesc_RRJ[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
		d3d11InputElementDesc_RRJ[0].InputSlot = 0;
		d3d11InputElementDesc_RRJ[0].AlignedByteOffset = 0;
		d3d11InputElementDesc_RRJ[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
		d3d11InputElementDesc_RRJ[0].InstanceDataStepRate = 0;


		d3d11InputElementDesc_RRJ[1].SemanticName = "COLOR";
		d3d11InputElementDesc_RRJ[1].SemanticIndex = 0;
		d3d11InputElementDesc_RRJ[1].Format = DXGI_FORMAT_R32G32B32_FLOAT;
		d3d11InputElementDesc_RRJ[1].InputSlot = 1;
		d3d11InputElementDesc_RRJ[1].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
		d3d11InputElementDesc_RRJ[1].AlignedByteOffset = 0;
		d3d11InputElementDesc_RRJ[1].InstanceDataStepRate = 0;


		d3d11InputElementDesc_RRJ[2].SemanticName = "NORMAL"; 
		d3d11InputElementDesc_RRJ[2].SemanticIndex = 0;
		d3d11InputElementDesc_RRJ[2].Format = DXGI_FORMAT_R32G32B32_FLOAT;
		d3d11InputElementDesc_RRJ[2].InputSlot = 2;
		d3d11InputElementDesc_RRJ[2].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
		d3d11InputElementDesc_RRJ[2].AlignedByteOffset = 0;
		d3d11InputElementDesc_RRJ[2].InstanceDataStepRate = 0;


		d3d11InputElementDesc_RRJ[3].SemanticName = "TEXCOORD";
		d3d11InputElementDesc_RRJ[3].SemanticIndex = 0;
		d3d11InputElementDesc_RRJ[3].Format = DXGI_FORMAT_R32G32_FLOAT;
		d3d11InputElementDesc_RRJ[3].InputSlot = 3;
		d3d11InputElementDesc_RRJ[3].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
		d3d11InputElementDesc_RRJ[3].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
		d3d11InputElementDesc_RRJ[3].InstanceDataStepRate = 0;



		hr_RRJ = gpID3D11Device_RRJ->CreateInputLayout(
			d3d11InputElementDesc_RRJ,
			_ARRAYSIZE(d3d11InputElementDesc_RRJ),
			pID3DBlob_VertexShaderCode_RRJ->GetBufferPointer(),
			pID3DBlob_VertexShaderCode_RRJ->GetBufferSize(),
			&gpID3D11InputLayout_RRJ);


		if(FAILED(hr_RRJ)){
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "ERROR: CreateInputLayout() Failed!!\n");
			fclose(gbFile_RRJ);
			return(hr_RRJ);
		}
		else{
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "SUCCESS: Input Layout Created!!\n");
			fclose(gbFile_RRJ);
		}


		gpID3D11DeviceContext_RRJ->IASetInputLayout(gpID3D11InputLayout_RRJ);


		pID3DBlob_VertexShaderCode_RRJ->Release();
		pID3DBlob_VertexShaderCode_RRJ = NULL;

		pID3DBlob_PixelShaderCode_RRJ->Release();
		pID3DBlob_PixelShaderCode_RRJ = NULL;

		pID3DBlob_Error_RRJ = NULL;



		/********** CUBE Position, Color, Normal, Texcoord **********/
		float cube_PCNT[] = {

			//Position 			//Color 			//Normal 			//Texcoord

			//Front
	 		-1.0f, 1.0f, -1.0f, 	1.0f, 0.0f, 0.0f,		0.0f, 0.0f, -1.0f,		0.0f, 0.0f,
	 		1.0f, 1.0f, -1.0f,		1.0f, 0.0f, 0.0f, 		0.0f, 0.0f, -1.0f,		1.0f, 0.0f,
	 		-1.0f, -1.0f, -1.0f,	1.0f, 0.0f, 0.0f,		0.0f, 0.0f, -1.0f,		0.0f, 1.0f,

			-1.0f, -1.0f, -1.0f,	1.0f, 0.0f, 0.0f,		0.0f, 0.0f, -1.0f,		0.0f, 1.0f,
			1.0f, 1.0f, -1.0f,		1.0f, 0.0f, 0.0f,		0.0f, 0.0f, -1.0f,		1.0f, 0.0f,
	 		1.0f, -1.0f, -1.0f,	1.0f, 0.0f, 0.0f,		0.0f, 0.0f, -1.0f,		1.0f, 1.0f,

	 		//Right
	 		1.0f, 1.0f, -1.0f,		0.0f, 1.0f, 0.0f,		1.0f, 0.0f, 0.0f,		0.0f, 0.0f,
	 		1.0f, 1.0f, 1.0f, 		0.0f, 1.0f, 0.0f,		1.0f, 0.0f, 0.0f,		1.0f, 0.0f,
	 		1.0f, -1.0f, -1.0f,	0.0f, 1.0f, 0.0f,		1.0f, 0.0f, 0.0f, 		0.0f, 1.0f,

	 		1.0f, -1.0f, -1.0f,	0.0f, 1.0f, 0.0f,		1.0f, 0.0f, 0.0f,		0.0f,	1.0f,
	 		1.0f, 1.0f, 1.0f, 		0.0f, 1.0f, 0.0f,		1.0f, 0.0f, 0.0f,		1.0f, 0.0f,
	 		1.0f, -1.0f, 1.0f,		0.0f, 1.0f, 0.0f,		1.0f, 0.0f, 0.0f,		1.0f, 1.0f,


	 		//Back
			1.0f, 1.0f, 1.0f,		0.0f, 0.0f, 1.0f,		0.0f, 0.0f, 1.0f,		0.0f, 0.0f,
			-1.0f, 1.0f, 1.0f,		0.0f, 0.0f, 1.0f,		0.0f, 0.0f, 1.0f,		1.0f, 0.0f,
			1.0f, -1.0f, 1.0f,		0.0f, 0.0f, 1.0f,		0.0f, 0.0f, 1.0f,		0.0f, 1.0f,

			1.0f, -1.0f, 1.0f,		0.0f, 0.0f, 1.0f,		0.0f, 0.0f, 1.0f,		0.0f, 1.0f,
			-1.0f, 1.0f, 1.0f,		0.0f, 0.0f, 1.0f,		0.0f, 0.0f, 1.0f,		1.0f, 0.0f,
			-1.0f, -1.0f, 1.0f,	0.0f, 0.0f, 1.0f,		0.0f, 0.0f, 1.0f,		1.0f, 1.0f,


			//Left
			-1.0f, 1.0f, 1.0f,		1.0f, 1.0f, 0.0f,		-1.0f, 0.0f, 0.0f,		0.0f, 0.0f,
			-1.0f, 1.0f, -1.0f,	1.0f, 1.0f, 0.0f,		-1.0f, 0.0f, 0.0f,		1.0f, 0.0f,
			-1.0f, -1.0f, 1.0f,	1.0f, 1.0f, 0.0f,		-1.0f, 0.0f, 0.0f,		0.0f, 1.0f,

			-1.0f, -1.0f, 1.0f,	1.0f, 1.0f, 0.0f,		-1.0f, 0.0f, 0.0f,		0.0f, 1.0f,
			-1.0f, 1.0f, -1.0f,	1.0f, 1.0f, 0.0f,		-1.0f, 0.0f, 0.0f,		1.0f, 0.0f,
			-1.0f, -1.0f, -1.0f,	1.0f, 1.0f, 0.0f,		-1.0f, 0.0f, 0.0f,		1.0f, 1.0f,

			//Top
			-1.0f, 1.0f, 1.0f,		0.0f, 1.0f, 1.0f,		0.0f, 1.0f, 0.0f,		0.0f, 0.0f,
			1.0f, 1.0f, 1.0f,		0.0f, 1.0f, 1.0f,		0.0f, 1.0f, 0.0f,		1.0f, 0.0f,
			-1.0f, 1.0f, -1.0f,	0.0f, 1.0f, 1.0f,		0.0f, 1.0f, 0.0f,		0.0f, 1.0f,

			-1.0f, 1.0f, -1.0f,	0.0f, 1.0f, 1.0f,		0.0f, 1.0f, 0.0f,		0.0f, 1.0f,
			1.0f, 1.0f, 1.0f,		0.0f, 1.0f, 1.0f,		0.0f, 1.0f, 0.0f,		1.0f, 0.0f,
			1.0f, 1.0f, -1.0f,		0.0f, 1.0f, 1.0f,		0.0f, 1.0f, 0.0f,		1.0f, 1.0f,

			//Bottom
			-1.0f, -1.0f, 1.0f,	1.0f, 0.0f, 1.0f,		0.0f, -1.0f, 0.0f,		0.0f, 0.0f,
			1.0f, -1.0f, 1.0f,		1.0f, 0.0f, 1.0f,		0.0f, -1.0f, 0.0f,		1.0f, 0.0f,
			-1.0f, -1.0f, -1.0f,	1.0f, 0.0f, 1.0f,		0.0f, -1.0f, 0.0f,		0.0f, 1.0f,

			-1.0f, -1.0f, -1.0f,	1.0f, 0.0f, 1.0f,		0.0f, -1.0f, 0.0f,		0.0f, 1.0f,
			1.0f, -1.0f, 1.0f,		1.0f, 0.0f, 1.0f,		0.0f, -1.0f, 0.0f,		1.0f, 0.0f,	
			1.0f, -1.0f, -1.0f,	1.0f, 0.0f, 1.0f,		0.0f, -1.0f, 0.0f,		1.0f, 1.0f,

		};



		/********** Cube Position Color Normal Texcoord **********/
		D3D11_BUFFER_DESC bufferDesc_Cube_PCNT_RRJ;
		ZeroMemory((void*)&bufferDesc_Cube_PCNT_RRJ, sizeof(D3D11_BUFFER_DESC));

		bufferDesc_Cube_PCNT_RRJ.Usage = D3D11_USAGE_DYNAMIC;
		bufferDesc_Cube_PCNT_RRJ.ByteWidth = sizeof(float) * _ARRAYSIZE(cube_PCNT);
		bufferDesc_Cube_PCNT_RRJ.BindFlags = D3D11_BIND_VERTEX_BUFFER;
		bufferDesc_Cube_PCNT_RRJ.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

		hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(
			&bufferDesc_Cube_PCNT_RRJ, 
			NULL, 
			&gpID3D11Buffer_VertexBuffer_Cube_PCNT_RRJ);

		if(FAILED(hr_RRJ)){
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "ERROR: Cube Position Color Normal Texcoord CreateBuffer()\n");
			fclose(gbFile_RRJ);
			return(hr_RRJ);
		}
		else{
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "SUCCESS: Cube Position Color Normal TexCoord CreateBuffer()\n");
			fclose(gbFile_RRJ);
		}


		/********** Cube Position Color Normal TexCoord Mapping **********/
		D3D11_MAPPED_SUBRESOURCE mappedSubresource_RRJ;
		ZeroMemory((void*)&mappedSubresource_RRJ, sizeof(D3D11_MAPPED_SUBRESOURCE));

		gpID3D11DeviceContext_RRJ->Map(gpID3D11Buffer_VertexBuffer_Cube_PCNT_RRJ, 0,
			D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource_RRJ);
		memcpy(mappedSubresource_RRJ.pData, cube_PCNT, sizeof(cube_PCNT));
		gpID3D11DeviceContext_RRJ->Unmap(gpID3D11Buffer_VertexBuffer_Cube_PCNT_RRJ, 0);




		/********** CONSTANT BUFFER **********/
		D3D11_BUFFER_DESC bufferDesc_ConstantBuffer_RRJ;
		ZeroMemory((void*)&bufferDesc_ConstantBuffer_RRJ, sizeof(D3D11_BUFFER_DESC));

		bufferDesc_ConstantBuffer_RRJ.Usage = D3D11_USAGE_DEFAULT;
		bufferDesc_ConstantBuffer_RRJ.ByteWidth = sizeof(CBUFFER);
		bufferDesc_ConstantBuffer_RRJ.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

		hr_RRJ = gpID3D11Device_RRJ->CreateBuffer(
			&bufferDesc_ConstantBuffer_RRJ,
			NULL,
			&gpID3D11Buffer_ConstantBuffer_RRJ
			);


		if(FAILED(hr_RRJ)){
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "ERROR: Constant Buffer CreateBuffer()\n");
			fclose(gbFile_RRJ);
			return(hr_RRJ);
		}
		else{
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "SUCCESS: Constant Buffer CreateBuffer()\n");
			fclose(gbFile_RRJ);	
		}


		gpID3D11DeviceContext_RRJ->VSSetConstantBuffers(0, 1, &gpID3D11Buffer_ConstantBuffer_RRJ);
		gpID3D11DeviceContext_RRJ->PSSetConstantBuffers(0, 1, &gpID3D11Buffer_ConstantBuffer_RRJ);




		/********** CULLING **********/
		D3D11_RASTERIZER_DESC rasterizerDesc_RRJ;
		ZeroMemory((void*)&rasterizerDesc_RRJ, sizeof(D3D11_RASTERIZER_DESC));

		rasterizerDesc_RRJ.CullMode = D3D11_CULL_NONE;
		rasterizerDesc_RRJ.FillMode = D3D11_FILL_SOLID;
		rasterizerDesc_RRJ.FrontCounterClockwise = FALSE;
		rasterizerDesc_RRJ.AntialiasedLineEnable = FALSE;
		rasterizerDesc_RRJ.DepthBias = 0;
		rasterizerDesc_RRJ.DepthBiasClamp = 0.0f;
		rasterizerDesc_RRJ.DepthClipEnable = TRUE;
		rasterizerDesc_RRJ.SlopeScaledDepthBias = 0.0f;
		rasterizerDesc_RRJ.ScissorEnable = FALSE;
		rasterizerDesc_RRJ.MultisampleEnable = FALSE;

		hr_RRJ = gpID3D11Device_RRJ->CreateRasterizerState(&rasterizerDesc_RRJ, &gpID3D11RasterizerState_RRJ);


		if(FAILED(hr_RRJ)){
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "ERROR: CreateRasterizerState()\n");
			fclose(gbFile_RRJ);
			return(hr_RRJ);
		}
		else{
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "SUCCESS: CreateRasterizerState()\n");
			fclose(gbFile_RRJ);	
		}


		gpID3D11DeviceContext_RRJ->RSSetState(gpID3D11RasterizerState_RRJ);



		/********** Shader Resource View **********/
		hr_RRJ = LoadD3DTexture(L"marble.bmp", &gpID3D11ShaderResourceView_RRJ);
		if(FAILED(hr_RRJ)){
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "ERROR: LoadD3DTexutre()\n");
			fclose(gbFile_RRJ);
			return(hr_RRJ);
		}
		else{
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "SUCCESS: LoadD3DTexutre()\n");
			fclose(gbFile_RRJ);	
		}


		/********** Sampler State **********/
		D3D11_SAMPLER_DESC d3dSamplerDesc_RRJ;
		ZeroMemory((void*)&d3dSamplerDesc_RRJ, sizeof(D3D11_SAMPLER_DESC));

		d3dSamplerDesc_RRJ.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
		d3dSamplerDesc_RRJ.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
		d3dSamplerDesc_RRJ.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
		d3dSamplerDesc_RRJ.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;


		hr_RRJ = gpID3D11Device_RRJ->CreateSamplerState(
			&d3dSamplerDesc_RRJ,
			&gpID3D11SamplerState_RRJ);

		if(FAILED(hr_RRJ)){
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "ERROR: CreateSamplerState()\n");
			fclose(gbFile_RRJ);
			return(hr_RRJ);
		}
		else{
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "SUCCESS: CreateSamplerState()\n");
			fclose(gbFile_RRJ);	
		}


		gPerspectiveProjectionMatrix_RRJ = XMMatrixIdentity();

		hr_RRJ = resize(WIN_WIDTH, WIN_HEIGHT);
		
		if(FAILED(hr_RRJ)){
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "ERROR: Warmup Resize()\n");
			fclose(gbFile_RRJ);
			return(hr_RRJ);
		}
		else{
			fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
			fprintf_s(gbFile_RRJ, "SUCCESS: Warmup Resize()\n");
			fclose(gbFile_RRJ);	
		}



		gClearColor_RRJ[0] = 0.0f;
		gClearColor_RRJ[1] = 0.0f;
		gClearColor_RRJ[2] = 0.0f;
		gClearColor_RRJ[3] = 1.0f;


		return(S_OK);
}


HRESULT LoadD3DTexture(const wchar_t *textureFileName, ID3D11ShaderResourceView **ppID3DShaderResourceView){

	HRESULT hr_RRJ = S_OK;

	hr_RRJ = CreateWICTextureFromFile(
		gpID3D11Device_RRJ,
		gpID3D11DeviceContext_RRJ,
		textureFileName,
		NULL,
		ppID3DShaderResourceView);

	
	fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
	switch(hr_RRJ){
		case S_OK:
			fprintf_s(gbFile_RRJ, "\t\t S_OK\n");
			break;

		case S_FALSE:
			fprintf_s(gbFile_RRJ, "\t\t S_FALSE \n");
			break;

		case E_NOTIMPL:
			fprintf_s(gbFile_RRJ, "\t\t E_NOTIMPL \n");
			break;

		case E_OUTOFMEMORY:
			fprintf_s(gbFile_RRJ, "\t\t E_OUTOFMEMORY \n");
			break;

		case E_INVALIDARG:	
			fprintf_s(gbFile_RRJ, "\t\t E_INVALIDARG \n");
			break;

		case E_FAIL:	
			fprintf_s(gbFile_RRJ, "\t\t E_FAIL \n");
			break;

		case DXGI_ERROR_WAS_STILL_DRAWING:
			fprintf_s(gbFile_RRJ, "\t\t DXGI_ERROR_WAS_STILL_DRAWING \n");
			break;

		case DXGI_ERROR_INVALID_CALL:
			fprintf_s(gbFile_RRJ, "\t\t DXGI_ERROR_INVALID_CALL \n");
			break;

		case D3D11_ERROR_DEFERRED_CONTEXT_MAP_WITHOUT_INITIAL_DISCARD:
			fprintf_s(gbFile_RRJ, "\t\t D3D11_ERROR_DEFERRED_CONTEXT_MAP_WITHOUT_INITIAL_DISCARD \n");
			break;

		case D3D11_ERROR_TOO_MANY_UNIQUE_VIEW_OBJECTS:
			fprintf_s(gbFile_RRJ, "\t\t D3D11_ERROR_TOO_MANY_UNIQUE_VIEW_OBJECTS \n");
			break;

		case D3D11_ERROR_TOO_MANY_UNIQUE_STATE_OBJECTS:
			fprintf_s(gbFile_RRJ, "\t\t D3D11_ERROR_TOO_MANY_UNIQUE_STATE_OBJECTS \n");
			break;

		case D3D11_ERROR_FILE_NOT_FOUND:
			fprintf_s(gbFile_RRJ, "\t\t D3D11_ERROR_FILE_NOT_FOUND \n");
			break;


		case E_ABORT:
			fprintf_s(gbFile_RRJ, "\t\t E_ABORT \n");
			break;

		case E_ACCESSDENIED:
			fprintf_s(gbFile_RRJ, "\t\t E_ACCESSDENIED \n");
			break;

		case E_HANDLE:
			fprintf_s(gbFile_RRJ, "\t\t E_HANDLE \n");
			break;

		case E_POINTER:
			fprintf_s(gbFile_RRJ, "\t\t E_POINTER \n");
			break;

		case E_UNEXPECTED:
			fprintf_s(gbFile_RRJ, "\t\t E_UNEXPECTED\n");
			break;


		case E_NOINTERFACE:
			fprintf_s(gbFile_RRJ, "\t\t E_NOINTERFACE \n");
			break;	
		
		default:
			fprintf_s(gbFile_RRJ, "\t\t UNKNOWN \n");
			break;
	}

	fclose(gbFile_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: CreateWICTextureFromFile()\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: CreateWICTexureFromFile()\n");
		fclose(gbFile_RRJ);	
	}

	return(hr_RRJ);

}


void uninitialize(void){

	if(gpID3D11SamplerState_RRJ){
		gpID3D11SamplerState_RRJ->Release();
		gpID3D11SamplerState_RRJ = NULL;
	}

	if(gpID3D11ShaderResourceView_RRJ){
		gpID3D11ShaderResourceView_RRJ->Release();
		gpID3D11ShaderResourceView_RRJ = NULL;
	}


	if(gpID3D11RasterizerState_RRJ){
		gpID3D11RasterizerState_RRJ->Release();
		gpID3D11RasterizerState_RRJ = NULL;
	}

	if(gpID3D11Buffer_ConstantBuffer_RRJ){
		gpID3D11Buffer_ConstantBuffer_RRJ->Release();
		gpID3D11Buffer_ConstantBuffer_RRJ = NULL;
	}

	if(gpID3D11Buffer_VertexBuffer_Cube_PCNT_RRJ){
		gpID3D11Buffer_VertexBuffer_Cube_PCNT_RRJ->Release();
		gpID3D11Buffer_VertexBuffer_Cube_PCNT_RRJ = NULL;
	}

	if(gpID3D11InputLayout_RRJ){
		gpID3D11InputLayout_RRJ->Release();
		gpID3D11InputLayout_RRJ = NULL;
	}


	if(gpID3D11DepthStencilView_RRJ){
		gpID3D11DepthStencilView_RRJ->Release();
		gpID3D11DepthStencilView_RRJ = NULL;
	}

	if(gpID3D11RenderTargetView_RRJ){
		gpID3D11RenderTargetView_RRJ->Release();
		gpID3D11RenderTargetView_RRJ = NULL;
	}


	if(gpID3D11PixelShader_RRJ){
		gpID3D11PixelShader_RRJ->Release();
		gpID3D11PixelShader_RRJ = NULL;
	}

	if(gpID3D11VertexShader_RRJ){
		gpID3D11VertexShader_RRJ->Release();
		gpID3D11VertexShader_RRJ = NULL;
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
		fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: Log Close!!\n");
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


	gpIDXGISwapChain_RRJ->ResizeBuffers(1, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, 0);

	ID3D11Texture2D *pID3D11Texture2D_BackBuffer_RRJ = NULL;

	gpIDXGISwapChain_RRJ->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&pID3D11Texture2D_BackBuffer_RRJ);


	hr_RRJ = gpID3D11Device_RRJ->CreateRenderTargetView(
		pID3D11Texture2D_BackBuffer_RRJ,
		NULL, 
		&gpID3D11RenderTargetView_RRJ
		);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: CreateRenderTargetView()\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: CreateRenderTargetView()\n");
		fclose(gbFile_RRJ);	
	}


	pID3D11Texture2D_BackBuffer_RRJ->Release();
	pID3D11Texture2D_BackBuffer_RRJ = NULL;



	/********** DEPTH STENCIL VIEW **********/
	D3D11_TEXTURE2D_DESC textureDesc_RRJ;
	ZeroMemory((void*)&textureDesc_RRJ, sizeof(D3D11_TEXTURE2D_DESC));

	textureDesc_RRJ.Usage = D3D11_USAGE_DEFAULT;
	textureDesc_RRJ.Width = width;
	textureDesc_RRJ.Height = height;
	textureDesc_RRJ.Format = DXGI_FORMAT_D32_FLOAT;
	textureDesc_RRJ.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	textureDesc_RRJ.MipLevels = 1;
	textureDesc_RRJ.ArraySize = 1;
	textureDesc_RRJ.SampleDesc.Count = 1;
	textureDesc_RRJ.SampleDesc.Quality = 0;
	textureDesc_RRJ.CPUAccessFlags = 0; //No CPU ACCESS!!
	textureDesc_RRJ.MiscFlags = 0;


	ID3D11Texture2D *pID3D11Texture2D_DepthBuffer_RRJ = NULL;

	hr_RRJ = gpID3D11Device_RRJ->CreateTexture2D(&textureDesc_RRJ, NULL, &pID3D11Texture2D_DepthBuffer_RRJ);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: CreateTexture2D()\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: CreateTexture2D()\n");
		fclose(gbFile_RRJ);	
	}


	D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc_RRJ;
	ZeroMemory((void*)&depthStencilViewDesc_RRJ, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));

	depthStencilViewDesc_RRJ.Format = DXGI_FORMAT_D32_FLOAT;
	depthStencilViewDesc_RRJ.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;

	hr_RRJ = gpID3D11Device_RRJ->CreateDepthStencilView(
		pID3D11Texture2D_DepthBuffer_RRJ,
		&depthStencilViewDesc_RRJ,
		&gpID3D11DepthStencilView_RRJ
		);

	if(FAILED(hr_RRJ)){
		fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "ERROR: CreateDepthStencilView()\n");
		fclose(gbFile_RRJ);
		return(hr_RRJ);
	}
	else{
		fopen_s(&gbFile_RRJ, gLogFileName_RRJ, "a+");
		fprintf_s(gbFile_RRJ, "SUCCESS: CreateDepthStencilView()\n");
		fclose(gbFile_RRJ);	
	}



	gpID3D11DeviceContext_RRJ->OMSetRenderTargets(1, 
		&gpID3D11RenderTargetView_RRJ,
		gpID3D11DepthStencilView_RRJ);



	/********** View Port **********/
	D3D11_VIEWPORT d3dViewPort_RRJ;

	d3dViewPort_RRJ.TopLeftX = 0.0f;
	d3dViewPort_RRJ.TopLeftY = 0.0f;
	d3dViewPort_RRJ.Width = (float)width;
	d3dViewPort_RRJ.Height = (float)height;
	d3dViewPort_RRJ.MinDepth = 0.0f;
	d3dViewPort_RRJ.MaxDepth = 1.0f;

	gpID3D11DeviceContext_RRJ->RSSetViewports(1, &d3dViewPort_RRJ);


	gPerspectiveProjectionMatrix_RRJ =XMMatrixPerspectiveFovLH(
		XMConvertToRadians(45.0f),
		(float)width / (float)height,
		0.1f,
		100.0f);


	return(S_OK);
}



void display(void){

	static float angle_Cube_RRJ = 0.0f;


	gpID3D11DeviceContext_RRJ->ClearRenderTargetView(gpID3D11RenderTargetView_RRJ, gClearColor_RRJ);
	gpID3D11DeviceContext_RRJ->ClearDepthStencilView(gpID3D11DepthStencilView_RRJ, D3D11_CLEAR_DEPTH, 1.0f, 0);

	//Position
	UINT stride_RRJ = sizeof(float) * 11;
	UINT offset_RRJ = sizeof(float) * 0;
	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(0, 1, &gpID3D11Buffer_VertexBuffer_Cube_PCNT_RRJ, &stride_RRJ, &offset_RRJ);

	//Color
	stride_RRJ = sizeof(float) * 11;
	offset_RRJ = sizeof(float) * 3;
	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(1, 1, &gpID3D11Buffer_VertexBuffer_Cube_PCNT_RRJ, &stride_RRJ, &offset_RRJ);

	//Normal
	stride_RRJ = sizeof(float) * 11;
	offset_RRJ = sizeof(float) * 6;
	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(2, 1, &gpID3D11Buffer_VertexBuffer_Cube_PCNT_RRJ, &stride_RRJ, &offset_RRJ);

	//TexCoord
	stride_RRJ = sizeof(float) * 11;
	offset_RRJ = sizeof(float) * 9;
	gpID3D11DeviceContext_RRJ->IASetVertexBuffers(3, 1, &gpID3D11Buffer_VertexBuffer_Cube_PCNT_RRJ, &stride_RRJ, &offset_RRJ);


	gpID3D11DeviceContext_RRJ->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);


	XMMATRIX worldMatrix_RRJ = XMMatrixIdentity();
	XMMATRIX viewMatrix_RRJ = XMMatrixIdentity();
	XMMATRIX translateMatrix_RRJ = XMMatrixIdentity();
	XMMATRIX r1 = XMMatrixIdentity();
	XMMATRIX r2 = XMMatrixIdentity();
	XMMATRIX r3 = XMMatrixIdentity();
	XMMATRIX rotateMatrix_RRJ = XMMatrixIdentity();

	translateMatrix_RRJ = XMMatrixTranslation(0.0f, 0.0f, 5.0f);

	r1 = XMMatrixRotationX(-angle_Cube_RRJ);
	r2 = XMMatrixRotationY(-angle_Cube_RRJ);
	r3 = XMMatrixRotationZ(-angle_Cube_RRJ);
	rotateMatrix_RRJ = r1 * r2 * r3;

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


	gpID3D11DeviceContext_RRJ->PSSetShaderResources(0, 1, &gpID3D11ShaderResourceView_RRJ);
	gpID3D11DeviceContext_RRJ->PSSetSamplers(0, 1, &gpID3D11SamplerState_RRJ);

	gpID3D11DeviceContext_RRJ->UpdateSubresource(gpID3D11Buffer_ConstantBuffer_RRJ,
		0, NULL, &constantBuffer_RRJ, 0, 0);


	gpID3D11DeviceContext_RRJ->Draw(6 * 6, 0);

	gpIDXGISwapChain_RRJ->Present(0, 0);

	angle_Cube_RRJ = angle_Cube_RRJ + 0.005f;
	if(angle_Cube_RRJ > 360.0f)
		angle_Cube_RRJ = 0.0f;

}

