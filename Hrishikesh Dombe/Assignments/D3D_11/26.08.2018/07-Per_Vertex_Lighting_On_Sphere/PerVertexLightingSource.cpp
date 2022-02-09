#include <Windows.h>
#include <stdio.h> 

#include <d3d11.h>
#include <d3dcompiler.h> //For Shader and Shader Compilation

#pragma warning(disable: 4838) //Typecasting warning supressed.
#include "XNAMath_204\xnamath.h" //For Maths

#include "Sphere.h" //New

#pragma comment (lib, "user32.lib")
#pragma comment (lib, "gdi32.lib")
#pragma comment (lib, "d3d11.lib")
#pragma comment (lib, "d3dcompiler.lib")

#pragma comment(lib, "Sphere.lib") //New

#define WIN_WIDTH 800
#define WIN_HEIGHT 600 

//Global Function Declarations
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

//Global Variable Declarations
FILE *gpFile = NULL;
char gszLogFileName[] = "Log.txt";

HWND ghWnd = NULL;

DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

bool gbActiveWindow = false;
bool gbEscapeKeyIsPressed = false;
bool gbFullScreen = false;

//New (For Sphere.dll)
float sphere_vertices[1146];
float sphere_normals[1146];
float sphere_textures[764];
unsigned short sphere_elements[2280];

unsigned int gNumVertices;
unsigned int gNumElements;

float gClearColor[4]; //RGBA
IDXGISwapChain *gpIDXGISwapChain = NULL;
ID3D11Device *gpID3D11Device = NULL;
ID3D11DeviceContext *gpID3D11DeviceContext = NULL;
ID3D11RenderTargetView *gpID3D11RenderTargetView = NULL;
ID3D11RasterizerState *gpID3D11RasterizerState = NULL; //To disable culling.
ID3D11DepthStencilView *gpID3D11DepthStencilView = NULL; //For Depth

ID3D11VertexShader *gpID3D11VertexShader = NULL;
ID3D11PixelShader *gpID3D11PixelShader = NULL;
ID3D11InputLayout *gpID3D11InputLayout = NULL;

ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Sphere_Position = NULL;
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Sphere_Normal = NULL;
ID3D11Buffer *gpID3D11Buffer_IndexBuffer_Sphere_Element = NULL; //Index Buffer Object for Sphere Element
ID3D11Buffer *gpID3D11Buffer_ConstantBuffer = NULL;

bool gbLight;

struct CBUFFER
{
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
	float MaterialShininess;
	UINT LKeyPressed;
};

/*struct CBUFFER
{
	XMMATRIX WorldMatrix;
	XMMATRIX ViewMatrix;
	XMMATRIX ProjectionMatrx;
	XMVECTOR u_La;
	XMVECTOR u_Ld;
	XMVECTOR u_Ls;
	XMVECTOR u_Ka;
	XMVECTOR u_Kd;
	XMVECTOR u_Ks;
	float u_material_shininess;
	XMVECTOR u_light_position;
	UINT u_lkeypressed;
};*/

XMMATRIX gPerspectiveProjectionMatrix;

//WinMain()
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	//Function Declarations
	HRESULT initialize(void);
	void display(void);
	void rotateObject(void);
	void uninitialize(void);

	//Variable Declarations
	WNDCLASSEX wndClass;
	HWND hWnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("Direct3D11_PerVetexLight");
	bool bDone = false;

	//Log File
	if (fopen_s(&gpFile, gszLogFileName, "w") != 0)
	{
		MessageBox(NULL, TEXT("Log File Cannont be created.\n Exitting..."), TEXT("Error"), MB_OK | MB_TOPMOST | MB_ICONSTOP);
		exit(0);
	}
	else
	{
		fprintf_s(gpFile, "Log File is successfully created!\n");
		fclose(gpFile);
	}

	//Initilaize WNDCLASSEX Structure
	wndClass.cbSize = sizeof(WNDCLASSEX);
	wndClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndClass.cbClsExtra = 0;
	wndClass.cbWndExtra = 0;
	wndClass.lpfnWndProc = WndProc;
	wndClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndClass.hInstance = hInstance;
	wndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndClass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndClass.lpszClassName = szClassName;
	wndClass.lpszMenuName = NULL;

	//Register
	RegisterClassEx(&wndClass);

	//Create Window
	hWnd = CreateWindow(szClassName,
		TEXT("Direct3D11 Per Vertex Lighting on a Steady sphere"),
		WS_OVERLAPPEDWINDOW,
		100,
		100,
		WIN_WIDTH,
		WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghWnd = hWnd;

	ShowWindow(hWnd, iCmdShow);
	SetForegroundWindow(hWnd);
	SetFocus(hWnd);

	//Initialize D3D
	HRESULT hr;
	hr = initialize(); //This was void till now in OpenGL
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "\nInitilaize() Failed. Exitting now..\n");
		fclose(gpFile);
		DestroyWindow(hWnd);
		hWnd = NULL;
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "\nInitilaize() Succeeded.\n");
		fclose(gpFile);
	}

	//Message Loop
	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
				bDone = true;
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			if (gbActiveWindow == true)
			{
				display();

				if (gbEscapeKeyIsPressed == true)
					bDone = true;
			}
		}
	}

	uninitialize();

	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	//Function Declarations
	HRESULT resize(int, int);
	void ToggleFullScreen(void);
	void uninitialize(void);

	//Variable Declarations
	HRESULT hr;

	static bool bIsLKeyPressed = false;

	//Code
	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			gbActiveWindow = true;
		else
			gbActiveWindow = false;
		break;

	case WM_ERASEBKGND:
		return(0);

	case WM_SIZE:
		if (gpID3D11DeviceContext)
		{
			hr = resize(LOWORD(lParam), HIWORD(lParam));

			if (FAILED(hr))
			{
				fopen_s(&gpFile, gszLogFileName, "a+");
				fprintf_s(gpFile, "\nresize() Failed.\n");
				fclose(gpFile);
				return(hr);
			}
			else
			{
				fopen_s(&gpFile, gszLogFileName, "a+");
				fprintf_s(gpFile, "\nresize() Succeeded.\n");
				fclose(gpFile);
			}
		}
		break;

	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			gbEscapeKeyIsPressed = true;
			break;

		case 0x46: //for 'F' or 'f'
			if (gbFullScreen == false)
			{
				ToggleFullScreen();
				gbFullScreen = true;
			}
			else
			{
				ToggleFullScreen();
				gbFullScreen = false;
			}
			break;

		case 0x4C: //For 'L' or 'l'
			if (bIsLKeyPressed == false)
			{
				gbLight = true;
				bIsLKeyPressed = true;
			}
			else
			{
				gbLight = false;
				bIsLKeyPressed = false;
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

	default:
		break;
	}
	return(DefWindowProc(hWnd, iMsg, wParam, lParam));
}

void ToggleFullScreen(void)
{
	//Variable Declarations
	MONITORINFO mi;

	//Code
	if (gbFullScreen == false)
	{
		dwStyle = GetWindowLong(ghWnd, GWL_STYLE);

		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			mi = { sizeof(MONITORINFO) };
			if (GetWindowPlacement(ghWnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghWnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghWnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghWnd,
					HWND_TOP,
					mi.rcMonitor.left,
					mi.rcMonitor.top,
					mi.rcMonitor.right - mi.rcMonitor.left,
					mi.rcMonitor.bottom - mi.rcMonitor.top,
					SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
	}

	else
	{
		//Code
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghWnd, &wpPrev);
		SetWindowPos(ghWnd,
			HWND_TOP,
			0,
			0,
			0,
			0,
			SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);

		ShowCursor(TRUE);
	}
}

HRESULT initialize(void)
{
	//Function Declarations
	HRESULT resize(int, int);
	void uninitialize(void);

	//Variable Declarations
	HRESULT hr;
	D3D_DRIVER_TYPE d3dDriverType;
	D3D_DRIVER_TYPE d3dDriverTypes[] = { D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP, D3D_DRIVER_TYPE_REFERENCE }; //Sequence is important.

	D3D_FEATURE_LEVEL d3dFeatureLevel_required = D3D_FEATURE_LEVEL_11_0;
	D3D_FEATURE_LEVEL d3dFeatureLevel_acquired = D3D_FEATURE_LEVEL_10_0; //Lowest, Default

	UINT createDeviceFlags = 0;
	UINT numDriverTypes = 0;
	UINT numFeatureLevels = 1; //This is based on required feature level

	//Code
	numDriverTypes = sizeof(d3dDriverTypes) / sizeof(d3dDriverTypes[0]); //Calculating Size of Array

	DXGI_SWAP_CHAIN_DESC dxgiSwapChainDesc;

	ZeroMemory((void *)&dxgiSwapChainDesc, sizeof(DXGI_SWAP_CHAIN_DESC));

	dxgiSwapChainDesc.BufferCount = 1;
	dxgiSwapChainDesc.BufferDesc.Width = WIN_WIDTH;
	dxgiSwapChainDesc.BufferDesc.Height = WIN_HEIGHT;
	dxgiSwapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	dxgiSwapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
	dxgiSwapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
	dxgiSwapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	dxgiSwapChainDesc.OutputWindow = ghWnd;
	dxgiSwapChainDesc.SampleDesc.Count = 1;
	dxgiSwapChainDesc.SampleDesc.Quality = 0;
	dxgiSwapChainDesc.Windowed = TRUE;

	for (UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes; driverTypeIndex++)
	{
		d3dDriverType = d3dDriverTypes[driverTypeIndex];

		hr = D3D11CreateDeviceAndSwapChain(
			NULL,
			d3dDriverType,
			NULL,
			createDeviceFlags,
			&d3dFeatureLevel_required,
			numFeatureLevels,
			D3D11_SDK_VERSION,
			&dxgiSwapChainDesc,
			&gpIDXGISwapChain,
			&gpID3D11Device,
			&d3dFeatureLevel_acquired,
			&gpID3D11DeviceContext
		);

		if (SUCCEEDED(hr))
			break;
	}

	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "\nD3D11CreateDeviceAndSwapChain() Failed.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "\nD3D11CreateDeviceAndSwapChain() Succeeded.\n");

		fprintf_s(gpFile, "The Chosen Driver is of: ");

		if (d3dDriverType == D3D_DRIVER_TYPE_HARDWARE)
		{
			fprintf_s(gpFile, "Hardware Type.\n");
		}
		else if (d3dDriverType == D3D_DRIVER_TYPE_WARP)
		{
			fprintf_s(gpFile, "Warp Type.\n");
		}
		else if (d3dDriverType == D3D_DRIVER_TYPE_REFERENCE)
		{
			fprintf_s(gpFile, "Reference Type.\n");
		}
		else
		{
			fprintf_s(gpFile, "Unknown Type.\n");
		}

		fprintf_s(gpFile, "The supported Highest Feature Level is: ");

		if (d3dFeatureLevel_acquired == D3D_FEATURE_LEVEL_11_0)
		{
			fprintf_s(gpFile, "11.0\n");
		}
		else if (d3dFeatureLevel_acquired == D3D_FEATURE_LEVEL_10_1)
		{
			fprintf_s(gpFile, "10.1\n");
		}
		else if (d3dFeatureLevel_acquired == D3D_FEATURE_LEVEL_10_0)
		{
			fprintf_s(gpFile, "10.0\n");
		}
		else
		{
			fprintf_s(gpFile, "Unknown.\n");
		}
		fclose(gpFile);
	}

	//Initialize Shader, Input Layouts, Constant Buffers, etc.

	//--------------------Vertex Shader--------------------
	//Shader Source Code
	const char *vertexShaderSourceCode =
		"cbuffer ConstantBuffer" \
		"{" \
		"float4x4 worldMatrix;" \
		"float4x4 viewMatrix;" \
		"float4x4 projectionMatrix;" \
		"float3 la;" \
		"float3 ld;" \
		"float3 ls;" \
		"float4 lightPosition;" \
		"float3 ka;" \
		"float3 kd;" \
		"float3 ks;" \
		"float material_shininess;" \
		"uint lKeyPressed;" \
		"}" \
		"struct vertex_output" \
		"{" \
		"float4 position : SV_POSITION;" \
		"float3 phong_ads_color : COLOR;" \
		"};" \
		"vertex_output main(float4 pos : POSITION, float4 normal : NORMAL)" \
		"{" \
		"vertex_output output;" \
		"if(lKeyPressed == 0)" \
		"{" \
		"float4 eyeCoordinates = mul(viewMatrix, mul(worldMatrix, pos));" \
		"float3 transformed_normals = normalize(mul((float3x3)mul(viewMatrix, worldMatrix), (float3)normal));" \
		"float3 light_direction = (float3)normalize(lightPosition - eyeCoordinates.xyz);" \
		"float tn_dot_ld = max(dot(transformed_normals, light_direction), 0.0);" \
		"float3 reflection_vector = reflect(-light_direction, transformed_normals);" \
		"float3 viewer_vector = normalize(-eyeCoordinates.xyz);" \
		"float3 ambient = mul(la, ka);" \
		"float3 diffuse = mul(ld, mul(kd, tn_dot_ld));" \
		"float3 specular = mul(ls, mul(ks, pow(max(dot(reflection_vector, viewer_vector), 0.0), material_shininess)));" \
		"output.phong_ads_color = ambient + diffuse + specular;" \
		"}" \
		"else" \
		"{" \
		"output.phong_ads_color = float3(1.0, 1.0, 1.0);" \
		"}" \
		"output.position = mul(projectionMatrix, mul(viewMatrix, mul(worldMatrix, pos)));" \
		"return(output);" \
		"}";
		/*"cbuffer ConstantBuffer" \
		"{" \
		"float4x4 worldMatrix;" \
		"float4x4 viewMatrix;" \
		"float4x4 projectionMatrix;" \
		"float4 u_La;" \
		"float4 u_Ld;" \
		"float4 u_Ls;" \
		"float4 u_Ka;" \
		"float4 u_Kd;" \
		"float4 u_Ks;" \
		"float u_material_shininess;" \
		"float4 u_light_position;" \
		"uint u_lkeypressed;" \
		"}" \
		"struct vertex_output" \
		"{" \
		"float4 vertex : SV_POSITION;" \
		"float4 diffuse_light : COLOR;" \
		"};" \
		"vertex_output main(float4 pos : POSITION, float4 normal : NORMAL )" \
		"{" \
		"vertex_output output;" \
		"if(u_lkeypressed == 0)" \
		"{" \
		"float4 eyeCoordinates = mul(viewMatrix,mul(worldMatrix,pos));" \
		"float3 transformed_normals = normalize(mul((float3x3)mul(worldMatrix,viewMatrix),(float3)normal));" \
		"float3 light_direction = (float3)normalize(u_light_position - eyeCoordinates);" \
		"float tn_dot_ld = max(dot(transformed_normals,light_direction),0.0);" \
		"float3 reflection_vector = reflect(-light_direction,transformed_normals);" \
		"float3 viewer_vector = normalize(-eyeCoordinates.xyz);" \
		"float3 ambient = mul((float3)u_La ,(float3)u_Ka);" \
		"float3 diffuse = mul((float3)u_Ld , mul((float3)u_Kd , float3(tn_dot_ld,0.0,0.0)));" \
		"float3 specular = mul((float3)u_Ls,mul((float3)u_Ks,float3(pow(max(dot(reflection_vector,viewer_vector),0.0),u_material_shininess),0.0,0.0)));" \
		"output.diffuse_light = float4(ambient + diffuse + specular,1.0);" \
		"}" \
		"else" \
		"{" \
		"output.diffuse_light = float4(1.0f,1.0f,1.0f,1.0f);" \
		"}" \
		"output.vertex = mul(mul(projectionMatrix,mul(viewMatrix,worldMatrix)),pos);" \
		"return(output);" \
		"}";*/

	ID3DBlob *pID3DBlob_VertexShaderCode = NULL; //Holds Shader's ByteCode
	ID3DBlob *pID3DBlob_Error = NULL; //Holds error, if any.

	//Compile Shader
	hr = D3DCompile
	(vertexShaderSourceCode,
		lstrlenA(vertexShaderSourceCode) + 1, //Source Code Length
		"VS", //Macro for Vertex Shader
		NULL, //Macro for Pointer
		D3D_COMPILE_STANDARD_FILE_INCLUDE, //Telling to include standard files of compiler
		"main", //Entry Point Function
		"vs_5_0", //Shader Feature Level
		0, //Compiler Constant
		0, //Effect Constant
		&pID3DBlob_VertexShaderCode,
		&pID3DBlob_Error
	);

	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf(gpFile, "D3DCompile() failed for Vertex Shader : %s.\n", (char *)pID3DBlob_Error->GetBufferPointer());
			fclose(gpFile);

			pID3DBlob_Error->Release();
			pID3DBlob_Error = NULL;

			return(hr);
		}
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf(gpFile, "D3DCompile() succeeded for Vertex Shader.\n");
		fclose(gpFile);
	}

	//Create Shader
	hr = gpID3D11Device->CreateVertexShader(pID3DBlob_VertexShaderCode->GetBufferPointer(),
		pID3DBlob_VertexShaderCode->GetBufferSize(),
		NULL,
		&gpID3D11VertexShader);

	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf(gpFile, "ID3D11Device:CreateVertexShader() failed.\n");
			fclose(gpFile);

			pID3DBlob_VertexShaderCode->Release();
			pID3DBlob_VertexShaderCode = NULL;

			return(hr);
		}
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf(gpFile, "ID3D11Device:CreateVertexShader() succeeded.\n");
		fclose(gpFile);
	}

	//Set Vertex Shader State of Pipeline
	gpID3D11DeviceContext->VSSetShader(gpID3D11VertexShader, 0, 0);

	//--------------------Pixel Shader--------------------
	//Shader Source Code
	const char *pixelShaderSourceCode =
		"struct vertex_output" \
		"{" \
		"float4 position : SV_POSITION;" \
		"float3 phong_ads_color : COLOR;" \
		"};" \
		"float4 main(vertex_output input) : SV_TARGET" \
		"{" \
		"return(float4(input.phong_ads_color, 1.0));" \
		"}";
		/*"float4 main(float4 position : SV_POSITION, float4 color : COLOR) : SV_TARGET" \
		"{" \
		"float4 final_color = color;" \
		"return(final_color);" \
		"}";*/

	ID3DBlob *pID3DBlob_PixelShaderCode = NULL; //Holds Shader's ByteCode

	//Re-initializing
	pID3DBlob_Error = NULL; //Holds error, if any.

	//Compile Shader
	hr = D3DCompile
	(pixelShaderSourceCode,
		lstrlenA(pixelShaderSourceCode) + 1, //Source Code Length
		"PS", //Macro for Pixel Shader
		NULL, //Macro for Pointer
		D3D_COMPILE_STANDARD_FILE_INCLUDE, //Telling to include standard files of compiler
		"main", //Entry Point Function
		"ps_5_0", //Shader Feature Level
		0, //Compiler Constant
		0, //Effect Constant
		&pID3DBlob_PixelShaderCode,
		&pID3DBlob_Error
	);

	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf(gpFile, "D3DCompile() failed for Pixel Shader : %s.\n", (char *)pID3DBlob_Error->GetBufferPointer());
			fclose(gpFile);

			pID3DBlob_Error->Release();
			pID3DBlob_Error = NULL;

			//Releasing Vertex Shader ByteCode
			pID3DBlob_VertexShaderCode->Release();
			pID3DBlob_VertexShaderCode = NULL;

			return(hr);
		}
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf(gpFile, "D3DCompile() succeeded for Pixel Shader.\n");
		fclose(gpFile);
	}

	//Create Shader
	hr = gpID3D11Device->CreatePixelShader(pID3DBlob_PixelShaderCode->GetBufferPointer(),
		pID3DBlob_PixelShaderCode->GetBufferSize(),
		NULL,
		&gpID3D11PixelShader);

	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf(gpFile, "ID3D11Device:CreatePixelShader() failed.\n");
			fclose(gpFile);

			//Releasing Pixel Shader ByteCode
			pID3DBlob_PixelShaderCode->Release();
			pID3DBlob_PixelShaderCode = NULL;

			//Releasing Vertex Shader ByteCode
			pID3DBlob_VertexShaderCode->Release();
			pID3DBlob_VertexShaderCode = NULL;

			return(hr);
		}
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf(gpFile, "ID3D11Device:CreatePixelShader() succeeded.\n");
		fclose(gpFile);
	}

	//Set Pixel Shader State of Pipeline
	gpID3D11DeviceContext->PSSetShader(gpID3D11PixelShader, 0, 0);

	//--------------------Input Layout--------------------
	//Fill Structure (Define Input Layout)
	D3D11_INPUT_ELEMENT_DESC inputElementDesc[2];

	ZeroMemory(&inputElementDesc, sizeof(D3D11_INPUT_ELEMENT_DESC));

	inputElementDesc[0].SemanticName = "POSITION";
	inputElementDesc[0].SemanticIndex = 0;
	inputElementDesc[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	inputElementDesc[0].InputSlot = 0;
	inputElementDesc[0].AlignedByteOffset = 0;
	inputElementDesc[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	inputElementDesc[0].InstanceDataStepRate = 0;

	inputElementDesc[1].SemanticName = "NORMAL";
	inputElementDesc[1].SemanticIndex = 0;
	inputElementDesc[1].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	inputElementDesc[1].InputSlot = 1;
	inputElementDesc[1].AlignedByteOffset = 0;
	inputElementDesc[1].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	inputElementDesc[1].InstanceDataStepRate = 0;

	//Create Input Layout
	hr = gpID3D11Device->CreateInputLayout(inputElementDesc,
		2,
		pID3DBlob_VertexShaderCode->GetBufferPointer(),
		pID3DBlob_VertexShaderCode->GetBufferSize(),
		&gpID3D11InputLayout);

	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf(gpFile, "ID3D11Device:CreateInputLayout() failed.\n");
			fclose(gpFile);

			//Releasing Pixel Shader ByteCode
			pID3DBlob_PixelShaderCode->Release();
			pID3DBlob_PixelShaderCode = NULL;

			//Releasing Vertex Shader ByteCode
			pID3DBlob_VertexShaderCode->Release();
			pID3DBlob_VertexShaderCode = NULL;

			return(hr);
		}
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf(gpFile, "ID3D11Device:CreateInputLayout() succeeded.\n");
		fclose(gpFile);
	}

	//Set Input Layout
	gpID3D11DeviceContext->IASetInputLayout(gpID3D11InputLayout);

	//Releasing Pixel Shader ByteCode
	pID3DBlob_PixelShaderCode->Release();
	pID3DBlob_PixelShaderCode = NULL;

	//Releasing Vertex Shader ByteCode
	pID3DBlob_VertexShaderCode->Release();
	pID3DBlob_VertexShaderCode = NULL;

	//New (Getting Data from DLL)---------------------------------------
	getSphereVertexData(sphere_vertices, sphere_normals, sphere_textures, sphere_elements);

	gNumVertices = getNumberOfSphereVertices();
	gNumElements = getNumberOfSphereElements();

	//--------------------VBO for Sphere Position--------------------
	//Create Vertex Buffer (This is VBO)
	D3D11_BUFFER_DESC bufferDesc;

	ZeroMemory(&bufferDesc, sizeof(D3D11_BUFFER_DESC));

	bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc.ByteWidth = sizeof(float) * ARRAYSIZE(sphere_vertices);
	bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr = gpID3D11Device->CreateBuffer(&bufferDesc, NULL, &gpID3D11Buffer_VertexBuffer_Sphere_Position);

	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf(gpFile, "ID3D11Device:CreateBuffer() failed for Vertex Buffer for Sphere Position.\n");
			fclose(gpFile);

			return(hr);
		}
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf(gpFile, "ID3D11Device:CreateBuffer() succeeded for Vertex Buffer for Sphere Position.\n");
		fclose(gpFile);
	}

	//Copy Vertices into "Above Buffer" which we created.
	D3D11_MAPPED_SUBRESOURCE mappedSubResource;

	//Copy Vertices into "Above Buffer" which we created.
	ZeroMemory(&mappedSubResource, sizeof(D3D11_MAPPED_SUBRESOURCE));

	//Map
	gpID3D11DeviceContext->Map(gpID3D11Buffer_VertexBuffer_Sphere_Position, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedSubResource);

	//Copy to Memory
	memcpy(mappedSubResource.pData, sphere_vertices, sizeof(sphere_vertices));

	//Unmap
	gpID3D11DeviceContext->Unmap(gpID3D11Buffer_VertexBuffer_Sphere_Position, 0);

	//--------------------VBO for Sphere Normal--------------------
	//Create Vertex Buffer (This is VBO)
	ZeroMemory(&bufferDesc, sizeof(D3D11_BUFFER_DESC));

	bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc.ByteWidth = sizeof(float) * ARRAYSIZE(sphere_normals);
	bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr = gpID3D11Device->CreateBuffer(&bufferDesc, NULL, &gpID3D11Buffer_VertexBuffer_Sphere_Normal);

	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf(gpFile, "ID3D11Device:CreateBuffer() failed for Vertex Buffer for Sphere Normal.\n");
			fclose(gpFile);

			return(hr);
		}
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf(gpFile, "ID3D11Device:CreateBuffer() succeeded for Vertex Buffer for Sphere Normal.\n");
		fclose(gpFile);
	}

	//Copy Vertices into "Above Buffer" which we created.
	ZeroMemory(&mappedSubResource, sizeof(D3D11_MAPPED_SUBRESOURCE));

	//Map
	gpID3D11DeviceContext->Map(gpID3D11Buffer_VertexBuffer_Sphere_Normal, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedSubResource);

	//Copy to Memory
	memcpy(mappedSubResource.pData, sphere_normals, sizeof(sphere_normals));

	//Unmap
	gpID3D11DeviceContext->Unmap(gpID3D11Buffer_VertexBuffer_Sphere_Normal, 0);

	//--------------------IBO for Sphere Elements--------------------
	//Create Index Buffer (This is IBO)
	ZeroMemory(&bufferDesc, sizeof(D3D11_BUFFER_DESC));

	bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc.ByteWidth = gNumElements * sizeof(short);
	bufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
	bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr = gpID3D11Device->CreateBuffer(&bufferDesc, NULL, &gpID3D11Buffer_IndexBuffer_Sphere_Element);

	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf(gpFile, "ID3D11Device:CreateBuffer() failed for Index Buffer for Sphere Element.\n");
			fclose(gpFile);

			return(hr);
		}
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf(gpFile, "ID3D11Device:CreateBuffer() succeeded for Index Buffer for Sphere Element.\n");
		fclose(gpFile);
	}

	//Copy Indices into above Index Buffer which we created.
	ZeroMemory(&mappedSubResource, sizeof(D3D11_MAPPED_SUBRESOURCE));

	//Map
	gpID3D11DeviceContext->Map(gpID3D11Buffer_IndexBuffer_Sphere_Element, NULL, D3D11_MAP_WRITE_DISCARD, NULL, &mappedSubResource);

	//Copy to Memory
	memcpy(mappedSubResource.pData, sphere_elements, gNumElements * sizeof(short));

	//Unmap
	gpID3D11DeviceContext->Unmap(gpID3D11Buffer_IndexBuffer_Sphere_Element, NULL);

	//--------------------UBO--------------------
	//Define Constant Buffer (This is UBO, related to Uniforms in OpenGL)
	D3D11_BUFFER_DESC bufferDesc_ConstantBuffer;

	ZeroMemory(&bufferDesc_ConstantBuffer, sizeof(D3D11_BUFFER_DESC));

	bufferDesc_ConstantBuffer.Usage = D3D11_USAGE_DEFAULT;
	bufferDesc_ConstantBuffer.ByteWidth = sizeof(CBUFFER);
	bufferDesc_ConstantBuffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

	hr = gpID3D11Device->CreateBuffer(&bufferDesc_ConstantBuffer, nullptr, &gpID3D11Buffer_ConstantBuffer);

	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf(gpFile, "ID3D11Device:CreateBuffer() failed for Constant Buffer.\n");
			fclose(gpFile);

			return(hr);
		}
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf(gpFile, "ID3D11Device:CreateBuffer() succeeded for Constant Buffer.\n");
		fclose(gpFile);
	}

	//Set Constant Buffer
	gpID3D11DeviceContext->VSSetConstantBuffers(0, 1, &gpID3D11Buffer_ConstantBuffer);
	//gpID3D11DeviceContext->PSSetConstantBuffers(0, 1, &gpID3D11Buffer_ConstantBuffer);

	//D3D Clear Color (Blue)
	gClearColor[0] = 0.0f; //R
	gClearColor[1] = 0.0f; //G
	gClearColor[2] = 0.0f; //B
	gClearColor[3] = 0.0f; //A

	//Set Projection Matrix to Identity
	gPerspectiveProjectionMatrix = XMMatrixIdentity();

	//To Disable Culling in DirectX which is by default on
	//Define Rasterizer Buffer
	D3D11_RASTERIZER_DESC rasterizerDesc;

	ZeroMemory(&rasterizerDesc, sizeof(D3D11_RASTERIZER_DESC));

	rasterizerDesc.AntialiasedLineEnable = FALSE; //Default Value
	rasterizerDesc.MultisampleEnable = FALSE; //Default Value
	rasterizerDesc.DepthBias = 0; //Default Value
	rasterizerDesc.DepthBiasClamp = 0.0; //Default Value
	rasterizerDesc.SlopeScaledDepthBias = 0.0; //Default Value
	rasterizerDesc.CullMode = D3D11_CULL_NONE; //Default Value: D3D11_CULL_BACK
	rasterizerDesc.DepthClipEnable = TRUE; //Default Value: FALSE
	rasterizerDesc.FillMode = D3D11_FILL_SOLID; //Default Value
	rasterizerDesc.FrontCounterClockwise = FALSE; //Default Value. Hence, DirectX is Clockwise.
	rasterizerDesc.ScissorEnable = FALSE; //Default Value

	//Create Rasterizer State
	gpID3D11Device->CreateRasterizerState(&rasterizerDesc, &gpID3D11RasterizerState);

	//Set Rasterizer State
	gpID3D11DeviceContext->RSSetState(gpID3D11RasterizerState);

	//Call resize() for the First Time
	hr = resize(WIN_WIDTH, WIN_HEIGHT);

	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "\nresize() Failed.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "\nresize() Succeeded.\n");
		fclose(gpFile);
	}

	return(S_OK);
}

HRESULT resize(int width, int height)
{
	//Code
	HRESULT hr = S_OK;

	//Free any size-dependent resources
	if (gpID3D11DepthStencilView)
	{
		gpID3D11DepthStencilView->Release();
		gpID3D11DepthStencilView = NULL;
	}

	if (gpID3D11RenderTargetView)
	{
		gpID3D11RenderTargetView->Release();
		gpID3D11RenderTargetView = NULL;
	}

	//Resize swap chain buffers accordingly
	gpIDXGISwapChain->ResizeBuffers(1, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, 0);

	//Again get back buffer from swap chain
	ID3D11Texture2D *pID3D11Texture2D_BackBuffer;
	gpIDXGISwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pID3D11Texture2D_BackBuffer);

	//Again get render target view from d3d11 device using above back buffer
	hr = gpID3D11Device->CreateRenderTargetView(pID3D11Texture2D_BackBuffer, NULL, &gpID3D11RenderTargetView);

	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "\nID3D11Device::CreateRenderTargetView() Failed.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "\nID3D11Device::CreateRenderTargetView() Succeeded.\n");
		fclose(gpFile);
	}

	pID3D11Texture2D_BackBuffer->Release();
	pID3D11Texture2D_BackBuffer = NULL;

	//Set render target view as render target
	gpID3D11DeviceContext->OMSetRenderTargets(1, &gpID3D11RenderTargetView, NULL);

	//New Depth Related Code
	D3D11_TEXTURE2D_DESC textureDesc;

	ZeroMemory(&textureDesc, sizeof(D3D11_TEXTURE2D_DESC));

	textureDesc.Width = width; //Width of Depth Buffer
	textureDesc.Height = height; //Height of Depth Buffer
	textureDesc.ArraySize = 1; //Because, it a depth buffer, not texture right now.
	textureDesc.MipLevels = 1; //Because, we are using it as Depth Buffer.
	textureDesc.SampleDesc.Count = 1; //1: min, 4: max
	textureDesc.SampleDesc.Quality = 0; //High Quality. 0: Low Quality. If Count = 1, Quality = 0; If Count =  4, Quality = 1.
	textureDesc.Format = DXGI_FORMAT_D32_FLOAT; //Similar to Depth Bits in PFD
	textureDesc.Usage = D3D11_USAGE_DEFAULT;
	textureDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL; //Depth Buffer
	textureDesc.CPUAccessFlags = 0;
	textureDesc.MiscFlags = 0;

	//Now, to get steps for converting Texture Buffer to the Depth Buffer:
	ID3D11Texture2D *pID3D11Texture2D_DepthBuffer = NULL;

	//Create Texture (Depth) Buffer
	hr = gpID3D11Device->CreateTexture2D(&textureDesc, NULL, &pID3D11Texture2D_DepthBuffer);

	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "\nID3D11Device::CreateTexture2D() Failed.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "\nID3D11Device::CreateTexture2D() Succeeded.\n");
		fclose(gpFile);
	}

	//Initialize Structure:
	D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;

	ZeroMemory(&depthStencilViewDesc, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));

	depthStencilViewDesc.Format = DXGI_FORMAT_D32_FLOAT;
	depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;

	//Create DepthStencilView (DSV)
	hr = gpID3D11Device->CreateDepthStencilView(pID3D11Texture2D_DepthBuffer, &depthStencilViewDesc, &gpID3D11DepthStencilView);

	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "\nID3D11Device::CreateDepthStencilView() Failed.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "\nID3D11Device::CreateDepthStencilView() Succeeded.\n");
		fclose(gpFile);
	}

	pID3D11Texture2D_DepthBuffer->Release();
	pID3D11Texture2D_DepthBuffer = NULL;

	//Set DSV as Render Target
	gpID3D11DeviceContext->OMSetRenderTargets(1, &gpID3D11RenderTargetView, gpID3D11DepthStencilView);

	//Set Viewport
	D3D11_VIEWPORT d3dViewPort;
	d3dViewPort.TopLeftX = 0;
	d3dViewPort.TopLeftY = 0;
	d3dViewPort.Width = (float)width;
	d3dViewPort.Height = (float)height;
	d3dViewPort.MinDepth = 0.0f;
	d3dViewPort.MaxDepth = 1.0f;

	gpID3D11DeviceContext->RSSetViewports(1, &d3dViewPort);

	gPerspectiveProjectionMatrix = XMMatrixPerspectiveFovLH(XMConvertToRadians(45.0f), (float)width / (float)height, 0.1f, 100.0f);

	return(hr);
}

void display(void)
{

	float lightAmbient[] = { 0.0f,0.0f,0.0f,1.0f };
	float lightDiffuse[] = { 1.0f,1.0f,1.0f,1.0f };
	float lightSpecular[] = { 1.0f,1.0f,1.0f,1.0f };
	float lightPosition[] = { 100.0f,100.0f,-100.0f,1.0f };

	float materialAmbient[] = { 0.0f,0.0f,0.0f,1.0f };
	float materialDiffuse[] = { 1.0f,1.0f,1.0f,1.0f };
	float materialSpecular[] = { 1.0f,1.0f,1.0f,1.0f };
	float materialShininess = 50.0f;

	gpID3D11DeviceContext->ClearRenderTargetView(gpID3D11RenderTargetView, gClearColor);

	//Clear DepthStencilView
	gpID3D11DeviceContext->ClearDepthStencilView(gpID3D11DepthStencilView, //Which DSV?
		D3D11_CLEAR_DEPTH, //What to Clear, Depth, Stencil or Both?
		1.0f, //Depth kitine clear karu?
		0 //Stencil kitine clear karu?
	);

	//--------------------Sphere Block--------------------
	//Select which Vertex Buffer to display
	UINT stride = sizeof(float) * 3;
	UINT offset = 0;

	//Set Vertex Buffer for Position here at runtime
	gpID3D11DeviceContext->IASetVertexBuffers(0, 1, &gpID3D11Buffer_VertexBuffer_Sphere_Position, &stride, &offset);

	//Set Vertex Buffer for Normal here at runtime
	gpID3D11DeviceContext->IASetVertexBuffers(1, 1, &gpID3D11Buffer_VertexBuffer_Sphere_Normal, &stride, &offset);

	//Set Index Buffer for Elements here at runtime
	gpID3D11DeviceContext->IASetIndexBuffer(gpID3D11Buffer_IndexBuffer_Sphere_Element, DXGI_FORMAT_R16_UINT, 0);

	//Select Geometry Primitive
	gpID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	//Translation
	XMMATRIX worldMatrix = XMMatrixIdentity();
	XMMATRIX viewMatrix = XMMatrixIdentity();
	XMMATRIX translationMatrix = XMMatrixIdentity();
	
	//Translate Z-Axis by 2.0f
	translationMatrix = XMMatrixTranslation(0.0f, 0.0f, 2.0f);

	//Multiply Rotation Matrix and Translation Matrix to obtain World Matrix
	worldMatrix = worldMatrix * translationMatrix;

	XMMATRIX wvMatrix = worldMatrix * viewMatrix;

	//Final World-View-Projection Matrix is:
	XMMATRIX wvpMatrix = wvMatrix * gPerspectiveProjectionMatrix;

	//Load Data into the Constant Buffer
	CBUFFER constantBuffer;

	ZeroMemory(&constantBuffer, sizeof(CBUFFER));

	constantBuffer.WorldMatrix = worldMatrix;
	constantBuffer.ViewMatrix = viewMatrix;
	constantBuffer.ProjectionMatrix = gPerspectiveProjectionMatrix;
	constantBuffer.La = XMVectorSet(lightAmbient[0], lightAmbient[1], lightAmbient[2], lightAmbient[3]);
	constantBuffer.Ld = XMVectorSet(lightDiffuse[0], lightDiffuse[1], lightDiffuse[2], lightDiffuse[3]);
	constantBuffer.Ls = XMVectorSet(lightSpecular[0], lightSpecular[1], lightSpecular[2], lightSpecular[3]);
	constantBuffer.LightPosition = XMVectorSet(lightPosition[0], lightPosition[1], lightPosition[2], lightPosition[3]);
	constantBuffer.Ka = XMVectorSet(materialAmbient[0], materialAmbient[1], materialAmbient[2], materialAmbient[3]);
	constantBuffer.Kd = XMVectorSet(materialDiffuse[0], materialDiffuse[1], materialDiffuse[2], materialDiffuse[3]);
	constantBuffer.Ks = XMVectorSet(materialSpecular[0], materialSpecular[1], materialSpecular[2], materialSpecular[3]);
	constantBuffer.MaterialShininess = materialShininess;
	/*constantBuffer.WorldMatrix = worldMatrix;
	constantBuffer.ViewMatrix = viewMatrix;
	constantBuffer.ProjectionMatrx = gPerspectiveProjectionMatrix;
	constantBuffer.u_La = XMVectorSet(lightAmbient[0], lightAmbient[1], lightAmbient[2], lightAmbient[3]);
	constantBuffer.u_Ld = XMVectorSet(lightDiffuse[0], lightDiffuse[1], lightDiffuse[2], lightDiffuse[3]);
	constantBuffer.u_Ls = XMVectorSet(lightSpecular[0], lightSpecular[1], lightSpecular[2], lightSpecular[3]);
	constantBuffer.u_light_position = XMVectorSet(lightPosition[0], lightPosition[1], lightPosition[2], lightPosition[3]);
	constantBuffer.u_Ka = XMVectorSet(materialAmbient[0], materialAmbient[1], materialAmbient[2], materialAmbient[3]);
	constantBuffer.u_Kd = XMVectorSet(materialDiffuse[0], materialDiffuse[1], materialDiffuse[2], materialDiffuse[3]);
	constantBuffer.u_Ks = XMVectorSet(materialSpecular[0], materialSpecular[1], materialSpecular[2], materialSpecular[3]);
	constantBuffer.u_material_shininess = materialShininess;*/
	if (gbLight == true)
	{
		gpFile = fopen("Log.txt", "a");
		fprintf(gpFile, "true\n");
		fclose(gpFile);
		constantBuffer.LKeyPressed = 1;
		//constantBuffer.u_lkeypressed = 1;

		/*constantBuffer.La = XMVectorSet(0.0f, 0.0f, 0.0f, 1.0f);
		constantBuffer.Ld = XMVectorSet(1.0f, 1.0f, 1.0f, 1.0f);
		constantBuffer.Ls = XMVectorSet(1.0f, 1.0f, 1.0f, 1.0f);
		constantBuffer.LightPosition = XMVectorSet(100.0f, 100.0f, -100.0f, 1.0f);

		constantBuffer.Ka = XMVectorSet(0.0f, 0.0f, 0.0f, 1.0f);
		constantBuffer.Kd = XMVectorSet(1.0f, 1.0f, 1.0f, 1.0f);
		constantBuffer.Ks = XMVectorSet(1.0f, 1.0f, 1.0f, 1.0f);
		constantBuffer.MaterialShininess = 50.0f;*/

		
	}
	else
	{
		gpFile = fopen("Log.txt", "a");
		fprintf(gpFile, "false\n");
		fclose(gpFile);
		constantBuffer.LKeyPressed = 0;
		//constantBuffer.u_lkeypressed = 0;
	}

	gpID3D11DeviceContext->UpdateSubresource(gpID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//Draw Vertex Buffer to Render Target
	gpID3D11DeviceContext->DrawIndexed(gNumElements, 0, 0);

	//Switch between Front and Back Buffers
	gpIDXGISwapChain->Present(0, 0);
}

void uninitialize(void)
{
	if (gpID3D11RasterizerState)
	{
		gpID3D11RasterizerState->Release();
		gpID3D11RasterizerState = NULL;
	}

	if (gpID3D11Buffer_ConstantBuffer)
	{
		gpID3D11Buffer_ConstantBuffer->Release();
		gpID3D11Buffer_ConstantBuffer = NULL;
	}

	if (gpID3D11InputLayout)
	{
		gpID3D11InputLayout->Release();
		gpID3D11InputLayout = NULL;
	}

	if (gpID3D11Buffer_IndexBuffer_Sphere_Element)
	{
		gpID3D11Buffer_IndexBuffer_Sphere_Element->Release();
		gpID3D11Buffer_IndexBuffer_Sphere_Element = NULL;
	}

	if (gpID3D11Buffer_VertexBuffer_Sphere_Normal)
	{
		gpID3D11Buffer_VertexBuffer_Sphere_Normal->Release();
		gpID3D11Buffer_VertexBuffer_Sphere_Normal = NULL;
	}

	if (gpID3D11Buffer_VertexBuffer_Sphere_Position)
	{
		gpID3D11Buffer_VertexBuffer_Sphere_Position->Release();
		gpID3D11Buffer_VertexBuffer_Sphere_Position = NULL;
	}

	if (gpID3D11PixelShader)
	{
		gpID3D11PixelShader->Release();
		gpID3D11PixelShader = NULL;
	}

	if (gpID3D11VertexShader)
	{
		gpID3D11VertexShader->Release();
		gpID3D11VertexShader = NULL;
	}

	if (gpID3D11DepthStencilView)
	{
		gpID3D11DepthStencilView->Release();
		gpID3D11DepthStencilView = NULL;
	}

	if (gpID3D11RenderTargetView)
	{
		gpID3D11RenderTargetView->Release();
		gpID3D11RenderTargetView = NULL;
	}

	if (gpIDXGISwapChain)
	{
		gpIDXGISwapChain->Release();
		gpIDXGISwapChain = NULL;
	}

	if (gpID3D11DeviceContext)
	{
		gpID3D11DeviceContext->Release();
		gpID3D11DeviceContext = NULL;
	}

	if (gpID3D11Device)
	{
		gpID3D11Device->Release();
		gpID3D11Device = NULL;
	}

	if (gpFile)
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "\nuninitialize() Succeeded.\n");
		fprintf_s(gpFile, "\nLog file is successfully closed!\n");
		fclose(gpFile);
	}
}
