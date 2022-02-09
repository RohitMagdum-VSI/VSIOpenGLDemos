#include<Windows.h>
#include<stdio.h>
#include "Sphere.h"

//D3D specific Header File
#include<d3d11.h>
//For D3D Shader Compilation 
#include<d3dcompiler.h>

//Supress Warning (Unsigned int to int data loss)
#pragma warning( disable:4838 )

//Math Header for D3D, we can also use DirectXMath but is has some issue(bugs) XNAMath is Cross Platform(Works on Desktop, XBOX, Phone)
#include"XNAMath_204\xnamath.h"

#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"user32.lib")
#pragma comment(lib,"gdi32.lib")
#pragma comment(lib,"D3dcompiler.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

FILE *gpFile = NULL;
char gszLogFileName[] = "Log.txt";

HWND ghwnd = NULL;

DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

bool gbActiveWindow = false;
bool gbEscapeKeyIsPressed = false;
bool gbFullscreen = false;
bool gbLight = false;

float gClearColor[4];
IDXGISwapChain *gpIDXGISwapChain = NULL;
ID3D11Device *gpID3D11Device = NULL;
ID3D11DeviceContext *gpID3D11DeviceContext = NULL;
ID3D11RenderTargetView *gpID3D11RenderTargetView = NULL;

ID3D11VertexShader *gpID3D11VertexShader = NULL;
ID3D11PixelShader *gpID3D11PixelShader = NULL;
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Sphere = NULL;
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Sphere_Normal = NULL;
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Sphere_Elements = NULL;
ID3D11InputLayout *gpID3D11InputLayout = NULL;
ID3D11Buffer *gpID3D11Buffer_ConstantBuffer = NULL;
ID3D11RasterizerState *gpID3D11RasterizerState = NULL;
ID3D11DepthStencilView *gpID3D11DepthStencilView = NULL;

struct CBUFFER
{
	XMMATRIX WorldViewMatrix;
	XMMATRIX ProjectionMatrx;
	XMVECTOR u_La;
	XMVECTOR u_Ka;
	XMVECTOR u_light_position;
	unsigned int u_lkeypressed;
};

XMMATRIX gPerspectiveProjectionMatrix;

float gAngle = 0.0f;

float sphere_vertices[1146];
float sphere_normals[1146];
float sphere_textures[764];
unsigned short sphere_elements[2280];
unsigned int gNumVertices, gNumElements;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	HRESULT initialize(void);
	void uninitialize(void);
	void update(void);
	void display(void);

	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("My Direct3D11");
	bool bDone = false;

	if (fopen_s(&gpFile, gszLogFileName, "w") != 0)
	{
		MessageBox(NULL, TEXT("Log File Cannot be Created\nExitting..."), TEXT("ERROR"), MB_OKCANCEL);
		exit(EXIT_FAILURE);
	}
	else
	{
		fprintf_s(gpFile, "Log File Created Successfully.\n");
		fclose(gpFile);
	}

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpszClassName = szClassName;
	wndclass.lpszMenuName = NULL;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);

	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, TEXT("Direct3D 11 3D Rotation"), WS_OVERLAPPEDWINDOW, 100, 100, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);

	ghwnd = hwnd;

	ShowWindow(hwnd, iCmdShow);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	HRESULT hr;

	hr = initialize();
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "Initailize() Failed. Exitting Now...\n");
		fclose(gpFile);
		DestroyWindow(hwnd);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "Initailize() Succeeded.\n");
		fclose(gpFile);
	}

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
				if (gbEscapeKeyIsPressed == true)
					bDone = true;
				update();
				display();
			}
		}
	}

	uninitialize();

	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	HRESULT resize(int, int);
	void ToggleFullscreen(void);
	void uninitialize(void);

	HRESULT hr;

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
				fprintf_s(gpFile, "Resize() Failed.\n");
				fclose(gpFile);
				return(hr);
			}
			else
			{
				fopen_s(&gpFile, gszLogFileName, "a+");
				fprintf_s(gpFile, "Resize() Succeeded.\n");
				fclose(gpFile);
			}
		}
		break;

	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			if (gbEscapeKeyIsPressed == false)
				gbEscapeKeyIsPressed = true;
			break;

		case 0x46:
			if (gbFullscreen == false)
			{
				ToggleFullscreen();
				gbFullscreen = true;
			}
			else
			{
				ToggleFullscreen();
				gbFullscreen = false;
			}
			break;

		case 0x4C:
			if (gbLight == false)
				gbLight = true;
			else
				gbLight = false;
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
		uninitialize();
		PostQuitMessage(0);
		break;
	default:
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullscreen(void)
{
	MONITORINFO mi = { sizeof(MONITORINFO) };

	if (gbFullscreen == false)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		if (dwStyle&WS_OVERLAPPEDWINDOW)
		{
			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
	}

	else
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);

		ShowCursor(TRUE);
	}
}

HRESULT initialize(void)
{
	void uninitialize(int);
	HRESULT resize(int, int);

	HRESULT hr;

	D3D_DRIVER_TYPE d3dDriverType;
	D3D_DRIVER_TYPE d3dDriverTypes[] = { D3D_DRIVER_TYPE_HARDWARE,D3D_DRIVER_TYPE_WARP,D3D_DRIVER_TYPE_REFERENCE };
	D3D_FEATURE_LEVEL d3dFeatureLevel_required = D3D_FEATURE_LEVEL_11_0;
	D3D_FEATURE_LEVEL d3dFeatureLevel_acquired = D3D_FEATURE_LEVEL_10_0;//Default, Lowest
	UINT createDeviceFlags = 0;
	UINT numDriverTypes = 0;
	UINT numFeatureLevels = 1;//Based upon D3DFeatureLevel_Required;

	numDriverTypes = sizeof(d3dDriverTypes) / sizeof(d3dDriverTypes[0]);

	fopen_s(&gpFile, gszLogFileName, "a+");
	fprintf_s(gpFile, "%d.\n", numDriverTypes);
	fclose(gpFile); 

	DXGI_SWAP_CHAIN_DESC dxgiSwapChainDesc;
	ZeroMemory((void*)&dxgiSwapChainDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
	dxgiSwapChainDesc.BufferCount = 1;
	dxgiSwapChainDesc.BufferDesc.Width = WIN_WIDTH;
	dxgiSwapChainDesc.BufferDesc.Height = WIN_HEIGHT;
	dxgiSwapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	dxgiSwapChainDesc.BufferDesc.RefreshRate.Numerator = 60;//FPS
	dxgiSwapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
	dxgiSwapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	dxgiSwapChainDesc.OutputWindow = ghwnd;
	dxgiSwapChainDesc.SampleDesc.Count = 4;
	dxgiSwapChainDesc.SampleDesc.Quality = 1;
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
			&gpID3D11DeviceContext);

		if (SUCCEEDED(hr))
			break;
	}

	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "D3D11CreateDeviceAndSwapChain() Failed.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "D3D11CreateDeviceAndSwapChain() Succeeded.\n");
		fprintf_s(gpFile, "The Chosen Driver Is Of ");

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

		fprintf_s(gpFile, "The Supported Highest Feature Level Is ");
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

	const char *vertexShaderSourceCode =
		"cbuffer ConstantBuffer" \
		"{" \
		"float4x4 worldViewMatrix;" \
		"float4x4 projectionMatrix;" \
		"float4 u_Ld;" \
		"float4 u_Kd;" \
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
		"if(u_lkeypressed == 1)" \
		"{" \
		"float4 eyeCoordinates = mul(worldViewMatrix,pos);" \
		"float3 tnorm = normalize(mul((float3x3)worldViewMatrix,(float3)normal));" \
		"float3 s = (float3)normalize(u_light_position - eyeCoordinates);" \
		"float4 temp_light = mul(u_Kd,max(dot(s,tnorm),0.0));" \
		"output.diffuse_light = mul(u_Ld,temp_light);" \
		"}" \
		"else"
		"{" \
		"output.diffuse_light = float4(1.0f,1.0f,1.0f,1.0f);" \
		"}" \
		"float4x4 mvpMatrix = mul(projectionMatrix,worldViewMatrix);" \
		"output.vertex = mul(mvpMatrix,pos);" \
		"return(output);" \
		"}";

	ID3DBlob *pID3DBlob_VertexShaderCode = NULL;
	ID3DBlob *pID3DBlob_Error = NULL;

	//lstrlenA return count without counting '/0' so we do +1
	//D3D_COMPILE_STANDARD_FILE_INCLUDE here we tell include the file that you think are important in Shader or we can also sepcify the files
	//vs_5_0 we tell Feature Level Of Shader
	//8th parameter Compile Constants
	//9th parameter Shader Constants
	//10th parameter is returned and contains Compiled Byte Code 
	hr = D3DCompile(vertexShaderSourceCode, lstrlenA(vertexShaderSourceCode) + 1, "VS", NULL, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "vs_5_0", 0, 0, &pID3DBlob_VertexShaderCode, &pID3DBlob_Error);

	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf_s(gpFile, "D3DCompile() Failed For Vertex Shader : %s \n", (char *)pID3DBlob_Error->GetBufferPointer());
			fclose(gpFile);
			pID3DBlob_Error->Release();
			pID3DBlob_Error = NULL;
			return(hr);
		}
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "D3DCompile() Succeeded For Vertex Shader. \n");
		fclose(gpFile);
	}

	//3rd Paramter is pointer to variables that should be used across the shader
	hr = gpID3D11Device->CreateVertexShader(pID3DBlob_VertexShaderCode->GetBufferPointer(), pID3DBlob_VertexShaderCode->GetBufferSize(), NULL, &gpID3D11VertexShader);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "gpID3D11Device::CreateVertexShader() Failed. \n");
		fclose(gpFile);
		pID3DBlob_Error->Release();
		pID3DBlob_Error = NULL;
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "gpID3D11Device::CreateVertexShader() Succeeded. \n");
		fclose(gpFile);
	}

	//2nd Parameter is Linkage Object Pointer Array
	//3rd Parameter is Index Of The Object Array
	gpID3D11DeviceContext->VSSetShader(gpID3D11VertexShader, 0, 0);

	const char *pixelShaderSourceCode =
		"float4 main(float4 position : SV_POSITION, float4 color : COLOR) : SV_TARGET" \
		"{" \
		"float4 final_color = color;" \
		"return(final_color);" \
		"}";

	ID3DBlob *pID3DBlob_PixelShaderCode = NULL;
	pID3DBlob_Error = NULL;

	hr = D3DCompile(pixelShaderSourceCode, lstrlenA(pixelShaderSourceCode) + 1, "PS", NULL, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "ps_5_0", 0, 0, &pID3DBlob_PixelShaderCode, &pID3DBlob_Error);

	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf_s(gpFile, "D3DCompile() Failed For Pixel Shader : %s\n", (char*)pID3DBlob_Error->GetBufferPointer());
			fclose(gpFile);
			pID3DBlob_Error->Release();
			pID3DBlob_Error = NULL;
			pID3DBlob_VertexShaderCode->Release();
			pID3DBlob_VertexShaderCode = NULL;
			return(hr);
		}
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "D3DCompile() Succeeded For Pixel Shader.\n");
		fclose(gpFile);
	}

	hr = gpID3D11Device->CreatePixelShader(pID3DBlob_PixelShaderCode->GetBufferPointer(), pID3DBlob_PixelShaderCode->GetBufferSize(), NULL, &gpID3D11PixelShader);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "gpID3D11Device::CreatePixelShader() Failed.\n");
		fclose(gpFile);
		pID3DBlob_PixelShaderCode->Release();
		pID3DBlob_PixelShaderCode = NULL;
		pID3DBlob_VertexShaderCode->Release();
		pID3DBlob_VertexShaderCode = NULL;
		pID3DBlob_Error->Release();
		pID3DBlob_Error = NULL;
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreatePixelShader() Succeeded.\n");
		fclose(gpFile);
	}

	gpID3D11DeviceContext->PSSetShader(gpID3D11PixelShader, 0, 0);

	//Create Input Layout 
	//Layout For Position
	D3D11_INPUT_ELEMENT_DESC inputElementDesc[2];
	ZeroMemory(inputElementDesc, sizeof(D3D11_INPUT_ELEMENT_DESC));
	inputElementDesc[0].SemanticName = "POSITION";
	inputElementDesc[0].SemanticIndex = 0;
	inputElementDesc[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	inputElementDesc[0].InputSlot = 0;
	inputElementDesc[0].AlignedByteOffset = 0;
	inputElementDesc[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	inputElementDesc[0].InstanceDataStepRate = 0;

	//Layout For Color
	inputElementDesc[1].SemanticName = "NORMAL";
	inputElementDesc[1].SemanticIndex = 0;
	inputElementDesc[1].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	inputElementDesc[1].InputSlot = 1;
	inputElementDesc[1].AlignedByteOffset = 0;
	inputElementDesc[1].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	inputElementDesc[1].InstanceDataStepRate = 0;

	hr = gpID3D11Device->CreateInputLayout(inputElementDesc, 2, pID3DBlob_VertexShaderCode->GetBufferPointer(), pID3DBlob_VertexShaderCode->GetBufferSize(), &gpID3D11InputLayout);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateInputLayout() Failed.\n");
		fclose(gpFile);
		pID3DBlob_VertexShaderCode->Release();
		pID3DBlob_VertexShaderCode = NULL;
		pID3DBlob_PixelShaderCode->Release();
		pID3DBlob_PixelShaderCode = NULL;
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateInputLayout() Succeeded.\n");
		fclose(gpFile);
	}

	gpID3D11DeviceContext->IASetInputLayout(gpID3D11InputLayout);
	pID3DBlob_VertexShaderCode->Release();
	pID3DBlob_VertexShaderCode = NULL;
	pID3DBlob_PixelShaderCode->Release();
	pID3DBlob_PixelShaderCode = NULL;

	/************Sphere**********/
	Sphere sphere;

	sphere.getSphereVertexData(sphere_vertices, sphere_normals, sphere_textures, sphere_elements);
	gNumVertices = sphere.getNumberOfSphereVertices();
	gNumElements = sphere.getNumberOfSphereElements();

	D3D11_BUFFER_DESC bufferDesc;
	ZeroMemory(&bufferDesc, sizeof(D3D11_BUFFER_DESC));
	bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc.ByteWidth = sizeof(sphere_vertices);
	bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr = gpID3D11Device->CreateBuffer(&bufferDesc, NULL, &gpID3D11Buffer_VertexBuffer_Sphere);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Failed For Vertex Buffer Sphere.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Succeeded For Vertex Buffer Sphere.\n");
		fclose(gpFile);
	}

	D3D11_MAPPED_SUBRESOURCE mappedSubresource;
	ZeroMemory(&mappedSubresource, sizeof(D3D11_MAPPED_SUBRESOURCE));
	gpID3D11DeviceContext->Map(gpID3D11Buffer_VertexBuffer_Sphere, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource);
	memcpy(mappedSubresource.pData, sphere_vertices, sizeof(sphere_vertices));
	gpID3D11DeviceContext->Unmap(gpID3D11Buffer_VertexBuffer_Sphere, 0);

	/***********Normals Sphere***********/

	ZeroMemory(&bufferDesc, sizeof(D3D11_BUFFER_DESC));
	bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc.ByteWidth = sizeof(float)*ARRAYSIZE(sphere_normals);
	bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr = gpID3D11Device->CreateBuffer(&bufferDesc, 0, &gpID3D11Buffer_VertexBuffer_Sphere_Normal);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Failed For Vertex Buffer Sphere Normal.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Succeeded For Vertex Buffer Sphere Normal.\n");
		fclose(gpFile);
	}

	ZeroMemory(&mappedSubresource, sizeof(D3D11_MAPPED_SUBRESOURCE));
	gpID3D11DeviceContext->Map(gpID3D11Buffer_VertexBuffer_Sphere_Normal, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource);
	memcpy(mappedSubresource.pData, sphere_normals, sizeof(sphere_normals));
	gpID3D11DeviceContext->Unmap(gpID3D11Buffer_VertexBuffer_Sphere_Normal, 0);

	ZeroMemory(&bufferDesc, sizeof(D3D10_BUFFER_DESC));
	bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc.ByteWidth = sizeof(unsigned short)*ARRAYSIZE(sphere_elements);
	bufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
	bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr = gpID3D11Device->CreateBuffer(&bufferDesc, 0, &gpID3D11Buffer_VertexBuffer_Sphere_Elements);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Failed For Vertex Buffer Sphere Elements.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Succeeded For Vertex Buffer Sphere Elements.\n");
		fclose(gpFile);
	}

	ZeroMemory(&mappedSubresource, sizeof(D3D11_MAPPED_SUBRESOURCE));
	gpID3D11DeviceContext->Map(gpID3D11Buffer_VertexBuffer_Sphere_Elements, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource);
	memcpy(mappedSubresource.pData, sphere_elements, sizeof(sphere_elements));
	gpID3D11DeviceContext->Unmap(gpID3D11Buffer_VertexBuffer_Sphere_Elements, 0);

	//Define and Set Constant Buffer
	D3D11_BUFFER_DESC bufferDesc_ConstantBuffer;
	ZeroMemory(&bufferDesc_ConstantBuffer, sizeof(D3D11_BUFFER_DESC));
	bufferDesc_ConstantBuffer.Usage = D3D11_USAGE_DEFAULT;
	bufferDesc_ConstantBuffer.ByteWidth = sizeof(CBUFFER);
	bufferDesc_ConstantBuffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

	hr = gpID3D11Device->CreateBuffer(&bufferDesc_ConstantBuffer, nullptr, &gpID3D11Buffer_ConstantBuffer);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Failed For Constant Buffer.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Succeeded For Constant Buffer.\n");
		fclose(gpFile);
	}

	gpID3D11DeviceContext->VSSetConstantBuffers(0, 1, &gpID3D11Buffer_ConstantBuffer);

	gClearColor[0] = 0.0f;
	gClearColor[1] = 0.0f;
	gClearColor[2] = 0.0f;
	gClearColor[3] = 1.0f;

	gPerspectiveProjectionMatrix = XMMatrixIdentity();

	D3D11_RASTERIZER_DESC rasterizerDesc;
	ZeroMemory(&rasterizerDesc, sizeof(D3D11_RASTERIZER_DESC));
	rasterizerDesc.AntialiasedLineEnable = FALSE;
	rasterizerDesc.MultisampleEnable = FALSE;
	rasterizerDesc.DepthBias = 0;
	rasterizerDesc.DepthBiasClamp = 0.0f;
	rasterizerDesc.CullMode = D3D11_CULL_NONE;
	rasterizerDesc.DepthClipEnable = TRUE;
	rasterizerDesc.FillMode = D3D11_FILL_SOLID;
	rasterizerDesc.FrontCounterClockwise = FALSE;
	rasterizerDesc.ScissorEnable = FALSE;
	
	hr = gpID3D11Device->CreateRasterizerState(&rasterizerDesc, &gpID3D11RasterizerState);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "gpID3D11Device::CreateRasterizerState() Failed.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "gpID3D11Device::CreateRasterizerState() Succeeded.\n");
		fclose(gpFile);
	}

	gpID3D11DeviceContext->RSSetState(gpID3D11RasterizerState);

	hr = resize(WIN_WIDTH, WIN_HEIGHT);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "resize() Failed.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "resize() Succeeded.\n");
		fclose(gpFile);
	}
	
	return(S_OK);
}

HRESULT resize(int width, int height)
{
	HRESULT hr = S_OK;

	if (gpID3D11DepthStencilView)
	{
		gpID3D11DepthStencilView->Release();
		gpID3D11DepthStencilView = NULL;
	}

	//Free any size dependent resources
	if (gpID3D11RenderTargetView)
	{
		gpID3D11RenderTargetView->Release();
		gpID3D11RenderTargetView = NULL;
	}

	//Resize the Swap Chain Buffer
	gpIDXGISwapChain->ResizeBuffers(1, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, 0);

	//Depth Buffer 
	D3D11_TEXTURE2D_DESC textureDesc;
	ZeroMemory(&textureDesc, sizeof(D3D11_TEXTURE2D_DESC));
	textureDesc.Width = width;
	textureDesc.Height = height;
	//Array Size is 1 because we are using buffer for Depth not for texture
	textureDesc.ArraySize = 1;
	//Atleast 1 MipLevel as we are using buffer for Depth
	textureDesc.MipLevels = 1;
	//Number of Samples 
	textureDesc.SampleDesc.Count = 4;
	//Quality of Samples (1 is for highest)
	textureDesc.SampleDesc.Quality = 1;
	//Similar to pfd.cDepthBits
	textureDesc.Format = DXGI_FORMAT_D32_FLOAT;
	textureDesc.Usage = D3D11_USAGE_DEFAULT;
	textureDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	textureDesc.CPUAccessFlags = 0;
	textureDesc.MiscFlags = 0;

	//Create 2D Texture for Depth from above Structure
	ID3D11Texture2D *pID3D11Texture2D = NULL;

	//2nd parameter is Subresource Data
	hr = gpID3D11Device->CreateTexture2D(&textureDesc, 0, &pID3D11Texture2D);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "gpID3D11Device::CreateTexture2D() Failed.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateTexture2D() Succeeded.\n");
		fclose(gpFile);
	}

	D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;
	ZeroMemory(&depthStencilViewDesc, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));
	depthStencilViewDesc.Format = DXGI_FORMAT_D32_FLOAT;
	depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;
	hr = gpID3D11Device->CreateDepthStencilView(pID3D11Texture2D, &depthStencilViewDesc, &gpID3D11DepthStencilView);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "gpID3D11Device::CreateDepthStencilView() Failed.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateDepthStencilView() Succeeded.\n");
		fclose(gpFile);
	}

	pID3D11Texture2D->Release();
	pID3D11Texture2D = NULL;

	//Again Get the Back Buffer From Swap Chain
	ID3D11Texture2D *pID3D11Texture2D_BackBuffer;
	gpIDXGISwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pID3D11Texture2D_BackBuffer);

	//Again get the render target view from d3d11 device using above back buffer
	hr = gpID3D11Device->CreateRenderTargetView(pID3D11Texture2D_BackBuffer, NULL, &gpID3D11RenderTargetView);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "gpID3D11Device::CreateRenderTargetView() Failed.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateRenderTargetView() Succeeded.\n");
		fclose(gpFile);
	}
	pID3D11Texture2D_BackBuffer->Release();
	pID3D11Texture2D_BackBuffer = NULL;

	//Set Render Target View as Render Target
	gpID3D11DeviceContext->OMSetRenderTargets(1, &gpID3D11RenderTargetView, gpID3D11DepthStencilView);

	//Set viewport
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
	//Clear Depth
	gpID3D11DeviceContext->ClearDepthStencilView(gpID3D11DepthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0.0f);
	//Clear Render Target View to chosen color
	gpID3D11DeviceContext->ClearRenderTargetView(gpID3D11RenderTargetView, gClearColor);

	UINT stride = sizeof(float) * 3;
	UINT offset = 0;

	XMMATRIX worldMatrix = XMMatrixIdentity();
	XMMATRIX viewMatrix = XMMatrixIdentity();
	XMMATRIX translationMatrix = XMMatrixIdentity();
	XMMATRIX rotationMatrix_X = XMMatrixIdentity();
	XMMATRIX rotationMatrix_Y = XMMatrixIdentity();
	XMMATRIX rotationMatrix_Z = XMMatrixIdentity();
	XMMATRIX scaleMatrix = XMMatrixIdentity();
	XMMATRIX worldviewMatrix;

	//Load the data into constant buffer
	CBUFFER constantBuffer;

	/*************Sphere*************/
	gpID3D11DeviceContext->IASetVertexBuffers(0, 1, &gpID3D11Buffer_VertexBuffer_Sphere, &stride, &offset);
	gpID3D11DeviceContext->IASetVertexBuffers(1, 1, &gpID3D11Buffer_VertexBuffer_Sphere_Normal, &stride, &offset);
	gpID3D11DeviceContext->IASetIndexBuffer(gpID3D11Buffer_VertexBuffer_Sphere_Elements, DXGI_FORMAT_R16_UINT, offset);

	gpID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	translationMatrix = XMMatrixTranslation(0.0f, 0.0f, 6.0f);

	rotationMatrix_X = XMMatrixRotationX(XMConvertToRadians(gAngle));
	rotationMatrix_Y = XMMatrixRotationY(XMConvertToRadians(gAngle));
	rotationMatrix_Z = XMMatrixRotationZ(XMConvertToRadians(gAngle));

	scaleMatrix = XMMatrixScaling(0.75f, 0.75f, 0.75f);

	worldMatrix = scaleMatrix * rotationMatrix_X * rotationMatrix_Y * rotationMatrix_Z * translationMatrix * worldMatrix;

	worldviewMatrix = worldMatrix * viewMatrix;

	constantBuffer.WorldViewMatrix = worldviewMatrix;
	constantBuffer.ProjectionMatrx = gPerspectiveProjectionMatrix;
	constantBuffer.u_La = XMVectorSet(1.0f, 1.0f, 1.0f, 1.0f);
	constantBuffer.u_Ka = XMVectorSet(0.5f, 0.5f, 0.5f, 1.0f);
	constantBuffer.u_light_position = XMVectorSet(0.0f, 0.0f, -10.0f, 0.0f);
	if (gbLight == true)
		constantBuffer.u_lkeypressed = 1;
	else
		constantBuffer.u_lkeypressed = 0;

	gpID3D11DeviceContext->UpdateSubresource(gpID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//gpID3D11DeviceContext->Draw(gNumVertices, 0);
	gpID3D11DeviceContext->DrawIndexed(gNumElements, 0, 0);

	//Switch between Front & Back Buffers(Like glSwapBuffer())
	gpIDXGISwapChain->Present(0, 0);
}

void update(void)
{
	gAngle = gAngle + 0.2f;
	if (gAngle >= 360.0f)
		gAngle = gAngle - 360.0f;
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

	if (gpID3D11Buffer_VertexBuffer_Sphere)
	{
		gpID3D11Buffer_VertexBuffer_Sphere->Release();
		gpID3D11Buffer_VertexBuffer_Sphere = NULL;
	}

	if (gpID3D11Buffer_VertexBuffer_Sphere_Normal)
	{
		gpID3D11Buffer_VertexBuffer_Sphere_Normal->Release();
		gpID3D11Buffer_VertexBuffer_Sphere_Normal = NULL;
	}

	if (gpID3D11Buffer_VertexBuffer_Sphere_Elements)
	{
		gpID3D11Buffer_VertexBuffer_Sphere_Elements->Release();
		gpID3D11Buffer_VertexBuffer_Sphere_Elements = NULL;
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
		fprintf_s(gpFile, "uninitialize() Succeeded.\n");
		fprintf_s(gpFile, "Log File Is Successfully Closed.\n");
		fclose(gpFile);
	}
}


