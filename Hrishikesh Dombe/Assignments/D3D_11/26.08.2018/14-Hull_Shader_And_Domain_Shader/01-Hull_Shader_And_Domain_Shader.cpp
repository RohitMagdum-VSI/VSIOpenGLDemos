#include<Windows.h>
#include<stdio.h>

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

float gClearColor[4];
IDXGISwapChain *gpIDXGISwapChain = NULL;
ID3D11Device *gpID3D11Device = NULL;
ID3D11DeviceContext *gpID3D11DeviceContext = NULL;
ID3D11RenderTargetView *gpID3D11RenderTargetView = NULL;

ID3D11VertexShader *gpID3D11VertexShader = NULL;
ID3D11PixelShader *gpID3D11PixelShader = NULL;
ID3D11HullShader *gpID3D11HullShader = NULL;
ID3D11DomainShader *gpID3D11DomainShader = NULL;
ID3D11Buffer *gpID3D11Buffer_VertexBuffer = NULL;
ID3D11InputLayout *gpID3D11InputLayout = NULL;
ID3D11Buffer *gpID3D11Buffer_ConstantBuffer_Hull_Shader = NULL;
ID3D11Buffer *gpID3D11Buffer_ConstantBuffer_Domain_Shader = NULL;
ID3D11Buffer *gpID3D11Buffer_ConstantBuffer_Pixel_Shader = NULL;

struct CBUFFER_Hull_Shader
{
	XMVECTOR Hull_constant_function_param;
};

struct CBUFFER_Domain_Shader
{
	XMMATRIX WorldViewProjectionMatrix;
};

struct CBUFFER_Pixel_Shader
{
	XMVECTOR Line_Color;
};

XMMATRIX gPerspectiveProjectionMatrix;

unsigned int NumberOfLineSegments = 1;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	HRESULT initialize(void);
	void uninitialize(void);
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

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, TEXT("Direct3D 11 Perspective Triangle"), WS_OVERLAPPEDWINDOW, 100, 100, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);

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

		case VK_UP:
			NumberOfLineSegments++;
			if (NumberOfLineSegments >= 50)
				NumberOfLineSegments = 50;
			break;

		case VK_DOWN: 
			NumberOfLineSegments--;
			if (NumberOfLineSegments <= 0)
				NumberOfLineSegments = 1;
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
		"struct vertex_output" \
		"{" \
		"float4 position : POSITION;" \
		"};" \
		"vertex_output main(float2 pos : POSITION)" \
		"{" \
		"vertex_output output;" \
		"output.position = float4(pos,0.0,1.0);" \
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

	const char *hullShaderSourceCode =
		"cbuffer ConstantBuffer" \
		"{" \
		"float4 hull_Constant_Function_Param;" \
		"}" \
		"struct vertex_output" \
		"{" \
		"float4 position : POSITION;" \
		"};" \
		"struct hull_constant_output" \
		"{" \
		"float edges[2] : SV_TESSFACTOR;" \
		"};" \
		"hull_constant_output hull_constant_function(void)" \
		"{" \
		"hull_constant_output output;" \
		"float numofstrips = hull_Constant_Function_Param[0];" \
		"float numofsegments = hull_Constant_Function_Param[1];" \
		"output.edges[0] = numofstrips;" \
		"output.edges[1] = numofsegments;" \
		"return(output);" \
		"}" \
		"struct hull_output" \
		"{" \
		"float4 position : POSITION;" \
		"};" \
		"[domain (\"isoline\")]" \
		"[partitioning(\"integer\")]" \
		"[outputtopology(\"line\")]" \
		"[outputcontrolpoints(4)]" \
		"[patchconstantfunc(\"hull_constant_function\")]" \
		"hull_output main(InputPatch<vertex_output,4> inputPatch, uint i : SV_OUTPUTCONTROLPOINTID)" \
		"{" \
		"hull_output output;" \
		"output.position = inputPatch[i].position;" \
		"return(output);" \
		"}";
		

	ID3DBlob *pID3DBlob_HullShaderCode = NULL;
	pID3DBlob_Error = NULL;

	//lstrlenA return count without counting '/0' so we do +1
	//D3D_COMPILE_STANDARD_FILE_INCLUDE here we tell include the file that you think are important in Shader or we can also sepcify the files
	//vs_5_0 we tell Feature Level Of Shader
	//8th parameter Compile Constants
	//9th parameter Shader Constants
	//10th parameter is returned and contains Compiled Byte Code 
	hr = D3DCompile(hullShaderSourceCode, lstrlenA(hullShaderSourceCode) + 1, "HS", NULL, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "hs_5_0", 0, 0, &pID3DBlob_HullShaderCode, &pID3DBlob_Error);

	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf_s(gpFile, "D3DCompile() Failed For Hull Shader : %s \n", (char *)pID3DBlob_Error->GetBufferPointer());
			fclose(gpFile);
			pID3DBlob_VertexShaderCode->Release();
			pID3DBlob_VertexShaderCode = NULL;
			pID3DBlob_HullShaderCode->Release();
			pID3DBlob_HullShaderCode = NULL;
			pID3DBlob_Error->Release();
			pID3DBlob_Error = NULL;
			return(hr);
		}
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "D3DCompile() Succeeded For Hull Shader. \n");
		fclose(gpFile);
	}

	//3rd Paramter is pointer to variables that should be used across the shader
	hr = gpID3D11Device->CreateHullShader(pID3DBlob_HullShaderCode->GetBufferPointer(), pID3DBlob_HullShaderCode->GetBufferSize(), NULL, &gpID3D11HullShader);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "gpID3D11Device::CreateHullShader() Failed. \n");
		fclose(gpFile);
		pID3DBlob_VertexShaderCode->Release();
		pID3DBlob_VertexShaderCode = NULL;
		pID3DBlob_HullShaderCode->Release();
		pID3DBlob_HullShaderCode = NULL;
		pID3DBlob_Error->Release();
		pID3DBlob_Error = NULL;
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "gpID3D11Device::CreateHullShader() Succeeded. \n");
		fclose(gpFile);
	}

	//2nd Parameter is Linkage Object Pointer Array
	//3rd Parameter is Index Of The Object Array
	gpID3D11DeviceContext->HSSetShader(gpID3D11HullShader, 0, 0);

	const char *domainShaderSourceCode =
		"cbuffer ConstantBuffer" \
		"{" \
		"float4x4 worldViewProjectionMatrix;" \
		"}" \
		"struct hull_output" \
		"{" \
		"float4 position : POSITION;" \
		"};" \
		"struct domain_output" \
		"{" \
		"float4 position : SV_POSITION;" \
		"};" \
		"struct hull_constant_output" \
		"{" \
		"float edges[2] : SV_TESSFACTOR;" \
		"};" \
		"[domain (\"isoline\")]" \
		"domain_output main(hull_constant_output input, OutputPatch<hull_output,4> output_patch, float2 tessCoord : SV_DOMAINLOCATION)" \
		"{" \
		"domain_output output;" \
		"float u = tessCoord.x;" \
		"float3 p0 = output_patch[0].position.xyz;"
		"float3 p1 = output_patch[1].position.xyz;"
		"float3 p2 = output_patch[2].position.xyz;"
		"float3 p3 = output_patch[3].position.xyz;"
		"float u1 = (1.0 - u);" \
		"float u2 = mul(u,u);" \
		"float b3 = mul(u2,u);" \
		"float b2 = mul(3.0,mul(u2,u1));" \
		"float b1 = mul(3.0,mul(u,mul(u1,u1)));" \
		"float b0 = mul(u1,mul(u1,u1));" \
		"float3 p = mul(p0,b0) + mul(p1,b1) + mul(p2,b2) + mul(p3,b3);" \
		"output.position = mul(worldViewProjectionMatrix,float4(p,1.0));" \
		"return(output);" \
		"}";

	ID3DBlob *pID3DBlob_DomainShaderCode = NULL;
	pID3DBlob_Error = NULL;

	//lstrlenA return count without counting '/0' so we do +1
	//D3D_COMPILE_STANDARD_FILE_INCLUDE here we tell include the file that you think are important in Shader or we can also sepcify the files
	//vs_5_0 we tell Feature Level Of Shader
	//8th parameter Compile Constants
	//9th parameter Shader Constants
	//10th parameter is returned and contains Compiled Byte Code 
	hr = D3DCompile(domainShaderSourceCode, lstrlenA(domainShaderSourceCode) + 1, "DS", NULL, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "ds_5_0", 0, 0, &pID3DBlob_DomainShaderCode, &pID3DBlob_Error);

	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf_s(gpFile, "D3DCompile() Failed For Hull Shader : %s \n", (char *)pID3DBlob_Error->GetBufferPointer());
			fclose(gpFile);
			pID3DBlob_VertexShaderCode->Release();
			pID3DBlob_VertexShaderCode = NULL;
			pID3DBlob_HullShaderCode->Release();
			pID3DBlob_HullShaderCode = NULL;
			pID3DBlob_DomainShaderCode->Release();
			pID3DBlob_DomainShaderCode = NULL;
			pID3DBlob_Error->Release();
			pID3DBlob_Error = NULL;
			return(hr);
		}
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "D3DCompile() Succeeded For Domain Shader. \n");
		fclose(gpFile);
	}

	//3rd Paramter is pointer to variables that should be used across the shader
	hr = gpID3D11Device->CreateDomainShader(pID3DBlob_DomainShaderCode->GetBufferPointer(), pID3DBlob_DomainShaderCode->GetBufferSize(), NULL, &gpID3D11DomainShader);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "gpID3D11Device::CreateDomainShader() Failed. \n");
		fclose(gpFile);
		pID3DBlob_VertexShaderCode->Release();
		pID3DBlob_VertexShaderCode = NULL;
		pID3DBlob_HullShaderCode->Release();
		pID3DBlob_HullShaderCode = NULL;
		pID3DBlob_DomainShaderCode->Release();
		pID3DBlob_DomainShaderCode = NULL;
		pID3DBlob_Error->Release();
		pID3DBlob_Error = NULL;
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "gpID3D11Device::CreateDomainShader() Succeeded. \n");
		fclose(gpFile);
	}

	//2nd Parameter is Linkage Object Pointer Array
	//3rd Parameter is Index Of The Object Array
	gpID3D11DeviceContext->DSSetShader(gpID3D11DomainShader, 0, 0);

	const char *pixelShaderSourceCode =
		"cbuffer ConstantBuffer" \
		"{" \
		"float4 lineColor : COLOR;" \
		"}" \
		"float4 main(float4 pos : SV_POSITION) : SV_TARGET" \
		"{" \
		"return(lineColor);" \
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
	D3D11_INPUT_ELEMENT_DESC inputElementDesc;
	ZeroMemory(&inputElementDesc, sizeof(D3D11_INPUT_ELEMENT_DESC));
	inputElementDesc.SemanticName = "POSITION";
	inputElementDesc.SemanticIndex = 0;
	inputElementDesc.Format = DXGI_FORMAT_R32G32_FLOAT;
	inputElementDesc.InputSlot = 0;
	inputElementDesc.AlignedByteOffset = 0;
	inputElementDesc.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	inputElementDesc.InstanceDataStepRate = 0;

	hr = gpID3D11Device->CreateInputLayout(&inputElementDesc, 1, pID3DBlob_VertexShaderCode->GetBufferPointer(), pID3DBlob_VertexShaderCode->GetBufferSize(), &gpID3D11InputLayout);
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

	//Left Hand Thumb Rule
	float vertices[] =
	{
		-1.0f,-1.0f,
		-0.5f,1.0f,
		0.5f,-1.0f,
		1.0f,1.0f
	};

	D3D11_BUFFER_DESC bufferDesc;
	ZeroMemory(&bufferDesc, sizeof(D3D11_BUFFER_DESC));
	bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc.ByteWidth = sizeof(float) * ARRAYSIZE(vertices);
	bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr = gpID3D11Device->CreateBuffer(&bufferDesc, NULL, &gpID3D11Buffer_VertexBuffer);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Failed For Vertex Buffer.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Succeeded For Vertex Buffer.\n");
		fclose(gpFile);
	}

	//Copy Vertices into the buffer
	D3D11_MAPPED_SUBRESOURCE mappedSubresource;
	ZeroMemory(&mappedSubresource, sizeof(D3D11_MAPPED_SUBRESOURCE));
	gpID3D11DeviceContext->Map(gpID3D11Buffer_VertexBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource);
	memcpy(mappedSubresource.pData, vertices, sizeof(vertices));
	gpID3D11DeviceContext->Unmap(gpID3D11Buffer_VertexBuffer, 0);

	//Define and Set Constant Buffer
	D3D11_BUFFER_DESC bufferDesc_ConstantBuffer;
	ZeroMemory(&bufferDesc_ConstantBuffer, sizeof(D3D11_BUFFER_DESC));
	bufferDesc_ConstantBuffer.Usage = D3D11_USAGE_DEFAULT;
	bufferDesc_ConstantBuffer.ByteWidth = sizeof(CBUFFER_Hull_Shader);
	bufferDesc_ConstantBuffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

	hr = gpID3D11Device->CreateBuffer(&bufferDesc_ConstantBuffer, nullptr, &gpID3D11Buffer_ConstantBuffer_Hull_Shader);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Failed For Constant Buffer For Hull Shader.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Succeeded For Constant Buffer For Hull Shader.\n");
		fclose(gpFile);
	}

	gpID3D11DeviceContext->HSSetConstantBuffers(0, 1, &gpID3D11Buffer_ConstantBuffer_Hull_Shader);

	ZeroMemory(&bufferDesc_ConstantBuffer, sizeof(D3D11_BUFFER_DESC));
	bufferDesc_ConstantBuffer.Usage = D3D11_USAGE_DEFAULT;
	bufferDesc_ConstantBuffer.ByteWidth = sizeof(CBUFFER_Domain_Shader);
	bufferDesc_ConstantBuffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

	hr = gpID3D11Device->CreateBuffer(&bufferDesc_ConstantBuffer, nullptr, &gpID3D11Buffer_ConstantBuffer_Domain_Shader);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Failed For Constant Buffer Domain Shader.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Succeeded For Constant Buffer Domain Shader.\n");
		fclose(gpFile);
	}

	gpID3D11DeviceContext->DSSetConstantBuffers(0, 1, &gpID3D11Buffer_ConstantBuffer_Domain_Shader);

	ZeroMemory(&bufferDesc_ConstantBuffer, sizeof(D3D11_BUFFER_DESC));
	bufferDesc_ConstantBuffer.Usage = D3D11_USAGE_DEFAULT;
	bufferDesc_ConstantBuffer.ByteWidth = sizeof(CBUFFER_Pixel_Shader);
	bufferDesc_ConstantBuffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

	hr = gpID3D11Device->CreateBuffer(&bufferDesc_ConstantBuffer, nullptr, &gpID3D11Buffer_ConstantBuffer_Pixel_Shader);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Failed For Constant Buffer Pixel Shader.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Succeeded For Constant Buffer Pixel Shader.\n");
		fclose(gpFile);
	}

	gpID3D11DeviceContext->PSSetConstantBuffers(0, 1, &gpID3D11Buffer_ConstantBuffer_Pixel_Shader);

	gClearColor[0] = 0.0f;
	gClearColor[1] = 0.0f;
	gClearColor[2] = 0.0f;
	gClearColor[3] = 1.0f;

	gPerspectiveProjectionMatrix = XMMatrixIdentity();

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

	//Free any size dependent resources
	if (gpID3D11RenderTargetView)
	{
		gpID3D11RenderTargetView->Release();
		gpID3D11RenderTargetView = NULL;
	}

	//Resize the Swap Chain Buffer
	gpIDXGISwapChain->ResizeBuffers(1, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, 0);

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
	gpID3D11DeviceContext->OMSetRenderTargets(1, &gpID3D11RenderTargetView, NULL);

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
	//Clear Render Target View to chosen color
	gpID3D11DeviceContext->ClearRenderTargetView(gpID3D11RenderTargetView, gClearColor);

	UINT stride = sizeof(float) * 2;
	UINT offset = 0;

	//IASetVertexBuffers like glVertexAttribPointer
	gpID3D11DeviceContext->IASetVertexBuffers(0, 1, &gpID3D11Buffer_VertexBuffer, &stride, &offset);

	gpID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_4_CONTROL_POINT_PATCHLIST);

	XMMATRIX worldMatrix = XMMatrixIdentity();
	XMMATRIX viewMatrix = XMMatrixIdentity();

	worldMatrix = XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	XMMATRIX wvpMatrix = worldMatrix * viewMatrix * gPerspectiveProjectionMatrix;

	//Load the data into constant buffer
	CBUFFER_Hull_Shader constantBuffer_Hull_Shader;
	constantBuffer_Hull_Shader.Hull_constant_function_param = XMVectorSet(1,NumberOfLineSegments,0.0,0.0);

	gpID3D11DeviceContext->UpdateSubresource(gpID3D11Buffer_ConstantBuffer_Hull_Shader, 0, NULL, &constantBuffer_Hull_Shader, 0, 0);

	CBUFFER_Domain_Shader constantBuffer_Domain_Shader;
	constantBuffer_Domain_Shader.WorldViewProjectionMatrix = wvpMatrix;

	gpID3D11DeviceContext->UpdateSubresource(gpID3D11Buffer_ConstantBuffer_Domain_Shader, 0, NULL, &constantBuffer_Domain_Shader, 0, 0);

	CBUFFER_Pixel_Shader constantBuffer_Pixel_Shader;
	constantBuffer_Pixel_Shader.Line_Color = XMVectorSet(0.0f,0.0f,1.0,1.0f);

	gpID3D11DeviceContext->UpdateSubresource(gpID3D11Buffer_ConstantBuffer_Pixel_Shader, 0, NULL, &constantBuffer_Pixel_Shader, 0, 0);


	//Draw vertex buffer to render target
	gpID3D11DeviceContext->Draw(4, 0);

	//Switch between Front & Back Buffers(Like glSwapBuffer())
	gpIDXGISwapChain->Present(0, 0);
}

void uninitialize(void)
{
	if(gpID3D11Buffer_ConstantBuffer_Domain_Shader)
	{
		gpID3D11Buffer_ConstantBuffer_Domain_Shader->Release();
		gpID3D11Buffer_ConstantBuffer_Domain_Shader = NULL;
	}

	if (gpID3D11Buffer_ConstantBuffer_Hull_Shader)
	{
		gpID3D11Buffer_ConstantBuffer_Hull_Shader->Release();
		gpID3D11Buffer_ConstantBuffer_Hull_Shader = NULL;
	}

	if (gpID3D11Buffer_ConstantBuffer_Pixel_Shader)
	{
		gpID3D11Buffer_ConstantBuffer_Pixel_Shader->Release();
		gpID3D11Buffer_ConstantBuffer_Pixel_Shader = NULL;
	}

	if (gpID3D11InputLayout)
	{
		gpID3D11InputLayout->Release();
		gpID3D11InputLayout = NULL;
	}

	if (gpID3D11Buffer_VertexBuffer)
	{
		gpID3D11Buffer_VertexBuffer->Release();
		gpID3D11Buffer_VertexBuffer = NULL;
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