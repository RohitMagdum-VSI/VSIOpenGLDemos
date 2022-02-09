#if 1
#include <Windows.h>
#include <stdio.h>

#include <d3d11.h>
#include <d3dcompiler.h>

#pragma warning(disable: 4838) // Suppress XNAMATH warning typecast of unsigned int to int
#include "..\..\include\XNAMath\xnamath.h"
#include "..\..\include\Texture\WICTextureLoader.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "D3dcompiler.lib")
#pragma comment(lib, "DirectXTK.lib")	//	For DirectX::CreateWICTextureFromFile().

#define WIN_WIDTH	800
#define WIN_HEIGHT	600

#define IMAGE_WIDTH		64
#define IMAGE_HEIGHT	64

BYTE g_arrWhiteImage[IMAGE_HEIGHT][IMAGE_WIDTH][4];
int g_iDigit = 0x31;

#define WINDOW_NAME		L"D3D-Tweak Smiley"
#define LOG_FILE		"log.txt"

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

//
//	Global variable.
//
FILE *g_pFile = NULL;

HWND g_hWnd = NULL;

DWORD g_dwStyle;
WINDOWPLACEMENT g_WindowPlacementPrev = { sizeof(WINDOWPLACEMENT) };

bool g_boFullScreen = false;
bool g_boActiveWindow = false;
bool g_boEscapeKeyPressed = false;

FLOAT g_fAngleTri = 0.0f;
FLOAT g_fAngleRect = 0.0f;

enum RTR_INPUT_SLOT
{
	RTR_INPUT_SLOT_POSITION = 0,
	RTR_INPUT_SLOT_COLOR,
	RTR_INPUT_SLOT_TEXTURE = RTR_INPUT_SLOT_COLOR,
	RTR_INPUT_SLOT_NORMAL,
};

float g_fClearColor[4];	//	RGBA
IDXGISwapChain *g_pIDXGISwapChain = NULL;	//	DXGI - DirectX Graphics Interface
ID3D11Device *g_pID3D11Device = NULL;
ID3D11DeviceContext *g_pID3D11DeviceContext = NULL;
ID3D11RenderTargetView *g_pID3D11RenderTargetView = NULL;
ID3D11DepthStencilView *g_pID3D11DepthStencilView = NULL;

ID3D11VertexShader *g_pID3D11VertexShader = NULL;	//	Vertex shader object
ID3D11PixelShader *g_pID3D11PixelShader = NULL;		//	same as fragment shader in OpenGL
ID3D11Buffer *g_pID3D11Buffer_VertexBufferCubePosition = NULL;	//	vbo_position in openGL
ID3D11Buffer *g_pID3D11Buffer_VertexBufferCubeTexture = NULL;	//	vbo_position in openGL
ID3D11InputLayout *g_pID3D11InputLayout = NULL;
ID3D11Buffer *g_pID3D11Buffer_ConstantBuffer = NULL;
ID3D11RasterizerState *g_pID3D11RasterizerState = NULL;

ID3D11ShaderResourceView *g_pID3D11ShaderResourceView_TextureCube = NULL;
ID3D11SamplerState *g_pID3D11SamplerState_TextureCube = NULL;


struct CBUFFER
{
	XMMATRIX WorldViewProjectionMatrix;
};

XMMATRIX g_PerspectiveProjectionMatrix;

void log_write(const char* msg)
{
	int iErrNo;

	iErrNo = fopen_s(&g_pFile, LOG_FILE, "a+");
	if (0 == iErrNo)
	{
		fprintf_s(g_pFile, msg);
		fclose(g_pFile);
	}
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	//	Function declarations
	HRESULT initialize();
	VOID display();
	VOID uninitialize();
	void update();

	MSG Msg;
	HWND hWnd;
	int iErrNo;
	int iMaxWidth;
	int iMaxHeight;
	WNDCLASSEX WndClass;
	bool boDone = false;

	iErrNo = fopen_s(&g_pFile, LOG_FILE, "w");
	if (0 != iErrNo)
	{
		MessageBox(NULL, L"Log file can not be created exiting", L"ERROR", MB_OK | MB_TOPMOST | MB_ICONSTOP);
		exit(0);
	}
	else
	{
		fprintf_s(g_pFile, "Log file is succesfuly opened \n");
		fclose(g_pFile);
	}

	//
	//	Initialize members of window class.
	//
	WndClass.cbSize = sizeof(WNDCLASSEX);
	WndClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;		//	Change:Added CS_OWNDC.
	WndClass.cbClsExtra = 0;
	WndClass.cbWndExtra = 0;
	WndClass.hInstance = hInstance;
	WndClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	WndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	WndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	WndClass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	WndClass.lpfnWndProc = WndProc;
	WndClass.lpszClassName = WINDOW_NAME;
	WndClass.lpszMenuName = NULL;

	//
	//	Register class.
	//
	RegisterClassEx(&WndClass);

	iMaxWidth = GetSystemMetrics(SM_CXFULLSCREEN);
	iMaxHeight = GetSystemMetrics(SM_CYFULLSCREEN);

	//
	//	Create Window.
	//
	hWnd = CreateWindowEx(
		WS_EX_APPWINDOW,	//	Change: New member get added for CreateWindowEx API.
		WINDOW_NAME,
		WINDOW_NAME,
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,		//	Change: Added styles -WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE
		(iMaxWidth - WIN_WIDTH) / 2,
		(iMaxHeight - WIN_HEIGHT) / 2,
		WIN_WIDTH,
		WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL
	);
	if (NULL == hWnd)
	{
		log_write("CreateWindowEx failed");

		return 0;
	}

	g_hWnd = hWnd;

	HRESULT hr;
	hr = initialize();
	if (FAILED(hr))
	{
		log_write("initialize failed");
		DestroyWindow(hWnd);
		hWnd = NULL;
	}

	ShowWindow(hWnd, SW_SHOW);
	SetForegroundWindow(hWnd);
	SetFocus(hWnd);

	//
	//	Message loop.
	//
	while (false == boDone)
	{
		if (PeekMessage(&Msg, NULL, 0, 0, PM_REMOVE))
		{
			if (WM_QUIT == Msg.message)
			{
				boDone = true;
			}
			else
			{
				TranslateMessage(&Msg);
				DispatchMessage(&Msg);
			}
		}
		else
		{
			if (true == g_boActiveWindow)
			{
				if (true == g_boEscapeKeyPressed)
				{
					boDone = true;
				}
				display();
				update();
			}
		}
	}

	uninitialize();

	return((int)Msg.wParam);
}


LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	VOID ToggleFullScreen();
	HRESULT resize(int iWidth, int iHeight);

	HRESULT hr;

	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (0 == HIWORD(wParam))
		{
			g_boActiveWindow = true;
		}
		else
		{
			g_boActiveWindow = false;
		}
		break;


		//case WM_ERASEBKGND:
		//return(0);

	case WM_SIZE:
		//
		//	If we have valid context then only call resize() otherwise program will crash
		if (g_pID3D11DeviceContext)
		{
			hr = resize(LOWORD(lParam), HIWORD(lParam));
			if (FAILED(hr))
			{
				log_write("resize() failed.\n");
			}
			else
			{
				log_write("resize() succeded.\n");
			}
		}
		break;

	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			g_boEscapeKeyPressed = true;
			break;

		case 'f':
		case 'F':
			if (false == g_boFullScreen)
			{
				ToggleFullScreen();
				g_boFullScreen = true;
			}
			else
			{
				ToggleFullScreen();
				g_boFullScreen = false;
			}
			break;

		case 0x31:
		case 0x32:
		case 0x33:
		case 0x34:
		case 0x61:
		case 0x62:
		case 0x63:
		case 0x64:
			g_iDigit = (int)wParam;
			break;

		default:
			break;
		}
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	default:
		break;
	}

	return (DefWindowProc(hWnd, iMsg, wParam, lParam));
}


VOID ToggleFullScreen()
{
	MONITORINFO MonitorInfo;

	if (false == g_boFullScreen)
	{
		g_dwStyle = GetWindowLong(g_hWnd, GWL_STYLE);

		if (g_dwStyle & WS_OVERLAPPEDWINDOW)
		{
			MonitorInfo = { sizeof(MonitorInfo) };

			if (GetWindowPlacement(g_hWnd, &g_WindowPlacementPrev) && GetMonitorInfo(MonitorFromWindow(g_hWnd, MONITORINFOF_PRIMARY), &MonitorInfo))
			{
				SetWindowLong(g_hWnd, GWL_STYLE, g_dwStyle & (~WS_OVERLAPPEDWINDOW));
				SetWindowPos(
					g_hWnd,
					HWND_TOP,
					MonitorInfo.rcMonitor.left,
					MonitorInfo.rcMonitor.top,
					MonitorInfo.rcMonitor.right - MonitorInfo.rcMonitor.left,
					MonitorInfo.rcMonitor.bottom - MonitorInfo.rcMonitor.top,
					SWP_NOZORDER | SWP_FRAMECHANGED
				);
			}
		}
		ShowCursor(FALSE);
	}
	else
	{
		SetWindowLong(g_hWnd, GWL_STYLE, g_dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(g_hWnd, &g_WindowPlacementPrev);
		SetWindowPos(g_hWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);

		ShowCursor(TRUE);
	}
}

HRESULT initialize()
{
	VOID uninitialize();
	HRESULT resize(int iWidth, int iHeight);
	HRESULT LoadD3DTexture(const WCHAR *pwszTextureName, ID3D11ShaderResourceView **ppID3D11ShaderResourceView);

	HRESULT hr;
	D3D_DRIVER_TYPE D3DDriverType;
	//
	//	Order of driver types are very importent.
	//	This order is nothing but the priority of required drivers.
	//	Hardware - 1st priority , warp - 2nd 
	//	D3D_DRIVER_TYPE_WARP(Windows advance rasterization platform) - From Win8(DirectX10) and above, software rendering is nothing but WARP.
	//	From Win8(DirectX10) and above, D3D become part of OS with the help of (CPU extensions)SSE 3,4,4.1. (SSE - Streaming SIMD extensions , SIMD - Single instruction multiple data).
	//
	D3D_DRIVER_TYPE arrD3DDriverTypes[] = { D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP, D3D_DRIVER_TYPE_REFERENCE };

	D3D_FEATURE_LEVEL D3DFeatureLevel_required = D3D_FEATURE_LEVEL_11_0;
	D3D_FEATURE_LEVEL D3DFeatureLevel_acquired = D3D_FEATURE_LEVEL_10_0;	//	default lowest

	UINT uiCreateDeviceFlags = 0;
	UINT uiNumDriverTypes = 0;
	UINT uiNumFeatureLevels = 1;	//	Based upon D3DFeatureLevel_required

	uiNumDriverTypes = sizeof(arrD3DDriverTypes) / sizeof(arrD3DDriverTypes[0]);

	DXGI_SWAP_CHAIN_DESC DXGISwapChainDesc;

	ZeroMemory(&DXGISwapChainDesc, sizeof(DXGISwapChainDesc));
	//
	//	Here we create back buffer and after that Direct3D will provide front buffer with the having same size as back buffer.
	//
	DXGISwapChainDesc.BufferCount = 1;
	DXGISwapChainDesc.BufferDesc.Width = WIN_WIDTH;
	DXGISwapChainDesc.BufferDesc.Height = WIN_HEIGHT;
	DXGISwapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	DXGISwapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
	DXGISwapChainDesc.BufferDesc.RefreshRate.Denominator = 1;

	DXGISwapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	DXGISwapChainDesc.OutputWindow = g_hWnd;
	DXGISwapChainDesc.SampleDesc.Count = 1;		//	Can be 1 to 4
	DXGISwapChainDesc.SampleDesc.Quality = 0;	// Both count and quality - MASA(Multi sampling anti alising)
	DXGISwapChainDesc.Windowed = TRUE;

	for (UINT uiDriverTypeIndex = 0; uiDriverTypeIndex < uiNumDriverTypes; uiDriverTypeIndex++)
	{
		D3DDriverType = arrD3DDriverTypes[uiDriverTypeIndex];

		hr = D3D11CreateDeviceAndSwapChain(
			NULL,
			D3DDriverType,
			NULL,
			uiCreateDeviceFlags,
			&D3DFeatureLevel_required,
			uiNumFeatureLevels,
			D3D11_SDK_VERSION,
			&DXGISwapChainDesc,
			&g_pIDXGISwapChain,
			&g_pID3D11Device,
			&D3DFeatureLevel_acquired,
			&g_pID3D11DeviceContext
		);
		if (SUCCEEDED(hr))
		{
			break;
		}
	}
	if (FAILED(hr))
	{
		log_write("D3D11CreateDeviceAndSwapChain() failed \n");
		return hr;
	}
	else
	{
		log_write("D3D11CreateDeviceAndSwapChain() Succeded \n");
		log_write("The choosen driver is of ");
		if (D3D_DRIVER_TYPE_HARDWARE == D3DDriverType)
		{
			log_write("D3D_DRIVER_TYPE_HARDWARE Type \n");
		}
		else if (D3D_DRIVER_TYPE_WARP == D3DDriverType)
		{
			log_write("D3D_DRIVER_TYPE_WARP Type \n");
		}
		else if (D3D_DRIVER_TYPE_REFERENCE == D3DDriverType)
		{
			log_write("D3D_DRIVER_TYPE_REFERENCE Type \n");
		}
		else
		{
			log_write("Unknown Type \n");
		}

		log_write("Supported highest feature is ");
		if (D3D_FEATURE_LEVEL_11_0 == D3DFeatureLevel_acquired)
		{
			log_write("D3D_FEATURE_LEVEL_11_0 \n");
		}
		else if (D3D_FEATURE_LEVEL_10_1 == D3DFeatureLevel_acquired)
		{
			log_write("D3D_FEATURE_LEVEL_10_1 \n");
		}
		else if (D3D_FEATURE_LEVEL_10_0 == D3DFeatureLevel_acquired)
		{
			log_write("D3D_FEATURE_LEVEL_10_0 \n");
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////
	//+	Initialize Shaders, input layouts, constant buffer

	/////////////////////////////////////////////////////////////////////////////////////////////
	//+	Vertex Shader

	const char *vertexShaderSourceCode =
		/*This constant buffer is same as uniform in OpenGL
		struct of cbuffer in the shader must match struct of the CBUFFER in program
		*/
		"cbuffer ConstantBuffer"		\
		"{"								\
		"float4x4 worldViewProjectionMatrix;"								\
		"}"								\
		"struct vertex_output"		\
		"{"								\
		"float4 Position:SV_POSITION;"								\
		"float2 texcoord:TEXCOORD;"								\
		"};"								\
		/*Here POSITION is like vPosition in OpenGL*/
		"vertex_output main(float4 pos:POSITION, float2 texcoord:TEXCOORD)	"/* : SV_POSITION"*/				\
		"{"								\
		"vertex_output output;"								\
		"output.Position = mul(worldViewProjectionMatrix, pos);"								\
		"output.texcoord = texcoord;"								\
		"return output;"								\
		"}";

	ID3DBlob *pID3DBlob_VertexShaderByteCode = NULL;
	ID3DBlob *pID3DBlob_Error = NULL;

	hr = D3DCompile(
		vertexShaderSourceCode,
		lstrlenA(vertexShaderSourceCode) + 1,	//	+ 1 for null character.
		"VS",
		NULL,	//	macros in shader program, currently we are not using any macro in program.
		D3D_COMPILE_STANDARD_FILE_INCLUDE,
		"main",	//	Entry point function in shader
		"vs_5_0",	//	Feature level
		0,	//	Compiler constants
		0, //	Effect constant
		&pID3DBlob_VertexShaderByteCode,
		&pID3DBlob_Error
	);
	if (FAILED(hr))
	{
		fopen_s(&g_pFile, LOG_FILE, "a+");
		fprintf_s(g_pFile, "D3DCompile failed for Vertex Shader.");
		if (NULL != pID3DBlob_Error)
		{
			fprintf_s(g_pFile, "Error %s.\n", (char*)pID3DBlob_Error->GetBufferPointer());
			pID3DBlob_Error->Release();
			pID3DBlob_Error = NULL;
		}
		fclose(g_pFile);

		return hr;
	}

	//	Note: Functions starts with "Create" mostly called on D3DDevice and 
	//		Functions starts with "Pipeline initial" called on Device context e.g ) VSSetShader().

	hr = g_pID3D11Device->CreateVertexShader(
		pID3DBlob_VertexShaderByteCode->GetBufferPointer(),
		pID3DBlob_VertexShaderByteCode->GetBufferSize(),
		NULL,	// To pass the data across the shaders
		&g_pID3D11VertexShader
	);
	if (FAILED(hr))
	{
		log_write("CreateVertexShader() failed.\n");
		//	Cleanup local only, global will be clean in uninitialize().
		pID3DBlob_VertexShaderByteCode->Release();
		pID3DBlob_VertexShaderByteCode = NULL;
		return hr;
	}

	g_pID3D11DeviceContext->VSSetShader(g_pID3D11VertexShader, 0, 0);

	//-	Vertex Shader
	/////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////
	//+	Pixel Shaders
	const char *pixelShaderSourceCode =
		"Texture2D myTexture2D;"		\
		"SamplerState mySamplerState;"		\
		"float4 main(float4 pos:SV_POSITION, float2 texcoord:TEXCOORD):SV_TARGET"		\
		"{"		\
		"float4 output_color;"		\
		/*myTexture2D.Sample ==  texture() in opengl*/
		"output_color = myTexture2D.Sample(mySamplerState, texcoord);"		\
		"return output_color;"		\
		"}";

	ID3DBlob *pID3DBlob_PixelShaderByteCode = NULL;
	pID3DBlob_Error = NULL;

	hr = D3DCompile(
		pixelShaderSourceCode,
		lstrlenA(pixelShaderSourceCode) + 1,
		"PS",
		NULL,
		D3D_COMPILE_STANDARD_FILE_INCLUDE,
		"main",
		"ps_5_0",
		0,
		0,
		&pID3DBlob_PixelShaderByteCode,
		&pID3DBlob_Error
	);
	if (FAILED(hr))
	{
		fopen_s(&g_pFile, LOG_FILE, "a+");
		fprintf_s(g_pFile, "D3DCompile failed for Pixel Shader.");
		if (NULL != pID3DBlob_Error)
		{
			fprintf_s(g_pFile, "Error %s.\n", (char*)pID3DBlob_Error->GetBufferPointer());
			pID3DBlob_Error->Release();
			pID3DBlob_Error = NULL;
		}
		pID3DBlob_VertexShaderByteCode->Release();
		pID3DBlob_VertexShaderByteCode = NULL;
		fclose(g_pFile);

		return hr;
	}

	hr = g_pID3D11Device->CreatePixelShader(
		pID3DBlob_PixelShaderByteCode->GetBufferPointer(),
		pID3DBlob_PixelShaderByteCode->GetBufferSize(),
		NULL,
		&g_pID3D11PixelShader
	);
	if (FAILED(hr))
	{
		log_write("CreatePixelShader() failed.\n");
		//	Cleanup local only, global will be clean in uninitialize().
		pID3DBlob_PixelShaderByteCode->Release();
		pID3DBlob_PixelShaderByteCode = NULL;
		pID3DBlob_VertexShaderByteCode->Release();
		pID3DBlob_VertexShaderByteCode = NULL;

		return hr;
	}

	g_pID3D11DeviceContext->PSSetShader(g_pID3D11PixelShader, 0, 0);

	//	byte code of pixel shader not required henceforth, we will keep byte code of vertex shader only.
	pID3DBlob_PixelShaderByteCode->Release();
	pID3DBlob_PixelShaderByteCode = NULL;

	//-	Pixel Shaders
	/////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////
	//+	Input layout

	//	Create and set input layout.
	D3D11_INPUT_ELEMENT_DESC InputElementDesc[2];
	InputElementDesc[RTR_INPUT_SLOT_POSITION].SemanticName = "POSITION";
	InputElementDesc[RTR_INPUT_SLOT_POSITION].SemanticIndex = 0;	//	A semantic index is only needed in a case where there is more than one element with the same semantic.we will use in case of structure.
	InputElementDesc[RTR_INPUT_SLOT_POSITION].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	InputElementDesc[RTR_INPUT_SLOT_POSITION].InputSlot = RTR_INPUT_SLOT_POSITION;//	An integer value that identifies the input-assembler (see input slot). Valid values are between 0 and 15, defined in D3D11.h.
	InputElementDesc[RTR_INPUT_SLOT_POSITION].AlignedByteOffset = 0;
	InputElementDesc[RTR_INPUT_SLOT_POSITION].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	InputElementDesc[RTR_INPUT_SLOT_POSITION].InstanceDataStepRate = 0;

	InputElementDesc[RTR_INPUT_SLOT_TEXTURE].SemanticName = "TEXCOORD";
	InputElementDesc[RTR_INPUT_SLOT_TEXTURE].SemanticIndex = 0;	//	A semantic index is only needed in a case where there is more than one element with the same semantic.we will use in case of structure.
	InputElementDesc[RTR_INPUT_SLOT_TEXTURE].Format = DXGI_FORMAT_R32G32_FLOAT;
	InputElementDesc[RTR_INPUT_SLOT_TEXTURE].InputSlot = RTR_INPUT_SLOT_TEXTURE;//	An integer value that identifies the input-assembler (see input slot). Valid values are between 0 and 15, defined in D3D11.h.
	InputElementDesc[RTR_INPUT_SLOT_TEXTURE].AlignedByteOffset = 0;
	InputElementDesc[RTR_INPUT_SLOT_TEXTURE].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	InputElementDesc[RTR_INPUT_SLOT_TEXTURE].InstanceDataStepRate = 0;

	hr = g_pID3D11Device->CreateInputLayout(
		InputElementDesc,
		2,
		pID3DBlob_VertexShaderByteCode->GetBufferPointer(),
		pID3DBlob_VertexShaderByteCode->GetBufferSize(),
		&g_pID3D11InputLayout
	);
	if (FAILED(hr))
	{
		log_write("CreateInputLayout() failed.\n");
		pID3DBlob_VertexShaderByteCode->Release();
		pID3DBlob_VertexShaderByteCode = NULL;

		return hr;
	}

	g_pID3D11DeviceContext->IASetInputLayout(g_pID3D11InputLayout);	//	Input assembler stage.
	pID3DBlob_VertexShaderByteCode->Release();
	pID3DBlob_VertexShaderByteCode = NULL;

	//-	Input layout
	/////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////
	//+	Create vertex buffer Square

	//	Winding - Clock wise
	//	Left hand rule
	//	Vertices to Draw Quad using Triangle strip
	float farrCubeVertices[] =
	{
		// SIDE 3 ( FRONT )
		// triangle 1
		-1.0f, +1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		// triangle 2
		-1.0f, -1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		+1.0f, -1.0f, 0.0f,
	};
	D3D11_BUFFER_DESC BufferDesc;
	ZeroMemory(&BufferDesc, sizeof(BufferDesc));
	BufferDesc.Usage = D3D11_USAGE_DYNAMIC;	//	DirectX prefer dynamic drawing.
	BufferDesc.ByteWidth = sizeof(float)* ARRAYSIZE(farrCubeVertices);
	BufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	BufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	hr = g_pID3D11Device->CreateBuffer(
		&BufferDesc,
		NULL,	//	Pass data dynamically.
		&g_pID3D11Buffer_VertexBufferCubePosition
	);
	if (FAILED(hr))
	{
		log_write("CreateBuffer() failed for vertex buffer.\n");

		return hr;
	}

	D3D11_MAPPED_SUBRESOURCE MappedSubresource;
	//	Copy vertices into above buffer.
	ZeroMemory(&MappedSubresource, sizeof(MappedSubresource));
	g_pID3D11DeviceContext->Map(
		g_pID3D11Buffer_VertexBufferCubePosition,	//	ID3D11Buffer Inherit from ID3D11Resource 
		0,	//Index number of subresource i.e like 0 = position, 1 = color
		D3D11_MAP_WRITE_DISCARD,	//	specifies the CPU's read and write permissions for a resource.
		0,	//	Flag that specifies what the CPU does when the GPU is busy. This flag is optional.Cpu will wait.
		&MappedSubresource
	);

	CopyMemory(MappedSubresource.pData, farrCubeVertices, sizeof(farrCubeVertices));

	g_pID3D11DeviceContext->Unmap(g_pID3D11Buffer_VertexBufferCubePosition, 0);

	//-	Create vertex buffer
	/////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////
	//+	Create color buffer

	float farrCubeTexture[] =
	{
		// SIDE 3 (FRONT)
		// triangle 1
		0.0f, 1.0f,
		1.0f, 1.0f,
		0.0f, 0.0f,
		// triangle 2
		0.0f, 0.0f,
		1.0f, 1.0f,
		1.0f, 0.0f,
	};

	ZeroMemory(&BufferDesc, sizeof(BufferDesc));
	BufferDesc.Usage = D3D11_USAGE_DYNAMIC;	//	DirectX prefer dynamic drawing.
	BufferDesc.ByteWidth = sizeof(float)* ARRAYSIZE(farrCubeTexture);
	BufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	BufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	hr = g_pID3D11Device->CreateBuffer(
		&BufferDesc,
		NULL,	//	Pass data dynamically.
		&g_pID3D11Buffer_VertexBufferCubeTexture
	);
	if (FAILED(hr))
	{
		log_write("CreateBuffer() failed for vertex buffer.\n");

		return hr;
	}

	//	Copy vertices into above buffer.
	ZeroMemory(&MappedSubresource, sizeof(MappedSubresource));
	g_pID3D11DeviceContext->Map(
		g_pID3D11Buffer_VertexBufferCubeTexture,	//	ID3D11Buffer Inherit from ID3D11Resource 
		0,	//Index number of subresource i.e like 0 = position, 1 = color
		D3D11_MAP_WRITE_DISCARD,	//	specifies the CPU's read and write permissions for a resource.
		0,	//	Flag that specifies what the CPU does when the GPU is busy. This flag is optional.Cpu will wait.
		&MappedSubresource
	);

	CopyMemory(MappedSubresource.pData, farrCubeTexture, sizeof(farrCubeTexture));

	g_pID3D11DeviceContext->Unmap(g_pID3D11Buffer_VertexBufferCubeTexture, 0);

	//-	Create color buffer
	/////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////
	//+	Create constant buffer
	D3D11_BUFFER_DESC BufferDesc_ConstantBuffer;
	ZeroMemory(&BufferDesc_ConstantBuffer, sizeof(BufferDesc_ConstantBuffer));
	BufferDesc_ConstantBuffer.Usage = D3D11_USAGE_DEFAULT;
	BufferDesc_ConstantBuffer.ByteWidth = sizeof(CBUFFER);
	BufferDesc_ConstantBuffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

	hr = g_pID3D11Device->CreateBuffer(&BufferDesc_ConstantBuffer, nullptr, &g_pID3D11Buffer_ConstantBuffer);
	if (FAILED(hr))
	{
		log_write("CreateBuffer() failed for constant buffer.\n");

		return hr;
	}

	g_pID3D11DeviceContext->VSSetConstantBuffers(
		0,//	Index into the device's zero-based array to begin setting constant buffers to (ranges from 0 to D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT - 1).
		1,	//	Number of buffers.
		&g_pID3D11Buffer_ConstantBuffer
	);

	//-	Create constant buffer
	/////////////////////////////////////////////////////////////////////////////////////////////

	//-	Initialize Shaders, input layouts, constant buffer
	/////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////
	//+	In case of animation , manually make back culling off.
	//+	By default backface culling is On. 
	D3D11_RASTERIZER_DESC RasterizerDesc;

	ZeroMemory(&RasterizerDesc, sizeof(RasterizerDesc));
	RasterizerDesc.AntialiasedLineEnable = FALSE;
	RasterizerDesc.MultisampleEnable = FALSE;
	RasterizerDesc.DepthBias = 0;	//	In case of shadow need to change.
	RasterizerDesc.DepthBiasClamp = 0.0f;
	RasterizerDesc.SlopeScaledDepthBias = 0.0f;
	RasterizerDesc.CullMode = D3D11_CULL_NONE;
	RasterizerDesc.DepthClipEnable = TRUE;
	RasterizerDesc.FillMode = D3D11_FILL_SOLID;
	RasterizerDesc.FrontCounterClockwise = FALSE;
	RasterizerDesc.ScissorEnable = FALSE;

	g_pID3D11Device->CreateRasterizerState(&RasterizerDesc, &g_pID3D11RasterizerState);

	g_pID3D11DeviceContext->RSSetState(g_pID3D11RasterizerState);

	//-	In case of animation , manually make back culling off.
	/////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////
	//+	Cube : Create texture resource and texture view

	hr = LoadD3DTexture(L"Smiley.bmp", &g_pID3D11ShaderResourceView_TextureCube);
	if (FAILED(hr))
	{
		log_write("LoadD3DTexture(Kundali) failed.");
		return hr;
	}

	//	Create the sampler state.
	D3D11_SAMPLER_DESC SamplerDesc;
	ZeroMemory(&SamplerDesc, sizeof(SamplerDesc));
	SamplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	SamplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	SamplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	SamplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;

	hr = g_pID3D11Device->CreateSamplerState(&SamplerDesc, &g_pID3D11SamplerState_TextureCube);
	if (FAILED(hr))
	{
		log_write("CreateSamplerState(Kundali) failed.");
		return hr;
	}

	//-	Cube : Create texture resource and texture view
	/////////////////////////////////////////////////////////////////////////////////////////////

	//	d3d clear color, analogus to glClearColor() in openGL.
	g_fClearColor[0] = 0.0f;
	g_fClearColor[1] = 0.0f;
	g_fClearColor[2] = 0.0f;
	g_fClearColor[3] = 1.0f;

	//	Set Projection matrix to orthographic projection.
	g_PerspectiveProjectionMatrix = XMMatrixIdentity();
	//
	//	Resize.
	//
	hr = resize(WIN_WIDTH, WIN_HEIGHT);
	if (FAILED(hr))
	{
		log_write("resize() failed.\n");
	}
	else
	{
		log_write("resize() succeded.\n");
	}

	return hr;
}

HRESULT
LoadD3DTexture(const WCHAR *pwszTextureName, ID3D11ShaderResourceView **ppID3D11ShaderResourceView)
{
	HRESULT hr;

	//	Create texture
	hr = DirectX::CreateWICTextureFromFile(g_pID3D11Device, g_pID3D11DeviceContext, pwszTextureName, NULL, ppID3D11ShaderResourceView);
	if (FAILED(hr))
	{
		log_write("CreateWICTextureFromFile() failed.");
		return hr;
	}

	return hr;
}


VOID
LoadD3DProceduralTexture(ID3D11ShaderResourceView **ppID3D11ShaderResourceView)
{
	//HRESULT hr;

	////	Create texture
	////hr = DirectX::CreateWICTextureFromMemory(g_pID3D11Device, g_pID3D11DeviceContext, pwszTextureName, NULL, ppID3D11ShaderResourceView);
	//if (FAILED(hr))
	//{
	//	log_write("CreateWICTextureFromFile() failed.");
	//	return hr;
	//}

	//return hr;
}

HRESULT resize(int iWidth, int iHeight)
{
	HRESULT hr = S_OK;

	//	Free any size-dependent resources.
	if (g_pID3D11DepthStencilView)
	{
		g_pID3D11DepthStencilView->Release();
		g_pID3D11DepthStencilView = NULL;
	}

	//	Free any size-dependent resources.
	if (g_pID3D11RenderTargetView)
	{
		g_pID3D11RenderTargetView->Release();
		g_pID3D11RenderTargetView = NULL;
	}

	//
	//	Resize swap chain buffer accordingly
	//
	g_pIDXGISwapChain->ResizeBuffers(1, iWidth, iHeight, DXGI_FORMAT_R8G8B8A8_UNORM, 0);

	//
	//	Again get back buffer from swap chain.
	//
	ID3D11Texture2D *pID3D11Texture2D_BackBuffer;
	g_pIDXGISwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (VOID**)&pID3D11Texture2D_BackBuffer);

	//
	//	Again get render target view from d3d11 device using above back buffer.
	//
	hr = g_pID3D11Device->CreateRenderTargetView(pID3D11Texture2D_BackBuffer, NULL, &g_pID3D11RenderTargetView);
	if (FAILED(hr))
	{
		log_write("CreateRenderTargetView() failed.\n");
		return hr;
	}

	//	Release temp resource.
	pID3D11Texture2D_BackBuffer->Release();
	pID3D11Texture2D_BackBuffer = NULL;

	//////////////////////////////////////////////////////////////////////////////////////
	//+	Make Depth on
	D3D11_TEXTURE2D_DESC TextureDesc;
	ZeroMemory(&TextureDesc, sizeof(TextureDesc));
	TextureDesc.Height = iHeight;
	TextureDesc.Width = iWidth;
	TextureDesc.ArraySize = 1;	//	No of array for depth
	TextureDesc.MipLevels = 1;
	TextureDesc.SampleDesc.Count = 1;	//	1 to 4 
	TextureDesc.SampleDesc.Quality = 0;
	TextureDesc.Format = DXGI_FORMAT_D32_FLOAT;
	TextureDesc.Usage = D3D11_USAGE_DEFAULT;
	TextureDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	TextureDesc.CPUAccessFlags = 0;
	TextureDesc.MiscFlags = 0;

	ID3D11Texture2D *pID3D11Texture2D_DepthBuffer = NULL;
	hr = g_pID3D11Device->CreateTexture2D(&TextureDesc, NULL, &pID3D11Texture2D_DepthBuffer);
	if (FAILED(hr))
	{
		log_write("CreateTexture2D() failed.\n");
		return hr;
	}

	D3D11_DEPTH_STENCIL_VIEW_DESC DepthStensilViewDesc;
	ZeroMemory(&DepthStensilViewDesc, sizeof(DepthStensilViewDesc));
	DepthStensilViewDesc.Format = DXGI_FORMAT_D32_FLOAT;
	DepthStensilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;	//	MS - Multi Sampling.

	hr = g_pID3D11Device->CreateDepthStencilView(pID3D11Texture2D_DepthBuffer, &DepthStensilViewDesc, &g_pID3D11DepthStencilView);
	if (FAILED(hr))
	{
		log_write("CreateDepthStencilView() failed.\n");

		pID3D11Texture2D_DepthBuffer->Release();
		pID3D11Texture2D_DepthBuffer = NULL;
		return hr;
	}

	pID3D11Texture2D_DepthBuffer->Release();
	pID3D11Texture2D_DepthBuffer = NULL;

	//-	Make Depth on
	//////////////////////////////////////////////////////////////////////////////////////

	//
	//	Set render target view and depth stensil view as render target.
	//
	g_pID3D11DeviceContext->OMSetRenderTargets(1, &g_pID3D11RenderTargetView, g_pID3D11DepthStencilView);

	//	Note: If you want to manipulate any stage of pipeline then it can only be done through device context
	//	and the method from device context is used always start from 2 pre-define initials like OM(output merger) in above method.

	//
	//	Set viewport.
	//
	D3D11_VIEWPORT D3DViewPort;
	D3DViewPort.TopLeftX = 0;
	D3DViewPort.TopLeftY = 0;
	D3DViewPort.Width = (float)iWidth;
	D3DViewPort.Height = (float)iHeight;
	D3DViewPort.MinDepth = 0.0f;
	D3DViewPort.MaxDepth = 1.0f;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);

	//	Set orthographic matrix.
	g_PerspectiveProjectionMatrix = XMMatrixPerspectiveFovLH(XMConvertToRadians(45), ((float)iWidth / (float)iHeight), 0.1f, 100.0f);

	return hr;
}

VOID display()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvpMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	XMMATRIX rotationMatrix;
	XMMATRIX rotationMatrixR1;
	XMMATRIX rotationMatrixR2;
	XMMATRIX rotationMatrixR3;
	CBUFFER constantBuffer;

	//	Clear render target view to a choosen color.
	g_pID3D11DeviceContext->ClearRenderTargetView(g_pID3D11RenderTargetView, g_fClearColor);

	g_pID3D11DeviceContext->ClearDepthStencilView(g_pID3D11DepthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0);

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Cube
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferCubePosition,
		&uiStride,
		&uiOffset
	);

	float arrQuadTextureCoord[] =
	{
		// SIDE 3 (FRONT)
		// triangle 1
		0.0f, 1.0f,
		1.0f, 1.0f,
		0.0f, 0.0f,
		// triangle 2
		0.0f, 0.0f,
		1.0f, 1.0f,
		1.0f, 0.0f,
	};

	switch (g_iDigit)
	{
	case 0x31://	1
	case 0x61:// Num pad 1
		arrQuadTextureCoord[0] = 0.0f;
		arrQuadTextureCoord[1] = 0.5f;

		arrQuadTextureCoord[2] = 0.5f;
		arrQuadTextureCoord[3] = 0.5f;

		arrQuadTextureCoord[4] = 0.0f;
		arrQuadTextureCoord[5] = 0.0f;

		arrQuadTextureCoord[6] = 0.0f;
		arrQuadTextureCoord[7] = 0.0f;

		arrQuadTextureCoord[8] = 0.5f;
		arrQuadTextureCoord[9] = 0.5f;

		arrQuadTextureCoord[10] = 0.5f;
		arrQuadTextureCoord[11] = 0.0f;

		break;

	case 0x32:
	case 0x62:
		arrQuadTextureCoord[0] = 0.0f;
		arrQuadTextureCoord[1] = 1.0f;

		arrQuadTextureCoord[2] = 1.0f;
		arrQuadTextureCoord[3] = 1.0f;

		arrQuadTextureCoord[4] = 0.0f;
		arrQuadTextureCoord[5] = 0.0f;

		arrQuadTextureCoord[6] = 0.0f;
		arrQuadTextureCoord[7] = 0.0f;

		arrQuadTextureCoord[8] = 1.0f;
		arrQuadTextureCoord[9] = 1.0f;

		arrQuadTextureCoord[10] = 1.0f;
		arrQuadTextureCoord[11] = 0.0f;

		break;

	case 0x33:
	case 0x63:
		arrQuadTextureCoord[0] = 0.0f;
		arrQuadTextureCoord[1] = 2.0f;

		arrQuadTextureCoord[2] = 2.0f;
		arrQuadTextureCoord[3] = 2.0f;

		arrQuadTextureCoord[4] = 0.0f;
		arrQuadTextureCoord[5] = 0.0f;

		arrQuadTextureCoord[6] = 0.0f;
		arrQuadTextureCoord[7] = 0.0f;

		arrQuadTextureCoord[8] = 2.0f;
		arrQuadTextureCoord[9] = 2.0f;

		arrQuadTextureCoord[10] = 2.0f;
		arrQuadTextureCoord[11] = 0.0f;

		break;

	case 0x34:
	case 0x64:
		arrQuadTextureCoord[0] = 0.5f;
		arrQuadTextureCoord[1] = 0.5f;

		arrQuadTextureCoord[2] = 0.5f;
		arrQuadTextureCoord[3] = 0.5f;

		arrQuadTextureCoord[4] = 0.5f;
		arrQuadTextureCoord[5] = 0.5f;

		arrQuadTextureCoord[6] = 0.5f;
		arrQuadTextureCoord[7] = 0.5f;

		arrQuadTextureCoord[8] = 0.5f;
		arrQuadTextureCoord[9] = 0.5f;

		arrQuadTextureCoord[10] = 0.5f;
		arrQuadTextureCoord[11] = 0.5f;

		break;
	}

	D3D11_MAPPED_SUBRESOURCE MappedSubresource;

	//	Copy vertices into above buffer.
	ZeroMemory(&MappedSubresource, sizeof(MappedSubresource));
	g_pID3D11DeviceContext->Map(
		g_pID3D11Buffer_VertexBufferCubeTexture,	//	ID3D11Buffer Inherit from ID3D11Resource 
		0,	//Index number of subresource i.e like 0 = position, 1 = color
		D3D11_MAP_WRITE_DISCARD,	//	specifies the CPU's read and write permissions for a resource.
		0,	//	Flag that specifies what the CPU does when the GPU is busy. This flag is optional.Cpu will wait.
		&MappedSubresource
	);

	CopyMemory(MappedSubresource.pData, arrQuadTextureCoord, sizeof(arrQuadTextureCoord));

	g_pID3D11DeviceContext->Unmap(g_pID3D11Buffer_VertexBufferCubeTexture, 0);

	uiStride = sizeof(float) * 2;
	uiOffset = 0;
	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_COLOR,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferCubeTexture,
		&uiStride,
		&uiOffset
	);

	//	Bind texture and sampler as pixel shader resource.
	g_pID3D11DeviceContext->PSSetShaderResources(0, 1, &g_pID3D11ShaderResourceView_TextureCube);
	g_pID3D11DeviceContext->PSSetSamplers(0, 1, &g_pID3D11SamplerState_TextureCube);

	//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	//worldMatrix = worldMatrix * rotationMatrix;
	//worldMatrix = worldMatrix * XMMatrixScaling(0.8f, 0.8f, 0.8f);
	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 4.0f);

	//	World view projection matrix.
	wvpMatrix = worldMatrix * viewMatrix * g_PerspectiveProjectionMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.WorldViewProjectionMatrix = wvpMatrix;

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw vertex buffer to render target.
	g_pID3D11DeviceContext->Draw(6, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

	//	Swap buffers, switch betwwn front and back buffers.
	//	1st parameter - no synchronization with vertical refresh rate(vertical blank). 
	//	2nd parameter - how many frames from how many buffers need to show. 
	g_pIDXGISwapChain->Present(0, 0);
}


void update()
{
	g_fAngleTri = g_fAngleTri + 0.1f;

	if (g_fAngleTri >= 360)
	{
		g_fAngleTri = 0.0f;
	}

	g_fAngleRect = g_fAngleRect + 0.1f;

	if (g_fAngleRect >= 360)
	{
		g_fAngleRect = 0.0f;
	}
}


VOID uninitialize()
{
	if (g_pID3D11SamplerState_TextureCube)
	{
		g_pID3D11SamplerState_TextureCube->Release();
		g_pID3D11SamplerState_TextureCube = NULL;
	}

	if (g_pID3D11ShaderResourceView_TextureCube)
	{
		g_pID3D11ShaderResourceView_TextureCube->Release();
		g_pID3D11ShaderResourceView_TextureCube = NULL;
	}

	if (g_pID3D11RasterizerState)
	{
		g_pID3D11RasterizerState->Release();
		g_pID3D11RasterizerState = NULL;
	}

	if (g_pID3D11Buffer_ConstantBuffer)
	{
		g_pID3D11Buffer_ConstantBuffer->Release();
		g_pID3D11Buffer_ConstantBuffer = NULL;
	}

	if (g_pID3D11InputLayout)
	{
		g_pID3D11InputLayout->Release();
		g_pID3D11InputLayout = NULL;
	}

	if (g_pID3D11Buffer_VertexBufferCubePosition)
	{
		g_pID3D11Buffer_VertexBufferCubePosition->Release();
		g_pID3D11Buffer_VertexBufferCubePosition = NULL;
	}

	if (g_pID3D11Buffer_VertexBufferCubeTexture)
	{
		g_pID3D11Buffer_VertexBufferCubeTexture->Release();
		g_pID3D11Buffer_VertexBufferCubeTexture = NULL;
	}

	if (g_pID3D11PixelShader)
	{
		g_pID3D11PixelShader->Release();
		g_pID3D11PixelShader = NULL;
	}

	if (g_pID3D11VertexShader)
	{
		g_pID3D11VertexShader->Release();
		g_pID3D11VertexShader = NULL;
	}

	if (g_pID3D11DepthStencilView)
	{
		g_pID3D11DepthStencilView->Release();
		g_pID3D11DepthStencilView = NULL;
	}

	if (g_pID3D11RenderTargetView)
	{
		g_pID3D11RenderTargetView->Release();
		g_pID3D11RenderTargetView = NULL;
	}

	if (g_pIDXGISwapChain)
	{
		g_pIDXGISwapChain->Release();
		g_pIDXGISwapChain = NULL;
	}

	if (g_pID3D11DeviceContext)
	{
		g_pID3D11DeviceContext->Release();
		g_pID3D11DeviceContext = NULL;
	}

	if (g_pID3D11Device)
	{
		g_pID3D11Device->Release();
		g_pID3D11Device = NULL;
	}

	if (true == g_boFullScreen)
	{
		SetWindowLong(g_hWnd, GWL_STYLE, g_dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(g_hWnd, &g_WindowPlacementPrev);
		SetWindowPos(g_hWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);

		ShowCursor(TRUE);
	}

	log_write("uninitialize() succeded \n");
	log_write("Log file is succesfuly closed \n");
}
#endif