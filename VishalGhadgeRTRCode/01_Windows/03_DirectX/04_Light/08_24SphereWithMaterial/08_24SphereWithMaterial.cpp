#if 1
#include <Windows.h>
#include <stdio.h>

#include <d3d11.h>
#include <d3dcompiler.h>

#pragma warning(disable: 4838) // Suppress XNAMATH warning typecast of unsigned int to int
#include "..\..\include\XNAMath\xnamath.h"
#include "..\..\include\Sphere.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "D3dcompiler.lib")
#pragma comment(lib, "Sphere.lib")

#define WIN_WIDTH	800
#define WIN_HEIGHT	600

#define WINDOW_NAME		L"D3D - 24 Sphere Material"
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

FLOAT g_fAngleLight = 0.0f;
char chAnimationAxis = 'X';

int g_iCurrentWidth;
int g_iCurrentHeight;

enum RTR_INPUT_SLOT
{
	RTR_INPUT_SLOT_POSITION = 0,
	RTR_INPUT_SLOT_COLOR,
	RTR_INPUT_SLOT_TEXTURE,
	RTR_INPUT_SLOT_NORMAL = RTR_INPUT_SLOT_COLOR,
};

float g_fClearColor[4];	//	RGBA
IDXGISwapChain *g_pIDXGISwapChain = NULL;	//	DXGI - DirectX Graphics Interface
ID3D11Device *g_pID3D11Device = NULL;
ID3D11DeviceContext *g_pID3D11DeviceContext = NULL;
ID3D11RenderTargetView *g_pID3D11RenderTargetView = NULL;
ID3D11DepthStencilView *g_pID3D11DepthStencilView = NULL;

ID3D11VertexShader *g_pID3D11VertexShader = NULL;	//	Vertex shader object
ID3D11PixelShader *g_pID3D11PixelShader = NULL;		//	same as fragment shader in OpenGL
ID3D11Buffer *g_pID3D11Buffer_VertexBufferSpherePosition = NULL;	//	vbo_position in openGL
ID3D11Buffer *g_pID3D11Buffer_VertexBufferSphereNormal = NULL;	//	vbo_position in openGL
ID3D11InputLayout *g_pID3D11InputLayout = NULL;
ID3D11Buffer *g_pID3D11Buffer_ConstantBuffer = NULL;
ID3D11RasterizerState *g_pID3D11RasterizerState = NULL;

ID3D11Buffer *g_pID3D11Buffer_IndexBuffer = NULL;

FLOAT g_farrLightAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
FLOAT g_farrLightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
FLOAT g_farrLightSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
FLOAT g_farrLightPosition[] = { 100.0f, 100.0f, -100.0f, 1.0f };

///////////////////////////////////////////////////////////////////////////
//+Material


//
//	Materail 00
//
FLOAT g_arrMaterial00Ambient[] = { 0.0215f, 0.1745f, 0.0215f, 1.0f };
FLOAT g_arrMaterial00Diffuse[] = { 0.07568f, 0.61424f, 0.07568f, 1.0f };
FLOAT g_arrMaterial00Specular[] = { 0.633f, 0.727811f, 0.633f, 1.0f };
FLOAT g_Material00Shininess = 0.6f * 128.0f;

//
//	Materail 10
//
FLOAT g_arrMaterial10Ambient[] = { 0.135f, 0.2225f, 0.1575f, 1.0f };
FLOAT g_arrMaterial10Diffuse[] = { 0.54f, 0.89f, 0.63f, 1.0f };
FLOAT g_arrMaterial10Specular[] = { 0.316228f, 0.316228f, 0.316228f, 1.0f };
FLOAT g_Material10Shininess = 0.1f * 128.0f;

//
//	Materail 20
//
FLOAT g_arrMaterial20Ambient[] = { 0.05375f, 0.05f, 0.06625f, 1.0f };
FLOAT g_arrMaterial20Diffuse[] = { 0.18275f, 0.17f, 0.22525f, 1.0f };
FLOAT g_arrMaterial20Specular[] = { 0.332741f, 0.328634f, 0.346435f, 1.0f };
FLOAT g_Material20Shininess = 0.3f * 128.0f;

//
//	Materail 30
//
FLOAT g_arrMaterial30Ambient[] = { 0.25f, 0.20725f, 0.20725f, 1.0f };
FLOAT g_arrMaterial30Diffuse[] = { 1.0f, 0.829f, 0.829f, 1.0f };
FLOAT g_arrMaterial30Specular[] = { 0.296648f, 0.296648f, 0.296648f, 1.0f };
FLOAT g_Material30Shininess = 0.088f * 128.0f;

//
//	Materail 40
//
FLOAT g_arrMaterial40Ambient[] = { 0.1745f, 0.01175f, 0.01175f, 1.0f };
FLOAT g_arrMaterial40Diffuse[] = { 0.61424f, 0.04136f, 0.04136f, 1.0f };
FLOAT g_arrMaterial40Specular[] = { 0.727811f, 0.626959f, 0.626959f, 1.0f };
FLOAT g_Material40Shininess = 0.6f * 128.0f;

//
//	Materail 50
//
FLOAT g_arrMaterial50Ambient[] = { 0.1f, 0.18725f, 0.1745f, 1.0f };
FLOAT g_arrMaterial50Diffuse[] = { 0.396f, 0.74151f, 0.69102f, 1.0f };
FLOAT g_arrMaterial50Specular[] = { 0.297254f, 0.30829f, 0.306678f, 1.0f };
FLOAT g_Material50Shininess = 0.1f * 128.0f;

//
//	Materail 01
//
FLOAT g_arrMaterial01Ambient[] = { 0.329412f, 0.223529f, 0.027451f, 1.0f };
FLOAT g_arrMaterial01Diffuse[] = { 0.780392f, 0.568627f, 0.113725f, 1.0f };
FLOAT g_arrMaterial01Specular[] = { 0.992157f, 0.941176f, 0.807843f, 1.0f };
FLOAT g_Material01Shininess = 0.21794872f * 128.0f;

//
//	Materail 11
//
FLOAT g_arrMaterial11Ambient[] = { 0.2125f, 0.1275f, 0.054f, 1.0f };
FLOAT g_arrMaterial11Diffuse[] = { 0.714f, 0.4284f, 0.18144f, 1.0f };
FLOAT g_arrMaterial11Specular[] = { 0.393548f, 0.271906f, 0.166721f, 1.0f };
FLOAT g_Material11Shininess = 0.2f * 128.0f;

//
//	Materail 21
//
FLOAT g_arrMaterial21Ambient[] = { 0.25f, 0.25f, 0.25f, 1.0f };
FLOAT g_arrMaterial21Diffuse[] = { 0.4f, 0.4f, 0.4f, 1.0f };
FLOAT g_arrMaterial21Specular[] = { 0.774597f, 0.774597f, 0.774597f, 1.0f };
FLOAT g_Material21Shininess = 0.6f * 128.0f;

//
//	Materail 31
//
FLOAT g_arrMaterial31Ambient[] = { 0.19125f, 0.0735f, 0.0225f, 1.0f };
FLOAT g_arrMaterial31Diffuse[] = { 0.7038f, 0.27048f, 0.0828f, 1.0f };
FLOAT g_arrMaterial31Specular[] = { 0.256777f, 0.137622f, 0.296648f, 1.0f };
FLOAT g_Material31Shininess = 0.1f * 128.0f;

//
//	Materail 41
//
FLOAT g_arrMaterial41Ambient[] = { 0.24725f, 0.1995f, 0.0745f, 1.0f };
FLOAT g_arrMaterial41Diffuse[] = { 0.75164f, 0.60648f, 0.22648f, 1.0f };
FLOAT g_arrMaterial41Specular[] = { 0.628281f, 0.555802f, 0.366065f, 1.0f };
FLOAT g_Material41Shininess = 0.4f * 128.0f;

//
//	Materail 51
//
FLOAT g_arrMaterial51Ambient[] = { 0.19225f, 0.19225f, 0.19225f, 1.0f };
FLOAT g_arrMaterial51Diffuse[] = { 0.50754f, 0.50754f, 0.50754f, 1.0f };
FLOAT g_arrMaterial51Specular[] = { 0.508273f, 0.508273f, 0.508273f, 1.0f };
FLOAT g_Material51Shininess = 0.4f * 128.0f;

//
//	Materail 02
//
FLOAT g_arrMaterial02Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
FLOAT g_arrMaterial02Diffuse[] = { 0.01f, 0.01f, 0.01f, 1.0f };
FLOAT g_arrMaterial02Specular[] = { 0.5f, 0.5f, 0.5f, 1.0f };
FLOAT g_Material02Shininess = 0.25f * 128.0f;

//
//	Materail 12
//
FLOAT g_arrMaterial12Ambient[] = { 0.0f, 0.1f, 0.06f, 1.0f };
FLOAT g_arrMaterial12Diffuse[] = { 0.0f, 0.50980392f, 0.50980392f, 1.0f };
FLOAT g_arrMaterial12Specular[] = { 0.50980392f, 0.50980392f, 0.50980392f, 1.0f };
FLOAT g_Material12Shininess = 0.25f * 128.0f;

//
//	Materail 22
//
FLOAT g_arrMaterial22Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
FLOAT g_arrMaterial22Diffuse[] = { 0.1f, 0.35f, 0.1f, 1.0f };
FLOAT g_arrMaterial22Specular[] = { 0.45f, 0.45f, 0.45f, 1.0f };
FLOAT g_Material22Shininess = 0.25f * 128.0f;

//
//	Materail 32
//
FLOAT g_arrMaterial32Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
FLOAT g_arrMaterial32Diffuse[] = { 0.5f, 0.0f, 0.0f, 1.0f };
FLOAT g_arrMaterial32Specular[] = { 0.7f, 0.6f, 0.6f, 1.0f };
FLOAT g_Material32Shininess = 0.25f * 128.0f;

//
//	Materail 42
//
FLOAT g_arrMaterial42Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
FLOAT g_arrMaterial42Diffuse[] = { 0.55f, 0.55f, 0.55f, 1.0f };
FLOAT g_arrMaterial42Specular[] = { 0.70f, 0.70f, 0.70f, 1.0f };
FLOAT g_Material42Shininess = 0.25f * 128.0f;

//
//	Materail 52
//
FLOAT g_arrMaterial52Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
FLOAT g_arrMaterial52Diffuse[] = { 0.5f, 0.5f, 0.0f, 1.0f };
FLOAT g_arrMaterial52Specular[] = { 0.60f, 0.60f, 0.50f, 1.0f };
FLOAT g_Material52Shininess = 0.25f * 128.0f;

//
//	Materail 03
//
FLOAT g_arrMaterial03Ambient[] = { 0.02f, 0.02f, 0.02f, 1.0f };
FLOAT g_arrMaterial03Diffuse[] = { 0.01f, 0.01f, 0.01f, 1.0f };
FLOAT g_arrMaterial03Specular[] = { 0.4f, 0.4f, 0.4f, 1.0f };
FLOAT g_Material03Shininess = 0.078125f * 128.0f;

//
//	Materail 13
//
FLOAT g_arrMaterial13Ambient[] = { 0.0f, 0.05f, 0.05f, 1.0f };
FLOAT g_arrMaterial13Diffuse[] = { 0.4f, 0.5f, 0.5f, 1.0f };
FLOAT g_arrMaterial13Specular[] = { 0.04f, 0.7f, 0.7f, 1.0f };
FLOAT g_Material13Shininess = 0.078125f * 128.0f;

//
//	Materail 23
//
FLOAT g_arrMaterial23Ambient[] = { 0.0f, 0.05f, 0.0f, 1.0f };
FLOAT g_arrMaterial23Diffuse[] = { 0.4f, 0.5f, 0.4f, 1.0f };
FLOAT g_arrMaterial23Specular[] = { 0.04f, 0.7f, 0.04f, 1.0f };
FLOAT g_Material23Shininess = 0.078125f * 128.0f;

//
//	Materail 33
//
FLOAT g_arrMaterial33Ambient[] = { 0.05f, 0.0f, 0.0f, 1.0f };
FLOAT g_arrMaterial33Diffuse[] = { 0.5f, 0.4f, 0.4f, 1.0f };
FLOAT g_arrMaterial33Specular[] = { 0.7f, 0.04f, 0.04f, 1.0f };
FLOAT g_Material33Shininess = 0.078125f * 128.0f;

//
//	Materail 43
//
FLOAT g_arrMaterial43Ambient[] = { 0.05f, 0.05f, 0.05f, 1.0f };
FLOAT g_arrMaterial43Diffuse[] = { 0.5f, 0.5f, 0.5f, 1.0f };
FLOAT g_arrMaterial43Specular[] = { 0.7f, 0.7f, 0.7f, 1.0f };
FLOAT g_Material43Shininess = 0.78125f * 128.0f;

//
//	Materail 53
//
FLOAT g_arrMaterial53Ambient[] = { 0.05f, 0.05f, 0.0f, 1.0f };
FLOAT g_arrMaterial53Diffuse[] = { 0.5f, 0.5f, 0.4f, 1.0f };
FLOAT g_arrMaterial53Specular[] = { 0.7f, 0.7f, 0.04f, 1.0f };
FLOAT g_Material53Shininess = 0.078125f * 128.0f;

//-Material
///////////////////////////////////////////////////////////////////////////


float g_farrSphereVertices[1146];
float g_farrSphereNormals[1146];
float g_farrSphereTextures[764];
unsigned short g_usharrSphereElements[2280];
UINT g_uiNumVertices;
UINT g_uiNumElements;

struct CBUFFER
{
	XMMATRIX worldMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX projectionMatrix;
	XMMATRIX rotationMatrix;
	XMVECTOR vecLA;
	XMVECTOR vecLD;
	XMVECTOR vecLS;
	XMVECTOR vecLightPosition;
	XMVECTOR vecKA;
	XMVECTOR vecKD;
	XMVECTOR vecKS;
	FLOAT fMaterialShininess;
	UINT uiKeyPressed;
};

XMMATRIX g_PerspectiveProjectionMatrix;

bool g_bLight = false;

void log_write(char* msg)
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
		g_iCurrentWidth = LOWORD(lParam);
		g_iCurrentHeight = HIWORD(lParam);
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
		case 'X':
		case 'x':
		case 'y':
		case 'Y':
		case 'z':
		case 'Z':
			chAnimationAxis = wParam;
			break;

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

		case 'l':
		case 'L':
			if (false == g_bLight)
			{
				g_bLight = true;
			}
			else
			{
				g_bLight = false;
			}
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
		"float4x4 worldMatrix;"								\
		"float4x4 viewMatrix;"								\
		"float4x4 projectionMatrix;"								\
		"float4x4 rotationMatrix;"								\
		"float4 vecLA;"								\
		"float4 vecLD;"								\
		"float4 vecLS;"								\
		"float4 vecLightPosition;"								\
		"float4 vecKA;"								\
		"float4 vecKD;"								\
		"float4 vecKS;"								\
		"float fMaterialShininess;"								\
		"uint uiKeyPressed;"								\
		"}"								\
		"struct vertex_output"		\
		"{"								\
		"float4 Position:SV_POSITION;"								\
		"float3 transformed_normals:NORMAL0;"								\
		"float3 light_direction:NORMAL1;"								\
		"float3 viewer_vector:NORMAL2;"								\
		"};"								\
		/*Here POSITION is like vPosition in OpenGL*/
		"vertex_output main(float4 pos:POSITION, float4 norm:NORMAL)"/* : SV_POSITION"*/				\
		"{"								\
		"vertex_output output;"								\
		"if (1 == uiKeyPressed)"								\
		"{"								\
		"float4 eyeCoordinates = mul(worldMatrix, pos);"								\
		"eyeCoordinates = mul(viewMatrix, eyeCoordinates);"								\
		"output.transformed_normals = mul((float3x3)worldMatrix,(float3)norm);"								\
		"output.transformed_normals= mul((float3x3)viewMatrix,output.transformed_normals);"								\
		"float4 rotated_light_direction = mul(rotationMatrix, vecLightPosition);"								\
		"output.light_direction= (float3)rotated_light_direction - eyeCoordinates.xyz;"								\
		"output.viewer_vector = -eyeCoordinates.xyz;"								\
		"}"								\
		"output.Position = mul(worldMatrix, pos);"								\
		"output.Position = mul(viewMatrix, output.Position);"								\
		"output.Position = mul(projectionMatrix, output.Position);"								\
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
		"cbuffer ConstantBuffer"		\
		"{"								\
		"float4x4 worldMatrix;"								\
		"float4x4 viewMatrix;"								\
		"float4x4 projectionMatrix;"								\
		"float4x4 rotationMatrix;"								\
		"float4 vecLA;"								\
		"float4 vecLD;"								\
		"float4 vecLS;"								\
		"float4 vecLightPosition;"								\
		"float4 vecKA;"								\
		"float4 vecKD;"								\
		"float4 vecKS;"								\
		"float fMaterialShininess;"								\
		"uint uiKeyPressed;"								\
		"}"								\
		"struct vertex_output"		\
		"{"								\
		"float4 Position:SV_POSITION;"								\
		"float3 transformed_normals:NORMAL0;"								\
		"float3 light_direction:NORMAL1;"								\
		"float3 viewer_vector:NORMAL2;"								\
		"};"								\
		"float4 main(float4 pos:SV_POSITION, vertex_output input):SV_TARGET"		\
		"{"		\
		"float4 output_color;"		\
		"if (1 == uiKeyPressed)"								\
		"{"								\
		"float3 normalized_transformed_normals= normalize(input.transformed_normals);"								\
		"float3 normalized_light_direction= normalize(input.light_direction);"								\
		"float3 normalized_viewer_vector = normalize(input.viewer_vector);"								\
		"float tn_dot_ld = max(dot(normalized_transformed_normals,normalized_light_direction), 0.0);"								\
		"float4 ambient = vecLA * vecKA;"								\
		"float4 diffuse = vecLD * vecKD * tn_dot_ld;"								\
		"float3 reflection_vector = reflect(-normalized_light_direction, normalized_transformed_normals);"								\
		"float4 specular = vecLS * vecKS * pow(max(dot(reflection_vector, normalized_viewer_vector), 0.0), fMaterialShininess);"								\
		"output_color = ambient + diffuse + specular;"								\
		"}"								\
		"else"								\
		"{"								\
		"output_color = float4(1.0f,1.0f,1.0f,1.0f) ;"								\
		"}"								\
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

	InputElementDesc[RTR_INPUT_SLOT_NORMAL].SemanticName = "NORMAL";
	InputElementDesc[RTR_INPUT_SLOT_NORMAL].SemanticIndex = 0;	//	A semantic index is only needed in a case where there is more than one element with the same semantic.we will use in case of structure.
	InputElementDesc[RTR_INPUT_SLOT_NORMAL].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	InputElementDesc[RTR_INPUT_SLOT_NORMAL].InputSlot = RTR_INPUT_SLOT_NORMAL;//	An integer value that identifies the input-assembler (see input slot). Valid values are between 0 and 15, defined in D3D11.h.
	InputElementDesc[RTR_INPUT_SLOT_NORMAL].AlignedByteOffset = 0;
	InputElementDesc[RTR_INPUT_SLOT_NORMAL].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	InputElementDesc[RTR_INPUT_SLOT_NORMAL].InstanceDataStepRate = 0;

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

	getSphereVertexData(g_farrSphereVertices, g_farrSphereNormals, g_farrSphereTextures, g_usharrSphereElements);
	g_uiNumVertices = ARRAYSIZE(g_farrSphereVertices);//getNumberOfSphereVertices();	Fixme: getNumberOfSphereVertices() returns 
	g_uiNumElements = getNumberOfSphereElements();

	D3D11_BUFFER_DESC BufferDesc;
	ZeroMemory(&BufferDesc, sizeof(BufferDesc));
	BufferDesc.Usage = D3D11_USAGE_DYNAMIC;	//	DirectX prefer dynamic drawing.
	BufferDesc.ByteWidth = sizeof(float)* ARRAYSIZE(g_farrSphereVertices);
	BufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	BufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	hr = g_pID3D11Device->CreateBuffer(
		&BufferDesc,
		NULL,	//	Pass data dynamically.
		&g_pID3D11Buffer_VertexBufferSpherePosition
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
		g_pID3D11Buffer_VertexBufferSpherePosition,	//	ID3D11Buffer Inherit from ID3D11Resource 
		0,	//Index number of subresource i.e like 0 = position, 1 = color
		D3D11_MAP_WRITE_DISCARD,	//	specifies the CPU's read and write permissions for a resource.
		0,	//	Flag that specifies what the CPU does when the GPU is busy. This flag is optional.Cpu will wait.
		&MappedSubresource
	);

	CopyMemory(MappedSubresource.pData, g_farrSphereVertices, sizeof(float)* ARRAYSIZE(g_farrSphereVertices));

	g_pID3D11DeviceContext->Unmap(g_pID3D11Buffer_VertexBufferSpherePosition, 0);

	//-	Create vertex buffer
	/////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////
	//+	Create normal buffer

	ZeroMemory(&BufferDesc, sizeof(BufferDesc));
	BufferDesc.Usage = D3D11_USAGE_DYNAMIC;	//	DirectX prefer dynamic drawing.
	BufferDesc.ByteWidth = sizeof(float)* ARRAYSIZE(g_farrSphereNormals);
	BufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	BufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	hr = g_pID3D11Device->CreateBuffer(
		&BufferDesc,
		NULL,	//	Pass data dynamically.
		&g_pID3D11Buffer_VertexBufferSphereNormal
	);
	if (FAILED(hr))
	{
		log_write("CreateBuffer() failed for vertex buffer.\n");

		return hr;
	}

	//	Copy vertices into above buffer.
	ZeroMemory(&MappedSubresource, sizeof(MappedSubresource));
	g_pID3D11DeviceContext->Map(
		g_pID3D11Buffer_VertexBufferSphereNormal,	//	ID3D11Buffer Inherit from ID3D11Resource 
		0,	//Index number of subresource i.e like 0 = position, 1 = color
		D3D11_MAP_WRITE_DISCARD,	//	specifies the CPU's read and write permissions for a resource.
		0,	//	Flag that specifies what the CPU does when the GPU is busy. This flag is optional.Cpu will wait.
		&MappedSubresource
	);

	CopyMemory(MappedSubresource.pData, g_farrSphereNormals, sizeof(float)* ARRAYSIZE(g_farrSphereNormals));

	g_pID3D11DeviceContext->Unmap(g_pID3D11Buffer_VertexBufferSphereNormal, 0);

	//-	Create color buffer
	/////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////
	//+	Create Index buffer

	ZeroMemory(&BufferDesc, sizeof(BufferDesc));
	BufferDesc.Usage = D3D11_USAGE_DYNAMIC;	//	DirectX prefer dynamic drawing.
	BufferDesc.ByteWidth = sizeof(g_usharrSphereElements[0]) * g_uiNumElements;
	BufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
	BufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	hr = g_pID3D11Device->CreateBuffer(
		&BufferDesc,
		NULL,	//	Pass data dynamically.
		&g_pID3D11Buffer_IndexBuffer
	);
	if (FAILED(hr))
	{
		log_write("CreateBuffer(g_pID3D11Buffer_IndexBuffer) failed for vertex buffer.\n");

		return hr;
	}

	//	Copy vertices into above buffer.
	ZeroMemory(&MappedSubresource, sizeof(MappedSubresource));
	g_pID3D11DeviceContext->Map(
		g_pID3D11Buffer_IndexBuffer,	//	ID3D11Buffer Inherit from ID3D11Resource 
		0,	//Index number of subresource i.e like 0 = position, 1 = color
		D3D11_MAP_WRITE_DISCARD,	//	specifies the CPU's read and write permissions for a resource.
		0,	//	Flag that specifies what the CPU does when the GPU is busy. This flag is optional.Cpu will wait.
		&MappedSubresource
	);

	CopyMemory(MappedSubresource.pData, g_usharrSphereElements, sizeof(g_usharrSphereElements[0]) * g_uiNumElements);

	g_pID3D11DeviceContext->Unmap(g_pID3D11Buffer_IndexBuffer, 0);

	//-	Create Index buffer
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

	g_pID3D11DeviceContext->PSSetConstantBuffers(
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

	//	d3d clear color, analogus to glClearColor() in openGL.
	g_fClearColor[0] = 0.25f;
	g_fClearColor[1] = 0.25f;
	g_fClearColor[2] = 0.25f;
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
	void Sphere00();
	void Sphere10();
	void Sphere20();
	void Sphere30();
	void Sphere40();
	void Sphere50();

	void Sphere01();
	void Sphere11();
	void Sphere21();
	void Sphere31();
	void Sphere41();
	void Sphere51();

	void Sphere02();
	void Sphere12();
	void Sphere22();
	void Sphere32();
	void Sphere42();
	void Sphere52();

	void Sphere03();
	void Sphere13();
	void Sphere23();
	void Sphere33();
	void Sphere43();
	void Sphere53();

	//	Clear render target view to a choosen color.
	g_pID3D11DeviceContext->ClearRenderTargetView(g_pID3D11RenderTargetView, g_fClearColor);

	g_pID3D11DeviceContext->ClearDepthStencilView(g_pID3D11DepthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0);

	float fHeightMulti = 0.05f;
	float fWidthMulti = 0.07f;
	//
	//	Set viewport.
	//
	D3D11_VIEWPORT D3DViewPort;
	ZeroMemory(&D3DViewPort, sizeof(D3DViewPort));
	D3DViewPort.MinDepth = 0.0f;
	D3DViewPort.MaxDepth = 1.0f;

	//
	//	First column,
	//
	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere00();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere10();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere20();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere30();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere40();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere50();

	
	//
	//	Second column,
	//
	fWidthMulti = fWidthMulti + 0.20f;
	fHeightMulti = 0.05f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere01();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere11();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere21();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere31();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere41();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere51();

	//
	//	Third column,
	//
	fWidthMulti = fWidthMulti + 0.20f;
	fHeightMulti = 0.05f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere02();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere12();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere22();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere32();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere42();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere52();

	//
	//	Fourth column,
	//
	fWidthMulti = fWidthMulti + 0.20f;
	fHeightMulti = 0.05f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere03();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere13();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere23();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere33();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere43();

	fHeightMulti = fHeightMulti + 0.15f;

	D3DViewPort.TopLeftX = (g_iCurrentWidth * fWidthMulti);
	D3DViewPort.TopLeftY = (g_iCurrentHeight * fHeightMulti);
	D3DViewPort.Width = (float)g_iCurrentWidth / 4;
	D3DViewPort.Height = (float)g_iCurrentHeight / 6;
	g_pID3D11DeviceContext->RSSetViewports(1, &D3DViewPort);
	Sphere53();

	//	Swap buffers, switch betwwn front and back buffers.
	//	1st parameter - no synchronization with vertical refresh rate(vertical blank). 
	//	2nd parameter - how many frames from how many buffers need to show. 
	g_pIDXGISwapChain->Present(0, 0);
}

void update()
{
	g_fAngleLight = g_fAngleLight + 0.005f;

	if (g_fAngleLight >= 360)
	{
		g_fAngleLight = 0.0f;
	}
}


///////////////////////Column 0 ////////////////////////////////////////////////////////////////////
void Sphere00()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.
																									
	//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial00Ambient[0], g_arrMaterial00Ambient[1], g_arrMaterial00Ambient[2], g_arrMaterial00Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial00Diffuse[0], g_arrMaterial00Diffuse[1], g_arrMaterial00Diffuse[2], g_arrMaterial00Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial00Specular[0], g_arrMaterial00Specular[1], g_arrMaterial00Specular[2], g_arrMaterial00Specular[3]);
		constantBuffer.fMaterialShininess = g_Material00Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere10()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial10Ambient[0], g_arrMaterial10Ambient[1], g_arrMaterial10Ambient[2], g_arrMaterial10Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial10Diffuse[0], g_arrMaterial10Diffuse[1], g_arrMaterial10Diffuse[2], g_arrMaterial10Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial10Specular[0], g_arrMaterial10Specular[1], g_arrMaterial10Specular[2], g_arrMaterial10Specular[3]);
		constantBuffer.fMaterialShininess = g_Material10Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere20()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial20Ambient[0], g_arrMaterial20Ambient[1], g_arrMaterial20Ambient[2], g_arrMaterial20Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial20Diffuse[0], g_arrMaterial20Diffuse[1], g_arrMaterial20Diffuse[2], g_arrMaterial20Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial20Specular[0], g_arrMaterial20Specular[1], g_arrMaterial20Specular[2], g_arrMaterial20Specular[3]);
		constantBuffer.fMaterialShininess = g_Material20Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere30()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial30Ambient[0], g_arrMaterial30Ambient[1], g_arrMaterial30Ambient[2], g_arrMaterial30Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial30Diffuse[0], g_arrMaterial30Diffuse[1], g_arrMaterial30Diffuse[2], g_arrMaterial30Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial30Specular[0], g_arrMaterial30Specular[1], g_arrMaterial30Specular[2], g_arrMaterial30Specular[3]);
		constantBuffer.fMaterialShininess = g_Material30Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere40()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial40Ambient[0], g_arrMaterial40Ambient[1], g_arrMaterial40Ambient[2], g_arrMaterial40Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial40Diffuse[0], g_arrMaterial40Diffuse[1], g_arrMaterial40Diffuse[2], g_arrMaterial40Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial40Specular[0], g_arrMaterial40Specular[1], g_arrMaterial40Specular[2], g_arrMaterial40Specular[3]);
		constantBuffer.fMaterialShininess = g_Material40Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere50()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial50Ambient[0], g_arrMaterial50Ambient[1], g_arrMaterial50Ambient[2], g_arrMaterial50Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial50Diffuse[0], g_arrMaterial50Diffuse[1], g_arrMaterial50Diffuse[2], g_arrMaterial50Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial50Specular[0], g_arrMaterial50Specular[1], g_arrMaterial50Specular[2], g_arrMaterial50Specular[3]);
		constantBuffer.fMaterialShininess = g_Material10Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////
}


///////////////////////Column 1 ////////////////////////////////////////////////////////////////////
void Sphere01()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial01Ambient[0], g_arrMaterial01Ambient[1], g_arrMaterial01Ambient[2], g_arrMaterial01Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial01Diffuse[0], g_arrMaterial01Diffuse[1], g_arrMaterial01Diffuse[2], g_arrMaterial01Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial01Specular[0], g_arrMaterial01Specular[1], g_arrMaterial01Specular[2], g_arrMaterial01Specular[3]);
		constantBuffer.fMaterialShininess = g_Material01Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere11()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial11Ambient[0], g_arrMaterial11Ambient[1], g_arrMaterial11Ambient[2], g_arrMaterial11Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial11Diffuse[0], g_arrMaterial11Diffuse[1], g_arrMaterial11Diffuse[2], g_arrMaterial11Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial11Specular[0], g_arrMaterial11Specular[1], g_arrMaterial11Specular[2], g_arrMaterial11Specular[3]);
		constantBuffer.fMaterialShininess = g_Material11Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere21()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial21Ambient[0], g_arrMaterial21Ambient[1], g_arrMaterial21Ambient[2], g_arrMaterial21Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial21Diffuse[0], g_arrMaterial21Diffuse[1], g_arrMaterial21Diffuse[2], g_arrMaterial21Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial21Specular[0], g_arrMaterial21Specular[1], g_arrMaterial21Specular[2], g_arrMaterial21Specular[3]);
		constantBuffer.fMaterialShininess = g_Material21Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere31()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial31Ambient[0], g_arrMaterial31Ambient[1], g_arrMaterial31Ambient[2], g_arrMaterial31Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial31Diffuse[0], g_arrMaterial31Diffuse[1], g_arrMaterial31Diffuse[2], g_arrMaterial31Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial31Specular[0], g_arrMaterial31Specular[1], g_arrMaterial31Specular[2], g_arrMaterial31Specular[3]);
		constantBuffer.fMaterialShininess = g_Material31Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere41()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial41Ambient[0], g_arrMaterial41Ambient[1], g_arrMaterial41Ambient[2], g_arrMaterial41Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial41Diffuse[0], g_arrMaterial41Diffuse[1], g_arrMaterial41Diffuse[2], g_arrMaterial41Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial41Specular[0], g_arrMaterial41Specular[1], g_arrMaterial41Specular[2], g_arrMaterial41Specular[3]);
		constantBuffer.fMaterialShininess = g_Material41Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere51()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial51Ambient[0], g_arrMaterial51Ambient[1], g_arrMaterial51Ambient[2], g_arrMaterial51Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial51Diffuse[0], g_arrMaterial51Diffuse[1], g_arrMaterial51Diffuse[2], g_arrMaterial51Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial51Specular[0], g_arrMaterial51Specular[1], g_arrMaterial51Specular[2], g_arrMaterial51Specular[3]);
		constantBuffer.fMaterialShininess = g_Material51Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////
}


///////////////////////Column 2 ////////////////////////////////////////////////////////////////////
void Sphere02()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial02Ambient[0], g_arrMaterial02Ambient[1], g_arrMaterial02Ambient[2], g_arrMaterial02Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial02Diffuse[0], g_arrMaterial02Diffuse[1], g_arrMaterial02Diffuse[2], g_arrMaterial02Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial02Specular[0], g_arrMaterial02Specular[1], g_arrMaterial02Specular[2], g_arrMaterial02Specular[3]);
		constantBuffer.fMaterialShininess = g_Material02Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere12()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial12Ambient[0], g_arrMaterial12Ambient[1], g_arrMaterial12Ambient[2], g_arrMaterial12Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial12Diffuse[0], g_arrMaterial12Diffuse[1], g_arrMaterial12Diffuse[2], g_arrMaterial12Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial12Specular[0], g_arrMaterial12Specular[1], g_arrMaterial12Specular[2], g_arrMaterial12Specular[3]);
		constantBuffer.fMaterialShininess = g_Material12Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere22()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial22Ambient[0], g_arrMaterial22Ambient[1], g_arrMaterial22Ambient[2], g_arrMaterial22Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial22Diffuse[0], g_arrMaterial22Diffuse[1], g_arrMaterial22Diffuse[2], g_arrMaterial22Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial22Specular[0], g_arrMaterial22Specular[1], g_arrMaterial22Specular[2], g_arrMaterial22Specular[3]);
		constantBuffer.fMaterialShininess = g_Material22Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere32()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial32Ambient[0], g_arrMaterial32Ambient[1], g_arrMaterial32Ambient[2], g_arrMaterial32Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial32Diffuse[0], g_arrMaterial32Diffuse[1], g_arrMaterial32Diffuse[2], g_arrMaterial32Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial32Specular[0], g_arrMaterial32Specular[1], g_arrMaterial32Specular[2], g_arrMaterial32Specular[3]);
		constantBuffer.fMaterialShininess = g_Material32Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere42()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial42Ambient[0], g_arrMaterial42Ambient[1], g_arrMaterial42Ambient[2], g_arrMaterial42Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial42Diffuse[0], g_arrMaterial42Diffuse[1], g_arrMaterial42Diffuse[2], g_arrMaterial42Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial42Specular[0], g_arrMaterial42Specular[1], g_arrMaterial42Specular[2], g_arrMaterial42Specular[3]);
		constantBuffer.fMaterialShininess = g_Material42Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere52()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial52Ambient[0], g_arrMaterial52Ambient[1], g_arrMaterial52Ambient[2], g_arrMaterial52Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial52Diffuse[0], g_arrMaterial52Diffuse[1], g_arrMaterial52Diffuse[2], g_arrMaterial52Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial52Specular[0], g_arrMaterial52Specular[1], g_arrMaterial52Specular[2], g_arrMaterial52Specular[3]);
		constantBuffer.fMaterialShininess = g_Material52Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////
}


///////////////////////Column 3 ////////////////////////////////////////////////////////////////////
void Sphere03()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial03Ambient[0], g_arrMaterial03Ambient[1], g_arrMaterial03Ambient[2], g_arrMaterial03Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial03Diffuse[0], g_arrMaterial03Diffuse[1], g_arrMaterial03Diffuse[2], g_arrMaterial03Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial03Specular[0], g_arrMaterial03Specular[1], g_arrMaterial03Specular[2], g_arrMaterial03Specular[3]);
		constantBuffer.fMaterialShininess = g_Material03Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere13()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial13Ambient[0], g_arrMaterial13Ambient[1], g_arrMaterial13Ambient[2], g_arrMaterial13Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial13Diffuse[0], g_arrMaterial13Diffuse[1], g_arrMaterial13Diffuse[2], g_arrMaterial13Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial13Specular[0], g_arrMaterial13Specular[1], g_arrMaterial13Specular[2], g_arrMaterial13Specular[3]);
		constantBuffer.fMaterialShininess = g_Material13Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere23()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial23Ambient[0], g_arrMaterial23Ambient[1], g_arrMaterial23Ambient[2], g_arrMaterial23Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial23Diffuse[0], g_arrMaterial23Diffuse[1], g_arrMaterial23Diffuse[2], g_arrMaterial23Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial23Specular[0], g_arrMaterial23Specular[1], g_arrMaterial23Specular[2], g_arrMaterial23Specular[3]);
		constantBuffer.fMaterialShininess = g_Material23Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere33()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial33Ambient[0], g_arrMaterial33Ambient[1], g_arrMaterial33Ambient[2], g_arrMaterial33Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial33Diffuse[0], g_arrMaterial33Diffuse[1], g_arrMaterial33Diffuse[2], g_arrMaterial33Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial33Specular[0], g_arrMaterial33Specular[1], g_arrMaterial33Specular[2], g_arrMaterial33Specular[3]);
		constantBuffer.fMaterialShininess = g_Material33Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere43()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial43Ambient[0], g_arrMaterial43Ambient[1], g_arrMaterial43Ambient[2], g_arrMaterial43Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial43Diffuse[0], g_arrMaterial43Diffuse[1], g_arrMaterial43Diffuse[2], g_arrMaterial43Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial43Specular[0], g_arrMaterial43Specular[1], g_arrMaterial43Specular[2], g_arrMaterial43Specular[3]);
		constantBuffer.fMaterialShininess = g_Material43Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

}


void Sphere53()
{
	UINT uiStride;
	UINT uiOffset;
	XMMATRIX wvMatrix;
	XMMATRIX viewMatrix;
	XMMATRIX worldMatrix;
	CBUFFER constantBuffer;
	XMMATRIX rotationMatrix;

	/////////////////////////////////////////////////////////////////////////////////////////
	//+ Draw Square
	uiStride = sizeof(float) * 3;
	uiOffset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_POSITION,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSpherePosition,
		&uiStride,
		&uiOffset
	);

	g_pID3D11DeviceContext->IASetVertexBuffers(
		RTR_INPUT_SLOT_NORMAL,	//	0 - Position, 1 - color
		1,
		&g_pID3D11Buffer_VertexBufferSphereNormal,
		&uiStride,
		&uiOffset
	);

	//	Set index buffer.
	g_pID3D11DeviceContext->IASetIndexBuffer(g_pID3D11Buffer_IndexBuffer, DXGI_FORMAT_R16_UINT, 0);	//	R16 maps with short.

																									//	Select geometry primitive.
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();

	worldMatrix = worldMatrix * XMMatrixTranslation(0.0f, 0.0f, 3.0f);

	//	World view projection matrix.
	//wvMatrix = worldMatrix * viewMatrix;

	//	Load the data to constant buffer.
	ZeroMemory(&constantBuffer, sizeof(constantBuffer));
	constantBuffer.worldMatrix = worldMatrix;
	constantBuffer.viewMatrix = viewMatrix;
	constantBuffer.projectionMatrix = g_PerspectiveProjectionMatrix;

	if (true == g_bLight)
	{
		if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationX(g_fAngleLight);
			g_farrLightPosition[1] = g_fAngleLight;
			g_farrLightPosition[0] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);

		}
		else if ('Y' == chAnimationAxis || 'Y' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationY(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}
		else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
		{
			constantBuffer.rotationMatrix = XMMatrixRotationZ(g_fAngleLight);
			g_farrLightPosition[0] = g_fAngleLight;
			g_farrLightPosition[1] = g_farrLightPosition[2] = g_farrLightPosition[3] = 0.0f;
			constantBuffer.vecLightPosition = XMVectorSet(g_farrLightPosition[0], g_farrLightPosition[1], g_farrLightPosition[2], g_farrLightPosition[3]);
		}

		constantBuffer.vecLA = XMVectorSet(g_farrLightAmbient[0], g_farrLightAmbient[1], g_farrLightAmbient[2], g_farrLightAmbient[3]);
		constantBuffer.vecLD = XMVectorSet(g_farrLightDiffuse[0], g_farrLightDiffuse[1], g_farrLightDiffuse[2], g_farrLightDiffuse[3]);
		constantBuffer.vecLS = XMVectorSet(g_farrLightSpecular[0], g_farrLightSpecular[1], g_farrLightSpecular[2], g_farrLightSpecular[3]);

		constantBuffer.vecKA = XMVectorSet(g_arrMaterial53Ambient[0], g_arrMaterial53Ambient[1], g_arrMaterial53Ambient[2], g_arrMaterial53Ambient[3]);
		constantBuffer.vecKD = XMVectorSet(g_arrMaterial53Diffuse[0], g_arrMaterial53Diffuse[1], g_arrMaterial53Diffuse[2], g_arrMaterial53Diffuse[3]);
		constantBuffer.vecKS = XMVectorSet(g_arrMaterial53Specular[0], g_arrMaterial53Specular[1], g_arrMaterial53Specular[2], g_arrMaterial53Specular[3]);
		constantBuffer.fMaterialShininess = g_Material53Shininess;

		constantBuffer.uiKeyPressed = 1;
	}
	else
	{
		constantBuffer.uiKeyPressed = 0;
	}

	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	//	Draw
	g_pID3D11DeviceContext->DrawIndexed(g_uiNumElements, 0, 0);

	//- Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////
}

VOID uninitialize()
{
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

	if (g_pID3D11Buffer_IndexBuffer)
	{
		g_pID3D11Buffer_IndexBuffer->Release();
		g_pID3D11Buffer_IndexBuffer = NULL;
	}

	if (g_pID3D11Buffer_VertexBufferSpherePosition)
	{
		g_pID3D11Buffer_VertexBufferSpherePosition->Release();
		g_pID3D11Buffer_VertexBufferSpherePosition = NULL;
	}

	if (g_pID3D11Buffer_VertexBufferSphereNormal)
	{
		g_pID3D11Buffer_VertexBufferSphereNormal->Release();
		g_pID3D11Buffer_VertexBufferSphereNormal = NULL;
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