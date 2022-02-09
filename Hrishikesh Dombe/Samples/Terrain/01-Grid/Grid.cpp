#include<Windows.h>
#include<windowsx.h>
#include<stdio.h>
#include<d3d11.h>
#include<d3dcompiler.h>
#include <DirectXMath.h>
#include "Vertices.h"
#include"..\..\..\..\Utilities\Camera\Camera.h"
#include"..\..\..\..\Utilities\Timer\Timer.h"
// #include"../../../../Utilities/Camera/Camera.h"
// #include"../../../../Utilities/Timer/Timer.h"

#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"D3dcompiler.lib")
#pragma comment(lib,"user32.lib")
#pragma comment(lib,"gdi32.lib")

using namespace DirectX;

#define WIN_WIDTH 800
#define	WIN_HEIGHT 600

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

FILE *gpFile = NULL;
char gszLogFileName[] = "Log.txt";

HWND ghwnd = NULL;

DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

bool gbActiveWindow = false;
bool gbIsEscapeKeyPressed = false;
bool gbFullscreen = false;

float gClearColor[4];
IDXGISwapChain *gpIDXGISwapChain = NULL;
ID3D11Device *gpID3D11Device = NULL;
ID3D11DeviceContext *gpID3D11DeviceContext = NULL;
ID3D11RenderTargetView *gpID3D11RenderTargetView = NULL;

ID3D11VertexShader *gpID3D11VertexShader = NULL;
ID3D11PixelShader *gpID3D11PixelShader = NULL;
ID3D11Buffer *gpID3D11Buffer_VertexBuffer = NULL;
ID3D11InputLayout *gpID3D11InputLayout = NULL;
ID3D11Buffer *gpID3D11Buffer_ConstantBuffer = NULL;

struct CBUFFER
{
	XMMATRIX WorldViewProjectionMatrix;
};

XMMATRIX gPerspectiveProjectionMatrix;

int giMouseLastXPos;
int giMouseLastYPos;
Camera mCam;
Timer mTimer;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	HRESULT initialize(void);
	void uninitialize(void);
	void update(void);
	void display(void);

	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("My D3D11 Window");
	bool bDone = false;

	if (fopen_s(&gpFile, gszLogFileName, "w") != 0)
	{
		MessageBox(NULL, TEXT("Log File Cannot be Created\nExitting."), TEXT("ERROR"), MB_OK);
		exit(EXIT_FAILURE);
	}
	else
	{
		fprintf_s(gpFile, "Log File Created Successfully");
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

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, TEXT("D3D11 Perspective Triangle"), WS_OVERLAPPEDWINDOW, 100, 100, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);

	ghwnd = hwnd;

	ShowWindow(hwnd, iCmdShow);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	HRESULT hr;

	hr = initialize();
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "Initialize() Failed. Exitting Now.\n");
		fclose(gpFile);
		uninitialize();
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "Initialize() Failed. Exitting Now.\n");
		fclose(gpFile);
	}

	mTimer.Reset();
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
				mTimer.Tick();
				if (gbIsEscapeKeyPressed == true)
					bDone = true;
				display();
				update();
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

	int x_Pos;
	int y_Pos;

	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
		{
			gbActiveWindow = true;
			mTimer.Start();
		}
		else
		{
			gbActiveWindow = false;
			mTimer.Stop();
		}
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
			if (gbIsEscapeKeyPressed == false)
				gbIsEscapeKeyPressed = true;
			else
				gbIsEscapeKeyPressed = false;
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

		// case 0x41://A
		// 	mCam.Strafe(-10.0f*mTimer.DeltaTime());
		// 	break;

		// case 0x44://D
		// 	mCam.Strafe(10.0f*mTimer.DeltaTime());
		// 	break;

		// case 0x53://S
		// 	mCam.Walk(-10.0f*mTimer.DeltaTime());
		// 	break;

		// case 0x57://W
		// 	mCam.Walk(10.0f*mTimer.DeltaTime());
		// 	break;

		default:
			break;
		}
		break;
	
	case WM_MOUSEMOVE:
		x_Pos = GET_X_LPARAM(lParam);
		y_Pos = GET_Y_LPARAM(lParam);
		if ((wParam & MK_LBUTTON) != 0)
		{
			float dx = XMConvertToRadians(0.25f*static_cast<float>(x_Pos - giMouseLastXPos));
			float dy = XMConvertToRadians(0.25f*static_cast<float>(y_Pos - giMouseLastYPos));

			mCam.Pitch(dy);
			mCam.RotateY(dx);
		}
		giMouseLastXPos = x_Pos;
		giMouseLastYPos = y_Pos;
		break;

	case WM_LBUTTONDOWN:
		break;

	case WM_CLOSE:
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
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghwnd, GWL_STYLE, dwStyle&~WS_OVERLAPPEDWINDOW);
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
	void uninitialize(void);
	HRESULT resize(int, int);

	HRESULT hr;

	D3D_DRIVER_TYPE d3dDriverType;
	D3D_DRIVER_TYPE d3dDriverTypes[] = { D3D_DRIVER_TYPE_HARDWARE,D3D_DRIVER_TYPE_WARP,D3D_DRIVER_TYPE_REFERENCE };
	D3D_FEATURE_LEVEL d3dFeature_Level_required = D3D_FEATURE_LEVEL_11_0;
	D3D_FEATURE_LEVEL d3dFeature_Level_acquired = D3D_FEATURE_LEVEL_10_0;
	UINT createDeviceFlags = 0;
	UINT numDriverTypes = 0;
	UINT numFeatureLevels = 1;

	numDriverTypes = sizeof(d3dDriverTypes) / sizeof(d3dDriverTypes[0]);

	fopen_s(&gpFile, gszLogFileName, "a+");
	fprintf_s(gpFile, "NumDriverTypes = %d.\n", numDriverTypes);
	fclose(gpFile);

	DXGI_SWAP_CHAIN_DESC dxgiSwapChainDesc;
	ZeroMemory((void*)&dxgiSwapChainDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
	dxgiSwapChainDesc.BufferCount = 1;
	dxgiSwapChainDesc.BufferDesc.Width = WIN_WIDTH;
	dxgiSwapChainDesc.BufferDesc.Height = WIN_HEIGHT;
	dxgiSwapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	dxgiSwapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
	dxgiSwapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
	dxgiSwapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	dxgiSwapChainDesc.OutputWindow = ghwnd;
	dxgiSwapChainDesc.SampleDesc.Count = 1;
	dxgiSwapChainDesc.SampleDesc.Quality = 0;
	dxgiSwapChainDesc.Windowed = TRUE;

	for(UINT driverTypeIndex = 0;driverTypeIndex < numDriverTypes; driverTypeIndex++)
	{
		d3dDriverType = d3dDriverTypes[driverTypeIndex];
		hr = D3D11CreateDeviceAndSwapChain(
			NULL,//Pointer to Video Adapter, Means pointer device to which display is connected
			d3dDriverType,
			NULL,//If Driver Type is D3D_DRIVER_TYPE_SOFTWARE, then we need to provide Handle to DLL which has implemented Software Render
			createDeviceFlags,//D3D11_CREATE_DEVICE_SINGLETHREADED
			&d3dFeature_Level_required,
			numFeatureLevels,
			D3D11_SDK_VERSION,
			&dxgiSwapChainDesc,
			&gpIDXGISwapChain,
			&gpID3D11Device,
			&d3dFeature_Level_acquired,
			&gpID3D11DeviceContext);

		if(SUCCEEDED(hr))
			break;
	}

	if(FAILED(hr))
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
		fprintf_s(gpFile, "The Chosen Driver Is Of");

		if(d3dDriverType == D3D_DRIVER_TYPE_HARDWARE)
		{
			fprintf_s(gpFile, "Hardware Type.\n");
		}
		else if(d3dDriverType == D3D_DRIVER_TYPE_WARP)
		{
			fprintf_s(gpFile, "Warp Type.\n");
		}
		else if(d3dDriverType == D3D_DRIVER_TYPE_REFERENCE)
		{
			fprintf_s(gpFile, "Reference Type.\n");
		}
		else
		{
			fprintf_s(gpFile,"Unknown Type.\n");
		}

		fprintf_s(gpFile, "The Supported Highest Feature Level Is");
		if(d3dFeature_Level_acquired == D3D_FEATURE_LEVEL_11_0)
		{
			fprintf_s(gpFile, "11.0\n");
		}
		else if(d3dFeature_Level_acquired == D3D_FEATURE_LEVEL_10_1)
		{
			fprintf_s(gpFile, "10.1\n");
		}
		else if(d3dFeature_Level_acquired == D3D_FEATURE_LEVEL_10_0)
		{
			fprintf_s(gpFile, "10.0\n");
		}
		else
		{
			fprintf_s(gpFile,"Unknown.\n");
		}

		fclose(gpFile);
	}

	const char *vertexShaderSourceCode =
		"cbuffer ConstantBuffer" \
		"{" \
		"float4x4 worldViewProjectionMatrix;" \
		"}" \
		"float4 main(float4 pos : POSITION) : SV_POSITION" \
		"{" \
		"float4 position = mul(worldViewProjectionMatrix, pos);" \
		"return(position);" \
		"}";

	ID3DBlob *pID3DBlob_VertexShaderCode = NULL;
	ID3DBlob *pID3DBlob_Error = NULL;

	hr = D3DCompile(vertexShaderSourceCode, lstrlenA(vertexShaderSourceCode) + 1, "VS", NULL, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "vs_5_0", 0, 0, &pID3DBlob_VertexShaderCode, &pID3DBlob_Error);
	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf_s(gpFile, "D3DCompile() Failed For Vertex Shader : %s.\n", (char *)pID3DBlob_Error->GetBufferPointer());
			fclose(gpFile);
			// pID3DBlob_VertexShaderCode->Release();
			// pID3DBlob_VertexShaderCode = NULL;
			pID3DBlob_Error->Release();
			pID3DBlob_Error = NULL;
			return(hr);
		}
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "D3DCompile() Succeeded For Vertex Shader.\n");
		fclose(gpFile);
	}
	
	hr = gpID3D11Device->CreateVertexShader(pID3DBlob_VertexShaderCode->GetBufferPointer(), pID3DBlob_VertexShaderCode->GetBufferSize(), NULL, &gpID3D11VertexShader);
	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf_s(gpFile, "CreateVertexShader() Failed.\n");
			fclose(gpFile);
			pID3DBlob_VertexShaderCode->Release();
			pID3DBlob_VertexShaderCode = NULL;
			pID3DBlob_Error->Release();
			pID3DBlob_Error = NULL;
			return(hr);
		}
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "CreateVertexShader() Succeeded.\n");
		fclose(gpFile);
	}

	gpID3D11DeviceContext->VSSetShader(gpID3D11VertexShader, 0, 0);

	const char *pixelShaderSourceCode =
		"float4 main(void) : SV_TARGET" \
		"{" \
		"return(float4(1.0f,1.0f,1.0f,1.0f));" \
		"}";

	ID3DBlob *pID3DBlob_PixelShaderCode = NULL;
	pID3DBlob_Error = NULL;

	hr = D3DCompile(pixelShaderSourceCode, lstrlenA(pixelShaderSourceCode) + 1, "PS", NULL, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "ps_5_0", 0, 0, &pID3DBlob_PixelShaderCode, &pID3DBlob_Error);
	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf_s(gpFile, "D3DCompile() Failed For Pixel Shader : %s.\n", (char *)pID3DBlob_Error->GetBufferPointer());
			fclose(gpFile);
			pID3DBlob_PixelShaderCode->Release();
			pID3DBlob_PixelShaderCode = NULL;
			pID3DBlob_VertexShaderCode->Release();
			pID3DBlob_VertexShaderCode = NULL;
			pID3DBlob_Error->Release();
			pID3DBlob_Error = NULL;
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
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf_s(gpFile, "CreatePixelShader() Failed.\n");
			fclose(gpFile);
			pID3DBlob_PixelShaderCode->Release();
			pID3DBlob_PixelShaderCode = NULL;
			pID3DBlob_VertexShaderCode->Release();
			pID3DBlob_VertexShaderCode = NULL;
			pID3DBlob_Error->Release();
			pID3DBlob_Error = NULL;
			return(hr);
		}
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "CreatePixelShader() Succeeded.\n");
		fclose(gpFile);
	}

	gpID3D11DeviceContext->PSSetShader(gpID3D11PixelShader, 0, 0);

	D3D11_INPUT_ELEMENT_DESC inputElementDesc;
	ZeroMemory(&inputElementDesc, sizeof(D3D11_INPUT_ELEMENT_DESC));
	inputElementDesc.SemanticName = "POSITION";
	inputElementDesc.SemanticIndex = 0;
	inputElementDesc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
	inputElementDesc.InputSlot = 0;
	inputElementDesc.AlignedByteOffset = 0;
	inputElementDesc.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	inputElementDesc.InstanceDataStepRate = 0;

	hr = gpID3D11Device->CreateInputLayout(&inputElementDesc, 1, pID3DBlob_VertexShaderCode->GetBufferPointer(), pID3DBlob_VertexShaderCode->GetBufferSize(), &gpID3D11InputLayout);
	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf_s(gpFile, "ID3D11Device::CreateInputLayout() Failed.\n");
			fclose(gpFile);
			pID3DBlob_PixelShaderCode->Release();
			pID3DBlob_PixelShaderCode = NULL;
			pID3DBlob_VertexShaderCode->Release();
			pID3DBlob_VertexShaderCode = NULL;
			pID3DBlob_Error->Release();
			pID3DBlob_Error = NULL;
			return(hr);
		}
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
	// pID3DBlob_Error->Release();
	// pID3DBlob_Error = NULL;

	//Left Hand Thumb Rule
	float vertices[] =
	{
		0.0f,1.0f,0.0f,//Apex
		1.0f,-1.0f,0.0f,//Right
		-1.0f,-1.0f,0.0f,//Left
	};

	D3D11_BUFFER_DESC bufferDesc;
	ZeroMemory(&bufferDesc, sizeof(D3D11_BUFFER_DESC));
	bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc.ByteWidth = sizeof(float)*ARRAYSIZE(gfVertices);
	bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;


	hr = gpID3D11Device->CreateBuffer(&bufferDesc, NULL, &gpID3D11Buffer_VertexBuffer);
	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Failed.\n");
			fclose(gpFile);
			return(hr);
		}
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Succeeded.\n");
		fclose(gpFile);
	}

	D3D11_MAPPED_SUBRESOURCE mappedSubresource;
	ZeroMemory(&mappedSubresource, sizeof(D3D11_MAPPED_SUBRESOURCE));
	gpID3D11DeviceContext->Map(gpID3D11Buffer_VertexBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource);
	memcpy(mappedSubresource.pData, gfVertices, sizeof(gfVertices));
	gpID3D11DeviceContext->Unmap(gpID3D11Buffer_VertexBuffer, 0);

	D3D11_BUFFER_DESC bufferDesc_ConstantBuffer;
	ZeroMemory(&bufferDesc_ConstantBuffer,sizeof(D3D11_BUFFER_DESC));
	bufferDesc_ConstantBuffer.Usage=D3D11_USAGE_DEFAULT;
	bufferDesc_ConstantBuffer.ByteWidth=sizeof(CBUFFER);
	bufferDesc_ConstantBuffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER;

	hr=gpID3D11Device->CreateBuffer(&bufferDesc_ConstantBuffer,nullptr,&gpID3D11Buffer_ConstantBuffer);
	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile, gszLogFileName, "a+");
			fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Failed For Constant Buffer.\n");
			fclose(gpFile);
			return(hr);
		}
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Succeeded For Constant Buffer.\n");
		fclose(gpFile);
	}

	gpID3D11DeviceContext->VSSetConstantBuffers(0,1,&gpID3D11Buffer_ConstantBuffer);

	gClearColor[0] = 0.0f;
	gClearColor[1] = 0.0f;
	gClearColor[2] = 0.0f;
	gClearColor[3] = 1.0f;

	gPerspectiveProjectionMatrix = XMMatrixIdentity();

	hr = resize(WIN_WIDTH, WIN_HEIGHT);
	if(FAILED(hr))
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
	if(gpID3D11RenderTargetView)
	{
		gpID3D11RenderTargetView->Release();
		gpID3D11RenderTargetView=NULL;
	}

	//Resize the Swap Chain Buffer
	gpIDXGISwapChain->ResizeBuffers(1,width,height,DXGI_FORMAT_R8G8B8A8_UNORM,0);

	ID3D11Texture2D *pID3D11Texture2D_BackBuffer;
	//Again Get the Back Buffer From Swap Chain
	gpIDXGISwapChain->GetBuffer(0,__uuidof(ID3D11Texture2D),(LPVOID*)&pID3D11Texture2D_BackBuffer);

	hr = gpID3D11Device->CreateRenderTargetView(pID3D11Texture2D_BackBuffer,NULL,&gpID3D11RenderTargetView);
	if(FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "gpID3D11Device::CreateRenderTargetView() Failed.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "gpID3D11Device::CreateRenderTargetView() Succeeded.\n");
		fclose(gpFile);	
	}
	pID3D11Texture2D_BackBuffer->Release();
	pID3D11Texture2D_BackBuffer = NULL;

	//Set Render Target View as Render Target
	gpID3D11DeviceContext->OMSetRenderTargets(1,&gpID3D11RenderTargetView,NULL);

	//Set Viewport
	D3D11_VIEWPORT d3dViewPort;
	d3dViewPort.TopLeftX = 0;
	d3dViewPort.TopLeftY = 0;
	d3dViewPort.Width = (float)width;
	d3dViewPort.Height = (float)height;
	d3dViewPort.MinDepth = 0.0f;
	d3dViewPort.MaxDepth = 1.0f;
	gpID3D11DeviceContext->RSSetViewports(1, &d3dViewPort);

	mCam.SetLens(0.25f * XM_PI, (float)width / (float)height, 1.0f, 1000.0f);
	//gPerspectiveProjectionMatrix = XMMatrixPerspectiveFovLH(XMConvertToRadians(45.0f),(float)width/(float)height,0.1f,100.0f);

	return(hr);
}

void display(void)
{
	//Clear Render Target View to chosen color
	gpID3D11DeviceContext->ClearRenderTargetView(gpID3D11RenderTargetView,gClearColor);

	UINT stride = sizeof(float) * 3;
	UINT offset = 0;

	gpID3D11DeviceContext->IASetVertexBuffers(0,1,&gpID3D11Buffer_VertexBuffer,&stride,&offset);

	gpID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);

	XMMATRIX worldMatrix = XMMatrixIdentity();
	XMMATRIX viewMatrix = XMMatrixIdentity();
	XMMATRIX translationMatrix = XMMatrixIdentity();

	worldMatrix = XMMatrixTranslation(-50.0f,-2.0f,3.0f);

	mCam.UpdateViewMatrix();
	viewMatrix = mCam.View();
	gPerspectiveProjectionMatrix = mCam.Projection();

	XMMATRIX wvpMatrix = worldMatrix * viewMatrix * gPerspectiveProjectionMatrix;

	CBUFFER constantBuffer;
	constantBuffer.WorldViewProjectionMatrix = wvpMatrix;

	gpID3D11DeviceContext->UpdateSubresource(gpID3D11Buffer_ConstantBuffer,0,NULL,&constantBuffer,0,0);

	gpID3D11DeviceContext->Draw(Vertex_Count,0);

	//Switch between Front and Back Buffers
	gpIDXGISwapChain->Present(0,0);
}

void update(void)
{
	// if(GetAsyncKeyState(0x41))//A
	// 	mCam.Strafe(-10.0f*deltaTime);

	// if(GetAsyncKeyState(0x44))//D
	// 	mCam.Strafe(10.0f*deltaTime);

	// if(GetAsyncKeyState(0x53))//S
	// 	mCam.Walk(-10.0f*deltaTime);

	// if(GetAsyncKeyState(0x57))//W
	// 	mCam.Walk(10.0f*deltaTime);

	mCam.UpdateCameraPosition(mTimer.DeltaTime());
}

void uninitialize(void)
{
	if(gpID3D11Buffer_ConstantBuffer)
	{
		gpID3D11Buffer_ConstantBuffer->Release();
		gpID3D11Buffer_ConstantBuffer = NULL;
	}

	if(gpID3D11InputLayout)
	{
		gpID3D11InputLayout->Release();
		gpID3D11InputLayout=NULL;
	}

	if(gpID3D11Buffer_VertexBuffer)
	{
		gpID3D11Buffer_VertexBuffer->Release();
		gpID3D11Buffer_VertexBuffer=NULL;
	}

	if(gpID3D11PixelShader)
	{
		gpID3D11PixelShader->Release();
		gpID3D11PixelShader=NULL;
	}

	if(gpID3D11VertexShader)
	{
		gpID3D11VertexShader->Release();
		gpID3D11VertexShader=NULL;
	}

	if(gpID3D11RenderTargetView)
	{
		gpID3D11RenderTargetView->Release();
		gpID3D11RenderTargetView = NULL;
	}

	if(gpIDXGISwapChain)
	{
		gpIDXGISwapChain->Release();
		gpIDXGISwapChain = NULL;
	}

	if(gpID3D11DeviceContext)
	{
		gpID3D11DeviceContext->Release();
		gpID3D11DeviceContext = NULL;
	}

	if(gpID3D11Device)
	{
		gpID3D11Device->Release();
		gpID3D11Device = NULL;
	}

	if(gpFile)
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "uninitialize() Succeeded.\n");
		fprintf_s(gpFile, "Log File Is Successfully Closed.\n");
		fclose(gpFile);
	}
}
