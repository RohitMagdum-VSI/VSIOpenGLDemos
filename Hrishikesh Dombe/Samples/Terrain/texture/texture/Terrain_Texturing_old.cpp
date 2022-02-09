#include<Windows.h>
#include<windowsx.h>
#include<stdio.h>
#include<d3d11.h>
#include<fstream>
#include<d3dcompiler.h>
#include <DirectXMath.h>
#include"Camera.h"
#include"Timer.h"
#include"Tga_Loader.h"

#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"D3dcompiler.lib")
#pragma comment(lib,"user32.lib")
#pragma comment(lib,"gdi32.lib")

using namespace DirectX;
using namespace std;

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
bool gbWireframe = false;

float gClearColor[4];
IDXGISwapChain *gpIDXGISwapChain = NULL;
ID3D11Device *gpID3D11Device = NULL;
ID3D11DeviceContext *gpID3D11DeviceContext = NULL;
ID3D11RenderTargetView *gpID3D11RenderTargetView = NULL;

ID3D11VertexShader *gpID3D11VertexShader = NULL;
ID3D11PixelShader *gpID3D11PixelShader = NULL;
ID3D11Buffer *gpID3D11Buffer_VertexBuffer = NULL;
ID3D11Buffer *gpID3D11Buffer_IndexBuffer = NULL;
ID3D11InputLayout *gpID3D11InputLayout = NULL;
ID3D11Buffer *gpID3D11Buffer_ConstantBuffer = NULL;
ID3D11RasterizerState *gpID3D11RasterizerState = NULL;
ID3D11DepthStencilView *gpID3D11DepthStencilView = NULL;
D3D11_RASTERIZER_DESC gRasterizerDesc;
ID3D11ShaderResourceView *gpID3D11ShaderResourceView_Texture = NULL;
ID3D11SamplerState *gpID3D11SamplerState_Texture = NULL;

struct CBUFFER
{
	XMMATRIX WorldViewProjectionMatrix;
};

XMMATRIX gPerspectiveProjectionMatrix;

struct HeightMapType
{
	float x,y,z;
};

struct ModelType
{
	float x,y,z;
	float tu, tv;
};

struct VertexType
{
	XMFLOAT4 position;
	XMFLOAT2 texture;
};

int giTerrainHeight, giTerrainWidth;
float gfHeightScale;
char *gTerrainFilename;
HeightMapType *gHeightMap;
ModelType *gTerrainModel;
int giVertex_Count, giIndex_Count;
VertexType *gfVertices;
unsigned long *gulIndices;

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
		fprintf_s(gpFile, "Initialize()Succeeded.\n");
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

		case 0x51:
			if(gbWireframe == false)
			{
				gRasterizerDesc.FillMode = D3D11_FILL_WIREFRAME;
				gbWireframe = true;
			}
			else
			{
				gRasterizerDesc.FillMode = D3D11_FILL_SOLID;
				gbWireframe = false;
			}
			hr = gpID3D11Device->CreateRasterizerState(&gRasterizerDesc,&gpID3D11RasterizerState);
			if (FAILED(hr))
			{
				fopen_s(&gpFile, gszLogFileName, "a+");
				fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Failed For CreateRasterizerState.\n");
				fclose(gpFile);
				return(hr);
			}
			else
			{
				fopen_s(&gpFile, gszLogFileName, "a+");
				fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Succeeded For CreateRasterizerState.\n");
				fclose(gpFile);
			}

			gpID3D11DeviceContext->RSSetState(gpID3D11RasterizerState);
			break;

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
	HRESULT LoadD3DTexture(const wchar_t *, ID3D11ShaderResourceView **);
	HRESULT resize(int, int);
	bool InitializeTerrain(char*);

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
	dxgiSwapChainDesc.SampleDesc.Count = 4;
	dxgiSwapChainDesc.SampleDesc.Quality = 1;
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
		"struct Vertex_Input" \
		"{" \
		"float4 position : POSITION;" \
		"float2 texcoord : TEXCOORD0;" \
		"};" \
		"struct Pixel_Input" \
		"{" \
		"float4 position : SV_POSITION;" \
		"float2 texcoord : TEXCOORD0;" \
		"};" \
		"Pixel_Input main(Vertex_Input input)" \
		"{" \
		"Pixel_Input output;" \
		"output.position = mul(worldViewProjectionMatrix, input.position);" \
		"output.texcoord = input.texcoord;" \
		"return(output);" \
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
		"struct Pixel_Input" \
		"{" \
		"float4 position : SV_POSITION;" \
		"float2 texcoord : TEXCOORD0;" \
		"};" \
		"Texture2D myTexture2D;" \
		"SamplerState mySamplerState;" \
		"float4 main(Pixel_Input input) : SV_TARGET" \
		"{" \
		//"float4 color = myTexture2D.Sample(mySamplerState,input.texcoord);" 
		"float4 color = float4(input.texcoord,1.0f,1.0f);" \
		"return(color);" \
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

	D3D11_INPUT_ELEMENT_DESC inputElementDesc[2];
	ZeroMemory(inputElementDesc, sizeof(D3D11_INPUT_ELEMENT_DESC));
	inputElementDesc[0].SemanticName = "POSITION";
	inputElementDesc[0].SemanticIndex = 0;
	inputElementDesc[0].Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	inputElementDesc[0].InputSlot = 0;
	inputElementDesc[0].AlignedByteOffset = 0;
	inputElementDesc[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	inputElementDesc[0].InstanceDataStepRate = 0;

	inputElementDesc[1].SemanticName = "TEXCOORD0";
	inputElementDesc[1].SemanticIndex = 0;
	inputElementDesc[1].Format = DXGI_FORMAT_R32G32_FLOAT;
	inputElementDesc[1].InputSlot = 0;
	inputElementDesc[1].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;//Also can use 12 as there are 3 vertices and they are float
	inputElementDesc[1].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	inputElementDesc[1].InstanceDataStepRate = 0;

	hr = gpID3D11Device->CreateInputLayout(inputElementDesc, 2, pID3DBlob_VertexShaderCode->GetBufferPointer(), pID3DBlob_VertexShaderCode->GetBufferSize(), &gpID3D11InputLayout);
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

	const char str[] = "setup.txt";
	bool Result = InitializeTerrain((char*)str);
	if(Result != true)
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "InitializeTerrain() Succeeded.\n");
		fclose(gpFile);
		return(E_FAIL);
	}

	giVertex_Count = (giTerrainWidth - 1) * (giTerrainHeight - 1) * 6;
	giIndex_Count = giVertex_Count;

	gfVertices = new VertexType[giVertex_Count];

	gulIndices = new unsigned long[giIndex_Count];

	int i;
	for(i=0;i<giVertex_Count;i++)
	{
		gfVertices[i].position = XMFLOAT4(gTerrainModel[i].x,gTerrainModel[i].y,gTerrainModel[i].z,1.0f);
		//gfVertices[i].texture = XMFLOAT2(gTerrainModel[i].tu,gTerrainModel[i].tv);
		gfVertices[i].texture = XMFLOAT2(1.0f, 1.0f);
		gulIndices[i] = i;
	}

	D3D11_BUFFER_DESC bufferDesc;
	ZeroMemory(&bufferDesc, sizeof(D3D11_BUFFER_DESC));
	bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc.ByteWidth = sizeof(VertexType)*(giVertex_Count);
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
	memcpy(mappedSubresource.pData, gfVertices, sizeof(VertexType)*(giVertex_Count));
	gpID3D11DeviceContext->Unmap(gpID3D11Buffer_VertexBuffer, 0);

	if(gfVertices)
	{
		delete gfVertices;
		gfVertices = NULL;
	}

	ZeroMemory(&bufferDesc,sizeof(D3D11_BUFFER_DESC));
	bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc.ByteWidth = sizeof(unsigned long)*giIndex_Count;
	bufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
	bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr = gpID3D11Device->CreateBuffer(&bufferDesc,NULL,&gpID3D11Buffer_IndexBuffer);
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

	ZeroMemory(&mappedSubresource,sizeof(D3D11_MAPPED_SUBRESOURCE));
	gpID3D11DeviceContext->Map(gpID3D11Buffer_IndexBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource);
	memcpy(mappedSubresource.pData,gulIndices,sizeof(unsigned long) * giIndex_Count);
	gpID3D11DeviceContext->Unmap(gpID3D11Buffer_IndexBuffer,0);

	if(gulIndices)
	{
		delete gulIndices;
		gulIndices = NULL;
	}

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
	gClearColor[2] = 1.0f;
	gClearColor[3] = 1.0f;

	gPerspectiveProjectionMatrix = XMMatrixIdentity();

	ZeroMemory(&gRasterizerDesc,sizeof(D3D11_RASTERIZER_DESC));
	gRasterizerDesc.AntialiasedLineEnable = FALSE;
	gRasterizerDesc.MultisampleEnable = FALSE;
	gRasterizerDesc.DepthBias = 0;
	gRasterizerDesc.DepthBiasClamp = 0.0f;
	gRasterizerDesc.CullMode = D3D11_CULL_NONE;
	gRasterizerDesc.DepthClipEnable = TRUE;
	gRasterizerDesc.FillMode = D3D11_FILL_SOLID;
	gRasterizerDesc.FrontCounterClockwise = FALSE;
	gRasterizerDesc.ScissorEnable = FALSE;

	hr = gpID3D11Device->CreateRasterizerState(&gRasterizerDesc,&gpID3D11RasterizerState);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Failed For CreateRasterizerState.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateBuffer() Succeeded For CreateRasterizerState.\n");
		fclose(gpFile);
	}

	gpID3D11DeviceContext->RSSetState(gpID3D11RasterizerState);

	/* TexMetadata *metadata;
	 ScratchImage scratchimage;
	
	 wchar_t filename[] = L"./dirt01d.tga";
	 hr = LoadFromTGAFile(filename, metadata, scratchimage);
	 if (FAILED(hr))
	 {
	 	fopen_s(&gpFile, gszLogFileName, "a+");
	 	fprintf_s(gpFile, "LoadFromTGAFile() Failed.\n");
	 	fclose(gpFile);
	 	return(hr);
	 }
	 else
	 {
	 	fopen_s(&gpFile, gszLogFileName, "a+");
	 	fprintf_s(gpFile, "LoadFromTGAFile() Succeeded.\n");
	 	fclose(gpFile);
	 }

	 const Image *image = scratchimage.GetImages();*/

	  int height, width;
	  unsigned char *tga_data = NULL;
	  Tga_Loader tga_Texture_Loader;
	  tga_data = tga_Texture_Loader.LoadTarga((char *)"./dirt01d.tga",height,width);
	  if(tga_data == NULL)
	  {
	  	fopen_s(&gpFile, gszLogFileName, "a+");
	  	fprintf_s(gpFile, "tga_Texture_Loader::LoadTarga() Failed.\n");
	  	fclose(gpFile);
	  	return(E_FAIL);
	  }

	 D3D11_TEXTURE2D_DESC textureDesc;
	 ZeroMemory(&textureDesc, sizeof(D3D11_TEXTURE2D_DESC));
	 D3D11_SHADER_RESOURCE_VIEW_DESC shader_Resource_View_Desc;
	 ZeroMemory(&shader_Resource_View_Desc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
	 ID3D11Texture2D *pID3D11Texture2D = NULL;
	 unsigned int rowPitch;

	 textureDesc.Height = height;
	 textureDesc.Width = width;
	 textureDesc.MipLevels = 0;
	 textureDesc.ArraySize = 1;
	 textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	 textureDesc.SampleDesc.Count = 1;
	 textureDesc.SampleDesc.Quality = 0;
	 textureDesc.Usage = D3D11_USAGE_DEFAULT;
	 textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
	 textureDesc.CPUAccessFlags = 0;
	 textureDesc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;

	 hr = gpID3D11Device->CreateTexture2D(&textureDesc, NULL, &pID3D11Texture2D);
	 if(FAILED(hr))
	 {
	 	fopen_s(&gpFile, gszLogFileName, "a+");
	 	fprintf_s(gpFile, "ID3D11Device::CreateTexture2D() Failed.\n");
	 	fclose(gpFile);
	 	return(hr);
	 }
	 else
	 {
	 	fopen_s(&gpFile, gszLogFileName, "a+");
	 	fprintf_s(gpFile, "ID3D11Device::CreateTexture2D() Succeeded.\n");
	 	fclose(gpFile);
	 }

	 rowPitch = (width * 4) * sizeof(unsigned char);
	 fopen_s(&gpFile, gszLogFileName, "a+");
	 fprintf_s(gpFile, "0.\n");
	 fclose(gpFile);

	 FILE *ptempFile;
	 fopen_s(&ptempFile, "tga_data.txt", "w");
	 for (int i = 0; i < height*width; i += 4)
		 fprintf(ptempFile, "%d,%d,%d,%d,\n", tga_data[i + 0], tga_data[i + 1], tga_data[i + 2], tga_data[i + 3]);
	 fclose(ptempFile);
	 gpID3D11DeviceContext->UpdateSubresource(pID3D11Texture2D, 0, NULL, tga_data, rowPitch, 0);

	 shader_Resource_View_Desc.Format = textureDesc.Format;
	 shader_Resource_View_Desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	 shader_Resource_View_Desc.Texture2D.MostDetailedMip = 0;
	 shader_Resource_View_Desc.Texture2D.MipLevels = -1;

	 fopen_s(&gpFile, gszLogFileName, "a+");
	 fprintf_s(gpFile, "2.\n");
	 fclose(gpFile);

	 hr = gpID3D11Device->CreateShaderResourceView(pID3D11Texture2D, &shader_Resource_View_Desc, &gpID3D11ShaderResourceView_Texture);
	 if(FAILED(hr))
	 {
	 	fopen_s(&gpFile, gszLogFileName, "a+");
	 	fprintf_s(gpFile, "ID3D11Device::CreateShaderResourceView() Failed.\n");
	 	fclose(gpFile);
	 	return(hr);
	 }
	 else
	 {
	 	fopen_s(&gpFile, gszLogFileName, "a+");
	 	fprintf_s(gpFile, "ID3D11Device::CreateShaderResourceView() Succeeded.\n");
	 	fclose(gpFile);
	 }

	 gpID3D11DeviceContext->GenerateMips(gpID3D11ShaderResourceView_Texture);

	 if(tga_data)
	 {
	 	delete tga_data;
	 	tga_data = NULL;
	 }

	/*hr=LoadD3DTexture(L"dirt01d.bmp",&gpID3D11ShaderResourceView_Texture);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "LoadD3DTexture() Failed For Kundali.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "LoadD3DTexture() Succeeded For Kundali.\n");
		fclose(gpFile);
	}*/

	D3D11_SAMPLER_DESC samplerDesc;
	ZeroMemory(&samplerDesc,sizeof(D3D11_SAMPLER_DESC));
	samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;

	hr=gpID3D11Device->CreateSamplerState(&samplerDesc,&gpID3D11SamplerState_Texture);
	if (FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateSamplerState() Failed For Pyramid Texture.\n");
		fclose(gpFile);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateSamplerState() Succeeded For Pyramid Texture.\n");
		fclose(gpFile);
	}

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

//HRESULT LoadD3DTexture(const wchar_t *textureFileName, ID3D11ShaderResourceView **ppID3D11ShaderResourceView)
//{
//	HRESULT hr;
//
//	hr = DirectX::CreateWICTextureFromFile(gpID3D11Device, gpID3D11DeviceContext, textureFileName, nullptr, ppID3D11ShaderResourceView);
//	if (FAILED(hr))
//	{
//		fopen_s(&gpFile, gszLogFileName, "a+");
//		fprintf_s(gpFile, "DirectX::CreateWICTextureFromFile() Failed For Texture From File.\n");
//		fclose(gpFile);
//		return(hr);
//	}
//	else
//	{
//		fopen_s(&gpFile, gszLogFileName, "a+");
//		fprintf_s(gpFile, "DirectX::CreateWICTextureFromFile() Succeeded For Texture From File.\n");
//		fclose(gpFile);
//	}
//
//	return(hr);
//}

bool InitializeTerrain(char *filename)
{
	bool LoadSetupFile(char*);
	bool LoadBitmapHeightMap();
	void ShutdownHeightMap();
	void SetTerrainCoordinates();
	bool BuildTerrainModel();
	void ShutdownTerrainModel();

	bool result;

	result = LoadSetupFile(filename);
	if(!result)
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "LoadSetupFile() Failed.\n");
		fclose(gpFile);
		return false;
	}

	result = LoadBitmapHeightMap();
	if(!result)
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "LoadBitmapHeightMap() Failed.\n");
		fclose(gpFile);
		return false;
	}

	SetTerrainCoordinates();

	result = BuildTerrainModel();
	if(!result)
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "BuildTerrainModel() Failed.\n");
		fclose(gpFile);
		return false;
	}

	//ShutdownHeightMap();

	//ShutdownTerrainModel();

	return true;
}

bool LoadSetupFile(char *filename)
{
	int stringLength;
	ifstream fin;
	char input;

	stringLength = 256;
	gTerrainFilename = new char[stringLength];
	if(!gTerrainFilename)
	{
		return false;
	}

	fin.open(filename);
	if(fin.fail())
	{
		return false;
	}

	fin.get(input);
	while(input!=':')
	{
		fin.get(input);
	}

	fin >> gTerrainFilename;

	fin.get(input);
	while(input!=':')
	{
		fin.get(input);
	}

	fin>>giTerrainHeight;

	fin.get(input);
	while(input!=':')
	{
		fin.get(input);
	}

	fin >> giTerrainWidth;

	fin.get(input);
	while(input!=':')
	{
		fin.get(input);
	}

	fin>>gfHeightScale;

	fin.close();

	return true;
}

bool LoadBitmapHeightMap()
{
	int error, imageSize, i, j, k, index;
	FILE *filePtr;
	unsigned long long count;
	BITMAPFILEHEADER bitmapFileHeader;
	BITMAPINFOHEADER bitmapInfoHeader;
	unsigned char *bitmapImage;
	unsigned char height;

	ZeroMemory((void*)&bitmapFileHeader,sizeof(BITMAPFILEHEADER));
	ZeroMemory((void*)&bitmapInfoHeader,sizeof(BITMAPINFOHEADER));

	gHeightMap = new HeightMapType[giTerrainWidth * giTerrainHeight];
	if(!gHeightMap)
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "new Failed.\n");
		fclose(gpFile);
		return false;
	}

	error = fopen_s(&filePtr,gTerrainFilename,"rb");
	if(error != 0)
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "fopen_s() Failed.\n");
		fclose(gpFile);
		return false;
	}

	count = fread(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);
	if(count!=1)
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "fread() Failed.\n");
		fclose(gpFile);
		return false;
	}

	count = fread(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER), 1, filePtr);
	if(count!=1)
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "fread() Failed.\n");
		fclose(gpFile);
		return false;
	}

	if((bitmapInfoHeader.biHeight != giTerrainHeight) || (bitmapInfoHeader.biWidth != giTerrainWidth))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "bitmapInfoHeader.biHeight = %ld, giTerrainHeight = %d, bitmapInfoHeader.biWidth = %ld, giTerrainWidth = %d, Size mismatch Failed.\n",bitmapInfoHeader.biHeight,giTerrainHeight,bitmapInfoHeader.biWidth, giTerrainWidth);
		fclose(gpFile);
		return false;
	}

	// Since we use non-divide by 2 dimensions (eg. 257x257) we need to add an extra byte to each line
	imageSize = giTerrainHeight * ((giTerrainWidth * 3) + 1);

	bitmapImage = new unsigned char[imageSize];
	if(!bitmapImage)
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "new Failed.\n");
		fclose(gpFile);
		return false;
	}

	fseek(filePtr, bitmapFileHeader.bfOffBits, SEEK_SET);

	count = fread(bitmapImage, 1, imageSize, filePtr);
	if(count != imageSize)
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "fread() Failed.\n");
		fclose(gpFile);
		return false;
	}

	error = fclose(filePtr);
	if(error!=0)
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "fclose() Failed.\n");
		fclose(gpFile);
		return false;
	}

	k = 0;

	for(j=0;j<giTerrainHeight;j++)
	{
		for(i=0;i<giTerrainWidth;i++)
		{
			height = bitmapImage[k];

			// Bitmaps are upside down so load bottom to top into the height map array.
			index = (giTerrainWidth * (giTerrainHeight -1 -j))+i;

			gHeightMap[index].y = (float)height;

			k+=3;
		}
		// Compensate for the extra byte at end of each line in non-divide by 2 bitmaps (eg. 257x257).
		k++;
	}

	delete []bitmapImage;
	bitmapImage = 0;

	delete []gTerrainFilename;
	gTerrainFilename = 0;

	return true;
}

void ShutdownHeightMap()
{
	if(gHeightMap)
	{
		delete []gHeightMap;
		gHeightMap = 0;
	}
}

void SetTerrainCoordinates()
{
	int i, j, index;

	for(j = 0; j < giTerrainHeight; j++)
	{
		for(i = 0; i < giTerrainWidth; i++)
		{
			index = (giTerrainWidth * j) + i;

			gHeightMap[index].x = (float)i;
			gHeightMap[index].z = -(float)j;

			gHeightMap[index].z += (float)(giTerrainHeight - 1);

			gHeightMap[index].y /= gfHeightScale;
		}
	}
}

bool BuildTerrainModel()
{
	int i, j, index, index1, index2, index3, index4;

	giVertex_Count = (giTerrainHeight - 1) * (giTerrainWidth - 1) *6;

	gTerrainModel = new ModelType[giVertex_Count];
	if(!gTerrainModel)
	{
		return false;
	}

	index = 0;

	for(j=0;j<(giTerrainHeight-1);j++)
	{
		for(i=0;i<(giTerrainWidth-1);i++)
		{
			index1 = (giTerrainWidth * j) +i;
			index2 = (giTerrainWidth * j) +(i+1);
			index3 = (giTerrainWidth * (j+1)) +i;
			index4 = (giTerrainWidth * (j+1)) +(i+1);

			gTerrainModel[index].x = gHeightMap[index1].x;
			gTerrainModel[index].y = gHeightMap[index1].y;
			gTerrainModel[index].z = gHeightMap[index1].z;
			gTerrainModel[index].tu = 0.0f;
			gTerrainModel[index].tv = 0.0f;
			index++;

			gTerrainModel[index].x = gHeightMap[index2].x;
			gTerrainModel[index].y = gHeightMap[index2].y;
			gTerrainModel[index].z = gHeightMap[index2].z;
			gTerrainModel[index].tu = 1.0f;
			gTerrainModel[index].tv = 0.0f;
			index++;

			gTerrainModel[index].x = gHeightMap[index3].x;
			gTerrainModel[index].y = gHeightMap[index3].y;
			gTerrainModel[index].z = gHeightMap[index3].z;
			gTerrainModel[index].tu = 0.0f;
			gTerrainModel[index].tv = 1.0f;
			index++;

			gTerrainModel[index].x = gHeightMap[index3].x;
			gTerrainModel[index].y = gHeightMap[index3].y;
			gTerrainModel[index].z = gHeightMap[index3].z;
			gTerrainModel[index].tu = 0.0f;
			gTerrainModel[index].tv = 1.0f;
			index++;

			gTerrainModel[index].x = gHeightMap[index2].x;
			gTerrainModel[index].y = gHeightMap[index2].y;
			gTerrainModel[index].z = gHeightMap[index2].z;
			gTerrainModel[index].tu = 1.0f;
			gTerrainModel[index].tv = 0.0f;
			index++;

			gTerrainModel[index].x = gHeightMap[index4].x;
			gTerrainModel[index].y = gHeightMap[index4].y;
			gTerrainModel[index].z = gHeightMap[index4].z;
			gTerrainModel[index].tu = 1.0f;
			gTerrainModel[index].tv = 1.0f;
			index++;
		}
	}
	return true;
}

void ShutdownTerrainModel()
{
	if(gTerrainModel)
	{
		delete []gTerrainModel;
		gTerrainModel = 0;
	}
}

HRESULT resize(int width, int height)
{
	HRESULT hr = S_OK;

	//Free any size dependent resources
	if(gpID3D11DepthStencilView)
	{
		gpID3D11DepthStencilView->Release();
		gpID3D11DepthStencilView = NULL;
	}

	//Free any size dependent resources
	if(gpID3D11RenderTargetView)
	{
		gpID3D11RenderTargetView->Release();
		gpID3D11RenderTargetView=NULL;
	}

	//Resize the Swap Chain Buffer
	gpIDXGISwapChain->ResizeBuffers(1,width,height,DXGI_FORMAT_R8G8B8A8_UNORM,0);

	D3D11_TEXTURE2D_DESC textureDesc;
	ZeroMemory(&textureDesc,sizeof(D3D11_TEXTURE2D_DESC));
	textureDesc.Width = width;
	textureDesc.Height = height;
	textureDesc.ArraySize = 1;
	textureDesc.MipLevels = 1;
	textureDesc.SampleDesc.Count = 4;
	textureDesc.SampleDesc.Quality = 1;
	textureDesc.Format = DXGI_FORMAT_D32_FLOAT;
	textureDesc.Usage = D3D11_USAGE_DEFAULT;
	textureDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	textureDesc.CPUAccessFlags = 0;
	textureDesc.MiscFlags = 0;

	ID3D11Texture2D *pID3D11Texture2D = NULL;
	
	hr = gpID3D11Device->CreateTexture2D(&textureDesc,0,&pID3D11Texture2D);
	if(FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateTexture2D() Failed.\n");
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
	ZeroMemory(&depthStencilViewDesc,sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));
	depthStencilViewDesc.Format = DXGI_FORMAT_D32_FLOAT;
	depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;
	hr = gpID3D11Device->CreateDepthStencilView(pID3D11Texture2D,&depthStencilViewDesc,&gpID3D11DepthStencilView);
	if(FAILED(hr))
	{
		fopen_s(&gpFile, gszLogFileName, "a+");
		fprintf_s(gpFile, "ID3D11Device::CreateDepthStencilView() Failed.\n");
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
	gpID3D11DeviceContext->OMSetRenderTargets(1,&gpID3D11RenderTargetView,gpID3D11DepthStencilView);
	//gpID3D11DeviceContext->OMSetRenderTargets(1,&gpID3D11RenderTargetView,NULL);

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
	gpID3D11DeviceContext->ClearDepthStencilView(gpID3D11DepthStencilView,D3D11_CLEAR_DEPTH,1.0f,0.0f);
	//Clear Render Target View to chosen color
	gpID3D11DeviceContext->ClearRenderTargetView(gpID3D11RenderTargetView,gClearColor);

	UINT stride = sizeof(VertexType);
	UINT offset = 0;

	gpID3D11DeviceContext->IASetVertexBuffers(0,1,&gpID3D11Buffer_VertexBuffer,&stride,&offset);
	gpID3D11DeviceContext->IASetIndexBuffer(gpID3D11Buffer_IndexBuffer, DXGI_FORMAT_R32_UINT,0);

	gpID3D11DeviceContext->PSSetShaderResources(0,1,&gpID3D11ShaderResourceView_Texture);
	gpID3D11DeviceContext->PSSetSamplers(0,1,&gpID3D11SamplerState_Texture);

	gpID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

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

	//gpID3D11DeviceContext->Draw(giVertex_Count,0);
	gpID3D11DeviceContext->DrawIndexed(giIndex_Count,0,0);

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
	if(gulIndices)
	{
		delete gulIndices;
		gulIndices = NULL;
	}

	if(gfVertices)
	{
		delete gfVertices;
		gfVertices = NULL;
	}
	
	if(gpID3D11DepthStencilView)
	{
		gpID3D11DepthStencilView->Release();
		gpID3D11DepthStencilView = NULL;
	}

	if (gpID3D11RasterizerState)
	{
		gpID3D11RasterizerState->Release();
		gpID3D11RasterizerState = NULL;
	}

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

	if(gpID3D11Buffer_IndexBuffer)
	{
		gpID3D11Buffer_IndexBuffer->Release();
		gpID3D11Buffer_IndexBuffer = NULL;
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
