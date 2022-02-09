#include <windows.h>
#include <windowsx.h>

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "OpenGL32.lib")

#include <stdio.h>
#include <gl\glew.h>
#include <gl\wglew.h>
#include <gl\GL.h>
#include <vector>
#include <string>
#include <sstream>

//#define GLM_FORCE_CUDA
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\quaternion.hpp>
#include <glm\gtc\random.hpp>
#include <glm\gtx\rotate_vector.hpp>
#include <glm\ext.hpp>
#include <glm\gtx\string_cast.hpp>
#include <glm\gtc\noise.hpp>

#include "Data.h"
#include "Camera/Camera.h"
#include "Camera/Player.h"
#include "LightsMaterial/LightsMaterial.h"
#include "Texture/Texture.h"
#include "ShaderLoad/LoadShaders.h"
#include "BasicShapes/BasicShapes.h"
#include "Skydome/Skydome.h"
#include "Grass/Grass.h"
#include "Terrain/Terrain.h"

//using namespace std;

bool gbEscapeKeyIsPressed = false;
bool gbIsFullscreen = false;
bool gbActiveWindow = false;

void ToggleFullscreen(void);

//BOOL gbDone = FALSE;

HWND ghWnd = NULL;
HDC ghDC = NULL;
HGLRC ghRC = NULL;

POINT ClickPoint, MovePoint, OldMovePoint;
GLboolean gbFirstMouse = GL_TRUE;

CCommon *pCCommon = NULL;
CCamera *pCCamera = NULL;
CPlayer *pCPlayer = NULL;
CSkyDome *pCSkyDome = NULL;
CGrass *pCGrass = NULL;
CTerrain *pCTerrain = NULL;

CDirectionalLight *pSunLight = NULL;
CPointLight *pLamp = NULL;
CSpotLight *pTorch = NULL;

CSphere *pSunSphere = NULL;

glm::vec3 CameraPosition;
glm::vec3 CameraFront;
glm::vec3 CameraUp;
glm::vec3 CameraLook;
glm::vec3 CameraRight;

GLboolean gbIsNight = GL_FALSE;
GLboolean gbIsDayNightCyclePaused = GL_FALSE;

DWORD dwStyle;

GLboolean gbIsPlaying = GL_FALSE;

WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
GLuint giWinHeight = 0;
GLuint giWinWidth = 0;
TCHAR str[255];

GLfloat gfLastX = giWinWidth / 2.0f;
GLfloat gfLastY = giWinHeight / 2.0f;

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

GLfloat gfFOVY = FOV_Y;

GLfloat gfAnglePyramid = 0.0f;
GLfloat gfAngleCube = 0.0f;

GLuint gShaderProgramObjectBasicShader;
GLuint gShaderProgramObjectBasicShaderTexture;

// HDR
GLuint gShaderObjectHDR;
GLuint gFrameBufferObjectHDR;
GLuint gRenderBufferObjectDepth;
GLuint gColorBufferTexture;

GLuint gVAOHDRQuad;
GLuint gVBOHDR;

void InitializeQuadHDR(void);
void RenderQuadHDR(void);


// General
GLuint gVAOPyramid;
GLuint gVAOCube;
GLuint gVAOGround;
GLuint gVAOLineGrid;
GLuint gVAOGroundGrid;
GLuint gVAOAxes;
GLuint gVAOLightCube;

GLuint gVBOPosition;
GLuint gVBOColor;
GLuint gVBONormal;
GLuint gVBOTexture;

GLint NumGridVertices = 0;

GLuint gModelMatrixUniform, gViewMatrixUniform, gProjectionMatrixUniform, gViewPositionUniform, gMVPMatrixUniform;
glm::mat4 gPerspectiveProjectionMatrix;
glm::mat4 gOrthographicProjectionMatrix;

GLint Samples;


//Timing related
double gd_resolution;
unsigned __int64 gu64_base;

float gfDeltaTime = 0.0f;
float gfElapsedTime = 0.0f;
int giFrameCount = 0;
__int64 giPreviousTime = 0;
__int64 giCurrentTime = 0;
__int64 giCountsPerSecond;
float fSecondsPerCount;
int fps__;

float LastFrame = 0.0f;

RECT screen;

std::stringstream strLightProperty;

////////////////////////////////
////////////////////////////////
FILE *pFilePosition = NULL;
FILE *pFileNormal = NULL;
FILE *pFileTexCoord = NULL;
FILE *pFileIndices = NULL;
FILE *pFileGeneratedTriangle = NULL;
FILE *pFileTangent = NULL;
///////////////////////////////////////////////
// Sun related
GLuint BitmapTextureSun;
GLuint DepthTexture;
GLuint SunTextures[4];
GLuint gFBOSun;
GLuint gVBOSun;

GLuint gVAOScreenFilledQuad;
GLuint gVBOPositionScreenFilledQuad;
GLuint gVBOTexCoordsScreenFilledQuad;

GLuint gVertexShaderObjectSunDepthTest;
GLuint gVertexShaderObjectBlur;
GLuint gVertexShaderObjectBlurHorizontal;
GLuint gVertexShaderObjectBlurVertical;
GLuint gVertexShaderObjectSunRays;

GLuint gFragmentShaderObjectSunDepthTest;
GLuint gFragmentShaderObjectBlurHorizontal;
GLuint gFragmentShaderObjectBlurVertical;
GLuint gFragmentShaderObjectSunRays;

GLuint gShaderProgramObjectSunDepthTest;
GLuint gShaderProgramObjectBlurHorizontal;
GLuint gShaderProgramObjectBlurVertical;
GLuint gShaderProgramObjectSunRays;

GLint giSunTextureWidth, giSunTextureHeight;

GLfloat ScreenFilledQuad[] = {
	0.0f, 0.0f,
	1.0f, 0.0f,
	1.0f, 1.0f,
	0.0f, 1.0f
};

GLfloat ScreenFilledQuadTexCoords[] = {
	0.0f, 0.0f,
	1.0f, 0.0f,
	1.0f, 1.0f,
	0.0f, 1.0f
};

//main function
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int nCmdShow)
{
	MSG msg;
	bool bDone = false;

	void Initialize(HINSTANCE hInstance, int Samples);
	void InitializeTimer(void);
	void CalculateDelta(void);
	void InitializeOpenGL(void);
	void Uninitialize(void);
	void Display(void);
	void Update(void);

	void CheckAsyncKeyDown(void);
	void CalculateFPS(void);

	pCCommon = new CCommon();

	if (fopen_s(&pCCommon->pLogFile, pCCommon->logfilepath, "w") != 0)
	{
		MessageBox(NULL, TEXT("Failed to create the log file. Exiting now."), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
	{
		fprintf_s(pCCommon->pLogFile, "Log file created successfully.\n");
	}

	// opening position file
	if (fopen_s(&pFilePosition, "Logs/01_Position.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Failed to create the position file. Exiting now."), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
	{
		fprintf_s(pCCommon->pLogFile, "position file created successfully.\n");
	}

	// opening normal file
	if (fopen_s(&pFileNormal, "Logs/02_Normal.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Failed to create the normal file. Exiting now."), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
	{
		fprintf_s(pCCommon->pLogFile, "normal file created successfully.\n");
	}

	// opening texcoord file
	if (fopen_s(&pFileTexCoord, "Logs/03_TexCoord.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Failed to create the texcoord file. Exiting now."), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
	{
		fprintf_s(pCCommon->pLogFile, "texcoord file created successfully.\n");
	}

	// opening indices
	if (fopen_s(&pFileIndices, "Logs/04_Indices.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Failed to create the indices file. Exiting now."), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
	{
		fprintf_s(pCCommon->pLogFile, "indices file created successfully.\n");
	}

	// opening tangent
	if (fopen_s(&pFileTangent, "Logs/05_Tangents.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Failed to create the tangents file. Exiting now."), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
	{
		fprintf_s(pCCommon->pLogFile, "tangents file created successfully.\n");
	}


	Initialize(hInstance, MSAA_SAMPLES);

	//display created window and update window
	ShowWindow(ghWnd, nCmdShow);
	SetForegroundWindow(ghWnd);
	SetFocus(ghWnd);

	InitializeTimer();

	fclose(pCCommon->pLogFile);
	InitializeOpenGL();

	// Game Loop
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
			CalculateDelta();
			CheckAsyncKeyDown();
			Update();
			Display();
			CalculateFPS();

			if (gbActiveWindow == true)
			{
				if (gbEscapeKeyIsPressed == true)
					bDone = true;
			}
		}
	}

	Uninitialize();
	return((int)msg.wParam);
}

void InitializeTimer(void)
{
	// Counts per second
	QueryPerformanceFrequency((LARGE_INTEGER *)&giCountsPerSecond);
	// seconds per count
	fSecondsPerCount = 1.0f / giCountsPerSecond;

	// previous time
	QueryPerformanceCounter((LARGE_INTEGER *)&giPreviousTime);
}

void CalculateDelta(void)
{
	// get current count
	QueryPerformanceCounter((LARGE_INTEGER *)&giCurrentTime);

	// Delta time
	gfDeltaTime = (giCurrentTime - giPreviousTime) * fSecondsPerCount;
	gfElapsedTime += gfDeltaTime;
}

void CalculateFPS(void)
{
	giFrameCount++;

	TCHAR str[255];

	// frames per second
	if (gfElapsedTime >= 1.0f)
	{
		fps__ = giFrameCount;

		swprintf_s(str, 255, TEXT("OpenGL Project FPS : %d"), fps__);
		SetWindowText(ghWnd, str);

		giFrameCount = 0;
		gfElapsedTime = 0.0f;
	}

	giPreviousTime = giCurrentTime;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	void Resize(int Width, int Height);
	void Uninitialize(void);

	void OnLButtonDown(INT, INT, DWORD);
	void OnLButtonUp(INT, INT, DWORD);
	void OnRButtonDown(INT, INT, DWORD);
	void OnRButtonUp(INT, INT, DWORD);
	void OnMouseMove(INT, INT, DWORD);
	void OnMButtonDown(INT, INT, DWORD);
	void OnMButtonUp(INT, INT, DWORD);
	void OnMouseWheelScroll(short zDelta);

	void OnUpArrowPress(void);
	void OnLeftArrowPress(void);
	void OnRightArrowPress(void);
	void OnDownArrowPress(void);
	void OnPageUpPress(void);
	void OnPageDownPress(void);

	void CheckAsyncKeyDown(void);


	switch (iMsg)
	{
	case WM_CREATE:
		break;

	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			gbActiveWindow = true;
		else
			gbActiveWindow = false;
		break;

		//case WM_ERASEBKGND:
		//return(0);

	case WM_SIZE:
		giWinWidth = LOWORD(lParam);
		giWinHeight = HIWORD(lParam);
		Resize(giWinWidth, giWinHeight);
		//Resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_LBUTTONDOWN:
		OnLButtonDown(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_LBUTTONUP:
		OnLButtonUp(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_RBUTTONDOWN:
		OnRButtonDown(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_RBUTTONUP:
		OnRButtonUp(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_MOUSEWHEEL:
		OnMouseWheelScroll(GET_WHEEL_DELTA_WPARAM(wParam));
		break;

	case WM_MOUSEMOVE:
		OnMouseMove(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_MBUTTONDOWN:
		OnMButtonDown(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_MBUTTONUP:
		OnMButtonUp(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_KEYDOWN:
		//CheckAsyncKeyDown();
		switch (wParam)
		{
		case VK_ESCAPE:
			gbEscapeKeyIsPressed = true;
			break;

		case VK_RETURN:
			if (gbIsFullscreen == false)
			{
				ToggleFullscreen();
				gbIsFullscreen = true;
			}
			else
			{
				ToggleFullscreen();
				gbIsFullscreen = false;
			}
			break;

		case 0x46:	//F key
			if (gbIsDayNightCyclePaused == GL_FALSE)
			{
				gbIsDayNightCyclePaused = GL_TRUE;
			}
			else
			{
				gbIsDayNightCyclePaused = GL_FALSE;
			}
			break;

			//case VK_UP:
			//	OnUpArrowPress();
			//	break;

			//case VK_DOWN:
			//	OnDownArrowPress();
			//	break;

			//case VK_LEFT:
			//	OnLeftArrowPress();
			//	break;

			//case VK_RIGHT:
			//	OnRightArrowPress();
			//	break;

			//	// PAGE UP key
			//case VK_PRIOR:
			//	OnPageUpPress();
			//	break;

			//	// PAGE DOWN KEY
			//case VK_NEXT:
			//	OnPageDownPress();
			//	break;

			// W key
		case 0x57:
			break;

			// S key
		case 0x53:
			break;

			// D key
		case 0x41:
			break;

			// A key
		case 0x44:
			break;

		case VK_NUMPAD8:
			break;

		case VK_NUMPAD2:
			break;

		case VK_NUMPAD4:
			break;

		case VK_NUMPAD6:
			break;

		case VK_NUMPAD7:
			break;

		case VK_NUMPAD1:
			break;

			// E key
		case 0x45:
			//cam.Roll(-1.0);
			break;

			// Q key
		case 0x51:
			//cam.Roll(1.0);
			break;

			// P key
		case 0x50:
			if (gbIsPlaying == GL_FALSE)
			{
				gbIsPlaying = GL_TRUE;

				if (gbIsFullscreen == false)
				{
					ToggleFullscreen();
					gbIsFullscreen = true;
				}

				GetWindowRect(ghWnd, &screen);
				SetCursorPos((int)(screen.right * 0.5), (int)(screen.bottom * 0.5));
				ShowCursor(FALSE);
			}
			else
			{
				gbIsPlaying = GL_FALSE;
				ShowCursor(TRUE);
			}
			break;

			// L key
			// cycle through lights
		case 0x4C:
			break;

		case VK_ADD:
			break;

		case VK_SUBTRACT:
			break;

		case VK_CONTROL:
			if (TOGGLE_CROUCH == TRUE)
			{
				if (pCPlayer->GetIsCrouching() == GL_TRUE)
				{
					pCPlayer->SetIsCrouching(GL_FALSE);
				}
				else if (pCPlayer->GetIsCrouching() == GL_FALSE)
				{
					pCPlayer->SetIsCrouching(GL_TRUE);
				}
			}
			break;

		case VK_SHIFT:
			if (TOGGLE_SPRINT == TRUE)
			{
				if (pCPlayer->GetIsSprinting() == GL_TRUE)
				{
					pCPlayer->SetIsSprinting(GL_FALSE);
				}
				else if (pCPlayer->GetIsSprinting() == GL_FALSE)
				{
					pCPlayer->SetIsSprinting(GL_TRUE);
				}
			}
			break;

		default:
			break;
		}
		break;

	case WM_CLOSE:
		Uninitialize();
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	default:
		break;
	}

	return(DefWindowProc(hWnd, iMsg, wParam, lParam));
}

void CheckAsyncKeyDown(void)
{
	if (GetAsyncKeyState(VK_UP) & 0x8000)
	{
		if (gbIsPlaying == GL_FALSE)
			pCCamera->ProcessNavigationKeys(CameraMovement::CAMERA_FORWARD, gfDeltaTime);
		else if (gbIsPlaying == GL_TRUE)
			pCPlayer->ControlPlayerMovement(PlayerMovement::PLAYER_FORWARD, gfDeltaTime);
	}

	if (GetAsyncKeyState(VK_DOWN) & 0x8000)
	{
		if (gbIsPlaying == GL_FALSE)
			pCCamera->ProcessNavigationKeys(CameraMovement::CAMERA_BACKWARD, gfDeltaTime);
		else if (gbIsPlaying == GL_TRUE)
			pCPlayer->ControlPlayerMovement(PlayerMovement::PLAYER_BACKWARD, gfDeltaTime);
	}

	if (GetAsyncKeyState(VK_LEFT) & 0x8000)
	{
		if (gbIsPlaying == GL_FALSE)
			pCCamera->ProcessNavigationKeys(CameraMovement::CAMERA_LEFT, gfDeltaTime);
		else if (gbIsPlaying == GL_TRUE)
			pCPlayer->ControlPlayerMovement(PlayerMovement::PLAYER_LEFT, gfDeltaTime);
	}

	if (GetAsyncKeyState(VK_RIGHT) & 0x8000)
	{
		if (gbIsPlaying == GL_FALSE)
			pCCamera->ProcessNavigationKeys(CameraMovement::CAMERA_RIGHT, gfDeltaTime);
		else if (gbIsPlaying == GL_TRUE)
			pCPlayer->ControlPlayerMovement(PlayerMovement::PLAYER_RIGHT, gfDeltaTime);
	}

	if (GetAsyncKeyState(VK_PRIOR) & 0x8000)
	{
		pCCamera->ProcessNavigationKeys(CameraMovement::CAMERA_UP, gfDeltaTime);
	}

	if (GetAsyncKeyState(VK_NEXT) & 0x8000)
	{
		pCCamera->ProcessNavigationKeys(CameraMovement::CAMERA_DOWN, gfDeltaTime);
	}

	if (GetAsyncKeyState(VK_SPACE) & 0x8000)
	{
		if (gbIsPlaying == GL_FALSE)
		{
			// code
		}
		else if (gbIsPlaying == GL_TRUE && pCPlayer->GetIsTouchingGround() == GL_TRUE)
			pCPlayer->ControlPlayerMovement(PlayerMovement::PLAYER_JUMP, gfDeltaTime);
	}

	if (TOGGLE_CROUCH == FALSE)
	{
		if (GetAsyncKeyState(VK_CONTROL))
		{
			if (pCPlayer->GetIsCrouching() == GL_FALSE)
			{
				pCPlayer->SetIsCrouching(GL_TRUE);
			}
		}
		else
		{
			if (pCPlayer->GetIsCrouching() == GL_TRUE)
			{
				pCPlayer->SetIsCrouching(GL_FALSE);
			}
		}
	}

	if (TOGGLE_SPRINT == FALSE)
	{
		if (GetAsyncKeyState(VK_SHIFT))
		{
			if (pCPlayer->GetIsSprinting() == GL_FALSE)
			{
				pCPlayer->SetIsSprinting(GL_TRUE);
			}
		}
		else
		{
			if (pCPlayer->GetIsSprinting() == GL_TRUE)
			{
				pCPlayer->SetIsSprinting(GL_FALSE);
			}
		}
	}

	if (gbIsPlaying == GL_TRUE)
	{
		pCPlayer->ControlMouseInput();
		SetCursorPos((int)(screen.right * 0.5), (int)(screen.bottom * 0.5));
	}

}

void ToggleFullscreen(void)
{
	HMONITOR hMonitor;
	MONITORINFO mi;
	BOOL bWindowPlacement;
	BOOL bMonitorInfo;

	if (gbIsFullscreen == FALSE)
	{
		dwStyle = GetWindowLong(ghWnd, GWL_STYLE);
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{

			mi.cbSize = { sizeof(MONITORINFO) };

			bWindowPlacement = GetWindowPlacement(ghWnd, &wpPrev);
			hMonitor = MonitorFromWindow(ghWnd, MONITOR_DEFAULTTOPRIMARY);
			bMonitorInfo = GetMonitorInfo(hMonitor, &mi);

			if (bWindowPlacement == TRUE && bMonitorInfo == TRUE)
			{
				SetWindowLong(ghWnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghWnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
	}
	else
	{
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghWnd, &wpPrev);
		SetWindowPos(ghWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
	}
}

void Initialize(HINSTANCE hInstance, int Samples)
{

	//variable declarations
	WNDCLASSEX wndclass;
	HWND hWnd;
	TCHAR szClassName[] = TEXT("RTROpenGLProject");
	TCHAR szAppName[] = TEXT("OpenGL Experimantal Project");

	void Initialize(HINSTANCE hInstance, int Samples);

	//WNDCLASSEX initialization
	wndclass.cbClsExtra = 0;
	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.cbWndExtra = 0;
	wndclass.hbrBackground = (HBRUSH)GetStockObject(GRAY_BRUSH);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hInstance = hInstance;
	wndclass.lpfnWndProc = WndProc;
	wndclass.lpszClassName = szClassName;
	wndclass.lpszMenuName = NULL;
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;

	//register wndclass
	RegisterClassEx(&wndclass);

	int iWidth = GetSystemMetrics(SM_CXSCREEN);
	int iHeight = GetSystemMetrics(SM_CYSCREEN);

	int iXFinalPosition = (iWidth / 2) - (WIN_WIDTH / 2);
	int iYFinalPosition = (iHeight / 2) - (WIN_HEIGHT / 2);

	//create window
	hWnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, szAppName, WS_OVERLAPPEDWINDOW, iXFinalPosition, iYFinalPosition, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);
	ghWnd = hWnd;

	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

	ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32;
	pfd.iLayerType = PFD_MAIN_PLANE;

	ghDC = GetDC(ghWnd);

	iPixelFormatIndex = ChoosePixelFormat(ghDC, &pfd);
	if (iPixelFormatIndex == 0)
	{
		fprintf(pCCommon->pLogFile, "ChoosePixelFormat() Error : Pixel Format Index is 0\n");
		//fclose(Essential.gpFile);
		//Essential.gpFile = NULL;

		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
	}

	static int MSAAPixelFormat = 0;

	if (SetPixelFormat(ghDC, MSAAPixelFormat == 0 ? iPixelFormatIndex : MSAAPixelFormat, &pfd) == false)
	{
		fprintf(pCCommon->pLogFile, "SetPixelFormat() Error : Failed to set the pixel format\n");
		//fclose(Essential.gpFile);
		//Essential.gpFile = NULL;

		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
	}

	ghRC = wglCreateContext(ghDC);
	if (ghRC == NULL)
	{
		fprintf(pCCommon->pLogFile, "wglCreateContext() Error : Rendering context ghRC is NULL\n");
		//fclose(Essential.gpFile);
		//Essential.gpFile = NULL;

		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
	}

	if (wglMakeCurrent(ghDC, ghRC) == false)
	{
		fprintf(pCCommon->pLogFile, "wglMakeCurrent() Error : wglMakeCurrent() Failed to set rendering context as current context\n");
		//fclose(Essential.gpFile);
		//Essential.gpFile = NULL;

		wglDeleteContext(ghRC);
		ghRC = NULL;
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
	}

	GLenum glew_error = glewInit();
	if (glew_error != GLEW_OK)
	{
		wsprintf(str, TEXT("Error: %s\n"), glewGetErrorString(glew_error));
		MessageBox(ghWnd, str, TEXT("GLEW Error"), MB_OK);

		wglDeleteContext(ghRC);
		ghRC = NULL;
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
	}

	//MessageBox(ghWnd, TEXT("Before if condition"), TEXT("Message"), MB_OK | MB_TOPMOST);
	if (MSAAPixelFormat == 0 && Samples > 0)
	{
		//MessageBox(ghWnd, TEXT("Inside first if condition"), TEXT("Message"), MB_OK | MB_TOPMOST);
		if (WGLEW_ARB_pixel_format && GLEW_ARB_multisample)
		{
			//MessageBox(ghWnd, TEXT("Inside second if condition"), TEXT("Message"), MB_OK | MB_TOPMOST);
			while (Samples > 0)
			{
				//MessageBox(ghWnd, TEXT("Inside while condition"), TEXT("Message"), MB_OK | MB_TOPMOST);

				UINT NumFormats = 0;

				int PFAttribs[] =
				{
					WGL_DRAW_TO_WINDOW_ARB, GL_TRUE,
					WGL_SUPPORT_OPENGL_ARB, GL_TRUE,
					WGL_DOUBLE_BUFFER_ARB, GL_TRUE,
					WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB,
					WGL_COLOR_BITS_ARB, 32,
					WGL_DEPTH_BITS_ARB, 24,
					WGL_ACCELERATION_ARB, WGL_FULL_ACCELERATION_ARB,
					WGL_SAMPLE_BUFFERS_ARB, GL_TRUE,
					WGL_SAMPLES_ARB, Samples,
					0
				};

				if (wglChoosePixelFormatARB(ghDC, PFAttribs, NULL, 1, &MSAAPixelFormat, &NumFormats) == TRUE && NumFormats > 0)
				{
					//MessageBox(ghWnd, TEXT("Inside third if condition"), TEXT("Message"), MB_OK | MB_TOPMOST);
					break;
				}
				Samples--;
			}

			//MessageBox(ghWnd, TEXT("Out of while condition"), TEXT("Message"), MB_OK | MB_TOPMOST);

			//wglDeleteContext(ghRC);
			//DestroyWindow(hWnd);
			//UnregisterClass(wndclass.lpszClassName, hInstance);
			return(Initialize(hInstance, Samples));
		}
		else
		{
			//MessageBox(ghWnd, TEXT("Inside else condition"), TEXT("Message"), MB_OK | MB_TOPMOST);
			Samples = 0;
		}
	}

	if (!GLEW_ARB_texture_non_power_of_two)
	{
		fprintf(pCCommon->pLogFile, "GL_ARB_texture_non_power_of_two not supported!\n");
	}

	if (!GLEW_ARB_depth_texture)
	{
		fprintf(pCCommon->pLogFile, "GLEW_ARB_depth_texture not supported!\n");
	}

	if (!GLEW_EXT_framebuffer_object)
	{
		fprintf(pCCommon->pLogFile, "GLEW_EXT_framebuffer_object not supported!\n");
	}
}

void InitializeOpenGL(void)
{
	void Uninitialize(void);
	void Resize(int, int);
	void InitializeLightCube(void);

	pSunLight = new CDirectionalLight();
	pLamp = new CPointLight[MAX_POINT_LIGHTS];
	pTorch = new CSpotLight();
	pCCamera = new CCamera();
	pCTerrain = new CTerrain();
	pCPlayer = new CPlayer(glm::vec3(0.0f, pCTerrain->GetHeightAt(glm::vec3(0.0f, 0.0f, 0.0f)), 0.0f), pCTerrain);
	pCSkyDome = new CSkyDome();
	pCGrass = new CGrass();
	pSunSphere = new CSphere();

	ShaderInfo BasicShader[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/BasicShader.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/BasicShader.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	gShaderProgramObjectBasicShader = LoadShaders(BasicShader);

	ShaderInfo BasicShaderTexture[] = 
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/BasicShaderTexture.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/BasicShaderTexture.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	gShaderProgramObjectBasicShaderTexture = LoadShaders(BasicShaderTexture);

	ShaderInfo SkyShader[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/AtmosphericScatter.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/AtmosphericScatter.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	ShaderInfo TerrainDay[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/TerrainDayTimeSugat.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/TerrainDayTimeSugat.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	ShaderInfo TerrainNight[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/TerrainNightMultilights.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/TerrainNightMultilights.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	/*ShaderInfo TerrainNormalMap[] = 
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/TerrainNightTimeSinglePointLightNormalMap.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/TerrainNightTimeSinglePointLightNormalMap.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	gShaderProgramObjectTerrainNightSinglePointNormalMap = LoadShaders(TerrainNormalMap);*/

	ShaderInfo GrassBasic[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/GrassBasicShader.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/GrassBasicShader.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	ShaderInfo SunDepthTest[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/SunDepthTest.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/SunDepthTest.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	gShaderProgramObjectSunDepthTest = LoadShaders(SunDepthTest);

	ShaderInfo BlurHorizontal[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/Blur.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/BlurHorizontal.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	gVertexShaderObjectBlurHorizontal = LoadShaders(BlurHorizontal);

	ShaderInfo BlurVertical[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/Blur.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/BlurVertical.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	gVertexShaderObjectBlurHorizontal = LoadShaders(BlurVertical);

	ShaderInfo SunRays[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/SunRays.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/SunRays.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	gShaderProgramObjectSunRays = LoadShaders(SunRays);

	ShaderInfo HDR[] = 
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/HDR.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/HDR.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	gShaderObjectHDR = LoadShaders(HDR);

	// initialize light values
	pSunLight->SetAmbient(glm::vec3(0.2f, 0.2f, 0.2f));
	pSunLight->SetDiffuse(glm::vec3(1.0f, 1.0f, 1.0f));
	pSunLight->SetSpecular(glm::vec3(1.0f, 1.0f, 1.0f));
	pSunLight->SetDirection(glm::vec3(0.0f, 1.0f, 0.0f));

	pLamp[0].SetAmbient(glm::vec3(0.980f * 0.1f, 0.980f * 0.1f, 0.824f * 0.1f));
	pLamp[0].SetDiffuse(glm::vec3(0.980f, 0.980f, 0.824f)); // Light Golden Yellow Rod
	pLamp[0].SetSpecular(glm::vec3(0.980f, 0.980f, 0.824f));
	pLamp[0].SetPosition(glm::vec3(10.0f, 20.0f, 10.0f));
	pLamp[0].SetAttenuation(1.0f, 0.07f, 0.017f); // 65 Units

	pLamp[1].SetAmbient(glm::vec3(0.980f * 0.1f, 0.980f * 0.1f, 0.824f * 0.1f));
	pLamp[1].SetDiffuse(glm::vec3(0.980f, 0.980f, 0.824f)); // Light Golden Yellow Rod
	pLamp[1].SetSpecular(glm::vec3(0.980f, 0.980f, 0.824f));
	pLamp[1].SetPosition(glm::vec3(10.0f, 20.0f, 20.0f));
	pLamp[1].SetAttenuation(1.0f, 0.07f, 0.017f); // 65 Units

	pTorch->SetAmbient(glm::vec3(0.980f * 0.1f, 0.980f * 0.1f, 0.824f * 0.1f));
	pTorch->SetDiffuse(glm::vec3(0.980f, 0.980f, 0.824f));
	pTorch->SetSpecular(glm::vec3(0.980f, 0.980f, 0.824f));
	pTorch->SetPosition(glm::vec3(0.0f, 0.0f, 0.0f));
	pTorch->SetDirection(glm::vec3(0.0f, 0.0f, 0.0f));

	//pTorch->SetAttenuation(1.0f, 0.022f, 0.0019f);	// 200 Units
	pTorch->SetAttenuation(1.0f, 0.22f, 0.20f);	// 20 Units
	pTorch->SetSpotCutOff(15.0f, 20.0f);

	// for debug light probes
	InitializeLightCube();
	InitializeQuadHDR();

	pCTerrain->LoadHeightMap("Resources/Raw/terrain0-16bbp-257x257.raw", 16, 257, 257);
	pCTerrain->InitializeTerrain(LoadShaders(TerrainDay), LoadShaders(TerrainNight));

	//////////////////////////////////////
	// Sky initialization

	pCSkyDome->InitializeSky(LoadShaders(SkyShader));
	pSunSphere->InitializeSphere(pCSkyDome->GetSunRadius(), 16, 16);

	// initialize screen filled quad
	glGenVertexArrays(1, &gVAOScreenFilledQuad);
	glBindVertexArray(gVAOScreenFilledQuad);

	// For position
	glGenBuffers(1, &gVBOPositionScreenFilledQuad);
	glBindBuffer(GL_ARRAY_BUFFER, gVBOPositionScreenFilledQuad);
	glBufferData(GL_ARRAY_BUFFER, sizeof(ScreenFilledQuad), ScreenFilledQuad, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_VERTEX, 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(OGL_ATTRIBUTE_VERTEX);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// For Texture
	glGenBuffers(1, &gVBOTexCoordsScreenFilledQuad);
	glBindBuffer(GL_ARRAY_BUFFER, gVBOTexCoordsScreenFilledQuad);
	glBufferData(GL_ARRAY_BUFFER, sizeof(ScreenFilledQuadTexCoords), ScreenFilledQuadTexCoords, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	//////////////////////////////////////

	//////////////////////////////////////
	glUseProgram(gShaderProgramObjectSunDepthTest);
	glUniform1i(glGetUniformLocation(gShaderProgramObjectSunDepthTest, "SunTexture"), 0);
	glUniform1i(glGetUniformLocation(gShaderProgramObjectSunDepthTest, "DepthTexture"), 1);
	glUseProgram(0);

	glUseProgram(gShaderProgramObjectSunRays);
	glUniform1i(glGetUniformLocation(gShaderProgramObjectSunRays, "LowBlurredSunTexture"), 0);
	glUniform1i(glGetUniformLocation(gShaderProgramObjectSunRays, "HighBlurredSunTexture"), 1);
	glUniform1f(glGetUniformLocation(gShaderProgramObjectSunRays, "Intensity"), 2.25f);
	glUniform3fv(glGetUniformLocation(gShaderProgramObjectSunRays, "Distortion"), 1, glm::value_ptr(glm::vec3(0.94f, 0.97f, 1.0f)));
	glUseProgram(0);

	glGenTextures(1, &DepthTexture);
	glGenTextures(4, SunTextures);

	glGenFramebuffers(1, &gFBOSun);

	////////////////////////////////////////
	/// Creating HDR framebuffers

	// configure floating point frame buffer
	glGenFramebuffers(1, &gFrameBufferObjectHDR);
	glGenRenderbuffers(1, &gRenderBufferObjectDepth);
	glGenTextures(1, &gColorBufferTexture);
	
	glBindTexture(GL_TEXTURE_2D, gColorBufferTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, giWinWidth, giWinHeight, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// create depth buffer (renderbuffer)
	glBindRenderbuffer(GL_RENDERBUFFER, gRenderBufferObjectDepth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, giWinWidth, giWinHeight);

	// attach buffers
	glBindFramebuffer(GL_FRAMEBUFFER, gFrameBufferObjectHDR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gColorBufferTexture, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, gRenderBufferObjectDepth);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		fopen_s(&pCCommon->pLogFile, "Logs/GeneralLog.txt", "a+");
		fprintf(pCCommon->pLogFile, "Framebuffer not complete!\n");
		fclose(pCCommon->pLogFile);
	}
	
	//glClearColor(0.3f, 0.0f, 0.5f, 1.0f);
	

	gPerspectiveProjectionMatrix = glm::mat4(1.0);
	gOrthographicProjectionMatrix = glm::mat4(1.0);

	///////////////////////////////////////////
	// Grass initilization
	pCGrass->InitGrass(pCPlayer, pCTerrain, LoadShaders(GrassBasic));
	///////////////////////////////////////////



	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);

	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	glEnable(GL_CULL_FACE);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_TEXTURE_2D);


	Resize(giWinWidth, giWinHeight);
	//Resize(WIN_WIDTH, WIN_HEIGHT);
}

void Display(void)
{
	void RenderScene(void);

	RenderScene();

	SwapBuffers(ghDC);
}

void RenderScene(void)
{
	void DrawPyramid(void);
	void DrawCube(void);
	void DrawGrid(void);
	void DrawGround(void);
	void DrawLightCube(glm::vec3 LightDiffuse);

	glClearColor(0.529f, 0.808f, 0.922f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Render scene into floating point framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, gFrameBufferObjectHDR);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//gPerspectiveProjectionMatrix = glm::perspective(glm::radians(gfFOVY), (GLfloat)giWinWidth / (GLfloat)giWinHeight, 0.1f, 5000.0f);
	glm::mat4 ModelMatrix = glm::mat4(1.0f);
	glm::mat4 TranslationMatrix = glm::mat4(1.0f);
	glm::mat4 RotationMatrix = glm::mat4(1.0f);
	glm::mat4 ScaleMatrix = glm::mat4(1.0f);
	glm::mat4 ViewMatrix = glm::mat4(1.0f);
	glm::mat4 ModelViewProjectionMatrix = glm::mat4(1.0f);

	if (gbIsPlaying == GL_FALSE)
	{
		CameraPosition = pCCamera->GetCameraPosition();
		ViewMatrix = pCCamera->GetViewMatrix();
		CameraFront = pCCamera->GetCameraFront();
		CameraUp = pCCamera->GetCameraUp();
		CameraLook = pCCamera->GetCameraFront();
		CameraRight = pCCamera->GetCameraRight();
	}
	else if (gbIsPlaying == GL_TRUE)
	{
		CameraPosition = pCPlayer->GetPlayerPosition();
		ViewMatrix = pCPlayer->GetViewMatrix();
		CameraFront = pCPlayer->GetCameraFront();
		CameraUp = pCPlayer->GetCameraUp();
		CameraLook = pCPlayer->GetCameraLook();
		CameraRight = pCPlayer->GetCameraSide();
	}

	GLfloat AmbientFactor = pCSkyDome->GetAmbientIntensity();
	GLfloat DiffuseFactor = pCSkyDome->GetDiffuseIntensity();

	if (AmbientFactor == 0.062500f && DiffuseFactor == 0.000000f)
	{
		gbIsNight = GL_TRUE;
	}
	else
	{
		gbIsNight = GL_FALSE;
	}

	// calculate the ambient and diffuse light color
	pSunLight->SetAmbient(glm::vec3(pCSkyDome->GetAmbientIntensity()));
	pSunLight->SetDiffuse(glm::vec3(pCSkyDome->GetLightColor() * pCSkyDome->GetDiffuseIntensity()));
	pSunLight->SetSpecular(glm::vec3(pCSkyDome->GetLightColor() * pCSkyDome->GetDiffuseIntensity()));
	pSunLight->SetDirection(glm::vec3(-pCSkyDome->GetSunWPos()));

	// calculate spotlight position and direction
	if (gbIsPlaying == GL_TRUE)
	{
		pTorch->SetPosition(pCPlayer->GetPlayerPosition());
		pTorch->SetDirection(pCPlayer->GetCameraFront());
	}
	else if (gbIsPlaying == GL_FALSE)
	{
		pTorch->SetPosition(pCCamera->GetCameraPosition());
		pTorch->SetDirection(pCCamera->GetCameraFront());
	}

	/////////////////////////////////////////////
	// Render the skydome
	
	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(CameraPosition.x, CameraPosition.y - pCSkyDome->GetInnerRadius(), CameraPosition.z));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;

	pCSkyDome->RenderSky(CameraPosition, ModelViewProjectionMatrix);

	////////////////////////////////////////////////////////

	///////////////////////////////////////////////////
	// Render the Terrain

	if (gbIsNight == GL_FALSE) 
	{
		glUseProgram(pCTerrain->GetShaderObject(TERRAIN_DAY));
	
		// Set Up the Sun Light for day time
		glUniform3fv(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_DAY), "SunLight.ambient"), 1, glm::value_ptr(pSunLight->GetAmbient()));
		glUniform3fv(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_DAY), "SunLight.diffuse"), 1, glm::value_ptr(pSunLight->GetDiffuse()));
		glUniform3fv(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_DAY), "SunLight.specular"), 1, glm::value_ptr(pSunLight->GetSpecular()));
		glUniform3fv(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_DAY), "SunLight.direction"), 1, glm::value_ptr(pSunLight->GetDirection()));

		ModelMatrix = glm::mat4(1.0f);
		TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
		ModelMatrix = ModelMatrix * TranslationMatrix;
		pCTerrain->RenderTerrain(ModelMatrix, ViewMatrix, gPerspectiveProjectionMatrix);

		glUseProgram(0);
	}
	else if (gbIsNight == GL_TRUE)
	{
		ModelMatrix = glm::mat4(1.0f);

		glUseProgram(pCTerrain->GetShaderObject(TERRAIN_NIGHT));

		glUniform1i(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_NIGHT), "num_lights"), MAX_POINT_LIGHTS);

		// send all point lights data
		for (int i = 0; i < MAX_POINT_LIGHTS; i++)
		{
			// light ads position
			strLightProperty.str("");
			strLightProperty << "PointLight[" << i << "].ambient";
			glUniform3fv(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_NIGHT), strLightProperty.str().c_str()), 1, glm::value_ptr(pLamp[i].GetAmbient()));

			strLightProperty.str("");
			strLightProperty << "PointLight[" << i << "].diffuse";
			glUniform3fv(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_NIGHT), strLightProperty.str().c_str()), 1, glm::value_ptr(pLamp[i].GetDiffuse()));

			strLightProperty.str("");
			strLightProperty << "PointLight[" << i << "].specular";
			glUniform3fv(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_NIGHT), strLightProperty.str().c_str()), 1, glm::value_ptr(pLamp[i].GetSpecular()));

			strLightProperty.str("");
			strLightProperty << "PointLight[" << i << "].position";
			glUniform3fv(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_NIGHT), strLightProperty.str().c_str()), 1, glm::value_ptr(pLamp[i].GetPosition()));

			// attenuation
			strLightProperty.str("");
			strLightProperty << "PointLight[" << i << "].constant_attenuation";
			glUniform1f(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_NIGHT), strLightProperty.str().c_str()), pLamp[i].GetConstantAttenuation());

			strLightProperty.str("");
			strLightProperty << "PointLight[" << i << "].linear_attenuation";
			glUniform1f(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_NIGHT), strLightProperty.str().c_str()), pLamp[1].GetLinearAttenuation());

			strLightProperty.str("");
			strLightProperty << "PointLight[" << i << "].quadratic_attenuation";
			glUniform1f(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_NIGHT), strLightProperty.str().c_str()), pLamp[i].GetQuadraticAttenuation());
		}

		// send spotlight data
		glUniform3fv(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_NIGHT), "Torch.ambient"), 1, glm::value_ptr(pTorch->GetAmbient()));
		glUniform3fv(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_NIGHT), "Torch.diffuse"), 1, glm::value_ptr(pTorch->GetDiffuse()));
		glUniform3fv(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_NIGHT), "Torch.specular"), 1, glm::value_ptr(pTorch->GetSpecular()));
		glUniform3fv(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_NIGHT), "Torch.position"), 1, glm::value_ptr(pTorch->GetPosition()));
		glUniform3fv(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_NIGHT), "Torch.direction"), 1, glm::value_ptr(pTorch->GetDirection()));

		glUniform1f(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_NIGHT), "Torch.constant_attenuation"), pTorch->GetConstantAttenuation());
		glUniform1f(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_NIGHT), "Torch.linear_attenuation"), pTorch->GetLinearAttenuation());
		glUniform1f(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_NIGHT), "Torch.quadratic_attenuation"), pTorch->GetQuadraticAttenuation());

		glUniform1f(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_NIGHT), "Torch.inner_cutoff"), glm::cos(glm::radians(pTorch->GetInnerSpotCutOff())));
		glUniform1f(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_NIGHT), "Torch.outer_cutoff"), glm::cos(glm::radians(pTorch->GetOuterSpotCutOff())));

		glUniform3fv(glGetUniformLocation(pCTerrain->GetShaderObject(TERRAIN_NIGHT), "u_view_position"), 1, glm::value_ptr(CameraPosition));
		pCTerrain->RenderTerrainNight(ModelMatrix, ViewMatrix, gPerspectiveProjectionMatrix);
		
		glUseProgram(0);

		////////////////////////////////////////////
		// render light cubes

		ModelMatrix = glm::mat4(1.0f);

		glUseProgram(gShaderProgramObjectBasicShader);

		for (int i = 0; i < MAX_POINT_LIGHTS; i++)
		{
			TranslationMatrix = glm::translate(glm::mat4(1.0f), pLamp[i].GetPosition());
			ModelMatrix = ModelMatrix * TranslationMatrix;

			ScaleMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.2f, 0.2f, 0.2f));
			ModelMatrix = ModelMatrix * ScaleMatrix;

			ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;
			glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShader, "u_mvp_matrix"), 1, GL_FALSE, glm::value_ptr(ModelViewProjectionMatrix));

			DrawLightCube(pLamp[i].GetDiffuse());

			ModelMatrix = glm::mat4(1.0f);		
			ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;
			glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShader, "u_mvp_matrix"), 1, GL_FALSE, glm::value_ptr(ModelViewProjectionMatrix));
		}
		glUseProgram(0);
	}

	/////////////////////////////////////////////
	// render grass
	pCGrass->RenderGrass(ModelMatrix, ViewMatrix, gPerspectiveProjectionMatrix);

	/////////////////////////////////////////////

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	

	//////////////////////////////////////////////
	// Now render the floating point color buffer to 2D quad and tonemap HDR colors to default framebuffer's (clamped) color range

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(gShaderObjectHDR);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gColorBufferTexture);
	glUniform1i(glGetUniformLocation(gShaderObjectHDR, "HDRTexture"), 0);
	
	glUniform1f(glGetUniformLocation(gShaderObjectHDR, "exposure"), 2.5f);
	RenderQuadHDR();
	glUseProgram(0);

		

		

	
	//////////////////////////////////////
	// render sun

	// copy the depth buffer to the texture
	//glBindTexture(GL_TEXTURE_2D, DepthTexture);
	//glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, giWinWidth, giWinHeight);
	//glBindTexture(GL_TEXTURE_2D, 0);

	//bool CalculateSunRays = false;
	//int Test = 0;
	//int Tests = 16;
	//float Angle = 0.0f;
	//float AngleInc = 360.0f / Tests;

	////glm::mat4 BiasMatrix = glm::mat4(glm::vec4(0.5f, 0.0f, 0.0f, 0.0f), glm::vec4(0.0f, 0.5f, 0.0f, 0.0f), glm::vec4(0.0f, 0.0f, 0.5f, 0.0f), glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));
	//glm::mat4x4 BiasMatrix = glm::mat4x4(0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.5f, 0.5f, 0.5f, 1.0f);
	//glm::mat4 ViewProjectionBiasMatrix = BiasMatrix * gPerspectiveProjectionMatrix * ViewMatrix;

	//glm::vec3 SunWPos = pCSkyDome->GetSunWPos();
	//GLfloat SunR = pCSkyDome->GetSunRadius();

	//while (Test < Tests && !CalculateSunRays)
	//{
	//	glm::vec3 SunWEdge = SunWPos + glm::rotate(CameraRight, Angle, CameraFront) * SunR;

	//	if (SunWEdge.y - CameraPosition.y > 0.0f)
	//	{
	//		glm::vec4 SunPosProj = ViewProjectionBiasMatrix * glm::vec4(SunWEdge, 1.0f);
	//		SunPosProj /= SunPosProj.w;

	//		CalculateSunRays |= (SunPosProj.x >= 0.0f && SunPosProj.x <= 1.0f && SunPosProj.y >= 0.0f && SunPosProj.y <= 1.0f && SunPosProj.z >= 0.0f && SunPosProj.z <= 1.0f);
	//	}

	//	Angle += AngleInc;
	//	Test++;
	//}

	//

	//// calculate sunrays
	//if (CalculateSunRays)
	//{
	//	glm::vec4 SunPosProj = ViewProjectionBiasMatrix * glm::vec4(SunWPos, 1.0f);
	//	SunPosProj /= SunPosProj.w;

	//	glViewport(0, 0, giSunTextureWidth, giSunTextureHeight);

	//	// render sun sphere
	//	glBindFramebuffer(GL_FRAMEBUFFER, gFBOSun);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, SunTextures[1], 0);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

	//	glClear(GL_COLOR_BUFFER_BIT);

	//	glUseProgram(gShaderProgramObjectBasicShader);
	//	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(SunWPos.x, SunWPos.y, SunWPos.z));
	//	ModelMatrix = ModelMatrix * TranslationMatrix;

	//	ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;
	//	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShader, "u_mvp_matrix"), 1, GL_FALSE, glm::value_ptr(ModelViewProjectionMatrix));

	//	glVertexAttrib3fv(OGL_ATTRIBUTE_COLOR, glm::value_ptr(pCSkyDome->GetSunColor()));
	//	pSunSphere->DrawSphere();
	//	glUseProgram(0);

	//	// test of sun sphere is behind the scene geometry
	//	glBindFramebuffer(GL_FRAMEBUFFER, gFBOSun);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, SunTextures[0], 0);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

	//	glUseProgram(gShaderProgramObjectSunDepthTest);

	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, SunTextures[1]);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectSunDepthTest, "SunTexture"), 0);

	//	glActiveTexture(GL_TEXTURE1);
	//	glBindTexture(GL_TEXTURE_2D, DepthTexture);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectSunDepthTest, "DepthTexture"), 1);

	//	glBindVertexArray(gVAOScreenFilledQuad);
	//	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	//	glBindVertexArray(0);
	//	glUseProgram(0);

	//	glActiveTexture(GL_TEXTURE1);
	//	glBindTexture(GL_TEXTURE_2D, 0);

	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, 0);

	//	// blur sun sphere horizontally (low)
	//	glBindFramebuffer(GL_FRAMEBUFFER, gFBOSun);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, SunTextures[3], 0);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

	//	glUseProgram(gShaderProgramObjectBlurHorizontal);
	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, SunTextures[0]);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectBlurHorizontal, "Texture"), 0);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectBlurHorizontal, "Width"), 1);
	//	glBindVertexArray(gVAOScreenFilledQuad);
	//	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	//	glBindVertexArray(0);
	//	glUseProgram(0);

	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, 0);

	//	// blur sun sphere vertically (low)
	//	glBindFramebuffer(GL_FRAMEBUFFER, gFBOSun);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, SunTextures[1], 0);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

	//	glUseProgram(gShaderProgramObjectBlurVertical);
	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, SunTextures[3]);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectBlurVertical, "Texture"), 0);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectBlurVertical, "Width"), 1);
	//	glBindVertexArray(gVAOScreenFilledQuad);
	//	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	//	glBindVertexArray(0);
	//	glUseProgram(0);

	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, 0);

	//	// blur sun sphere horizontally (high)
	//	glBindFramebuffer(GL_FRAMEBUFFER, gFBOSun);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, SunTextures[3], 0);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

	//	glUseProgram(gShaderProgramObjectBlurHorizontal);
	//	
	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, SunTextures[0]);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectBlurHorizontal, "Texture"), 0);
	//	
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectBlurHorizontal, "Width"), 10);
	//	glBindVertexArray(gVAOScreenFilledQuad);
	//	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	//	glBindVertexArray(0);
	//	glUseProgram(0);

	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, 0);

	//	// blur sun sphere vertically (high)
	//	glBindFramebuffer(GL_FRAMEBUFFER, gFBOSun);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, SunTextures[2], 0);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

	//	glUseProgram(gShaderProgramObjectBlurVertical);
	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, SunTextures[3]);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectBlurVertical, "Texture"), 0);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectBlurVertical, "Width"), 10);
	//	glBindVertexArray(gVAOScreenFilledQuad);
	//	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	//	glBindVertexArray(0);
	//	glUseProgram(0);

	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, 0);

	//	// blur sun sphere radially and calculate sunrays
	//	glBindFramebuffer(GL_FRAMEBUFFER, gFBOSun);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, SunTextures[3], 0);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

	//	glUseProgram(gShaderProgramObjectSunRays);
	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, SunTextures[1]);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectSunDepthTest, "LowBlurredSunTexture"), 0);

	//	glActiveTexture(GL_TEXTURE1);
	//	glBindTexture(GL_TEXTURE_2D, SunTextures[2]);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectSunDepthTest, "HighBlurredSunTexture"), 1);

	//	glUniform2fv(glGetUniformLocation(gShaderProgramObjectSunDepthTest, "SunPosProj"), 1, glm::value_ptr(glm::vec2(SunPosProj)));

	//	glBindVertexArray(gVAOScreenFilledQuad);
	//	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	//	glBindVertexArray(0);

	//	glUseProgram(0);

	//	glActiveTexture(GL_TEXTURE1);
	//	glBindTexture(GL_TEXTURE_2D, 0);

	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, 0);

	//	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//	glViewport(0, 0, giWinWidth, giWinHeight);
	//}

	//gOrthographicProjectionMatrix = glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f);
	//ModelMatrix = glm::mat4(1.0f);

	//if (CalculateSunRays)
	//{
	//	glUseProgram(gShaderProgramObjectBasicShaderTexture);
	//	ModelViewProjectionMatrix = gOrthographicProjectionMatrix * ViewMatrix * ModelMatrix;
	//	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShaderTexture, "u_mvp_matrix"), 1, GL_FALSE, glm::value_ptr(ModelViewProjectionMatrix));

	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, SunTextures[3]);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectBasicShaderTexture, "SunTexture"), 0);

	//	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);
	//	glEnable(GL_BLEND);

	//	glBindVertexArray(gVAOScreenFilledQuad);
	//	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	//	glBindVertexArray(0);

	//	glDisable(GL_BLEND);
	//	glUseProgram(0);
	//	glBindTexture(GL_TEXTURE_2D, 0);
	//}
}

void Update(void)
{
	if (gbIsDayNightCyclePaused == GL_FALSE)
		pCSkyDome->UpdateSky(gfDeltaTime);

	gfAnglePyramid += 0.001f;
	if (gfAnglePyramid >= 360.0f)
		gfAnglePyramid = 0.0f;

	gfAngleCube += 0.1f;
	if (gfAngleCube >= 360.0f)
		gfAngleCube = 0.0f;

	//pCPlayer->gfGroundHeight = pCTerrain->GetHeightAt(pCPlayer->GetPlayerPosition());
	pCPlayer->PlayerUpdate(gfDeltaTime);

	pCGrass->ControlGrassWaving(gfDeltaTime);
}

void Resize(int Width, int Height)
{
	glm::mat4 SetInfiniteProjectionMatrix(GLfloat Left, GLfloat Right, GLfloat Bottom, GLfloat Top, GLfloat Near, GLfloat Far);

	GLfloat fLeft = 0.0f;
	GLfloat fRight = 0.0f;
	GLfloat fBottom = 0.0f;
	GLfloat fTop = 0.0f;
	GLfloat fNear = FRUSTUM_NEAR;
	GLfloat fFar = FRUSTUM_FAR;


	if (Height == 0)
		Height = 1;

	if (Width == 0)
		Width = 1;

	//if (Width > Height) 
	//{
	//	glViewport(0, (Height - Width) / 2, (GLsizei)Width, (GLsizei)Width);
	//}
	//else 
	//{
	//	glViewport((Width - Height) / 2, 0, (GLsizei)Height, (GLsizei)Height);
	//}

	glViewport(0, 0, (GLsizei)Width, (GLsizei)Height);

	fTop = fNear * (GLfloat)tan(gfFOVY / 360.0 * glm::pi<float>());
	fRight = fTop * ((float)Width / (float)Height);

	fBottom = -fTop;
	fLeft = -fTop * ((float)Width / (float)Height);

	gPerspectiveProjectionMatrix = SetInfiniteProjectionMatrix(fLeft, fRight, fBottom, fTop, fNear, fFar);

	/*glBindTexture(GL_TEXTURE_2D, DepthTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, Width, Height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	giSunTextureWidth = Width / 2;
	giSunTextureHeight = Height / 2;

	for (int i = 0; i < 4; i++)
	{
		glBindTexture(GL_TEXTURE_2D, SunTextures[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, giSunTextureWidth, giSunTextureHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	glUseProgram(gShaderProgramObjectBlurHorizontal);
	glUniform1f(glGetUniformLocation(gShaderProgramObjectBlurHorizontal, "odw"), 1.0f / (float)giSunTextureWidth);
	glUseProgram(gShaderProgramObjectBlurVertical);
	glUniform1f(glGetUniformLocation(gShaderProgramObjectBlurVertical, "odh"), 1.0f / (float)giSunTextureHeight);
	glUseProgram(0);*/


	glBindTexture(GL_TEXTURE_2D, gColorBufferTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, giWinWidth, giWinHeight, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// create depth buffer (renderbuffer)
	glBindRenderbuffer(GL_RENDERBUFFER, gRenderBufferObjectDepth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, giWinWidth, giWinHeight);

	// attach buffers
	glBindFramebuffer(GL_FRAMEBUFFER, gFrameBufferObjectHDR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gColorBufferTexture, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, gRenderBufferObjectDepth);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		fopen_s(&pCCommon->pLogFile, "Logs/GeneralLog.txt", "a+");
		fprintf(pCCommon->pLogFile, "Framebuffer not complete!\n");
		fclose(pCCommon->pLogFile);
	}
}

glm::mat4 SetInfiniteProjectionMatrix(GLfloat Left, GLfloat Right, GLfloat Bottom, GLfloat Top, GLfloat Near, GLfloat Far)
{
	glm::mat4 result(glm::mat4(1.0f));

	if ((Right == Left) || (Top == Bottom) || (Near == Far) || (Near < 0.0) || (Far < 0.0))
		return result;

	result[0][0] = (2.0f * Near) / (Right - Left);
	result[1][1] = (2.0f * Near) / (Top - Bottom);

	result[2][0] = (Right + Left) / (Right - Left);
	result[2][1] = (Top + Bottom) / (Top - Bottom);
	//result[2][2] = -(Far + Near) / (Far - Near);
	result[2][2] = -1.0f;
	result[2][3] = -1.0f;

	//result[3][2] = -(2.0f * Far * Near) / (Far - Near);
	result[3][2] = -(2.0f * Near);
	result[3][3] = 0.0f;

	return result;
}

void Uninitialize(void)
{
	pCTerrain->DeleteTerrain();

	if (gbIsFullscreen == true)
	{
		dwStyle = GetWindowLong(ghWnd, GWL_STYLE);
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghWnd, &wpPrev);
		SetWindowPos(ghWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
	}

	if (gVAOPyramid)
	{
		glDeleteVertexArrays(1, &gVAOPyramid);
		gVAOPyramid = 0;
	}

	if (gVAOCube)
	{
		glDeleteVertexArrays(1, &gVAOCube);
		gVAOCube = 0;
	}

	if (gVBOPosition)
	{
		glDeleteVertexArrays(1, &gVBOPosition);
		gVBOPosition = 0;
	}

	if (gVBOColor)
	{
		glDeleteVertexArrays(1, &gVBOColor);
		gVBOColor = 0;
	}

	//delete[] pCCamera;
	//delete[] pCPlayer;
	//delete[] pCTerrain;
	//delete[] pCSkyDome;
	//delete[] pCGrass;

	glUseProgram(0);

	wglMakeCurrent(NULL, NULL);

	wglDeleteContext(ghRC);
	ghRC = NULL;

	ReleaseDC(ghWnd, ghDC);
	ghDC = NULL;

	

	if (pCCommon->pLogFile)
	{
		fopen_s(&pCCommon->pLogFile, "Logs/GeneralLog.txt", "a+");
		fprintf(pCCommon->pLogFile, "Log File Is Successfully Closed.\n");
		fclose(pCCommon->pLogFile);
		pCCommon->pLogFile = NULL;
	}

	DestroyWindow(ghWnd);
}


void InitializeGrid(GLdouble fExt, GLdouble fSt)
{
	// ground grid
	GLdouble fExtent = FloorExtent;
	GLdouble fStep = StepSize;
	GLdouble y = 0.0f;
	GLdouble iLine;

	GLdouble i = std::fmod(FloorExtent, StepSize);
	if (i != 0)
	{
		Uninitialize();
		exit(0);
	}

	NumGridVertices = (((FloorExtent * 2) / StepSize) + 1) * 2 * 6;
	GLdouble fGridVertices[(((FloorExtent * 2) / StepSize) + 1) * 2 * 6];
	GLfloat fGridColor[(((FloorExtent * 2) / StepSize) + 1) * 2 * 6];

	static int VertexCounter = 0, ColorCounter = 0;

	for (iLine = -fExtent; iLine <= fExtent; iLine += fStep)
	{
		fGridVertices[VertexCounter++] = iLine;
		fGridVertices[VertexCounter++] = y;
		fGridVertices[VertexCounter++] = fExtent;

		fGridVertices[VertexCounter++] = iLine;
		fGridVertices[VertexCounter++] = y;
		fGridVertices[VertexCounter++] = -fExtent;

		fGridVertices[VertexCounter++] = fExtent;
		fGridVertices[VertexCounter++] = y;
		fGridVertices[VertexCounter++] = iLine;

		fGridVertices[VertexCounter++] = -fExtent;
		fGridVertices[VertexCounter++] = y;
		fGridVertices[VertexCounter++] = iLine;

		//glVertex3d(iLine, y, fExtent);    // Draw Z lines			
		//glVertex3d(iLine, y, -fExtent);

		//glVertex3d(fExtent, y, iLine);
		//glVertex3d(-fExtent, y, iLine);

		//fprintf_s(gpFile, "%f %f %f\n", iLine, y, fExtent);
		//fprintf_s(gpFile, "%f %f %f\n\n", iLine, y, -fExtent);

		//fprintf_s(gpFile, "%f %f %f\n", fExtent, y, iLine);
		//fprintf_s(gpFile, "%f %f %f\n\n", -fExtent, y, iLine);
	}

	/*fprintf_s(gpFile, "\n\n\n\n fGridVertices\n\n");
	for (int i = 0; i < NumGridVertices; i += 12)
	{
	fprintf_s(gpFile, "%f, %f, %f,\n", fGridVertices[i], fGridVertices[i + 1], fGridVertices[i + 2]);
	fprintf_s(gpFile, "%f, %f, %f,\n\n", fGridVertices[i + 3], fGridVertices[i + 4], fGridVertices[i + 5]);

	fprintf_s(gpFile, "%f, %f, %f,\n", fGridVertices[i + 6], fGridVertices[i + 7], fGridVertices[i + 8]);
	fprintf_s(gpFile, "%f, %f, %f,\n\n", fGridVertices[i + 9], fGridVertices[i + 10], fGridVertices[i + 11]);
	}*/

	for (int i = 0; i < NumGridVertices; i++)
	{
		fGridColor[i] = 0.5f;
	}


	// For grid
	glGenVertexArrays(1, &gVAOLineGrid);
	glBindVertexArray(gVAOLineGrid);

	// For grid position
	glGenBuffers(1, &gVBOPosition);

	glBindBuffer(GL_ARRAY_BUFFER, gVBOPosition);
	glBufferData(GL_ARRAY_BUFFER, sizeof(fGridVertices), fGridVertices, GL_STATIC_DRAW);

	glVertexAttribPointer(OGL_ATTRIBUTE_VERTEX, 3, GL_DOUBLE, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(OGL_ATTRIBUTE_VERTEX);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// For grid color

	glGenBuffers(1, &gVBOColor);

	glBindBuffer(GL_ARRAY_BUFFER, gVBOColor);
	glBufferData(GL_ARRAY_BUFFER, sizeof(fGridColor), fGridColor, GL_STATIC_DRAW);

	glVertexAttribPointer(OGL_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(OGL_ATTRIBUTE_COLOR);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);
}


void PyramidInitialization(void)
{
	// Pyramid vertices
	const GLfloat PyramidVertices[] =
	{
		// Front Face
		0.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,

		// Right Face
		0.0f, 1.0f, 0.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f,

		// Back Face
		0.0f, 1.0f, 0.0f,
		1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,

		// Left Face
		0.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f
	};

	const GLfloat PyramidNormals[] =
	{
		0.0f, 0.447214f, 0.894427f,
		0.0f, 0.447214f, 0.894427f,
		0.0f, 0.447214f, 0.894427f,

		0.894427f, 0.447214f, 0.0f,
		0.894427f, 0.447214f, 0.0f,
		0.894427f, 0.447214f, 0.0f,

		0.0f, 0.447214f, -0.894427f,
		0.0f, 0.447214f, -0.894427f,
		0.0f, 0.447214f, -0.894427f,

		-0.894427f, 0.447214f, 0.0f,
		-0.894427f, 0.447214f, 0.0f,
		-0.894427f, 0.447214f, 0.0f
	};

	const GLfloat PyramidTexcoords[] =
	{
		0.5f, 1.0f,	// Front apex
		0.0f, 0.0f,	// Front bottom left
		1.0f, 0.0f,	// Front bottom right

		0.5f, 1.0f, // Right apex
		1.0f, 0.0f, // Right bottom left
		0.0f, 0.0f,	// Right bottom right

		0.5f, 1.0f,	// Back apex
		1.0f, 0.0f,	// Back bottom left
		0.0f, 0.0f,	// Back bottom right

		0.5f, 1.0f,	// Left apex
		0.0f, 0.0f,	// Left bottom left
		1.0f, 0.0f	// Left bottom right
	};

	// For Triangle
	glGenVertexArrays(1, &gVAOPyramid);
	glBindVertexArray(gVAOPyramid);

	// For triangle position
	glGenBuffers(1, &gVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, gVBOPosition);
	glBufferData(GL_ARRAY_BUFFER, sizeof(PyramidVertices), PyramidVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(OGL_ATTRIBUTE_VERTEX);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// For triangle normal

	glGenBuffers(1, &gVBONormal);
	glBindBuffer(GL_ARRAY_BUFFER, gVBONormal);
	glBufferData(GL_ARRAY_BUFFER, sizeof(PyramidNormals), PyramidNormals, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(OGL_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// For triangle Texture
	glGenBuffers(1, &gVBOTexture);
	glBindBuffer(GL_ARRAY_BUFFER, gVBOTexture);
	glBufferData(GL_ARRAY_BUFFER, sizeof(PyramidTexcoords), PyramidTexcoords, GL_STATIC_DRAW);

	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void InitializeLightCube(void)
{
	const GLfloat LightCubeVertices[] =
	{
		// Front Face
		1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,

		// Right Face
		1.0f, 1.0f, -1.0f,
		1.0f, 1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f,

		// Back Face
		-1.0f, 1.0f, -1.0f,
		1.0f, 1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,

		// Left Face
		-1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f,

		// Top Face
		1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,

		// Bottom Face
		1.0f, -1.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,
		-1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f
	};

	// For Quad
	glGenVertexArrays(1, &gVAOLightCube);
	glBindVertexArray(gVAOLightCube);

	// For Quad position
	glGenBuffers(1, &gVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, gVBOPosition);
	glBufferData(GL_ARRAY_BUFFER, sizeof(LightCubeVertices), LightCubeVertices, GL_STATIC_DRAW);

	glVertexAttribPointer(OGL_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(OGL_ATTRIBUTE_VERTEX);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void DrawLightCube(glm::vec3 LightDiffuse)
{
	glBindVertexArray(gVAOLightCube);

	glVertexAttrib3fv(OGL_ATTRIBUTE_COLOR, glm::value_ptr(LightDiffuse));

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 20, 4);
	glBindVertexArray(0);
}

void InitializeQuadHDR(void)
{
	float quadVertices[] = {
		// positions        // texture Coords
		-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
		1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
		1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
	};
	// setup plane VAO
	glGenVertexArrays(1, &gVAOHDRQuad);
	glBindVertexArray(gVAOHDRQuad);

	glGenBuffers(1, &gVBOHDR);
	glBindBuffer(GL_ARRAY_BUFFER, gVBOHDR);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_VERTEX);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);
}

void RenderQuadHDR(void)
{	
	glBindVertexArray(gVAOHDRQuad);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}

void OnLButtonDown(INT LeftClickX, INT LeftClickY, DWORD LeftClickFlags)
{
	RECT Clip;
	//RECT PrevClip;

	SetCapture(ghWnd);
	//GetClipCursor(&PrevClip);
	GetWindowRect(ghWnd, &Clip);
	ClipCursor(&Clip);

	GetCursorPos(&ClickPoint);
	//ShowCursor(FALSE);
}

void OnLButtonUp(INT LeftUpX, INT LeftUpY, DWORD LeftUpFlags)
{
	ClipCursor(NULL);
	ReleaseCapture();
	SetCursorPos(ClickPoint.x, ClickPoint.y);
	ShowCursor(TRUE);
}

void OnRButtonDown(INT RightClickX, INT RightClickY, DWORD RightClickFlags)
{
	RECT Clip;
	//RECT PrevClip;

	SetCapture(ghWnd);
	//GetClipCursor(&PrevClip);
	GetWindowRect(ghWnd, &Clip);
	ClipCursor(&Clip);

	GetCursorPos(&ClickPoint);
	ShowCursor(FALSE);
}

void OnRButtonUp(INT RightUpX, INT RightUpY, DWORD RightUpFlags)
{
	ClipCursor(NULL);
	ReleaseCapture();
	SetCursorPos(ClickPoint.x, ClickPoint.y);
	ShowCursor(TRUE);
}

void OnMButtonDown(INT MiddleDownX, INT MiddleDownY, DWORD MiddleDownFlags)
{
	RECT Clip;
	//RECT PrevClip;

	SetCapture(ghWnd);
	//GetClipCursor(&PrevClip);
	GetWindowRect(ghWnd, &Clip);
	ClipCursor(&Clip);

	GetCursorPos(&ClickPoint);
	ShowCursor(FALSE);
}

void OnMButtonUp(INT MiddleUpX, INT MiddleUpY, DWORD MiddleUpFlags)
{
	ClipCursor(NULL);
	ReleaseCapture();
	SetCursorPos(ClickPoint.x, ClickPoint.y);
	ShowCursor(TRUE);
}

void OnMouseMove(INT MouseMoveX, INT MouseMoveY, DWORD Flags)
{
	//fprintf(stream, "+X Threshold : %f -X Threshold : %f +Y Threshold : %f -Y Threshold : %f\n", iPositiveXThreshold, iNegativeXThreshold, iPositiveYThreshold, iNegativeYThreshold);
	//fprintf(stream, "On Mouse Move X : %d Y : %d\n", MouseMoveX, MouseMoveY);
	//ShowCursor(FALSE);


	if (Flags & MK_LBUTTON)
	{
		// Code
	}

	if (gbIsPlaying == GL_FALSE)
	{
		if (gbFirstMouse)
		{
			gfLastX = (GLfloat)MouseMoveX;
			gfLastY = (GLfloat)MouseMoveY;
			gbFirstMouse = GL_FALSE;
		}

		GLfloat fXOffset = MouseMoveX - gfLastX;
		GLfloat fYOffset = gfLastY - MouseMoveY;

		gfLastX = (GLfloat)MouseMoveX;
		gfLastY = (GLfloat)MouseMoveY;

		if (Flags & MK_RBUTTON)
		{
			pCCamera->ProcessMouseMovement(fXOffset, fYOffset, GL_TRUE);
		}
	}
	else if (gbIsPlaying == GL_TRUE)
	{
		if (gbFirstMouse)
		{
			gfLastX = (GLfloat)MouseMoveX;
			gfLastY = (GLfloat)MouseMoveY;
			gbFirstMouse = GL_FALSE;
		}

		//Player->ControlMouseInput(MouseMoveX, MouseMoveY);
	}
	ShowCursor(TRUE);
}

void OnMouseWheelScroll(short zDelta)
{
	void Resize(int Width, int Height);

	if (zDelta > 0)
	{
		gfFOVY -= ZOOM_FACTOR;
		if (gfFOVY <= 0.1f)
			gfFOVY = 0.1f;
	}
	else
	{
		gfFOVY += ZOOM_FACTOR;
		if (gfFOVY >= 45.0f)
			gfFOVY = 45.0f;
	}

	Resize(giWinWidth, giWinHeight);
}


void ScreenPositionToWorldRay(int MouseX, int MouseY, int ScreenWidth, int ScreenHeight, glm::mat4 ViewMatrix, glm::mat4 ProjectionMatrix, glm::vec3 &OutRayOrigin, glm::vec3 &OutRayDirection)
{
	// the ray start and end position in NDC
	glm::vec4 RayStartNDC(((float)MouseX / (float)ScreenWidth - 0.5f) * 2.0f, ((float)MouseY / (float)ScreenWidth - 0.5f) * 2.0f, -1.0f, 1.0f);
	glm::vec4 RayEndNDC(((float)MouseX / (float)ScreenWidth - 0.5f) * 2.0f, ((float)MouseY / (float)ScreenWidth - 0.5f) * 2.0f, 0.0f, 1.0f);

	// the projection matrix goes from camera space to NDC
	// hence to go from NDC to cmaera space find inverse of projection matrix
	glm::mat4 InverseProjectionMatrix = glm::inverse(ProjectionMatrix);

	// the view matrix goes from world space to camera space
	// hence to go from camera space to world space find inverse of view matrix
	glm::mat4 InverseViewMatrix = glm::inverse(ViewMatrix);

	glm::vec4 RayStartCamera = InverseProjectionMatrix * RayStartNDC;
	RayStartCamera /= RayStartCamera.w;

	glm::vec4 RayStartWorld = InverseViewMatrix * RayStartCamera;
	RayStartWorld /= RayStartWorld.w;

	glm::vec4 RayEndCamera = InverseProjectionMatrix * RayEndNDC;
	RayEndCamera /= RayEndCamera.w;

	glm::vec4 RayEndWorld = InverseViewMatrix * RayEndCamera;
	RayEndWorld /= RayEndWorld.w;

	// ray direction in world
	glm::vec3 RayDirWorld(RayEndWorld - RayStartWorld);
	RayDirWorld = glm::normalize(RayDirWorld);

	OutRayOrigin = glm::vec3(RayStartWorld);
	OutRayDirection = glm::normalize(RayDirWorld);
}

// parameters
// ray origin, in world space
// ray direction (not target position), in world space
// minimum x, y, z coordinates of the mesh (axis aligned bounding box) when not transformed
// maximum x, y, z coordinates. Often AABBMin * -1 if your mesh is centered (but this is not always a case)
// model matrix for transformation applied to the mesh so that bounding box will be transformed accordingly
// output: distance between ray origin and the intersection with the bounding box
bool TestRayOBBIntersection(glm::vec3 RayOrigin, glm::vec3 RayDirection, glm::vec3 AABBMin, glm::vec3 AABBMax, glm::mat4 ModelMatrix, float &IntersectionDistance)
{
	float tMin = 0.0f;
	float tMax = 100000.0f;

	// oriented bounding box position in world space
	glm::vec3 OBBPositionWorldSpace(ModelMatrix[3].x, ModelMatrix[3].y, ModelMatrix[3].z);

	glm::vec3 Delta = OBBPositionWorldSpace - RayOrigin;

	// test intersection with the 2 planes perpendicular to the OBB's X axis
	{
		glm::vec3 XAxis(ModelMatrix[0].x, ModelMatrix[0].y, ModelMatrix[0].z);
		float e = glm::dot(XAxis, Delta);
		float f = glm::dot(RayDirection, XAxis);

		if (fabs(f) > 0.001f)
		{
			float t1 = (e + AABBMin.x) / f; // intersection with the left plane
			float t2 = (e + AABBMax.x) / f; // intersection with the right plane
											// t1 and t2 now contain distance between ray origin and rat-plane intersections

											// t1 to represent the nearest intersection,
											// if it's not the case, invert t1 and t2
			if (t1 > t2)
			{
				// swap t1 and t2
				float w = t1;
				t1 = t2;
				t2 = w;
			}

			// tMax is the nearest "far" intersection (amongst the x, y and z planes pair)
			if (t2 < tMax)
				tMax = t2;

			// tMin is the farthest "near" intersection (amongst the x, y and z planes pair)
			if (t1 > tMin)
				tMin = t1;

			// if "far" is closer than "near", then there is nointersection
			if (tMax < tMin)
				return false;
		}
		else
		{
			// if the ray is almost parallel to the planes, so they don't have any intersection
			if (-e + AABBMin.x > 0.0f || -e + AABBMax.x < 0.0f)
				return false;
		}
	}

	// test intersection with the 2 planes perpendicular to the OBB's Y axis
	{
		glm::vec3 YAxis(ModelMatrix[0].x, ModelMatrix[0].y, ModelMatrix[0].z);
		float e = glm::dot(YAxis, Delta);
		float f = glm::dot(RayDirection, YAxis);

		if (fabs(f) > 0.001f)
		{
			float t1 = (e + AABBMin.y) / f;
			float t2 = (e + AABBMax.y) / f;

			if (t1 > t2)
			{
				float w = t1;
				t1 = t2;
				t2 = w;
			}

			if (t2 < tMax)
				tMax = t2;

			if (t1 > tMin)
				tMin = t1;

			if (tMin > tMax)
				return false;
		}
		else
		{
			if (-e + AABBMin.y > 0.0f || -e + AABBMax.y < 0.0f)
				return false;
		}
	}

	// test intersection with the 2 planes perpendicular to the OBB's Z axis
	{
		glm::vec3 ZAxis(ModelMatrix[2].x, ModelMatrix[0].y, ModelMatrix[0].z);
		float e = glm::dot(ZAxis, Delta);
		float f = glm::dot(RayDirection, ZAxis);

		if (fabs(f) > 0.001f)
		{
			float t1 = (e + AABBMin.z) / f;
			float t2 = (e + AABBMax.z) / f;

			if (t1 > t2)
			{
				float w = t1;
				t1 = t2;
				t2 = w;
			}

			if (t2 < tMax)
				tMax = t2;

			if (t1 > tMin)
				tMin = t1;

			if (tMin > tMax)
				return false;
		}
		else
		{
			if (-e + AABBMin.z > 0.0f || -e + AABBMax.z < 0.0f)
				return false;
		}
	}

	IntersectionDistance = tMin;
	return true;
}
