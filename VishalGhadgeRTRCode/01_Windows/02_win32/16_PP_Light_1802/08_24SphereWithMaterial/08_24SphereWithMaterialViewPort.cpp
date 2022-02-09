#if 1
#include <Windows.h>
#include<stdio.h>
#include<gl\glew.h>
#include <gl\GL.h>
#include "..\..\include\vmath.h"
#include "..\..\include\Sphere.h"

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "Sphere.lib")

#define WIN_WIDTH	800
#define WIN_HEIGHT	600

using namespace vmath;

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

VOID resize(int, int);

int g_iCurrentWidth;
int g_iCurrentHeight;

//
//	Global variable.
//
HWND g_hWnd;
HDC g_hdc;
HGLRC g_hRC;

DWORD g_dwStyle;
WINDOWPLACEMENT g_WindowPlacementPrev = { sizeof(WINDOWPLACEMENT) };

bool g_boFullScreen = false;
bool g_boActiveWindow = false;
bool g_boEscapeKeyPressed = false;

#define	CLASS_NAME		TEXT("PP : 24 sphere's with Material -  Per fragment Fong")

#define LOG_FILE_NAME	("log.txt")
//	Handle to log file
FILE *g_fpLogFile = NULL;

enum
{
	RTR_ATTRIBUTE_POSITION = 0,
	RTR_ATTRIBUTE_COLOR,
	RTR_ATTRIBUTE_NORMAL,
	RTR_ATTRIBUTE_TEXTURE0
};

float g_farrSphereVertices[1146];
float g_farrSphereNormals[1146];
float g_farrSphereTextures[764];
unsigned short g_uiarrSphereElements[2280];
GLuint g_gluiNumVertices;
GLuint g_gluiNumElements;

GLuint g_gluiShaderObjectVertex;
GLuint g_gluiShaderObjectFragment;
GLuint g_gluiShaderObjectProgram;

GLuint g_gluiVAOSphere;
GLuint g_gluiVBOPosition;
GLuint g_gluiVBONormal;
GLuint g_gluiVBOElement;

/////////////////////////////////////////////////////////////////
//+Uniforms.
GLuint g_gluiModelMat4Uniform;
GLuint g_gluiViewMat4Uniform;
GLuint g_gluiProjectionMat4Uniform;
GLuint g_gluiRotationMat4Uniform;

GLuint g_gluiLKeyPressedUniform;

GLuint g_gluiLaVec3Uniform;	//	light ambient
GLuint g_gluiLdVec3Uniform;	//	light diffuse
GLuint g_gluiLsVec3Uniform;	//	light specular
GLuint g_gluiLightPositionVec4Uniform;

GLuint g_gluiKaVec3Uniform;//	Material ambient
GLuint g_gluiKdVec3Uniform;//	Material diffuse
GLuint g_gluiKsVec3Uniform;//	Material specular
GLuint g_gluiMaterialShininessUniform;
//-Uniforms.
/////////////////////////////////////////////////////////////////

//
//	Light R == Red Light
//
GLfloat g_glfarrLightAmbient[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfarrLightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfarrLightSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfarrLightPosition[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Give position runtime.

///////////////////////////////////////////////////////////////////////////
//+Material
//
//	Materail 00
//
GLfloat g_arrMaterial00Ambient[] = { 0.0215f, 0.1745f, 0.0215f, 1.0f };
GLfloat g_arrMaterial00Diffuse[] = { 0.07568f, 0.61424f, 0.07568f, 1.0f };
GLfloat g_arrMaterial00Specular[] = { 0.633f, 0.727811f, 0.633f, 1.0f };
GLfloat g_Material00Shininess = 0.6f * 128.0f;

//
//	Materail 10
//
GLfloat g_arrMaterial10Ambient[] = { 0.135f, 0.2225f, 0.1575f, 1.0f };
GLfloat g_arrMaterial10Diffuse[] = { 0.54f, 0.89f, 0.63f, 1.0f };
GLfloat g_arrMaterial10Specular[] = { 0.316228f, 0.316228f, 0.316228f, 1.0f };
GLfloat g_Material10Shininess = 0.1f * 128.0f;

//
//	Materail 20
//
GLfloat g_arrMaterial20Ambient[] = { 0.05375f, 0.05f, 0.06625f, 1.0f };
GLfloat g_arrMaterial20Diffuse[] = { 0.18275f, 0.17f, 0.22525f, 1.0f };
GLfloat g_arrMaterial20Specular[] = { 0.332741f, 0.328634f, 0.346435f, 1.0f };
GLfloat g_Material20Shininess = 0.3f * 128.0f;

//
//	Materail 30
//
GLfloat g_arrMaterial30Ambient[] = { 0.25f, 0.20725f, 0.20725f, 1.0f };
GLfloat g_arrMaterial30Diffuse[] = { 1.0f, 0.829f, 0.829f, 1.0f };
GLfloat g_arrMaterial30Specular[] = { 0.296648f, 0.296648f, 0.296648f, 1.0f };
GLfloat g_Material30Shininess = 0.088f * 128.0f;

//
//	Materail 40
//
GLfloat g_arrMaterial40Ambient[] = { 0.1745f, 0.01175f, 0.01175f, 1.0f };
GLfloat g_arrMaterial40Diffuse[] = { 0.61424f, 0.04136f, 0.04136f, 1.0f };
GLfloat g_arrMaterial40Specular[] = { 0.727811f, 0.626959f, 0.626959f, 1.0f };
GLfloat g_Material40Shininess = 0.6f * 128.0f;

//
//	Materail 50
//
GLfloat g_arrMaterial50Ambient[] = { 0.1f, 0.18725f, 0.1745f, 1.0f };
GLfloat g_arrMaterial50Diffuse[] = { 0.396f, 0.74151f, 0.69102f, 1.0f };
GLfloat g_arrMaterial50Specular[] = { 0.297254f, 0.30829f, 0.306678f, 1.0f };
GLfloat g_Material50Shininess = 0.1f * 128.0f;

//
//	Materail 01
//
GLfloat g_arrMaterial01Ambient[] = { 0.329412f, 0.223529f, 0.027451f, 1.0f };
GLfloat g_arrMaterial01Diffuse[] = { 0.780392f, 0.568627f, 0.113725f, 1.0f };
GLfloat g_arrMaterial01Specular[] = { 0.992157f, 0.941176f, 0.807843f, 1.0f };
GLfloat g_Material01Shininess = 0.21794872f * 128.0f;

//
//	Materail 11
//
GLfloat g_arrMaterial11Ambient[] = { 0.2125f, 0.1275f, 0.054f, 1.0f };
GLfloat g_arrMaterial11Diffuse[] = { 0.714f, 0.4284f, 0.18144f, 1.0f };
GLfloat g_arrMaterial11Specular[] = { 0.393548f, 0.271906f, 0.166721f, 1.0f };
GLfloat g_Material11Shininess = 0.2f * 128.0f;

//
//	Materail 21
//
GLfloat g_arrMaterial21Ambient[] = { 0.25f, 0.25f, 0.25f, 1.0f };
GLfloat g_arrMaterial21Diffuse[] = { 0.4f, 0.4f, 0.4f, 1.0f };
GLfloat g_arrMaterial21Specular[] = { 0.774597f, 0.774597f, 0.774597f, 1.0f };
GLfloat g_Material21Shininess = 0.6f * 128.0f;

//
//	Materail 31
//
GLfloat g_arrMaterial31Ambient[] = { 0.19125f, 0.0735f, 0.0225f, 1.0f };
GLfloat g_arrMaterial31Diffuse[] = { 0.7038f, 0.27048f, 0.0828f, 1.0f };
GLfloat g_arrMaterial31Specular[] = { 0.256777f, 0.137622f, 0.296648f, 1.0f };
GLfloat g_Material31Shininess = 0.1f * 128.0f;

//
//	Materail 41
//
GLfloat g_arrMaterial41Ambient[] = { 0.24725f, 0.1995f, 0.0745f, 1.0f };
GLfloat g_arrMaterial41Diffuse[] = { 0.75164f, 0.60648f, 0.22648f, 1.0f };
GLfloat g_arrMaterial41Specular[] = { 0.628281f, 0.555802f, 0.366065f, 1.0f };
GLfloat g_Material41Shininess = 0.4f * 128.0f;

//
//	Materail 51
//
GLfloat g_arrMaterial51Ambient[] = { 0.19225f, 0.19225f, 0.19225f, 1.0f };
GLfloat g_arrMaterial51Diffuse[] = { 0.50754f, 0.50754f, 0.50754f, 1.0f };
GLfloat g_arrMaterial51Specular[] = { 0.508273f, 0.508273f, 0.508273f, 1.0f };
GLfloat g_Material51Shininess = 0.4f * 128.0f;

//
//	Materail 02
//
GLfloat g_arrMaterial02Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial02Diffuse[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial02Specular[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_Material02Shininess = 0.25f * 128.0f;

//
//	Materail 12
//
GLfloat g_arrMaterial12Ambient[] = { 0.0f, 0.1f, 0.06f, 1.0f };
GLfloat g_arrMaterial12Diffuse[] = { 0.0f, 0.50980392f, 0.50980392f, 1.0f };
GLfloat g_arrMaterial12Specular[] = { 0.50980392f, 0.50980392f, 0.50980392f, 1.0f };
GLfloat g_Material12Shininess = 0.25f * 128.0f;

//
//	Materail 22
//
GLfloat g_arrMaterial22Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial22Diffuse[] = { 0.1f, 0.35f, 0.1f, 1.0f };
GLfloat g_arrMaterial22Specular[] = { 0.45f, 0.45f, 0.45f, 1.0f };
GLfloat g_Material22Shininess = 0.25f * 128.0f;

//
//	Materail 32
//
GLfloat g_arrMaterial32Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial32Diffuse[] = { 0.5f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial32Specular[] = { 0.7f, 0.6f, 0.6f, 1.0f };
GLfloat g_Material32Shininess = 0.25f * 128.0f;

//
//	Materail 42
//
GLfloat g_arrMaterial42Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial42Diffuse[] = { 0.55f, 0.55f, 0.55f, 1.0f };
GLfloat g_arrMaterial42Specular[] = { 0.70f, 0.70f, 0.70f, 1.0f };
GLfloat g_Material42Shininess = 0.25f * 128.0f;

//
//	Materail 52
//
GLfloat g_arrMaterial52Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial52Diffuse[] = { 0.5f, 0.5f, 0.0f, 1.0f };
GLfloat g_arrMaterial52Specular[] = { 0.60f, 0.60f, 0.50f, 1.0f };
GLfloat g_Material52Shininess = 0.25f * 128.0f;

//
//	Materail 03
//
GLfloat g_arrMaterial03Ambient[] = { 0.02f, 0.02f, 0.02f, 1.0f };
GLfloat g_arrMaterial03Diffuse[] = { 0.01f, 0.01f, 0.01f, 1.0f };
GLfloat g_arrMaterial03Specular[] = { 0.4f, 0.4f, 0.4f, 1.0f };
GLfloat g_Material03Shininess = 0.078125f * 128.0f;

//
//	Materail 13
//
GLfloat g_arrMaterial13Ambient[] = { 0.0f, 0.05f, 0.05f, 1.0f };
GLfloat g_arrMaterial13Diffuse[] = { 0.4f, 0.5f, 0.5f, 1.0f };
GLfloat g_arrMaterial13Specular[] = { 0.04f, 0.7f, 0.7f, 1.0f };
GLfloat g_Material13Shininess = 0.078125f * 128.0f;

//
//	Materail 23
//
GLfloat g_arrMaterial23Ambient[] = { 0.0f, 0.05f, 0.0f, 1.0f };
GLfloat g_arrMaterial23Diffuse[] = { 0.4f, 0.5f, 0.4f, 1.0f };
GLfloat g_arrMaterial23Specular[] = { 0.04f, 0.7f, 0.04f, 1.0f };
GLfloat g_Material23Shininess = 0.078125f * 128.0f;

//
//	Materail 33
//
GLfloat g_arrMaterial33Ambient[] = { 0.05f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial33Diffuse[] = { 0.5f, 0.4f, 0.4f, 1.0f };
GLfloat g_arrMaterial33Specular[] = { 0.7f, 0.04f, 0.04f, 1.0f };
GLfloat g_Material33Shininess = 0.078125f * 128.0f;

//
//	Materail 43
//
GLfloat g_arrMaterial43Ambient[] = { 0.05f, 0.05f, 0.05f, 1.0f };
GLfloat g_arrMaterial43Diffuse[] = { 0.5f, 0.5f, 0.5f, 1.0f };
GLfloat g_arrMaterial43Specular[] = { 0.7f, 0.7f, 0.7f, 1.0f };
GLfloat g_Material43Shininess = 0.78125f * 128.0f;

//
//	Materail 53
//
GLfloat g_arrMaterial53Ambient[] = { 0.05f, 0.05f, 0.0f, 1.0f };
GLfloat g_arrMaterial53Diffuse[] = { 0.5f, 0.5f, 0.4f, 1.0f };
GLfloat g_arrMaterial53Specular[] = { 0.7f, 0.7f, 0.04f, 1.0f };
GLfloat g_Material53Shininess = 0.078125f * 128.0f;

//-Material
///////////////////////////////////////////////////////////////////////////
mat4 g_matPerspectiveProjection;

bool g_bAnimate = false;
bool g_bLight = false;
int g_iLightType = 1;	//	1 for vertex light else fragment light.

GLfloat g_fAngleLight = 0.0f;

char chAnimationAxis = 'x';
GLfloat g_fRotateX = 1.0f;
GLfloat g_fRotateY = 0.0f;
GLfloat g_fRotateZ = 0.0f;


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	VOID initialize();
	VOID display();
	VOID uninitialize();
	void update();

	MSG Msg;
	int x, y;
	HWND hWnd;
	int iMaxWidth;
	int iMaxHeight;
	WNDCLASSEX WndClass;
	bool boDone = false;
	TCHAR szClassName[] = CLASS_NAME;

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
	WndClass.lpszClassName = szClassName;
	WndClass.lpszMenuName = NULL;

	//
	//	Register class.
	//
	RegisterClassEx(&WndClass);

	iMaxWidth = GetSystemMetrics(SM_CXFULLSCREEN);
	iMaxHeight = GetSystemMetrics(SM_CYFULLSCREEN);

	x = (iMaxWidth - WIN_WIDTH) / 2;
	y = (iMaxHeight - WIN_HEIGHT) / 2;

	//
	//	Create Window.
	//
	hWnd = CreateWindowEx(
		WS_EX_APPWINDOW,	//	Change: New member get added for CreateWindowEx API.
		szClassName,
		CLASS_NAME,
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,		//	Change: Added styles -WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE
		x,
		y,
		WIN_WIDTH,
		WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL
		);
	if (NULL == hWnd)
	{
		return 0;
	}

	g_hWnd = hWnd;

	initialize();

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

				if (true == g_bAnimate)
				{
					update();
				}
			}
		}
	}

	uninitialize();

	return((int)Msg.wParam);
}


LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	VOID ToggleFullScreen();
	void InitLight();

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
		resize(LOWORD(lParam), HIWORD(lParam));
		break;


	case WM_CHAR:
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

		case VK_ESCAPE:
			g_boEscapeKeyPressed = true;
			break;

		case 'a':
		case 'A':
			if (false == g_bAnimate)
			{
				g_bAnimate = true;
			}
			else
			{
				g_bAnimate = false;
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

VOID initialize()
{
	void uninitialize();
	void resize(int, int);

	HDC hDC;
	int iPixelFormatIndex;
	PIXELFORMATDESCRIPTOR pfd;

	fopen_s(&g_fpLogFile, LOG_FILE_NAME, "w");
	if (NULL == g_fpLogFile)
	{
		uninitialize();
		return;
	}

	ZeroMemory(&pfd, sizeof(pfd));

	//
	//	Init Pixel format descriptor structure.
	//
	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32;	//	Change 1: for 3d

	g_hdc = GetDC(g_hWnd);

	hDC = GetDC(g_hWnd);

	ReleaseDC(g_hWnd, hDC);

	iPixelFormatIndex = ChoosePixelFormat(g_hdc, &pfd);
	if (0 == iPixelFormatIndex)
	{
		ReleaseDC(g_hWnd, g_hdc);
		g_hdc = NULL;
	}

	if (FALSE == SetPixelFormat(g_hdc, iPixelFormatIndex, &pfd))
	{
		ReleaseDC(g_hWnd, g_hdc);
		g_hdc = NULL;
	}

	g_hRC = wglCreateContext(g_hdc);
	if (NULL == g_hRC)
	{
		ReleaseDC(g_hWnd, g_hdc);
		g_hdc = NULL;
	}

	if (FALSE == wglMakeCurrent(g_hdc, g_hRC))
	{
		wglDeleteContext(g_hRC);
		g_hRC = NULL;
		ReleaseDC(g_hWnd, g_hdc);
		g_hdc = NULL;
	}

	GLenum glewError = glewInit();
	if (GLEW_OK != glewError)
	{
		uninitialize();
		fprintf(g_fpLogFile, "glewInit() failed, Error :%d", glewError);
		return;
	}

	fprintf(g_fpLogFile, "\n Version : %s", glGetString(GL_VERSION));
	fprintf(g_fpLogFile, "\n Shader Version : %s", glGetString(GL_SHADING_LANGUAGE_VERSION));

	////////////////////////////////////////////////////////////////////
	//+	Shader code

	////////////////////////////////////////////////////////////////////
	//+	Vertex shader.

	fprintf(g_fpLogFile, "==>Vertex Shader.");

	//	Create shader.
	g_gluiShaderObjectVertex = glCreateShader(GL_VERTEX_SHADER);

	//	Provide source code.
	const GLchar *szVertexShaderSourceCode =
		"#version 430 core"							\
		"\n"										\
		"in vec4 vPosition;"						\
		"in vec3 vNormal;"							\
		"uniform mat4 u_model_matrix;"	\
		"uniform mat4 u_view_matrix;"	\
		"uniform mat4 u_projection_matrix;"	\
		"uniform mat4 u_rotation_matrix;"	\
		"uniform vec4 u_light_position;"		\
		"uniform int u_L_key_pressed;"			\
		"out vec3 transformed_normals;"			\
		"out vec3 light_direction;"			\
		"out vec3 viewer_vector;"			\
		"void main(void)"							\
		"{"											\
		"if (1 == u_L_key_pressed)"										\
		"{"											\
		"vec4 eyeCoordinates = u_view_matrix * u_model_matrix * vPosition;"											\
		"transformed_normals = normalize(mat3(u_view_matrix * u_model_matrix) * vNormal);"											\
		"vec4 rotated_light_position = u_rotation_matrix * u_light_position;"											\
		"light_direction = normalize(vec3(rotated_light_position) - eyeCoordinates.xyz);"											\
		"viewer_vector = normalize(-eyeCoordinates.xyz);"											\

		"transformed_normals = normalize(transformed_normals);"											\
		"light_direction = normalize(light_direction);"											\
		"viewer_vector = normalize(viewer_vector);"											\
		"}"											\
		"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;"	\
		"}";

	glShaderSource(g_gluiShaderObjectVertex, 1, &szVertexShaderSourceCode, NULL);

	//	Compile shader.
	glCompileShader(g_gluiShaderObjectVertex);

	GLint gliCompileStatus;
	GLint gliInfoLogLength;
	char *pszInfoLog = NULL;
	GLsizei glsiWritten;
	glGetShaderiv(g_gluiShaderObjectVertex, GL_COMPILE_STATUS, &gliCompileStatus);
	if (GL_FALSE == gliCompileStatus)
	{
		glGetShaderiv(g_gluiShaderObjectVertex, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength <= 0)
		{
			fprintf(g_fpLogFile, "GL_INFO_LOG_LENGTH is less than 0.");
			uninitialize();
			exit(0);
		}

		pszInfoLog = (char*)malloc(gliInfoLogLength);
		if (NULL == pszInfoLog)
		{
			fprintf(g_fpLogFile, "malloc failed.");
			uninitialize();
			exit(0);
		}

		glGetShaderInfoLog(g_gluiShaderObjectVertex, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//-	Vertex shader.
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+	Fragment shader.

	fprintf(g_fpLogFile, "==>Fragment Shader.");

	//	Create shader.
	g_gluiShaderObjectFragment = glCreateShader(GL_FRAGMENT_SHADER);

	//	Provide source code.
	const GLchar *szFragmentShaderSourceCode =
		"#version 430 core"							\
		"\n"										\
		"in vec3 transformed_normals;"			\
		"in vec3 light_direction;"			\
		"in vec3 viewer_vector;"			\
		"uniform vec3 u_La;	"				\
		"uniform vec3 u_Ld;	"				\
		"uniform vec3 u_Ls;	"				\
		"uniform vec3 u_Ka;"					\
		"uniform vec3 u_Kd;"					\
		"uniform vec3 u_Ks;"					\
		"uniform float u_material_shininess;"		\
		"uniform int u_L_key_pressed;"			\
		"out vec4 vFragColor;"						\
		"void main(void)"							\
		"{"											\
		"vec3 phong_ads_color;"					\
		"if (1 == u_L_key_pressed)"										\
		"{"											\
		"vec3 normalized_transformed_normals = normalize(transformed_normals);"											\
		"vec3 normalized_light_direction = normalize(light_direction);"											\
		"vec3 normalized_viewer_vector = normalize(viewer_vector);"											\
		"float tn_dot_ld = max(dot(normalized_transformed_normals, normalized_light_direction), 0.0);"											\
		"vec3 ambient = u_La * u_Ka;"											\
		"vec3 diffuse = u_Ld * u_Kd * tn_dot_ld;"											\
		"vec3 reflection_vector = reflect(-normalized_light_direction, normalized_transformed_normals);"											\
		"vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector, normalized_viewer_vector), 0.0), u_material_shininess);"											\
		"phong_ads_color = ambient + diffuse + specular;"											\
		"}"											\
		"else"											\
		"{"											\
		"	phong_ads_color = vec3(1.0,1.0,1.0);"											\
		"}"											\
		"vFragColor = vec4(phong_ads_color, 1.0);"					\
		"}";

	glShaderSource(g_gluiShaderObjectFragment, 1, &szFragmentShaderSourceCode, NULL);

	//	Compile shader.
	glCompileShader(g_gluiShaderObjectFragment);

	gliCompileStatus = 0;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;
	glsiWritten = 0;
	glGetShaderiv(g_gluiShaderObjectFragment, GL_COMPILE_STATUS, &gliCompileStatus);
	if (GL_FALSE == gliCompileStatus)
	{
		glGetShaderiv(g_gluiShaderObjectFragment, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength <= 0)
		{
			fprintf(g_fpLogFile, "Fragment : GL_INFO_LOG_LENGTH is less than 0.");
			uninitialize();
			exit(0);
		}

		pszInfoLog = (char*)malloc(gliInfoLogLength);
		if (NULL == pszInfoLog)
		{
			fprintf(g_fpLogFile, "Fragment : malloc failed.");
			uninitialize();
			exit(0);
		}

		glGetShaderInfoLog(g_gluiShaderObjectFragment, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//-	Fragment shader.
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+	Shader program.

	//	Create.
	g_gluiShaderObjectProgram = glCreateProgram();

	//	Attach vertex shader to shader program.
	glAttachShader(g_gluiShaderObjectProgram, g_gluiShaderObjectVertex);

	//	Attach Fragment shader to shader program.
	glAttachShader(g_gluiShaderObjectProgram, g_gluiShaderObjectFragment);

	//	IMP:Pre-link binding of shader program object with vertex shader position attribute.
	glBindAttribLocation(g_gluiShaderObjectProgram, RTR_ATTRIBUTE_POSITION, "vPosition");

	glBindAttribLocation(g_gluiShaderObjectProgram, RTR_ATTRIBUTE_NORMAL, "vNormal");

	//	Link shader.
	glLinkProgram(g_gluiShaderObjectProgram);

	GLint gliLinkStatus;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;
	glsiWritten = 0;
	glGetShaderiv(g_gluiShaderObjectProgram, GL_LINK_STATUS, &gliLinkStatus);
	if (GL_FALSE == gliLinkStatus)
	{
		glGetProgramiv(g_gluiShaderObjectProgram, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength <= 0)
		{
			fprintf(g_fpLogFile, "Link : GL_INFO_LOG_LENGTH is less than 0.");
			uninitialize();
			exit(0);
		}

		pszInfoLog = (char*)malloc(gliInfoLogLength);
		if (NULL == pszInfoLog)
		{
			fprintf(g_fpLogFile, "Link : malloc failed.");
			uninitialize();
			exit(0);
		}

		glGetProgramInfoLog(g_gluiShaderObjectProgram, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//-	Shader program.
	////////////////////////////////////////////////////////////////////

	//-	Shader code
	////////////////////////////////////////////////////////////////////

	//
	//	The actual locations assigned to uniform variables are not known until the program object is linked successfully.
	//	After a program object has been linked successfully, the index values for uniform variables remain fixed until the next link command occurs.
	//
	g_gluiModelMat4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_model_matrix");
	if (-1 == g_gluiModelMat4Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_model_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiViewMat4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_view_matrix");
	if (-1 == g_gluiViewMat4Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_view_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiProjectionMat4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_projection_matrix");
	if (-1 == g_gluiProjectionMat4Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_projection_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiRotationMat4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_rotation_matrix");
	if (-1 == g_gluiRotationMat4Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_rotation_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLKeyPressedUniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_L_key_pressed");
	if (-1 == g_gluiLKeyPressedUniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_L_key_pressed) failed.");
		uninitialize();
		exit(0);
	}

	//	Red Light
	g_gluiLaVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_La");
	if (-1 == g_gluiLaVec3Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_La) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLdVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Ld");
	if (-1 == g_gluiLdVec3Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Ld) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLsVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Ls");
	if (-1 == g_gluiLsVec3Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Ls) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLightPositionVec4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_light_position");
	if (-1 == g_gluiLightPositionVec4Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_light_position) failed.");
		uninitialize();
		exit(0);
	}

	//	Light Material
	g_gluiKaVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Ka");
	if (-1 == g_gluiKaVec3Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Ka) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiKdVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Kd");
	if (-1 == g_gluiKdVec3Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Kd) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiKsVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Ks");
	if (-1 == g_gluiKsVec3Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Ks) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiMaterialShininessUniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_material_shininess");
	if (-1 == g_gluiMaterialShininessUniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_material_shininess) failed.");
		uninitialize();
		exit(0);
	}

	////////////////////////////////////////////////////////////////////
	//+	Vertices,color, shader attribute, vbo,vao initialization.

	getSphereVertexData(g_farrSphereVertices, g_farrSphereNormals, g_farrSphereTextures, g_uiarrSphereElements);
	g_gluiNumVertices = getNumberOfSphereVertices();
	g_gluiNumElements = getNumberOfSphereElements();

	glGenVertexArrays(1, &g_gluiVAOSphere);	//	It is like recorder.
	glBindVertexArray(g_gluiVAOSphere);

	////////////////////////////////////////////////////////////////////
	//+ Vertex position
	glGenBuffers(1, &g_gluiVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, g_gluiVBOPosition);

	glBufferData(GL_ARRAY_BUFFER, sizeof(g_farrSphereVertices), g_farrSphereVertices, GL_STATIC_DRAW);

	glVertexAttribPointer(RTR_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(RTR_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//- Vertex position
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+ Vertex Normal
	glGenBuffers(1, &g_gluiVBONormal);
	glBindBuffer(GL_ARRAY_BUFFER, g_gluiVBONormal);

	glBufferData(GL_ARRAY_BUFFER, sizeof(g_farrSphereNormals), g_farrSphereNormals, GL_STATIC_DRAW);

	glVertexAttribPointer(RTR_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(RTR_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//- Vertex Normal
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+ Vertex Element
	glGenBuffers(1, &g_gluiVBOElement);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(g_uiarrSphereElements), g_uiarrSphereElements, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	//- Vertex Element
	////////////////////////////////////////////////////////////////////

	glBindVertexArray(0);

	//-	Vertices,color, shader attribute, vbo,vao initialization.
	////////////////////////////////////////////////////////////////////

	//glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClearColor(0.25f, 0.25f, 0.25f, 0.0f);	//	Grey color.

	//+	Change 2 For 3D
	glClearDepth(1.0f);

	glEnable(GL_DEPTH_TEST);

	glDepthFunc(GL_LEQUAL);

	//
	//	Optional.
	//
	glShadeModel(GL_SMOOTH);

	//
	//	We will always cull back faces for better performance.
	//	We will this in case of 3-D rotation/graphics.
	//
	glEnable(GL_CULL_FACE);

	//-	Change 2 For 3D

	//	See orthographic projection matrix to identity.
	g_matPerspectiveProjection = mat4::identity();

	//
	//	Resize.
	//
	resize(WIN_WIDTH, WIN_HEIGHT);
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


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);	//	Change 3 for 3D

	//	Start using opengl program.
	glUseProgram(g_gluiShaderObjectProgram);

	if (true == g_bLight)
	{
		glUniform1i(g_gluiLKeyPressedUniform, 1);

		glUniform3fv(g_gluiLaVec3Uniform, 1, g_glfarrLightAmbient);	//	Ambient
		glUniform3fv(g_gluiLdVec3Uniform, 1, g_glfarrLightDiffuse);	//	Diffuse
		glUniform3fv(g_gluiLsVec3Uniform, 1, g_glfarrLightSpecular);	//	Specular
	}
	else
	{
		glUniform1i(g_gluiLKeyPressedUniform, 0);
	}

	float fHeightMulti = 0.05f;
	float fWidthMulti = 0.07f;

	//
	//	First column,
	//
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere50();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere40();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere30();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere20();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere10();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere00();

	//
	//	Second column.
	//
	fWidthMulti = fWidthMulti + 0.20f;
	fHeightMulti = 0.05f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere51();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere41();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere31();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere21();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere11();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere01();

	//
	//	Third column.
	//
	fWidthMulti = fWidthMulti + 0.20f;
	fHeightMulti = 0.05f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere52();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere42();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere32();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere22();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere12();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere02();

	//
	//	Fourth column.
	//
	fWidthMulti = fWidthMulti + 0.20f;
	fHeightMulti = 0.05f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere53();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere43();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere33();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere23();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere13();

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	Sphere03();

	//	Stop using opengl program.
	glUseProgram(0);

	SwapBuffers(g_hdc);
}

/////X0

void Sphere00()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial00Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial00Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial00Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material00Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere10()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial10Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial10Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial10Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material10Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere20()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial20Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial20Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial20Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material20Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere30()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial30Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial30Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial30Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material30Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere40()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial40Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial40Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial40Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material40Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere50()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial50Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial50Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial50Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material50Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

/////X1

void Sphere01()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial01Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial01Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial01Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material01Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere11()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial11Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial11Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial11Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material11Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere21()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial21Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial21Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial21Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material21Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere31()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial31Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial31Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial31Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material31Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere41()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial41Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial41Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial41Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material41Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere51()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial51Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial51Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial51Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material51Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

/////X1

void Sphere02()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial02Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial02Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial02Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material02Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere12()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial12Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial12Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial12Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material12Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere22()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial22Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial22Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial22Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material22Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere32()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial32Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial32Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial32Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material32Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere42()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial42Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial42Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial42Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material42Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere52()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial52Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial52Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial52Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material52Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

/////X3

void Sphere03()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial03Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial03Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial03Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material03Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere13()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial13Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial13Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial13Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material13Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere23()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial23Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial23Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial23Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material23Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere33()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial33Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial33Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial33Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material33Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere43()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial43Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial43Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial43Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material43Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}

void Sphere53()
{
	mat4 matModel;
	mat4 matView;
	mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotation = mat4::identity();

	matModel = translate(0.0f, 0.0f, -2.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial53Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial53Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial53Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material53Shininess);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);
}


void update()
{
	g_fAngleLight = g_fAngleLight + 0.1f;
	if (g_fAngleLight >= 720)
		//if (g_fAngleLight >= 360)
	{
		//	Fixme: Search proper way to avoid hitch in light animation.
		g_fAngleLight = 360.0f;
	}
}


VOID resize(int iWidth, int iHeight)
{
	if (0 == iHeight)
	{
		iHeight = 1;
	}

	//	perspective(float fovy, float aspect, float n, float f)
	g_matPerspectiveProjection = perspective(45, (GLfloat)(iWidth/4) / (GLfloat)(iHeight/6), 0.1f, 100.0f);

	glViewport(0, 0, iWidth, iHeight);
}


VOID uninitialize()
{
	if (true == g_boFullScreen)
	{
		SetWindowLong(g_hWnd, GWL_STYLE, g_dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(g_hWnd, &g_WindowPlacementPrev);
		SetWindowPos(g_hWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);

		ShowCursor(TRUE);
	}

	if (g_gluiVBONormal)
	{
		glDeleteBuffers(1, &g_gluiVBONormal);
		g_gluiVBONormal = 0;
	}

	if (g_gluiVBOPosition)
	{
		glDeleteBuffers(1, &g_gluiVBOPosition);
		g_gluiVBOPosition = 0;
	}

	if (g_gluiVBOElement)
	{
		glDeleteBuffers(1, &g_gluiVBOElement);
		g_gluiVBOElement = 0;
	}

	if (g_gluiVAOSphere)
	{
		glDeleteVertexArrays(1, &g_gluiVAOSphere);
		g_gluiVAOSphere = 0;
	}

	if (g_gluiShaderObjectVertex)
	{
		glDetachShader(g_gluiShaderObjectProgram, g_gluiShaderObjectVertex);
		glDeleteShader(g_gluiShaderObjectVertex);
		g_gluiShaderObjectVertex = 0;
	}

	if (g_gluiShaderObjectFragment)
	{
		glDetachShader(g_gluiShaderObjectProgram, g_gluiShaderObjectFragment);
		glDeleteShader(g_gluiShaderObjectFragment);
		g_gluiShaderObjectFragment = 0;
	}

	//
	//	Unlink shader program
	//	This will be useful when detach multiple shaders in loop.
	//	1.glUseProgram(Shader_Program_Object)
	//	2.Get Attach shader list
	//	3.Detach i loop.
	//	4.glUseProgram(0)
	//
	glUseProgram(0);

	if (g_gluiShaderObjectProgram)
	{
		glDeleteProgram(g_gluiShaderObjectProgram);
		g_gluiShaderObjectProgram = 0;
	}

	wglMakeCurrent(NULL, NULL);
	wglDeleteContext(g_hRC);
	g_hRC = NULL;

	ReleaseDC(g_hWnd, g_hdc);
	g_hdc = NULL;

	DestroyWindow(g_hWnd);
	g_hWnd = NULL;

	if (g_fpLogFile)
	{
		fprintf(g_fpLogFile, "\n Log file succesfuly closed.");
		fclose(g_fpLogFile);
		g_fpLogFile = NULL;
	}
}
#endif
