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

#define	CLASS_NAME		TEXT("PP : 3 lights on steady sphere -  Per vertex and fragment")

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

GLuint g_gluiShaderObjectVertexPerVertexLight;
GLuint g_gluiShaderObjectFragmentPerVertexLight;
GLuint g_gluiShaderObjectProgramPerVertexLight;

GLuint g_gluiShaderObjectVertexPerFragmentLight;
GLuint g_gluiShaderObjectFragmentPerFragmentLight;
GLuint g_gluiShaderObjectProgramPerFragmentLight;

GLuint g_gluiVAOSphere;
GLuint g_gluiVBOPosition;
GLuint g_gluiVBONormal;
GLuint g_gluiVBOElement;

/////////////////////////////////////////////////////////////////
//+Uniforms.

//	0 th uniform for Per Vertex Light
//	1 th uniform for Per Fragment Light
#define UNIFORM_INDEX_PER_VERTEX	0
#define UNIFORM_INDEX_PER_FRAGMENT	1
#define NUM_LIGHT_TYPE				2

GLuint g_gluiModelMat4Uniform[NUM_LIGHT_TYPE];
GLuint g_gluiViewMat4Uniform[NUM_LIGHT_TYPE];
GLuint g_gluiProjectionMat4Uniform[NUM_LIGHT_TYPE];
GLuint g_gluiRotationRMat4Uniform[NUM_LIGHT_TYPE];
GLuint g_gluiRotationGMat4Uniform[NUM_LIGHT_TYPE];
GLuint g_gluiRotationBMat4Uniform[NUM_LIGHT_TYPE];

GLuint g_gluiLKeyPressedUniform[NUM_LIGHT_TYPE];
GLuint g_gluiSKeyPressedUniform[NUM_LIGHT_TYPE];

GLuint g_gluiLaRVec3Uniform[NUM_LIGHT_TYPE];	//	light ambient
GLuint g_gluiLdRVec3Uniform[NUM_LIGHT_TYPE];	//	light diffuse
GLuint g_gluiLsRVec3Uniform[NUM_LIGHT_TYPE];	//	light specular
GLuint g_gluiLightPositionRVec4Uniform[NUM_LIGHT_TYPE];

GLuint g_gluiLaGVec3Uniform[NUM_LIGHT_TYPE];	//	light ambient
GLuint g_gluiLdGVec3Uniform[NUM_LIGHT_TYPE];	//	light diffuse
GLuint g_gluiLsGVec3Uniform[NUM_LIGHT_TYPE];	//	light specular
GLuint g_gluiLightPositionGVec4Uniform[NUM_LIGHT_TYPE];

GLuint g_gluiLaBVec3Uniform[NUM_LIGHT_TYPE];	//	light ambient
GLuint g_gluiLdBVec3Uniform[NUM_LIGHT_TYPE];	//	light diffuse
GLuint g_gluiLsBVec3Uniform[NUM_LIGHT_TYPE];	//	light specular
GLuint g_gluiLightPositionBVec4Uniform[NUM_LIGHT_TYPE];

GLuint g_gluiKaVec3Uniform[NUM_LIGHT_TYPE];//	Material ambient
GLuint g_gluiKdVec3Uniform[NUM_LIGHT_TYPE];//	Material diffuse
GLuint g_gluiKsVec3Uniform[NUM_LIGHT_TYPE];//	Material specular
GLuint g_gluiMaterialShininessUniform[NUM_LIGHT_TYPE];
//-Uniforms.
/////////////////////////////////////////////////////////////////

//
//	Light R == Red Light
//
GLfloat g_glfarrLightRAmbient[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Decides general light
GLfloat g_glfarrLightRDiffuse[] = { 1.0f, 0.0f, 0.0f, 0.0f };	//	Decides color of light
GLfloat g_glfarrLightRSpecular[] = { 1.0f, 0.0f, 0.0f, 0.0f };	//	Decides height of light
GLfloat g_glfarrLightRPosition[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Runtime give position 

//
//	Light G == Green Light
//
GLfloat g_glfarrLightGAmbient[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Decides general light
GLfloat g_glfarrLightGDiffuse[] = { 0.0f, 1.0f, 0.0f, 0.0f };	//	Decides color of light
GLfloat g_glfarrLightGSpecular[] = { 0.0f, 1.0f, 0.0f, 0.0f };	//	Decides height of light
GLfloat g_glfarrLightGPosition[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Runtime give position 

//
//	Light B == Blue Light
//
GLfloat g_glfarrLightBAmbient[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Decides general light
GLfloat g_glfarrLightBDiffuse[] = { 0.0f, 0.0f, 1.0f, 0.0f };	//	Decides color of light
GLfloat g_glfarrLightBSpecular[] = { 0.0f, 0.0f, 1.0f, 0.0f };	//	Decides height of light
GLfloat g_glfarrLightBPosition[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Runtime give position 


GLfloat g_glfarrMaterialAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_glfarrMaterialDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfarrMaterialSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfMaterialShininess = 50.0f;

mat4 g_matPerspectiveProjection;

bool g_bAnimate = true;
bool g_bLight = false;
int g_iLightType = 2;	//	1 for vertex light else fragment light.

GLfloat g_fAngleRed = 1.0;
GLfloat g_fAngleGreen = 1.0;
GLfloat g_fAngleBlue = 1.0;

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
		resize(LOWORD(lParam), HIWORD(lParam));
		break;


	case WM_CHAR:
		switch (wParam)
		{
		case 'Q':
		case 'q':
			g_boEscapeKeyPressed = true;
			break;

		case 'f':
		case 'F':
			g_iLightType = 2;
			break;

		case 'v':
		case 'V':
			g_iLightType = 1;
			break;

		case VK_ESCAPE:
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
	//+	Vertex shader - Per vertex light

	fprintf(g_fpLogFile, "==>Vertex Shader: Per vertex.");

	//	Create shader.
	g_gluiShaderObjectVertexPerVertexLight = glCreateShader(GL_VERTEX_SHADER);

	//	Provide source code.
	const GLchar *szVertexShaderSourceCodePerVertexLight =
		"#version 430 core"							\
		"\n"										\
		"in vec4 vPosition;"						\
		"in vec3 vNormal;"							\
		"uniform mat4 u_model_matrix;"	\
		"uniform mat4 u_view_matrix;"	\
		"uniform mat4 u_projection_matrix;"	\
		"uniform mat4 u_rotation_matrixR;"	\
		"uniform mat4 u_rotation_matrixG;"	\
		"uniform mat4 u_rotation_matrixB;"	\
		"uniform int u_L_key_pressed;"			\
		"uniform vec3 u_LaR;	"				\
		"uniform vec3 u_LdR;	"				\
		"uniform vec3 u_LsR;	"				\
		"uniform vec4 u_light_positionR;"		\
		"uniform vec3 u_LaG;	"				\
		"uniform vec3 u_LdG;	"				\
		"uniform vec3 u_LsG;	"				\
		"uniform vec4 u_light_positionG;"		\
		"uniform vec3 u_LaB;	"				\
		"uniform vec3 u_LdB;	"				\
		"uniform vec3 u_LsB;	"				\
		"uniform vec4 u_light_positionB;"		\
		"uniform vec3 u_Ka;"					\
		"uniform vec3 u_Kd;"					\
		"uniform vec3 u_Ks;"					\
		"uniform float u_material_shininess;"		\
		"out vec3 out_phong_ads_color;"			\
		"void main(void)"							\
		"{"											\
			"if (1 == u_L_key_pressed)"										\
			"{"											\
				"vec4 eyeCoordinates = u_view_matrix * u_model_matrix * vPosition;"											\
				"vec3 transformed_normals = normalize(mat3(u_view_matrix * u_model_matrix) * vNormal);"											\
				"vec4 rotated_light_positionR = u_rotation_matrixR * u_light_positionR;"											\
				"vec4 rotated_light_positionG = u_rotation_matrixG * u_light_positionG;"											\
				"vec4 rotated_light_positionB = u_rotation_matrixB * u_light_positionB;"											\
				"vec3 light_directionR = normalize(vec3(rotated_light_positionR) - eyeCoordinates.xyz);"											\
				"vec3 light_directionG = normalize(vec3(rotated_light_positionG) - eyeCoordinates.xyz);"											\
				"vec3 light_directionB = normalize(vec3(rotated_light_positionB) - eyeCoordinates.xyz);"											\
				"vec3 viewer_vector = normalize(-eyeCoordinates.xyz);"											\
				/*Red Light*/
				"float tn_dot_ldR = max(dot(transformed_normals, light_directionR), 0.0);"											\
				"vec3 ambientR = u_LaR * u_Ka;"											\
				"vec3 diffuseR = u_LdR * u_Kd * tn_dot_ldR;"											\
				"vec3 reflection_vectorR = reflect(-light_directionR, transformed_normals);"											\
				"vec3 specularR = u_LsR * u_Ks * pow(max(dot(reflection_vectorR, viewer_vector), 0.0), u_material_shininess);"											\
				/*Green Light*/
				"float tn_dot_ldG = max(dot(transformed_normals, light_directionG), 0.0);"											\
				"vec3 ambientG = u_LaG * u_Ka;"											\
				"vec3 diffuseG = u_LdG * u_Kd * tn_dot_ldG;"											\
				"vec3 reflection_vectorG = reflect(-light_directionG, transformed_normals);"											\
				"vec3 specularG = u_LsG * u_Ks * pow(max(dot(reflection_vectorG, viewer_vector), 0.0), u_material_shininess);"											\
				/*Blue Light*/
				"float tn_dot_ldB = max(dot(transformed_normals, light_directionB), 0.0);"											\
				"vec3 ambientB = u_LaB * u_Ka;"											\
				"vec3 diffuseB = u_LdB * u_Kd * tn_dot_ldB;"											\
				"vec3 reflection_vectorB = reflect(-light_directionB, transformed_normals);"											\
				"vec3 specularB = u_LsB * u_Ks * pow(max(dot(reflection_vectorB, viewer_vector), 0.0), u_material_shininess);"											\
				"out_phong_ads_color = ambientR + ambientG + ambientB + diffuseR + diffuseG + diffuseB + specularR + specularG + specularB;"											\
			"}"											\
			"else"											\
			"{"											\
				"out_phong_ads_color = vec3(1.0,1.0,1.0);"											\
			"}"											\
			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;"	\
		"}";

	glShaderSource(g_gluiShaderObjectVertexPerVertexLight, 1, &szVertexShaderSourceCodePerVertexLight, NULL);

	//	Compile shader.
	glCompileShader(g_gluiShaderObjectVertexPerVertexLight);

	GLint gliCompileStatus;
	GLint gliInfoLogLength;
	char *pszInfoLog = NULL;
	GLsizei glsiWritten;
	glGetShaderiv(g_gluiShaderObjectVertexPerVertexLight, GL_COMPILE_STATUS, &gliCompileStatus);
	if (GL_FALSE == gliCompileStatus)
	{
		glGetShaderiv(g_gluiShaderObjectVertexPerVertexLight, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
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

		glGetShaderInfoLog(g_gluiShaderObjectVertexPerVertexLight, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//+	Vertex shader - Per vertex light
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+	Vertex shader - Per fragment light.

	fprintf(g_fpLogFile, "==>Vertex Shader: Per Fragment.");

	//	Create shader.
	g_gluiShaderObjectVertexPerFragmentLight = glCreateShader(GL_VERTEX_SHADER);

	//	Provide source code.
	const GLchar *szVertexShaderSourceCodePerFragmentLight =
		"#version 430 core"							\
		"\n"										\
		"in vec4 vPosition;"						\
		"in vec3 vNormal;"							\
		"uniform mat4 u_model_matrix;"	\
		"uniform mat4 u_view_matrix;"	\
		"uniform mat4 u_projection_matrix;"	\
		"uniform mat4 u_rotation_matrixR;"	\
		"uniform mat4 u_rotation_matrixG;"	\
		"uniform mat4 u_rotation_matrixB;"	\
		"uniform int u_L_key_pressed;"			\
		"uniform vec3 u_LaR;	"				\
		"uniform vec3 u_LdR;	"				\
		"uniform vec3 u_LsR;	"				\
		"uniform vec4 u_light_positionR;"		\
		"uniform vec3 u_LaG;	"				\
		"uniform vec3 u_LdG;	"				\
		"uniform vec3 u_LsG;	"				\
		"uniform vec4 u_light_positionG;"		\
		"uniform vec3 u_LaB;	"				\
		"uniform vec3 u_LdB;	"				\
		"uniform vec3 u_LsB;	"				\
		"uniform vec4 u_light_positionB;"		\
		"uniform vec3 u_Ka;"					\
		"uniform vec3 u_Kd;"					\
		"uniform vec3 u_Ks;"					\
		"uniform float u_material_shininess;"		\
		"out vec3 transformed_normals;"			\
		"out vec3 light_directionR;"			\
		"out vec3 light_directionG;"			\
		"out vec3 light_directionB;"			\
		"out vec3 viewer_vector;"			\
		"out vec3 out_phong_ads_color;"			\
		"void main(void)"							\
		"{"											\
			"if (1 == u_L_key_pressed)"										\
			"{"											\
				"vec4 eyeCoordinates = u_view_matrix * u_model_matrix * vPosition;"											\
				"transformed_normals = normalize(mat3(u_view_matrix * u_model_matrix) * vNormal);"											\
				"vec4 rotated_light_positionR = u_rotation_matrixR * u_light_positionR;"											\
				"vec4 rotated_light_positionG = u_rotation_matrixG * u_light_positionG;"											\
				"vec4 rotated_light_positionB = u_rotation_matrixB * u_light_positionB;"											\
				"light_directionR = normalize(vec3(rotated_light_positionR) - eyeCoordinates.xyz);"											\
				"light_directionG = normalize(vec3(rotated_light_positionG) - eyeCoordinates.xyz);"											\
				"light_directionB = normalize(vec3(rotated_light_positionB) - eyeCoordinates.xyz);"											\
				"viewer_vector = normalize(-eyeCoordinates.xyz);"											\
				"transformed_normals = normalize(transformed_normals);"											\
				"light_directionR = normalize(light_directionR);"											\
				"light_directionG = normalize(light_directionG);"											\
				"light_directionB = normalize(light_directionB);"											\
				"viewer_vector = normalize(viewer_vector);"											\
			"}"											\
			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;"	\
		"}";

	glShaderSource(g_gluiShaderObjectVertexPerFragmentLight, 1, &szVertexShaderSourceCodePerFragmentLight, NULL);

	//	Compile shader.
	glCompileShader(g_gluiShaderObjectVertexPerFragmentLight);

	gliCompileStatus;
	gliInfoLogLength;
	pszInfoLog = NULL;
	glGetShaderiv(g_gluiShaderObjectVertexPerFragmentLight, GL_COMPILE_STATUS, &gliCompileStatus);
	if (GL_FALSE == gliCompileStatus)
	{
		glGetShaderiv(g_gluiShaderObjectVertexPerFragmentLight, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
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

		glGetShaderInfoLog(g_gluiShaderObjectVertexPerFragmentLight, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//-	Vertex shader - Per fragment light.
	////////////////////////////////////////////////////////////////////


	////////////////////////////////////////////////////////////////////
	//+	Fragment shader:Per Vertex .

	fprintf(g_fpLogFile, "==>Fragment Shader:Per Vertex.");

	//	Create shader.
	g_gluiShaderObjectFragmentPerVertexLight = glCreateShader(GL_FRAGMENT_SHADER);

	//	Provide source code.
	const GLchar *szFragmentShaderSourceCodePerVertexLight =
		"#version 430 core"							\
		"\n"										\
		"in vec3 out_phong_ads_color;"				\
		"out vec4 vFragColor;"						\
		"void main(void)"							\
		"{"											\
			"vec3 phong_ads_color;"					\
			"vFragColor = vec4(out_phong_ads_color, 1.0);"					\
		"}";

	glShaderSource(g_gluiShaderObjectFragmentPerVertexLight, 1, &szFragmentShaderSourceCodePerVertexLight, NULL);

	//	Compile shader.
	glCompileShader(g_gluiShaderObjectFragmentPerVertexLight);

	gliCompileStatus = 0;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;
	glsiWritten = 0;
	glGetShaderiv(g_gluiShaderObjectFragmentPerVertexLight, GL_COMPILE_STATUS, &gliCompileStatus);
	if (GL_FALSE == gliCompileStatus)
	{
		glGetShaderiv(g_gluiShaderObjectFragmentPerVertexLight, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
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

		glGetShaderInfoLog(g_gluiShaderObjectFragmentPerVertexLight, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//-	Fragment shader:Per Vertex .
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+	Fragment shader: Per Fragment.

	fprintf(g_fpLogFile, "==>Fragment Shader: Per Fragment.");

	//	Create shader.
	g_gluiShaderObjectFragmentPerFragmentLight = glCreateShader(GL_FRAGMENT_SHADER);

	//	Provide source code.
	const GLchar *szFragmentShaderSourceCodePerFragmentLight =
		"#version 430 core"							\
		"\n"										\
		"in vec3 transformed_normals;"			\
		"in vec3 light_directionR;"			\
		"in vec3 light_directionG;"			\
		"in vec3 light_directionB;"			\
		"in vec3 viewer_vector;"			\
		"uniform vec3 u_LaR;	"				\
		"uniform vec3 u_LdR;	"				\
		"uniform vec3 u_LsR;	"				\
		"uniform vec3 u_LaG;	"				\
		"uniform vec3 u_LdG;	"				\
		"uniform vec3 u_LsG;	"				\
		"uniform vec3 u_LaB;	"				\
		"uniform vec3 u_LdB;	"				\
		"uniform vec3 u_LsB;	"				\
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
				"vec3 normalized_light_directionR = normalize(light_directionR);"											\
				"vec3 normalized_light_directionG = normalize(light_directionG);"											\
				"vec3 normalized_light_directionB = normalize(light_directionB);"											\
				"vec3 normalized_viewer_vector = normalize(viewer_vector);"											\
				/*Red Light*/
				"float tn_dot_ldR = max(dot(normalized_transformed_normals, normalized_light_directionR), 0.0);"											\
				"vec3 ambientR = u_LaR * u_Ka;"											\
				"vec3 diffuseR = u_LdR * u_Kd * tn_dot_ldR;"											\
				"vec3 reflection_vectorR = reflect(-normalized_light_directionR, normalized_transformed_normals);"											\
				"vec3 specularR = u_LsR * u_Ks * pow(max(dot(reflection_vectorR, normalized_viewer_vector), 0.0), u_material_shininess);"											\
				/*Green Light*/
				"float tn_dot_ldG = max(dot(normalized_transformed_normals, normalized_light_directionG), 0.0);"											\
				"vec3 ambientG = u_LaG * u_Ka;"											\
				"vec3 diffuseG = u_LdG * u_Kd * tn_dot_ldG;"											\
				"vec3 reflection_vectorG = reflect(-normalized_light_directionG, normalized_transformed_normals);"											\
				"vec3 specularG = u_LsG * u_Ks * pow(max(dot(reflection_vectorG, normalized_viewer_vector), 0.0), u_material_shininess);"											\
				/*Blue Light*/
				"float tn_dot_ldB = max(dot(normalized_transformed_normals, normalized_light_directionB), 0.0);"											\
				"vec3 ambientB = u_LaB * u_Ka;"											\
				"vec3 diffuseB = u_LdB * u_Kd * tn_dot_ldB;"											\
				"vec3 reflection_vectorB = reflect(-normalized_light_directionB, normalized_transformed_normals);"											\
				"vec3 specularB = u_LsB * u_Ks * pow(max(dot(reflection_vectorB, normalized_viewer_vector), 0.0), u_material_shininess);"											\
				"phong_ads_color = ambientR + ambientG + ambientB + diffuseR + diffuseG + diffuseB + specularR + specularG + specularB;"											\
			"}"											\
			"else"											\
			"{"											\
			"	phong_ads_color = vec3(1.0,1.0,1.0);"											\
			"}"											\
			"vFragColor = vec4(phong_ads_color, 1.0);"					\
			"}";

	glShaderSource(g_gluiShaderObjectFragmentPerFragmentLight, 1, &szFragmentShaderSourceCodePerFragmentLight, NULL);

	//	Compile shader.
	glCompileShader(g_gluiShaderObjectFragmentPerFragmentLight);

	gliCompileStatus = 0;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;
	glsiWritten = 0;
	glGetShaderiv(g_gluiShaderObjectFragmentPerFragmentLight, GL_COMPILE_STATUS, &gliCompileStatus);
	if (GL_FALSE == gliCompileStatus)
	{
		glGetShaderiv(g_gluiShaderObjectFragmentPerFragmentLight, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
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

		glGetShaderInfoLog(g_gluiShaderObjectFragmentPerFragmentLight, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//-	Fragment shader: Per Fragment.
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+	Shader program: Per Vertex.

	//	Create.
	g_gluiShaderObjectProgramPerVertexLight = glCreateProgram();

	//	Attach vertex shader to shader program.
	glAttachShader(g_gluiShaderObjectProgramPerVertexLight, g_gluiShaderObjectVertexPerVertexLight);

	//	Attach Fragment shader to shader program.
	glAttachShader(g_gluiShaderObjectProgramPerVertexLight, g_gluiShaderObjectFragmentPerVertexLight);

	//	IMP:Pre-link binding of shader program object with vertex shader position attribute.
	glBindAttribLocation(g_gluiShaderObjectProgramPerVertexLight, RTR_ATTRIBUTE_POSITION, "vPosition");

	glBindAttribLocation(g_gluiShaderObjectProgramPerVertexLight, RTR_ATTRIBUTE_NORMAL, "vNormal");

	//	Link shader.
	glLinkProgram(g_gluiShaderObjectProgramPerVertexLight);

	GLint gliLinkStatus;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;
	glsiWritten = 0;
	glGetProgramiv(g_gluiShaderObjectProgramPerVertexLight, GL_LINK_STATUS, &gliLinkStatus);
	if (GL_FALSE == gliLinkStatus)
	{
		glGetProgramiv(g_gluiShaderObjectProgramPerVertexLight, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
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

		glGetProgramInfoLog(g_gluiShaderObjectProgramPerVertexLight, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//+	Shader program: Per Vertex.
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+	Shader program: Per Fragment.

	//	Create.
	g_gluiShaderObjectProgramPerFragmentLight = glCreateProgram();

	//	Attach vertex shader to shader program.
	glAttachShader(g_gluiShaderObjectProgramPerFragmentLight, g_gluiShaderObjectVertexPerFragmentLight);

	//	Attach Fragment shader to shader program.
	glAttachShader(g_gluiShaderObjectProgramPerFragmentLight, g_gluiShaderObjectFragmentPerFragmentLight);

	//	IMP:Pre-link binding of shader program object with vertex shader position attribute.
	glBindAttribLocation(g_gluiShaderObjectProgramPerFragmentLight, RTR_ATTRIBUTE_POSITION, "vPosition");

	glBindAttribLocation(g_gluiShaderObjectProgramPerFragmentLight, RTR_ATTRIBUTE_NORMAL, "vNormal");

	//	Link shader.
	glLinkProgram(g_gluiShaderObjectProgramPerFragmentLight);

	gliLinkStatus = 0;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;
	glsiWritten = 0;
	glGetProgramiv(g_gluiShaderObjectProgramPerFragmentLight, GL_LINK_STATUS, &gliLinkStatus);
	if (GL_FALSE == gliLinkStatus)
	{
		glGetProgramiv(g_gluiShaderObjectProgramPerFragmentLight, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
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

		glGetProgramInfoLog(g_gluiShaderObjectProgramPerFragmentLight, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//+	Shader program: Per Fragment.
	////////////////////////////////////////////////////////////////////

	//-	Shader code
	////////////////////////////////////////////////////////////////////

	//
	//	The actual locations assigned to uniform variables are not known until the program object is linked successfully.
	//	After a program object has been linked successfully, the index values for uniform variables remain fixed until the next link command occurs.
	//

	//+	Per vertex uniform
	g_gluiModelMat4Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_model_matrix");
	if (-1 == g_gluiModelMat4Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_model_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiViewMat4Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_view_matrix");
	if (-1 == g_gluiViewMat4Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_view_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiProjectionMat4Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_projection_matrix");
	if (-1 == g_gluiProjectionMat4Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_projection_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiRotationRMat4Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_rotation_matrixR");
	if (-1 == g_gluiRotationRMat4Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_rotation_matrixR) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiRotationGMat4Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_rotation_matrixG");
	if (-1 == g_gluiRotationGMat4Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_rotation_matrixG) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiRotationBMat4Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_rotation_matrixB");
	if (-1 == g_gluiRotationBMat4Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_rotation_matrixB) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLKeyPressedUniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_L_key_pressed");
	if (-1 == g_gluiLKeyPressedUniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_L_key_pressed) failed.");
		uninitialize();
		exit(0);
	}

	//	Red Light
	g_gluiLaRVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_LaR");
	if (-1 == g_gluiLaRVec3Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LaR) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLdRVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_LdR");
	if (-1 == g_gluiLdRVec3Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LdR) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLsRVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_LsR");
	if (-1 == g_gluiLsRVec3Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LsR) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLightPositionRVec4Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_light_positionR");
	if (-1 == g_gluiLightPositionRVec4Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_light_positionR) failed.");
		uninitialize();
		exit(0);
	}

	//	Green Light
	g_gluiLaGVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_LaG");
	if (-1 == g_gluiLaGVec3Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LaR) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLdGVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_LdG");
	if (-1 == g_gluiLdGVec3Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LdG) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLsGVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_LsG");
	if (-1 == g_gluiLsGVec3Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LsG) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLightPositionGVec4Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_light_positionG");
	if (-1 == g_gluiLightPositionGVec4Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_light_positionG) failed.");
		uninitialize();
		exit(0);
	}

	//	Blue Light
	g_gluiLaBVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_LaB");
	if (-1 == g_gluiLaBVec3Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LaB) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLdBVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_LdB");
	if (-1 == g_gluiLdBVec3Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LdB) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLsBVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_LsB");
	if (-1 == g_gluiLsBVec3Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LsB) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLightPositionBVec4Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_light_positionB");
	if (-1 == g_gluiLightPositionBVec4Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_light_positionB) failed.");
		uninitialize();
		exit(0);
	}

	//	Light Material
	g_gluiKaVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_Ka");
	if (-1 == g_gluiKaVec3Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Ka) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiKdVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_Kd");
	if (-1 == g_gluiKdVec3Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Kd) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiKsVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_Ks");
	if (-1 == g_gluiKsVec3Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Ks) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiMaterialShininessUniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_material_shininess");
	if (-1 == g_gluiMaterialShininessUniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_material_shininess) failed.");
		uninitialize();
		exit(0);
	}
	//-	Per vertex uniform.

	//+	Per fragment uniform.
	g_gluiModelMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_model_matrix");
	if (-1 == g_gluiModelMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_model_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiViewMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_view_matrix");
	if (-1 == g_gluiViewMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_view_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiProjectionMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_projection_matrix");
	if (-1 == g_gluiProjectionMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_projection_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiRotationRMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_rotation_matrixR");
	if (-1 == g_gluiRotationRMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_rotation_matrixR) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiRotationGMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_rotation_matrixG");
	if (-1 == g_gluiRotationGMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_rotation_matrixG) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiRotationBMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_rotation_matrixB");
	if (-1 == g_gluiRotationBMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_rotation_matrixB) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLKeyPressedUniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_L_key_pressed");
	if (-1 == g_gluiLKeyPressedUniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_L_key_pressed) failed.");
		uninitialize();
		exit(0);
	}

	//	Red Light
	g_gluiLaRVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_LaR");
	if (-1 == g_gluiLaRVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LaR) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLdRVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_LdR");
	if (-1 == g_gluiLdRVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LdR) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLsRVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_LsR");
	if (-1 == g_gluiLsRVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LsR) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLightPositionRVec4Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_light_positionR");
	if (-1 == g_gluiLightPositionRVec4Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_light_positionR) failed.");
		uninitialize();
		exit(0);
	}

	//	Green Light
	g_gluiLaGVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_LaG");
	if (-1 == g_gluiLaGVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LaR) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLdGVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_LdG");
	if (-1 == g_gluiLdGVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LdG) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLsGVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_LsG");
	if (-1 == g_gluiLsGVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LsG) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLightPositionGVec4Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_light_positionG");
	if (-1 == g_gluiLightPositionGVec4Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_light_positionG) failed.");
		uninitialize();
		exit(0);
	}

	//	Blue Light
	g_gluiLaBVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_LaB");
	if (-1 == g_gluiLaBVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LaB) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLdBVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_LdB");
	if (-1 == g_gluiLdBVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LdB) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLsBVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_LsB");
	if (-1 == g_gluiLsBVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LsB) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLightPositionBVec4Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_light_positionB");
	if (-1 == g_gluiLightPositionBVec4Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_light_positionB) failed.");
		uninitialize();
		exit(0);
	}

	//	Light Material
	g_gluiKaVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_Ka");
	if (-1 == g_gluiKaVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Ka) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiKdVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_Kd");
	if (-1 == g_gluiKdVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Kd) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiKsVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_Ks");
	if (-1 == g_gluiKsVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Ks) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiMaterialShininessUniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_material_shininess");
	if (-1 == g_gluiMaterialShininessUniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_material_shininess) failed.");
		uninitialize();
		exit(0);
	}
	//-	Per fragment uniform.

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

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

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
	int index;
	mat4 matModel;
	mat4 matView;
	mat4 matRotationR;
	mat4 matRotationG;
	mat4 matRotationB;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);	//	Change 3 for 3D

	//	Start using opengl program.
	if (1 == g_iLightType)
	{
		glUseProgram(g_gluiShaderObjectProgramPerVertexLight);
		index = UNIFORM_INDEX_PER_VERTEX;
	}
	else
	{
		glUseProgram(g_gluiShaderObjectProgramPerFragmentLight);
		index = UNIFORM_INDEX_PER_FRAGMENT;
	}

	matRotationR = mat4::identity();
	matRotationG = mat4::identity();
	matRotationB = mat4::identity();

	if (true == g_bLight)
	{
		glUniform1i(g_gluiLKeyPressedUniform[index], 1);

		//	Red Light
		glUniform3fv(g_gluiLaRVec3Uniform[index], 1, g_glfarrLightRAmbient);	//	Ambient
		glUniform3fv(g_gluiLdRVec3Uniform[index], 1, g_glfarrLightRDiffuse);	//	Diffuse
		glUniform3fv(g_gluiLsRVec3Uniform[index], 1, g_glfarrLightRSpecular);	//	Specular

		//	Green Light
		glUniform3fv(g_gluiLaGVec3Uniform[index], 1, g_glfarrLightGAmbient);	//	Ambient
		glUniform3fv(g_gluiLdGVec3Uniform[index], 1, g_glfarrLightGDiffuse);	//	Diffuse
		glUniform3fv(g_gluiLsGVec3Uniform[index], 1, g_glfarrLightGSpecular);	//	Specular

		//	Blue Light
		glUniform3fv(g_gluiLaBVec3Uniform[index], 1, g_glfarrLightBAmbient);	//	Ambient
		glUniform3fv(g_gluiLdBVec3Uniform[index], 1, g_glfarrLightBDiffuse);	//	Diffuse
		glUniform3fv(g_gluiLsBVec3Uniform[index], 1, g_glfarrLightBSpecular);	//	Specular

		glUniform3fv(g_gluiKaVec3Uniform[index], 1, g_glfarrMaterialAmbient);
		glUniform3fv(g_gluiKdVec3Uniform[index], 1, g_glfarrMaterialDiffuse);
		glUniform3fv(g_gluiKsVec3Uniform[index], 1, g_glfarrMaterialSpecular);
		glUniform1f(g_gluiMaterialShininessUniform[index], g_glfMaterialShininess);
	}
	else
	{
		glUniform1i(g_gluiLKeyPressedUniform[index], 0);
	}

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = mat4::identity();
	matView = mat4::identity();
	matRotationR = mat4::identity();
	matRotationG = mat4::identity();
	matRotationB = mat4::identity();

	matModel = translate(0.0f, 0.0f, -3.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform[index], 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform[index], 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform[index], 1, GL_FALSE, g_matPerspectiveProjection);

	matRotationR = rotate(g_fAngleRed, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
	g_glfarrLightRPosition[1] = g_fAngleRed;
	glUniform4fv(g_gluiLightPositionRVec4Uniform[index], 1, g_glfarrLightRPosition);
	glUniformMatrix4fv(g_gluiRotationRMat4Uniform[index], 1, GL_FALSE, matRotationR);

	matRotationG = rotate(g_fAngleGreen, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
	g_glfarrLightGPosition[0] = g_fAngleGreen;
	glUniform4fv(g_gluiLightPositionGVec4Uniform[index], 1, g_glfarrLightGPosition);
	glUniformMatrix4fv(g_gluiRotationGMat4Uniform[index], 1, GL_FALSE, matRotationG);

	matRotationB = rotate(g_fAngleBlue, 0.0f, 0.0f, 1.0f);		//	Z-axis rotation
	g_glfarrLightBPosition[0] = g_fAngleBlue;
	glUniform4fv(g_gluiLightPositionBVec4Uniform[index], 1, g_glfarrLightBPosition);
	glUniformMatrix4fv(g_gluiRotationBMat4Uniform[index], 1, GL_FALSE, matRotationB);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);

	//	Stop using opengl program.
	glUseProgram(0);

	SwapBuffers(g_hdc);
}


void update()
{
	g_fAngleRed = g_fAngleRed + 0.1f;
	if (g_fAngleRed >= 360)
	{
		g_fAngleRed = 0.0f;
	}

	g_fAngleGreen = g_fAngleGreen + 0.1f;
	if (g_fAngleGreen >= 360)
	{
		g_fAngleGreen = 0.0f;
	}

	g_fAngleBlue = g_fAngleBlue + 0.1f;
	if (g_fAngleBlue >= 360)
	{
		g_fAngleBlue = 0.0f;
	}
}


VOID resize(int iWidth, int iHeight)
{
	if (0 == iHeight)
	{
		iHeight = 1;
	}

	//	perspective(float fovy, float aspect, float n, float f)
	g_matPerspectiveProjection = perspective(45, (GLfloat)iWidth / (GLfloat)iHeight, 0.1f, 100.0f);

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

	if (g_gluiShaderObjectVertexPerVertexLight)
	{
		glDetachShader(g_gluiShaderObjectProgramPerVertexLight, g_gluiShaderObjectVertexPerVertexLight);
		glDeleteShader(g_gluiShaderObjectVertexPerVertexLight);
		g_gluiShaderObjectVertexPerVertexLight = 0;
	}

	if (g_gluiShaderObjectVertexPerFragmentLight)
	{
		glDetachShader(g_gluiShaderObjectProgramPerFragmentLight, g_gluiShaderObjectVertexPerFragmentLight);
		glDeleteShader(g_gluiShaderObjectVertexPerFragmentLight);
		g_gluiShaderObjectVertexPerFragmentLight = 0;
	}

	if (g_gluiShaderObjectFragmentPerVertexLight)
	{
		glDetachShader(g_gluiShaderObjectProgramPerVertexLight, g_gluiShaderObjectFragmentPerVertexLight);
		glDeleteShader(g_gluiShaderObjectFragmentPerVertexLight);
		g_gluiShaderObjectFragmentPerVertexLight = 0;
	}

	if (g_gluiShaderObjectFragmentPerFragmentLight)
	{
		glDetachShader(g_gluiShaderObjectProgramPerFragmentLight, g_gluiShaderObjectFragmentPerFragmentLight);
		glDeleteShader(g_gluiShaderObjectFragmentPerFragmentLight);
		g_gluiShaderObjectFragmentPerFragmentLight = 0;
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

	if (g_gluiShaderObjectProgramPerVertexLight)
	{
		glDeleteProgram(g_gluiShaderObjectProgramPerVertexLight);
		g_gluiShaderObjectProgramPerVertexLight = 0;
	}

	if (g_gluiShaderObjectProgramPerFragmentLight)
	{
		glDeleteProgram(g_gluiShaderObjectProgramPerFragmentLight);
		g_gluiShaderObjectProgramPerFragmentLight = 0;
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
