#include<windows.h>
#include<C:\glew\include\GL\glew.h>
#include<gl/GL.h>
#include<stdio.h>
#include<math.h>
#include"../../Resources/vmath.h"
#include"../../Resources/Sphere.h"

#pragma comment(lib,"User32.lib")
#pragma comment(lib,"GDI32.lib")
#pragma comment(lib,"C:\\glew\\lib\\Release\\Win32\\glew32.lib")
#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"..\\..\\Resources\\Sphere.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

using namespace vmath;
		
enum
{
	HAD_ATTRIBUTE_POSITION = 0,
	HAD_ATTRIBUTE_COLOR,
	HAD_ATTRIBUTE_NORMAL,
	HAD_ATTRIBUTE_TEXTURE0,
};

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

FILE *gpFile;
HWND ghwnd;
HDC ghdc;
HGLRC ghrc;
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
bool gbActiveWindow = false;
bool gbFullscreen = false;
bool gbIsEscapeKeyPressed = false;

GLuint gVertexShaderObject;
GLuint gFragmentShaderObject;
GLuint gShaderProgramObject;

GLuint gVao_Sphere;
GLuint gVbo_Position, gVbo_Normal, gVbo_Elements;

GLuint gModelMatrixUniform, gViewMatrixUniform, gProjectionMatrixUniform;
GLuint gLKeyPressedUniform;

GLuint gLaUniform, gLdUniform, gLsUniform;
GLuint gLightPositionUniform;

GLuint gKaUniform, gKdUniform, gKsUniform;
GLuint gMaterialShininessUniform;

GLfloat gAngle_Sphere;

GLfloat lightAmbient[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat lightDiffuse[] = { 1.0f,1.0f,1.0f,1.0f };
GLfloat lightSpecular[] = { 1.0f,1.0f,1.0f,1.0f };
GLfloat lightPosition[] = { 100.0f,100.0f,100.0f,1.0f };

GLfloat material_ambient_1[] = { 0.0215f,0.1745f,0.0215f,1.0f };
GLfloat material_diffuse_1[] = { 0.07568f,0.61424f,0.07568f,1.0f };
GLfloat material_specular_1[] = { 0.633f,0.727811f,0.633f,1.0f };
GLfloat material_shininess_1 = 0.6f * 128.0f;

GLfloat material_ambient_2[] = { 0.135f,0.2225f,0.1575f,1.0f };
GLfloat material_diffuse_2[] = { 0.54f,0.89f,0.63f,1.0f };
GLfloat material_specular_2[] = { 0.316228f,0.316228f,0.316228f,1.0f };
GLfloat material_shininess_2 = 0.1f * 128.0f;

GLfloat material_ambient_3[] = { 0.05375f,0.05f,0.06625f,1.0f };
GLfloat material_diffuse_3[] = { 0.18275f,0.17f,0.22525f,1.0f };
GLfloat material_specular_3[] = { 0.332741f,0.328634f,0.346435f,1.0f };
GLfloat material_shininess_3 = 0.3f * 128.0f;

GLfloat material_ambient_4[] = { 0.25f,0.20725f,0.20725f,1.0f };
GLfloat material_diffuse_4[] = { 1.0f,0.829f,0.829f,1.0f };
GLfloat material_specular_4[] = { 0.296648f,0.296648f,0.296648f,1.0f };
GLfloat material_shininess_4 = 0.088f * 128.0f;

GLfloat material_ambient_5[] = { 0.1745f,0.01175f,0.01175f,1.0f };
GLfloat material_diffuse_5[] = { 0.61424f,0.04136f,0.04136f,1.0f };
GLfloat material_specular_5[] = { 0.727811f,0.626959f,0.626959f,1.0f };
GLfloat material_shininess_5 = 0.6f * 128.0f;

GLfloat material_ambient_6[] = { 0.1f,0.18725f,0.1745f,1.0f };
GLfloat material_diffuse_6[] = { 0.396f,0.74151f,0.69102f,1.0f };
GLfloat material_specular_6[] = { 0.297254f,0.30829f,0.306678f,1.0f };
GLfloat material_shininess_6 = 0.1f * 128.0f;

GLfloat material_ambient_7[] = { 0.329412f,0.223529f,0.027451f,1.0f };
GLfloat material_diffuse_7[] = { 0.780392f,0.568627f,0.113725f,1.0f };
GLfloat material_specular_7[] = { 0.992157f,0.941176f,0.807843f,1.0f };
GLfloat material_shininess_7 = 0.21794872f * 128.0f;

GLfloat material_ambient_8[] = { 0.2125f,0.1275f,0.054f,1.0f };
GLfloat material_diffuse_8[] = { 0.714f,0.4284f,0.18144f,1.0f };
GLfloat material_specular_8[] = { 0.393548f,0.271906f,0.166721f,1.0f };
GLfloat material_shininess_8 = 0.2f * 128.0f;

GLfloat material_ambient_9[] = { 0.25f,0.25f,0.25f,1.0f };
GLfloat material_diffuse_9[] = { 0.4f,0.4f,0.4f,1.0f };
GLfloat material_specular_9[] = { 0.774597f,0.774597f,0.774597f,1.0f };
GLfloat material_shininess_9 = 0.6f * 128.0f;

GLfloat material_ambient_10[] = { 0.19125f,0.0735f,0.0225f,1.0f };
GLfloat material_diffuse_10[] = { 0.7038f,0.27048f,0.0828f,1.0f };
GLfloat material_specular_10[] = { 0.256777f,0.137622f,0.086014f,1.0f };
GLfloat material_shininess_10 = 0.1f * 128.0f;

GLfloat material_ambient_11[] = { 0.24725f,0.1995f,0.0745f,1.0f };
GLfloat material_diffuse_11[] = { 0.75164f,0.60648f,0.22648f,1.0f };
GLfloat material_specular_11[] = { 0.628281f,0.555802f,0.366065f,1.0f };
GLfloat material_shininess_11 = 0.4f * 128.0f;

GLfloat material_ambient_12[] = { 0.19225f,0.19225f,0.19225f,1.0f };
GLfloat material_diffuse_12[] = { 0.50754f,0.50754f,0.50754f,1.0f };
GLfloat material_specular_12[] = { 0.508273f,0.508273f,0.508273f,1.0f };
GLfloat material_shininess_12 = 0.4f * 128.0f;

GLfloat material_ambient_13[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat material_diffuse_13[] = { 0.01f,0.01f,0.01f,1.0f };
GLfloat material_specular_13[] = { 0.5f,0.5f,0.5f,1.0f };
GLfloat material_shininess_13 = 0.25f * 128.0f;

GLfloat material_ambient_14[] = { 0.0f,0.1f,0.06f,1.0f };
GLfloat material_diffuse_14[] = { 0.0f,0.50980392f,0.50980392f,1.0f };
GLfloat material_specular_14[] = { 0.50196078f,0.50196078f,0.50196078f,1.0f };
GLfloat material_shininess_14 = 0.25f * 128.0f;

GLfloat material_ambient_15[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat material_diffuse_15[] = { 0.1f,0.35f,0.1f,1.0f };
GLfloat material_specular_15[] = { 0.45f,0.55f,0.45f,1.0f };
GLfloat material_shininess_15 = 0.25f * 128.0f;

GLfloat material_ambient_16[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat material_diffuse_16[] = { 0.5f,0.0f,0.0f,1.0f };
GLfloat material_specular_16[] = { 0.7f,0.6f,0.6f,1.0f };
GLfloat material_shininess_16 = 0.25f * 128.0f;

GLfloat material_ambient_17[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat material_diffuse_17[] = { 0.55f,0.55f,0.55f,1.0f };
GLfloat material_specular_17[] = { 0.70f,0.70f,0.70f,1.0f };
GLfloat material_shininess_17 = 0.25f * 128.0f;

GLfloat material_ambient_18[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat material_diffuse_18[] = { 0.5f,0.5f,0.0f,1.0f };
GLfloat material_specular_18[] = { 0.6f,0.6f,0.5f,1.0f };
GLfloat material_shininess_18 = 0.25f * 128.0f;

GLfloat material_ambient_19[] = { 0.02f,0.02f,0.02f,1.0f };
GLfloat material_diffuse_19[] = { 0.1f,0.1f,0.1f,1.0f };
GLfloat material_specular_19[] = { 0.4f,0.4f,0.4f,1.0f };
GLfloat material_shininess_19 = 0.078125f * 128.0f;

GLfloat material_ambient_20[] = { 0.0f,0.05f,0.05f,1.0f };
GLfloat material_diffuse_20[] = { 0.4f,0.5f,0.5f,1.0f };
GLfloat material_specular_20[] = { 0.04f,0.7f,0.7f,1.0f };
GLfloat material_shininess_20 = 0.078125f * 128.0f;

GLfloat material_ambient_21[] = { 0.0f,0.05f,0.0f,1.0f };
GLfloat material_diffuse_21[] = { 0.4f,0.5f,0.4f,1.0f };
GLfloat material_specular_21[] = { 0.04f,0.7f,0.04f,1.0f };
GLfloat material_shininess_21 = 0.078125f * 128.0f;

GLfloat material_ambient_22[] = { 0.05f,0.0f,0.0f,1.0f };
GLfloat material_diffuse_22[] = { 0.5f,0.4f,0.4f,1.0f };
GLfloat material_specular_22[] = { 0.7f,0.04f,0.04f,1.0f };
GLfloat material_shininess_22 = 0.078125f * 128.0f;

GLfloat material_ambient_23[] = { 0.05f,0.05f,0.05f,1.0f };
GLfloat material_diffuse_23[] = { 0.5f,0.5f,0.5f,1.0f };
GLfloat material_specular_23[] = { 0.7f,0.7f,0.7f,1.0f };
GLfloat material_shininess_23 = 0.078125f * 128.0f;

GLfloat material_ambient_24[] = { 0.05f,0.05f,0.0f,1.0f };
GLfloat material_diffuse_24[] = { 0.5f,0.5f,0.4f,1.0f };
GLfloat material_specular_24[] = { 0.7f,0.7f,0.04f,1.0f };
GLfloat material_shininess_24 = 0.078125f * 128.0f;


GLfloat sphere_vertices[1146];
GLfloat sphere_normals[1146];
GLfloat sphere_textures[764];
unsigned short sphere_elements[2280];
unsigned int gNumVertices, gNumElements;

mat4 gPerspectiveProjectionMatrix;

bool gbAnimate = false;
bool gbLight = false;
bool gbIsAKeyPressed = false;
bool gbIsLKeyPressed = false;
bool gbIsXKeyPressed = false;
bool gbIsYKeyPressed = false;
bool gbIsZKeyPressed = false;

int giHeight, giWidth;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	void initialize(void);
	void display(void);
	void update(void);
	void uninitialize(int);
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("My App");
	bool bDone = false;

	if (fopen_s(&gpFile, "Log.txt", "w") != NULL)
	{
		MessageBox(NULL, TEXT("Cannot Create Log File !!!"), TEXT("Error"), MB_OK);
		exit(EXIT_FAILURE);
	}
	else
		fprintf(gpFile, "Log File Created Successfully...\n");

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.hInstance = hInstance;
	wndclass.lpszClassName = szClassName;
	wndclass.lpszMenuName = NULL;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);

	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, TEXT("OpenGLPP : 3D Rotation"), WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE, 100, 100, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);
	if (hwnd == NULL)
	{
		fprintf(gpFile, "Cannot Create Window...\n");
		uninitialize(1);
	}

	ghwnd = hwnd;

	ShowWindow(hwnd, iCmdShow);
	SetFocus(hwnd);
	SetForegroundWindow(hwnd);

	initialize();

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
				if (gbIsEscapeKeyPressed == true)
					bDone = true;
				if (gbAnimate == true)
					update();
				display();
			}
		}
	}

	uninitialize(0);
	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	void resize(int, int);
	void ToggleFullscreen(void);
	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			gbActiveWindow = true;
		else
			gbActiveWindow = false;
		break;
	case WM_CREATE:
		break;
	case WM_SIZE:
		giWidth = LOWORD(lParam);
		giHeight = HIWORD(lParam);
		resize(LOWORD(lParam), HIWORD(lParam));
		break;
	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			gbIsEscapeKeyPressed = true;
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

		case 0x41:
			if (gbIsAKeyPressed == false)
			{
				gbAnimate = true;
				gbIsAKeyPressed = true;
			}
			else
			{
				gbAnimate = false;
				gbIsAKeyPressed = false;
			}
			break;

		case 0x4C:
			if (gbIsLKeyPressed == false)
			{
				gbLight = true;
				gbIsLKeyPressed = true;
			}
			else
			{
				gbLight = false;
				gbIsLKeyPressed = false;
			}
			break;

		case 0x58:
			if (gbIsXKeyPressed == false)
			{
				gbIsXKeyPressed = true;
				gbIsYKeyPressed = false;
				gbIsZKeyPressed = false;
			}
			else
				gbIsXKeyPressed = false;
			break;

		case 0x59:
			if (gbIsYKeyPressed == false)
			{
				gbIsXKeyPressed = false;
				gbIsYKeyPressed = true;
				gbIsZKeyPressed = false;
			}
			else
				gbIsYKeyPressed = false;
			break;

		case 0x5A:
			if (gbIsZKeyPressed == false)
			{
				gbIsXKeyPressed = false;
				gbIsYKeyPressed = false;
				gbIsZKeyPressed = true;
			}
			else
				gbIsZKeyPressed = false;
			break;
		}
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void initialize(void)
{
	void resize(int, int);
	void uninitialize(int);
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

	ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 24;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32;

	ghdc = GetDC(ghwnd);
	if (ghdc == NULL)
	{
		fprintf(gpFile, "GetDC() Failed.\n");
		uninitialize(1);
	}

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0)
	{
		fprintf(gpFile, "ChoosePixelFormat() Failed.\n");
		uninitialize(1);
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		fprintf(gpFile, "SetPixelFormat() Failed.\n");
		uninitialize(1);
	}

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		fprintf(gpFile, "wglCreateContext() Failed.\n");
		uninitialize(1);
	}

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		fprintf(gpFile, "wglMakeCurrent() Failed");
		uninitialize(1);
	}

	GLenum glew_error = glewInit();
	if (glew_error != GLEW_OK)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	getSphereVertexData(sphere_vertices, sphere_normals, sphere_textures, sphere_elements);

	gNumVertices = getNumberOfSphereVertices();
	gNumElements = getNumberOfSphereElements();
	//Vertex Shader
	gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *vertexShaderSourceCode =
		"#version 450" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vNormal;" \
		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"uniform int u_lighting_enabled;" \
		"uniform vec4 u_light_position;" \
		"out vec3 transformed_normals;" \
		"out vec3 light_direction;" \
		"out vec3 viewer_vector;" \
		"void main(void)" \
		"{" \
		"if(u_lighting_enabled==1)" \
		"{" \
		"vec4 eye_coordinates = u_view_matrix*u_model_matrix*vPosition;" \
		"transformed_normals = mat3(u_view_matrix*u_model_matrix)*vNormal;" \
		"light_direction = vec3(u_light_position)-eye_coordinates.xyz;" \
		"viewer_vector = -eye_coordinates.xyz;" \
		"}" \
		"gl_Position = u_projection_matrix*u_view_matrix*u_model_matrix*vPosition;" \
		"}";

	glShaderSource(gVertexShaderObject, 1, (const GLchar **)&vertexShaderSourceCode, NULL);

	glCompileShader(gVertexShaderObject);
	GLint iInfoLogLength = 0;
	GLint iShaderCompiledStatus = 0;
	char *szInfoLog = NULL;

	glGetShaderiv(gVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(gVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gVertexShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Vertex Shader Compilation Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize(1);
				exit(0);
			}
		}
	}

	//Fragment Shader
	gFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *fragmentShaderSourceCode =
		"#version 450" \
		"\n" \
		"in vec3 transformed_normals;" \
		"in vec3 light_direction;" \
		"in vec3 viewer_vector;" \
		"out vec4 FragColor;" \
		"uniform vec3 u_La;" \
		"uniform vec3 u_Ld;" \
		"uniform vec3 u_Ls;" \
		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_material_shininess;" \
		"uniform int u_lighting_enabled;" \
		"void main(void)" \
		"{" \
		"vec3 phong_ads_color;" \
		"if(u_lighting_enabled == 1)" \
		"{" \
		"vec3 normalized_transformed_normals = normalize(transformed_normals);" \
		"vec3 normalized_light_direction = normalize(light_direction);" \
		"vec3 normalized_viewer_vector = normalize(viewer_vector);" \
		"vec3 ambient = u_La * u_Ka;" \
		"float tn_dot_ld = max(dot(normalized_transformed_normals,normalized_light_direction),0.0);" \
		"vec3 diffuse = u_Ld * u_Kd * tn_dot_ld;" \
		"vec3 reflection_vector = reflect(-normalized_light_direction,normalized_transformed_normals);" \
		"vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector,normalized_viewer_vector),0.0),u_material_shininess);" \
		"phong_ads_color = ambient + diffuse + specular;" \
		"}" \
		"else" \
		"{" \
		"phong_ads_color = vec3(1.0f,1.0f,1.0f);" \
		"}" \
		"FragColor = vec4(phong_ads_color,1.0);" \
		"}";

	glShaderSource(gFragmentShaderObject, 1, (const GLchar **)&fragmentShaderSourceCode, NULL);

	glCompileShader(gFragmentShaderObject);

	glGetShaderiv(gFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(gFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gFragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Fragment Shader Compilation Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize(1);
				exit(0);
			}
		}
	}

	//Shader Program
	gShaderProgramObject = glCreateProgram();

	glAttachShader(gShaderProgramObject, gVertexShaderObject);

	glAttachShader(gShaderProgramObject, gFragmentShaderObject);

	glBindAttribLocation(gShaderProgramObject, HAD_ATTRIBUTE_POSITION, "vPosition");

	glBindAttribLocation(gShaderProgramObject, HAD_ATTRIBUTE_NORMAL, "vNormal");

	glLinkProgram(gShaderProgramObject);

	GLint iShaderProgramLinkStatus = 0;

	glGetProgramiv(gShaderProgramObject, GL_LINK_STATUS, &iShaderProgramLinkStatus);
	if (iShaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Shader Program Link Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize(1);
				exit(0);
			}
		}
	}

	gModelMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_model_matrix");
	gViewMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_view_matrix");
	gProjectionMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_projection_matrix");

	gLKeyPressedUniform = glGetUniformLocation(gShaderProgramObject, "u_lighting_enabled");

	gLaUniform = glGetUniformLocation(gShaderProgramObject, "u_La");
	gLdUniform = glGetUniformLocation(gShaderProgramObject, "u_Ld");
	gLsUniform = glGetUniformLocation(gShaderProgramObject, "u_Ls");

	gLightPositionUniform = glGetUniformLocation(gShaderProgramObject, "u_light_position");

	gKaUniform = glGetUniformLocation(gShaderProgramObject, "u_Ka");
	gKdUniform = glGetUniformLocation(gShaderProgramObject, "u_Kd");
	gKsUniform = glGetUniformLocation(gShaderProgramObject, "u_Ks");

	gMaterialShininessUniform = glGetUniformLocation(gShaderProgramObject, "u_material_shininess");

	/*****************VAO For Cube*****************/
	glGenVertexArrays(1, &gVao_Sphere);
	glBindVertexArray(gVao_Sphere);

	/*****************Cube Position****************/
	glGenBuffers(1, &gVbo_Position);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_vertices), sphere_vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/*****************Cube Color****************/
	glGenBuffers(1, &gVbo_Normal);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Normal);
	glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_normals), sphere_normals, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &gVbo_Elements);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(sphere_elements), sphere_elements, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glShadeModel(GL_SMOOTH);
	//glHint(GL_PERSPECTIVE_CORRECTION_HINT , GL_NICEST);
	//glEnable(GL_CULL_FACE);
	
	glClearColor(0.25f, 0.25f, 0.25f, 0.0f);

	gPerspectiveProjectionMatrix = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
}

void display(void)
{
	GLfloat fradius = 10.0f;
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	void Draw_Sphere(void);
	void Draw_Sphere_1(void);
	void Draw_Sphere_2(void);
	void Draw_Sphere_3(void);
	void Draw_Sphere_4(void);
	void Draw_Sphere_5(void);
	void Draw_Sphere_6(void);
	void Draw_Sphere_7(void);
	void Draw_Sphere_8(void);
	void Draw_Sphere_9(void);
	void Draw_Sphere_10(void);
	void Draw_Sphere_11(void);
	void Draw_Sphere_12(void);
	void Draw_Sphere_13(void);
	void Draw_Sphere_14(void);
	void Draw_Sphere_15(void);
	void Draw_Sphere_16(void);
	void Draw_Sphere_17(void);
	void Draw_Sphere_18(void);
	void Draw_Sphere_19(void);
	void Draw_Sphere_20(void);
	void Draw_Sphere_21(void);
	void Draw_Sphere_22(void);
	void Draw_Sphere_23(void);
	void Draw_Sphere_24(void);
	//Use Shader Program Object
	glUseProgram(gShaderProgramObject);

	if (gbLight == true)
	{
		if (gbIsXKeyPressed == true)
		{
			lightPosition[0] = 0.0f;
			lightPosition[1] = (GLfloat)sin(gAngle_Sphere)*fradius;
			lightPosition[2] = (GLfloat)(cos(gAngle_Sphere)*fradius - 2.0f);
		}
		else if (gbIsYKeyPressed == true)
		{
			lightPosition[0] = (GLfloat)sin(gAngle_Sphere)*fradius;
			lightPosition[2] = (GLfloat)(cos(gAngle_Sphere)*fradius - 2.0f);
			lightPosition[1] = 0.0f;
		}
		else if (gbIsZKeyPressed == true)
		{
			lightPosition[0] = (GLfloat)sin(gAngle_Sphere)*fradius;
			lightPosition[1] = (GLfloat)cos(gAngle_Sphere)*fradius;
			lightPosition[2] = -2.0f;
		}
		else if (gbIsXKeyPressed == false && gbIsYKeyPressed == false && gbIsZKeyPressed == false)
		{
			lightPosition[0] = 0.0f;
			lightPosition[1] = 0.0f;
			lightPosition[2] = 0.0f;
		}

		glUniform1i(gLKeyPressedUniform, 1);

		glUniform3fv(gLaUniform, 1, lightAmbient);
		glUniform3fv(gLdUniform, 1, lightDiffuse);
		glUniform3fv(gLsUniform, 1, lightSpecular);
		glUniform4fv(gLightPositionUniform, 1, lightPosition);
	}
	else
	{
		glUniform1i(gLKeyPressedUniform, 0);
	}

	mat4 modelMatrix = mat4::identity();
	mat4 viewMatrix = mat4::identity();
	mat4 scaleMatrix = mat4::identity();

	modelMatrix = translate(0.0f, 0.0f, -4.0f);

	scaleMatrix = scale((GLfloat)giHeight/(GLfloat)giWidth, 1.0f, 1.0f);

	//modelMatrix = modelMatrix * scaleMatrix;

	glUniformMatrix4fv(gModelMatrixUniform, 1, GL_FALSE, modelMatrix);

	glUniformMatrix4fv(gViewMatrixUniform, 1, GL_FALSE, viewMatrix);

	glUniformMatrix4fv(gProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	glViewport(0, giHeight * 5 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_1();
	glViewport(0, giHeight * 4 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_2();
	glViewport(0, giHeight * 3 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_3();
	glViewport(0, giHeight * 2 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_4();
	glViewport(0, giHeight * 1 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_5();
	glViewport(0, -30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_6();
	glViewport(giWidth / 4, giHeight * 5 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_7();
	glViewport(giWidth / 4, giHeight * 4 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_8();
	glViewport(giWidth / 4, giHeight * 3 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_9();
	glViewport(giWidth / 4, giHeight * 2 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_10();
	glViewport(giWidth / 4, giHeight * 1 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_11();
	glViewport(giWidth / 4, -30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_12();
	glViewport(giWidth / 2, giHeight * 5 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_13();
	glViewport(giWidth / 2, giHeight * 4 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_14();
	glViewport(giWidth / 2, giHeight * 3 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_15();
	glViewport(giWidth / 2, giHeight * 2 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_16();
	glViewport(giWidth / 2, giHeight * 1 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_17();
	glViewport(giWidth / 2, -30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_18();
	glViewport((giWidth /2) + (giWidth /4), giHeight * 5 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_19();
	glViewport((giWidth / 2) + (giWidth / 4), giHeight * 4 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_20();
	glViewport((giWidth / 2) + (giWidth / 4), giHeight * 3 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_21();
	glViewport((giWidth / 2) + (giWidth / 4), giHeight * 2 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_22();
	glViewport((giWidth / 2) + (giWidth / 4), giHeight * 1 / 6 - 30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_23();
	glViewport((giWidth / 2) + (giWidth / 4), -30, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	Draw_Sphere_24();

	glBindVertexArray(0);

	glUseProgram(0);

	SwapBuffers(ghdc);
}

void update(void)
{
	gAngle_Sphere = gAngle_Sphere + 0.01f;
	if (gAngle_Sphere >= 360.0f)
		gAngle_Sphere = 0.0f;
}

void Draw_Sphere_1(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_1);
		glUniform3fv(gKdUniform, 1, material_diffuse_1);
		glUniform3fv(gKsUniform, 1, material_specular_1);
		glUniform1f(gMaterialShininessUniform, material_shininess_1);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_2(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_2);
		glUniform3fv(gKdUniform, 1, material_diffuse_2);
		glUniform3fv(gKsUniform, 1, material_specular_2);
		glUniform1f(gMaterialShininessUniform, material_shininess_2);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_3(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_3);
		glUniform3fv(gKdUniform, 1, material_diffuse_3);
		glUniform3fv(gKsUniform, 1, material_specular_3);
		glUniform1f(gMaterialShininessUniform, material_shininess_3);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_4(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_4);
		glUniform3fv(gKdUniform, 1, material_diffuse_4);
		glUniform3fv(gKsUniform, 1, material_specular_4);
		glUniform1f(gMaterialShininessUniform, material_shininess_4);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_5(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_5);
		glUniform3fv(gKdUniform, 1, material_diffuse_5);
		glUniform3fv(gKsUniform, 1, material_specular_5);
		glUniform1f(gMaterialShininessUniform, material_shininess_5);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_6(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_6);
		glUniform3fv(gKdUniform, 1, material_diffuse_6);
		glUniform3fv(gKsUniform, 1, material_specular_6);
		glUniform1f(gMaterialShininessUniform, material_shininess_6);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_7(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_7);
		glUniform3fv(gKdUniform, 1, material_diffuse_7);
		glUniform3fv(gKsUniform, 1, material_specular_7);
		glUniform1f(gMaterialShininessUniform, material_shininess_7);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_8(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_8);
		glUniform3fv(gKdUniform, 1, material_diffuse_8);
		glUniform3fv(gKsUniform, 1, material_specular_8);
		glUniform1f(gMaterialShininessUniform, material_shininess_8);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_9(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_9);
		glUniform3fv(gKdUniform, 1, material_diffuse_9);
		glUniform3fv(gKsUniform, 1, material_specular_9);
		glUniform1f(gMaterialShininessUniform, material_shininess_9);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_10(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_10);
		glUniform3fv(gKdUniform, 1, material_diffuse_10);
		glUniform3fv(gKsUniform, 1, material_specular_10);
		glUniform1f(gMaterialShininessUniform, material_shininess_10);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_11(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_11);
		glUniform3fv(gKdUniform, 1, material_diffuse_11);
		glUniform3fv(gKsUniform, 1, material_specular_11);
		glUniform1f(gMaterialShininessUniform, material_shininess_11);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}


void Draw_Sphere_12(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_12);
		glUniform3fv(gKdUniform, 1, material_diffuse_12);
		glUniform3fv(gKsUniform, 1, material_specular_12);
		glUniform1f(gMaterialShininessUniform, material_shininess_12);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_13(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_13);
		glUniform3fv(gKdUniform, 1, material_diffuse_13);
		glUniform3fv(gKsUniform, 1, material_specular_13);
		glUniform1f(gMaterialShininessUniform, material_shininess_13);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_14(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_14);
		glUniform3fv(gKdUniform, 1, material_diffuse_14);
		glUniform3fv(gKsUniform, 1, material_specular_14);
		glUniform1f(gMaterialShininessUniform, material_shininess_14);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}


void Draw_Sphere_15(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_15);
		glUniform3fv(gKdUniform, 1, material_diffuse_15);
		glUniform3fv(gKsUniform, 1, material_specular_15);
		glUniform1f(gMaterialShininessUniform, material_shininess_15);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_16(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_16);
		glUniform3fv(gKdUniform, 1, material_diffuse_16);
		glUniform3fv(gKsUniform, 1, material_specular_16);
		glUniform1f(gMaterialShininessUniform, material_shininess_16);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_17(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_17);
		glUniform3fv(gKdUniform, 1, material_diffuse_17);
		glUniform3fv(gKsUniform, 1, material_specular_17);
		glUniform1f(gMaterialShininessUniform, material_shininess_17);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_18(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_18);
		glUniform3fv(gKdUniform, 1, material_diffuse_18);
		glUniform3fv(gKsUniform, 1, material_specular_18);
		glUniform1f(gMaterialShininessUniform, material_shininess_18);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_19(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_19);
		glUniform3fv(gKdUniform, 1, material_diffuse_19);
		glUniform3fv(gKsUniform, 1, material_specular_19);
		glUniform1f(gMaterialShininessUniform, material_shininess_19);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_20(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_20);
		glUniform3fv(gKdUniform, 1, material_diffuse_20);
		glUniform3fv(gKsUniform, 1, material_specular_20);
		glUniform1f(gMaterialShininessUniform, material_shininess_20);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_21(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_21);
		glUniform3fv(gKdUniform, 1, material_diffuse_21);
		glUniform3fv(gKsUniform, 1, material_specular_21);
		glUniform1f(gMaterialShininessUniform, material_shininess_21);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_22(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_22);
		glUniform3fv(gKdUniform, 1, material_diffuse_22);
		glUniform3fv(gKsUniform, 1, material_specular_22);
		glUniform1f(gMaterialShininessUniform, material_shininess_22);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_23(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_23);
		glUniform3fv(gKdUniform, 1, material_diffuse_23);
		glUniform3fv(gKsUniform, 1, material_specular_23);
		glUniform1f(gMaterialShininessUniform, material_shininess_23);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void Draw_Sphere_24(void)
{
	if (gbLight == true)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_24);
		glUniform3fv(gKdUniform, 1, material_diffuse_24);
		glUniform3fv(gKsUniform, 1, material_specular_24);
		glUniform1f(gMaterialShininessUniform, material_shininess_24);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

void resize(int width, int height)
{
	if (height == 0)
		height = 1;
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	gPerspectiveProjectionMatrix = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
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
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}
}

void uninitialize(int i_Exit_Flag)
{
	if (gbFullscreen == false)
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}

	if (gVao_Sphere)
	{
		glDeleteVertexArrays(1, &gVao_Sphere);
		gVao_Sphere = 0;
	}

	if (gVbo_Position)
	{
		glDeleteBuffers(1, &gVbo_Position);
		gVbo_Position = 0;
	}

	if (gVbo_Normal)
	{
		glDeleteBuffers(1, &gVbo_Normal);
		gVbo_Normal = 0;
	}

	if (gVbo_Elements)
	{
		glDeleteBuffers(1, &gVbo_Elements);
		gVbo_Elements = 0;
	}

	//Detach Shader 
	glDetachShader(gShaderProgramObject, gVertexShaderObject);
	glDetachShader(gShaderProgramObject, gFragmentShaderObject);

	//Delete Shader
	glDeleteShader(gVertexShaderObject);
	gVertexShaderObject = 0;

	glDeleteShader(gFragmentShaderObject);
	gFragmentShaderObject = 0;

	//Delete Program
	glDeleteProgram(gShaderProgramObject);
	gShaderProgramObject = 0;

	//Stray call to glUseProgram(0)
	glUseProgram(0);

	wglMakeCurrent(NULL, NULL);

	if (ghrc != NULL)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}

	if (ghdc != NULL)
	{
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	if (i_Exit_Flag == 0)
	{
		fprintf(gpFile, "Log File Closed Successfully");
	}
	else if (i_Exit_Flag == 1)
	{
		fprintf(gpFile, "Log File Closed Erroniously");
	}

	fclose(gpFile);
	gpFile = NULL;

	DestroyWindow(ghwnd);
}