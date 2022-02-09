/*#include<Windows.h>
#include<stdio.h>

#include<gl\glew.h>
#include<gl\GL.h>

#include "vmath.h"

#include "Sphere.h" //New

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")

#pragma comment(lib, "Sphere.lib") //New
*/

#include<windows.h>
#include<C:\glew\include\GL\glew.h>
#include<gl/GL.h>
#include<stdio.h>
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
	NRK_ATTRIBUTE_VERTEX = 0,
	NRK_ATTRIBUTE_COLOR,
	NRK_ATTRIBUTE_NORMAL,
	NRK_ATTRIBUTE_TEXTURE0,
};


//Prototype of WndProc() declared globally
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

//Global Variable Declarations
FILE *gpFile = NULL;

HWND ghWnd = NULL;
HDC ghdc = NULL;
HGLRC ghrc = NULL;

DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

bool gbActiveWindow = false;
bool gbQKeyIsPressed = false;
bool gbFullScreen = false;

GLuint gVertexShaderObjectForPerVertexLighting;
GLuint gFragmentShaderObjectForPerFragmentLighting;
GLuint gShaderProgramObjectForPerVertexLighting;

GLuint gVertexShaderObjectForPerFragmentLighting;
GLuint gFragmentShaderObjectForPerVertexLighting;
GLuint gShaderProgramObjectForPerFragmentLighting;

//New (For Sphere.dll)
float sphere_vertices[1146];
float sphere_normals[1146];
float sphere_textures[764];
unsigned short sphere_elements[2280];

GLuint gNumVertices;
GLuint gNumElements;

//New
GLuint gVao_Sphere;
GLuint gVbo_Sphere_Position;
GLuint gVbo_Sphere_Normal;
GLuint gVbo_Sphere_Element;

GLuint gModelMatrixUniform, gViewMatrixUniform, gProjectionMatrixUniform;

GLuint gLKeyPressedUniform, gVKeyPressedUniform, gFKeyPressedUniform;

//For Lights
GLuint gLA_Uniform;
GLuint gLD_Uniform;
GLuint gLS_Uniform;
GLuint gLightPositionUniform;

//For Materials
GLuint gKA_Uniform;
GLuint gKD_Uniform;
GLuint gKS_Uniform;
GLuint gMaterialShininessUniform;

mat4 gPerspectiveProjectionMatrix;

bool gbLight;
bool gbVertexPhong; //New
bool gbFragmentPhong; //New

GLfloat lightAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat lightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat lightSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat lightPosition[] = { 100.0f, 100.0f, 100.0f, 1.0f };

GLfloat materialAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat materialDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialShininess = 50.0f;

//main()
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	//Function Prototype
	void initialize(void);
	void display(void);
	void uninitialize(void);

	//Variable Declarations
	WNDCLASSEX wndClass;
	HWND hWnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("PerVertexFragmentToggle_PP");
	bool bDone = false;

	//Code
	//Create Log File
	if (fopen_s(&gpFile, "Log.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Log file cannot be created. \nExitting.....\n"), TEXT("Error"), MB_OK | MB_TOPMOST | MB_ICONSTOP);
		exit(0);
	}
	else
	{
		fprintf(gpFile, "Log File is successfully opened!\n");
	}

	//Initialiing members of struct WNDCLASSEX
	wndClass.cbSize = sizeof(WNDCLASSEX);
	wndClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndClass.cbClsExtra = 0;
	wndClass.cbWndExtra = 0;
	wndClass.hInstance = hInstance;
	wndClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndClass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndClass.lpfnWndProc = WndProc;
	wndClass.lpszClassName = szClassName;
	wndClass.lpszMenuName = NULL;

	//registering Class
	RegisterClassEx(&wndClass);

	//Create Window
	hWnd = CreateWindowEx(WS_EX_APPWINDOW,
		szClassName,
		TEXT("Per Vertex and Per Fragment Phong Lighting Toggle on steady Sphere"),
		WS_OVERLAPPEDWINDOW,
		100,
		100,
		WIN_WIDTH,
		WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL
	);

	ghWnd = hWnd;

	ShowWindow(hWnd, iCmdShow);
	SetForegroundWindow(hWnd);
	SetFocus(hWnd);

	//initilaize
	initialize();

	//Message Loop
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
				display();

				if (gbQKeyIsPressed == true)
					bDone = true;
			}
		}
	}
	uninitialize();

	return((int)msg.wParam);
}

//WndProc
LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	//Function Prototype
	void resize(int, int);
	void ToggleFullScreen(void);
	void uninitialize(void);

	//Variable Declarations
	static bool bIsFKeyPressed = false;
	static bool bIsLKeyPressed = false;
	static bool bIsVKeyPressed = false;

	//Code
	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			gbActiveWindow = true;
		else
			gbActiveWindow = false;
		break;

	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_KEYDOWN:
		switch (wParam)
		{

		case 0x51: //For 'Q' or 'q'
			if (gbQKeyIsPressed == false)
				gbQKeyIsPressed = true;
			break;

		case VK_ESCAPE:
			if (gbFullScreen == false)
			{
				ToggleFullScreen();
				gbFullScreen = true;
			}
			else
			{
				ToggleFullScreen();
				gbFullScreen = false;
			}
			break;

		/*case 0x46: //For 'F' or 'f' 
			if (bIsFKeyPressed == false)
			{
				gbFragmentPhong = true;
				gbVertexPhong = false;
				bIsFKeyPressed = true;
			}
			else
			{
				gbFragmentPhong = false;
				gbVertexPhong = true;
				bIsFKeyPressed = false;
			}
			break;*/

		case 0x4C: //For 'L' or 'l'
			if (bIsLKeyPressed == false)
			{
				gbLight = true;
				bIsLKeyPressed = true;
			}
			else
			{
				gbLight = false;
				bIsLKeyPressed = false;
			}
			break;

		case 0x56: //For 'V' or 'v'
			if (bIsVKeyPressed == false)
			{
				gbVertexPhong = true;
				gbFragmentPhong = false;
				bIsVKeyPressed = true;
			}
			else
			{
				gbVertexPhong = false;
				gbFragmentPhong = true;
				bIsVKeyPressed = false;
			}
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
		PostQuitMessage(0);
		break;

	default:
		break;
	}
	return(DefWindowProc(hWnd, iMsg, wParam, lParam));
}

void ToggleFullScreen(void)
{
	//Variable Declarations
	MONITORINFO mi;

	//Code
	if (gbFullScreen == false)
	{
		dwStyle = GetWindowLong(ghWnd, GWL_STYLE);

		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			mi = { sizeof(MONITORINFO) };

			if (GetWindowPlacement(ghWnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghWnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghWnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghWnd,
					HWND_TOP,
					mi.rcMonitor.left,
					mi.rcMonitor.top,
					mi.rcMonitor.right - mi.rcMonitor.left,
					mi.rcMonitor.bottom - mi.rcMonitor.top,
					SWP_NOZORDER | SWP_FRAMECHANGED
				);
			}
		}
		ShowCursor(FALSE);
	}

	else
	{
		//Code
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghWnd, &wpPrev);
		SetWindowPos(ghWnd,
			HWND_TOP,
			0,
			0,
			0,
			0,
			SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED
		);

		ShowCursor(TRUE);
	}
}

void initialize(void)
{
	//Function Prototypes
	void resize(int, int);
	void uninitialize(void);

	//Variable Declarations
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

	//Code
	ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

	//Initialization of structure PIXELFORMATDESCRIPTOR
	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32;

	ghdc = GetDC(ghWnd);

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);

	if (iPixelFormatIndex == 0)
	{
		ReleaseDC(ghWnd, ghdc);
		ghdc = NULL;
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == false)
	{
		ReleaseDC(ghWnd, ghdc);
		ghdc = NULL;
	}

	ghrc = wglCreateContext(ghdc);

	if (ghrc == NULL)
	{
		ReleaseDC(ghWnd, ghdc);
		ghdc = NULL;
	}

	if (wglMakeCurrent(ghdc, ghrc) == false)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
		ReleaseDC(ghWnd, ghdc);
		ghdc = NULL;
	}

	//--------------------GLEW--------------------
	GLenum glew_error = glewInit();

	if (glew_error != GLEW_OK)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
		ReleaseDC(ghWnd, ghdc);
		ghdc = NULL;
	}

	//Print OpenGL and GLSL Versions to file
	fprintf(gpFile, "OpenGL Version: %s\n", glGetString(GL_VERSION));
	fprintf(gpFile, "GLSL Version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	//--------------------Vertex Shader For Per Vertex Lighting--------------------
	//Create Shader
	gVertexShaderObjectForPerVertexLighting = glCreateShader(GL_VERTEX_SHADER);

	//Provide 'Per Vertex Lighting' Source Code to Shader
	const GLchar *pVLVertexShaderSourceCode =
		"#version 460" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vNormal;" \
		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"uniform int u_lighting_enabled;" \
		"uniform vec3 u_La;" \
		"uniform vec3 u_Ld;" \
		"uniform vec3 u_Ls;" \
		"uniform vec4 u_light_position;" \
		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_material_shininess;" \
		"uniform int u_vertex_lighting_enabled;" \
		"out vec3 phong_ads_color;" \
		"void main(void)" \
		"{" \
		"if(u_lighting_enabled == 1)" \
		"{" \
		"vec4 eyeCoordinates = u_view_matrix * u_model_matrix * vPosition;" \
		"vec3 transformed_normals = normalize(mat3(u_view_matrix * u_model_matrix) * vNormal);" \
		"vec3 light_direction = normalize(vec3(u_light_position) - eyeCoordinates.xyz);" \
		"float tn_dot_ld = max(dot(transformed_normals, light_direction), 0.0f);" \
		"vec3 ambient = u_La * u_Ka;" \
		"vec3 diffuse = u_Ld * u_Kd * tn_dot_ld;" \
		"vec3 reflection_vector = reflect(-light_direction, transformed_normals);" \
		"vec3 viewer_vector = normalize(-eyeCoordinates.xyz);" \
		"vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector, viewer_vector), 0.0f), u_material_shininess);" \
		"phong_ads_color = ambient + diffuse + specular;" \
		"}" \
		"else" \
		"{" \
		"phong_ads_color = vec3(1.0f, 1.0f, 1.0f);" \
		"}" \
		"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
		"}";

	glShaderSource(gVertexShaderObjectForPerVertexLighting, 1, (const GLchar **)&pVLVertexShaderSourceCode, NULL);

	//Compile Vertex Shader For Per Vertex Lighting
	glCompileShader(gVertexShaderObjectForPerVertexLighting);
	GLint iInfoLogLength = 0;
	GLint iShaderCompiledStatus = 0;
	char *szInfoLog = NULL;
	glGetShaderiv(gVertexShaderObjectForPerVertexLighting, GL_COMPILE_STATUS, &iShaderCompiledStatus);

	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(gVertexShaderObjectForPerVertexLighting, GL_INFO_LOG_LENGTH, &iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);

			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gVertexShaderObjectForPerVertexLighting, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Vertex Shader For Per Vertex Lighting Compilation Log: %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize();
				exit(0);
			}
		}
	}

	//--------------------Fragment Shader For Per Vertex Lighting--------------------
	//Create Shader
	gFragmentShaderObjectForPerVertexLighting = glCreateShader(GL_FRAGMENT_SHADER);

	//Provide 'Per Vertex Lighting' Source Code to Shader
	const GLchar *pVLFragmentShaderSourceCode =
		"#version 460" \
		"\n" \
		"in vec3 phong_ads_color;" \
		"out vec4 FragColor;" \
		"uniform int u_lighting_enabled;" \
		"uniform int u_vertex_lighting_enabled;" \
		"void main(void)" \
		"{" \
		"FragColor = vec4(phong_ads_color, 1.0f);" \
		"}";

	glShaderSource(gFragmentShaderObjectForPerVertexLighting, 1, (const GLchar **)&pVLFragmentShaderSourceCode, NULL);

	//Compile Shader
	glCompileShader(gFragmentShaderObjectForPerVertexLighting);
	glGetShaderiv(gFragmentShaderObjectForPerVertexLighting, GL_COMPILE_STATUS, &iShaderCompiledStatus);

	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(gFragmentShaderObjectForPerVertexLighting, GL_INFO_LOG_LENGTH, &iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);

			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gFragmentShaderObjectForPerVertexLighting, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Fragment Shader Compilation Log: %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize();
				exit(0);
			}
		}
	}
	
	//--------------------Vertex Shader For Per Fragment Lighting--------------------
	//Create Shader
	gVertexShaderObjectForPerFragmentLighting = glCreateShader(GL_VERTEX_SHADER);

	//Provide 'Per Fragment Lighting' Source Code to Shader
	const GLchar *pFLVertexShaderSourceCode =
		"#version 460" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vNormal;" \
		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"uniform vec4 u_light_position;" \
		"uniform int u_lighting_enabled;" \
		"uniform int u_fragment_lighting_enabled;" \
		"out vec3 transformed_normals;" \
		"out vec3 light_direction;" \
		"out vec3 viewer_vector;" \
		"void main(void)" \
		"{" \
		"if(u_lighting_enabled == 1)" \
		"{" \
		"vec4 eyeCoordinates = u_view_matrix * u_model_matrix * vPosition;" \
		"transformed_normals = mat3(u_view_matrix * u_model_matrix) * vNormal;" \
		"light_direction = vec3(u_light_position) - eyeCoordinates.xyz;" \
		"viewer_vector = -eyeCoordinates.xyz;" \
		"}" \
		"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
		"}";

	glShaderSource(gVertexShaderObjectForPerFragmentLighting, 1, (const GLchar **)&pFLVertexShaderSourceCode, NULL);

	//Compile Vertex Shader For Per Fragment Lighting
	glCompileShader(gVertexShaderObjectForPerFragmentLighting);
	glGetShaderiv(gVertexShaderObjectForPerFragmentLighting, GL_COMPILE_STATUS, &iShaderCompiledStatus);

	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(gVertexShaderObjectForPerFragmentLighting, GL_INFO_LOG_LENGTH, &iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);

			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gVertexShaderObjectForPerFragmentLighting, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Vertex Shader For Per Vertex Lighting Compilation Log: %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize();
				exit(0);
			}
		}
	}
	
	//--------------------Fragment Shader For Per Fragment Lighting--------------------
	//Create Shader
	gFragmentShaderObjectForPerFragmentLighting = glCreateShader(GL_FRAGMENT_SHADER);

	//Provide 'Per Fragment Lighting' Source Code to Shader
	const GLchar *pFLFragmentShaderSourceCode =
		"#version 460" \
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
		"uniform int u_fragment_lighting_enabled;" \
		"void main(void)" \
		"{" \
		"vec3 phong_ads_color;" \
		"if(u_lighting_enabled == 1)" \
		"{" \
		"vec3 normalized_transformed_normals = normalize(transformed_normals);" \
		"vec3 normalized_light_direction = normalize(light_direction);" \
		"vec3 normalized_viewer_vector = normalize(viewer_vector);" \
		"vec3 ambient = u_La * u_Ka;" \
		"float tn_dot_ld = max(dot(normalized_transformed_normals, normalized_light_direction), 0.0f);" \
		"vec3 diffuse = u_Ld * u_Kd * tn_dot_ld;" \
		"vec3 reflection_vector = reflect(-normalized_light_direction, normalized_transformed_normals);" \
		"vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector, normalized_viewer_vector), 0.0f), u_material_shininess);" \
		"phong_ads_color = ambient + diffuse + specular;" \
		"}" \
		"else" \
		"{" \
		"phong_ads_color = vec3(1.0f, 1.0f, 1.0f);" \
		"}" \
		"FragColor = vec4(phong_ads_color, 1.0f);" \
		"}";
		
	glShaderSource(gFragmentShaderObjectForPerFragmentLighting, 1, (const GLchar **)&pFLFragmentShaderSourceCode, NULL);

	//Compile Shader
	glCompileShader(gFragmentShaderObjectForPerFragmentLighting);
	glGetShaderiv(gFragmentShaderObjectForPerFragmentLighting, GL_COMPILE_STATUS, &iShaderCompiledStatus);

	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(gFragmentShaderObjectForPerFragmentLighting, GL_INFO_LOG_LENGTH, &iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);

			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gFragmentShaderObjectForPerFragmentLighting, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Fragment Shader Compilation Log: %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize();
				exit(0);
			}
		}
	}

	//--------------------Shader Program For Per Vertex Lighting--------------------
	//Create
	gShaderProgramObjectForPerVertexLighting = glCreateProgram();

	//Attach Vertex Shader to Shader Program
	glAttachShader(gShaderProgramObjectForPerVertexLighting, gVertexShaderObjectForPerVertexLighting);

	//Attach Frgament Shader to Shader Program
	glAttachShader(gShaderProgramObjectForPerVertexLighting, gFragmentShaderObjectForPerVertexLighting);

	//Pre-link Binding of Shader Program Object with Vertex Shader Position Attribute
	glBindAttribLocation(gShaderProgramObjectForPerVertexLighting, NRK_ATTRIBUTE_VERTEX, "vPosition");

	glBindAttribLocation(gShaderProgramObjectForPerVertexLighting, NRK_ATTRIBUTE_NORMAL, "vNormal");

	//Link Shader
	glLinkProgram(gShaderProgramObjectForPerVertexLighting);
	GLint iShaderProgramLinkStatus = 0;
	glGetProgramiv(gShaderProgramObjectForPerVertexLighting, GL_LINK_STATUS, &iShaderProgramLinkStatus);

	if (iShaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(gShaderProgramObjectForPerVertexLighting, GL_INFO_LOG_LENGTH, &iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);

			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObjectForPerVertexLighting, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Shader Program For Per Vertex Lighting Link Log: %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize();
				exit(0);
			}
		}
	}

	//Get Uniform Locations For Per Vertex Lighting
	gModelMatrixUniform = glGetUniformLocation(gShaderProgramObjectForPerVertexLighting, "u_model_matrix");

	gViewMatrixUniform = glGetUniformLocation(gShaderProgramObjectForPerVertexLighting, "u_view_matrix");

	gProjectionMatrixUniform = glGetUniformLocation(gShaderProgramObjectForPerVertexLighting, "u_projection_matrix");

	//Whether 'L'/'l' key is pressed or not
	gLKeyPressedUniform = glGetUniformLocation(gShaderProgramObjectForPerVertexLighting, "u_lighting_enabled");

	//Ambient Color Intensity of Light
	gLA_Uniform = glGetUniformLocation(gShaderProgramObjectForPerVertexLighting, "u_La");

	//Diffuse Color Intensity of Light
	gLD_Uniform = glGetUniformLocation(gShaderProgramObjectForPerVertexLighting, "u_Ld");

	//Specular Color Intensity of Light
	gLS_Uniform = glGetUniformLocation(gShaderProgramObjectForPerVertexLighting, "u_Ls");

	//Position of Light
	gLightPositionUniform = glGetUniformLocation(gShaderProgramObjectForPerVertexLighting, "u_light_position");

	//Ambient Reflective Color Intensity of Material
	gKA_Uniform = glGetUniformLocation(gShaderProgramObjectForPerVertexLighting, "u_Ka");

	//Diffuse Reflective Color Intensity of Material
	gKD_Uniform = glGetUniformLocation(gShaderProgramObjectForPerVertexLighting, "u_Kd");

	//Specular Reflective Color Intensity of Material
	gKS_Uniform = glGetUniformLocation(gShaderProgramObjectForPerVertexLighting, "u_Ks");

	//Shininess of Material (Value is conventionally between 1 to 200
	gMaterialShininessUniform = glGetUniformLocation(gShaderProgramObjectForPerVertexLighting, "u_material_shininess");

	//---------------------------------------------------------------------------------------------------------------//
	
	//--------------------Shader Program For Per Fragment Lighting--------------------
	//Create Shader Program Object 
	gShaderProgramObjectForPerFragmentLighting = glCreateProgram();

	//Attach Vertex Shader to Shader Program
	glAttachShader(gShaderProgramObjectForPerFragmentLighting, gVertexShaderObjectForPerFragmentLighting);

	//Attach Frgament Shader to Shader Program
	glAttachShader(gShaderProgramObjectForPerFragmentLighting, gFragmentShaderObjectForPerFragmentLighting);

	//Pre-link Binding of Shader Program Object with Vertex Shader Position Attribute
	glBindAttribLocation(gShaderProgramObjectForPerFragmentLighting, NRK_ATTRIBUTE_VERTEX, "vPosition");

	glBindAttribLocation(gShaderProgramObjectForPerFragmentLighting, NRK_ATTRIBUTE_NORMAL, "vNormal");

	//Link Shader For Per Fragment Lighting
	glLinkProgram(gShaderProgramObjectForPerFragmentLighting);
	iShaderProgramLinkStatus = 0;
	glGetProgramiv(gShaderProgramObjectForPerFragmentLighting, GL_LINK_STATUS, &iShaderProgramLinkStatus);

	if (iShaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(gShaderProgramObjectForPerFragmentLighting, GL_INFO_LOG_LENGTH, &iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);

			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObjectForPerFragmentLighting, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Shader Program For Per Fragment Lighting Link Log: %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize();
				exit(0);
			}
		}
	}

	//Get Uniform Locations For Per Fragment Lighting
	gModelMatrixUniform = glGetUniformLocation(gShaderProgramObjectForPerFragmentLighting, "u_model_matrix");

	gViewMatrixUniform = glGetUniformLocation(gShaderProgramObjectForPerFragmentLighting, "u_view_matrix");

	gProjectionMatrixUniform = glGetUniformLocation(gShaderProgramObjectForPerFragmentLighting, "u_projection_matrix");

	//Whether 'L'/'l' key is pressed or not
	gLKeyPressedUniform = glGetUniformLocation(gShaderProgramObjectForPerFragmentLighting, "u_lighting_enabled");

	//Ambient Color Intensity of Light
	gLA_Uniform = glGetUniformLocation(gShaderProgramObjectForPerFragmentLighting, "u_La");

	//Diffuse Color Intensity of Light
	gLD_Uniform = glGetUniformLocation(gShaderProgramObjectForPerFragmentLighting, "u_Ld");

	//Specular Color Intensity of Light
	gLS_Uniform = glGetUniformLocation(gShaderProgramObjectForPerFragmentLighting, "u_Ls");

	//Position of Light
	gLightPositionUniform = glGetUniformLocation(gShaderProgramObjectForPerFragmentLighting, "u_light_position");

	//Ambient Reflective Color Intensity of Material
	gKA_Uniform = glGetUniformLocation(gShaderProgramObjectForPerFragmentLighting, "u_Ka");

	//Diffuse Reflective Color Intensity of Material
	gKD_Uniform = glGetUniformLocation(gShaderProgramObjectForPerFragmentLighting, "u_Kd");

	//Specular Reflective Color Intensity of Material
	gKS_Uniform = glGetUniformLocation(gShaderProgramObjectForPerFragmentLighting, "u_Ks");

	//Shininess of Material (Value is conventionally between 1 to 200
	gMaterialShininessUniform = glGetUniformLocation(gShaderProgramObjectForPerFragmentLighting, "u_material_shininess");


	//Vertices, Colors, Shader Attribs, Vao, Vbo Initializations

	//New (Getting Data from DLL)---------------------------------------
	getSphereVertexData(sphere_vertices, sphere_normals, sphere_textures, sphere_elements);

	gNumVertices = getNumberOfSphereVertices();
	gNumElements = getNumberOfSphereElements();

	//--------------------Vao for Sphere--------------------
	glGenVertexArrays(1, &gVao_Sphere); //Taking Cassette
	glBindVertexArray(gVao_Sphere); //Binding with Vao

	//--------------------Vbo for Sphere Position--------------------
	glGenBuffers(1, &gVbo_Sphere_Position);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Sphere_Position); //Binding with Vbo for Sphere Position
	glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_vertices), sphere_vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(NRK_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(NRK_ATTRIBUTE_VERTEX);

	glBindBuffer(GL_ARRAY_BUFFER, 0); //Unbinding with Vabo for Sphere Position

	//--------------------Vbo for Sphere Normal--------------------
	glGenBuffers(1, &gVbo_Sphere_Normal);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Sphere_Normal); //Binding with Vbo for Cube Normal
	glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_normals), sphere_normals, GL_STATIC_DRAW);

	glVertexAttribPointer(NRK_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(NRK_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0); //Unbinding with Vbo for Sphere Normal

	//--------------------Vbo for Sphere Element--------------------
	glGenBuffers(1, &gVbo_Sphere_Element);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Sphere_Element); //Binding with Vbo for Sphere Element
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(sphere_elements), sphere_elements, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); //Unbinding with Vbo for Sphere Element

	glBindVertexArray(0); //Unbinding with Vao

	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable(GL_CULL_FACE);

	//Set Background Color
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f); //Black Color

	//Set Perspective Matrix to Identity Matrix
	gPerspectiveProjectionMatrix = mat4::identity();

	gbLight = false;
	gbVertexPhong = true;
	gbFragmentPhong = false;

	resize(WIN_WIDTH, WIN_HEIGHT);
}

void display(void)
{
	//Code
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//Start using OpenGL Program Object

	if(gbVertexPhong == true)
		glUseProgram(gShaderProgramObjectForPerVertexLighting);

	else if (gbFragmentPhong == true)
		glUseProgram(gShaderProgramObjectForPerFragmentLighting);

	if (gbLight == true)
	{
		//Set 'u_lighting_enabled' uniform
		glUniform1i(gLKeyPressedUniform, 1);

		//Setting Light's Properties
		glUniform3fv(gLA_Uniform, 1, lightAmbient);
		glUniform3fv(gLD_Uniform, 1, lightDiffuse);
		glUniform3fv(gLS_Uniform, 1, lightSpecular);
		glUniform4fv(gLightPositionUniform, 1, lightPosition);

		//Setting Material's Properties
		glUniform3fv(gKA_Uniform, 1, materialAmbient);
		glUniform3fv(gKD_Uniform, 1, materialDiffuse);
		glUniform3fv(gKS_Uniform, 1, materialSpecular);
		glUniform1f(gMaterialShininessUniform, materialShininess);
	}
	else
	{
		//Set 'u_lighting_enabled' uniform
		glUniform1i(gLKeyPressedUniform, 0);
	}

	//OpenGL Drawing

	//Set All Matrices to Identity
	mat4 modelMatrix = mat4::identity();
	mat4 viewMatrix = mat4::identity();

	//Translate Z-Axis by -2.0f
	modelMatrix = translate(0.0f, 0.0f, -2.0f);

	//Pass modelView Matrix to the Vertex Shader in 'u_model_matrix' shader variable 
	//whose position value we already calculated in initialize by using glGetUniformLocation()
	glUniformMatrix4fv(gModelMatrixUniform, 1, GL_FALSE, modelMatrix);

	//Pass modelView Matrix to the Vertex Shader in 'u_view_matrix' shader variable 
	//whose position value we already calculated in initialize by using glGetUniformLocation()
	glUniformMatrix4fv(gViewMatrixUniform, 1, GL_FALSE, viewMatrix);

	//Pass modelView Matrix to the Vertex Shader in 'u_projection_matrix' shader variable 
	//whose position value we already calculated in initialize by using glGetUniformLocation()
	glUniformMatrix4fv(gProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	//Bind with Vao
	glBindVertexArray(gVao_Sphere);

	//Draw
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Sphere_Element);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);

	glBindVertexArray(0); //Unbinding with Vao

	//Stop Using OpenGL Program Object
	glUseProgram(0);

	SwapBuffers(ghdc);
}

void resize(int width, int height)
{
	if (height == 0)
		height = 1;
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	gPerspectiveProjectionMatrix = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void uninitialize(void)
{
	if (gbFullScreen == true)
	{
		dwStyle = GetWindowLong(ghWnd, GWL_STYLE);
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghWnd, &wpPrev);
		SetWindowPos(ghWnd,
			HWND_TOP,
			0,
			0,
			0,
			0,
			SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED
		);

		ShowCursor(TRUE);
	}

	//Destroy Vao_Sphere
	if (gVao_Sphere)
	{
		glDeleteVertexArrays(1, &gVao_Sphere);
		gVao_Sphere = 0;
	}

	//Destroy Vbo Position
	if (gVbo_Sphere_Position)
	{
		glDeleteBuffers(1, &gVbo_Sphere_Position);
		gVbo_Sphere_Position = 0;
	}

	//Destroy Vbo Normal
	if (gVbo_Sphere_Normal)
	{
		glDeleteBuffers(1, &gVbo_Sphere_Normal);
		gVbo_Sphere_Normal = 0;
	}

	//Destroy Vbo Element
	if (gVbo_Sphere_Element)
	{
		glDeleteBuffers(1, &gVbo_Sphere_Element);
		gVbo_Sphere_Element = 0;
	}

	//Detach Vertex Shader from Shader Program Object For Per Vertex Lighting
	glDetachShader(gShaderProgramObjectForPerVertexLighting, gVertexShaderObjectForPerVertexLighting);

	//Detach Fragment Shader from Shader Program Object For Per Vertex Lighting
	glDetachShader(gShaderProgramObjectForPerVertexLighting, gFragmentShaderObjectForPerVertexLighting);
	
	//Delete Vertex Shader Object
	glDeleteShader(gVertexShaderObjectForPerVertexLighting);
	gVertexShaderObjectForPerVertexLighting = 0;

	//Delete Fragment Shader Object
	glDeleteShader(gFragmentShaderObjectForPerVertexLighting);
	gFragmentShaderObjectForPerVertexLighting = 0;
	
	//Delete Shader Program Object For Per Vertex Lighting
	glDeleteProgram(gShaderProgramObjectForPerVertexLighting);
	gShaderProgramObjectForPerVertexLighting = 0;

	//Detach Vertex Shader from Shader Program Object For Per Fragment Lighting
	glDetachShader(gShaderProgramObjectForPerFragmentLighting, gVertexShaderObjectForPerFragmentLighting);

	//Detach Fragment Shader from Shader Program Object For Per Fragment Lighting
	glDetachShader(gShaderProgramObjectForPerFragmentLighting, gFragmentShaderObjectForPerFragmentLighting);

	//Delete Vertex Shader Object
	glDeleteShader(gVertexShaderObjectForPerFragmentLighting);
	gVertexShaderObjectForPerFragmentLighting = 0;

	//Delete Fragment Shader Object
	glDeleteShader(gFragmentShaderObjectForPerFragmentLighting);
	gFragmentShaderObjectForPerFragmentLighting = 0;
	
	//Delete Shader Program Object For Per Fragment Lighting
	glDeleteProgram(gShaderProgramObjectForPerFragmentLighting);
	gShaderProgramObjectForPerFragmentLighting = 0;

	//De-Select the Rendering Context
	wglMakeCurrent(NULL, NULL);

	//Delete the Rendering Context
	wglDeleteContext(ghrc);
	ghrc = NULL;

	//Delete the Device Context
	ReleaseDC(ghWnd, ghdc);
	ghdc = NULL;

	if (gpFile)
	{
		fprintf(gpFile, "Log File is successfully closed.\n");
		fclose(gpFile);
		gpFile = NULL;
	}
}
