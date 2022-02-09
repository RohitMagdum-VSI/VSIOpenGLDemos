#include<windows.h>
#include<C:\glew\include\GL\glew.h>
#include<gl/GL.h>
#include<stdio.h>
#include"../../Resources/vmath.h"
#include"../../Resources/Sphere.h"
#include<math.h>

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

GLuint gLaUniform_red, gLdUniform_red, gLsUniform_red;
GLuint glightPosition_Uniform_red;

GLuint gLaUniform_blue, gLdUniform_blue, gLsUniform_blue;
GLuint glightPosition_Uniform_blue;

GLuint gLaUniform_green, gLdUniform_green, gLsUniform_green;
GLuint glightPosition_Uniform_green;

GLuint gKaUniform, gKdUniform, gKsUniform;
GLuint gMaterialShininessUniform;

GLuint gToggleShaderUniform;

GLfloat gAngle_Sphere;

GLfloat lightAmbient_Red[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat lightDiffuse_Red[] = { 1.0f,0.0f,0.0f,1.0f };
GLfloat lightSpecular_Red[] = { 1.0f,0.0f,0.0f,1.0f };
GLfloat lightPosition_Red[] = { 0.0f,10.0f,10.0f,0.0f };

GLfloat lightAmbient_Blue[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat lightDiffuse_Blue[] = { 0.0f,0.0f,1.0f,1.0f };
GLfloat lightSpecular_Blue[] = { 0.0f,0.0f,1.0f,1.0f };
GLfloat lightPosition_Blue[] = { 10.0f,10.0f,0.0f,0.0f };

GLfloat lightAmbient_Green[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat lightDiffuse_Green[] = { 0.0f,1.0f,0.0f,1.0f };
GLfloat lightSpecular_Green[] = { 0.0f,1.0f,0.0f,1.0f };
GLfloat lightPosition_Green[] = { 10.0f,0.0f,10.0f,0.0f };

GLfloat materialAmbient[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat materialDiffuse[] = { 1.0f,1.0f,1.0f,1.0f };
GLfloat materialSpecular[] = { 1.0f,1.0f,1.0f,1.0f };
GLfloat materialShininess = 50.0f;

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
bool gbToggleShaderFlag = false;

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

		case 0x54://T
			if (gbToggleShaderFlag == false)
				gbToggleShaderFlag = true;
			else
				gbToggleShaderFlag = false;
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
		"uniform vec4 u_light_position_red;" \
		"uniform vec4 u_light_position_blue;" \
		"uniform vec4 u_light_position_green;" \
		"uniform int u_toggle_shader;" \
		"uniform vec3 u_La_red;" \
		"uniform vec3 u_Ld_red;" \
		"uniform vec3 u_Ls_red;" \
		"uniform vec3 u_La_blue;" \
		"uniform vec3 u_Ld_blue;" \
		"uniform vec3 u_Ls_blue;" \
		"uniform vec3 u_La_green;" \
		"uniform vec3 u_Ld_green;" \
		"uniform vec3 u_Ls_green;" \
		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_material_shininess;" \
		"out vec3 transformed_normals;" \
		"out vec3 light_direction_red;" \
		"out vec3 light_direction_blue;" \
		"out vec3 light_direction_green;" \
		"out vec3 viewer_vector;" \
		"out vec3 phong_ads_color_vertex;" \
		"void main(void)" \
		"{" \
		"if(u_lighting_enabled==1)" \
		"{" \
		"vec4 eye_coordinates = u_view_matrix*u_model_matrix*vPosition;" \
		"transformed_normals = mat3(u_view_matrix*u_model_matrix)*vNormal;" \
		"light_direction_red = vec3(u_light_position_red)-eye_coordinates.xyz;" \
		"light_direction_blue = vec3(u_light_position_blue)-eye_coordinates.xyz;" \
		"light_direction_green = vec3(u_light_position_green)-eye_coordinates.xyz;" \
		"viewer_vector = -eye_coordinates.xyz;" \
		"if(u_toggle_shader == 1)" \
		"{" \
		"vec3 normalized_transformed_normals = normalize(transformed_normals);" \
		"vec3 normalized_light_direction_red = normalize(light_direction_red);" \
		"vec3 normalized_light_direction_blue = normalize(light_direction_blue);" \
		"vec3 normalized_light_direction_green = normalize(light_direction_green);" \
		"vec3 normalized_viewer_vector = normalize(viewer_vector);" \
		"vec3 ambient = u_La_red * u_Ka + u_La_blue * u_Ka + u_La_green * u_Ka;" \
		"float tn_dot_ld_red = max(dot(normalized_transformed_normals,normalized_light_direction_red),0.0);" \
		"float tn_dot_ld_blue = max(dot(normalized_transformed_normals,normalized_light_direction_blue),0.0);" \
		"float tn_dot_ld_green = max(dot(normalized_transformed_normals,normalized_light_direction_green),0.0);" \
		"vec3 diffuse = u_Ld_red * u_Kd * tn_dot_ld_red + u_Ld_blue * u_Kd * tn_dot_ld_blue + u_Ld_green * u_Kd * tn_dot_ld_green;" \
		"vec3 reflection_vector_red = reflect(-normalized_light_direction_red,normalized_transformed_normals);" \
		"vec3 reflection_vector_blue = reflect(-normalized_light_direction_blue,normalized_transformed_normals);" \
		"vec3 reflection_vector_green = reflect(-normalized_light_direction_green,normalized_transformed_normals);" \
		"vec3 specular = u_Ls_red * u_Ks * pow(max(dot(reflection_vector_red,normalized_viewer_vector),0.0),u_material_shininess) + u_Ls_blue * u_Ks * pow(max(dot(reflection_vector_blue,normalized_viewer_vector),0.0),u_material_shininess) + u_Ls_green * u_Ks * pow(max(dot(reflection_vector_green,normalized_viewer_vector),0.0),u_material_shininess);" \
		"phong_ads_color_vertex = ambient + diffuse + specular;" \
		"}" \
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
		"in vec3 light_direction_red;" \
		"in vec3 light_direction_blue;" \
		"in vec3 light_direction_green;" \
		"in vec3 viewer_vector;" \
		"in vec3 phong_ads_color_vertex;" \
		"out vec4 FragColor;" \
		"uniform vec3 u_La_red;" \
		"uniform vec3 u_Ld_red;" \
		"uniform vec3 u_Ls_red;" \
		"uniform vec3 u_La_blue;" \
		"uniform vec3 u_Ld_blue;" \
		"uniform vec3 u_Ls_blue;" \
		"uniform vec3 u_La_green;" \
		"uniform vec3 u_Ld_green;" \
		"uniform vec3 u_Ls_green;" \
		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_material_shininess;" \
		"uniform int u_lighting_enabled;" \
		"uniform int u_toggle_shader;" \
		"void main(void)" \
		"{" \
		"vec3 phong_ads_color;" \
		"if(u_lighting_enabled == 1)" \
		"{" \
		"if(u_toggle_shader == 0)" \
		"{" \
		"vec3 normalized_transformed_normals = normalize(transformed_normals);" \
		"vec3 normalized_light_direction_red = normalize(light_direction_red);" \
		"vec3 normalized_light_direction_blue = normalize(light_direction_blue);" \
		"vec3 normalized_light_direction_green = normalize(light_direction_green);" \
		"vec3 normalized_viewer_vector = normalize(viewer_vector);" \
		"float tn_dot_ld_red = max(dot(normalized_transformed_normals,normalized_light_direction_red),0.0);" \
		"float tn_dot_ld_blue = max(dot(normalized_transformed_normals,normalized_light_direction_blue),0.0);" \
		"float tn_dot_ld_green = max(dot(normalized_transformed_normals,normalized_light_direction_green),0.0);" \
		"vec3 reflection_vector_red = reflect(-normalized_light_direction_red,normalized_transformed_normals);" \
		"vec3 reflection_vector_blue = reflect(-normalized_light_direction_blue,normalized_transformed_normals);" \
		"vec3 reflection_vector_green = reflect(-normalized_light_direction_green,normalized_transformed_normals);" \
		"vec3 ambient = u_La_red * u_Ka + u_La_blue * u_Ka + u_La_green * u_Ka;" \
		"vec3 diffuse = u_Ld_red * u_Kd * tn_dot_ld_red + u_Ld_blue * u_Kd * tn_dot_ld_blue + u_Ld_green * u_Kd * tn_dot_ld_green;" \
		"vec3 specular = u_Ls_red * u_Ks * pow(max(dot(reflection_vector_red,normalized_viewer_vector),0.0),u_material_shininess) + u_Ls_blue * u_Ks * pow(max(dot(reflection_vector_blue,normalized_viewer_vector),0.0),u_material_shininess) + u_Ls_green * u_Ks * pow(max(dot(reflection_vector_green,normalized_viewer_vector),0.0),u_material_shininess);" \
		"phong_ads_color = ambient + diffuse + specular;" \
		"}" \
		"else" \
		"{" \
		"phong_ads_color = phong_ads_color_vertex;" \
		"}" \
		"}" \
		"else" \
		"{" \
		"phong_ads_color = vec3(1.0,1.0,1.0);" \
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

	gLaUniform_red = glGetUniformLocation(gShaderProgramObject, "u_La_red");
	gLdUniform_red = glGetUniformLocation(gShaderProgramObject, "u_Ld_red");
	gLsUniform_red = glGetUniformLocation(gShaderProgramObject, "u_Ls_red");

	glightPosition_Uniform_red = glGetUniformLocation(gShaderProgramObject, "u_light_position_red");

	gLaUniform_blue = glGetUniformLocation(gShaderProgramObject, "u_La_blue");
	gLdUniform_blue = glGetUniformLocation(gShaderProgramObject, "u_Ld_blue");
	gLsUniform_blue = glGetUniformLocation(gShaderProgramObject, "u_Ls_blue");

	glightPosition_Uniform_blue = glGetUniformLocation(gShaderProgramObject, "u_light_position_blue");

	gLaUniform_green = glGetUniformLocation(gShaderProgramObject, "u_La_green");
	gLdUniform_green = glGetUniformLocation(gShaderProgramObject, "u_Ld_green");
	gLsUniform_green = glGetUniformLocation(gShaderProgramObject, "u_Ls_green");

	glightPosition_Uniform_green = glGetUniformLocation(gShaderProgramObject, "u_light_position_green");

	gKaUniform = glGetUniformLocation(gShaderProgramObject, "u_Ka");
	gKdUniform = glGetUniformLocation(gShaderProgramObject, "u_Kd");
	gKsUniform = glGetUniformLocation(gShaderProgramObject, "u_Ks");

	gMaterialShininessUniform = glGetUniformLocation(gShaderProgramObject, "u_material_shininess");
	gToggleShaderUniform = glGetUniformLocation(gShaderProgramObject, "u_toggle_shader");

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

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	GLfloat X_Coord_Of_Light;
	GLfloat Y_Coord_Of_Light;

	//Use Shader Program Object
	glUseProgram(gShaderProgramObject);

	if (gbLight == true)
	{
		X_Coord_Of_Light = sin(gAngle_Sphere);
		Y_Coord_Of_Light = cos(gAngle_Sphere);

		lightPosition_Blue[0] = X_Coord_Of_Light * 2.0f;
		lightPosition_Blue[1] = Y_Coord_Of_Light * 2.0f;

		lightPosition_Red[1] = X_Coord_Of_Light * 2.0f;
		lightPosition_Red[2] = Y_Coord_Of_Light * 2.0f;

		lightPosition_Green[0] = X_Coord_Of_Light * 2.0f;
		lightPosition_Green[2] = Y_Coord_Of_Light * 2.0f;

		if (gbToggleShaderFlag == false)
			glUniform1i(gToggleShaderUniform, 1);
		else
			glUniform1i(gToggleShaderUniform, 0);

		glUniform1i(gLKeyPressedUniform, 1);

		glUniform3fv(gLaUniform_red, 1, lightAmbient_Red);
		glUniform3fv(gLdUniform_red, 1, lightDiffuse_Red);
		glUniform3fv(gLsUniform_red, 1, lightSpecular_Red);
		glUniform4fv(glightPosition_Uniform_red, 1, lightPosition_Red);

		glUniform3fv(gLaUniform_blue, 1, lightAmbient_Blue);
		glUniform3fv(gLdUniform_blue, 1, lightDiffuse_Blue);
		glUniform3fv(gLsUniform_blue, 1, lightSpecular_Blue);
		glUniform4fv(glightPosition_Uniform_blue, 1, lightPosition_Blue);

		glUniform3fv(gLaUniform_green, 1, lightAmbient_Green);
		glUniform3fv(gLdUniform_green, 1, lightDiffuse_Green);
		glUniform3fv(gLsUniform_green, 1, lightSpecular_Green);
		glUniform4fv(glightPosition_Uniform_green, 1, lightPosition_Green);

		glUniform3fv(gKaUniform, 1, materialAmbient);
		glUniform3fv(gKdUniform, 1, materialDiffuse);
		glUniform3fv(gKsUniform, 1, materialSpecular);
		glUniform1f(gMaterialShininessUniform, materialShininess);
	}
	else
	{
		glUniform1i(gLKeyPressedUniform, 0);
	}

	mat4 modelMatrix = mat4::identity();
	mat4 viewMatrix = mat4::identity();
	mat4 scaleMatrix = mat4::identity();

	modelMatrix = translate(0.0f, 0.0f, -2.0f);

	scaleMatrix = scale(0.5f, 0.5f, 0.5f);

	//modelMatrix = modelMatrix * scaleMatrix;

	glUniformMatrix4fv(gModelMatrixUniform, 1, GL_FALSE, modelMatrix);

	glUniformMatrix4fv(gViewMatrixUniform, 1, GL_FALSE, viewMatrix);

	glUniformMatrix4fv(gProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);

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