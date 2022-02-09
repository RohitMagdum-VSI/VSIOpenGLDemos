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

GLuint gVao_Geometry;
GLuint gVbo_Position_Pyramid, gVbo_Normal_Pyramid;

GLuint gModelMatrixUniform, gViewMatrixUniform, gProjectionMatrixUniform;
GLuint gLKeyPressedUniform;

GLuint gLaUniform_red, gLdUniform_red, gLsUniform_red;
GLuint glightPosition_Uniform_red;

GLuint gLaUniform_blue, gLdUniform_blue, gLsUniform_blue;
GLuint glightPosition_Uniform_blue;

GLuint gKaUniform, gKdUniform, gKsUniform;
GLuint gMaterialShininessUniform;	

GLuint gShaderToggleUniform;

GLfloat gAngle_Sphere;

GLfloat lightAmbient_Red[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat lightDiffuse_Red[] = { 1.0f,0.0f,0.0f,1.0f };
GLfloat lightSpecular_Red[] = { 1.0f,0.0f,0.0f,1.0f };
GLfloat lightPosition_Red[] = { 100.0f,100.0f,100.0f,1.0f };

GLfloat lightAmbient_Blue[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat lightDiffuse_Blue[] = { 0.0f,0.0f,1.0f,1.0f };
GLfloat lightSpecular_Blue[] = { 0.0f,0.0f,1.0f,1.0f };
GLfloat lightPosition_Blue[] = { -100.0f,100.0f,100.0f,1.0f };

GLfloat materialAmbient[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat materialDiffuse[] = { 1.0f,1.0f,1.0f,1.0f };
GLfloat materialSpecular[] = { 1.0f,1.0f,1.0f,1.0f };
GLfloat materialShininess = 50.0f;

mat4 gPerspectiveProjectionMatrix;

bool gbAnimate = false;
bool gbLight = false;
bool gbIsAKeyPressed = false;
bool gbIsLKeyPressed = false;
bool gbShaderToggleFlag = false;

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
		case VK_ESCAPE://ESC
			gbIsEscapeKeyPressed = true;
			break;
		case 0x46://F
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

		case 0x41://A
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

		case 0x4C://L
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
			if (gbShaderToggleFlag == false)
			{
				gbShaderToggleFlag = true;
			}
			else
			{
				gbShaderToggleFlag = false;
			}
			break;
/*
		case 0x43://C
			if (gbIsCKeyPressed == false)
			{
				gbIsCKeyPressed = true;
				gbIsPKeyPressed = false;
				gbIsSKeyPressed = false;
			}
			else
				gbIsCKeyPressed = false;
			break;

		case 0x50://P
			if (gbIsPKeyPressed == false)
			{
				gbIsPKeyPressed = true;
				gbIsCKeyPressed = false;
				gbIsSKeyPressed = false;
			}
			else
				gbIsPKeyPressed = false;
			break;

		case 0x53://S
			if (gbIsSKeyPressed == false)
			{
				gbIsSKeyPressed = true;
				gbIsPKeyPressed = false;
				gbIsCKeyPressed = false;
			}
			else
				gbIsSKeyPressed = false;
			break;*/
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

	fprintf(gpFile, "OpenGL Version : %s \n", (const char*)glGetString(GL_VERSION));
	fprintf(gpFile, "OpenGL Shading Language Version : %s \n", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

	//Vertex Shader
	gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *vertexShaderSourceCode =
		"#version 460" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vNormal;" \
		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"uniform int u_lighting_enabled;" \
		"uniform vec4 u_light_position_red;" \
		"uniform vec4 u_light_position_blue;" \
		"uniform int u_toggle_shader;" \
		"uniform vec3 u_La_red;" \
		"uniform vec3 u_Ld_red;" \
		"uniform vec3 u_Ls_red;" \
		"uniform vec3 u_La_blue;" \
		"uniform vec3 u_Ld_blue;" \
		"uniform vec3 u_Ls_blue;" \
		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_material_shininess;" \
		"out vec3 transformed_normals;" \
		"out vec3 light_direction_red;" \
		"out vec3 light_direction_blue;" \
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
		"viewer_vector = -eye_coordinates.xyz;" \
		"if(u_toggle_shader == 1)" \
		"{" \
		"vec3 normalized_transformed_normals = normalize(transformed_normals);" \
		"vec3 normalized_light_direction_red = normalize(light_direction_red);" \
		"vec3 normalized_light_direction_blue = normalize(light_direction_blue);" \
		"vec3 normalized_viewer_vector = normalize(viewer_vector);" \
		"vec3 ambient = u_La_red * u_Ka + u_La_blue * u_Ka;" \
		"float tn_dot_ld_red = max(dot(normalized_transformed_normals,normalized_light_direction_red),0.0);" \
		"float tn_dot_ld_blue = max(dot(normalized_transformed_normals,normalized_light_direction_blue),0.0);" \
		"vec3 diffuse = u_Ld_red * u_Kd * tn_dot_ld_red + u_Ld_blue * u_Kd * tn_dot_ld_blue;" \
		"vec3 reflection_vector_red = reflect(-normalized_light_direction_red,normalized_transformed_normals);" \
		"vec3 reflection_vector_blue = reflect(-normalized_light_direction_blue,normalized_transformed_normals);" \
		"vec3 specular = u_Ls_red * u_Ks * pow(max(dot(reflection_vector_red,normalized_viewer_vector),0.0),u_material_shininess) + u_Ls_blue * u_Ks * pow(max(dot(reflection_vector_blue,normalized_viewer_vector),0.0),u_material_shininess);" \
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
		"#version 460" \
		"\n" \
		"in vec3 transformed_normals;" \
		"in vec3 light_direction_red;" \
		"in vec3 light_direction_blue;" \
		"in vec3 viewer_vector;" \
		"in vec3 phong_ads_color_vertex;" \
		"out vec4 FragColor;" \
		"uniform vec3 u_La_red;" \
		"uniform vec3 u_Ld_red;" \
		"uniform vec3 u_Ls_red;" \
		"uniform vec3 u_La_blue;" \
		"uniform vec3 u_Ld_blue;" \
		"uniform vec3 u_Ls_blue;" \
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
		"vec3 normalized_viewer_vector = normalize(viewer_vector);" \
		"float tn_dot_ld_red = max(dot(normalized_transformed_normals,normalized_light_direction_red),0.0);" \
		"vec3 reflection_vector_red = reflect(-normalized_light_direction_red,normalized_transformed_normals);" \
		"vec3 ambient = u_La_red * u_Ka + u_La_blue * u_Ka;" \
		"float tn_dot_ld_blue = max(dot(normalized_transformed_normals,normalized_light_direction_blue),0.0);" \
		"vec3 diffuse = u_Ld_red * u_Kd * tn_dot_ld_red + u_Ld_blue * u_Kd * tn_dot_ld_blue;" \
		"vec3 reflection_vector_blue = reflect(-normalized_light_direction_blue,normalized_transformed_normals);" \
		"vec3 specular = u_Ls_red * u_Ks * pow(max(dot(reflection_vector_red,normalized_viewer_vector),0.0),u_material_shininess) + u_Ls_blue * u_Ks * pow(max(dot(reflection_vector_blue,normalized_viewer_vector),0.0),u_material_shininess);" \
		"phong_ads_color = ambient + diffuse + specular;" \
		"}" \
		"else" \
		"{"\
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

	GLfloat pyramidVertices[] =
	{
		0.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,

		0.0f, 1.0f, 0.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f,

		0.0f,1.0f,0.0f,
		1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,-1.0f,

		0.0f,1.0f,0.0f,
		-1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,1.0f
	};

	GLfloat pyramidNormal[] =
	{
		0.0f, 0.447214f, 0.894428f,
		0.0f, 0.447214f, 0.894428f,
		0.0f, 0.447214f, 0.894428f,

		0.894428f, 0.447214f, 0.0f,
		0.894428f, 0.447214f, 0.0f,
		0.894428f, 0.447214f, 0.0f,

		0.0f, 0.447214f, -0.894428f,
		0.0f, 0.447214f, -0.894428f,
		0.0f, 0.447214f, -0.894428f,

		-0.894428f, 0.447214f, 0.0f,
		-0.894428f, 0.447214f, 0.0f,
		-0.894428f, 0.447214f, 0.0f
	};

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

	gKaUniform = glGetUniformLocation(gShaderProgramObject, "u_Ka");
	gKdUniform = glGetUniformLocation(gShaderProgramObject, "u_Kd");
	gKsUniform = glGetUniformLocation(gShaderProgramObject, "u_Ks");

	gMaterialShininessUniform = glGetUniformLocation(gShaderProgramObject, "u_material_shininess");

	gShaderToggleUniform = glGetUniformLocation(gShaderProgramObject, "u_toggle_shader");

	/*****************VAO For Sphere*****************/
	glGenVertexArrays(1, &gVao_Geometry);
	glBindVertexArray(gVao_Geometry);

	/*****************Pyramid Position****************/
	glGenBuffers(1, &gVbo_Position_Pyramid);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Position_Pyramid);
	glBufferData(GL_ARRAY_BUFFER, sizeof(pyramidVertices), pyramidVertices, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/*****************Pyramid Lights****************/
	glGenBuffers(1, &gVbo_Normal_Pyramid);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Normal_Pyramid);
	glBufferData(GL_ARRAY_BUFFER, sizeof(pyramidNormal), pyramidNormal, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

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
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	
	//Use Shader Program Object
	glUseProgram(gShaderProgramObject);

	if (gbLight == true)
	{
		if (gbShaderToggleFlag == true)
		{
			glUniform1i(gShaderToggleUniform, 0);
		}
		else
		{
			glUniform1i(gShaderToggleUniform, 1);
		}

		glUniform1i(gLKeyPressedUniform, 1);

		glUniform3fv(gLaUniform_red, 1, lightAmbient_Red);
		glUniform3fv(gLdUniform_red, 1, lightDiffuse_Red);
		glUniform3fv(gLsUniform_red, 1, lightSpecular_Red);
		glUniform4fv(glightPosition_Uniform_red, 1, lightPosition_Red);

		glUniform3fv(gLaUniform_blue, 1, lightAmbient_Blue);
		glUniform3fv(gLdUniform_blue, 1, lightDiffuse_Blue);
		glUniform3fv(gLsUniform_blue, 1, lightSpecular_Blue);
		glUniform4fv(glightPosition_Uniform_blue, 1, lightPosition_Blue);

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
	mat4 rotationMatrix = mat4::identity();

	modelMatrix = translate(0.0f, 0.0f, -5.0f);

	rotationMatrix = rotate(gAngle_Sphere, 0.0f, 1.0f, 0.0f);

	modelMatrix = modelMatrix * rotationMatrix;

	glUniformMatrix4fv(gModelMatrixUniform, 1, GL_FALSE, modelMatrix);

	glUniformMatrix4fv(gViewMatrixUniform, 1, GL_FALSE, viewMatrix);

	glUniformMatrix4fv(gProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	glBindVertexArray(gVao_Geometry);

	glDrawArrays(GL_TRIANGLES, 0, 12);

	glBindVertexArray(0);

	glUseProgram(0);

	SwapBuffers(ghdc);
}

void update(void)
{
	gAngle_Sphere = gAngle_Sphere + 0.1f;
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

	if (gVao_Geometry)
	{
		glDeleteVertexArrays(1, &gVao_Geometry);
		gVao_Geometry = 0;
	}

	if (gVbo_Normal_Pyramid)
	{
		glDeleteBuffers(1, &gVbo_Normal_Pyramid);
		gVbo_Normal_Pyramid = 0;
	}

	if (gVbo_Position_Pyramid)
	{
		glDeleteBuffers(1, &gVbo_Position_Pyramid);
		gVbo_Position_Pyramid = 0;
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