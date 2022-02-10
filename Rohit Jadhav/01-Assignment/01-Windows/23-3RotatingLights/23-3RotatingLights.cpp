#include<Windows.h>
#include<stdio.h>
#include<math.h>
#include<GL/glew.h>
#include<GL/gl.h>
#include"vmath.h"
#include"Sphere.h"

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "Sphere.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

using namespace vmath;

enum {
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOORD0
};

//For Fullscreen
bool bIsFullScreen_RRJ = false;
HWND ghwnd_RRJ = NULL;
WINDOWPLACEMENT wpPrev_RRJ;
DWORD dwStyle_RRJ;

//For SuperMan
bool bActiveWindow_RRJ = false;
HDC ghdc_RRJ = NULL;
HGLRC ghrc_RRJ = NULL;

//For Error
FILE *gbFile_RRJ = NULL;

//For Shader
GLuint shaderProgramObject_PV_RRJ;
GLuint shaderProgramObject_PF_RRJ;

//For Projection
mat4 perspectiveProjectionMatrix_RRJ;

//For Uniform
GLuint modelMatrixUniform_RRJ;
GLuint viewMatrixUniform_RRJ;
GLuint projectionMatrixUniform_RRJ;


GLuint red_LaUniform_RRJ;
GLuint red_LdUniform_RRJ;
GLuint red_LsUniform_RRJ;
GLuint red_lightPositionUniform_RRJ;

GLuint green_LaUniform_RRJ;
GLuint green_LdUniform_RRJ;
GLuint green_LsUniform_RRJ;
GLuint green_lightPositionUniform_RRJ;

GLuint blue_LaUniform_RRJ;
GLuint blue_LdUniform_RRJ;
GLuint blue_LsUniform_RRJ;
GLuint blue_lightPositionUniform_RRJ;


GLuint KaUniform_RRJ;
GLuint KdUniform_RRJ;
GLuint KsUniform_RRJ;
GLuint shininessUniform_RRJ;
GLuint LKeyPressUniform_RRJ;


//For Light
int iWhichLight_RRJ = 1;
bool bLight_RRJ = false;

GLfloat lightAmbient_Red_RRJ[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat lightDiffuse_Red_RRJ[] = { 1.0f, 0.0f, 0.0f, 1.0f };
GLfloat lightSpecular_Red_RRJ[] = { 1.0f, 0.0f, 0.0f, 1.0f };
GLfloat lightPosition_Red_RRJ[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat angle_red_RRJ = 0.0f;

GLfloat lightAmbient_Green_RRJ[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat lightDiffuse_Green_RRJ[] = { 0.0f, 1.0f, 0.0f, 1.0f };
GLfloat lightSpecular_Green_RRJ[] = { 0.0f, 1.0f, 0.0f, 1.0f };
GLfloat lightPosition_Green_RRJ[] = { 0.0, 0.0f, 0.0f, 1.0f };
GLfloat angle_green_RRJ = 0.0f;

GLfloat lightAmbient_Blue_RRJ[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat lightDiffuse_Blue_RRJ[] = { 0.0f, 0.0f, 1.0f, 1.0f };
GLfloat lightSpecular_Blue_RRJ[] = { 0.0f, 0.0f, 1.0f, 1.0f };
GLfloat lightPosition_Blue_RRJ[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat angle_blue_RRJ = 0.0f;


//For Material
GLfloat materialAmbient_RRJ[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat materialDiffuse_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialSpecular_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialShininess_RRJ = 128.0f;

//For Sphere
GLuint vao_Sphere_RRJ;
GLuint vbo_Sphere_Position_RRJ;
GLuint vbo_Sphere_Normal_RRJ;
GLuint vbo_Sphere_Element_RRJ;
float sphere_vertices_RRJ[1146];
float sphere_normals_RRJ[1146];
float sphere_textures_RRJ[764];
unsigned short sphere_elements_RRJ[2280];
unsigned int gNumVertices_RRJ;
unsigned int gNumElements_RRJ;



LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR szCmdLine, int iCmdShow) {

	int initialize(void);
	void display(void);
	void ToggleFullScreen(void);
	void uninitialize(void);
	void update(void);

	bool bDone_RRJ = false;
	int iRet_RRJ;

	if (fopen_s(&gbFile_RRJ, "Log.txt", "w") != 0) {
		MessageBox(NULL, TEXT("Error Creating Log File"), TEXT("Error"), MB_OK);
		exit(1);
	}
	else
		fprintf(gbFile_RRJ, "Log Created!!\n");


	WNDCLASSEX wndclass_RRJ;
	HWND hwnd_RRJ;
	MSG msg_RRJ;
	TCHAR szName_RRJ[] = TEXT("RohitRJadhav-PP-3Lights");

	wndclass_RRJ.lpszClassName = szName_RRJ;
	wndclass_RRJ.lpszMenuName = NULL;
	wndclass_RRJ.lpfnWndProc = WndProc;

	wndclass_RRJ.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass_RRJ.cbSize = sizeof(WNDCLASSEX);
	wndclass_RRJ.cbClsExtra = 0;
	wndclass_RRJ.cbWndExtra = 0;

	wndclass_RRJ.hInstance = hInstance;
	wndclass_RRJ.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass_RRJ.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass_RRJ.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass_RRJ.hCursor = LoadCursor(NULL, IDC_ARROW);

	RegisterClassEx(&wndclass_RRJ);

	hwnd_RRJ = CreateWindowEx(WS_EX_APPWINDOW,
		szName_RRJ,
		TEXT("RohitRJadhav-PP-3Lights"),
		WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_VISIBLE,
		100, 100,
		WIN_WIDTH, WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghwnd_RRJ = hwnd_RRJ;

	SetForegroundWindow(hwnd_RRJ);
	SetFocus(hwnd_RRJ);

	iRet_RRJ = initialize();
	if (iRet_RRJ == 1) {
		fprintf(gbFile_RRJ, "ChoosePixelFormat() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else if (iRet_RRJ == 2) {
		fprintf(gbFile_RRJ, "SetPixelFormat() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else if (iRet_RRJ == 3) {
		fprintf(gbFile_RRJ, "wglCreateContext() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else if (iRet_RRJ == 4) {
		fprintf(gbFile_RRJ, "wglMakeCurrent() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else
		fprintf(gbFile_RRJ, "Initialize Done!!\n");




	ShowWindow(hwnd_RRJ, iCmdShow);
	ToggleFullScreen();


	while (bDone_RRJ == false) {
		if (PeekMessage(&msg_RRJ, NULL, 0, 0, PM_REMOVE)) {
			if (msg_RRJ.message == WM_QUIT)
				bDone_RRJ = true;
			else {
				TranslateMessage(&msg_RRJ);
				DispatchMessage(&msg_RRJ);
			}
		}
		else {
			if (bActiveWindow_RRJ == true) {
				update();
			}
			display();
		}
	}

	return((int)msg_RRJ.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam) {

	void resize(int, int);
	void uninitialize(void);
	void ToggleFullScreen(void);

	switch (iMsg) {
	case WM_SETFOCUS:
		bActiveWindow_RRJ = true;
		break;

	case WM_KILLFOCUS:
		bActiveWindow_RRJ = false;
		break;

	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_CHAR:
		switch (wParam) {
		case VK_ESCAPE:
			ToggleFullScreen();
			break;


		case 'q':
		case 'Q':
			DestroyWindow(hwnd);
			break;

		case 'F':
		case 'f':
			iWhichLight_RRJ = 2;
			break;

		case 'V':
		case 'v':
			iWhichLight_RRJ = 1;
			break;

		case 'L':
		case 'l':
			if (bLight_RRJ == false)
				bLight_RRJ = true;
			else
				bLight_RRJ = false;
			break;
		}
		break;


	case WM_ERASEBKGND:
		return(0);

	case WM_DESTROY:
		uninitialize();
		PostQuitMessage(0);
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullScreen(void) {

	MONITORINFO mi_RRJ;

	if (bIsFullScreen_RRJ == false) {
		dwStyle_RRJ = GetWindowLong(ghwnd_RRJ, GWL_STYLE);
		mi_RRJ = { sizeof(MONITORINFO) };
		if (dwStyle_RRJ & WS_OVERLAPPEDWINDOW) {
			if (GetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ) && GetMonitorInfo(MonitorFromWindow(ghwnd_RRJ, MONITORINFOF_PRIMARY), &mi_RRJ)) {
				SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd_RRJ,
					HWND_TOP,
					mi_RRJ.rcMonitor.left,
					mi_RRJ.rcMonitor.top,
					(mi_RRJ.rcMonitor.right - mi_RRJ.rcMonitor.left),
					(mi_RRJ.rcMonitor.bottom - mi_RRJ.rcMonitor.top),
					SWP_NOZORDER | SWP_FRAMECHANGED);
				ShowCursor(FALSE);
				bIsFullScreen_RRJ = true;
			}
		}

	}
	else {
		SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ);
		SetWindowPos(ghwnd_RRJ,
			HWND_TOP,
			0, 0, 0, 0,
			SWP_NOSIZE | SWP_NOMOVE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
		bIsFullScreen_RRJ = false;
	}
}

int initialize(void) {

	void resize(int, int);
	void uninitialize(void);

	GLuint vertexShaderObject_PV_RRJ;
	GLuint fragmentShaderObject_PV_RRJ;

	GLuint vertexShaderObject_PF_RRJ;
	GLuint fragmentShaderObject_PF_RRJ;

	GLenum result_RRJ;
	PIXELFORMATDESCRIPTOR pfd_RRJ;
	int iPixelFormatIndex_RRJ;

	ghdc_RRJ = GetDC(ghwnd_RRJ);

	memset(&pfd_RRJ, 0, sizeof(PIXELFORMATDESCRIPTOR));

	pfd_RRJ.nVersion = 1;
	pfd_RRJ.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd_RRJ.dwFlags = PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER;
	pfd_RRJ.iPixelType = PFD_TYPE_RGBA;

	pfd_RRJ.cColorBits = 32;
	pfd_RRJ.cRedBits = 8;
	pfd_RRJ.cGreenBits = 8;
	pfd_RRJ.cBlueBits = 8;
	pfd_RRJ.cAlphaBits = 8;
	pfd_RRJ.cDepthBits = 24;

	iPixelFormatIndex_RRJ = ChoosePixelFormat(ghdc_RRJ, &pfd_RRJ);
	if (iPixelFormatIndex_RRJ == 0)
		return(1);

	if (SetPixelFormat(ghdc_RRJ, iPixelFormatIndex_RRJ, &pfd_RRJ) == GL_FALSE)
		return(2);

	ghrc_RRJ = wglCreateContext(ghdc_RRJ);
	if (ghrc_RRJ == NULL)
		return(3);

	if (wglMakeCurrent(ghdc_RRJ, ghrc_RRJ) == GL_FALSE)
		return(4);


	result_RRJ = glewInit();
	if (result_RRJ != GLEW_OK) {
		fprintf(gbFile_RRJ, "glewInit() Failed!!\n");
		DestroyWindow(ghwnd_RRJ);
	}



	/********** Vertex Shader Per Vertex *********/
	vertexShaderObject_PV_RRJ = glCreateShader(GL_VERTEX_SHADER);
	const char *szVertexShaderCode_PV_RRJ =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vNormals;" \


		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \

		"uniform vec3 u_Red_La;" \
		"uniform vec3 u_Red_Ld;" \
		"uniform vec3 u_Red_Ls;" \
		"uniform vec4 u_Red_light_position;" \

		"uniform vec3 u_Green_La;" \
		"uniform vec3 u_Green_Ld;" \
		"uniform vec3 u_Green_Ls;" \
		"uniform vec4 u_Green_light_position;" \

		"uniform vec3 u_Blue_La;" \
		"uniform vec3 u_Blue_Ld;" \
		"uniform vec3 u_Blue_Ls;" \
		"uniform vec4 u_Blue_light_position;" \

		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_shininess;" \

		"uniform int u_L_keypress;" \

		"out vec3 phongLight;"

		"void main(void)" \
		"{" \
		"if(u_L_keypress == 1){" \
		"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" \

		"vec3 RedSource = normalize(vec3(u_Red_light_position - eyeCoordinate));" \
		"vec3 GreenSource = normalize(vec3(u_Green_light_position - eyeCoordinate));" \
		"vec3 BlueSource = normalize(vec3(u_Blue_light_position - eyeCoordinate));" \

		"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" \
		"vec3 Normal = normalize(normalMatrix * vNormals);" \

		"float SRed_Dot_N = max(dot(RedSource, Normal), 0.0);" \
		"float SGreen_Dot_N = max(dot(GreenSource, Normal), 0.0);" \
		"float SBlue_Dot_N = max(dot(BlueSource, Normal), 0.0);" \

		"vec3 RedReflection = reflect(-RedSource, Normal);" \
		"vec3 GreenReflection = reflect(-GreenSource, Normal);" \
		"vec3 BlueReflection = reflect(-BlueSource, Normal);" \

		"vec3 Viewer = normalize(vec3(-eyeCoordinate.xyz));" \


		"float RRed_Dot_V = max(dot(RedReflection, Viewer), 0.0);" \
		"vec3 ambientRed = u_Red_La * u_Ka;" \
		"vec3 diffuseRed = u_Red_Ld * u_Kd * SRed_Dot_N;" \
		"vec3 specularRed = u_Red_Ls * u_Ks * pow(RRed_Dot_V, u_shininess);" \
		"vec3 Red = ambientRed + diffuseRed + specularRed;" \


		"float RGreen_Dot_V = max(dot(GreenReflection, Viewer), 0.0);" \
		"vec3 ambientGreen = u_Green_La * u_Ka;" \
		"vec3 diffuseGreen = u_Green_Ld * u_Kd * SGreen_Dot_N;" \
		"vec3 specularGreen = u_Green_Ls * u_Ks * pow(RGreen_Dot_V, u_shininess);" \
		"vec3 Green = ambientGreen + diffuseGreen + specularGreen;" \


		"float RBlue_Dot_V = max(dot(BlueReflection, Viewer), 0.0);" \
		"vec3 ambientBlue = u_Blue_La * u_Ka;" \
		"vec3 diffuseBlue = u_Blue_Ld * u_Kd * SBlue_Dot_N;" \
		"vec3 specularBlue = u_Blue_Ls * u_Ks * pow(RBlue_Dot_V, u_shininess);" \
		"vec3 Blue = ambientBlue + diffuseBlue + specularBlue;" \

		"phongLight = Red + Green + Blue;" \

		"}" \
		"else{" \
			"phongLight = vec3(1.0, 1.0, 1.0);" \
		"}" \
		"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;"
		"}";

	glShaderSource(vertexShaderObject_PV_RRJ, 1, (const GLchar**)&szVertexShaderCode_PV_RRJ, NULL);

	glCompileShader(vertexShaderObject_PV_RRJ);

	GLint iShaderCompileStatus_RRJ;
	GLint iInfoLogLength_RRJ;
	GLchar *szInfoLog_RRJ = NULL;
	glGetShaderiv(vertexShaderObject_PV_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(vertexShaderObject_PV_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(vertexShaderObject_PV_RRJ, iInfoLogLength_RRJ,
					&written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Per Vertex Lighting Vertex Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}


	/********** Fragment Shader Per Vertex *********/
	fragmentShaderObject_PV_RRJ = glCreateShader(GL_FRAGMENT_SHADER);
	const char* szFragmentShaderCode_PV_RRJ =
		"#version 450 core" \
		"\n" \
		"in vec3 phongLight;" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
		"FragColor = vec4(phongLight, 1.0);" \
		"}";


	glShaderSource(fragmentShaderObject_PV_RRJ, 1,
		(const GLchar**)&szFragmentShaderCode_PV_RRJ, NULL);

	glCompileShader(fragmentShaderObject_PV_RRJ);

	iShaderCompileStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(fragmentShaderObject_PV_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(fragmentShaderObject_PV_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject_PV_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Per Vertex Lighting Fragment Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}


	/********* Program Object For Per Vertex Lighting **********/
	shaderProgramObject_PV_RRJ = glCreateProgram();

	glAttachShader(shaderProgramObject_PV_RRJ, vertexShaderObject_PV_RRJ);
	glAttachShader(shaderProgramObject_PV_RRJ, fragmentShaderObject_PV_RRJ);

	glBindAttribLocation(shaderProgramObject_PV_RRJ, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(shaderProgramObject_PV_RRJ, AMC_ATTRIBUTE_NORMAL, "vNormals");

	glLinkProgram(shaderProgramObject_PV_RRJ);

	GLint iProgramLinkingStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;
	glGetProgramiv(shaderProgramObject_PV_RRJ, GL_LINK_STATUS, &iProgramLinkingStatus_RRJ);
	if (iProgramLinkingStatus_RRJ == GL_FALSE) {
		glGetProgramiv(shaderProgramObject_PV_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetProgramInfoLog(shaderProgramObject_PV_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Shader Program Object Linking Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}

	modelMatrixUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_model_matrix");
	viewMatrixUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_view_matrix");
	projectionMatrixUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_projection_matrix");
	
	red_LaUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Red_La");
	red_LdUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Red_Ld");
	red_LsUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Red_Ls");
	red_lightPositionUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Red_light_position");

	green_LaUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Green_La");
	green_LdUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Green_Ld");
	green_LsUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Green_Ls");
	green_lightPositionUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Green_light_position");

	blue_LaUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Blue_La");
	blue_LdUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Blue_Ld");
	blue_LsUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Blue_Ls");
	blue_lightPositionUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Blue_light_position");


	KaUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ka");
	KdUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Kd");
	KsUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ks");
	shininessUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_shininess");
	LKeyPressUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_L_keypress");








	/********** Vertex Shader Per Fragment Lighting *********/
	vertexShaderObject_PF_RRJ = glCreateShader(GL_VERTEX_SHADER);
	const char *szVertexShaderCode_PF_RRJ =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vNormals;" \

		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \

		"uniform vec4 u_Red_light_position;" \
		"uniform vec4 u_Green_light_position;" \
		"uniform vec4 u_Blue_light_position;" \

		"out vec3 lightDirectionRed_VS;" \
		"out vec3 lightDirectionGreen_VS;" \
		"out vec3 lightDirectionBlue_VS;" \

		"out vec3 Normal_VS;" \
		"out vec3 Viewer_VS;" \

		"void main(void)" \
		"{"
		"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" \
		"lightDirectionRed_VS = vec3(u_Red_light_position - eyeCoordinate);" \
		"lightDirectionGreen_VS = vec3(u_Green_light_position - eyeCoordinate);" \
		"lightDirectionBlue_VS = vec3(u_Blue_light_position - eyeCoordinate);" \
		
		"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" \
		"Normal_VS = vec3(normalMatrix * vNormals);" \
		"Viewer_VS = vec3(-eyeCoordinate);" \
		"gl_Position =	u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
		"}";


	glShaderSource(vertexShaderObject_PF_RRJ, 1, (const GLchar**)&szVertexShaderCode_PF_RRJ, NULL);

	glCompileShader(vertexShaderObject_PF_RRJ);


	glGetShaderiv(vertexShaderObject_PF_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(vertexShaderObject_PF_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(vertexShaderObject_PF_RRJ, iInfoLogLength_RRJ,
					&written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Per Fragment Lighting Vertex Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}


	/********** Fragment Shader Per Fragment *********/
	fragmentShaderObject_PF_RRJ = glCreateShader(GL_FRAGMENT_SHADER);
	const char* szFragmentShaderCode_PF_RRJ =
		"#version 450 core" \
		"\n" \

		"in vec3 lightDirectionRed_VS;" \
		"in vec3 lightDirectionGreen_VS;" \
		"in vec3 lightDirectionBlue_VS;" \

		"in vec3 Normal_VS;" \
		"in vec3 Viewer_VS;" \

		"uniform vec3 u_Red_La;" \
		"uniform vec3 u_Red_Ld;" \
		"uniform vec3 u_Red_Ls;" \

		"uniform vec3 u_Green_La;" \
		"uniform vec3 u_Green_Ld;" \
		"uniform vec3 u_Green_Ls;" \

		"uniform vec3 u_Blue_La;" \
		"uniform vec3 u_Blue_Ld;" \
		"uniform vec3 u_Blue_Ls;" \


		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_shininess;" \
		"uniform int u_L_keypress;" \


		"out vec4 FragColor;" \


		"void main(void)" \
		"{" \
		"vec3 phongLight;" \
		"if(u_L_keypress == 1){" \
		"vec3 RedLightDirection = normalize(lightDirectionRed_VS);" \
		"vec3 GreenLightDirection = normalize(lightDirectionGreen_VS);" \
		"vec3 BlueLightDirection = normalize(lightDirectionBlue_VS);" \
		
		
		"vec3 Normal = normalize(Normal_VS);" \

		"float LRed_Dot_N = max(dot(RedLightDirection, Normal), 0.0);" \
		"float LGreen_Dot_N = max(dot(GreenLightDirection, Normal), 0.0);" \
		"float LBlue_Dot_N = max(dot(BlueLightDirection, Normal), 0.0);" \
		
		
		"vec3 RedReflection = reflect(-RedLightDirection, Normal);" \
		"vec3 GreenReflection = reflect(-GreenLightDirection, Normal);" \
		"vec3 BlueReflection = reflect(-BlueLightDirection, Normal);" \
		
		
		"vec3 Viewer = normalize(Viewer_VS);" \


		"float RRed_Dot_V = max(dot(RedReflection, Viewer), 0.0);" \
		"float RGreen_Dot_V = max(dot(GreenReflection, Viewer), 0.0);" \
		"float RBlue_Dot_V = max(dot(BlueReflection, Viewer), 0.0);" \



		"vec3 ambientRed = u_Red_La * u_Ka;" \
		"vec3 diffuseRed = u_Red_Ld * u_Kd * LRed_Dot_N;" \
		"vec3 specularRed = u_Red_Ls * u_Ks * pow(RRed_Dot_V, u_shininess);" \
		"vec3 Red = ambientRed + diffuseRed + specularRed;" \


		"vec3 ambientGreen = u_Green_La * u_Ka;" \
		"vec3 diffuseGreen = u_Green_Ld * u_Kd * LGreen_Dot_N;" \
		"vec3 specularGreen = u_Green_Ls * u_Ks * pow(RGreen_Dot_V, u_shininess);" \
		"vec3 Green = ambientGreen + diffuseGreen + specularGreen;" \

		"vec3 ambientBlue = u_Blue_La * u_Ka;" \
		"vec3 diffuseBlue = u_Blue_Ld * u_Kd * LBlue_Dot_N;" \
		"vec3 specularBlue = u_Blue_Ls * u_Ks * pow(RBlue_Dot_V, u_shininess);" \
		"vec3 Blue = ambientBlue + diffuseBlue + specularBlue;" \

		"phongLight = Red + Green + Blue;" \

		"}" \
		"else{" \
		"phongLight = vec3(1.0, 1.0, 1.0);" \
		"}" \
		"FragColor = vec4(phongLight, 1.0);" \
		"}";


	glShaderSource(fragmentShaderObject_PF_RRJ, 1,
		(const GLchar**)&szFragmentShaderCode_PF_RRJ, NULL);

	glCompileShader(fragmentShaderObject_PF_RRJ);

	iShaderCompileStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(fragmentShaderObject_PF_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(fragmentShaderObject_PF_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject_PF_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Per Fragment Lighting Fragment Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}


	/********* Program Object For Per Fragment Lighting **********/
	shaderProgramObject_PF_RRJ = glCreateProgram();

	glAttachShader(shaderProgramObject_PF_RRJ, vertexShaderObject_PF_RRJ);
	glAttachShader(shaderProgramObject_PF_RRJ, fragmentShaderObject_PF_RRJ);

	glBindAttribLocation(shaderProgramObject_PF_RRJ, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(shaderProgramObject_PF_RRJ, AMC_ATTRIBUTE_NORMAL, "vNormals");

	glLinkProgram(shaderProgramObject_PF_RRJ);


	glGetProgramiv(shaderProgramObject_PF_RRJ, GL_LINK_STATUS, &iProgramLinkingStatus_RRJ);
	if (iProgramLinkingStatus_RRJ == GL_FALSE) {
		glGetProgramiv(shaderProgramObject_PF_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetProgramInfoLog(shaderProgramObject_PF_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Shader Program Object Linking Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}

	modelMatrixUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_model_matrix");
	viewMatrixUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_view_matrix");
	projectionMatrixUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_projection_matrix");
	
	red_LaUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Red_La");
	red_LdUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Red_Ld");
	red_LsUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Red_Ls");
	red_lightPositionUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Red_light_position");

	green_LaUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Green_La");
	green_LdUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Green_Ld");
	green_LsUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Green_Ls");
	green_lightPositionUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Green_light_position");

	blue_LaUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Blue_La");
	blue_LdUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Blue_Ld");
	blue_LsUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Blue_Ls");
	blue_lightPositionUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Blue_light_position");

	KaUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Ka");
	KdUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Kd");
	KsUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Ks");
	shininessUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_shininess");
	LKeyPressUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_L_keypress");






	/********** Position, Normal and Elements **********/
	getSphereVertexData(sphere_vertices_RRJ, sphere_normals_RRJ, sphere_textures_RRJ, sphere_elements_RRJ);
	gNumVertices_RRJ = getNumberOfSphereVertices();
	gNumElements_RRJ = getNumberOfSphereElements();



	/********** Sphere Vao **********/
	glGenVertexArrays(1, &vao_Sphere_RRJ);
	glBindVertexArray(vao_Sphere_RRJ);

	/********** Position **********/
	glGenBuffers(1, &vbo_Sphere_Position_RRJ);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Sphere_Position_RRJ);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(sphere_vertices_RRJ),
		sphere_vertices_RRJ,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/********** Normals **********/
	glGenBuffers(1, &vbo_Sphere_Normal_RRJ);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Sphere_Normal_RRJ);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(sphere_normals_RRJ),
		sphere_normals_RRJ,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	/********** Element Vbo **********/
	glGenBuffers(1, &vbo_Sphere_Element_RRJ);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(sphere_elements_RRJ), sphere_elements_RRJ, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


	glBindVertexArray(0);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	perspectiveProjectionMatrix_RRJ = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}



void uninitialize(void) {

	if (bIsFullScreen_RRJ == true) {
		SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ);
		SetWindowPos(ghwnd_RRJ,
			HWND_TOP,
			0, 0, 0, 0,
			SWP_NOZORDER | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
		bIsFullScreen_RRJ = false;
	}

	if (vbo_Sphere_Element_RRJ) {
		glDeleteBuffers(1, &vbo_Sphere_Element_RRJ);
		vbo_Sphere_Element_RRJ = 0;
	}

	if (vbo_Sphere_Normal_RRJ) {
		glDeleteBuffers(1, &vbo_Sphere_Normal_RRJ);
		vbo_Sphere_Normal_RRJ = 0;
	}

	if (vbo_Sphere_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Sphere_Position_RRJ);
		vbo_Sphere_Position_RRJ = 0;
	}

	if (vao_Sphere_RRJ) {
		glDeleteVertexArrays(1, &vao_Sphere_RRJ);
		vao_Sphere_RRJ = 0;
	}


	GLsizei ShaderCount;
	GLsizei ShaderNumber;

	if (shaderProgramObject_PV_RRJ) {
		glUseProgram(shaderProgramObject_PV_RRJ);

		glGetProgramiv(shaderProgramObject_PV_RRJ, GL_ATTACHED_SHADERS, &ShaderCount);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount);
		if (pShader) {
			glGetAttachedShaders(shaderProgramObject_PV_RRJ, ShaderCount, &ShaderCount, pShader);
			for (ShaderNumber = 0; ShaderNumber < ShaderCount; ShaderNumber++) {
				glDetachShader(shaderProgramObject_PV_RRJ, pShader[ShaderNumber]);
				glDeleteShader(pShader[ShaderNumber]);
				pShader[ShaderNumber] = 0;
			}
			free(pShader);
			pShader = NULL;
		}
		glDeleteProgram(shaderProgramObject_PV_RRJ);
		shaderProgramObject_PV_RRJ = 0;
		glUseProgram(0);
	}


	ShaderCount = 0;
	ShaderNumber = 0;
	if (shaderProgramObject_PF_RRJ) {
		glUseProgram(shaderProgramObject_PF_RRJ);

		glGetProgramiv(shaderProgramObject_PF_RRJ, GL_ATTACHED_SHADERS, &ShaderCount);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount);
		if (pShader) {
			glGetAttachedShaders(shaderProgramObject_PF_RRJ, ShaderCount, &ShaderCount, pShader);
			for (ShaderNumber = 0; ShaderNumber < ShaderCount; ShaderNumber++) {
				glDetachShader(shaderProgramObject_PF_RRJ, pShader[ShaderNumber]);
				glDeleteShader(pShader[ShaderNumber]);
				pShader[ShaderNumber] = 0;
			}
			free(pShader);
			pShader = NULL;
		}
		glDeleteProgram(shaderProgramObject_PF_RRJ);
		shaderProgramObject_PF_RRJ = 0;
		glUseProgram(0);
	}


	if (wglGetCurrentContext() == ghrc_RRJ) {
		wglMakeCurrent(NULL, NULL);
	}

	if (ghrc_RRJ) {
		wglDeleteContext(ghrc_RRJ);
		ghrc_RRJ = NULL;
	}

	if (ghdc_RRJ) {
		ReleaseDC(ghwnd_RRJ, ghdc_RRJ);
		ghdc_RRJ = NULL;
	}

	if (gbFile_RRJ) {
		fprintf(gbFile_RRJ, "Log Close!!\n");
		fclose(gbFile_RRJ);
		gbFile_RRJ = NULL;
	}
}


void resize(int width, int height) {
	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	perspectiveProjectionMatrix_RRJ = mat4::identity();
	perspectiveProjectionMatrix_RRJ = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void display(void) {

	mat4 translateMatrix_RRJ;
	mat4 rotateMatrix_RRJ;
	mat4 modelMatrix_RRJ;
	mat4 viewMatrix_RRJ;

	void rotateRedLight(GLfloat);
	void rotateGreenLight(GLfloat);
	void rotateBlueLight(GLfloat);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (iWhichLight_RRJ == 1)
		glUseProgram(shaderProgramObject_PV_RRJ);
	else
		glUseProgram(shaderProgramObject_PF_RRJ);

	translateMatrix_RRJ = mat4::identity();
	rotateMatrix_RRJ = mat4::identity();
	modelMatrix_RRJ = mat4::identity();
	viewMatrix_RRJ = mat4::identity();

	translateMatrix_RRJ = translate(0.0f, 0.0f, -1.50f);
	modelMatrix_RRJ = modelMatrix_RRJ * translateMatrix_RRJ;

	glUniformMatrix4fv(modelMatrixUniform_RRJ, 1, GL_FALSE, modelMatrix_RRJ);
	glUniformMatrix4fv(viewMatrixUniform_RRJ, 1, GL_FALSE, viewMatrix_RRJ);
	glUniformMatrix4fv(projectionMatrixUniform_RRJ, 1, GL_FALSE, perspectiveProjectionMatrix_RRJ);

	if (bLight_RRJ == true) {

		glUniform1i(LKeyPressUniform_RRJ, 1);

		rotateRedLight(angle_red_RRJ);
		rotateGreenLight(angle_green_RRJ);
		rotateBlueLight(angle_blue_RRJ);
		
		glUniform3fv(red_LaUniform_RRJ, 1, lightAmbient_Red_RRJ);
		glUniform3fv(red_LdUniform_RRJ, 1, lightDiffuse_Red_RRJ);
		glUniform3fv(red_LsUniform_RRJ, 1, lightSpecular_Red_RRJ);
		glUniform4fv(red_lightPositionUniform_RRJ, 1, lightPosition_Red_RRJ);

		glUniform3fv(green_LaUniform_RRJ, 1, lightAmbient_Green_RRJ);
		glUniform3fv(green_LdUniform_RRJ, 1, lightDiffuse_Green_RRJ);
		glUniform3fv(green_LsUniform_RRJ, 1, lightSpecular_Green_RRJ);
		glUniform4fv(green_lightPositionUniform_RRJ, 1, lightPosition_Green_RRJ);

		glUniform3fv(blue_LaUniform_RRJ, 1, lightAmbient_Blue_RRJ);
		glUniform3fv(blue_LdUniform_RRJ, 1, lightDiffuse_Blue_RRJ);
		glUniform3fv(blue_LsUniform_RRJ, 1, lightSpecular_Blue_RRJ);
		glUniform4fv(blue_lightPositionUniform_RRJ, 1, lightPosition_Blue_RRJ);


		glUniform3fv(KaUniform_RRJ, 1, materialAmbient_RRJ);
		glUniform3fv(KdUniform_RRJ, 1, materialDiffuse_RRJ);
		glUniform3fv(KsUniform_RRJ, 1, materialSpecular_RRJ);
		glUniform1f(shininessUniform_RRJ, materialShininess_RRJ);
	}
	else
		glUniform1i(LKeyPressUniform_RRJ, 0);

	glBindVertexArray(vao_Sphere_RRJ);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ);
	glDrawElements(GL_TRIANGLES, gNumElements_RRJ, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	glUseProgram(0);
	SwapBuffers(ghdc_RRJ);

}

void rotateRedLight(float angle) {
	lightPosition_Red_RRJ[1] = (float)(5.0f * sin(angle));
	lightPosition_Red_RRJ[2] = (float)(5.0f * cos(angle));
}

void rotateGreenLight(float angle) {
	lightPosition_Green_RRJ[0] = (float)(5.0f * sin(angle));
	lightPosition_Green_RRJ[2] = (float)(5.0f * cos(angle));
}

void rotateBlueLight(float angle) {
	lightPosition_Blue_RRJ[0] = (float)(5.0f * cos(angle));
	lightPosition_Blue_RRJ[1] = (float)(5.0f * sin(angle));
}




void update() {
	angle_red_RRJ = angle_red_RRJ + 0.020f;
	angle_green_RRJ = angle_green_RRJ + 0.02f;
	angle_blue_RRJ = angle_blue_RRJ + 0.02f;

	if (angle_red_RRJ > 360.0f)
		angle_red_RRJ = 0.0f;

	if (angle_green_RRJ > 360.0f)
		angle_green_RRJ = 0.0f;

	if (angle_blue_RRJ > 360.0f)
		angle_blue_RRJ = 0.0f;
 }
