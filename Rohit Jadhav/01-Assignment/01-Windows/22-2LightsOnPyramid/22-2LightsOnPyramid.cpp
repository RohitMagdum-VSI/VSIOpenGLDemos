#include<Windows.h>
#include<stdio.h>
#include<GL/glew.h>
#include<GL/gl.h>
#include"vmath.h"

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")

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
GLuint LaRed_Uniform_RRJ;
GLuint LdRed_Uniform_RRJ;
GLuint LsRed_Uniform_RRJ;
GLuint lightPositionRed_Uniform_RRJ;
GLuint LaBlue_Uniform_RRJ;
GLuint LdBlue_Uniform_RRJ;
GLuint LsBlue_Uniform_RRJ;
GLuint lightPositionBlue_Uniform_RRJ;
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
GLfloat lightPosition_Red_RRJ[] = { -2.0f, 0.0f, 0.0f, 1.0f };

GLfloat lightAmbient_Blue_RRJ[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat lightDiffuse_Blue_RRJ[] = { 0.0f, 0.0f, 1.0f, 1.0f };
GLfloat lightSpecular_Blue_RRJ[] = { 0.0f, 0.0f, 1.0f, 1.0f };
GLfloat lightPosition_Blue_RRJ[] = { 2.0f, 0.0f, 0.0f, 1.0f };


//For Material
GLfloat materialAmbient_RRJ[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat materialDiffuse_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialSpecular_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialShininess_RRJ = 128.0f;

//For Pyramid
GLuint texture_Pyramid_RRJ;
GLuint vao_Pyramid_RRJ;
GLuint vbo_Pyramid_Position_RRJ;
GLuint vbo_Pyramid_Normal_RRJ;
GLfloat angle_Pyramid_RRJ = 0.0f;



LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR szCmdLine, int iCmdShow) {

	int initialize(void);
	void display(void);
	void ToggleFullScreen(void);
	void uninitialize(void);

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
	TCHAR szName_RRJ[] = TEXT("RohitRJadhav-PP-2LightsOnPyramid");

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
		TEXT("RohitRJadhav-PP-2LightsOnPyramid"),
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
		"uniform vec3 u_La_Red;" \
		"uniform vec3 u_Ld_Red;" \
		"uniform vec3 u_Ls_Red;" \
		"uniform vec4 u_light_position_Red;" \
		"uniform vec3 u_La_Blue;" \
		"uniform vec3 u_Ld_Blue;" \
		"uniform vec3 u_Ls_Blue;" \
		"uniform vec4 u_light_position_Blue;" \
		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_shininess;" \
		"uniform int u_L_keypress;" \
		"out vec3 phongLight;"
		"void main(void)" \
		"{" \
		"vec3 phongRed_Light;" \
		"vec3 phongBlue_Light;" \
		"if(u_L_keypress == 1){" \
		"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" \

		"vec3 Source_Red = normalize(vec3(u_light_position_Red - eyeCoordinate));" \
		"vec3 Source_Blue = normalize(vec3(u_light_position_Blue - eyeCoordinate));" \

		"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" \
		"vec3 Normal = normalize(normalMatrix * vNormals);" \

		"float SRed_Dot_N = max(dot(Source_Red, Normal), 0.0);" \
		"float SBlue_Dot_N = max(dot(Source_Blue, Normal), 0.0);" \

		"vec3 Reflection_Red = reflect(-Source_Red, Normal);" \
		"vec3 Reflection_Blue = reflect(-Source_Blue, Normal);" \

		"vec3 Viewer = normalize(vec3(-eyeCoordinate.xyz));" \

		"float RRed_Dot_V = max(dot(Reflection_Red, Viewer), 0.0);" \
		"float RBlue_Dot_V = max(dot(Reflection_Blue, Viewer), 0.0);" \

		"vec3 ambient_Red = u_La_Red * u_Ka;" \
		"vec3 diffuse_Red = u_Ld_Red * u_Kd * SRed_Dot_N;" \
		"vec3 specular_Red = u_Ls_Red * u_Ks * pow(RRed_Dot_V, u_shininess);" \
		"phongRed_Light = ambient_Red + diffuse_Red + specular_Red;" \

		"vec3 ambient_Blue = u_La_Blue * u_Ka;" \
		"vec3 diffuse_Blue = u_Ld_Blue * u_Kd * SBlue_Dot_N;" \
		"vec3 specular_Blue = u_Ls_Blue * u_Ks * pow(RBlue_Dot_V, u_shininess);" \
		"phongBlue_Light = ambient_Blue + diffuse_Blue + specular_Blue;" \

		"phongLight = phongRed_Light + phongBlue_Light;" \

		"}" \
		"else{" \
		"phongLight = vec3(1.0, 1.0, 1.0);" \
		"}" \
		"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
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
		"FragColor = vec4(phongLight, 0.0);" \
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

	LaRed_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_La_Red");
	LdRed_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ld_Red");
	LsRed_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ls_Red");
	lightPositionRed_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_light_position_Red");

	LaBlue_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_La_Blue");
	LdBlue_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ld_Blue");
	LsBlue_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ls_Blue");
	lightPositionBlue_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_light_position_Blue");

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
		"uniform vec4 u_light_position_Red;" \
		"uniform vec4 u_light_position_Blue;" \
		"out vec3 lightDirectionRed_VS;" \
		"out vec3 lightDirectionBlue_VS;" \
		"out vec3 Normal_VS;" \
		"out vec3 Viewer_VS;" \
		"void main(void)" \
		"{"
		"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" \
		"lightDirectionRed_VS = vec3(u_light_position_Red - eyeCoordinate);" \
		"lightDirectionBlue_VS = vec3(u_light_position_Blue - eyeCoordinate);" \
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
		"in vec3 lightDirectionBlue_VS;" \
		"in vec3 Normal_VS;" \
		"in vec3 Viewer_VS;" \
		"uniform vec3 u_La_Red;" \
		"uniform vec3 u_Ld_Red;" \
		"uniform vec3 u_Ls_Red;" \

		"uniform vec3 u_La_Blue;" \
		"uniform vec3 u_Ld_Blue;" \
		"uniform vec3 u_Ls_Blue;" \

		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \

		"uniform float u_shininess;" \
		"uniform int u_L_keypress;" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
		"vec3 phongRed_Light;" \
		"vec3 phongBlue_Light;" \
		"vec3 phongLight;" \
		"if(u_L_keypress == 1){" \
		"vec3 LightDirection_Red = normalize(lightDirectionRed_VS);" \
		"vec3 LightDirection_Blue = normalize(lightDirectionBlue_VS);" \

		"vec3 Normal = normalize(Normal_VS);" \
		"float LRed_Dot_N = max(dot(LightDirection_Red, Normal), 0.0);" \
		"float LBlue_Dot_N = max(dot(LightDirection_Blue, Normal), 0.0);" \

		"vec3 ReflectionRed = reflect(-LightDirection_Red, Normal);" \
		"vec3 ReflectionBlue = reflect(-LightDirection_Blue, Normal);" \

		"vec3 Viewer = normalize(Viewer_VS);" \

		"float RRed_Dot_V = max(dot(ReflectionRed, Viewer), 0.0);" \
		"float RBlue_Dot_V = max(dot(ReflectionBlue, Viewer), 0.0);" \

		"vec3 ambient_Red = u_La_Red * u_Ka;" \
		"vec3 diffuse_Red = u_Ld_Red * u_Kd * LRed_Dot_N;" \
		"vec3 specular_Red = u_Ls_Red * u_Ks * pow(RRed_Dot_V, u_shininess);" \
		"phongRed_Light = ambient_Red + diffuse_Red + specular_Red;" \

		"vec3 ambient_Blue = u_La_Blue * u_Ka;" \
		"vec3 diffuse_Blue = u_Ld_Blue * u_Kd * LBlue_Dot_N;" \
		"vec3 specular_Blue = u_Ls_Blue * u_Ks * pow(RBlue_Dot_V, u_shininess);" \
		"phongBlue_Light = ambient_Blue + diffuse_Blue + specular_Blue;" \

		"phongLight = phongRed_Light + phongBlue_Light;" \

			"}" \
			"else{" \
				"phongLight = vec3(1.0, 1.0, 1.0);" \
			"}" \
			"FragColor = vec4(phongLight, 0.0);" \
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

	modelMatrixUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_model_matrix");
	viewMatrixUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_view_matrix");
	projectionMatrixUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_projection_matrix");

	LaRed_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_La_Red");
	LdRed_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ld_Red");
	LsRed_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ls_Red");
	lightPositionRed_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_light_position_Red");

	LaBlue_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_La_Blue");
	LdBlue_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ld_Blue");
	LsBlue_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ls_Blue");
	lightPositionBlue_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_light_position_Blue");

	KaUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ka");
	KdUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Kd");
	KsUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ks");
	shininessUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_shininess");
	LKeyPressUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_L_keypress");




	/********** Positions **********/
	GLfloat Pyramid_Vertices_RRJ[] = {
		//Face
		0.0f, 1.0f, 0.0f,
		-1.0, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		//Right
		0.0f, 1.0f, 0.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f,
		//Back
		0.0f, 1.0f, 0.0f,
		1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		//Left
		0.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f
	};

	GLfloat Pyramid_Normals_RRJ[] = {
		//Face
		0.0f, 0.447214f, 0.894427f,
		0.0f, 0.447214f, 0.894427f,
		0.0f, 0.447214f, 0.894427f,


		//Right
		0.894427f, 0.447214f, 0.0f,
		0.894427f, 0.447214f, 0.0f,
		0.894427f, 0.447214f, 0.0f,


		//Back
		0.0f, 0.447214f, -0.894427f,
		0.0f, 0.447214f, -0.894427f,
		0.0f, 0.447214f, -0.894427f,

		//Left
		-0.894427f, 0.447214f, 0.0f,
		-0.894427f, 0.447214f, 0.0f,
		-0.894427f, 0.447214f, 0.0f,
	};



	/********* Vao Pyramid **********/
	glGenVertexArrays(1, &vao_Pyramid_RRJ);
	glBindVertexArray(vao_Pyramid_RRJ);

	/********** Position *********/
	glGenBuffers(1, &vbo_Pyramid_Position_RRJ);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Pyramid_Position_RRJ);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Pyramid_Vertices_RRJ),
		Pyramid_Vertices_RRJ,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	/********** Normals **********/
	glGenBuffers(1, &vbo_Pyramid_Normal_RRJ);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Pyramid_Normal_RRJ);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Pyramid_Normals_RRJ),
		Pyramid_Normals_RRJ,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

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

	if (vbo_Pyramid_Normal_RRJ) {
		glDeleteBuffers(1, &vbo_Pyramid_Normal_RRJ);
		vbo_Pyramid_Normal_RRJ = 0;
	}

	if (vbo_Pyramid_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Pyramid_Position_RRJ);
		vbo_Pyramid_Position_RRJ = 0;
	}

	if (vao_Pyramid_RRJ) {
		glDeleteVertexArrays(1, &vao_Pyramid_RRJ);
		vao_Pyramid_RRJ = 0;
	}

	GLsizei ShaderCount_RRJ;
	GLsizei ShaderNumber_RRJ;

	if (shaderProgramObject_PV_RRJ) {
		glUseProgram(shaderProgramObject_PV_RRJ);

		glGetProgramiv(shaderProgramObject_PV_RRJ, GL_ATTACHED_SHADERS, &ShaderCount_RRJ);

		GLuint *pShader_RRJ = (GLuint*)malloc(sizeof(GLuint) * ShaderCount_RRJ);
		if (pShader_RRJ) {
			glGetAttachedShaders(shaderProgramObject_PV_RRJ, ShaderCount_RRJ, &ShaderCount_RRJ, pShader_RRJ);
			for (ShaderNumber_RRJ = 0; ShaderNumber_RRJ < ShaderCount_RRJ; ShaderNumber_RRJ++) {
				glDetachShader(shaderProgramObject_PV_RRJ, pShader_RRJ[ShaderNumber_RRJ]);
				glDeleteShader(pShader_RRJ[ShaderNumber_RRJ]);
				pShader_RRJ[ShaderNumber_RRJ] = 0;
			}
			free(pShader_RRJ);
			pShader_RRJ = NULL;
		}
		glDeleteProgram(shaderProgramObject_PV_RRJ);
		shaderProgramObject_PV_RRJ = 0;
		glUseProgram(0);
	}


	ShaderCount_RRJ = 0;
	ShaderNumber_RRJ = 0;
	if (shaderProgramObject_PF_RRJ) {
		glUseProgram(shaderProgramObject_PF_RRJ);

		glGetProgramiv(shaderProgramObject_PF_RRJ, GL_ATTACHED_SHADERS, &ShaderCount_RRJ);

		GLuint *pShader_RRJ = (GLuint*)malloc(sizeof(GLuint) * ShaderCount_RRJ);
		if (pShader_RRJ) {
			glGetAttachedShaders(shaderProgramObject_PF_RRJ, ShaderCount_RRJ, &ShaderCount_RRJ, pShader_RRJ);
			for (ShaderNumber_RRJ = 0; ShaderNumber_RRJ < ShaderCount_RRJ; ShaderNumber_RRJ++) {
				glDetachShader(shaderProgramObject_PF_RRJ, pShader_RRJ[ShaderNumber_RRJ]);
				glDeleteShader(pShader_RRJ[ShaderNumber_RRJ]);
				pShader_RRJ[ShaderNumber_RRJ] = 0;
			}
			free(pShader_RRJ);
			pShader_RRJ = NULL;
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
	static GLfloat angle_RRJ = 0.0f;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (iWhichLight_RRJ == 1)
		glUseProgram(shaderProgramObject_PV_RRJ);
	else
		glUseProgram(shaderProgramObject_PF_RRJ);

	translateMatrix_RRJ = mat4::identity();
	rotateMatrix_RRJ = mat4::identity();
	modelMatrix_RRJ = mat4::identity();
	viewMatrix_RRJ = mat4::identity();

	translateMatrix_RRJ = translate(0.0f, 0.0f, -4.0f);
	rotateMatrix_RRJ = rotate(angle_RRJ, 0.0f, 1.0f, 0.0f);
	modelMatrix_RRJ = modelMatrix_RRJ * translateMatrix_RRJ * rotateMatrix_RRJ;

	glUniformMatrix4fv(modelMatrixUniform_RRJ, 1, GL_FALSE, modelMatrix_RRJ);
	glUniformMatrix4fv(viewMatrixUniform_RRJ, 1, GL_FALSE, viewMatrix_RRJ);
	glUniformMatrix4fv(projectionMatrixUniform_RRJ, 1, GL_FALSE, perspectiveProjectionMatrix_RRJ);

	if (bLight_RRJ == true) {

		glUniform1i(LKeyPressUniform_RRJ, 1);
		glUniform3fv(LaRed_Uniform_RRJ, 1, lightAmbient_Red_RRJ);
		glUniform3fv(LdRed_Uniform_RRJ, 1, lightDiffuse_Red_RRJ);
		glUniform3fv(LsRed_Uniform_RRJ, 1, lightSpecular_Red_RRJ);
		glUniform4fv(lightPositionRed_Uniform_RRJ, 1, lightPosition_Red_RRJ);

		glUniform3fv(LaBlue_Uniform_RRJ, 1, lightAmbient_Blue_RRJ);
		glUniform3fv(LdBlue_Uniform_RRJ, 1, lightDiffuse_Blue_RRJ);
		glUniform3fv(LsBlue_Uniform_RRJ, 1, lightSpecular_Blue_RRJ);
		glUniform4fv(lightPositionBlue_Uniform_RRJ, 1, lightPosition_Blue_RRJ);

		glUniform3fv(KaUniform_RRJ, 1, materialAmbient_RRJ);
		glUniform3fv(KdUniform_RRJ, 1, materialDiffuse_RRJ);
		glUniform3fv(KsUniform_RRJ, 1, materialSpecular_RRJ);
		glUniform1f(shininessUniform_RRJ, materialShininess_RRJ);
	}
	else
		glUniform1i(LKeyPressUniform_RRJ, 0);

	glBindVertexArray(vao_Pyramid_RRJ);
	glDrawArrays(GL_TRIANGLES,
		0,
		12);
	glBindVertexArray(0);


	angle_RRJ += 0.5f;

	glUseProgram(0);
	SwapBuffers(ghdc_RRJ);

}

