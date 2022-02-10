#include<Windows.h>
#include<stdio.h>
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
	AMC_ATTRIBUTE_TEXCOORD0,
};	

//For Fullscreen
bool bIsFullScreen_RRJ = false;
HWND ghwnd_RRJ = NULL;
WINDOWPLACEMENT wpPrev_RRJ = {sizeof(WINDOWPLACEMENT)};
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
GLuint LaUniform_RRJ;
GLuint LdUniform_RRJ;
GLuint LsUniform_RRJ;
GLuint lightPositionUniform_RRJ;
GLuint KaUniform_RRJ;
GLuint KdUniform_RRJ;
GLuint KsUniform_RRJ;
GLuint shininessUniform_RRJ;
GLuint LKeyPressUniform_RRJ;


//For Light
int iWhichLight_RRJ = 1;
bool bLight_RRJ = false;
GLfloat lightAmbient_RRJ[] = {0.0f, 0.0f, 0.0f, 0.0f};
GLfloat lightDiffuse_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
GLfloat lightSpecular_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
GLfloat lightPosition_RRJ[] = {0.0f, 0.0f, 0.0f, 1.0f};



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


//For Viewport Toggling
int iViewPortNo_RRJ = 1;
float windowWidth_RRJ = 0.0f;
float windowHeight_RRJ = 0.0f;

//For Light Rotationn
const int X_ROT = 1;
const int Y_ROT = 2;
const int Z_ROT = 3;
int iWhichRotation_RRJ = X_ROT;
float angle_Sphere_RRJ = 0.0f;


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
	TCHAR szName_RRJ[] = TEXT("RohitRJadhav-PP-24Spheres");

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
		TEXT("RohitRJadhav-PP-24Spheres"),
		WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_VISIBLE,
		100, 100,
		WIN_WIDTH, WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghwnd_RRJ = hwnd_RRJ;

	

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
	SetForegroundWindow(hwnd_RRJ);
	SetFocus(hwnd_RRJ);
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
		windowWidth_RRJ = LOWORD(lParam);
		windowHeight_RRJ = HIWORD(lParam);
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


		case 'X':
		case 'x':
			iWhichRotation_RRJ = X_ROT;
			break;

		case 'Y':
		case 'y':
			iWhichRotation_RRJ = Y_ROT;
			break;

		case 'Z':
		case 'z':
			iWhichRotation_RRJ = Z_ROT;
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
	GLuint fragmentShaderObject_PF;

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
		"uniform vec3 u_La;" \
		"uniform vec3 u_Ld;" \
		"uniform vec3 u_Ls;" \
		"uniform vec4 u_light_position;" \
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
				"vec3 Source = normalize(vec3(u_light_position - eyeCoordinate));" \
				"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" \
				"vec3 Normal = normalize(normalMatrix * vNormals);" \
				"float S_Dot_N = max(dot(Source, Normal), 0.0);" \
				"vec3 Reflection = reflect(-Source, Normal);" \
				"vec3 Viewer = normalize(vec3(-eyeCoordinate.xyz));" \
				"float R_Dot_V = max(dot(Reflection, Viewer), 0.0);" \
				"vec3 ambient = u_La * u_Ka;" \
				"vec3 diffuse = u_Ld * u_Kd * S_Dot_N;" \
				"vec3 specular = u_Ls * u_Ks * pow(R_Dot_V, u_shininess);" \
				"phongLight = ambient + diffuse + specular;" \
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
	LaUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_La");
	LdUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ld");
	LsUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ls");
	lightPositionUniform_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_light_position");
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
		"uniform vec4 u_light_position;" \
		"out vec3 lightDirection_VS;" \
		"out vec3 Normal_VS;" \
		"out vec3 Viewer_VS;" \
		"void main(void)" \
		"{"
			"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" \
			"lightDirection_VS = vec3(u_light_position - eyeCoordinate);" \
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
	fragmentShaderObject_PF = glCreateShader(GL_FRAGMENT_SHADER);
	const char* szFragmentShaderCode_PF_RRJ =
		"#version 450 core" \
		"\n" \
		"in vec3 lightDirection_VS;" \
		"in vec3 Normal_VS;" \
		"in vec3 Viewer_VS;" \
		"uniform vec3 u_La;" \
		"uniform vec3 u_Ld;" \
		"uniform vec3 u_Ls;" \
		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3	 u_Ks;" \
		"uniform float u_shininess;" \
		"uniform int u_L_keypress;" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
			"vec3 phongLight;" \
			"if(u_L_keypress == 1){" \
				"vec3 LightDirection = normalize(lightDirection_VS);" \
				"vec3 Normal = normalize(Normal_VS);" \
				"float L_Dot_N = max(dot(LightDirection, Normal), 0.0);" \
				"vec3 Reflection = reflect(-LightDirection, Normal);" \
				"vec3 Viewer = normalize(Viewer_VS);" \
				"float R_Dot_V = max(dot(Reflection, Viewer), 0.0);" \
				"vec3 ambient = u_La * u_Ka;" \
				"vec3 diffuse = u_Ld * u_Kd * L_Dot_N;" \
				"vec3 specular = u_Ls * u_Ks * pow(R_Dot_V, u_shininess);" \
				"phongLight = ambient + diffuse + specular;" \
			"}" \
			"else{" \
				"phongLight = vec3(1.0, 1.0, 1.0);" \
			"}" \
			"FragColor = vec4(phongLight, 0.0);" \
		"}";


	glShaderSource(fragmentShaderObject_PF, 1,
		(const GLchar**)&szFragmentShaderCode_PF_RRJ, NULL);

	glCompileShader(fragmentShaderObject_PF);

	iShaderCompileStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(fragmentShaderObject_PF, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(fragmentShaderObject_PF, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject_PF, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
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
	glAttachShader(shaderProgramObject_PF_RRJ, fragmentShaderObject_PF);

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
	LaUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_La");
	LdUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Ld");
	LsUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Ls");
	lightPositionUniform_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_light_position");
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

	glClearColor(0.250f, 0.250f, 0.250f, 1.0f);

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

	GLsizei w = (GLsizei)width;
	GLsizei h = (GLsizei)height;

	if(iViewPortNo_RRJ == 1)							/************ 1st SET ***********/
		glViewport( 0, 5 * h / 6, w / 6, h / 6);
	else if(iViewPortNo_RRJ == 2)
		glViewport( 0, 4 * h / 6, w / 6,  h / 6);
	else if(iViewPortNo_RRJ == 3)
		glViewport( 0, 3 * h / 6, w / 6, h / 6);
	else if(iViewPortNo_RRJ == 4)
		glViewport( 0, 2 * h / 6, w / 6, h / 6);
	else if(iViewPortNo_RRJ == 5)
		glViewport( 0, 1 * h / 6, w / 6, h / 6);
	else if(iViewPortNo_RRJ == 6)
		glViewport( 0, 0, w / 6, h / 6);
	else if(iViewPortNo_RRJ == 7)						/************ 2nd SET ***********/
		glViewport( 1 * w / 4, 5 * h / 6, w / 6, h / 6);
	else if(iViewPortNo_RRJ == 8)
		glViewport( 1 * w / 4, 4 * h / 6, w / 6,  h / 6);
	else if(iViewPortNo_RRJ == 9)
		glViewport( 1 * w / 4, 3 * h / 6, w / 6, h / 6);
	else if(iViewPortNo_RRJ == 10)
		glViewport( 1 * w / 4, 2 * h / 6, w / 6, h / 6);
	else if(iViewPortNo_RRJ == 11)
		glViewport( 1 * w / 4, 1 * h / 6, w / 6, h / 6);
	else if(iViewPortNo_RRJ == 12)
		glViewport( 1 * w / 4, 0, w / 6, h / 6);
	else if(iViewPortNo_RRJ == 13)						/************ 3rd SET ***********/
		glViewport( 2 * w / 4, 5 * h / 6, w / 6, h / 6);
	else if(iViewPortNo_RRJ == 14)						
		glViewport( 2 * w / 4, 4 * h / 6, w / 6,  h / 6);
	else if(iViewPortNo_RRJ == 15)
		glViewport( 2 * w / 4, 3 * h / 6, w / 6, h / 6);
	else if(iViewPortNo_RRJ == 16)
		glViewport( 2 * w / 4, 2 * h / 6, w / 6, h / 6);
	else if(iViewPortNo_RRJ == 17)
		glViewport( 2 * w / 4, 1 * h / 6, w / 6, h / 6);
	else if(iViewPortNo_RRJ == 18)						
		glViewport( 2 * w / 4, 0, w / 6, h / 6);
	else if(iViewPortNo_RRJ == 19)						/************ 4th SET ***********/
		glViewport( 3 * w / 4, 5 * h / 6, w / 6, h / 6);
	else if(iViewPortNo_RRJ == 20)
		glViewport( 3 * w / 4, 4 * h / 6, w / 6,  h / 6);
	else if(iViewPortNo_RRJ == 21)
		glViewport( 3 * w / 4, 3 * h / 6, w / 6, h / 6);
	else if(iViewPortNo_RRJ == 22)
		glViewport( 3 * w / 4, 2 * h / 6, w / 6, h / 6);
	else if(iViewPortNo_RRJ == 23)
		glViewport( 3 * w / 4, 1 * h / 6, w / 6, h / 6);
	else if(iViewPortNo_RRJ == 24)
		glViewport( 3 * w / 4, 0, w / 6, h / 6);


	perspectiveProjectionMatrix_RRJ = mat4::identity();
	perspectiveProjectionMatrix_RRJ = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

//For Matrix
mat4 translateMatrix_RRJ;
mat4 modelMatrix_RRJ;
mat4 viewMatrix_RRJ;



void display(void) {

	void draw24Spheres();


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	if(iWhichLight_RRJ == 1){
		//Per Vertex

		glUseProgram(shaderProgramObject_PV_RRJ);

			draw24Spheres();

		glUseProgram(0);
	}
	else{
		glUseProgram(shaderProgramObject_PF_RRJ);

			draw24Spheres();

		glUseProgram(0);

	}

	SwapBuffers(ghdc_RRJ);
}


void update(){
	angle_Sphere_RRJ += 0.001f;
	if(angle_Sphere_RRJ > 360.0f)
		angle_Sphere_RRJ = 0.0;
}

void draw24Spheres(){


	void rotateX(float);
	void rotateY(float);
	void rotateZ(float);

	float materialAmbient_RRJ[] = {0.0f, 0.0f, 0.0f, 1.0f};
	float materialDiffuse_RRJ[] = {0.0f, 0.0f, 0.0f, 1.0f};
	float materialSpecular_RRJ[] = {0.0f, 0.0f, 0.0f, 1.0f};
	float materialShininess_RRJ = 0.0f;


	for(int i = 1 ; i <= 24; i++){


		if(i == 1){
			materialAmbient_RRJ[0] = 0.0215;
			materialAmbient_RRJ[1] = 0.1745;
			materialAmbient_RRJ[2] = 0.215;
			

			materialDiffuse_RRJ[0] = 0.07568;
			materialDiffuse_RRJ[1] = 0.61424;
			materialDiffuse_RRJ[2] = 0.07568;
			

			materialSpecular_RRJ[0] = 0.633;
			materialSpecular_RRJ[1] = 0.727811;
			materialSpecular_RRJ[2] = 0.633;
			

			materialShininess_RRJ = 0.6 * 128;

		}
		else if(i == 2){
			materialAmbient_RRJ[0] = 0.135;
			materialAmbient_RRJ[1] = 0.2225;
			materialAmbient_RRJ[2] = 0.1575;
			

			materialDiffuse_RRJ[0] = 0.54;
			materialDiffuse_RRJ[1] = 0.89;
			materialDiffuse_RRJ[2] = 0.63;
			

			materialSpecular_RRJ[0] = 0.316228;
			materialSpecular_RRJ[1] = 0.316228;
			materialSpecular_RRJ[2] = 0.316228;
			

			materialShininess_RRJ = 0.1 * 128;
		}
		else if(i == 3){
			materialAmbient_RRJ[0] = 0.05375;
			materialAmbient_RRJ[1] = 0.05;
			materialAmbient_RRJ[2] = 0.06625;
			

			materialDiffuse_RRJ[0] = 0.18275;
			materialDiffuse_RRJ[1] = 0.17;
			materialDiffuse_RRJ[2] = 0.22525;
			

			materialSpecular_RRJ[0] = 0.332741;
			materialSpecular_RRJ[1] = 0.328634;
			materialSpecular_RRJ[2] = 0.346435;
			

			materialShininess_RRJ = 0.3 * 128;
		}
		else if(i == 4){
			materialAmbient_RRJ[0] = 0.25;
			materialAmbient_RRJ[1] = 0.20725;
			materialAmbient_RRJ[2] = 0.20725;
			

			materialDiffuse_RRJ[0] = 1.0;
			materialDiffuse_RRJ[1] = 0.829;
			materialDiffuse_RRJ[2] = 0.829;
			

			materialSpecular_RRJ[0] = 0.296648;
			materialSpecular_RRJ[1] = 0.296648;
			materialSpecular_RRJ[2] = 0.296648;
			

			materialShininess_RRJ = 0.088 * 128;
		}
		else if(i == 5){
			materialAmbient_RRJ[0] = 0.1745;
			materialAmbient_RRJ[1] = 0.01175;
			materialAmbient_RRJ[2] = 0.01175;
			

			materialDiffuse_RRJ[0] = 0.61424;
			materialDiffuse_RRJ[1] = 0.04136;
			materialDiffuse_RRJ[2] = 0.04136;
			

			materialSpecular_RRJ[0] = 0.727811;
			materialSpecular_RRJ[1] = 0.626959;
			materialSpecular_RRJ[2] = 0.626959;
			

			materialShininess_RRJ = 0.6 * 128;
		}
		else if(i == 6){
			materialAmbient_RRJ[0] = 0.1;
			materialAmbient_RRJ[1] = 0.18725;
			materialAmbient_RRJ[2] = 0.1745;
			

			materialDiffuse_RRJ[0] = 0.396;
			materialDiffuse_RRJ[1] = 0.74151;
			materialDiffuse_RRJ[2] = 0.69102;
			

			materialSpecular_RRJ[0] = 0.297254;
			materialSpecular_RRJ[1] = 0.30829;
			materialSpecular_RRJ[2] = 0.306678;
			

			materialShininess_RRJ = 0.1 * 128;
		}
		else if(i == 7){
			materialAmbient_RRJ[0] = 0.329412;
			materialAmbient_RRJ[1] = 0.223529;
			materialAmbient_RRJ[2] = 0.027451;
			

			materialDiffuse_RRJ[0] = 0.780392;
			materialDiffuse_RRJ[1] = 0.568627;
			materialDiffuse_RRJ[2] = 0.113725;
			

			materialSpecular_RRJ[0] = 0.992157;
			materialSpecular_RRJ[1] = 0.941176;
			materialSpecular_RRJ[2] = 0.807843;
			

			materialShininess_RRJ = 0.21794872 * 128;
		}
		else if(i == 8){
			materialAmbient_RRJ[0] = 0.2125;
			materialAmbient_RRJ[1] = 0.1275;
			materialAmbient_RRJ[2] = 0.054;
			

			materialDiffuse_RRJ[0] = 0.714;
			materialDiffuse_RRJ[1] = 0.4284;
			materialDiffuse_RRJ[2] = 0.18144;
			

			materialSpecular_RRJ[0] = 0.393548;
			materialSpecular_RRJ[1] = 0.271906;
			materialSpecular_RRJ[2] = 0.166721;
			

			materialShininess_RRJ = 0.2 * 128;
		}
		else if(i == 9){
			materialAmbient_RRJ[0] = 0.25;
			materialAmbient_RRJ[1] = 0.25;
			materialAmbient_RRJ[2] = 0.25;
			

			materialDiffuse_RRJ[0] = 0.4;
			materialDiffuse_RRJ[1] = 0.4;
			materialDiffuse_RRJ[2] = 0.4;
			

			materialSpecular_RRJ[0] = 0.774597;
			materialSpecular_RRJ[1] = 0.774597;
			materialSpecular_RRJ[2] = 0.774597;
			

			materialShininess_RRJ = 0.6 * 128;
		}
		else if(i == 10){
			materialAmbient_RRJ[0] = 0.19125;
			materialAmbient_RRJ[1] = 0.0735;
			materialAmbient_RRJ[2] = 0.0225;
			

			materialDiffuse_RRJ[0] = 0.7038;
			materialDiffuse_RRJ[1] = 0.27048;
			materialDiffuse_RRJ[2] = 0.0828;
			

			materialSpecular_RRJ[0] = 0.256777;
			materialSpecular_RRJ[1] = 0.137622;
			materialSpecular_RRJ[2] = 0.086014;
			

			materialShininess_RRJ = 0.1 * 128;
		}
		else if(i == 11){
			materialAmbient_RRJ[0] = 0.24725;
			materialAmbient_RRJ[1] = 0.1995;
			materialAmbient_RRJ[2] = 0.0745;
			

			materialDiffuse_RRJ[0] = 0.75164;
			materialDiffuse_RRJ[1] = 0.60648;
			materialDiffuse_RRJ[2] = 0.22648;
			

			materialSpecular_RRJ[0] = 0.628281;
			materialSpecular_RRJ[1] = 0.555802;
			materialSpecular_RRJ[2] = 0.366065;
			

			materialShininess_RRJ = 0.4 * 128;
		}
		else if(i == 12){
			materialAmbient_RRJ[0] = 0.19225;
			materialAmbient_RRJ[1] = 0.19225;
			materialAmbient_RRJ[2] = 0.19225;
			

			materialDiffuse_RRJ[0] = 0.50754;
			materialDiffuse_RRJ[1] = 0.50754;
			materialDiffuse_RRJ[2] = 0.50754;
			

			materialSpecular_RRJ[0] = 0.508273;
			materialSpecular_RRJ[1] = 0.508273;
			materialSpecular_RRJ[2] = 0.508273;
			

			materialShininess_RRJ = 0.4 * 128;
		}
		else if(i == 13){
			materialAmbient_RRJ[0] = 0.0;
			materialAmbient_RRJ[1] = 0.0;
			materialAmbient_RRJ[2] = 0.0;
			

			materialDiffuse_RRJ[0] = 0.01;
			materialDiffuse_RRJ[1] = 0.01;
			materialDiffuse_RRJ[2] = 0.01;
			

			materialSpecular_RRJ[0] = 0.5;
			materialSpecular_RRJ[1] = 0.5;
			materialSpecular_RRJ[2] = 0.5;
			

			materialShininess_RRJ = 0.25 * 128;
		}
		else if(i == 14){
			materialAmbient_RRJ[0] = 0.0;
			materialAmbient_RRJ[1] = 0.1;
			materialAmbient_RRJ[2] = 0.06;
			

			materialDiffuse_RRJ[0] = 0.0;
			materialDiffuse_RRJ[1] = 0.50980392;
			materialDiffuse_RRJ[2] = 0.52980392;
			

			materialSpecular_RRJ[0] = 0.50196078;
			materialSpecular_RRJ[1] = 0.50196078;
			materialSpecular_RRJ[2] = 0.50196078;
			

			materialShininess_RRJ = 0.25 * 128;
		}
		else if(i == 15){
			materialAmbient_RRJ[0] = 0.0;
			materialAmbient_RRJ[1] = 0.0;
			materialAmbient_RRJ[2] = 0.0;
			

			materialDiffuse_RRJ[0] = 0.1;
			materialDiffuse_RRJ[1] = 0.35;
			materialDiffuse_RRJ[2] = 0.1;
			

			materialSpecular_RRJ[0] = 0.45;
			materialSpecular_RRJ[1] = 0.55;
			materialSpecular_RRJ[2] = 0.45;
			

			materialShininess_RRJ = 0.25 * 128;
		}
		else if(i == 16){
			materialAmbient_RRJ[0] = 0.0;
			materialAmbient_RRJ[1] = 0.0;
			materialAmbient_RRJ[2] = 0.0;
			

			materialDiffuse_RRJ[0] = 0.5;
			materialDiffuse_RRJ[1] = 0.0;
			materialDiffuse_RRJ[2] = 0.0;
			

			materialSpecular_RRJ[0] = 0.7;
			materialSpecular_RRJ[1] = 0.6;
			materialSpecular_RRJ[2] = 0.6;
			

			materialShininess_RRJ = 0.25 * 128;
		}
		else if(i == 17){
			materialAmbient_RRJ[0] = 0.0;
			materialAmbient_RRJ[1] = 0.0;
			materialAmbient_RRJ[2] = 0.0;
			

			materialDiffuse_RRJ[0] = 0.55;
			materialDiffuse_RRJ[1] = 0.55;
			materialDiffuse_RRJ[2] = 0.55;
			

			materialSpecular_RRJ[0] = 0.70;
			materialSpecular_RRJ[1] = 0.70;
			materialSpecular_RRJ[2] = 0.70;
			

			materialShininess_RRJ = 0.25 * 128;
		}
		else if(i == 18){
			materialAmbient_RRJ[0] = 0.0;
			materialAmbient_RRJ[1] = 0.0;
			materialAmbient_RRJ[2] = 0.0;
			

			materialDiffuse_RRJ[0] = 0.5;
			materialDiffuse_RRJ[1] = 0.5;
			materialDiffuse_RRJ[2] = 0.0;
			

			materialSpecular_RRJ[0] = 0.60;
			materialSpecular_RRJ[1] = 0.60;
			materialSpecular_RRJ[2] = 0.50;
			

			materialShininess_RRJ = 0.25 * 128;
		}
		else if(i == 19){
			materialAmbient_RRJ[0] = 0.02;
			materialAmbient_RRJ[1] = 0.02;
			materialAmbient_RRJ[2] = 0.02;
			

			materialDiffuse_RRJ[0] = 0.01;
			materialDiffuse_RRJ[1] = 0.01;
			materialDiffuse_RRJ[2] = 0.01;
			

			materialSpecular_RRJ[0] = 0.4;
			materialSpecular_RRJ[1] = 0.4;
			materialSpecular_RRJ[2] = 0.4;
			

			materialShininess_RRJ = 0.078125 * 128;
		}
		else if(i == 20){
			materialAmbient_RRJ[0] = 0.0;
			materialAmbient_RRJ[1] = 0.05;
			materialAmbient_RRJ[2] = 0.05;
			

			materialDiffuse_RRJ[0] = 0.4;
			materialDiffuse_RRJ[1] = 0.5;
			materialDiffuse_RRJ[2] = 0.5;
			

			materialSpecular_RRJ[0] = 0.04;
			materialSpecular_RRJ[1] = 0.7;
			materialSpecular_RRJ[2] = 0.7;
			

			materialShininess_RRJ = 0.078125 * 128;
		}
		else if(i == 21){
			materialAmbient_RRJ[0] = 0.0;
			materialAmbient_RRJ[1] = 0.05;
			materialAmbient_RRJ[2] = 0.0;
			

			materialDiffuse_RRJ[0] = 0.4;
			materialDiffuse_RRJ[1] = 0.5;
			materialDiffuse_RRJ[2] = 0.4;
			

			materialSpecular_RRJ[0] = 0.04;
			materialSpecular_RRJ[1] = 0.7;
			materialSpecular_RRJ[2] = 0.04;
			

			materialShininess_RRJ = 0.078125 * 128;
		}
		else if(i == 22){
			materialAmbient_RRJ[0] = 0.05;
			materialAmbient_RRJ[1] = 0.0;
			materialAmbient_RRJ[2] = 0.0;
			

			materialDiffuse_RRJ[0] = 0.5;
			materialDiffuse_RRJ[1] = 0.4;
			materialDiffuse_RRJ[2] = 0.4;
			

			materialSpecular_RRJ[0] = 0.70;
			materialSpecular_RRJ[1] = 0.04;
			materialSpecular_RRJ[2] = 0.04;
			

			materialShininess_RRJ = 0.078125 * 128;
		}
		else if(i == 23){
			materialAmbient_RRJ[0] = 0.05;
			materialAmbient_RRJ[1] = 0.05;
			materialAmbient_RRJ[2] = 0.05;
			

			materialDiffuse_RRJ[0] = 0.5;
			materialDiffuse_RRJ[1] = 0.5;
			materialDiffuse_RRJ[2] = 0.5;
			

			materialSpecular_RRJ[0] = 0.70;
			materialSpecular_RRJ[1] = 0.70;
			materialSpecular_RRJ[2] = 0.70;
			

			materialShininess_RRJ = 0.078125 * 128;
		}
		else if(i == 24){
			materialAmbient_RRJ[0] = 0.05;
			materialAmbient_RRJ[1] = 0.05;
			materialAmbient_RRJ[2] = 0.0;
			

			materialDiffuse_RRJ[0] = 0.5;
			materialDiffuse_RRJ[1] = 0.5;
			materialDiffuse_RRJ[2] = 0.4;
			

			materialSpecular_RRJ[0] = 0.70;
			materialSpecular_RRJ[1] = 0.70;
			materialSpecular_RRJ[2] = 0.04;
			

			materialShininess_RRJ = 0.078125 * 128;
		}


		iViewPortNo_RRJ = i;
		resize(windowWidth_RRJ, windowHeight_RRJ);


		translateMatrix_RRJ = mat4::identity();
		modelMatrix_RRJ = mat4::identity();
		viewMatrix_RRJ = mat4::identity();

		translateMatrix_RRJ = translate(0.0f, 0.0f, -1.5f);
		modelMatrix_RRJ = modelMatrix_RRJ * translateMatrix_RRJ;

		glUniformMatrix4fv(modelMatrixUniform_RRJ, 1, false, modelMatrix_RRJ);
		glUniformMatrix4fv(viewMatrixUniform_RRJ, 1, false, viewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_RRJ, 1, false, perspectiveProjectionMatrix_RRJ);

	


			
		if(bLight_RRJ == true){
			//Per Vertex

			if(iWhichRotation_RRJ == X_ROT)
				rotateX(angle_Sphere_RRJ);
			else if(iWhichRotation_RRJ == Y_ROT)
				rotateY(angle_Sphere_RRJ);
			else if(iWhichRotation_RRJ == Z_ROT)
				rotateZ(angle_Sphere_RRJ);

			update();


			glUniform1i(LKeyPressUniform_RRJ, 1);

			glUniform3fv(LaUniform_RRJ, 1, lightAmbient_RRJ);
			glUniform3fv(LdUniform_RRJ, 1, lightDiffuse_RRJ);
			glUniform3fv(LsUniform_RRJ, 1, lightSpecular_RRJ);
			glUniform4fv(lightPositionUniform_RRJ, 1, lightPosition_RRJ);

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

		glBindVertexArray(0);
	}
}

void rotateX(float angle){
	lightPosition_RRJ[0] = 0.0;
	lightPosition_RRJ[1] = 15.0 * sin(angle);
	lightPosition_RRJ[2] = 15.0 * cos(angle);
}

void rotateY(float angle){
	lightPosition_RRJ[0] = 15.0 * cos(angle);
	lightPosition_RRJ[1] = 0.0;
	lightPosition_RRJ[2] = 15.0 * sin(angle);
}


void rotateZ(float angle){
	lightPosition_RRJ[0] = 15.0 * cos(angle);
	lightPosition_RRJ[1] = 15.0 * sin(angle);
	lightPosition_RRJ[2] = 0.0;
}


