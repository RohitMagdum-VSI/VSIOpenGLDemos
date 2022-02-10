#include<Windows.h>
#include<stdio.h>
#include<gl/glew.h>
#include<GL/GL.h>
#include"vmath.h"
#include"Resource.h"
#include"Sphere.h"
#include<assert.h>

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "Sphere.lib")


enum {
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOORD0
};

using namespace vmath;

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

//For FullScreen
bool bIsFullScreen = false;
HWND ghwnd = NULL;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
DWORD dwStyle;

//For SuperMan
bool bActiveWindow = false;
HDC ghdc = NULL;
HGLRC ghrc = NULL;

//For Error
FILE *gbFile = NULL;

//For Shader Program Object;
GLint gShaderProgramObject;

//For Perspective Matric
mat4 gPerspectiveProjectionMatrix;


//For Cube
GLuint texture_CubeMap;
GLuint vao_Cube;
GLuint vbo_Cube_Position;
GLuint vbo_Cube_Texture;
GLfloat angle_Cube = 360.0f;

//For Uniforms
GLuint samplerUniform;
GLuint cubeMapToggleUniform;




//For Uniform
GLuint modelMatrixUniform;
GLuint viewMatrixUniform;
GLuint projectionMatrixUniform;


GLuint red_LaUniform;
GLuint red_LdUniform;
GLuint red_LsUniform;
GLuint red_lightPositionUniform;

GLuint green_LaUniform;
GLuint green_LdUniform;
GLuint green_LsUniform;
GLuint green_lightPositionUniform;

GLuint blue_LaUniform;
GLuint blue_LdUniform;
GLuint blue_LsUniform;
GLuint blue_lightPositionUniform;


GLuint KaUniform;
GLuint KdUniform;
GLuint KsUniform;
GLuint shininessUniform;
GLuint LKeyPressUniform;


//For Light
int iWhichLight = 1;
bool bLight = false;

GLfloat lightAmbient_Red[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat lightDiffuse_Red[] = { 1.0f, 0.0f, 0.0f, 1.0f };
GLfloat lightSpecular_Red[] = { 1.0f, 0.0f, 0.0f, 1.0f };
GLfloat lightPosition_Red[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat angle_red = 0.0f;

GLfloat lightAmbient_Green[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat lightDiffuse_Green[] = { 0.0f, 1.0f, 0.0f, 1.0f };
GLfloat lightSpecular_Green[] = { 0.0f, 1.0f, 0.0f, 1.0f };
GLfloat lightPosition_Green[] = { 0.0, 0.0f, 0.0f, 1.0f };
GLfloat angle_green = 0.0f;

GLfloat lightAmbient_Blue[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat lightDiffuse_Blue[] = { 0.0f, 0.0f, 1.0f, 1.0f };
GLfloat lightSpecular_Blue[] = { 0.0f, 0.0f, 1.0f, 1.0f };
GLfloat lightPosition_Blue[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat angle_blue = 0.0f;


//For Material
GLfloat materialAmbient[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat materialDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialShininess = 128.0f;

//For Sphere
GLuint vao_Sphere;
GLuint vbo_Sphere_Position;
GLuint vbo_Sphere_Normal;
GLuint vbo_Sphere_Element;
float sphere_vertices[1146];
float sphere_normals[1146];
float sphere_textures[764];
unsigned short sphere_elements[2280];
unsigned int gNumVertices;
unsigned int gNumElements;






LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow) {

	if (fopen_s(&gbFile, "Log.txt", "w") != 0) {
		MessageBox(NULL, TEXT("Log Creation Failed!!\n"), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
		fprintf(gbFile, "Log Created!!\n");

	
	int initialize(void);
	void display(void);
	void update(void);
	void ToggleFullScreen(void);

	int iRet;
	bool bDone = false;

	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szName[] = TEXT("RohitRJadhav-PP-EnvMap2");

	wndclass.lpszClassName = szName;
	wndclass.lpszMenuName = NULL;
	wndclass.lpfnWndProc = WndProc;

	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.cbWndExtra = 0;
	wndclass.cbClsExtra = 0;

	wndclass.hInstance = hInstance;
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);

	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szName,
		TEXT("RohitRJadhav-PP-EnvMap2"),
		WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_VISIBLE,
		100, 100,
		WIN_WIDTH, WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghwnd = hwnd;

	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	iRet = initialize();
	if (iRet == -1) {
		fprintf(gbFile, "ChoosePixelFormat() Failed!!\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -2) {
		fprintf(gbFile, "SetPixelFormat() Failed!!\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -3) {
		fprintf(gbFile, "wglMakeCurrent() Failed!!\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -4) {
		fprintf(gbFile, "wglMakeCurrent() Failed!!\n");
		DestroyWindow(hwnd);
	}
	else
		fprintf(gbFile, "initialize() done!!\n");



	ShowWindow(hwnd, iCmdShow);
	ToggleFullScreen();

	while (bDone == false) {
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
			if (msg.message == WM_QUIT)
				bDone = true;
			else {
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else {
			if (bActiveWindow == true) {
				update();
			}
			display();
		}
	}
	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam) {

	void uninitialize(void);
	void ToggleFullScreen(void);
	void resize(int, int);

	switch (iMsg) {
	case WM_SETFOCUS:
		bActiveWindow = true;
		break;
	case WM_KILLFOCUS:
		bActiveWindow = false;
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

		case 'L':
		case 'l':
			if (bLight == false)
				bLight = true;
			else
				bLight = false;
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

	MONITORINFO mi;

	if (bIsFullScreen == false) {
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		mi = { sizeof(MONITORINFO) };
		if (dwStyle & WS_OVERLAPPEDWINDOW) {
			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi)) {
				SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd,
					HWND_TOP,
					mi.rcMonitor.left,
					mi.rcMonitor.top,
					(mi.rcMonitor.right - mi.rcMonitor.left),
					(mi.rcMonitor.bottom - mi.rcMonitor.top),
					SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
		bIsFullScreen = true;
	}
	else {
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd,
			HWND_TOP,
			0, 0, 0, 0,
			SWP_NOZORDER | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
		bIsFullScreen = false;
	}
}

int initialize(void) {

	void resize(int, int);
	void uninitialize(void);
	GLuint LoadCubeMapTexture();

	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;
	GLenum Result;

	//Shader Object;
	GLint iVertexShaderObject;
	GLint iFragmentShaderObject;


	memset(&pfd, NULL, sizeof(PIXELFORMATDESCRIPTOR));

	ghdc = GetDC(ghwnd);

	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER | PFD_DRAW_TO_WINDOW;
	pfd.iPixelType = PFD_TYPE_RGBA;

	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;

	pfd.cDepthBits = 32;

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0)
		return(-1);

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
		return(-2);

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
		return(-3);

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
		return(-4);

	Result = glewInit();
	if (Result != GLEW_OK) {
		fprintf(gbFile, "glewInit() Failed!!\n");
		uninitialize();
		DestroyWindow(ghwnd);
		exit(1);
	}

	/********** Vertex Shader **********/
	iVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *szVertexShaderSourceCode =
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

				
		"uniform int u_CubeMapToggle;" \

		"out vec4 eyeCoord_VS;" \
		"out vec3 Normal_VS;" \
		"out vec3 Viewer_VS;" \
		"out vec3 outTexCoord_VS;" \






		"void main(void)" \
		"{" \

			"if(u_CubeMapToggle == 1) {" \

				"outTexCoord_VS = vPosition.xyz;" \
				//"outTexCoord_VS.y = -outTexCoord_VS.y;" \

				"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \

			"}" \

			"else {" \

				"eyeCoord_VS = u_view_matrix * u_model_matrix * vPosition;" \

				"lightDirectionRed_VS = vec3(u_Red_light_position - eyeCoord_VS);" \
				"lightDirectionGreen_VS = vec3(u_Green_light_position - eyeCoord_VS);" \
				"lightDirectionBlue_VS = vec3(u_Blue_light_position - eyeCoord_VS);" \
				
				"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" \
				"Normal_VS = vec3(normalMatrix * vNormals);" \
				"Viewer_VS = vec3(-eyeCoord_VS);" \


				"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \

			"}" \

		"}";

	glShaderSource(iVertexShaderObject, 1,
		(const GLchar**)&szVertexShaderSourceCode, NULL);

	glCompileShader(iVertexShaderObject);

	GLint iShaderCompileStatus;
	GLint iInfoLogLength;
	GLchar *szInfoLog = NULL;
	glGetShaderiv(iVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE) {
		glGetShaderiv(iVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetShaderInfoLog(iVertexShaderObject, iInfoLogLength,
					&written, szInfoLog);
				fprintf(gbFile, "Vertex Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				DestroyWindow(ghwnd);
				exit(0);
			}
		}
	}

	/********** Fragment Shader **********/
	iFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *szFragmentShaderSourceCode =
		"#version 450 core" \
		"\n" \

		"out vec4 FragColor;" \


		"in vec3 outTexCoord;" \
		"uniform samplerCube u_sampler;" \
		"in vec3 outTexCoord_VS;" \
		"uniform int u_CubeMapToggle;" \

		"in vec3 lightDirectionRed_VS;" \
		"in vec3 lightDirectionGreen_VS;" \
		"in vec3 lightDirectionBlue_VS;" \

		"in vec3 Normal_VS;" \
		"in vec4 eyeCoord_VS;" \
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

		"void main(void)" \
		"{" \
		
			"if(u_CubeMapToggle == 1){" \

				"FragColor = texture(u_sampler, outTexCoord_VS);" \

			"}" \

			"else{" \


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

				
				"vec3 r = reflect(eyeCoord_VS.xyz, normalize(Normal_VS));" \
				//"r.y = -r.y;" \
				
				"vec4 envCol = texture(u_sampler, r);" \
				"vec4 color = mix(envCol, vec4(phongLight,1.0f), 0.5);" \

				"FragColor = color;" \
 
			"}" \


		"}";

	glShaderSource(iFragmentShaderObject, 1,
		(const GLchar**)&szFragmentShaderSourceCode, NULL);

	glCompileShader(iFragmentShaderObject);

	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(iFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE) {
		glGetShaderiv(iFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetShaderInfoLog(iFragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gbFile, "Fragment Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				DestroyWindow(ghwnd);
				exit(0);
			}
		}
	}

	/********** Shader Program Object **********/
	gShaderProgramObject = glCreateProgram();

	glAttachShader(gShaderProgramObject, iVertexShaderObject);
	glAttachShader(gShaderProgramObject, iFragmentShaderObject);

	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_TEXCOORD0, "vTexCoord");
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_NORMAL, "vNormals");

	glLinkProgram(gShaderProgramObject);

	GLint iProgramLinkingStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	glGetProgramiv(gShaderProgramObject, GL_LINK_STATUS, &iProgramLinkingStatus);
	if (iProgramLinkingStatus == GL_FALSE) {
		glGetProgramiv(gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gbFile, "Shader Program Object Linking Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				DestroyWindow(ghwnd);
				exit(0);
			}
		}
	}

	modelMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_model_matrix");
	viewMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_view_matrix");
	projectionMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_projection_matrix");
	
	red_LaUniform = glGetUniformLocation(gShaderProgramObject, "u_Red_La");
	red_LdUniform = glGetUniformLocation(gShaderProgramObject, "u_Red_Ld");
	red_LsUniform = glGetUniformLocation(gShaderProgramObject, "u_Red_Ls");
	red_lightPositionUniform = glGetUniformLocation(gShaderProgramObject, "u_Red_light_position");

	green_LaUniform = glGetUniformLocation(gShaderProgramObject, "u_Green_La");
	green_LdUniform = glGetUniformLocation(gShaderProgramObject, "u_Green_Ld");
	green_LsUniform = glGetUniformLocation(gShaderProgramObject, "u_Green_Ls");
	green_lightPositionUniform = glGetUniformLocation(gShaderProgramObject, "u_Green_light_position");

	blue_LaUniform = glGetUniformLocation(gShaderProgramObject, "u_Blue_La");
	blue_LdUniform = glGetUniformLocation(gShaderProgramObject, "u_Blue_Ld");
	blue_LsUniform = glGetUniformLocation(gShaderProgramObject, "u_Blue_Ls");
	blue_lightPositionUniform = glGetUniformLocation(gShaderProgramObject, "u_Blue_light_position");

	KaUniform = glGetUniformLocation(gShaderProgramObject, "u_Ka");
	KdUniform = glGetUniformLocation(gShaderProgramObject, "u_Kd");
	KsUniform = glGetUniformLocation(gShaderProgramObject, "u_Ks");
	shininessUniform = glGetUniformLocation(gShaderProgramObject, "u_shininess");
	LKeyPressUniform = glGetUniformLocation(gShaderProgramObject, "u_L_keypress");

	samplerUniform = glGetUniformLocation(gShaderProgramObject, "u_sampler");

	cubeMapToggleUniform = glGetUniformLocation(gShaderProgramObject, "u_CubeMapToggle");



	/********** Positions **********/
	GLfloat Cube_Vertices[] = {
		//Top
		1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,
		//Bottom
		1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		//Front
		1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		//Back
		1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,
		//Right
		1.0f, 1.0f, -1.0f,
		1.0f, 1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f,
		//Left
		-1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f,


	};


	
	/********** Vao Cube **********/
	glGenVertexArrays(1, &vao_Cube);
	glBindVertexArray(vao_Cube);

	/******** Position **********/
	glGenBuffers(1, &vbo_Cube_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Cube_Position);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Cube_Vertices),
		Cube_Vertices,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	glBindVertexArray(0);


	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);




	

	/********** Position, Normal and Elements **********/
	getSphereVertexData(sphere_vertices, sphere_normals, sphere_textures, sphere_elements);
	gNumVertices = getNumberOfSphereVertices();
	gNumElements = getNumberOfSphereElements();



	/********** Sphere Vao **********/
	glGenVertexArrays(1, &vao_Sphere);
	glBindVertexArray(vao_Sphere);

	/********** Position **********/
	glGenBuffers(1, &vbo_Sphere_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Sphere_Position);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(sphere_vertices),
		sphere_vertices,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/********** Normals **********/
	glGenBuffers(1, &vbo_Sphere_Normal);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Sphere_Normal);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(sphere_normals),
		sphere_normals,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	/********** Element Vbo **********/
	glGenBuffers(1, &vbo_Sphere_Element);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(sphere_elements), sphere_elements, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


	glBindVertexArray(0);




	// ***** Cube Map *****
	glEnable(GL_TEXTURE_CUBE_MAP);
	texture_CubeMap = LoadCubeMapTexture();

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	gPerspectiveProjectionMatrix = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}




GLuint LoadCubeMapTexture(void) {

	void uninitialize(void);

	
	GLuint texture = 0;
	HBITMAP hBitmap = NULL;
	BITMAP bmp;


	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_CUBE_MAP, texture);

	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

		

	// *** Positive X (Right Side) ***
	hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL), MAKEINTRESOURCE(ID_RIGHT), IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);
	if(hBitmap == NULL){
		fprintf(gbFile, "LoadCubeMapTexture: LoadImage() Failed for Right\n");
		uninitialize();
		exit(0);
	}
	else{
		GetObject(hBitmap, sizeof(BITMAP), &bmp);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 
			0, GL_RGB, 
			bmp.bmWidth, bmp.bmHeight, 0,
			GL_BGR_EXT, GL_UNSIGNED_BYTE, bmp.bmBits);
	}
	DeleteObject(hBitmap);
	hBitmap =  NULL;


	// *** Negative X (Left Side) ***
	hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL), MAKEINTRESOURCE(ID_LEFT), IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);
	if(hBitmap == NULL){
		fprintf(gbFile, "LoadCubeMapTexture: LoadImage() Failed for Left\n");
		uninitialize();
		exit(0);
	}
	else{
		GetObject(hBitmap, sizeof(BITMAP), &bmp);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 
			0, GL_RGB, 
			bmp.bmWidth, bmp.bmHeight, 0,
			GL_BGR_EXT, GL_UNSIGNED_BYTE, bmp.bmBits);
	}
	DeleteObject(hBitmap);
	hBitmap =  NULL;


	// *** Positive Y (Top Side) ***
	hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL), MAKEINTRESOURCE(ID_TOP), IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);
	if(hBitmap == NULL){
		fprintf(gbFile, "LoadCubeMapTexture: LoadImage() Failed for Bottom\n");
		uninitialize();
		exit(0);
	}
	else{
		GetObject(hBitmap, sizeof(BITMAP), &bmp);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 
			0, GL_RGB, 
			bmp.bmWidth, bmp.bmHeight, 0,
			GL_BGR_EXT, GL_UNSIGNED_BYTE, bmp.bmBits);
	}
	DeleteObject(hBitmap);
	hBitmap =  NULL;


	// *** Negative Y (Bottom Side) ***
	hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL), MAKEINTRESOURCE(ID_BOTTOM), IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);
	if(hBitmap == NULL){
		fprintf(gbFile, "LoadCubeMapTexture: LoadImage() Failed for Top\n");
		uninitialize();
		exit(0);
	}
	else{
		GetObject(hBitmap, sizeof(BITMAP), &bmp);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 
			0, GL_RGB, 
			bmp.bmWidth, bmp.bmHeight, 0,
			GL_BGR_EXT, GL_UNSIGNED_BYTE, bmp.bmBits);
	}
	DeleteObject(hBitmap);
	hBitmap =  NULL;


	// *** Positive Z (Front Side) ***
	hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL), MAKEINTRESOURCE(ID_FRONT), IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);
	if(hBitmap == NULL){
		fprintf(gbFile, "LoadCubeMapTexture: LoadImage() Failed for Front\n");
		uninitialize();
		exit(0);
	}
	else{
		GetObject(hBitmap, sizeof(BITMAP), &bmp);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 
			0, GL_RGB, 
			bmp.bmWidth, bmp.bmHeight, 0,
			GL_BGR_EXT, GL_UNSIGNED_BYTE, bmp.bmBits);
	}
	DeleteObject(hBitmap);
	hBitmap =  NULL;


	// *** Negative Z (Back Side) ***
	hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL), MAKEINTRESOURCE(ID_BACK), IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);
	if(hBitmap == NULL){
		fprintf(gbFile, "LoadCubeMapTexture: LoadImage() Failed for Back\n");
		uninitialize();
		exit(0);
	}
	else{
		GetObject(hBitmap, sizeof(BITMAP), &bmp);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 
			0, GL_RGB, 
			bmp.bmWidth, bmp.bmHeight, 0,
			GL_BGR_EXT, GL_UNSIGNED_BYTE, bmp.bmBits);
	}
	DeleteObject(hBitmap);
	hBitmap =  NULL;


	glGenerateMipmap(GL_TEXTURE_CUBE_MAP);


	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);


	return(texture);
}

void uninitialize(void) {

	if (bIsFullScreen == true) {
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd,
			HWND_TOP,
			0, 0, 0, 0,
			SWP_NOZORDER | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
		bIsFullScreen = false;
	}

	if (vbo_Sphere_Element) {
		glDeleteBuffers(1, &vbo_Sphere_Element);
		vbo_Sphere_Element = 0;
	}

	if (vbo_Sphere_Normal) {
		glDeleteBuffers(1, &vbo_Sphere_Normal);
		vbo_Sphere_Normal = 0;
	}

	if (vbo_Sphere_Position) {
		glDeleteBuffers(1, &vbo_Sphere_Position);
		vbo_Sphere_Position = 0;
	}

	if (vao_Sphere) {
		glDeleteVertexArrays(1, &vao_Sphere);
		vao_Sphere = 0;
	}


	if(texture_CubeMap){
		glDeleteTextures(1, &texture_CubeMap);
		texture_CubeMap = 0;
	}


	if (vbo_Cube_Texture) {
		glDeleteBuffers(1, &vbo_Cube_Texture);
		vbo_Cube_Texture = 0;
	}

	if (vbo_Cube_Position) {
		glDeleteBuffers(1, &vbo_Cube_Position);
		vbo_Cube_Position = 0;
	}

	if (vao_Cube) {
		glDeleteVertexArrays(1, &vao_Cube);
		vao_Cube = 0;
	}



	GLsizei ShaderCount;
	GLsizei ShaderNumber;

	if (gShaderProgramObject) {
		glUseProgram(gShaderProgramObject);

		glGetProgramiv(gShaderProgramObject, GL_ATTACHED_SHADERS, &ShaderCount);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount);
		if (pShader) {
			glGetAttachedShaders(gShaderProgramObject, ShaderCount,
				&ShaderCount, pShader);
			for (ShaderNumber = 0; ShaderNumber < ShaderCount; ShaderNumber++) {
				glDetachShader(gShaderProgramObject, pShader[ShaderNumber]);
				glDeleteShader(pShader[ShaderNumber]);
				pShader[ShaderNumber] = 0;
			}
			free(pShader);
			pShader = NULL;
		}
		glDeleteProgram(gShaderProgramObject);
		gShaderProgramObject = 0;
		glUseProgram(0);
	}

	if (bIsFullScreen == true) {
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd,
			HWND_TOP,
			0, 0, 0, 0,
			SWP_NOZORDER | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
		bIsFullScreen = false;
	}

	if (wglGetCurrentContext() == ghrc) {
		wglMakeCurrent(NULL, NULL);
	}

	if (ghrc) {
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}

	if (ghdc) {
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	if (gbFile) {
		fprintf(gbFile, "Log Close!!\n");
		fclose(gbFile);
		gbFile = NULL;
	}
}

void resize(int width, int height) {
	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	gPerspectiveProjectionMatrix = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void display(void) {

	void rotateRedLight(GLfloat);
	void rotateGreenLight(GLfloat);
	void rotateBlueLight(GLfloat);

	mat4 translateMatrix;
	mat4 scaleMatrix;
	mat4 rotateMatrix;
	mat4 modelMatrix;
	mat4 viewMatrix;
	

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(gShaderProgramObject);


	/********** Cube **********/
	translateMatrix = mat4::identity();
	rotateMatrix = mat4::identity();
	scaleMatrix = mat4::identity();
	modelMatrix = mat4::identity();
	viewMatrix = mat4::identity();


	translateMatrix = translate(0.0f, 0.0f, -6.0f);
	scaleMatrix = scale(10.9f, 10.9f,10.9f);
	//rotateMatrix = rotate(angle_Cube, 0.0f, 0.0f);
	//rotateMatrix = rotate(0.0f, angle_Cube, 0.0f);
	modelMatrix =  translateMatrix * scaleMatrix * rotateMatrix;
	

	glUniformMatrix4fv(modelMatrixUniform, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(viewMatrixUniform, 1, GL_FALSE, viewMatrix);
	glUniformMatrix4fv(projectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);
	glUniform1i(cubeMapToggleUniform, 1);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, texture_CubeMap);
	glUniform1i(samplerUniform, 0);

	glBindVertexArray(vao_Cube);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 20, 4);
	glBindVertexArray(0);




	// ********** Sphere **********
	translateMatrix = mat4::identity();
	rotateMatrix = mat4::identity();
	modelMatrix = mat4::identity();
	viewMatrix = mat4::identity();

	translateMatrix = translate(0.0f, 0.0f, -3.50f);
	//rotateMatrix = rotate(0.0f, angle_Cube, 0.0f);
	modelMatrix = modelMatrix * translateMatrix * rotateMatrix;

	glUniformMatrix4fv(modelMatrixUniform, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(viewMatrixUniform, 1, GL_FALSE, viewMatrix);
	glUniformMatrix4fv(projectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);
	glUniform1i(cubeMapToggleUniform, 0);


	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, texture_CubeMap);
	glUniform1i(samplerUniform, 0);


	if (bLight == true) {

		glUniform1i(LKeyPressUniform, 1);

		rotateRedLight(angle_red);
		rotateGreenLight(angle_green);
		rotateBlueLight(angle_blue);
		
		glUniform3fv(red_LaUniform, 1, lightAmbient_Red);
		glUniform3fv(red_LdUniform, 1, lightDiffuse_Red);
		glUniform3fv(red_LsUniform, 1, lightSpecular_Red);
		glUniform4fv(red_lightPositionUniform, 1, lightPosition_Red);

		glUniform3fv(green_LaUniform, 1, lightAmbient_Green);
		glUniform3fv(green_LdUniform, 1, lightDiffuse_Green);
		glUniform3fv(green_LsUniform, 1, lightSpecular_Green);
		glUniform4fv(green_lightPositionUniform, 1, lightPosition_Green);

		glUniform3fv(blue_LaUniform, 1, lightAmbient_Blue);
		glUniform3fv(blue_LdUniform, 1, lightDiffuse_Blue);
		glUniform3fv(blue_LsUniform, 1, lightSpecular_Blue);
		glUniform4fv(blue_lightPositionUniform, 1, lightPosition_Blue);


		glUniform3fv(KaUniform, 1, materialAmbient);
		glUniform3fv(KdUniform, 1, materialDiffuse);
		glUniform3fv(KsUniform, 1, materialSpecular);
		glUniform1f(shininessUniform, materialShininess);
	}
	else
		glUniform1i(LKeyPressUniform, 0);

	glBindVertexArray(vao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);



	glUseProgram(0);

	SwapBuffers(ghdc);
}



void rotateRedLight(float angle) {
	lightPosition_Red[1] = (float)(5.0f * sin(angle));
	lightPosition_Red[2] = (float)(5.0f * cos(angle));
}

void rotateGreenLight(float angle) {
	lightPosition_Green[0] = (float)(5.0f * sin(angle));
	lightPosition_Green[2] = (float)(5.0f * cos(angle));
}

void rotateBlueLight(float angle) {
	lightPosition_Blue[0] = (float)(5.0f * cos(angle));
	lightPosition_Blue[1] = (float)(5.0f * sin(angle));
}




void update() {
	angle_red = angle_red + 0.020f;
	angle_green = angle_green + 0.02f;
	angle_blue = angle_blue + 0.02f;

	if (angle_red > 360.0f)
		angle_red = 0.0f;

	if (angle_green > 360.0f)
		angle_green = 0.0f;

	if (angle_blue > 360.0f)
		angle_blue = 0.0f;

	angle_Cube = angle_Cube - 0.5f;

	if (angle_Cube < 0.0f)
		angle_Cube = 360.0f;
 }
