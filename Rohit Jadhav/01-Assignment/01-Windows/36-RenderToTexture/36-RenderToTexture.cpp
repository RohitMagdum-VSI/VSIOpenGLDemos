#include<Windows.h>
#include<stdio.h>
#include<gl/glew.h>
#include<GL/GL.h>
#include"vmath.h"
#include<assert.h>



#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")

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
bool bIsFullScreen_RRJ = false;
HWND ghwnd_RRJ = NULL;
WINDOWPLACEMENT wpPrev_RRJ = { sizeof(WINDOWPLACEMENT) };
DWORD dwStyle_RRJ;

//For SuperMan
bool bActiveWindow_RRJ = false;
HDC ghdc_RRJ = NULL;
HGLRC ghrc_RRJ = NULL;

//For Error
FILE *gbFile_RRJ = NULL;

//For Shader Program Object;
GLint gShaderProgramObject_RRJ;

//For Perspective Matric
mat4 gPerspectiveProjectionMatrix_RRJ;


//For Cube
GLuint texture_Cube_RRJ;
GLuint vao_Cube_RRJ;
GLuint vbo_Cube_Position_RRJ;
GLuint vbo_Cube_Texture_RRJ;
GLfloat angle_Cube_RRJ = 360.0f;

//For Uniforms
GLuint mvpUniform_RRJ;
GLuint samplerUniform_RRJ;
GLuint choiceUniform_RRJ;

//For Framebuffer
GLuint frameBufferObject_RRJ;
GLuint renderBufferObject_Depth_RRJ;	
GLint viewPortWidth_RRJ;
GLint viewPortHeight_RRJ;
GLint viewPort_FBO_Width_RRJ = 1024;
GLint viewPort_FBO_Height_RRJ = 1024;

//For Model
GLuint vao_Model_RRJ;
GLuint vbo_Model_Position_RRJ;
GLuint vbo_Model_Normal_RRJ;
GLuint vbo_Model_Element_RRJ;


//For Uniform
GLuint modelMatrixUniform_RRJ;
GLuint viewMatrixUniform_RRJ;
GLuint projectionMatrixUniform_RRJ;
GLuint La_Uniform_RRJ;
GLuint Ld_Uniform_RRJ;
GLuint Ls_Uniform_RRJ;
GLuint lightPositionUniform_RRJ;
GLuint Ka_Uniform_RRJ;
GLuint Kd_Uniform_RRJ;
GLuint Ks_Uniform_RRJ;
GLuint materialShininessUniform_RRJ;
GLuint LKeyPressUniform_RRJ;


//For Lights
bool bLights_RRJ = false;
GLfloat lightAmbient_RRJ[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat lightDiffuse_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat lightSpecular_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat lightPosition_RRJ[] = { 100.0f, 100.0f, 100.0f, 1.0f };

//For Material
GLfloat materialAmbient_RRJ[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat materialDiffuse_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialSpecular_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialShininess_RRJ = 128.0f;


//Model Loading
struct VecFloat {

	float *pData;
	int iSize;
};


#define RRJ_SUCCESS 1
#define RRJ_ERROR 0


struct VecFloat *pVecFloat_Model_Vertices = NULL;
struct VecFloat *pVecFloat_Model_Normals = NULL;
struct VecFloat *pVecFloat_Model_Texcoord = NULL;

struct VecFloat *pVecFloat_Model_Sorted_Vertices = NULL;
struct VecFloat *pVecFloat_Model_Sorted_Normals = NULL;
struct VecFloat *pVecFloat_Model_Sorted_Texcoord = NULL;
struct VecFloat *pVecFloat_Model_Elements = NULL;

int PushBackVecFloat(struct VecFloat*, float);
void ShowVecFloat(struct VecFloat*);
struct VecFloat* CreateVecFloat(void);
int DestroyVecFloat(struct VecFloat*);


FILE *gbFile_Model = NULL;
FILE *gbFile_Vertices = NULL;
FILE *gbFile_Normals = NULL;
FILE *gbFile_TexCoord = NULL;
FILE *gbFile_FaceIndices = NULL;


LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow) {

	if (fopen_s(&gbFile_RRJ, "Log.txt", "w") != 0) {
		MessageBox(NULL, TEXT("Log Creation Failed!!\n"), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
		fprintf_s(gbFile_RRJ, "Log Created!!\n");




	fopen_s(&gbFile_Model, "head.txt", "r");
	if (gbFile_Model == NULL) {
		MessageBox(NULL, TEXT("ERROR: Main(): Model File fopen() Failed!!\n"), TEXT("ERROR"), MB_OK);
		exit(0);
	}

	fopen_s(&gbFile_Vertices, "Vertices.txt", "w");
	if (gbFile_Vertices == NULL) {
		MessageBox(NULL, TEXT("ERROR: Vertices.txt Creation Failed!!\n"), TEXT("ERROR"), MB_OK);
		exit(0);
	}

	fopen_s(&gbFile_Normals, "Normals.txt", "w");
	if (gbFile_Normals == NULL) {
		MessageBox(NULL, TEXT("ERROR: Normals.txt Failed!!"), TEXT("ERROR"), MB_OK);
		exit(0);
	}

	fopen_s(&gbFile_TexCoord, "Texcoord.txt", "w");
	if (gbFile_TexCoord == NULL) {
		MessageBox(NULL, TEXT("ERROR: Texture.txt Failed!!"), TEXT("ERROR"), MB_OK);
		exit(0);
	}

	fopen_s(&gbFile_FaceIndices, "Face.txt", "w");
	if (gbFile_FaceIndices == NULL) {
		MessageBox(NULL, TEXT("ERROR: Face.txt"), TEXT("ERROR"), MB_OK);
		exit(0);
	}





	int initialize(void);
	void display(void);
	void update(void);
	void ToggleFullScreen(void);

	int iRet_RRJ;
	bool bDone_RRJ = false;

	WNDCLASSEX wndclass_RRJ;
	HWND hwnd;
	MSG msg_RRJ;
	TCHAR szName_RRJ[] = TEXT("RohitRJadhav-PP-RenderToTexture");

	wndclass_RRJ.lpszClassName = szName_RRJ;
	wndclass_RRJ.lpszMenuName = NULL;
	wndclass_RRJ.lpfnWndProc = WndProc;

	wndclass_RRJ.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass_RRJ.cbSize = sizeof(WNDCLASSEX);
	wndclass_RRJ.cbWndExtra = 0;
	wndclass_RRJ.cbClsExtra = 0;

	wndclass_RRJ.hInstance = hInstance;
	wndclass_RRJ.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass_RRJ.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass_RRJ.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass_RRJ.hCursor = LoadCursor(NULL, IDC_ARROW);

	RegisterClassEx(&wndclass_RRJ);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szName_RRJ,
		TEXT("RohitRJadhav-PP-RenderToTexture"),
		WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_VISIBLE,
		100, 100,
		WIN_WIDTH, WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghwnd_RRJ = hwnd;

	SetForegroundWindow(hwnd);
	SetFocus(hwnd);
	ShowWindow(hwnd, iCmdShow);


	iRet_RRJ = initialize();
	if (iRet_RRJ == -1) {
		fprintf_s(gbFile_RRJ, "ChoosePixelFormat() Failed!!\n");
		DestroyWindow(hwnd);
	}
	else if (iRet_RRJ == -2) {
		fprintf_s(gbFile_RRJ, "SetPixelFormat() Failed!!\n");
		DestroyWindow(hwnd);
	}
	else if (iRet_RRJ == -3) {
		fprintf_s(gbFile_RRJ, "wglMakeCurrent() Failed!!\n");
		DestroyWindow(hwnd);
	}
	else if (iRet_RRJ == -4) {
		fprintf_s(gbFile_RRJ, "wglMakeCurrent() Failed!!\n");
		DestroyWindow(hwnd);
	}
	else
		fprintf_s(gbFile_RRJ, "initialize() done!!\n");



	
	//ToggleFullScreen();

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

	void uninitialize(void);
	void ToggleFullScreen(void);
	void resize(int, int);

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
			DestroyWindow(hwnd);
			break;

		case 'F':
		case 'f':
			ToggleFullScreen();
			break;


		case 'L':
		case 'l':
			if(bLights_RRJ == false)
				bLights_RRJ = true;
			else
				bLights_RRJ = false;
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

	if (bIsFullScreen_RRJ == false) {
		dwStyle_RRJ = GetWindowLong(ghwnd_RRJ, GWL_STYLE);
		mi = { sizeof(MONITORINFO) };
		if (dwStyle_RRJ & WS_OVERLAPPEDWINDOW) {
			if (GetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ) && GetMonitorInfo(MonitorFromWindow(ghwnd_RRJ, MONITORINFOF_PRIMARY), &mi)) {
				SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd_RRJ,
					HWND_TOP,
					mi.rcMonitor.left,
					mi.rcMonitor.top,
					(mi.rcMonitor.right - mi.rcMonitor.left),
					(mi.rcMonitor.bottom - mi.rcMonitor.top),
					SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
		bIsFullScreen_RRJ = true;
	}
	else {
		SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ);
		SetWindowPos(ghwnd_RRJ,
			HWND_TOP,
			0, 0, 0, 0,
			SWP_NOZORDER | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
		bIsFullScreen_RRJ = false;
	}
}

int initialize(void) {

	void resize(int, int);
	void uninitialize(void);
	GLuint loadTexture(void);
	void LoadModel(void);

	PIXELFORMATDESCRIPTOR pfd_RRJ;
	int iPixelFormatIndex_RRJ;
	GLenum Result;

	//Shader Object;
	GLint iVertexShaderObject_RRJ;
	GLint iFragmentShaderObject_RRJ;


	memset(&pfd_RRJ, NULL, sizeof(PIXELFORMATDESCRIPTOR));

	ghdc_RRJ = GetDC(ghwnd_RRJ);

	pfd_RRJ.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd_RRJ.nVersion = 1;
	pfd_RRJ.dwFlags = PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER | PFD_DRAW_TO_WINDOW;
	pfd_RRJ.iPixelType = PFD_TYPE_RGBA;

	pfd_RRJ.cColorBits = 32;
	pfd_RRJ.cRedBits = 8;
	pfd_RRJ.cGreenBits = 8;
	pfd_RRJ.cBlueBits = 8;
	pfd_RRJ.cAlphaBits = 8;

	pfd_RRJ.cDepthBits = 32;

	iPixelFormatIndex_RRJ = ChoosePixelFormat(ghdc_RRJ, &pfd_RRJ);
	if (iPixelFormatIndex_RRJ == 0)
		return(-1);

	if (SetPixelFormat(ghdc_RRJ, iPixelFormatIndex_RRJ, &pfd_RRJ) == FALSE)
		return(-2);

	ghrc_RRJ = wglCreateContext(ghdc_RRJ);
	if (ghrc_RRJ == NULL)
		return(-3);

	if (wglMakeCurrent(ghdc_RRJ, ghrc_RRJ) == FALSE)
		return(-4);

	Result = glewInit();
	if (Result != GLEW_OK) {
		fprintf_s(gbFile_RRJ, "glewInit() Failed!!\n");
		uninitialize();
		DestroyWindow(ghwnd_RRJ);
		exit(1);
	}

	/********** Vertex Shader **********/
	iVertexShaderObject_RRJ = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *szVertexShaderSourceCode_RRJ =
		"#version 450 core" \
		"\n" \

		//For Choice
		"uniform int u_choice;" \

		//For Attribute
		"in vec4 vPosition;" \
		"in vec3 vNormal;" \
		"in vec2 vTexCoord;" \
		"out vec2 outTexCoord;" \

		//For Cube
		"uniform mat4 u_mvp_matrix;" \

		//For Model and Light
		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"uniform vec4 u_light_position;" \
		"out vec3 viewer_vector_VS;" \
		"out vec3 tNorm_VS;" \
		"out vec3 lightDirection_VS;" \


		"void main(void)" \
		"{" \
			//For Cube	
			"if(u_choice == 1) { " \

				"gl_Position = u_mvp_matrix * vPosition;" \
				"outTexCoord = vTexCoord;" \

			"}" \

			//For Model
			"else if(u_choice == 2) {" \

				"vec4 eye_coordinate = u_view_matrix * u_model_matrix * vPosition;" \
				"viewer_vector_VS = vec3(-eye_coordinate);" \
				"tNorm_VS = mat3(u_view_matrix * u_model_matrix) * vNormal;" \
				"lightDirection_VS = vec3(u_light_position - eye_coordinate);" \
				"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \

			"}" \
		"}";

	glShaderSource(iVertexShaderObject_RRJ, 1,
		(const GLchar**)&szVertexShaderSourceCode_RRJ, NULL);

	glCompileShader(iVertexShaderObject_RRJ);

	GLint iShaderCompileStatus_RRJ;
	GLint iInfoLogLength_RRJ;
	GLchar *szInfoLog_RRJ = NULL;
	glGetShaderiv(iVertexShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(iVertexShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(iVertexShaderObject_RRJ, iInfoLogLength_RRJ,
					&written, szInfoLog_RRJ);
				fprintf_s(gbFile_RRJ, "Vertex Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}

	/********** Fragment Shader **********/
	iFragmentShaderObject_RRJ = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *szFragmentShaderSourceCode_RRJ =
		"#version 450 core" \
		"\n" \

		//For Choice
		"uniform int u_choice;" \
		
		//For Cube
		"in vec2 outTexCoord;" \
		"uniform sampler2D u_sampler;" \


		//For Model
		"in vec3 viewer_vector_VS;" \
		"in vec3 tNorm_VS;" \
		"in vec3 lightDirection_VS;" \
		"uniform vec3 u_La;" \
		"uniform vec3 u_Ld;" \
		"uniform vec3 u_Ls;" \
		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_shininess;" \
		"uniform int u_LKeyPress;" \
		
		//For Output
		"out vec4 FragColor;" \

		"void main(void)" \
		"{" \

			//For Cube
			"if(u_choice == 1) { " \
					
				"FragColor = texture(u_sampler, outTexCoord);" \

			"}" \

			//For Model
			"else if (u_choice == 2){" \
				
				"if(u_LKeyPress == 1){" \
					"vec3 normalize_viewer_vector = normalize(viewer_vector_VS);" \
					"vec3 normalize_tNorm = normalize(tNorm_VS);" \
					"vec3 normalize_lightDirection = normalize(lightDirection_VS);" \
					"vec3 reflection_vector = reflect(-normalize_lightDirection, normalize_tNorm);" \
					"float s_dot_n = max(dot(normalize_lightDirection, normalize_tNorm), 0.0);" \
					"float r_dot_v = max(dot(reflection_vector, normalize_viewer_vector), 0.0);" \
					"vec3 ambient = u_La * u_Ka;" \
					"vec3 diffuse = u_Ld * u_Kd * s_dot_n;" \
					"vec3 specular = u_Ls * u_Ks * pow(r_dot_v, u_shininess);" \
					"vec3 Phong_ADS_Light = ambient + diffuse + specular;" \
					"FragColor = vec4(Phong_ADS_Light, 1.0);" \
				"}" \
				"else{" \
					"FragColor = vec4(1.0, 1.0, 1.0, 1.0);" \
				"}" \

			"}" \
		"}";

	glShaderSource(iFragmentShaderObject_RRJ, 1,
		(const GLchar**)&szFragmentShaderSourceCode_RRJ, NULL);

	glCompileShader(iFragmentShaderObject_RRJ);

	iShaderCompileStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(iFragmentShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(iFragmentShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(iFragmentShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf_s(gbFile_RRJ, "Fragment Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}

	/********** Shader Program Object **********/
	gShaderProgramObject_RRJ = glCreateProgram();

	glAttachShader(gShaderProgramObject_RRJ, iVertexShaderObject_RRJ);
	glAttachShader(gShaderProgramObject_RRJ, iFragmentShaderObject_RRJ);

	glBindAttribLocation(gShaderProgramObject_RRJ, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(gShaderProgramObject_RRJ, AMC_ATTRIBUTE_TEXCOORD0, "vTexCoord");
	glBindAttribLocation(gShaderProgramObject_RRJ, AMC_ATTRIBUTE_NORMAL, "vNormal");

	glLinkProgram(gShaderProgramObject_RRJ);

	GLint iProgramLinkingStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;
	glGetProgramiv(gShaderProgramObject_RRJ, GL_LINK_STATUS, &iProgramLinkingStatus_RRJ);
	if (iProgramLinkingStatus_RRJ == GL_FALSE) {
		glGetProgramiv(gShaderProgramObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf_s(gbFile_RRJ, "Shader Program Object Linking Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}

	
	//For Cube
	mvpUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_mvp_matrix");
	samplerUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_sampler");
	choiceUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ , "u_choice");

	//For Model
	modelMatrixUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_model_matrix");
	viewMatrixUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_view_matrix");
	projectionMatrixUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_projection_matrix");
	La_Uniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_La");
	Ld_Uniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_Ld");
	Ls_Uniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_Ls");
	lightPositionUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_light_position");
	Ka_Uniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_Ka");
	Kd_Uniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_Kd");
	Ks_Uniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_Ks");
	materialShininessUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_shininess");
	LKeyPressUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_LKeyPress");



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
		-1.0f, -1.0f, 1.0f
	};


	/************ TexCoord **********/
	GLfloat Cube_TexCoord[] = {
		//Top
		1.0f, 1.0,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//Back
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//Face
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//Back
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//Right
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//Left
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};



	/********** Vao Cube **********/
	glGenVertexArrays(1, &vao_Cube_RRJ);
	glBindVertexArray(vao_Cube_RRJ);

		/******** Position **********/
		glGenBuffers(1, &vbo_Cube_Position_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Cube_Position_RRJ);
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


		/********** Texture ***********/
		glGenBuffers(1, &vbo_Cube_Texture_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Cube_Texture_RRJ);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(Cube_TexCoord),
			Cube_TexCoord,
			GL_STATIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOORD0,
			2,
			GL_FLOAT,
			GL_FALSE,
			0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOORD0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);




	/********** Position, Normal and Elements **********/
	pVecFloat_Model_Vertices = CreateVecFloat();
	pVecFloat_Model_Normals = CreateVecFloat();
	pVecFloat_Model_Texcoord = CreateVecFloat();

	pVecFloat_Model_Elements = CreateVecFloat();

	pVecFloat_Model_Sorted_Vertices = CreateVecFloat();
	pVecFloat_Model_Sorted_Normals = CreateVecFloat();
	pVecFloat_Model_Sorted_Texcoord = CreateVecFloat();

	LoadModel();


	/********** Model Vao **********/
	glGenVertexArrays(1, &vao_Model_RRJ);
	glBindVertexArray(vao_Model_RRJ);

		/********** Position **********/
		glGenBuffers(1, &vbo_Model_Position_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Model_Position_RRJ);
		glBufferData(GL_ARRAY_BUFFER,
			pVecFloat_Model_Sorted_Vertices->iSize * sizeof(float),
			pVecFloat_Model_Sorted_Vertices->pData,
			GL_STATIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
			3,
			GL_FLOAT,
			GL_FALSE,
			0, NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);



		/********** Normals **********/
		glGenBuffers(1, &vbo_Model_Normal_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Model_Normal_RRJ);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(float) * pVecFloat_Model_Sorted_Normals->iSize,
			pVecFloat_Model_Sorted_Normals->pData,
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



	/********** FRAMEBUFFER **********/
	glGenFramebuffers(1, &frameBufferObject_RRJ);
	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferObject_RRJ);

		/********** Texture **********/
		glGenTextures(1, &texture_Cube_RRJ);
		glBindTexture(GL_TEXTURE_2D, texture_Cube_RRJ);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, 
			GL_RGBA, 
			viewPort_FBO_Width_RRJ, viewPort_FBO_Width_RRJ, 0, 
			GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_Cube_RRJ, 0);


		/********** For Depth **********/
		glGenRenderbuffers(1, &renderBufferObject_Depth_RRJ);
		glBindRenderbuffer(GL_RENDERBUFFER, renderBufferObject_Depth_RRJ);	
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, viewPort_FBO_Width_RRJ, viewPort_FBO_Width_RRJ);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, renderBufferObject_Depth_RRJ);




		/********** Checking *********/
		if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
			fprintf_s(gbFile_RRJ, "ERROR: glCheckFramebufferStatus\n");
			uninitialize();
			DestroyWindow(ghwnd_RRJ);
		}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);


	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	gPerspectiveProjectionMatrix_RRJ = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}



void uninitialize(void) {

	if(renderBufferObject_Depth_RRJ){
		glDeleteRenderbuffers(1, &renderBufferObject_Depth_RRJ);
		renderBufferObject_Depth_RRJ = 0;
	}

	if(texture_Cube_RRJ){
		glDeleteTextures(1, &texture_Cube_RRJ);
		texture_Cube_RRJ = 0;
	}

	if(frameBufferObject_RRJ){
		glDeleteFramebuffers(1, &frameBufferObject_RRJ);
		frameBufferObject_RRJ = 0;
	}



	if (pVecFloat_Model_Sorted_Texcoord) {
		DestroyVecFloat(pVecFloat_Model_Sorted_Texcoord);
		pVecFloat_Model_Sorted_Texcoord = NULL;
	}

	if (pVecFloat_Model_Sorted_Normals) {
		DestroyVecFloat(pVecFloat_Model_Sorted_Normals);
		pVecFloat_Model_Sorted_Normals = NULL;
	}


	if (pVecFloat_Model_Sorted_Vertices) {
		DestroyVecFloat(pVecFloat_Model_Sorted_Vertices);
		pVecFloat_Model_Sorted_Vertices = NULL;
	}


	if (pVecFloat_Model_Normals) {
		DestroyVecFloat(pVecFloat_Model_Normals);
		pVecFloat_Model_Normals = NULL;
	}

	if (pVecFloat_Model_Texcoord) {
		DestroyVecFloat(pVecFloat_Model_Texcoord);
		pVecFloat_Model_Texcoord = NULL;
	}

	if (pVecFloat_Model_Vertices) {
		DestroyVecFloat(pVecFloat_Model_Vertices);
		pVecFloat_Model_Vertices = NULL;
	}


	if (pVecFloat_Model_Elements) {
		DestroyVecFloat(pVecFloat_Model_Elements);
		pVecFloat_Model_Elements = NULL;
	}



	if (vbo_Model_Element_RRJ) {
		glDeleteBuffers(1, &vbo_Model_Element_RRJ);
		vbo_Model_Element_RRJ = 0;
	}

	if (vbo_Model_Normal_RRJ) {
		glDeleteBuffers(1, &vbo_Model_Normal_RRJ);
		vbo_Model_Normal_RRJ = 0;
	}

	if (vbo_Model_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Model_Position_RRJ);
		vbo_Model_Position_RRJ = 0;
	}

	if (vao_Model_RRJ) {
		glDeleteVertexArrays(1, &vao_Model_RRJ);
		vao_Model_RRJ = 0;
	}	




	if (vbo_Cube_Texture_RRJ) {
		glDeleteBuffers(1, &vbo_Cube_Texture_RRJ);
		vbo_Cube_Texture_RRJ = 0;
	}

	if (vbo_Cube_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Cube_Position_RRJ);
		vbo_Cube_Position_RRJ = 0;
	}

	if (vao_Cube_RRJ) {
		glDeleteVertexArrays(1, &vao_Cube_RRJ);
		vao_Cube_RRJ = 0;
	}


	GLsizei ShaderCount;
	GLsizei ShaderNumber;

	if (gShaderProgramObject_RRJ) {
		glUseProgram(gShaderProgramObject_RRJ);

		glGetProgramiv(gShaderProgramObject_RRJ, GL_ATTACHED_SHADERS, &ShaderCount);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount);
		if (pShader) {
			glGetAttachedShaders(gShaderProgramObject_RRJ, ShaderCount,
				&ShaderCount, pShader);
			for (ShaderNumber = 0; ShaderNumber < ShaderCount; ShaderNumber++) {
				glDetachShader(gShaderProgramObject_RRJ, pShader[ShaderNumber]);
				glDeleteShader(pShader[ShaderNumber]);
				pShader[ShaderNumber] = 0;
			}
			free(pShader);
			pShader = NULL;
		}
		glDeleteProgram(gShaderProgramObject_RRJ);
		gShaderProgramObject_RRJ = 0;
		glUseProgram(0);
	}

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
		fprintf_s(gbFile_RRJ, "Log Close!!\n");
		fclose(gbFile_RRJ);
		gbFile_RRJ = NULL;
	}
}




void resize(int width, int height) {
	if (height == 0)
		height = 1;

	viewPortWidth_RRJ = width;
	viewPortHeight_RRJ = height;

}

void display(void) {

	mat4 translateMatrix_RRJ;
	mat4 rotateMatrix_RRJ;
	mat4 modelMatrix_RRJ;
	mat4 viewMatrix_RRJ;
	mat4 modelViewProjectionMatrix_RRJ;

	static GLfloat angle_Model_RRJ = 0.0f;


	//For Model As a Texture

	glViewport(0, 0, (GLsizei)viewPort_FBO_Width_RRJ, (GLsizei)viewPort_FBO_Width_RRJ);
	gPerspectiveProjectionMatrix_RRJ = mat4::identity();
	gPerspectiveProjectionMatrix_RRJ = perspective(45.0f, (float)viewPort_FBO_Width_RRJ / (GLfloat)viewPort_FBO_Width_RRJ, 0.1f, 100.0f);

	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferObject_RRJ);
		glClearColor(0.0f, 0.0f, 0.10f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		glUseProgram(gShaderProgramObject_RRJ);



		/********** Model **********/
		translateMatrix_RRJ = mat4::identity();
		rotateMatrix_RRJ = mat4::identity();
		modelMatrix_RRJ = mat4::identity();
		viewMatrix_RRJ = mat4::identity();


		translateMatrix_RRJ = translate(0.0f, 0.0f, -8.0f);
		rotateMatrix_RRJ = rotate(0.0f, angle_Model_RRJ, 0.0f);
		modelMatrix_RRJ = modelMatrix_RRJ * translateMatrix_RRJ * rotateMatrix_RRJ;

		//For Shader Toggle
		glUniform1i(choiceUniform_RRJ, 2);


		glUniformMatrix4fv(modelMatrixUniform_RRJ, 1, GL_FALSE, modelMatrix_RRJ);
		glUniformMatrix4fv(viewMatrixUniform_RRJ, 1, GL_FALSE, viewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);

		if (bLights_RRJ == true) {
			glUniform1i(LKeyPressUniform_RRJ, 1);


			glUniform3fv(La_Uniform_RRJ, 1, lightAmbient_RRJ);
			glUniform3fv(Ld_Uniform_RRJ, 1, lightDiffuse_RRJ);
			glUniform3fv(Ls_Uniform_RRJ, 1, lightSpecular_RRJ);
			glUniform4fv(lightPositionUniform_RRJ, 1, lightPosition_RRJ);

			glUniform3fv(Ka_Uniform_RRJ, 1, materialAmbient_RRJ);
			glUniform3fv(Kd_Uniform_RRJ, 1, materialDiffuse_RRJ);
			glUniform3fv(Ks_Uniform_RRJ, 1, materialSpecular_RRJ);
			glUniform1f(materialShininessUniform_RRJ, materialShininess_RRJ);
		}
		else
			glUniform1i(LKeyPressUniform_RRJ, 0);


		glBindVertexArray(vao_Model_RRJ);
			glDrawArrays(GL_TRIANGLES, 0, pVecFloat_Model_Sorted_Vertices->iSize / 3);
		glBindVertexArray(0);

		glUseProgram(0);
		

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	angle_Model_RRJ = angle_Model_RRJ + 1.0f;
	if(angle_Model_RRJ > 360.0f)
		angle_Model_RRJ = 0.0f;


	//For Cube

	glViewport(0, 0, (GLsizei)viewPortWidth_RRJ, (GLsizei)viewPortHeight_RRJ);
	gPerspectiveProjectionMatrix_RRJ = mat4::identity();
	gPerspectiveProjectionMatrix_RRJ = perspective(45.0f, (float)viewPortWidth_RRJ / (float)viewPortHeight_RRJ, 0.1f, 100.0f);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(gShaderProgramObject_RRJ);

	/********** Cube **********/
	translateMatrix_RRJ = mat4::identity();
	rotateMatrix_RRJ = mat4::identity();
	modelMatrix_RRJ = mat4::identity();
	modelViewProjectionMatrix_RRJ = mat4::identity();


	translateMatrix_RRJ = translate(0.0f, 0.0f, -6.0f);
	rotateMatrix_RRJ = rotate(angle_Cube_RRJ, 0.0f, 0.0f) * rotate(0.0f, angle_Cube_RRJ, 0.0f) * rotate(0.0f, 0.0f, angle_Cube_RRJ);
	modelMatrix_RRJ = modelMatrix_RRJ * translateMatrix_RRJ * rotateMatrix_RRJ;
	modelViewProjectionMatrix_RRJ = gPerspectiveProjectionMatrix_RRJ * viewMatrix_RRJ * modelMatrix_RRJ;

	glUniformMatrix4fv(mvpUniform_RRJ,
		1,
		GL_FALSE,
		modelViewProjectionMatrix_RRJ);


	//For Shader Toggle
	glUniform1i(choiceUniform_RRJ, 1);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture_Cube_RRJ);
	glUniform1i(samplerUniform_RRJ, 0);

	glBindVertexArray(vao_Cube_RRJ);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
		glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
		glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
		glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
		glDrawArrays(GL_TRIANGLE_FAN, 20, 4);
	glBindVertexArray(0);

	glUseProgram(0);


	SwapBuffers(ghdc_RRJ);
}


void update(void) {

	angle_Cube_RRJ = angle_Cube_RRJ - 1.0f;
	if (angle_Cube_RRJ < 0.0f)
		angle_Cube_RRJ = 360.0f;
}



struct VecFloat* CreateVecFloat(void) {

	struct VecFloat *pTemp = NULL;

	pTemp = (struct VecFloat*)malloc(sizeof(struct VecFloat));
	if (pTemp == NULL) {
		fprintf_s(gbFile_RRJ, "ERROR: CreateVecInt(): Malloc() Failed!\n");
		exit(0);
	}

	memset((void*)pTemp, 0, sizeof(struct VecFloat));

	return(pTemp);
}


int PushBackVecFloat(struct VecFloat *pVec, float data) {

	pVec->pData = (float*)realloc(pVec->pData, sizeof(struct VecFloat) * (pVec->iSize + 1));

	assert(pVec->pData);

	pVec->iSize = pVec->iSize + 1;
	pVec->pData[pVec->iSize - 1] = data;
	//fprintf_s(gbFile_RRJ, "iSize: %d   iData: %f\n", pVec->iSize, pVec->pData[pVec->iSize - 1]);

	return(RRJ_SUCCESS);
}


void ShowVecFloat(struct VecFloat *pVec) {

	for (int i = 0; i < pVec->iSize; i++)
		fprintf_s(gbFile_RRJ, "P[%d]: %f\t", i, pVec->pData[i]);
}


int DestroyVecFloat(struct VecFloat *pVec) {


	free(pVec->pData);
	pVec->pData = NULL;
	pVec->iSize = 0;
	free(pVec);
	pVec = NULL;

	return(RRJ_SUCCESS);
}



void LoadModel(void) {

	char buffer[1024];
	char *firstToken = NULL;
	char *My_Strtok(char*, char);
	const char *space = " ";
	char *cContext = NULL;


	while (fgets(buffer, 1024, gbFile_Model) != NULL) {

		firstToken = strtok_s(buffer, space, &cContext);

		if (strcmp(firstToken, "v") == 0) {
			//Vertices
			float x, y, z;
			x = (float)atof(strtok_s(NULL, space, &cContext));
			y = (float)atof(strtok_s(NULL, space, &cContext));
			z = (float)atof(strtok_s(NULL, space, &cContext));

			//fprintf_s(gbFile_Vertices, "%f/%f/%f\n", x, y, z);

			PushBackVecFloat(pVecFloat_Model_Vertices, x);
			PushBackVecFloat(pVecFloat_Model_Vertices, y);
			PushBackVecFloat(pVecFloat_Model_Vertices, z);

		}
		else if (strcmp(firstToken, "vt") == 0) {
			//Texture

			float u, v;
			u = (float)atof(strtok_s(NULL, space, &cContext));
			v = (float)atof(strtok_s(NULL, space, &cContext));

			//fprintf_s(gbFile_TexCoord, "%f/%f\n", u, v);
			PushBackVecFloat(pVecFloat_Model_Texcoord, u);
			PushBackVecFloat(pVecFloat_Model_Texcoord, v);
		}
		else if (strcmp(firstToken, "vn") == 0) {
			//Normals

			float x, y, z;
			x = (float)atof(strtok_s(NULL, space, &cContext));
			y = (float)atof(strtok_s(NULL, space, &cContext));
			z = (float)atof(strtok_s(NULL, space, &cContext));

			//fprintf_s(gbFile_Normals, "%f/%f/%f\n", x, y, z);
			PushBackVecFloat(pVecFloat_Model_Normals, x);
			PushBackVecFloat(pVecFloat_Model_Normals, y);
			PushBackVecFloat(pVecFloat_Model_Normals, z);

		}
		else if (strcmp(firstToken, "f") == 0) {
			//Faces


			for (int i = 0; i < 3; i++) {

				char *faces = strtok_s(NULL, space, &cContext);
				int v, vt, vn;
				v = atoi(My_Strtok(faces, '/')) - 1;
				vt = atoi(My_Strtok(faces, '/')) - 1;
				vn = atoi(My_Strtok(faces, '/')) - 1;

				float x, y, z;

				//Sorted Vertices
				x = pVecFloat_Model_Vertices->pData[(v * 3) + 0];
				y = pVecFloat_Model_Vertices->pData[(v * 3) + 1];
				z = pVecFloat_Model_Vertices->pData[(v * 3) + 2];

				PushBackVecFloat(pVecFloat_Model_Sorted_Vertices, x);
				PushBackVecFloat(pVecFloat_Model_Sorted_Vertices, y);
				PushBackVecFloat(pVecFloat_Model_Sorted_Vertices, z);


				//Sorted Normals
				x = pVecFloat_Model_Normals->pData[(vn * 3) + 0];
				y = pVecFloat_Model_Normals->pData[(vn * 3) + 1];
				z = pVecFloat_Model_Normals->pData[(vn * 3) + 2];

				PushBackVecFloat(pVecFloat_Model_Sorted_Normals, x);
				PushBackVecFloat(pVecFloat_Model_Sorted_Normals, y);
				PushBackVecFloat(pVecFloat_Model_Sorted_Normals, z);


				//Sorted Texcoord;
				x = pVecFloat_Model_Texcoord->pData[(vt * 2) + 0];
				y = pVecFloat_Model_Texcoord->pData[(vt * 2) + 1];

				PushBackVecFloat(pVecFloat_Model_Sorted_Texcoord, x);
				PushBackVecFloat(pVecFloat_Model_Sorted_Texcoord, y);



				//Face Elements
				PushBackVecFloat(pVecFloat_Model_Elements, v);

				//fprintf_s(gbFile_FaceIndices, "%d/ %d/ %d     ", v, vt, vn);
			}
			//fprintf_s(gbFile_FaceIndices, "\n");


		}


	}


}

char gBuffer[128];

char* My_Strtok(char* str, char delimiter) {

	static int  i = 0;
	int  j = 0;
	char c;


	while ((c = str[i]) != delimiter && c != '\0') {
		gBuffer[j] = c;
		j = j + 1;
		i = i + 1;
	}

	gBuffer[j] = '\0';


	if (c == '\0') {
		i = 0;
	}
	else
		i = i + 1;


	return(gBuffer);
}



