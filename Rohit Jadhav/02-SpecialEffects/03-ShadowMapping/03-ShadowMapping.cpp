#include<Windows.h>
#include<stdio.h>
#include<gl/glew.h>
#include<GL/GL.h>
#include"vmath.h"
#include"02-vbmLoader.h"

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
mat4 gPerspectiveProjectionMatrix_ShadowMap;

//For Platform
GLuint vao_Rect;
GLuint vbo_Rect_Position;
GLuint vbo_Rect_Normal;

// For Simple Texture
GLuint vao_TestRect;
GLuint vbo_TestRect_Position;
GLuint vbo_TestRect_Tex;

	
// For Light Uniform	
GLuint La_Uniform;
GLuint Ld_Uniform;
GLuint Ls_Uniform;
GLuint lightPositionUniform;
GLuint Ka_Uniform;
GLuint Kd_Uniform;
GLuint Ks_Uniform;
GLuint materialShininessUniform;
GLuint LKeyPressUniform;


//For Lights
bool bLights = false;
GLfloat lightAmbient[] = { 0.50f, 0.50f, 0.50f, 0.0f };
GLfloat lightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat lightSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat lightPosition[] = { 10.0f, 10.0f, 10.0f, 1.0f };

//For Material
GLfloat materialAmbient[] = { 0.50f, 0.50f, 0.50f, 0.0f };
GLfloat materialDiffuse[] = { 1.f, 1.0f, 1.0f, 1.0f };
GLfloat materialSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialShininess = 50.0f;




//For Uniform
GLuint modelMatrixUniform; 
GLuint viewMatrixUniform;
GLuint projectionMatrixUniform;
GLuint shadowMatrix_Uniform;
GLuint sampler2dShadow_Uniform;
GLuint samplerUniform;
GLuint toggleUniform;


//For Framebuffer
GLuint gDepth_FrameBufferObject;
GLuint texture_Depth;
#define DEPTH_MAP_WIDTH 1024
#define DEPTH_MAP_HEIGHT 1024


//For VBM Model Loading
GLuint vao_vbmModel;
GLuint vbo_vbmModel_PNT;
GLuint vbo_vbmModel_Indices;

P_VBM_HEADER gpHeader = NULL;
P_VBM_ATTRIB_HEADER gpAttribs = NULL;
P_VBM_FRAME_HEADER gpFrames = NULL;
unsigned char *gpData = NULL;


//For Viewport
GLsizei gViewPortWidth;
GLsizei gViewPortHeight;


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
	void ToggleFullScreen(void);

	int iRet;
	bool bDone = false;

	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szName[] = TEXT("RohitRJadhav-PP-SimpleShadow");

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
		TEXT("RohitRJadhav-PP-SimpleShadow"),
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
				//update();
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
			DestroyWindow(hwnd);
			break;

		case 'F':
		case 'f':
			ToggleFullScreen();
			break;

		case 'L':
		case 'l':
			if(bLights == true)
				bLights = false;
			else
				bLights = true;
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
		"in vec3 vNormal;" \
		"in vec2 vTex;" \

		"out vec2 outTex;" \

		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"uniform mat4 u_shadow_matrix;" \

		"uniform vec4 u_light_position;" \

		"out vec3 viewer_vector_VS;" \
		"out vec3 tNorm_VS;" \
		"out vec3 lightDirection_VS;" \

		"out vec4 shadowCoord_VS;" \

		"uniform int u_Toggle;" \

		"void main(void)" \
		"{" \
			
			"if(u_Toggle == 1) { " \

				//For Shadow Map
				"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \

			"}" \
			"else{ " \

				//For Shadow Processing
				"vec4 eye_coordinate = u_view_matrix * u_model_matrix * vPosition;" \
				"viewer_vector_VS = vec3(-eye_coordinate);" \
				"tNorm_VS = mat3(u_view_matrix * u_model_matrix) * vNormal;" \
				"lightDirection_VS = vec3(u_light_position - eye_coordinate);" \

				"shadowCoord_VS = u_shadow_matrix * u_model_matrix * vPosition;" \

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

		"uniform int u_Toggle;" \

		"out vec4 FragColor;" \

		"in vec2 outTex;" \
		"uniform sampler2D u_sampler;" \


		"uniform sampler2DShadow u_sampler2dShadow;" \
		"in vec4 shadowCoord_VS;" \

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

		"void main(void)" \
		"{" \

			"if(u_Toggle == 1) { " \
				//For Shadow
				"FragColor = vec4(1.0f);" \

			"}" \

			"else{" \

				//For Lights Procressing

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
					

					//For Shadow	
					"float f = textureProj(u_sampler2dShadow, shadowCoord_VS);" \

					"vec3 Phong_ADS_Light = ambient + f * (diffuse + specular);" \

					"FragColor = vec4(Phong_ADS_Light, 1.0f);" \

				"}" \
				"else{" \
					"FragColor = vec4(1.0f, 1.0f, 1.0f, 1.0);" \
				"}" \

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
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_NORMAL, "vNormal");
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_TEXCOORD0, "vTex");


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
	shadowMatrix_Uniform = glGetUniformLocation(gShaderProgramObject, "u_shadow_matrix");
	samplerUniform = glGetUniformLocation(gShaderProgramObject, "u_sampler");


	sampler2dShadow_Uniform = glGetUniformLocation(gShaderProgramObject, "u_sampler2dShadow");

	La_Uniform = glGetUniformLocation(gShaderProgramObject, "u_La");
	Ld_Uniform = glGetUniformLocation(gShaderProgramObject, "u_Ld");
	Ls_Uniform = glGetUniformLocation(gShaderProgramObject, "u_Ls");
	lightPositionUniform = glGetUniformLocation(gShaderProgramObject, "u_light_position");
	Ka_Uniform = glGetUniformLocation(gShaderProgramObject, "u_Ka");
	Kd_Uniform = glGetUniformLocation(gShaderProgramObject, "u_Kd");
	Ks_Uniform = glGetUniformLocation(gShaderProgramObject, "u_Ks");
	materialShininessUniform = glGetUniformLocation(gShaderProgramObject, "u_shininess");
	LKeyPressUniform = glGetUniformLocation(gShaderProgramObject, "u_LKeyPress");

	toggleUniform = glGetUniformLocation(gShaderProgramObject, "u_Toggle");


	/********** Position and Normal For Platform **********/
	GLfloat Rect_Vertices[] = {
		 -500.0f, -50.0f, -500.0f, 1.0f,
	        -500.0f, -50.0f,  500.0f, 1.0f,
	         500.0f, -50.0f,  500.0f, 1.0f,
	         500.0f, -50.0f, -500.0f, 1.0f,
	};

	
	GLfloat Rect_Normal[] = {
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
	};


	// *** For Simple Rect Tex ***
	GLfloat TestRect_Vertices[] = {
		1.0f, 1.0f, 0.0f,
		-1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
	};


	GLfloat TestRect_Tex[] = {
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
	};



	/********** Platform **********/
	glGenVertexArrays(1, &vao_Rect);
	glBindVertexArray(vao_Rect);

		/********** Position **********/
		glGenBuffers(1, &vbo_Rect_Position);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Position);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(Rect_Vertices),
			Rect_Vertices,
			GL_STATIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
			4,
			GL_FLOAT,
			GL_FALSE,
			0, NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/********** Normal **********/
		glGenBuffers(1, &vbo_Rect_Normal);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Normal);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(Rect_Normal),
			Rect_Normal,
			GL_STATIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL,
			3,
			GL_FLOAT,
			GL_FALSE,
			0, NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);




	/********** Vao Rect On Which We Apply Texture **********/
	glGenVertexArrays(1, &vao_TestRect);
	glBindVertexArray(vao_TestRect);

		/********** Position **********/
		glGenBuffers(1, &vbo_TestRect_Position);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_TestRect_Position);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(TestRect_Vertices),
			TestRect_Vertices,
			GL_STATIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
			3,
			GL_FLOAT,
			GL_FALSE,
			0, NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/********** Texture **********/
		glGenBuffers(1, &vbo_TestRect_Tex);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_TestRect_Tex);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(TestRect_Tex),
			TestRect_Tex,
			GL_STATIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOORD0,
			2,
			GL_FLOAT,
			GL_FALSE,
			0, NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOORD0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);



	// ********** Loading .vbm Model **********
	vbmLoader("armadillo_low.vbm", 0, 1, 2);




	// ********** For FrameBuffer **********
	glGenFramebuffers(1, &gDepth_FrameBufferObject);
	glBindFramebuffer(GL_FRAMEBUFFER, gDepth_FrameBufferObject);


		// ********** For Shadow Depth Map Texture **********
		glGenTextures(1, &texture_Depth);
		glBindTexture(GL_TEXTURE_2D, texture_Depth);

		glTexImage2D(GL_TEXTURE_2D,
			0,
			GL_DEPTH_COMPONENT32, 
			//GL_RGBA,
			DEPTH_MAP_WIDTH, DEPTH_MAP_HEIGHT,
			0,
			GL_DEPTH_COMPONENT,
			//GL_RGBA,
			GL_FLOAT,
			//GL_UNSIGNED_BYTE,
			NULL);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);


		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	
		glBindTexture(GL_TEXTURE_2D, 0);


		glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, texture_Depth, 0);
		//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture_Depth, 0);
		
		// ********** For Disable Color From Default Framebuffer For Only Depth Values **********
		glDrawBuffer(GL_NONE);
		
		GLenum stat;
		// ********** Checking *********
		if((stat = glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE){


			switch(stat){
				case GL_FRAMEBUFFER_UNDEFINED :
					fprintf(gbFile, "ERROR: 1\n");
					break;

				case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT :
					fprintf(gbFile, "ERROR: 2\n");
					break;

				case  GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT :
					fprintf(gbFile, "ERROR: 3\n");
					break;

				case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER :
					fprintf(gbFile, "ERROR: 4\n");
					break;

				case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER :
					fprintf(gbFile, "ERROR: 5\n");
					break;

				case GL_FRAMEBUFFER_UNSUPPORTED:
					fprintf(gbFile, "ERROR: 6\n");
					break;

				case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE :
					fprintf(gbFile, "ERROR: 7\n");
					break;

				case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS :
					fprintf(gbFile, "ERROR: 8\n");
					break;

				default:
					fprintf(gbFile, "ERROR: 9\n");
					break;
			}

			fprintf(gbFile, "ERROR: glCheckFramebufferStatus()\n");
			uninitialize();
			DestroyWindow(ghwnd);
		}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	gPerspectiveProjectionMatrix = mat4::identity();
	gPerspectiveProjectionMatrix_ShadowMap = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}


void uninitialize(void) {

	
	if(gDepth_FrameBufferObject){
		glDeleteFramebuffers(1, &gDepth_FrameBufferObject);
		gDepth_FrameBufferObject = 0;
	}

	if(texture_Depth){
		glDeleteTextures(1, &texture_Depth);
		texture_Depth = 0;
	}


	// ***** Model *****
	if(gpFrames){
		free(gpFrames);
		gpFrames = NULL;
		fprintf(gbFile, "vbmLoader: gpFrames freed\n");
	}

	if(gpAttribs){
		free(gpAttribs);
		gpAttribs = NULL;
		fprintf(gbFile, "vbmLoader: gpAttribs freed\n");
	}

	if(gpData){
		free(gpData);
		gpData = NULL;
		fprintf(gbFile, "vbmLoader: gpData freed\n");
	}

	if(vbo_vbmModel_Indices){
		glDeleteBuffers(1, &vbo_vbmModel_Indices);
		vbo_vbmModel_Indices = 0;
	}

	if (vbo_vbmModel_PNT) {
		glDeleteBuffers(1, &vbo_vbmModel_PNT);
		vbo_vbmModel_PNT = 0;
	}

	if (vao_vbmModel) {
		glDeleteVertexArrays(1, &vao_vbmModel);
		vao_vbmModel = 0;
	}

	// ***** Platform *****
	if (vbo_TestRect_Tex) {
		glDeleteBuffers(1, &vbo_TestRect_Tex);
		vbo_TestRect_Tex = 0;
	}

	if (vbo_TestRect_Position) {
		glDeleteBuffers(1, &vbo_TestRect_Position);
		vbo_TestRect_Position = 0;
	}

	if (vao_TestRect) {
		glDeleteVertexArrays(1, &vao_TestRect);
		vao_TestRect = 0;
	}


	// ***** Platform *****
	if (vbo_Rect_Normal) {
		glDeleteBuffers(1, &vbo_Rect_Normal);
		vbo_Rect_Normal = 0;
	}

	if (vbo_Rect_Position) {
		glDeleteBuffers(1, &vbo_Rect_Position);
		vbo_Rect_Position = 0;
	}

	if (vao_Rect) {
		glDeleteVertexArrays(1, &vao_Rect);
		vao_Rect = 0;
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

	//glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	//gPerspectiveProjectionMatrix = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);

	gViewPortWidth = (GLsizei)width;
	gViewPortHeight = (GLsizei)height;
}

void display(void) {


	mat4 translateMatrix;
	mat4 rotateMatrix;
	mat4 scaleMatrix;
	mat4 modelMatrix;
	mat4 viewMatrix;
	mat4 scaleBiasMatrix = mat4(vec4(0.5f, 0.0f, 0.0f, 0.0f),
						vec4(0.0f, 0.5f, 0.0f, 0.0f),
						vec4(0.0f, 0.0f, 0.5f, 0.0f),
						vec4(0.5f, 0.5f, 0.5f, 1.0f));

	mat4 shadowMatrix;

	static GLfloat angle = 0.0f;

	float t = float(GetTickCount() & 0xFFFF) / float(0xFFFF);
	
	vec3 light_position = vec3(sinf(t * 6.0f * 3.141592f) * 300.0f,
						200.0f,
						cosf(t * 4.0f * 3.141592f) * 100.0f + 250.0f);



	glViewport(0, 0, (GLsizei)DEPTH_MAP_WIDTH, (GLsizei)DEPTH_MAP_HEIGHT);
	gPerspectiveProjectionMatrix_ShadowMap = mat4::identity();
	gPerspectiveProjectionMatrix_ShadowMap = frustum(-1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 800.0f);

	// ********** For Creating Shadow Map **********
	glBindFramebuffer(GL_FRAMEBUFFER, gDepth_FrameBufferObject);


		glEnable(GL_CULL_FACE);
		glClearDepth(1.0f);
		glClear(GL_DEPTH_BUFFER_BIT);

		translateMatrix = mat4::identity();
		scaleMatrix = mat4::identity();
		rotateMatrix = mat4::identity();
		modelMatrix = mat4::identity();
		viewMatrix = mat4::identity();
		shadowMatrix = mat4::identity();
		
		// ********** Render **********
		glUseProgram(gShaderProgramObject);

		//translateMatrix = translate(0.0f, 0.0f, -300.0f);
		rotateMatrix = rotate(t * 720.0f, 0.0f, 1.0f, 0.0f);
		//rotateMatrix = rotate(angle, 0.0f, 1.0f, 0.0f);
		//scaleMatrix = scale(0.02f, 0.02f, 0.02f);

		//angle = angle + 0.5f;

		modelMatrix = translateMatrix * scaleMatrix * rotateMatrix;
		viewMatrix = lookat(light_position, vec3(0.0f), vec3(0.0f, 1.0f, 0.0f));
		
		shadowMatrix = scaleBiasMatrix * gPerspectiveProjectionMatrix_ShadowMap * viewMatrix;


		glUniformMatrix4fv(modelMatrixUniform, 1, GL_FALSE, modelMatrix);
		glUniformMatrix4fv(viewMatrixUniform, 1, GL_FALSE, viewMatrix);
		glUniformMatrix4fv(projectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix_ShadowMap);

	
		glUniform1i(toggleUniform, 1);

		glEnable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(2.0f, 4.0f);

			glBindVertexArray(vao_vbmModel);

	           	if(gpHeader->num_indices)
	           		glDrawElements(GL_TRIANGLES, gpFrames[0].count, GL_UNSIGNED_INT, (GLvoid *)(gpFrames[0].first * sizeof(GLuint)));
	       		else
				glDrawArrays(GL_TRIANGLES, gpFrames[0].first, gpFrames[0].count);

			glBindVertexArray(0);

			// Platform
			glBindVertexArray(vao_Rect);
			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
			glBindVertexArray(0);

		glDisable(GL_POLYGON_OFFSET_FILL);
		
		glUseProgram(0);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);




	// ********** For Appling Shadow **********
	GLfloat aspect = (GLfloat)gViewPortHeight / (GLfloat)gViewPortWidth;

	glViewport(0, 0, gViewPortWidth, gViewPortHeight);
	gPerspectiveProjectionMatrix = frustum(-1.0f, 1.0f, -aspect, aspect, 1.0f, 800.0f);

	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	glUseProgram(gShaderProgramObject);

		translateMatrix = mat4::identity();
		modelMatrix = mat4::identity();
		rotateMatrix = mat4::identity();
		scaleMatrix = mat4::identity();
		viewMatrix = mat4::identity();

		modelMatrix = rotate(t * 720.0f, 0.0f, 1.0f, 0.0f);
	
		viewMatrix = translate(0.0f, 0.0f, -300.0f);


		glUniform1i(toggleUniform, 3);

		glUniformMatrix4fv(modelMatrixUniform, 1, GL_FALSE, modelMatrix);
		glUniformMatrix4fv(viewMatrixUniform, 1, GL_FALSE, viewMatrix);
		glUniformMatrix4fv(projectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

		glUniformMatrix4fv(shadowMatrix_Uniform, 1, GL_FALSE, shadowMatrix);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture_Depth);
		glGenerateMipmap(GL_TEXTURE_2D);
		glUniform1i(sampler2dShadow_Uniform, 0);

		if (bLights == true) {
			glUniform1i(LKeyPressUniform, 1);

			glUniform3fv(La_Uniform, 1, lightAmbient);
			glUniform3fv(Ld_Uniform, 1, lightDiffuse);
			glUniform3fv(Ls_Uniform, 1, lightSpecular);
			glUniform4fv(lightPositionUniform, 1, light_position);

			glUniform3fv(Ka_Uniform, 1, materialAmbient);
			glUniform3fv(Kd_Uniform, 1, materialDiffuse);
			glUniform3fv(Ks_Uniform, 1, materialSpecular);
			glUniform1f(materialShininessUniform, materialShininess);
		}
		else
			glUniform1i(LKeyPressUniform, 0);

		//Model
		glBindVertexArray(vao_vbmModel);
	
           	if(gpHeader->num_indices)
           		glDrawElements(GL_TRIANGLES, gpFrames[0].count, GL_UNSIGNED_INT, (GLvoid *)(gpFrames[0].first * sizeof(GLuint)));
       		else
			glDrawArrays(GL_TRIANGLES, gpFrames[0].first, gpFrames[0].count);
		glBindVertexArray(0);


		if (bLights == true) {
			glUniform1i(LKeyPressUniform, 1);

			glUniform3fv(La_Uniform, 1, lightAmbient);
			glUniform3fv(Ld_Uniform, 1, lightDiffuse);
			glUniform3fv(Ls_Uniform, 1, lightSpecular);
			glUniform4fv(lightPositionUniform, 1, light_position);

			glUniform3fv(Ka_Uniform, 1, vec3(0.1f, 0.1f, 0.1f));
			glUniform3fv(Kd_Uniform, 1, vec3(0.1f, 0.5f, 0.1f));
			glUniform3fv(Ks_Uniform, 1, vec3(0.1f, 0.1f, 0.1f));
			glUniform1f(materialShininessUniform, materialShininess);
		}
		else
			glUniform1i(LKeyPressUniform, 0);		


		// Platform
		glBindVertexArray(vao_Rect);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		glBindVertexArray(0);


	glUseProgram(0);

	angle = angle + 0.5f;

	SwapBuffers(ghdc);
}




void vbmLoader(char *fileName, int iVertexIndex, int iNormalIndex, int iTexcoord0Index){

	fprintf(gbFile, "vbmLoader: Entered\n");

	void uninitialize(void);

	FILE *pFile = NULL;

	pFile = fopen(fileName, "rb");
	if(pFile == NULL){
		fprintf(gbFile, "vbmLoader: fopen() Failed\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}


	fseek(pFile, 0, SEEK_END);
	size_t fileSize = ftell(pFile);
	fseek(pFile, 0, SEEK_SET);

	fprintf(gbFile, "vbmLoader: FileSize : %d\n", fileSize);

	gpData = (unsigned char*)malloc(sizeof(unsigned char) * fileSize);
	if(gpData == NULL){
		fprintf(gbFile, "vbmLoader: malloc() failed\n");

		fclose(pFile);
		pFile = NULL;

		uninitialize();
		DestroyWindow(ghwnd);
	}


	fread(gpData, fileSize, 1, pFile);
	fclose(pFile);
	pFile = NULL;

	unsigned char *pRawData = NULL; 


	gpHeader = (P_VBM_HEADER)gpData;

	fprintf(gbFile, "vbmLoader: num_indices: %d\n", gpHeader->num_indices);

	pRawData = gpData + 
			sizeof(VBM_HEADER) + 
			(sizeof(VBM_ATTRIB_HEADER) * gpHeader->num_attribs) + 
			(sizeof(VBM_FRAME_HEADER) * gpHeader->num_frames);


	// ****** Attrib Header Starting Address ******		
	P_VBM_ATTRIB_HEADER pAttribHeader = (P_VBM_ATTRIB_HEADER)(gpData + sizeof(VBM_HEADER));

	// ****** Frame Header Starting Address ******	
	P_VBM_FRAME_HEADER pFrameHeader =  (P_VBM_FRAME_HEADER)(gpData + 
													sizeof(VBM_HEADER) + 
													(sizeof(VBM_ATTRIB_HEADER) * gpHeader->num_attribs));



	// ***** For Attribs headers *****
	gpAttribs = (P_VBM_ATTRIB_HEADER)malloc(sizeof(VBM_ATTRIB_HEADER) * gpHeader->num_attribs);
	if(gpAttribs == NULL){
		fprintf(gbFile, "vbmLoader: Malloc() for Attrib Header Failed\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}

	memcpy(gpAttribs, pAttribHeader, sizeof(VBM_ATTRIB_HEADER) * gpHeader->num_attribs);
	fprintf(gbFile, "vbmLoader: Attrib: %d\n", gpHeader->num_attribs);



	// ***** For Frame headers *****
	gpFrames = (P_VBM_FRAME_HEADER)malloc(sizeof(VBM_FRAME_HEADER) * gpHeader->num_frames);
	if(gpFrames == NULL){
		fprintf(gbFile, "vbmLoader: Malloc() for Frame Header Failed\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}	

	memcpy(gpFrames, pFrameHeader, sizeof(VBM_FRAME_HEADER) * gpHeader->num_frames);
	fprintf(gbFile, "vbmLoader: Frame: %d\n", gpHeader->num_frames);


	glGenVertexArrays(1, &vao_vbmModel);
	glBindVertexArray(vao_vbmModel);
	glGenBuffers(1, &vbo_vbmModel_PNT);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_vbmModel_PNT);

	unsigned int iTotalDataSize = 0;

	for(int i = 0; i < gpHeader->num_attribs; i++){
		int attribIndex = i;

		if(attribIndex == 0)
			attribIndex = iVertexIndex;
		else if(attribIndex == 1)
			attribIndex = iNormalIndex;
		else if(attribIndex == 2)
			attribIndex = iTexcoord0Index;


		glVertexAttribPointer(attribIndex, gpAttribs[i].components, gpAttribs[i].type, GL_FALSE, 0, (void*)iTotalDataSize);
		glEnableVertexAttribArray(attribIndex);

		fprintf(gbFile, "vbmLoader: %d Attrib Added\n", attribIndex);
		iTotalDataSize = iTotalDataSize + (sizeof(GLfloat) * gpAttribs[i].components * gpHeader->num_vertices); 
	}

	glBufferData(GL_ARRAY_BUFFER, iTotalDataSize, pRawData, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	// ***** Indices *****
	if(gpHeader->num_indices){

		glGenBuffers(1, &vbo_vbmModel_Indices);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_vbmModel_Indices);

		unsigned int element_size;

		switch(gpHeader->index_type){
			case GL_UNSIGNED_SHORT:
				element_size = sizeof(GLushort);
				break;

			default:
				element_size = sizeof(GLuint);
				break;
		}
	       
	        glBufferData(GL_ELEMENT_ARRAY_BUFFER, gpHeader->num_indices * element_size, gpData + iTotalDataSize, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		fprintf(gbFile, "vbmLoader: vbo_vbmModel_Indices\n");
	}


	fprintf(gbFile, "vbmLoader: Frames: first: %d\n", gpFrames[0].first);
	fprintf(gbFile, "vbmLoader: Frames: count: %d\n", gpFrames[0].count);


	glBindVertexArray(0);

	fprintf(gbFile, "vbmLoader: Done\n");
}
