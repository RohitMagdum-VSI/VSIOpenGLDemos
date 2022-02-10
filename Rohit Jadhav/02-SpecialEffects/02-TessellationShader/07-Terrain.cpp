#include<windows.h>
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
	AMC_ATTRIBUTE_POSITION = 1,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOORD0,
};

//For Fullscreen
bool bIsFullScreen_RRJ = false;
WINDOWPLACEMENT wpPrev_RRJ = { sizeof(WINDOWPLACEMENT) };
DWORD dwStyle_RRJ;
HWND ghwnd_RRJ;

//For OpenGL
bool bActiveWindow_RRJ = false;
HDC ghdc_RRJ = NULL;
HGLRC ghrc_RRJ = NULL;

//For Error
FILE *gbFile_RRJ = NULL;

//For Shader
GLuint vertexShaderObject_RRJ;
GLuint tessellationControlShaderObject_RRJ;
GLuint tessellationEvaluationShaderObject_RRJ;
GLuint fragmentShaderObject_RRJ;
GLuint shaderProgramObject_RRJ;

GLuint vao_Rect_RRJ;
GLuint vbo_Rect_Position_RRJ;

//For Uniform
GLuint mvpUniform_RRJ;

mat4 perspectiveProjectionMatrix_RRJ;


//For Texture
GLuint textureNoise_RRJ;
GLuint textureColor_RRJ;
GLuint samplerNoiseUniform_RRJ;
GLuint samplerColorUniform_RRJ;

//For Depth
GLfloat dmapDepth_RRJ = 0.0f;
GLuint dmapDepthUniform_RRJ;



//For KTX File
struct Header{
	unsigned char identifier[12];
	unsigned int endianness;
	unsigned int glType;
	unsigned int glTypeSize;
	unsigned int glFormat;
	unsigned int glInternalFormat;
	unsigned int glBaseInternalFormat;
	unsigned int pixelWidth;
	unsigned int pixelHeight;
	unsigned int pixelDepth;
	unsigned int arrayElements;
	unsigned int faces;
	unsigned int mipLevels;
	unsigned int keyPairBytes;
};



LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow) {
	

	int initialize(void);
	void ToggleFullScreen(void);
	void display(void);

	WNDCLASSEX wndclass_RRJ;
	HWND hwnd_RRJ;
	MSG msg_RRJ;
	TCHAR szName_RRJ[] = TEXT("Rohit_R_Jadhav-07-Terrain");

	int iRet;
	bool bDone = false;

	fopen_s(&gbFile_RRJ, "Log.txt", "w");
	if (gbFile_RRJ == NULL) {
		MessageBox(NULL, TEXT("Log Creation Failed!!"), TEXT("ERROR"), MB_OK);
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: Log Created!!\n");


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
		TEXT("Rohit_R_Jadhav-07-Terrain"),
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		100, 100,
		WIN_WIDTH, WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghwnd_RRJ = hwnd_RRJ;

	iRet = initialize();
	if (iRet == -1) {
		fprintf(gbFile_RRJ, "ERROR: ChoosePixelFormat() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else if (iRet == -2) {
		fprintf(gbFile_RRJ, "ERROR: SetPixelFormat() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else if (iRet == -3) {
		fprintf(gbFile_RRJ, "ERROR: wglCreateContext() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else if (iRet == -4) {
		fprintf(gbFile_RRJ, "ERROR: wglMakeCurrent() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: Initialize() Done!!\n");

	
	ShowWindow(hwnd_RRJ, iCmdShow);
	SetForegroundWindow(hwnd_RRJ);
	SetFocus(hwnd_RRJ);
	ToggleFullScreen();

	while (bDone == false) {
		if (PeekMessage(&msg_RRJ, NULL, 0, 0, PM_REMOVE)) {
			if (msg_RRJ.message == WM_QUIT)
				bDone = true;
			else {
				TranslateMessage(&msg_RRJ);
				DispatchMessage(&msg_RRJ);
			}
		}
		else {
			if (bActiveWindow_RRJ == true) {
				//Update
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
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			bActiveWindow_RRJ = true;
		else
			bActiveWindow_RRJ = false;
		break;

	case WM_ERASEBKGND:
		return(0);

	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_KEYDOWN:
		switch (wParam) {
		case 0X46: //F
			ToggleFullScreen();
			break;

		case VK_UP:
			dmapDepth_RRJ = dmapDepth_RRJ + 0.1f;
			break;

		case VK_DOWN:
			dmapDepth_RRJ = dmapDepth_RRJ - 0.1f;
			break;


		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;
		}
		break;

	case WM_CLOSE:
		uninitialize();
		break;

	case WM_DESTROY:
		uninitialize();
		PostQuitMessage(0);
		break;

	}

	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullScreen(void) {
	
	MONITORINFO mi_RRJ = { sizeof(MONITORINFO) };

	if (bIsFullScreen_RRJ == false) {
		dwStyle_RRJ = GetWindowLong(ghwnd_RRJ, GWL_STYLE);
		if (dwStyle_RRJ & WS_OVERLAPPEDWINDOW) {
			if (GetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ) && GetMonitorInfo(MonitorFromWindow(ghwnd_RRJ, MONITORINFOF_PRIMARY), &mi_RRJ)) {
				SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd_RRJ, HWND_TOP,
					mi_RRJ.rcMonitor.left,
					mi_RRJ.rcMonitor.top,
					(mi_RRJ.rcMonitor.right - mi_RRJ.rcMonitor.left),
					(mi_RRJ.rcMonitor.bottom - mi_RRJ.rcMonitor.top),
					SWP_NOZORDER | SWP_FRAMECHANGED);
			}
			ShowCursor(FALSE);
			bIsFullScreen_RRJ = true;
		}
	}
	else {
		SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ);
		SetWindowPos(ghwnd_RRJ, HWND_TOP,
			0, 0, 0, 0,
			SWP_NOZORDER | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
		bIsFullScreen_RRJ = false;
	}
}

int initialize(void) {

	void uninitialize(void);
	void resize(int, int);
	int LoadKTXTexture(const char*, GLuint*);

	PIXELFORMATDESCRIPTOR pfd_RRJ;
	int iPixelFormatIndex_RRJ;
	GLenum result_RRJ;

	ghdc_RRJ = GetDC(ghwnd_RRJ);

	memset(&pfd_RRJ, 0, sizeof(PIXELFORMATDESCRIPTOR));

	pfd_RRJ.nVersion = 1;
	pfd_RRJ.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd_RRJ.iPixelType = PFD_TYPE_RGBA;
	pfd_RRJ.dwFlags = PFD_DOUBLEBUFFER | PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW;

	pfd_RRJ.cColorBits = 32;
	pfd_RRJ.cRedBits = 8;
	pfd_RRJ.cGreenBits = 8;
	pfd_RRJ.cBlueBits = 8;
	pfd_RRJ.cAlphaBits = 8;
	pfd_RRJ.cDepthBits = 24;

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


	result_RRJ = glewInit();
	if (result_RRJ != GLEW_OK) {
		fprintf(gbFile_RRJ, "ERROR: glewInit() Failed!!\n");
		uninitialize();
		DestroyWindow(ghwnd_RRJ);
	}


	/********** VERTEX SHADER **********/
	vertexShaderObject_RRJ = glCreateShader(GL_VERTEX_SHADER);
	const char *vertexShaderSourceCode_RRJ =
		"#version 450" \
		"\n" \

		"in vec4 vPosition;" \

		"out VS_OUT { " \
			"vec2 texCoord;" \
		"}vs_out;" \


		"void main(void) {" \

			"int x = gl_InstanceID & 63;" \
			"int y = gl_InstanceID >> 6;" \
			"vec2 offset = vec2(x, y);" \

			"vs_out.texCoord = (vPosition.xz + offset + vec2(0.5f)) / 64.0f;" \

			"gl_Position = vPosition + vec4((x - 32), 0.0f, (y - 32), 0.0f);" \
		"}";

	glShaderSource(vertexShaderObject_RRJ, 1, (const char**)&vertexShaderSourceCode_RRJ, NULL);
	glCompileShader(vertexShaderObject_RRJ);

	int iInfoLogLength_RRJ;
	int iShaderCompileStatus_RRJ;
	char *szInfoLog_RRJ = NULL;

	glGetShaderiv(vertexShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(vertexShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ) {
				GLsizei written;

				glGetShaderInfoLog(vertexShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "VERTEX SHADER ERROR: \n %s", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
			}
		}
	}




	/********** TESSELLATION CONTROL SHADER **********/
	tessellationControlShaderObject_RRJ = glCreateShader(GL_TESS_CONTROL_SHADER);
	const char* tessellationControlShaderSourceCode_RRJ =
		"#version 450" \
		"\n" \
		"layout (vertices = 4) out;" \
		
		"uniform mat4 u_mvp_matrix;" \

		// *** From VS ***
		"in VS_OUT { " \

			"vec2 texCoord;" \

		"}tcs_in[];" \


		// *** For TES ***
		"out TCS_OUT{" \

			"vec2 texCoord;" \

		"}tcs_out[];" \


		"void main(void) {" \
				
			"if(gl_InvocationID == 0) {" \
			
				"vec4 p0 = u_mvp_matrix * gl_in[0].gl_Position;" \
				"vec4 p1 = u_mvp_matrix * gl_in[1].gl_Position;" \
				"vec4 p2 = u_mvp_matrix * gl_in[2].gl_Position;" \
				"vec4 p3 = u_mvp_matrix * gl_in[3].gl_Position;" \

				"p0 = p0 / p0.w;" \
				"p1 = p1 / p1.w;" \
				"p2 = p2 / p2.w;" \
				"p3 = p3 / p3.w;" \


				"if(p0.z == 0 || p1.z == 0 || p2.z == 0 || p3.z ==0) {" \

					"gl_TessLevelOuter[0] = 0.0f;" \
					"gl_TessLevelOuter[1] = 0.0f;" \
					"gl_TessLevelOuter[2] = 0.0f;" \
					"gl_TessLevelOuter[3] = 0.0f;" \

				"}" \
				"else { " \

					"float f0 = length(p2.xy - p0.xy) * 16.0f + 1.0f;" \
					"float f1 = length(p3.xy - p2.xy) * 16.0f + 1.0f;" \
					"float f2 = length(p3.xy - p1.xy) * 16.0f + 1.0f;" \
					"float f3 = length(p1.xy - p0.xy) * 16.0f + 1.0f;" \

					"gl_TessLevelOuter[0] = f0;" \
					"gl_TessLevelOuter[1] = f1;" \
					"gl_TessLevelOuter[2] = f2;" \
					"gl_TessLevelOuter[3] = f3;" \

					"gl_TessLevelInner[0] = min(f1, f3);" \
					"gl_TessLevelInner[1] = min(f0, f2);" \

				"}" \


			"}" \

			"gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;" \
			"tcs_out[gl_InvocationID].texCoord = tcs_in[gl_InvocationID].texCoord;" \

		"}";


	glShaderSource(tessellationControlShaderObject_RRJ, 1,
		(const char**)&tessellationControlShaderSourceCode_RRJ, NULL);

	glCompileShader(tessellationControlShaderObject_RRJ);

	iShaderCompileStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;
	glGetShaderiv(tessellationControlShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(tessellationControlShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ) {
				GLsizei written;
				glGetShaderInfoLog(tessellationControlShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "TESSELLATION CONTROL SHADER ERROR: \n %s", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
			}
		}
	}



	/********** TESSELLATION EVALUATION SHADER **********/
	tessellationEvaluationShaderObject_RRJ = glCreateShader(GL_TESS_EVALUATION_SHADER);
	const char *tessellationEvaluationShaderSourceCode_RRJ =
		"#version 450" \
		"\n" \
		"layout (quads) in;" \

		// *** From TCS ***
		"in TCS_OUT {" \

			"vec2 texCoord;" \

		"}tes_in[];" \


		// *** For FS ***
		"out TES_OUT{" \

			"vec2 texCoord;" \

		"}tes_out;" \

		
		"uniform mat4 u_mvp_matrix;" \
		"uniform sampler2D u_samplerNoise;" \
		"uniform float u_dmap_depth;" \


		"void main(void) {" \

			"vec2 tc1 = mix(tes_in[0].texCoord, tes_in[1].texCoord, gl_TessCoord.x);" \
			"vec2 tc2 = mix(tes_in[2].texCoord, tes_in[3].texCoord, gl_TessCoord.x);" \
			"vec2 tc = mix(tc2, tc1, gl_TessCoord.y);" \

			"vec4 p1 = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);" \
			"vec4 p2 = mix(gl_in[2].gl_Position, gl_in[3].gl_Position, gl_TessCoord.x);" \
			"vec4 p = mix(p2, p1, gl_TessCoord.y);" \

			"p.y = p.y + texture(u_samplerNoise, tc).r * u_dmap_depth;" \

			"gl_Position = u_mvp_matrix * p;" \
			"tes_out.texCoord = tc;" \

		"}";	



	glShaderSource(tessellationEvaluationShaderObject_RRJ, 1,
		(const char**)&tessellationEvaluationShaderSourceCode_RRJ, NULL);

	glCompileShader(tessellationEvaluationShaderObject_RRJ);

	iShaderCompileStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(tessellationEvaluationShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(tessellationEvaluationShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ) {
				GLsizei written;
				glGetShaderInfoLog(tessellationEvaluationShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "TESSELLATION EVALUATION SHADER ERROR: \n %s", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
			}
		}
	}



	/********** FRAGMENT SHADER **********/
	fragmentShaderObject_RRJ = glCreateShader(GL_FRAGMENT_SHADER);
	const char *fragmentShaderSourceCode_RRJ =
		"#version 450" \
		"\n" \
		
		// *** From TES ***
		"in TES_OUT{" \
			"vec2 texCoord;" \
		"}fs_in;" \

		"uniform sampler2D u_samplerColor;" \
		"out vec4 FragColor;" \


		"void main(void) {" \
			"FragColor = texture(u_samplerColor, fs_in.texCoord);" \
		"}";


	glShaderSource(fragmentShaderObject_RRJ, 1, 
		(const char**)&fragmentShaderSourceCode_RRJ, NULL);

	glCompileShader(fragmentShaderObject_RRJ);

	iInfoLogLength_RRJ = 0;
	iShaderCompileStatus_RRJ = 0;
	szInfoLog_RRJ = NULL;
	
	glGetShaderiv(fragmentShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(fragmentShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ) {
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "FRAGMENT SHADER ERROR: \n %s", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
			}
		}
	}


	/********** SHADER PROGRAM **********/
	shaderProgramObject_RRJ = glCreateProgram();

	glAttachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
	glAttachShader(shaderProgramObject_RRJ, tessellationControlShaderObject_RRJ);
	glAttachShader(shaderProgramObject_RRJ, tessellationEvaluationShaderObject_RRJ);
	glAttachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);

	glBindAttribLocation(shaderProgramObject_RRJ, AMC_ATTRIBUTE_POSITION, "vPosition");

	glLinkProgram(shaderProgramObject_RRJ);

	int iProgramLinkStatus_RRJ;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;
	glGetProgramiv(shaderProgramObject_RRJ, GL_LINK_STATUS, &iProgramLinkStatus_RRJ);
	if (iProgramLinkStatus_RRJ == GL_FALSE) {
		glGetProgramiv(shaderProgramObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ) {
				GLsizei written;
				glGetProgramInfoLog(shaderProgramObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "SHADER PROGRAM ERROR: %s", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
			}
		}
	}


	mvpUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_mvp_matrix");

	samplerNoiseUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_samplerNoise");
	samplerColorUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_samplerColor");
	dmapDepthUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_dmap_depth");

	/********** LINE COORDINATES **********/
	float rect_Vertices_RRJ[] = { 
			
		// *** See Coordinate Closely ***

		-0.5f, 0.0f, -0.5f, 1.0f,
		0.5f, 0.0f, -0.5f, 1.0f,
		-0.5f, 0.0f, 0.5f, 1.0f,
		0.5f, 0.0f, 0.5f, 1.0f,
	};


	LoadKTXTexture("terragen1.ktx", &textureNoise_RRJ);

	LoadKTXTexture("terragen_color.ktx", &textureColor_RRJ);


	glGenVertexArrays(1, &vao_Rect_RRJ);
	glBindVertexArray(vao_Rect_RRJ);

		/********** Position **********/
		glGenBuffers(1, &vbo_Rect_Position_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Position_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(rect_Vertices_RRJ), rect_Vertices_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	glLineWidth(3.0f);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	resize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}


void uninitialize(void) {


	if (bIsFullScreen_RRJ == true) {
		SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ);
		SetWindowPos(ghwnd_RRJ, HWND_TOP,
			0, 0, 0, 0,
			SWP_NOZORDER | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
	}

	if(textureColor_RRJ){
		glDeleteTextures(1, &textureColor_RRJ);
		textureColor_RRJ = 0;
	}


	if(textureNoise_RRJ){
		glDeleteTextures(1, &textureNoise_RRJ);
		textureNoise_RRJ = 0;
	}

	if (vbo_Rect_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Rect_Position_RRJ);
		vbo_Rect_Position_RRJ = 0;
	}

	if (vao_Rect_RRJ) {
		glDeleteVertexArrays(1, &vao_Rect_RRJ);
		vao_Rect_RRJ = 0;
	}

	if (shaderProgramObject_RRJ) {
		glUseProgram(shaderProgramObject_RRJ);
			
		/*GLint shaderCount_RRJ;
		GLint shaderNo_RRJ;

		glGetShaderiv(shaderProgramObject_RRJ, GL_ATTACHED_SHADERS, &shaderCount_RRJ);
		fprintf(gbFile_RRJ, "INFO: ShaderCount: %d\n", shaderCount_RRJ);
		GLuint *pShaders = (GLuint*)malloc(sizeof(GLuint*) * shaderCount_RRJ);
		if (pShaders) {
			glGetAttachedShaders(shaderProgramObject_RRJ, shaderCount_RRJ, &shaderCount_RRJ, pShaders);
			for (shaderNo_RRJ = 0; shaderNo_RRJ < shaderCount_RRJ; shaderNo_RRJ++) {
				glDetachShader(shaderProgramObject_RRJ, pShaders[shaderNo_RRJ]);
				glDeleteShader(pShaders[shaderNo_RRJ]);
				pShaders[shaderNo_RRJ] = 0;
			}
			free(pShaders);
			pShaders = NULL;
		}*/


		if (fragmentShaderObject_RRJ) {
			glDetachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);
			glDeleteShader(fragmentShaderObject_RRJ);
			fragmentShaderObject_RRJ = 0;
		}

		if (tessellationEvaluationShaderObject_RRJ) {
			glDetachShader(shaderProgramObject_RRJ, tessellationEvaluationShaderObject_RRJ);
			glDeleteShader(tessellationEvaluationShaderObject_RRJ);
			tessellationEvaluationShaderObject_RRJ = 0;
		}

		if (tessellationControlShaderObject_RRJ) {
			glDetachShader(shaderProgramObject_RRJ, tessellationControlShaderObject_RRJ);
			glDeleteShader(tessellationControlShaderObject_RRJ);
			tessellationControlShaderObject_RRJ = 0;
		}

		if (vertexShaderObject_RRJ) {
			glDetachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
			glDeleteShader(vertexShaderObject_RRJ);
			vertexShaderObject_RRJ = 0;
		}


		glUseProgram(0);
		glDeleteProgram(shaderProgramObject_RRJ);
		shaderProgramObject_RRJ = 0;
	}


	if (wglGetCurrentContext() == ghrc_RRJ)
		wglMakeCurrent(NULL, NULL);

	if (ghrc_RRJ) {
		wglDeleteContext(ghrc_RRJ);
		ghrc_RRJ = NULL;
	}

	if (ghdc_RRJ) {
		ReleaseDC(ghwnd_RRJ, ghdc_RRJ);
		ghdc_RRJ = NULL;
	}

	if (gbFile_RRJ) {
		fprintf(gbFile_RRJ, "SUCCESS: Log Close!!\n");
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
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	mat4 modelViewMatrix_RRJ;
	mat4 modelViewProjectionMatrix_RRJ;


	glUseProgram(shaderProgramObject_RRJ);

	modelViewMatrix_RRJ = mat4::identity();
	modelViewProjectionMatrix_RRJ = mat4::identity();

	modelViewMatrix_RRJ = translate(0.0f, -5.00f - dmapDepth_RRJ, -50.0f);
	modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;


	glUniformMatrix4fv(mvpUniform_RRJ, 1, GL_FALSE, modelViewProjectionMatrix_RRJ);
	glUniform1f(dmapDepthUniform_RRJ, dmapDepth_RRJ);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, textureNoise_RRJ);
	glUniform1i(samplerNoiseUniform_RRJ, 0);

	//glBindTexture(GL_TEXTURE_2D, 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, textureColor_RRJ);
	glUniform1i(samplerColorUniform_RRJ, 1);

	glBindVertexArray(vao_Rect_RRJ);
	glPatchParameteri(GL_PATCH_VERTICES, 4);
	glDrawArraysInstanced(GL_PATCHES, 0, 4, 64 * 64);
	glBindVertexArray(0);



	glUseProgram(0);

	SwapBuffers(ghdc_RRJ);
}




int LoadKTXTexture(const char* filename, GLuint *texture){

	void uninitialize(void);
	unsigned int swap32(unsigned int);
	unsigned int calculate_stride(struct Header&, unsigned int, unsigned int);
	unsigned int calculate_face_size(struct Header&);

	FILE *pFile = NULL;
	GLuint iTemp = 0;
	int iRetval = 0;

	struct Header h;
	size_t data_start, data_end, data_total_size;

	unsigned char *data = NULL;
	GLenum target = GL_NONE;


	unsigned char identifier[] = {
		0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 
		0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
	};


	pFile = fopen(filename, "rb");

	if(pFile == NULL){
		fprintf(gbFile_RRJ, "LoadKTXTexture : %s file open failed\n", filename);
		uninitialize();
		DestroyWindow(ghwnd_RRJ);
	}
	else
		fprintf(gbFile_RRJ, "LoadKTXTexture : %s file open\n", filename);


	if(fread(&h, sizeof(struct Header), 1, pFile) != 1){
		goto fail_read;
	}


	if(memcmp(h.identifier, identifier, sizeof(identifier)) != 0){
		goto fail_header;
	}



	fprintf(gbFile_RRJ, "LoadKTXTexture : Endianness : 0x%x\n", h.endianness);


	if(h.endianness == 0x04030201){
		// No Swap Needed
	}
	else if(h.endianness == 0x01020304){

		h.endianness = swap32(h.endianness);
		h.glType = swap32(h.glType);
		h.glTypeSize = swap32(h.glTypeSize);
		h.glFormat = swap32(h.glFormat);
		h.glInternalFormat = swap32(h.glInternalFormat);
		h.glBaseInternalFormat = swap32(h.glBaseInternalFormat);
		h.pixelWidth = swap32(h.pixelWidth);
		h.pixelHeight = swap32(h.pixelHeight);
		h.pixelDepth = swap32(h.pixelDepth);
		h.arrayElements = swap32(h.arrayElements);
		h.faces = swap32(h.faces);
		h.mipLevels = swap32(h.mipLevels);
		h.keyPairBytes = swap32(h.keyPairBytes);
	}
	else
		goto fail_header;




	if(h.pixelHeight == 0){
		
		// *** 1D Texture ***

		if(h.arrayElements == 0){
			target = GL_TEXTURE_1D;
		}
		else
			target = GL_TEXTURE_1D_ARRAY;

	}
	else if(h.pixelDepth == 0){

		// *** 2D Texture ***

		if(h.arrayElements == 0){
			
			if(h.faces == 0)
				target = GL_TEXTURE_2D;
			else
				target = GL_TEXTURE_CUBE_MAP;
		
		}
		else{
			
			if(h.faces == 0)
				target = GL_TEXTURE_2D_ARRAY;
			else
				target = GL_TEXTURE_CUBE_MAP_ARRAY;
		
		}
	}
	else{

		// *** 3D Texture ***

		target = GL_TEXTURE_3D;
	}




	// ***** Check for Insanity *****
	if(target == GL_NONE || (h.pixelWidth == 0) || (h.pixelHeight == 0 && h.pixelDepth != 0))
		goto fail_header;	



	if(*texture == 0){
		glGenTextures(1, texture);
	}


	glBindTexture(target, *texture);

	data_start = ftell(pFile) + h.keyPairBytes;
	fseek(pFile, 0, SEEK_END);
	data_end = ftell(pFile);

	data_total_size = data_end - data_start;


	fseek(pFile, data_start, SEEK_SET);

	data = (unsigned char*)malloc(sizeof(unsigned char) * data_total_size);
	if(data == NULL){
		fprintf(gbFile_RRJ, "LoadKTXTexture : Memory Allocation Failed\n");
		uninitialize();
		DestroyWindow(ghwnd_RRJ);
	}

	
	memset(data, 0, data_total_size);

	fread(data, 1, data_total_size, pFile);	


	if(h.mipLevels == 0)
		h.mipLevels = 1;

	fprintf(gbFile_RRJ, "LoadKTXTexture : Width : %d Height : %d\n", h.pixelWidth, h.pixelHeight);


	switch(target){

		case GL_TEXTURE_1D:
			glTexStorage1D(GL_TEXTURE_1D, h.mipLevels, h.glInternalFormat, h.pixelWidth);
			glTexSubImage1D(GL_TEXTURE_1D, 0, 0, h.pixelWidth, h.glFormat, h.glInternalFormat, data);
			break;


		case GL_TEXTURE_2D:

			if(h.glType == GL_NONE){
				glCompressedTexImage2D(GL_TEXTURE_2D, 0, h.glInternalFormat, h.pixelWidth, h.pixelHeight, 0, 420*380 / 2, data);
			}
			else{

				glTexStorage2D(GL_TEXTURE_2D, h.mipLevels, h.glInternalFormat, h.pixelWidth, h.pixelHeight);

				{
					unsigned char *ptr = data;
					unsigned int height = h.pixelHeight;
					unsigned int width = h.pixelWidth;

					glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
					for(unsigned int i = 0; i < h.mipLevels; i++){

						glTexSubImage2D(GL_TEXTURE_2D, i, 0, 0, width, height, h.glFormat, h.glType, ptr);

						ptr = ptr + height * calculate_stride(h, width, 1);

						height >>= 1;
						width >>= 1;

						if(!height)
							height = 1;
						if(!width)
							width = 1;
					}
				}

			}

			break;


		case GL_TEXTURE_3D:

			glTexStorage3D(GL_TEXTURE_3D, h.mipLevels, h.glInternalFormat, h.pixelWidth, h.pixelHeight, h.pixelDepth);
			glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, h.pixelWidth, h.pixelHeight, h.pixelDepth, h.glFormat, h.glType, data);

			break;

		case GL_TEXTURE_1D_ARRAY:
			glTexStorage2D(GL_TEXTURE_1D_ARRAY, h.mipLevels, h.glInternalFormat, h.pixelWidth, h.arrayElements);
			glTexSubImage2D(GL_TEXTURE_1D_ARRAY, 0, 0, 0, h.pixelWidth, h.arrayElements, h.glFormat, h.glType, data);
			break;


		case GL_TEXTURE_2D_ARRAY:
			glTexStorage3D(GL_TEXTURE_2D_ARRAY, h.mipLevels, h.glInternalFormat, h.pixelWidth, h.pixelHeight, h.arrayElements);
			glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, h.pixelWidth, h.pixelHeight, h.arrayElements, h.glFormat, h.glType, data);
			break;

		case GL_TEXTURE_CUBE_MAP:
			glTexStorage2D(GL_TEXTURE_CUBE_MAP, h.mipLevels, h.glInternalFormat, h.pixelWidth, h.pixelHeight);

			{
				unsigned int face_size = calculate_face_size(h);
				for(unsigned int i = 0; i < h.faces; i++){
					glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, 0, 0, h.pixelWidth, h.pixelHeight, h.glFormat, h.glType, data + face_size * i);
				}
			}
			break;

		case GL_TEXTURE_CUBE_MAP_ARRAY:

			glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY, h.mipLevels, h.glInternalFormat, h.pixelWidth, h.pixelHeight, h.arrayElements);
			glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, 0, 0, 0, 0, h.pixelWidth, h.pixelHeight, h.faces * h.arrayElements, h.glFormat, h.glType, data);

			break;

		default:
			goto fail_target;
	}


	if(h.mipLevels == 1)
		glGenerateMipmap(target);

	iRetval = (int)texture;


fail_target:

	if(data){
		free(data);
		data = NULL;
	}


fail_header:
fail_read:

	if(pFile){
		fclose(pFile);
		pFile = NULL;
	}

	return(iRetval);
}


unsigned int swap32(unsigned int u32){

	union{
		unsigned int u32;
		unsigned char u8[4];
	}a, b;

	a.u32 = u32;

	b.u8[0] = a.u8[3];
	b.u8[1] = a.u8[2];
	b.u8[2] = a.u8[1];
	b.u8[3] = a.u8[0];

	return(b.u32);
}

unsigned int calculate_stride(struct Header &h, unsigned int width, unsigned int pad = 4){

	unsigned int channels = 0;

	switch(h.glInternalFormat){

		case GL_RED:
			channels = 1;
			break;

		case GL_RG:
			channels = 2;
			break;

		case GL_BGR:
		case GL_RGB:
			channels = 3;
			break;

		case GL_BGRA:
		case GL_RGBA:
			channels = 4;
			break;
	}

	unsigned int  stride = h.glTypeSize * channels * width;

	stride = (stride + (pad - 1)) & ~(pad - 1);

	return(stride);
}


unsigned int calculate_face_size(struct Header &h){

	unsigned int stride = calculate_stride(h, h.pixelWidth, 4);

	return(stride * h.pixelHeight);
}

