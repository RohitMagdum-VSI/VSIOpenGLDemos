#include<Windows.h>
#include<stdio.h>
#include<gl/glew.h>
#include<GL/GL.h>
#include"vmath.h"
#include"01-ComputeShader.h"
#include<mmsystem.h>


#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "winmm.lib")


enum {
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOORD0
};

enum{
	RRJ_COMPUTE_POS = 0,
	RRJ_COMPUTE_VELO = 1,
};


using namespace vmath;

#define WIN_WIDTH 800
#define WIN_HEIGHT 600
#define RRJ_GROUP_X 4000
#define RRJ_MAX_PARTICALS 128 * RRJ_GROUP_X
#define RRJ_MAX_ATTRACTOR 32

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


//For Points
GLuint vao_Points_RRJ;
GLuint vbo_Points_Pos_RRJ;
GLuint vbo_Points_Velo_RRJ;



//For Uniform
GLuint modelMatrixUniform_RRJ;
GLuint viewMatrixUniform_RRJ;
GLuint projectionMatrixUniform_RRJ;



//For Compute Shader
GLuint gComputeShaderObject_RRJ;
GLuint gComputeShaderProgramObject_RRJ;

//For Compute Shader Output
GLuint texturePos_RRJ;
GLuint textureVelo_RRJ;

//For Compute Shader Uniform
GLuint timeUniform_RRJ;



// *** For Attractor for "UNIFORM BLOCK(Concept)" ***
GLuint vbo_Attractor_RRJ;
GLfloat fAttractorMass[RRJ_MAX_ATTRACTOR];



// *** Start ***
bool gbStart_RRJ = false;


// *** For Time ***
GLfloat gTime_RRJ = 1.5f;


// *** For Sound ***
HINSTANCE ghInst_RRJ = NULL;

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow) {

	if (fopen_s(&gbFile_RRJ, "Log.txt", "w") != 0) {
		MessageBox(NULL, TEXT("Log Creation Failed!!\n"), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "Log Created!!\n");

	int initialize(void);
	void display(void);
	void ToggleFullScreen(void);

	ghInst_RRJ = hInstance;

	int iRet_RRJ;
	bool bDone_RRJ = false;

	WNDCLASSEX wndclass_RRJ;
	HWND hwnd_RRJ;
	MSG msg_RRJ;
	TCHAR szName_RRJ[] = TEXT("RohitRJadhav-PP-05-ParticalSystem");

	wndclass_RRJ.lpszClassName = szName_RRJ;
	wndclass_RRJ.lpszMenuName = NULL;
	wndclass_RRJ.lpfnWndProc = WndProc;

	wndclass_RRJ.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass_RRJ.cbSize = sizeof(WNDCLASSEX);
	wndclass_RRJ.cbWndExtra = 0;
	wndclass_RRJ.cbClsExtra = 0;

	wndclass_RRJ.hInstance = hInstance;
	wndclass_RRJ.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass_RRJ.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(RRJ_ICON));
	wndclass_RRJ.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(RRJ_ICON));
	wndclass_RRJ.hCursor = LoadCursor(NULL, IDC_ARROW);

	RegisterClassEx(&wndclass_RRJ);

	hwnd_RRJ = CreateWindowEx(WS_EX_APPWINDOW,
		szName_RRJ,
		TEXT("RohitRJadhav-PP-05-ParticalSystem"),
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
	if (iRet_RRJ == -1) {
		fprintf(gbFile_RRJ, "ChoosePixelFormat() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else if (iRet_RRJ == -2) {
		fprintf(gbFile_RRJ, "SetPixelFormat() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else if (iRet_RRJ == -3) {
		fprintf(gbFile_RRJ, "wglCreateContext() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else if (iRet_RRJ == -4) {
		fprintf(gbFile_RRJ, "wglMakeCurrent() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else
		fprintf(gbFile_RRJ, "initialize() done!!\n");



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
				//update();
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

		case 's':
		case 'S':
			if(gbStart_RRJ == false){
				PlaySound(MAKEINTRESOURCE(RRJ_WAVE), ghInst_RRJ, SND_RESOURCE | SND_ASYNC);
				gbStart_RRJ = true;
			}
			else{
				gbStart_RRJ = false;
			}
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

	//GLfloat RandomNoInRangeV2(GLfloat, GLfloat);
	
	GLfloat RandomFloat(void);
	vec3 RandomVec3(GLfloat, GLfloat);

	PIXELFORMATDESCRIPTOR pfd_RRJ;
	int iPixelFormatIndex_RRJ;
	GLenum Result_RRJ;

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

	Result_RRJ = glewInit();
	if (Result_RRJ != GLEW_OK) {
		fprintf(gbFile_RRJ, "glewInit() Failed!!\n");
		uninitialize();
		DestroyWindow(ghwnd_RRJ);
		exit(1);
	}



	// ********** Compute Shader ********
	gComputeShaderObject_RRJ = glCreateShader(GL_COMPUTE_SHADER);

	const GLchar *szComputeShaderSourceCode_RRJ = 
		"#version 450 core" \
		"\n" \

		"layout (local_size_x = 128) in;" \

		"layout (std140, binding = 0) uniform attractor_block { " \
			"vec4 attractor[64];" \
		"};" \



		"layout (rgba32f, binding = 0) uniform imageBuffer u_Position;" \
		"layout (rgba32f, binding = 1) uniform imageBuffer u_Velocity;" \

		"uniform float u_Time;" \

		"void main(void) {" \

			"vec4 pos = imageLoad(u_Position, int(gl_GlobalInvocationID.x));" \
			"vec4 velo = imageLoad(u_Velocity, int(gl_GlobalInvocationID.x));" \

			"pos.xyz += velo.xyz * u_Time;" \

			"pos.w -= 0.0001f * u_Time;" \


			// *** Here We Update the velocity w.r.t Attractor ***
			"for(int i = 0; i < 4; i++){" \
				"vec3 dist = (attractor[i].xyz - pos.xyz);" \
				"velo.xyz += u_Time * u_Time * attractor[i].w * normalize(dist) / (dot(dist, dist) + 10.0f);" \

			"}" \

			// *** If Partical Dies ***
			"if(pos.w <= 0.0f) { " \
				"pos.xyz = -pos.xyz * 0.005f;" \
				"velo.xyz *= 0.01f;" \
				"pos.w += 1.0f;" \
			"}" \

			"imageStore(u_Position, int(gl_GlobalInvocationID.x), pos);" \
			"imageStore(u_Velocity, int(gl_GlobalInvocationID.x), velo);" \
		
		"}";

	glShaderSource(gComputeShaderObject_RRJ, 1, (const GLchar**)&szComputeShaderSourceCode_RRJ, NULL);

	glCompileShader(gComputeShaderObject_RRJ);

	GLint iComputeShaderCompilationStatus = 0;
	GLint iInfoLogLengthComputeShader = 0;
	GLchar *szInfoLogComputeShader = NULL;

	glGetShaderiv(gComputeShaderObject_RRJ, GL_COMPILE_STATUS, &iComputeShaderCompilationStatus);
	if(iComputeShaderCompilationStatus == GL_FALSE){
		glGetShaderiv(gComputeShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLengthComputeShader);
		if(iInfoLogLengthComputeShader > 0){
			szInfoLogComputeShader = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLengthComputeShader);
			if(szInfoLogComputeShader){
				GLsizei writen;
				glGetShaderInfoLog(gComputeShaderObject_RRJ, iInfoLogLengthComputeShader, &writen, szInfoLogComputeShader);
				fprintf_s(gbFile_RRJ, "Compute Shader Compilation Error : %s\n", szInfoLogComputeShader);
				free(szInfoLogComputeShader);
				szInfoLogComputeShader = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
			}
		}
	}



	// ********** Compute Program **********
	gComputeShaderProgramObject_RRJ = glCreateProgram();

	glAttachShader(gComputeShaderProgramObject_RRJ, gComputeShaderObject_RRJ);

	glLinkProgram(gComputeShaderProgramObject_RRJ);

	GLint iComputeProgramLinkingStatus;
	iInfoLogLengthComputeShader = 0;
	szInfoLogComputeShader = NULL;

	glGetProgramiv(gComputeShaderProgramObject_RRJ, GL_LINK_STATUS, &iComputeProgramLinkingStatus);
	if(iComputeProgramLinkingStatus == GL_FALSE){
		glGetProgramiv(gComputeShaderProgramObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLengthComputeShader);
		if(iInfoLogLengthComputeShader > 0){
			szInfoLogComputeShader = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLengthComputeShader);
			if(szInfoLogComputeShader){
				GLsizei written;
				glGetProgramInfoLog(gComputeShaderProgramObject_RRJ, iInfoLogLengthComputeShader, &written, szInfoLogComputeShader);
				fprintf_s(gbFile_RRJ, "Compute Program Linking Error : %s\n", szInfoLogComputeShader);
				free(szInfoLogComputeShader);
				szInfoLogComputeShader = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
			}
		}
	}

	timeUniform_RRJ = glGetUniformLocation(gComputeShaderProgramObject_RRJ, "u_Time");




	/********** Vertex Shader **********/
	iVertexShaderObject_RRJ = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *szVertexShaderSourceCode_RRJ =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \

		"out float outIntensity;" \

		"void main(void)" \
		"{" \
			"outIntensity = vPosition.w;" \
			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vec4(vPosition.xyz, 1.0f);" \
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
				fprintf(gbFile_RRJ, "Vertex Shader Compilation Error: %s\n", szInfoLog_RRJ);
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

	const GLchar *szFragmentShaderSourceCode =
		"#version 450 core" \
		"\n" \
		
		"out vec4 FragColor;" \
		"in float outIntensity;" \

		"void main(void)" \
		"{" \

			//"vec4 color1 = mix(vec4(0.0f, 0.2f, 1.0f, 1.0f), vec4(0.2f, 0.05f, 0.0f, 1.0f), outIntensity);" \
		
			//"vec4 color2 = mix(vec4(0.0f, 0.0f, 0.0f, 1.0f), vec4(0.2f, 0.05f, 0.0f, 1.0f), outIntensity);" \

			"vec4 color3 = mix(vec4(0.0f, 0.0f, 0.0f, 1.0f), vec4(0.0f, 0.20f, 0.80f, 1.0f), outIntensity);" \

			//"vec4 color = mix(color3, vec4(0.2f, 0.05f, 0.0f, 1.0f), outIntensity);" \

			"FragColor = color3;" \
		"}";

	glShaderSource(iFragmentShaderObject_RRJ, 1,
		(const GLchar**)&szFragmentShaderSourceCode, NULL);

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
				fprintf(gbFile_RRJ, "Fragment Shader Compilation Error: %s\n", szInfoLog_RRJ);
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
	

	glLinkProgram(gShaderProgramObject_RRJ);

	GLint iProgramLinkingStatus = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;
	glGetProgramiv(gShaderProgramObject_RRJ, GL_LINK_STATUS, &iProgramLinkingStatus);
	if (iProgramLinkingStatus == GL_FALSE) {
		glGetProgramiv(gShaderProgramObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Shader Program Object Linking Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}

	modelMatrixUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_model_matrix");
	viewMatrixUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_view_matrix");
	projectionMatrixUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_projection_matrix");
	
	



	// ********** Points Vao **********
	glGenVertexArrays(1, &vao_Points_RRJ);
	glBindVertexArray(vao_Points_RRJ);


		// ********** Position **********
		glGenBuffers(1, &vbo_Points_Pos_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Points_Pos_RRJ);
		glBufferData(GL_ARRAY_BUFFER, RRJ_MAX_PARTICALS * sizeof(vec4), NULL, GL_DYNAMIC_COPY);

		vec4 *fPosition = (vec4*)glMapBufferRange(
			GL_ARRAY_BUFFER, 
			0, 
			RRJ_MAX_PARTICALS * sizeof(vec4), 
			GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

		for(int i = 0; i < RRJ_MAX_PARTICALS; i++){
			/*fPosition[i] = vec4(RandomNoInRangeV2(-10.0f, 10.0f), 
						RandomNoInRangeV2(-10.0f, 10.0f),
						RandomNoInRangeV2(-10.0f, 10.0f),
						RandomNumber());*/


			fPosition[i] = vec4(RandomVec3(-10.0f, 10.0f), RandomFloat());
			

		}

		glUnmapBuffer(GL_ARRAY_BUFFER);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

		
		// ********** Creating TBO **********
		glGenTextures(1, &texturePos_RRJ);
		glBindTexture(GL_TEXTURE_BUFFER, texturePos_RRJ);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, vbo_Points_Pos_RRJ);

		glBindImageTexture(RRJ_COMPUTE_POS, texturePos_RRJ, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

		glBindTexture(GL_TEXTURE_BUFFER, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		

		// ********** Velocity **********
		glGenBuffers(1, &vbo_Points_Velo_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Points_Velo_RRJ);
		glBufferData(GL_ARRAY_BUFFER, RRJ_MAX_PARTICALS * sizeof(vec4), NULL, GL_DYNAMIC_COPY);

		vec4 *fVelocity = (vec4*)glMapBufferRange(
			GL_ARRAY_BUFFER, 
			0,
			RRJ_MAX_PARTICALS * sizeof(vec4),
			GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

		for(int i = 0; i < RRJ_MAX_PARTICALS; i++){
			//fVelocity[i] = vec4(RandomVec3(-0.1f, 0.1f), 0.0f);
		
			/*fVelocity[i] = vec4(RandomNoInRangeV2(-0.1f, 0.1f),
						RandomNoInRangeV2(-0.1f, 0.1f),
						RandomNoInRangeV2(-0.1f, 0.1f),
						0.0f);*/

			fVelocity[i] = vec4(RandomVec3(-0.1f, 0.1f), 0.0f);
		}

		glUnmapBuffer(GL_ARRAY_BUFFER);


		glGenTextures(1, &textureVelo_RRJ);
		glBindTexture(GL_TEXTURE_BUFFER, textureVelo_RRJ);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, vbo_Points_Velo_RRJ);

		glBindImageTexture(RRJ_COMPUTE_VELO, textureVelo_RRJ, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
			
		glBindTexture(GL_TEXTURE_BUFFER, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


	glBindVertexArray(0);




	// ********** For Attractors **********
	glGenBuffers(1, &vbo_Attractor_RRJ);
	glBindBuffer(GL_UNIFORM_BUFFER, vbo_Attractor_RRJ);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(vec4) * RRJ_MAX_ATTRACTOR, NULL, GL_STATIC_DRAW);
	
	for(int i = 0; i < RRJ_MAX_ATTRACTOR; i++){
		fAttractorMass[i] = 0.5f + RandomFloat() * 0.5f;
	}


	glBindBufferBase(GL_UNIFORM_BUFFER, 0, vbo_Attractor_RRJ);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);



	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	
	// ********** For Blending **********
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);


	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	gPerspectiveProjectionMatrix_RRJ = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}

void uninitialize(void) {

	if(texturePos_RRJ){
		glDeleteTextures(1, &texturePos_RRJ);
		texturePos_RRJ = 0;
	}

	if(textureVelo_RRJ){
		glDeleteTextures(1, &textureVelo_RRJ);
		textureVelo_RRJ = 0;
	}


	if(vbo_Attractor_RRJ){
		glDeleteBuffers(1, &vbo_Attractor_RRJ);
		vbo_Attractor_RRJ = 0;
	}


	if(vbo_Points_Velo_RRJ){
		glDeleteBuffers(1, &vbo_Points_Velo_RRJ);
		vbo_Points_Velo_RRJ = 0;
	}

	if(vbo_Points_Pos_RRJ){
		glDeleteBuffers(1, &vbo_Points_Pos_RRJ);
		vbo_Points_Pos_RRJ = 0;
	}

	if(vao_Points_RRJ){
		glDeleteVertexArrays(1, &vao_Points_RRJ);
		vao_Points_RRJ = 0;
	}


	GLsizei ShaderCount_RRJ;
	GLsizei ShaderNumber_RRJ;

	if (gShaderProgramObject_RRJ) {
		glUseProgram(gShaderProgramObject_RRJ);

		glGetProgramiv(gShaderProgramObject_RRJ, GL_ATTACHED_SHADERS, &ShaderCount_RRJ);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount_RRJ);
		if (pShader) {
			glGetAttachedShaders(gShaderProgramObject_RRJ, ShaderCount_RRJ, &ShaderCount_RRJ, pShader);
			for (ShaderNumber_RRJ = 0; ShaderNumber_RRJ < ShaderCount_RRJ; ShaderNumber_RRJ++) {
				glDetachShader(gShaderProgramObject_RRJ, pShader[ShaderNumber_RRJ]);
				glDeleteShader(pShader[ShaderNumber_RRJ]);
				pShader[ShaderNumber_RRJ] = 0;
			}
			free(pShader);
			pShader = NULL;
		}
		glDeleteProgram(gShaderProgramObject_RRJ);
		gShaderProgramObject_RRJ = 0;
		glUseProgram(0);
	}


	if(gComputeShaderProgramObject_RRJ){
		glUseProgram(gComputeShaderProgramObject_RRJ);
			
			if(gComputeShaderObject_RRJ){
				glDetachShader(gComputeShaderProgramObject_RRJ, gComputeShaderObject_RRJ);
				glDeleteShader(gComputeShaderObject_RRJ);
				gComputeShaderObject_RRJ = 0;
			}

		glUseProgram(0);
		glDeleteProgram(gComputeShaderProgramObject_RRJ);
		gComputeShaderProgramObject_RRJ = 0;
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
		fprintf(gbFile_RRJ, "Log Close!!\n");
		fclose(gbFile_RRJ);
		gbFile_RRJ = NULL;
	}
}

void resize(int width, int height) {
	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	gPerspectiveProjectionMatrix_RRJ = mat4::identity();
	gPerspectiveProjectionMatrix_RRJ = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 1000.0f);
}


void display(void) {

	GLfloat RandomNoInRange(GLfloat, GLfloat);
	GLfloat RandomFloat(void);


	mat4 translateMatrix_RRJ;
	mat4 rotateMatrix_RRJ;
	mat4 modelMatrix_RRJ;
	mat4 viewMatrix_RRJ;

	GLfloat fTime_RRJ;
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	if(gbStart_RRJ == true){

		// ********** For Time **********
		static GLint iStartTick_RRJ = GetTickCount() - 100000;
		GLint iCurrentTick_RRJ = GetTickCount();
		fTime_RRJ = ((iStartTick_RRJ - iCurrentTick_RRJ) &  0xFFFFF) / float(0xFFFFF);
		
		/*fprintf(gbFile_RRJ, "\n");
		fprintf(gbFile_RRJ, "Start Tick : 0x%X\n", iStartTick_RRJ);
		fprintf(gbFile_RRJ, "Current Tick : 0x%X\n", iCurrentTick_RRJ);
		fprintf(gbFile_RRJ, "Last Tick : 0x%X\n", iLastTick_RRJ);
		fprintf(gbFile_RRJ, "Start - Current : 0x%X\n", (iStartTick_RRJ - iCurrentTick_RRJ));
		fprintf(gbFile_RRJ, "(Start - Current) & 0xFFFFF : 0x%X\n", (iStartTick_RRJ - iCurrentTick_RRJ) & 0xFFFFF);
		fprintf(gbFile_RRJ,  "((Start - Current) & 0xFFFFF) / (float)0xFFFFF : %f\n", fTime_RRJ);*/


		// *********** For Attractors ***********
		glBindBuffer(GL_UNIFORM_BUFFER, vbo_Attractor_RRJ);

		vec4 *pAttractor = (vec4*)glMapBufferRange(
			GL_UNIFORM_BUFFER, 
			0,
			RRJ_MAX_ATTRACTOR * sizeof(vec4),
			GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

		for(int  i = 0; i < RRJ_MAX_ATTRACTOR; i++){
			pAttractor[i] = vec4(
				sinf(fTime_RRJ * (float)(i + 4) * 7.5f * 20.0f) * 50.0f,
				cosf(fTime_RRJ * (float)(i + 7) * 3.9f * 20.0f) * 50.0f,
				sinf(fTime_RRJ * (float)(i + 3) * 5.3f * 20.0f) * cosf(fTime_RRJ * (float)(i + 5) * 9.1f) * 100.0f,
				fAttractorMass[i]);	// <<-- MASS of Attractor

		}

		glUnmapBuffer(GL_UNIFORM_BUFFER);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);



		// *********** Compute Shader **********
		glUseProgram(gComputeShaderProgramObject_RRJ);

			glBindVertexArray(vao_Points_RRJ);

				glUniform1f(timeUniform_RRJ,  gTime_RRJ);
				glDispatchCompute(RRJ_GROUP_X, 1, 1);
				glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
			
			glBindVertexArray(0);
			
		glUseProgram(0);


		// ********** Rendering **********
		glUseProgram(gShaderProgramObject_RRJ);

			/********** Points **********/
			translateMatrix_RRJ = mat4::identity();
			rotateMatrix_RRJ = mat4::identity();
			modelMatrix_RRJ = mat4::identity();
			viewMatrix_RRJ = mat4::identity();


			static GLfloat fRotateY_RRJ = 0.0f;

			translateMatrix_RRJ = translate(0.0f, 0.0f, -300.00f);
			rotateMatrix_RRJ = rotate(fRotateY_RRJ, 0.0f, 1.0f, 0.0f);
			modelMatrix_RRJ = modelMatrix_RRJ * translateMatrix_RRJ * rotateMatrix_RRJ;

			fRotateY_RRJ = fRotateY_RRJ + 0.15f;

			glUniformMatrix4fv(modelMatrixUniform_RRJ, 1, GL_FALSE, modelMatrix_RRJ);
			glUniformMatrix4fv(viewMatrixUniform_RRJ, 1, GL_FALSE, viewMatrix_RRJ);
			glUniformMatrix4fv(projectionMatrixUniform_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);


			//glPointSize(1.25f);
			glBindVertexArray(vao_Points_RRJ);
				glDrawArrays(GL_POINTS, 0, RRJ_MAX_PARTICALS);
			glBindVertexArray(0);

		glUseProgram(0);

	}


	SwapBuffers(ghdc_RRJ);
}



GLfloat RandomNoInRangeV2(GLfloat fStart, GLfloat fEnd){

	GLfloat RandomFloat(void);

	GLfloat fDiff_RRJ = fabs(fEnd - fStart);

	GLfloat fNumber_RRJ = fStart + (fDiff_RRJ / (GLfloat)RAND_MAX);

	return(fNumber_RRJ);
}



GLfloat RandomFloat(void){

	static unsigned int seed = 0x13371337;
	unsigned int k = 16807;
	float res;
	unsigned int temp;

	seed = seed * k;

	temp = seed ^ (seed >> 4) ^ (seed << 15);

	*((unsigned int*)&res) = (temp >> 9) | 0X3F800000;

	return(res - 1.0f); 
}

vec3 RandomVec3(GLfloat fStart, GLfloat fEnd){

	vec3 temp = vec3(
		RandomFloat() * 2.0f - 1.0f,
		RandomFloat() * 2.0f - 1.0f,
		RandomFloat() * 2.0f - 1.0f);

	temp = normalize(temp);

	temp = temp * (RandomFloat() * (fEnd - fStart) + fStart);

	return(temp);
}
