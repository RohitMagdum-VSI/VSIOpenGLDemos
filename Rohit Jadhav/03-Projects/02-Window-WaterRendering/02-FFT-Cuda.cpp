#include<Windows.h>
#include<stdio.h>
#include<gl/glew.h>
#include<GL/GL.h>
#include"vmath.h"


#include<cuda_gl_interop.h>
#include<cuda_runtime.h>
#include<cuda.h>
#include<cufft.h>


#include"02-FFT-Cuda.h"


#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")

#pragma comment(lib, "cudart.lib")

enum {
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOORD0,
	AMC_ATTRIBUTE_HEIGHTMAP,
	AMC_ATTRIBUTE_SLOPE,
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


//For Wave
bool bOnGPU_RRJ = false;
cudaError_t error_RRJ;
cufftHandle fftPlan_RRJ;

GLuint vao_Wave_RRJ;
GLuint vbo_WavePos_RRJ;
GLuint vbo_WaveHeight_RRJ;
GLuint vbo_WaveSlope_RRJ;
GLuint vbo_WaveElements_RRJ;


//For Cuda Interop
struct cudaGraphicsResource *gHeight_GraphicsResource_RRJ;
struct cudaGraphicsResource *gSlope_GraphicsResource_RRJ;


//For Wave Data
float2 *h_h0 = NULL;
float2 *d_h0 = NULL;

float2 *d_ht = NULL;
float2 *d_Slope = NULL;

//For Pointer To Device Object used in display
float *g_hPtr = NULL;
float2 *g_sPtr = NULL;


//For Wave
#define RRJ_PI 3.1415926535f

const GLuint gMeshSize = 512;
const GLuint gSpectrumW = gMeshSize + 4;
const GLuint gSpectrumH = gMeshSize + 1;

const GLfloat gfGravity = 9.81f;
const GLfloat gfAmplitude = 1E-7f;
const GLfloat gfPatchSize = 100.0f;
GLfloat gfWindSpeed = 5.0f;
GLfloat gfWindDirection = RRJ_PI / 3.0f;
GLfloat gfDirDepend = 0.07f;
GLfloat gfAnimationTime = 1000.0f;
GLfloat gfAnimFactor = 0.050f;



//For VS Uniform
GLuint modelMatrixUniform_RRJ;
GLuint viewMatrixUniform_RRJ;
GLuint projectionMatrixUniform_RRJ;

GLuint meshSizeUniform_RRJ;
GLuint heightScaleUniform_RRJ;
GLuint chopinessUniform_RRJ;

// For FS Uniform
GLuint deepColorUniform_RRJ;
GLuint shallowColorUniform_RRJ;
GLuint skyColorUniform_RRJ;
GLuint lightDirectionUniform_RRJ;

GLfloat deepColor[] = {0.0f, 0.1f, 0.4f, 1.0f};
GLfloat shallowColor[] = {0.1f, 0.3f, 0.3f, 1.0f};
GLfloat skyColor[] = {1.0f, 1.0f, 1.0f, 1.0f};
GLfloat lightDir[] = {0.0f, 1.0f, 0.0f};
GLfloat heightScale = 0.1f;
GLfloat chopiness = 1.0f;
GLfloat mesh[] = {gMeshSize, gMeshSize};

GLfloat wavePos[gMeshSize * gMeshSize * 4];


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

	int iRet_RRJ;
	bool bDone_RRJ = false;

	WNDCLASSEX wndclass_RRJ;
	HWND hwnd_RRJ;
	MSG msg_RRJ;
	TCHAR szName_RRJ[] = TEXT("RohitRJadhav-PP-02-FFT-Cuda");

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

	hwnd_RRJ = CreateWindowEx(WS_EX_APPWINDOW,
		szName_RRJ,
		TEXT("RohitRJadhav-PP-02-FFT-Cuda"),
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
		fprintf(gbFile_RRJ, "wglMakeCurrent() Failed!!\n");
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

		case 'G':
		case 'g':
			bOnGPU_RRJ = true;
			break;

		case 'C':
		case 'c':
			bOnGPU_RRJ = false;
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


	int devCount_RRJ = 0;
	error_RRJ = cudaGetDeviceCount(&devCount_RRJ);
	if(error_RRJ != cudaSuccess){
		fprintf(gbFile_RRJ, "cudaGetDeviceCount() Failed!!\n");
		uninitialize();
		exit(0);
	}
	else if(devCount_RRJ == 0){
		fprintf(gbFile_RRJ, "devCount_RRJ == 0\n");
		uninitialize();
		exit(0);
	}
	else{
		fprintf(gbFile_RRJ, "DevCount: %d\n", devCount_RRJ);
		cudaSetDevice(0);
	}




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

	/********** Vertex Shader **********/
	iVertexShaderObject_RRJ = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *szVertexShaderSourceCode_RRJ =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in float vHeight;" \
		"in vec2 vSlope;" \

		"uniform mat4 u_modelMatrix;" \
		"uniform mat4 u_viewMatrix;" \
		"uniform mat4 u_projectionMatrix;" \

		"uniform float u_heightScale;" \
		"uniform float u_chopiness;" \
		"uniform vec2 u_meshSize;" \


		"out vec3 eyeSpacePos_VS;" \
		"out vec3 worldSpaceNormal_VS;" \
		"out vec3 eyeSpaceNormal_VS;" \



		"void main(void)" \
		"{" \
			"mat3 normalMatrix = mat3(u_viewMatrix * u_modelMatrix); " \

			"vec3 normal = normalize(cross(vec3(0.0f, vSlope.y * u_heightScale, 2.0f / u_meshSize.x), vec3(2.0f / u_meshSize.y, vSlope.x * u_heightScale, 0.0f)));" \

			"worldSpaceNormal_VS = normal;" \

			"vec4 pos = vec4(vPosition.x, vHeight * u_heightScale, vPosition.z, 1.0f);" \
			"gl_Position = u_projectionMatrix * u_viewMatrix * u_modelMatrix * pos;" \

			"eyeSpacePos_VS = vec3(u_viewMatrix * u_modelMatrix * pos);" \
			"eyeSpaceNormal_VS = vec3(normalMatrix * normal);" \

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

	const GLchar *szFragmentShaderSourceCode_RRJ =
		"#version 450 core" \
		"\n" \

		"in vec3 eyeSpacePos_VS;" \
		"in vec3 worldSpaceNormal_VS;" \
		"in vec3 eyeSpaceNormal_VS;" \

		"uniform vec4 u_deepColor;" \
		"uniform vec4 u_shallowColor;" \
		"uniform vec4 u_skyColor;" \
		"uniform vec3 u_lightDirection;" \


		"out vec4 FragColor;" \

		"void main(void)" \
		"{" \
			
			"vec3 eyeSpacePosVec = normalize(eyeSpacePos_VS);" \
			"vec3 eyeSpaceNorVec = normalize(eyeSpaceNormal_VS);" \
			"vec3 worldSpaceNorVec = normalize(worldSpaceNormal_VS);" \


			"float facing = max(0.0f, dot(eyeSpaceNormal_VS, -eyeSpacePos_VS));" \
			"float fresnel = pow(1.0f - facing, 5.0f);" \
			"float diffuse = max(0.0f, dot(worldSpaceNormal_VS, u_lightDirection));" \

			"vec4 waterColor = u_deepColor;" \

			// "vec4 waterColor = mix(u_shallowColor, u_deepColor, facing);"

			"FragColor = waterColor * diffuse + u_skyColor * fresnel;" \
			
			//"FragColor = vec4(1.0f);" \

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
	glBindAttribLocation(gShaderProgramObject_RRJ, AMC_ATTRIBUTE_HEIGHTMAP, "vHeight");
	glBindAttribLocation(gShaderProgramObject_RRJ, AMC_ATTRIBUTE_SLOPE, "vSlope");

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
				fprintf(gbFile_RRJ, "Shader Program Object Linking Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}


	modelMatrixUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_modelMatrix");
	viewMatrixUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_viewMatrix");
	projectionMatrixUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_projectionMatrix");

	heightScaleUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_heightScale");
	chopinessUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_chopiness");
	meshSizeUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_meshSize");

	deepColorUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_deepColor");
	shallowColorUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_shallowColor");
	skyColorUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_skyColor");
	lightDirectionUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_lightDirection");





	// ***** Wave Position *****
	memset(wavePos, 0.0f, sizeof(GLfloat) * gMeshSize * gMeshSize * 4);

	int index = 0;

	for(int y = 0; y < gMeshSize; y++){

		for(int x = 0; x < gMeshSize; x++){

			index = (y * gMeshSize * 4) + (x * 4);	

			GLfloat u = x / (GLfloat)(gMeshSize - 1);
			GLfloat v = y / (GLfloat)(gMeshSize - 1);

			//fprintf(gbFile_RRJ, "Index : %d\n", index);

			wavePos[index + 0] = u * 2.0f - 1.0f;
			wavePos[index + 1] = 0.0f;
			wavePos[index + 2] = v * 2.0f - 1.0f;
			wavePos[index + 3] = 1.0f;

		}
	}

	GLuint size = ((gMeshSize * 2) + 2) * (gMeshSize -1) * sizeof(GLuint);
	//For Information ((w * 2) + 2) * (h - 1) * size(GLuint);


	// ***** For Wave *****
	glGenVertexArrays(1, &vao_Wave_RRJ);
	glBindVertexArray(vao_Wave_RRJ);

		// ***** Position *****
		glGenBuffers(1, &vbo_WavePos_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_WavePos_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 4 * gMeshSize * gMeshSize, wavePos, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		// ***** Slope *****
		glGenBuffers(1, &vbo_WaveSlope_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_WaveSlope_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * gMeshSize * gMeshSize, NULL, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_SLOPE, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_SLOPE);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		// ***** Height *****
		glGenBuffers(1, &vbo_WaveHeight_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_WaveHeight_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * gMeshSize * gMeshSize, NULL, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_HEIGHTMAP, 1, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_HEIGHTMAP);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		// ***** Index *****
		glGenBuffers(1, &vbo_WaveElements_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_WaveElements_RRJ);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);

		GLuint *indices = (GLuint*)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);

		if(indices == NULL){
			fprintf(gbFile_RRJ, "ERROR: glMapBuffer() For Elements Failed\n");
			uninitialize();
			DestroyWindow(ghwnd_RRJ);
		}


		for(int y = 0; y < gMeshSize - 1; y++){

			for(int x = 0; x < gMeshSize; x++){

				*indices++ = y * gMeshSize + x;
				*indices++ = (y + 1) * gMeshSize + x;

			}

			*indices++ = (y + 1) * gMeshSize + (gMeshSize - 1);
			*indices++ = (y + 1) * gMeshSize;
		}

		glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


	glBindVertexArray(0);



	// ***** Graphics Resource For Height Map *****
	error_RRJ = cudaGraphicsGLRegisterBuffer(
				&gHeight_GraphicsResource_RRJ, 
				vbo_WaveHeight_RRJ, 
				cudaGraphicsMapFlagsWriteDiscard);

	if(error_RRJ != cudaSuccess){
		fprintf(gbFile_RRJ, "ERROR: cudaGraphicsGLRegisterBuffer() For HeightMap Failed\n");
		uninitialize();
		DestroyWindow(ghwnd_RRJ);
	}


	// ***** Graphics Resource For Slope *****
	error_RRJ = cudaGraphicsGLRegisterBuffer(
				&gSlope_GraphicsResource_RRJ,
				vbo_WaveSlope_RRJ,
				cudaGraphicsMapFlagsWriteDiscard);

	if(error_RRJ != cudaSuccess){
		fprintf(gbFile_RRJ, "ERROR: cudaGraphicsGLRegisterBuffer() For Slope Failed\n");
		uninitialize();
		DestroyWindow(ghwnd_RRJ);
	}



	// ***** Memory For Arrays *****
	GLuint spectrumSize = gSpectrumW * gSpectrumH * sizeof(float2);
	GLuint meshSize = gMeshSize * gMeshSize * sizeof(float);
	

	// *** Host Memory ***
	h_h0 = (float2*)malloc(spectrumSize);
	if(h_h0 == NULL){
		fprintf(gbFile_RRJ, "ERROR: malloc() for h_h0 Failed\n");
		uninitialize();
		DestroyWindow(ghwnd_RRJ);
	}

	// *** Device Memory ***
	error_RRJ = cudaMalloc((void**)&d_h0, spectrumSize);
	if(error_RRJ != cudaSuccess){
		fprintf(gbFile_RRJ, "ERROR: cudaMalloc() Failed for d_h0 with : %s\n", cudaGetErrorString(error_RRJ));
		uninitialize();
		DestroyWindow(ghwnd_RRJ);
	}

	error_RRJ = cudaMalloc((void**)&d_ht, meshSize);
	if(error_RRJ != cudaSuccess){
		fprintf(gbFile_RRJ, "ERROR: cudaMalloc() failed for d_ht with : %s\n", cudaGetErrorString(error_RRJ));
		uninitialize();
		DestroyWindow(ghwnd_RRJ);
	}


	error_RRJ = cudaMalloc((void**)&d_Slope, meshSize);
	if(error_RRJ != cudaSuccess){
		fprintf(gbFile_RRJ, "ERROR: cudaMalloc() failed for d_Slope with : %s\n", cudaGetErrorString(error_RRJ));
		uninitialize();
		DestroyWindow(ghwnd_RRJ);
	}



	// ***** Genenrate Initial Height Field *****
	Generate_H0(h_h0);

	error_RRJ = cudaMemcpy(d_h0, h_h0, spectrumSize, cudaMemcpyHostToDevice);
	if(error_RRJ != cudaSuccess){
		fprintf(gbFile_RRJ, "ERROR: cudaMemcpy() failed for h_h0 -> d_h0 with : %s\n", cudaGetErrorString(error_RRJ));
		uninitialize();
		DestroyWindow(ghwnd_RRJ);
	}


	cufftResult fftResult;

	fftResult = cufftPlan2d(&fftPlan_RRJ, gMeshSize, gMeshSize, CUFFT_C2C);
	if(fftResult != CUFFT_SUCCESS){
		fprintf(gbFile_RRJ, "ERROR: cufftPlan2d");
		uninitialize();
		DestroyWindow(ghwnd_RRJ);
	}



	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);
	
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	gPerspectiveProjectionMatrix_RRJ = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}

void uninitialize(void) {

	
	cudaGraphicsUnregisterResource(gHeight_GraphicsResource_RRJ);
	cudaGraphicsUnregisterResource(gSlope_GraphicsResource_RRJ);


	cufftDestroy(fftPlan_RRJ);

	if(d_Slope){
		cudaFree(d_Slope);
		d_Slope = NULL;
	}

	if(d_ht){
		cudaFree(d_ht);
		d_ht = NULL;
	}

	if(d_h0){
		cudaFree(d_h0);
		d_h0 = NULL;
	}

	if(h_h0){
		free(h_h0);
		h_h0 = NULL;
	}


	if(vbo_WaveElements_RRJ){
		glDeleteBuffers(1, &vbo_WaveElements_RRJ);
		vbo_WaveElements_RRJ = 0;
	}

	if(vbo_WaveHeight_RRJ){
		glDeleteBuffers(1, &vbo_WaveHeight_RRJ);
		vbo_WaveHeight_RRJ = 0;
	}

	if(vbo_WaveSlope_RRJ){
		glDeleteBuffers(1, &vbo_WaveSlope_RRJ);
		vbo_WaveSlope_RRJ = 0;
	}

	if(vbo_WavePos_RRJ){
		glDeleteBuffers(1, &vbo_WavePos_RRJ);
		vbo_WavePos_RRJ = NULL;
	}

	if(vao_Wave_RRJ){
		glDeleteVertexArrays(1, &vao_Wave_RRJ);
		vao_Wave_RRJ = NULL;
	}




	GLsizei ShaderCount_RRJ;
	GLsizei ShaderNumber_RRJ;

	if (gShaderProgramObject_RRJ) {
		glUseProgram(gShaderProgramObject_RRJ);

		glGetProgramiv(gShaderProgramObject_RRJ, GL_ATTACHED_SHADERS, &ShaderCount_RRJ);

		GLuint *pShader_RRJ = (GLuint*)malloc(sizeof(GLuint) * ShaderCount_RRJ);
		if (pShader_RRJ) {
			glGetAttachedShaders(gShaderProgramObject_RRJ, ShaderCount_RRJ,
				&ShaderCount_RRJ, pShader_RRJ);
			for (ShaderNumber_RRJ = 0; ShaderNumber_RRJ < ShaderCount_RRJ; ShaderNumber_RRJ++) {
				glDetachShader(gShaderProgramObject_RRJ, pShader_RRJ[ShaderNumber_RRJ]);
				glDeleteShader(pShader_RRJ[ShaderNumber_RRJ]);
				pShader_RRJ[ShaderNumber_RRJ] = 0;
			}
			free(pShader_RRJ);
			pShader_RRJ = NULL;
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
		fprintf(gbFile_RRJ, "Log Close!!\n");
		fclose(gbFile_RRJ);
		gbFile_RRJ = NULL;
	}
}

void resize(int width, int height) {
	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	gPerspectiveProjectionMatrix_RRJ = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}


void display(void) {

	
	mat4 TranslateMatrix_RRJ;
	mat4 ModelMatrix_RRJ;
	mat4 ViewMatrix_RRJ;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(gShaderProgramObject_RRJ);

	TranslateMatrix_RRJ = mat4::identity();
	ModelMatrix_RRJ = mat4::identity();
	ViewMatrix_RRJ = mat4::identity();


	TranslateMatrix_RRJ = translate(0.0f, 0.0f, -1.0f);
	ModelMatrix_RRJ = ModelMatrix_RRJ * TranslateMatrix_RRJ * rotate(10.0f, 1.0f, 0.0f, 0.0f);
	


	glUniformMatrix4fv(modelMatrixUniform_RRJ, 1, GL_FALSE, ModelMatrix_RRJ);
	glUniformMatrix4fv(viewMatrixUniform_RRJ, 1, GL_FALSE, ViewMatrix_RRJ);
	glUniformMatrix4fv(projectionMatrixUniform_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	



		if(bOnGPU_RRJ == true){
			

			cudaGenerateSpectrumKernel(d_h0, d_ht, gSpectrumW, gMeshSize, gMeshSize, gfAnimationTime, gfPatchSize);

			//FFT
			cufftResult fftResult;
			fftResult = cufftExecC2C(fftPlan_RRJ, d_ht, d_ht, CUFFT_INVERSE);
			if(fftResult != CUFFT_SUCCESS){
				fprintf(gbFile_RRJ, "ERROR: cufftExecC2C() failed");
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
			}


			// *** HeightMap ***
			size_t numOfBytes = 0;
			error_RRJ = cudaGraphicsMapResources(1, &gHeight_GraphicsResource_RRJ, 0);
			if(error_RRJ != cudaSuccess){
				fprintf(gbFile_RRJ, "ERROR: cudaGraphicsMapResource() failed for gHeight_GraphicsResource_RRJ\n");
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
			}


			error_RRJ = cudaGraphicsResourceGetMappedPointer((void**)&g_hPtr, &numOfBytes, gHeight_GraphicsResource_RRJ);
			if(error_RRJ != cudaSuccess){
				fprintf(gbFile_RRJ, "ERROR: cudaGraphicsResourceGetMappedPointer() failed for gHeight_GraphicsResource_RRJ\n");
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
			}


			cudaUpdateHeightMapKernel(g_hPtr, d_ht, gMeshSize, gMeshSize);



			// *** Slope ***
			error_RRJ = cudaGraphicsMapResources(1, &gSlope_GraphicsResource_RRJ, 0);
			if(error_RRJ != cudaSuccess){
				fprintf(gbFile_RRJ, "ERROR: cudaGraphicsMapResource() failed for gSlope_GraphicsResource_RRJ\n");
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
			}


			error_RRJ = cudaGraphicsResourceGetMappedPointer((void**)&g_sPtr, &numOfBytes, gSlope_GraphicsResource_RRJ);
			if(error_RRJ != cudaSuccess){
				fprintf(gbFile_RRJ, "ERROR: cudaGraphicsResourceGetMappedPointer() failed for gSlope_GraphicsResource_RRJ\n");
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
			}

			cudaCalculateSlopeKernel(g_hPtr, g_sPtr, gMeshSize, gMeshSize);


			// *** Unmap ***
			error_RRJ = cudaGraphicsUnmapResources(1, &gSlope_GraphicsResource_RRJ, 0);
			if(error_RRJ != cudaSuccess){
				fprintf(gbFile_RRJ, "ERROR: cudaGraphicsUnmapResource() failed for gSlope_GraphicsResource_RRJ\n");
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
			}

			error_RRJ = cudaGraphicsUnmapResources(1, &gHeight_GraphicsResource_RRJ, 0);
			if(error_RRJ != cudaSuccess){
				fprintf(gbFile_RRJ, "ERROR: cudaGraphicsUnmapResource() failed for gHeight_GraphicsResource_RRJ\n");
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
			}


			glUniform4fv(deepColorUniform_RRJ, 1, deepColor);
			glUniform4fv(shallowColorUniform_RRJ, 1, shallowColor);
			glUniform4fv(skyColorUniform_RRJ, 1, skyColor);
			glUniform3fv(lightDirectionUniform_RRJ, 1, lightDir);

			glUniform1f(heightScaleUniform_RRJ, heightScale);
			glUniform1f(chopinessUniform_RRJ, chopiness);
			glUniform2fv(meshSizeUniform_RRJ, 1, mesh);

			glBindVertexArray(vao_Wave_RRJ);
			//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
				
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_WaveElements_RRJ);

			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glDrawElements(GL_TRIANGLE_STRIP, ((gMeshSize * 2) + 2) * (gMeshSize - 1),  GL_UNSIGNED_INT, 0);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

			glBindVertexArray(0);


		}
		else{
			//CPU

			

		}




	glUseProgram(0);

	SwapBuffers(ghdc_RRJ);


	//animationTime_RRJ = 1.0;
	gfAnimationTime = gfAnimationTime + gfAnimFactor;

}


GLfloat urand(void){
	return(rand() / RAND_MAX);
}


GLfloat Gauss(void){

	GLfloat u1 = urand();
	GLfloat u2 = urand();

	if(u1 < 1e-6)
		u1 = 1e-6;


	GLfloat ret = sqrt(-2 * logf(u1)) * cosf(-2 * RRJ_PI * u2);
	return(ret);
}



// Phillips spectrum
GLfloat Philips(float Kx, float Ky, float windDir, float windSpeed, float A, float dirDepend){

	float k_square = Kx * Kx + Ky * Ky;

	if(k_square == 0.0f)
		return(0.0f);


	float L = windSpeed * windSpeed / gfGravity;

	float k_x = Kx / sqrt(k_square);
	float k_y = Ky / sqrt(k_square);

	float w_dot_k = k_x * cosf(windDir) + k_y * sinf(windDir);

	float philips = A * exp(-1.0f /(k_square * L * L)) / (k_square * k_square) * w_dot_k * w_dot_k;

	if(w_dot_k < 0.0f){
		philips = philips * dirDepend;
	}

	return(philips);
}


void Generate_H0(float2 *h0){


	for(unsigned int y = 0; y <= gMeshSize; y++){

		for(unsigned int x = 0; x <= gMeshSize; x++){

			float kx = (-(int)gMeshSize / 2 + x) * (2.0f * RRJ_PI / gfPatchSize);
			float ky = (-(int)gMeshSize / 2 + y) * (2.0f * RRJ_PI / gfPatchSize);

			float p = sqrt(Philips(kx, ky, gfWindDirection, gfWindSpeed, gfAmplitude, gfDirDepend));

			if(kx == 0.0f && ky == 0.0f)
				p = 0.0f;

			float Er = Gauss();
			float Ei = Gauss();

			float h0_r = Er * p * sqrt(0.5f);
			float h0_i = Ei * p * sqrt(0.5f);

			int i = y * gSpectrumW + x;
			h0[i].x = h0_r;
			h0[i].y = h0_i;
 		}
	}
}


