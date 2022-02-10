#include<Windows.h>
#include<stdio.h>
#include<gl/glew.h>
#include<GL/GL.h>
#include"vmath.h"
#include"Sphere.h"


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
extern FILE *gbFile = NULL;

//For Shader Program Object;
GLint gShaderProgramObject;

//For Perspective Matric
mat4 gPerspectiveProjectionMatrix;

//For Rectangle
GLuint vao_Rect;
GLuint vbo_Rect_Position;
GLuint vbo_Rect_Tex;

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
GLfloat angle_Sphere = 0.0f;


//For Uniform
GLuint mvpUniform;
GLuint noiseFactorUniform;
GLuint octaveUniform;
GLuint lacunarityUniform;
GLuint gainUniform;
GLuint noiseAnimationUniform;


GLfloat gfNoiseFactor = 0.0f;
GLfloat gfLacunarity = 2.0f;
GLfloat gfGain = 0.50f;
GLuint giOctaves = 6;
GLfloat gfvNoiseAnimation[] = { 0.0f, 0.0f};


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
	TCHAR szName[] = TEXT("RohitRJadhav-PP-Noise-04-fbm_inside_fbm");

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
		TEXT("RohitRJadhav-PP-Noise-04-fbm_inside_fbm"),
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

		case 'w':
		case 'W':
			gfNoiseFactor += 1.0f;
			break;

		case 'S':
		case 's':
			gfNoiseFactor -= 1.0f;
			break;


		case 'O':
			giOctaves += 1;
			break;

		case 'o':
			giOctaves -= 1;
			break;

		case 'L':
			gfLacunarity += 0.2f;
			break;

		case  'l':
			gfLacunarity -= 0.2f;
			break;

		case 'G':
			gfGain += 0.2f;
			break;

		case 'g':
			gfGain -= 0.2f;
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
		"out vec3 outTexCoord;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void)" \
		"{" \
			"gl_Position = u_mvp_matrix * vPosition;" \
			"outTexCoord = vPosition.xyz;" \
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

		
		"in vec3 outTexCoord;" \
		"uniform float u_fNoiseFactor;" \
		"uniform vec2 u_v2NoiseAnimation;" \

		"uniform int u_iOctaves;" \
		"uniform float u_fLacunarity;" \
		"uniform float u_fGain;" \

		"out vec4 v4FragColor;" \
		

		"float value_random(vec2 st) {" \

			"float fDotProduct = dot(st.xy, vec2(12.9898, 78.233));" \
		
			"float fSin = sin(fDotProduct);" \
			
			"float fRet = fract(fSin * 43758.5453123);" \

			"return(fRet);" \

		"}" \


		"float value_noise(vec2 st){" \

			"vec2 v2IntPart = floor(st);" \
			"vec2 v2FractPart = fract(st);" \

			"float a = value_random(v2IntPart);" \
			"float b = value_random(v2IntPart + vec2(1.0f, 0.0f));" \
			"float c = value_random(v2IntPart + vec2(0.0f, 1.0f));" \
			"float d = value_random(v2IntPart + vec2(1.0f, 1.0f));" \

			"vec2 v2MixIntensity = v2FractPart * v2FractPart * (3.0f - 2.0f * v2FractPart);" \

			"float ab = mix(a, b, v2MixIntensity.x);" \
			"float cd = mix(c, d, v2MixIntensity.x);" \
			"float all = mix(ab, cd, v2MixIntensity.y);" \

			"return(all);" \

		"}" \


		"float fbm_noise(vec2 st){" \

			"float value = 0.0f;" \
			"float amplitude = 0.5f;" \
			"float frequency = 1.0f;" \

			"for(int i = 0; i < u_iOctaves; i++){" \

				"value = value + (amplitude * value_noise(frequency * st));" \

				"frequency = frequency * u_fLacunarity;" \

				"amplitude = amplitude * u_fGain;" \

			"}" \

			"return(value);" \

		"}" \





		"void main(void)" \
		"{" \
			
			
			"vec2 v2st = outTexCoord.xy;" \

			"v2st *= u_fNoiseFactor;" \

			"vec3 v3Color = vec3(0.0f, 0.0f, 0.0f);" \

			"vec2 p = v2st;" \

			"vec2 q = vec2(0.0f, 0.0f);" \

			"q.x = fbm_noise(p + vec2(0.0f) + 0.65f * u_v2NoiseAnimation.x);" \
			"q.y = fbm_noise(p + vec2(1.0f) + 0.65f * u_v2NoiseAnimation.y);" \
			

			"vec2 r = vec2(0.0f, 0.0f);" \

			"r.x = fbm_noise(p + 4.0f * q + vec2(1.7f, 9.2f) + 0.15 * u_v2NoiseAnimation.x);" \
			"r.y = fbm_noise(p + 4.0f * q + vec2(8.3f, 2.8f) + 0.15 * u_v2NoiseAnimation.y);" \

			"float f = fbm_noise(p + 4.0f * r);" \

			 // color = mix(vec3(0.101961,0.619608,0.666667),
				//                 vec3(0.666667,0.666667,0.498039),
				//                 clamp((f*f)*4.0,0.0,1.0));

				//     color = mix(color,
				//                 vec3(0,0,0.164706),
				//                 clamp(length(q),0.0,1.0));

				//     color = mix(color,
				//                 vec3(0.666667,1,1),
				//                 clamp(length(r.x),0.0,1.0));

   	// 		gl_FragColor = vec4((f*f*f+.6*f*f+.5*f)*color,1.);


			"v3Color = mix(vec3(0.101961f, 0.619608f, 0.666667f), vec3(0.666667f, 0.666667f, 0.498039f), clamp(f * f * 4.0f, 0.0f, 1.0f));" \

			"v3Color = mix(v3Color, vec3(0.0f, 0.0f, 0.164706f), clamp(length(q), 0.0f, 1.0f));" \

			//"v3Color = mix(v3Color, vec3(0.666667f, 1.0f, 1.0f), clamp(length(r.x), 0.0f, 1.0f));" \
			
			"v3Color = mix(v3Color, vec3(0.666667f, 1.0f, 0.0f), clamp(length(r.x), 0.0f, 1.0f));" \


			"v3Color = (f * f * f + 0.6 * f * f + 0.5 * f) * v3Color;" \

			"v4FragColor = vec4(v3Color, 1.0f);" \


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

	mvpUniform = glGetUniformLocation(gShaderProgramObject, "u_mvp_matrix");
	noiseFactorUniform = glGetUniformLocation(gShaderProgramObject, "u_fNoiseFactor");
	octaveUniform = glGetUniformLocation(gShaderProgramObject, "u_iOctaves");
	lacunarityUniform = glGetUniformLocation(gShaderProgramObject, "u_fLacunarity");
	gainUniform = glGetUniformLocation(gShaderProgramObject, "u_fGain");
	noiseAnimationUniform = glGetUniformLocation(gShaderProgramObject, "u_v2NoiseAnimation");




	/********** TexCoord **********/
	GLfloat Rect_Tex[] = {
		0.0f, 0.0f,
		0.0f, 1.0f,
		1.0f, 1.0f,
		1.0f, 0.0f
	};

	GLfloat Rect_Pos[] = {
		1.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
	};

	


	/********** Vao Rectangle **********/
	glGenVertexArrays(1, &vao_Rect);
	glBindVertexArray(vao_Rect);

	/********** Position **********/
	glGenBuffers(1, &vbo_Rect_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Position);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Rect_Pos),
		Rect_Pos,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0);



	/********** Texture **********/
	glGenBuffers(1, &vbo_Rect_Tex);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Tex);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Rect_Tex),
		Rect_Tex,
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



	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);


	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	gPerspectiveProjectionMatrix = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}




void uninitialize(void) {

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


	if (vbo_Rect_Tex) {
		glDeleteBuffers(1, &vbo_Rect_Tex);
		vbo_Rect_Tex = 0;
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

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	//fprintf(gbFile, "%d/ %d\n", width, height);

	gPerspectiveProjectionMatrix = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}




void display(void) {

	static GLfloat angle = 0.0f;

	mat4 translateMatrix;
	mat4 modelViewMatrix;
	mat4 modelViewProjectionMatrix;

	

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(gShaderProgramObject);

	translateMatrix = mat4::identity();
	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();

	// translateMatrix = translate(-1.0f, -1.0f, -3.0f) * scale(2.0f, 2.0f, 1.0f);

	translateMatrix = translate(0.0f, 0.0f, -3.0f) * scale(10.0f, 10.0f, 10.0f) ; //* rotate(angle, 0.0f, 1.0f, 0.0f);

	modelViewMatrix = modelViewMatrix * translateMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(mvpUniform,
		1,
		GL_FALSE,
		modelViewProjectionMatrix);

	glUniform1f(noiseFactorUniform, gfNoiseFactor);
	glUniform1f(lacunarityUniform, gfLacunarity);
	glUniform1f(gainUniform, gfGain);
	glUniform2fv(noiseAnimationUniform, 1, gfvNoiseAnimation);
	glUniform1i(octaveUniform, giOctaves);


	// glBindVertexArray(vao_Rect);
	// glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	// glBindVertexArray(0);

	glBindVertexArray(vao_Sphere);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	glUseProgram(0);

	SwapBuffers(ghdc);

	gfvNoiseAnimation[0] += 0.01f;
	gfvNoiseAnimation[1] += 0.01f;

	angle = angle + 0.1f;
	if(angle > 360.0f)
		angle = 0.0f;
	
}

