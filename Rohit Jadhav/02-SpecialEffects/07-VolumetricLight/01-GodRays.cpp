#include<Windows.h>
#include<stdio.h>
#include<gl/glew.h>
#include<GL/GL.h>
#include"vmath.h"
#include"Sphere.h"
#include"Resource.h"


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

//For Shader
GLuint gShaderProgramObject_Models;
GLuint gShaderProgramObject_GodRays;

//For Projection
mat4 gPerspectiveProjectionMatrix;
mat4 gOrthoProjectionMatrix;


//For Cube
GLuint texture_Cube;
GLuint kundali_Cube;
GLuint vao_Cube;
GLuint vbo_Cube_Position;
GLuint vbo_Cube_Texture;
GLfloat angle_Cube = 360.0f;

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


//For Movement
#define MOVE_CUBE 1
#define MOVE_SPHERE 2
GLuint giWhichMovement = MOVE_CUBE;
GLfloat gfMoveFactor = 0.50f;

GLfloat cubeX = -5.0f;
GLfloat cubeY = 0.0f;
GLfloat cubeZ = -4.0f;
GLfloat sphereX = 0.0f;
GLfloat sphereY = 0.0f;
GLfloat sphereZ = -8.0f;
GLfloat XOffset = 0.0f;
GLfloat YOffset = 0.0f;




//For Uniforms
GLuint mvpUniform;
GLuint samplerUniform;
GLuint choiceUniform;


//For GodRays

//For Rectangle
GLuint vao_Rect;
GLuint vbo_Rect_Position;
GLuint vbo_Rect_TexCoord;

GLuint samplerFirstPassUniform;
GLuint mvpOrtho_Uniform_GodRays;
GLuint mvpPerspective_Uniform_GodRays;
GLuint viewPortUniform_GodRays;


GLuint exposureUniform;
GLuint decayUniform;
GLuint densityUniform;
GLuint weightUniform;
GLuint lightPositionUniform;

GLfloat gfExposure = 0.0034f;
GLfloat gfDecay = 1.0f;
GLfloat gfDensity = 0.95f;
GLfloat gfWeight = 5.65f;



//For Framebuffer
GLuint frameBufferObject;
GLuint renderBufferObject_Depth;	

#define SCALE_RATE 2

GLint viewPortWidth = 1366;
GLint viewPortHeight = 768;

GLint viewPort_FBO_Width = viewPortWidth / SCALE_RATE;
GLint viewPort_FBO_Height = viewPortHeight / SCALE_RATE;

// GLfloat gfvLightPosition[2] = {viewPort_FBO_Width, viewPort_FBO_Height};
GLfloat gfvLightPosition[4] = {0.0f, 0.0f, 0.0f, 1.0f};



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
	TCHAR szName[] = TEXT("RohitRJadhav-PP-GodRays-01-GodRays");

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
		TEXT("RohitRJadhav-PP-GodRays-01-GodRays"),
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

	case WM_KEYDOWN:
		switch(wParam){
			case VK_UP:
			if(giWhichMovement == MOVE_CUBE)
				cubeY += gfMoveFactor;
			else
				sphereY += gfMoveFactor;
			gfvLightPosition[1] = sphereY + YOffset;

			break;

		case VK_DOWN:
			if(giWhichMovement == MOVE_CUBE)
				cubeY -= gfMoveFactor;
			else
				sphereY -= gfMoveFactor;

			gfvLightPosition[1] = sphereY + YOffset;
			break;

		
		case VK_RIGHT:
			if(giWhichMovement == MOVE_CUBE)
				cubeX += gfMoveFactor;
			else
				sphereX += gfMoveFactor;
			gfvLightPosition[0] = sphereX + XOffset;
			break;


		case VK_LEFT:
			if(giWhichMovement == MOVE_CUBE)
				cubeX -= gfMoveFactor;
			else
				sphereX -= gfMoveFactor;
			gfvLightPosition[0] = sphereX + XOffset;
			break;
		}
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


		case 'S':
		case 's':
			giWhichMovement = MOVE_SPHERE;
			break;

		case 'C':
		case 'c':
			giWhichMovement = MOVE_CUBE;
			break;


		case 'x':
		case 'X':
			fprintf(gbFile, "%f, %f, %f\n", sphereX, sphereY, sphereZ);
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
	void initialize_Objects();
	void initialize_GodRays();

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

		
	initialize_Objects();
	initialize_GodRays();	


	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);


	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	gPerspectiveProjectionMatrix = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}


void initialize_Objects(void){

	void uninitialize(void);
	void resize(int, int);
	GLuint LoadTexture(TCHAR imageResourceID[]);

	//Shader Object;
	GLint iVertexShaderObject;
	GLint iFragmentShaderObject;


	/********** Vertex Shader **********/
	iVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *szVertexShaderSourceCode =
		"#version 450 core" \
		"\n" \

		//For Choice
		"uniform int u_choice;" \

		//For Attribute
		"in vec4 vPosition;" \
		"in vec2 vTexCoord;" \
		"out vec2 outTexCoord;" \

		//For Cube
		"uniform mat4 u_mvp_matrix;" \

		"void main(void)" \
		"{" \
			"gl_Position = u_mvp_matrix * vPosition;" \
			"outTexCoord = vTexCoord;" \

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
				exit(0);
			}
		}
	}

	/********** Fragment Shader **********/
	iFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *szFragmentShaderSourceCode =
		"#version 450 core" \
		"\n" \

		//For Choice
		"uniform int u_choice;" \

		//For Cube
		"in vec2 outTexCoord;" \
		"uniform sampler2D u_sampler;" \

		//For Output
		"out vec4 FragColor;" \

		"void main(void)" \
		"{" \


			"if(u_choice == 1){" \
				"FragColor = vec4(1.0f, 0.5f, 0.0f, 1.0f);" \
			"}" \

			"else if(u_choice == 2){" \
				"FragColor = texture(u_sampler, outTexCoord);" \
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
				exit(0);
			}
		}
	}

	/********** Shader Program Object **********/
	gShaderProgramObject_Models = glCreateProgram();

	glAttachShader(gShaderProgramObject_Models, iVertexShaderObject);
	glAttachShader(gShaderProgramObject_Models, iFragmentShaderObject);

	glBindAttribLocation(gShaderProgramObject_Models, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(gShaderProgramObject_Models, AMC_ATTRIBUTE_TEXCOORD0, "vTexCoord");

	glLinkProgram(gShaderProgramObject_Models);

	GLint iProgramLinkingStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	glGetProgramiv(gShaderProgramObject_Models, GL_LINK_STATUS, &iProgramLinkingStatus);
	if (iProgramLinkingStatus == GL_FALSE) {
		glGetProgramiv(gShaderProgramObject_Models, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject_Models, iInfoLogLength, &written, szInfoLog);
				fprintf(gbFile, "Shader Program Object Linking Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				exit(0);
			}
		}
	}

	



	//For Cube
	mvpUniform = glGetUniformLocation(gShaderProgramObject_Models, "u_mvp_matrix");
	samplerUniform = glGetUniformLocation(gShaderProgramObject_Models, "u_sampler");
	choiceUniform = glGetUniformLocation(gShaderProgramObject_Models, "u_choice");



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


		/********** Texture ***********/
		glGenBuffers(1, &vbo_Cube_Texture);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Cube_Texture);
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


	glEnable(GL_TEXTURE_2D);
	kundali_Cube = LoadTexture(MAKEINTRESOURCE(ID_BITMAP_KUNDALI));

}



GLuint LoadTexture(TCHAR imageResourceID[]) {

	HBITMAP hBitmap = NULL;
	BITMAP bmp;
	GLuint texture = 0;

	hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL), imageResourceID, IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);

	if (hBitmap) {

		GetObject(hBitmap, sizeof(BITMAP), &bmp);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

		glTexImage2D(GL_TEXTURE_2D,
			0,
			GL_RGB,
			bmp.bmWidth, bmp.bmHeight, 0,
			GL_BGR_EXT,
			GL_UNSIGNED_BYTE,
			bmp.bmBits);

		glGenerateMipmap(GL_TEXTURE_2D);

		DeleteObject(hBitmap);
		glBindTexture(GL_TEXTURE_2D, 0);
		
	}
	return(texture);
}

void initialize_GodRays(void){


	void uninitialize(void);
	void resize(int, int);
	
	//Shader Object;
	GLint iVertexShaderObject;
	GLint iFragmentShaderObject;


	/********** Vertex Shader **********/
	iVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *szVertexShaderSourceCode =
		"#version 450 core" \
		"\n" \

		//For Choice
		"uniform int u_choice;" \

		//For Attribute
		"in vec4 vPosition;" \
		"in vec2 vTexCoord;" \
		"out vec2 outTexCoord;" \

		// For Screen Space Light Position;
		"uniform mat4 u_mvp_perspective;" \
		"uniform vec4 u_v4LightPosition;" \
		"out vec4 out_v4LightPosition;" \
		

		//For Cube
		"uniform mat4 u_mvp_ortho;" \



		"void main(void)" \
		"{" \
			"gl_Position = u_mvp_ortho * vPosition;" \
			//"out_v4LightPosition = u_mvp_perspective * vec4(0.370f, 0.20f, 0.0f, 1.0f);" \
			
			//"out_v4LightPosition = u_mvp_perspective * vec4(u_v2LightPosition, 0.0f, 1.0f);" \
			//"out_v4LightPosition /= vec4(out_v4LightPosition.w);" \
			
			"outTexCoord = vTexCoord;" \

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
				exit(0);
			}
		}
	}

	/********** Fragment Shader **********/
	iFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *szFragmentShaderSourceCode =
		"#version 450 core" \
		"\n" \

		//For Choice
		"uniform int u_choice;" \

		//For Cube
		"in vec2 outTexCoord;" \
		"uniform sampler2D u_sampler_firstpass;" \

		//For Light Position
		"uniform mat4 u_mvp_perspective;" \
		"uniform vec4 u_v4LightPosition;" \
		"uniform vec4 u_v4ViewPort;" \
		"in vec4 out_v4LightPosition;" \

		//For God Rays
		"uniform float u_fExposure;" \
		"uniform float u_fDecay;" \
		"uniform float u_fDensity;" \
		"uniform float u_fWeight;" \
		
		"const int NUM_SAMPLES = 75;" \


		//For Output
		"out vec4 FragColor;" \

		"void main(void)" \
		"{" \



			"vec4 lightPos = u_mvp_perspective * u_v4LightPosition;" \
			"lightPos /= vec4(1.0f);" \

			// Here 0.5f tells to shift the orgin to specific location in XY plane!!!!
			
			//"lightPos += vec4(1.0f + 0.50f, 1.0f + 0.50f, 1.0f,1.0f);" \

			"lightPos += vec4(1.0f, 1.0f, 1.0f,1.0f);" \
			
			"lightPos *= vec4(0.5f);" \

			
			"vec2 v2DeltaTexCoord = outTexCoord.xy - lightPos.xy;" \

	
			"vec2 v2TexCoord = outTexCoord;" \
			"v2DeltaTexCoord *= 1.0f / float(NUM_SAMPLES) * u_fDensity;" \
			"float fIlluminationDecay = 1.0f;" \

			"for(int i = 0; i < NUM_SAMPLES; i++){" \
				"v2TexCoord = v2TexCoord - v2DeltaTexCoord;" \
				"vec4 v4Sample = texture2D(u_sampler_firstpass, v2TexCoord);" \

				"v4Sample = v4Sample * fIlluminationDecay * u_fWeight;" \

				"FragColor += v4Sample;" \

				"fIlluminationDecay *= u_fDecay;" \

			"}" \

			"FragColor *= u_fExposure;" \

			// "FragColor = vec4(out_v4LightPosition.xy, 0.0f, 1.0f);" \
		
			//"FragColor = texture2D(u_sampler_firstpass, outTexCoord);;" \

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
				exit(0);
			}
		}
	}

	/********** Shader Program Object **********/
	gShaderProgramObject_GodRays = glCreateProgram();

	glAttachShader(gShaderProgramObject_GodRays, iVertexShaderObject);
	glAttachShader(gShaderProgramObject_GodRays, iFragmentShaderObject);

	glBindAttribLocation(gShaderProgramObject_GodRays, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(gShaderProgramObject_GodRays, AMC_ATTRIBUTE_TEXCOORD0, "vTexCoord");

	glLinkProgram(gShaderProgramObject_GodRays);

	GLint iProgramLinkingStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	glGetProgramiv(gShaderProgramObject_GodRays, GL_LINK_STATUS, &iProgramLinkingStatus);
	if (iProgramLinkingStatus == GL_FALSE) {
		glGetProgramiv(gShaderProgramObject_GodRays, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject_GodRays, iInfoLogLength, &written, szInfoLog);
				fprintf(gbFile, "Shader Program Object Linking Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				exit(0);
			}
		}
	}


	mvpOrtho_Uniform_GodRays = glGetUniformLocation(gShaderProgramObject_GodRays, "u_mvp_ortho");
	mvpPerspective_Uniform_GodRays = glGetUniformLocation(gShaderProgramObject_GodRays, "u_mvp_perspective");
	samplerFirstPassUniform = glGetUniformLocation(gShaderProgramObject_GodRays, "u_sampler_firstpass");
	exposureUniform = glGetUniformLocation(gShaderProgramObject_GodRays, "u_fExposure");
	decayUniform = glGetUniformLocation(gShaderProgramObject_GodRays, "u_fDecay");
	densityUniform = glGetUniformLocation(gShaderProgramObject_GodRays, "u_fDensity");
	weightUniform = glGetUniformLocation(gShaderProgramObject_GodRays, "u_fWeight");
	lightPositionUniform = glGetUniformLocation(gShaderProgramObject_GodRays, "u_v4LightPosition");
	viewPortUniform_GodRays = glGetUniformLocation(gShaderProgramObject_GodRays, "u_v4ViewPort");




	/********** Position and TexCoord **********/
	GLfloat Rect_Vertices[] = {
		1.0f, 1.0f, 0.0f,
		-1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
	};

	GLfloat Rect_TexCoord[] = {
		1.0f, 1.0f,
		0.0f, 1.0f, 
		0.0f, 0.0f,
		1.0f, 0.0f,
	};



	/********** Vao Rect On Which We Apply Texture **********/
	glGenVertexArrays(1, &vao_Rect);
	glBindVertexArray(vao_Rect);

		/********** Position **********/
		glGenBuffers(1, &vbo_Rect_Position);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Position);
		glBufferData(GL_ARRAY_BUFFER, 
				sizeof(Rect_Vertices),
				Rect_Vertices,
				GL_DYNAMIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
					3,
					GL_FLOAT,
					GL_FALSE,
					0, NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/********** Texture **********/
		glGenBuffers(1, &vbo_Rect_TexCoord);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_TexCoord);
		glBufferData(GL_ARRAY_BUFFER,
				sizeof(Rect_TexCoord),
				Rect_TexCoord,
				GL_STATIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOORD0,
					2,
					GL_FLOAT,
					GL_FALSE,
					0, NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOORD0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);



	/********** FRAMEBUFFER **********/
	glGenFramebuffers(1, &frameBufferObject);
	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferObject);

		/********** Texture **********/
		glGenTextures(1, &texture_Cube);
		glBindTexture(GL_TEXTURE_2D, texture_Cube);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, 
			GL_RGBA, 
			viewPort_FBO_Width, viewPort_FBO_Height, 0, 
			GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_Cube, 0);


		/********** For Depth **********/
		glGenRenderbuffers(1, &renderBufferObject_Depth);
		glBindRenderbuffer(GL_RENDERBUFFER, renderBufferObject_Depth);	
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, viewPort_FBO_Width, viewPort_FBO_Height);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, renderBufferObject_Depth);




		/********** Checking *********/
		if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
			fprintf(gbFile, "ERROR: glCheckFramebufferStatus\n");
			uninitialize();
			exit(0);
		}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

}


void uninitialize_Models(void){

	if(kundali_Cube){
		glDeleteTextures(1, &kundali_Cube);
		kundali_Cube = 0;
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

	if (gShaderProgramObject_Models) {
		glUseProgram(gShaderProgramObject_Models);

		glGetProgramiv(gShaderProgramObject_Models, GL_ATTACHED_SHADERS, &ShaderCount);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount);
		if (pShader) {
			glGetAttachedShaders(gShaderProgramObject_Models, ShaderCount, &ShaderCount, pShader);
			for (ShaderNumber = 0; ShaderNumber < ShaderCount; ShaderNumber++) {
				glDetachShader(gShaderProgramObject_Models, pShader[ShaderNumber]);
				glDeleteShader(pShader[ShaderNumber]);
				pShader[ShaderNumber] = 0;
			}
			free(pShader);
			pShader = NULL;
		}
		glUseProgram(0);
		glDeleteProgram(gShaderProgramObject_Models);
		gShaderProgramObject_Models = 0;

	}

}


void uninitialize_GodRays(void){


	if(texture_Cube){
		glDeleteTextures(1, &texture_Cube);
		texture_Cube = 0;
	}


	if(renderBufferObject_Depth){
		glDeleteRenderbuffers(1, &renderBufferObject_Depth);
		renderBufferObject_Depth = 0;
	}

	if(frameBufferObject){
		glDeleteFramebuffers(1, &frameBufferObject);
		frameBufferObject = 0;
	}



	if(vbo_Rect_TexCoord){
		glDeleteBuffers(1, &vbo_Rect_TexCoord);
		vbo_Rect_TexCoord = 0;
	}

	if(vbo_Rect_Position){
		glDeleteBuffers(1, &vbo_Rect_Position);
		vbo_Rect_Position = 0;
	}

	if(vao_Rect){
		glDeleteVertexArrays(1, &vao_Rect);
		vao_Rect = 0;
	}

	GLsizei ShaderCount;
	GLsizei ShaderNumber;

	if (gShaderProgramObject_GodRays) {
		glUseProgram(gShaderProgramObject_GodRays);

		glGetProgramiv(gShaderProgramObject_GodRays, GL_ATTACHED_SHADERS, &ShaderCount);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount);
		if (pShader) {
			glGetAttachedShaders(gShaderProgramObject_GodRays, ShaderCount, &ShaderCount, pShader);
			for (ShaderNumber = 0; ShaderNumber < ShaderCount; ShaderNumber++) {
				glDetachShader(gShaderProgramObject_GodRays, pShader[ShaderNumber]);
				glDeleteShader(pShader[ShaderNumber]);
				pShader[ShaderNumber] = 0;
			}
			free(pShader);
			pShader = NULL;
		}
		glUseProgram(0);
		glDeleteProgram(gShaderProgramObject_GodRays);
		gShaderProgramObject_GodRays = 0;

	}
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


	uninitialize_Models();

	uninitialize_GodRays();

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
}



void display(void) {

	void update(void);

	mat4 translateMatrix;
	mat4 rotateMatrix;
	mat4 modelMatrix;
	mat4 viewMatrix;
	mat4 modelViewProjectionMatrix;
	mat4 lightPosition;

	static GLfloat angle_Model = 0.0f;


	//For Model As a Texture

	glViewport(0, 0, (GLsizei)viewPort_FBO_Width, (GLsizei)viewPort_FBO_Height);
	gPerspectiveProjectionMatrix = mat4::identity();
	gPerspectiveProjectionMatrix = perspective(45.0f, (float)viewPort_FBO_Width / (GLfloat)viewPort_FBO_Height, 0.1f, 100.0f);

	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferObject);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		glUseProgram(gShaderProgramObject_Models);


		// Sphere
		translateMatrix = mat4::identity();
		modelMatrix = mat4::identity();
		viewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();

		translateMatrix = translate(sphereX, sphereY, sphereZ);
		

		modelMatrix = modelMatrix * translateMatrix;
		modelViewProjectionMatrix = gPerspectiveProjectionMatrix * viewMatrix * modelMatrix;

		glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);
		glUniform1i(choiceUniform, 1);

		glBindVertexArray(vao_Sphere);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element);
			glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindVertexArray(0);


		// Cube
		translateMatrix = mat4::identity();
		modelMatrix = mat4::identity();
		viewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();

		translateMatrix = translate(cubeX, cubeY, cubeZ);
		rotateMatrix = rotate(angle_Cube, 1.0f, 0.0f, 0.0f) * rotate(angle_Cube, 0.0f, 1.0f, 0.0f) * rotate(angle_Cube, 0.0f, 0.0f, 1.0f);
		modelMatrix = modelMatrix * translateMatrix * scale(0.1f, 0.1f, 0.1f) * rotateMatrix;
		modelViewProjectionMatrix = gPerspectiveProjectionMatrix * viewMatrix * modelMatrix;

		glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);
		glUniform1i(choiceUniform, 2);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, 0);
		glUniform1i(samplerUniform, 0);

		glBindVertexArray(vao_Cube);
			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
			glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
			glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
			glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
			glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
			glDrawArrays(GL_TRIANGLE_FAN, 20, 4);
		glBindVertexArray(0);
		

		glUseProgram(0);
		

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	

	//Draw Scene
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(0, 0, (GLsizei)viewPortWidth, (GLsizei)viewPortHeight);
	gPerspectiveProjectionMatrix = mat4::identity();
	gPerspectiveProjectionMatrix = perspective(45.0f, (float)viewPortWidth / (float)viewPortHeight, 0.1f, 100.0f);
	gOrthoProjectionMatrix = ortho(0, viewPortWidth, 0, viewPortHeight, -100.0f, 100.0f);
	glUseProgram(gShaderProgramObject_Models);


		// Sphere
		translateMatrix = mat4::identity();
		modelMatrix = mat4::identity();
		viewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();

		translateMatrix = translate(sphereX, sphereY, sphereZ);

		modelMatrix = modelMatrix * translateMatrix;
		modelViewProjectionMatrix = gPerspectiveProjectionMatrix * viewMatrix * modelMatrix;

		glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);
		glUniform1i(choiceUniform, 1);

		glBindVertexArray(vao_Sphere);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element);
			glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindVertexArray(0);


		//Cube
		translateMatrix = mat4::identity();
		modelMatrix = mat4::identity();
		viewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();

		translateMatrix = translate(cubeX, cubeY, cubeZ);
		rotateMatrix = rotate(angle_Cube, 1.0f, 0.0f, 0.0f) * rotate(angle_Cube, 0.0f, 1.0f, 0.0f) * rotate(angle_Cube, 0.0f, 0.0f, 1.0f);
		modelMatrix = modelMatrix * translateMatrix * scale(0.1f, 0.1f, 0.1f) * rotateMatrix;
		modelViewProjectionMatrix = gPerspectiveProjectionMatrix * viewMatrix * modelMatrix;

		glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);
		glUniform1i(choiceUniform, 2);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, kundali_Cube);
		glUniform1i(samplerUniform, 0);

		glBindVertexArray(vao_Cube);
			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
			glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
			glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
			glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
			glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
			glDrawArrays(GL_TRIANGLE_FAN, 20, 4);
		glBindVertexArray(0);
		

	glUseProgram(0);




	//For God Rays
	glViewport(0, 0, (GLsizei)viewPortWidth, (GLsizei)viewPortHeight);
	gOrthoProjectionMatrix = mat4::identity();
	gOrthoProjectionMatrix = ortho(
					-viewPortWidth / 2.0f, viewPortWidth / 2.0f,	// L, R
					-viewPortHeight/ 2.0f, viewPortHeight / 2.0f,	// B, T
					-1.0f, 1.0f);						// N, F

	gPerspectiveProjectionMatrix = mat4::identity();
	gPerspectiveProjectionMatrix = perspective(45.0f, (float)viewPortWidth / (float)viewPortHeight, 0.1f, 100.0f);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_DEPTH_BUFFER_BIT);

	glUseProgram(gShaderProgramObject_GodRays);


		// For Rectangle
		modelMatrix = mat4::identity();
		viewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();
		modelMatrix = modelMatrix;
		modelViewProjectionMatrix = gOrthoProjectionMatrix * viewMatrix * modelMatrix;

		glUniformMatrix4fv(mvpOrtho_Uniform_GodRays, 1, GL_FALSE, modelViewProjectionMatrix);

		//For Light Source
		modelMatrix = mat4::identity();
		viewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();
		modelViewProjectionMatrix = gPerspectiveProjectionMatrix * viewMatrix * modelMatrix;

		glUniformMatrix4fv(mvpPerspective_Uniform_GodRays, 1, GL_FALSE, modelViewProjectionMatrix);


		
		glEnable(GL_TEXTURE_2D);                    // Enable 2D Texture Mapping
		glDisable(GL_DEPTH_TEST);                   // Disable Depth Testing
		glBlendFunc(GL_SRC_ALPHA,GL_ONE);           // Set Blending Mode
		glEnable(GL_BLEND);  

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture_Cube);
		glUniform1i(samplerFirstPassUniform, 0);

		glUniform1f(exposureUniform, gfExposure);
		glUniform1f(decayUniform, gfDecay);
		glUniform1f(densityUniform, gfDensity);
		glUniform1f(weightUniform, gfWeight);
		glUniform2fv(lightPositionUniform, 1, gfvLightPosition);
		glUniform4fv(viewPortUniform_GodRays, 1, vec4(0.0f, 0.0f, viewPortWidth, viewPortHeight));


		GLfloat Rect_Vertices[] = {
			viewPortWidth / 2.0f, viewPortHeight / 2.0f, 0.0f,
			-viewPortWidth / 2.0f, viewPortHeight / 2.0f, 0.0f,
			-viewPortWidth / 2.0f, -viewPortHeight / 2.0f, 0.0f,
			viewPortWidth / 2.0f, -viewPortHeight / 2.0f, 0.0f,
		};

		glBindVertexArray(vao_Rect);

			glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Position);
			glBufferData(GL_ARRAY_BUFFER, sizeof(Rect_Vertices), Rect_Vertices, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		glBindVertexArray(0);


		glEnable(GL_DEPTH_TEST);                    // Enable Depth Testing
		glDisable(GL_TEXTURE_2D);                   // Disable 2D Texture Mapping
		glDisable(GL_BLEND);                        // Disable Blending
		glBindTexture(GL_TEXTURE_2D,0); 

	
	glUseProgram(0);

	update();

	SwapBuffers(ghdc);
}


void update(void) {

	angle_Cube -= 1.5f;
	if(angle_Cube < 0.0f)
		angle_Cube = 360.0f;

	cubeX += 0.01f;
	
}

