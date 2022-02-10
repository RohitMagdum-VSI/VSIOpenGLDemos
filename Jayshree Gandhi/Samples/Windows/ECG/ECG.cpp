#include<windows.h>
#include<GL/glew.h>  
#include<gl/GL.h>
#include<stdio.h>
#include"vmath.h"

#include"data_ecg.h"

#pragma comment(lib,"glew32.lib")
#pragma comment(lib,"opengl32.lib")

using namespace vmath;

enum
{
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOORD0
};

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

#define ECG_DATA_BUFFER_SIZE 1024

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

bool gbFullScreen = false;
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
HWND ghWnd = NULL;
HDC ghdc = NULL;
HGLRC ghrc = NULL;
bool gbActiveWindow = false;
FILE *gpFile = NULL;

void ToggleFullScreen(void);
int initialize(void);
void resize(int, int);
void display(void);
void update(void);
void uninitialize(void);

GLuint gShaderProgramObject;
GLuint vao;
GLuint vao_line;

GLuint vbo_position;

GLuint mvpUniform;

mat4 perspectiveProjectionMatrix;

struct MyPoint
{
	GLfloat x;
	GLfloat y;
};

MyPoint point[1024];

int gWidth;
int gHeight;
float ratio;
int counter = 0;
bool isSpaceKeyPressed = false;

int size = 1024;
float myScale = 0.5f;
float offset_y = 0.0f;
float space;
float gpos;

//void egc(int);
//void createECGData(int, int, float, float);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szAppName[] = TEXT("MY OGL WINDOW");
	bool bDone = false;
	int iRet = 0;

	if (fopen_s(&gpFile, "Log.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Log file can not be created"), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
	{
		fprintf(gpFile, "Log file created successfully...\n");
	}

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.lpszClassName = szAppName;
	wndclass.lpszMenuName = NULL;
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szAppName,
		TEXT("My Double buffer Window - Jayshree"),
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		100,
		100,
		WIN_WIDTH,
		WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghWnd = hwnd;

	iRet = initialize();
	if (iRet == -1)
	{
		fprintf(gpFile, "ChoosePixelFormat() Failed\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -2)
	{
		fprintf(gpFile, "SetPixelFormat() Failed\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -3)
	{
		fprintf(gpFile, "wglCreateContext() Failed\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -4)
	{
		fprintf(gpFile, "wglMakeCurrent() Failed\n");
		DestroyWindow(hwnd);
	}
	else
	{
		fprintf(gpFile, "Initialization succeded\n");
	}

	ToggleFullScreen();
	ShowWindow(hwnd, iCmdShow);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
			{
				bDone = true;
			}
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			//Play game here
			if (gbActiveWindow == true)
			{
				//code
				//here call update
				update();
			}
			display();
		}
	}

	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{

	switch (iMsg)
	{
	case WM_SETFOCUS:
		gbActiveWindow = true;
		break;

	case WM_KILLFOCUS:
		gbActiveWindow = false;
		break;

	case WM_SIZE:
		gWidth = LOWORD(lParam);
		gHeight = HIWORD(lParam);

		ratio = gWidth / gHeight;
		resize(gWidth, gHeight);
		break;

	case WM_ERASEBKGND:
		return(0);
		break;

	case WM_CLOSE:
		DestroyWindow(hwnd);
		break;

	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;

		case VK_SPACE:
			if (isSpaceKeyPressed == false)
			{
				isSpaceKeyPressed = true;
			}
			else
			{
				isSpaceKeyPressed = false;
			}
			break;
		}
		break;

	case WM_DESTROY:
		uninitialize();
		PostQuitMessage(0);
		break;
	}

	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullScreen(void)
{
	MONITORINFO mi;

	if (gbFullScreen == false)
	{
		dwStyle = GetWindowLong(ghWnd, GWL_STYLE);

		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			mi = { sizeof(MONITORINFO) };

			if (GetWindowPlacement(ghWnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghWnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghWnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);

				SetWindowPos(ghWnd,
					HWND_TOP,
					mi.rcMonitor.left,
					mi.rcMonitor.top,
					mi.rcMonitor.right - mi.rcMonitor.left,
					mi.rcMonitor.bottom - mi.rcMonitor.top,
					SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}

		ShowCursor(FALSE);
		gbFullScreen = true;
	}
}

int initialize(void)
{

	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;
	GLenum result;

	ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32;

	ghdc = GetDC(ghWnd);

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0)
	{
		return(-1);
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		return(-2);
	}

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		return(-3);
	}

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		return(-4);
	}

	//On the extensions requred for PP
	result = glewInit();
	if (result != GLEW_OK)
	{
		fprintf(gpFile, "\nglewInit() failed...\n");
		uninitialize();
		DestroyWindow(ghWnd);
	}

	GLuint vertexShaderObject;
	GLuint fragmentShaderObject;


	//***************** 1. VERTEX SHADER ************************************ 
	//define vertex shader object
	//create vertex shader object
	vertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	//Write vertex shader code
	const GLchar *vertexShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec4 vColor;" \
		"uniform mat4 u_mvp_matrix;" \
		"out vec4 out_color;" \
		"void main(void)" \
		"{" \
		"gl_Position = u_mvp_matrix * vPosition;" \
		"out_color = vColor;" \
		"}";

	//specify above source code to vertex shader object
	glShaderSource(vertexShaderObject,//to whom?
		1,//how many strings
		(const GLchar **)&vertexShaderSourceCode,//address of string
		NULL);// NULL specifes that there is only one string with fixed length

	//Compile the vertex shader
	glCompileShader(vertexShaderObject);


	//Error checking for compilation:
	GLint iShaderCompileStatus = 0;
	GLint iInfoLogLength = 0;
	GLchar *szInfoLog = NULL;

	//Step 1 : Call glGetShaderiv() to get comiple status of particular shader
	glGetShaderiv(vertexShaderObject, // whose?
		GL_COMPILE_STATUS,//what to get?
		&iShaderCompileStatus);//in what?

	//Step 2 : Check shader compile status for GL_FALSE
	if (iShaderCompileStatus == GL_FALSE)
	{
		//Step 3 : If GL_FALSE , call glGetShaderiv() again , but this time to get info log length
		glGetShaderiv(vertexShaderObject,
			GL_INFO_LOG_LENGTH,
			&iInfoLogLength);

		//Step 4 : if info log length > 0 , call glGetShaderInfoLog()
		if (iInfoLogLength > 0)
		{
			//allocate memory to pointer
			szInfoLog = (GLchar *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;

				glGetShaderInfoLog(vertexShaderObject,//whose?
					iInfoLogLength,//length?
					&written,//might have not used all, give that much only which have been used in what?
					szInfoLog);//store in what?

				fprintf(gpFile, "\nVertex Shader Compilation Log : %s\n", szInfoLog);

				//free the memory
				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}


	//************************** 2. FRAGMENT SHADER ********************************
	//define fragment shader object
	//create fragment shader object
	fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	//write fragment shader code
	const GLchar *fragmentShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"in vec4 out_color;" \
		"out vec4 fragColor;" \
		"void main(void)" \
		"{" \
		"fragColor = out_color;" \
		"}";

	//specify the above source code to fragment shader object
	glShaderSource(fragmentShaderObject,
		1,
		(const GLchar **)&fragmentShaderSourceCode,
		NULL);

	//compile the fragment shader
	glCompileShader(fragmentShaderObject);


	//Error checking for compilation
	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	//Step 1 : Call glGetShaderiv() to get comiple status of particular shader
	glGetShaderiv(fragmentShaderObject, // whose?
		GL_COMPILE_STATUS,//what to get?
		&iShaderCompileStatus);//in what?

	//Step 2 : Check shader compile status for GL_FALSE
	if (iShaderCompileStatus == GL_FALSE)
	{
		//Step 3 : If GL_FALSE , call glGetShaderiv() again , but this time to get info log length
		glGetShaderiv(fragmentShaderObject,
			GL_INFO_LOG_LENGTH,
			&iInfoLogLength);

		//Step 4 : if info log length > 0 , call glGetShaderInfoLog()
		if (iInfoLogLength > 0)
		{
			//allocate memory to pointer
			szInfoLog = (GLchar *)malloc(iInfoLogLength);

			if (szInfoLog != NULL)
			{
				GLsizei written;

				glGetShaderInfoLog(fragmentShaderObject,//whose?
					iInfoLogLength,//length?
					&written,//might have not used all, give that much only which have been used in what?
					szInfoLog);//store in what?

				fprintf(gpFile, "\nFragment Shader Compilation Log : %s\n", szInfoLog);

				//free the memory
				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}


	//create shader program object
	gShaderProgramObject = glCreateProgram();

	//Attach vertex shader to shader program
	glAttachShader(gShaderProgramObject,//to whom?
		vertexShaderObject);//what to attach?

	//Attach fragment shader to shader program
	glAttachShader(gShaderProgramObject,
		fragmentShaderObject);

	//Pre-Linking binding to vertex attribute
	glBindAttribLocation(gShaderProgramObject,
		AMC_ATTRIBUTE_POSITION,
		"vPosition");

	glBindAttribLocation(gShaderProgramObject,
		AMC_ATTRIBUTE_COLOR,
		"vColor");

	//Link the shader program
	glLinkProgram(gShaderProgramObject);//link to whom?


	//Error checking for linking
	GLint iProgramLinkStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	//Step 1 : Call glGetShaderiv() to get comiple status of particular shader
	glGetProgramiv(gShaderProgramObject, // whose?
		GL_LINK_STATUS,//what to get?
		&iProgramLinkStatus);//in what?

	//Step 2 : Check shader compile status for GL_FALSE
	if (iProgramLinkStatus == GL_FALSE)
	{
		//Step 3 : If GL_FALSE , call glGetShaderiv() again , but this time to get info log length
		glGetProgramiv(gShaderProgramObject,
			GL_INFO_LOG_LENGTH,
			&iInfoLogLength);

		//Step 4 : if info log length > 0 , call glGetShaderInfoLog()
		if (iInfoLogLength > 0)
		{
			//allocate memory to pointer
			szInfoLog = (GLchar *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;

				glGetProgramInfoLog(gShaderProgramObject,//whose?
					iInfoLogLength,//length?
					&written,//might have not used all, give that much only which have been used in what?
					szInfoLog);//store in what?

				fprintf(gpFile, "\nShader Program Linking Log : %s\n", szInfoLog);

				//free the memory
				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}


	//Post-Linking reteriving uniform location
	mvpUniform = glGetUniformLocation(gShaderProgramObject,
		"u_mvp_matrix");

	//above is the preparation of data transfer from CPU to GPU 
	//i.e glBindAttribLocation() & glGetUniformLocation()

	//array initialization (glBegin() and glEnd())
	space = 4.4f / size * (float)(ratio);
	float pos = -size * space / 2.0f;
	gpos = pos;
	/*
	for (int i = 0; i < 1000; i++)
	{
		//float data = myScale * data_ecg[i] + offset_y;
		float data = myScale * data_ecg[i];

		point[i].x = pos;
		point[i].y = data;

		pos = pos + space;
	}*/

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glGenBuffers(1, &vbo_position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_position);

	glBufferData(GL_ARRAY_BUFFER,
		2 * 2 * sizeof(float),
		NULL,
		GL_DYNAMIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		2,
		GL_FLOAT,
		GL_FALSE,
		0,
		NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glVertexAttrib3f(AMC_ATTRIBUTE_COLOR,
		0.0f,
		1.0f,
		0.0f);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	perspectiveProjectionMatrix = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);

	return(0);
}


void resize(int width, int height)
{
	if (height == 0)
	{
		height = 1;
	}

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	perspectiveProjectionMatrix = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void display(void)
{

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//Binding Opengl code to shader program object
	glUseProgram(gShaderProgramObject);

	//matrices
	mat4 modelViewMatrix;
	mat4 modelViewProjectionMatrix;
	mat4 translationMatrix;

	//make identity
	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();
	translationMatrix = mat4::identity();

	//do necessary transformation

	translationMatrix = translate(0.0f, 0.0f, -3.0f);

	//do necessary matrix multiplication
	//this was internally done by gluOrtho() in ffp
	modelViewMatrix = translationMatrix;
	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

	//send necessary matrices to shader in respective uniforms
	glUniformMatrix4fv(mvpUniform,//which uniform?
		1,//how many matrices
		GL_FALSE,//have to transpose?
		modelViewProjectionMatrix);//actual matrix

	//bind with vao


	if (isSpaceKeyPressed == false)
	{
		glBindVertexArray(vao);

		float data = myScale * data_ecg[counter];

		point[counter].x = gpos;
		point[counter].y = data;

		gpos = gpos + space;

		glLineWidth(3.0f);

		glBindBuffer(GL_ARRAY_BUFFER, vbo_position);

		glBufferData(GL_ARRAY_BUFFER,
			sizeof(point),
			point,
			GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glDrawArrays(GL_LINE_STRIP,
			1,
			counter);

		glBindVertexArray(0);
	}
	else
	{

		glBindVertexArray(vao);

		//float data = 0.0f;

		point[counter].x = gpos;
		point[counter].y = 0.0f;

		gpos = gpos + space;

		glLineWidth(3.0f);

		glBindBuffer(GL_ARRAY_BUFFER, vbo_position);

		glBufferData(GL_ARRAY_BUFFER,
			sizeof(point),
			point,
			GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glDrawArrays(GL_LINE_STRIP,
			1,
			counter);

		glBindVertexArray(0);
	}

	//unbinding program
	glUseProgram(0);


	SwapBuffers(ghdc);

}

void update(void)
{
	counter = counter + 1;
	if (counter > 1000)
	{
		counter = 0;
		gpos = -size * space / 2.0f;
	}
}


void uninitialize(void)
{

	if (vbo_position)
	{
		glDeleteBuffers(1, &vbo_position);
		vbo_position = 0;
	}

	if (vao)
	{
		glDeleteVertexArrays(1, &vao);
		vao = 0;
	}

	if (gShaderProgramObject)
	{
		GLsizei shaderCount;
		GLsizei shaderNumber;

		glUseProgram(gShaderProgramObject);

		//ask the program how many shaders are attached to you?
		glGetProgramiv(gShaderProgramObject,
			GL_ATTACHED_SHADERS,
			&shaderCount);

		GLuint *pShaders = (GLuint *)malloc(sizeof(GLuint) * shaderCount);

		if (pShaders)
		{
			//fprintf(gpFile, "\npshaders sucessful\n");

			//get shaders
			glGetAttachedShaders(gShaderProgramObject,
				shaderCount,
				&shaderCount,
				pShaders);

			for (shaderNumber = 0; shaderNumber < shaderCount; shaderNumber++)
			{
				//detach
				glDetachShader(gShaderProgramObject,
					pShaders[shaderNumber]);

				//delete
				glDeleteShader(pShaders[shaderNumber]);

				//explicit 0
				pShaders[shaderNumber] = 0;
			}

			free(pShaders);
		}

		glDeleteProgram(gShaderProgramObject);
		gShaderProgramObject = 0;

		glUseProgram(0);
	}

	if (gbFullScreen == true)
	{
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);

		SetWindowPlacement(ghWnd, &wpPrev);

		SetWindowPos(ghWnd,
			HWND_TOP,
			0,
			0,
			0,
			0,
			SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER);

		ShowCursor(TRUE);
	}

	if (wglGetCurrentContext() == ghrc)
	{
		wglMakeCurrent(NULL, NULL);
	}

	if (ghrc)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}

	if (ghdc)
	{
		ReleaseDC(ghWnd, ghdc);
		ghdc = NULL;
	}


	if (gpFile)
	{
		fprintf(gpFile, "Log file closed successfully\n");
		fclose(gpFile);
		gpFile = NULL;
	}
}

/*void createECGData(int myOffset, int mySize, float myOffset_y, float myScale)
{
	const float mySpace = 2.0f / mySize * ratio;

	float myPos = - mySize * mySpace / 2.0f;

	for (int i = myOffset; i < mySize + myOffset; i++)
	{
		const float myData = myScale * data_ecg[i] + myOffset_y;

		point[i].x = myPos;
		point[i].y = myData;

		myPos = myPos + mySpace;
	}
}*/
