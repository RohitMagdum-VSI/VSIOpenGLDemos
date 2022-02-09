#include<windows.h>
#include<C:\glew\include\GL\glew.h>
#include<gl/GL.h>
#include<stdio.h>
#include<math.h>
#include"vmath.h"

#pragma comment(lib,"User32.lib")
#pragma comment(lib,"GDI32.lib")
#pragma comment(lib,"C:\\glew\\lib\\Release\\Win32\\glew32.lib")
#pragma comment(lib,"opengl32.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

#define VERTEX_ARRAY 1
#define COLOR_ARRAY 2
#define VELOCITY_ARRAY 3
#define START_TIME_ARRAY 4

using namespace vmath;

enum
{
	HAD_ATTRIBUTE_VERTEX = 0,
	HAD_ATTRIBUTE_COLOR,
	HAD_ATTRIBUTE_NORMAL,
	HAD_ATTRIBUTE_TEXTURE0,
	HAD_ATTRIBUTE_VELOCITIES,
	HAD_ATTRIBUTE_START_TIME,
};

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

FILE *gpFile;
HWND ghwnd;
HDC ghdc;
HGLRC ghrc;
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
bool gbActiveWindow = false;
bool gbFullscreen = false;
bool gbIsEscapeKeyPressed = false;

GLuint gVertexShaderObject;
GLuint gFragmentShaderObject;
GLuint gGeometryShaderObject;
GLuint gShaderProgramObject;

GLuint vao;
GLuint vbo_pos, vbo_color, vbo_velocities, vbo_start_time;

/*************Variables for Particle Engine************/
GLint arrayWidth, arrayHeight;
GLfloat *verts = NULL;
GLfloat *colors = NULL;
GLfloat *velocities = NULL;
GLfloat *startTimes = NULL;
float Particle_Time;

GLfloat gAngle;

GLuint gbParticleTimeUniform;
GLuint gbPerpectiveProjectionMatrixUniform;

mat4 gPerpectiveProjectionMatrix;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	void initialize(void);
	void display(void);
	void update(void);
	void uninitialize(int);
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("My App");
	bool bDone = false;

	if (fopen_s(&gpFile, "Log.txt", "w") != NULL)
	{
		MessageBox(NULL, TEXT("Cannot Create Log File !!!"), TEXT("Error"), MB_OK);
		exit(EXIT_FAILURE);
	}
	else
		fprintf(gpFile, "Log File Created Successfully...\n");

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.hInstance = hInstance;
	wndclass.lpszClassName = szClassName;
	wndclass.lpszMenuName = NULL;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);

	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, TEXT("OpenGL Programmable Pipeline First Shader Program"), WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE, 100, 100, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);
	if (hwnd == NULL)
	{
		fprintf(gpFile, "Cannot Create Window...\n");
		uninitialize(1);
	}

	ghwnd = hwnd;

	ShowWindow(hwnd, iCmdShow);
	SetFocus(hwnd);
	SetForegroundWindow(hwnd);

	initialize();

	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
				bDone = true;
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			if (gbActiveWindow == true)
			{
				if (gbIsEscapeKeyPressed == true)
					bDone = true;
				update();
				display();
			}
		}
	}

	uninitialize(0);
	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	void resize(int, int);
	void ToggleFullscreen(void);
	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			gbActiveWindow = true;
		else
			gbActiveWindow = false;
		break;
	case WM_CREATE:
		break;
	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;
	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			gbIsEscapeKeyPressed = true;
			break;
		case 0x46:
			if (gbFullscreen == false)
			{
				ToggleFullscreen();
				gbFullscreen = true;
			}
			else
			{
				ToggleFullscreen();
				gbFullscreen = false;
			}
			break;
		}
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void initialize(void)
{
	void resize(int, int);
	void createPoints(GLint, GLint);
	void uninitialize(int);
	
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

	ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 24;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32;

	ghdc = GetDC(ghwnd);
	if (ghdc == NULL)
	{
		fprintf(gpFile, "GetDC() Failed.\n");
		uninitialize(1);
	}

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0)
	{
		fprintf(gpFile, "ChoosePixelFormat() Failed.\n");
		uninitialize(1);
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		fprintf(gpFile, "SetPixelFormat() Failed.\n");
		uninitialize(1);
	}

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		fprintf(gpFile, "wglCreateContext() Failed.\n");
		uninitialize(1);
	}

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		fprintf(gpFile, "wglMakeCurrent() Failed");
		uninitialize(1);
	}

	GLenum glew_error = glewInit();
	if (glew_error != GLEW_OK)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	createPoints(100, 100);

	//Vertex Shader
	gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *vertexShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"uniform float Time;" \
		"in vec4 MCVertex;" \
		"in vec3 MColor;" \
		"in vec3 Velocity;" \
		"in float StartTime;" \
		"out vec3 Color;" \
		"void main(void)"\
		"{"\
		"vec4 vert;" \
		"vec3 Background = MColor;" \
		"float t = Time - StartTime;" \
		"if(t>=0.0)" \
		"{" \
		"vert = MCVertex + vec4(Velocity * t , 0.0);" \
		"vert.y -= 4.9*t*t;" \
		"Color = MColor;" \
		"}" \
		"else" \
		"{" \
		"vert=MCVertex;" \
		"Color = Background;" \
		"}" \
		"gl_Position = vert;" \
		"}";

	glShaderSource(gVertexShaderObject, 1, (const GLchar **)&vertexShaderSourceCode, NULL);

	glCompileShader(gVertexShaderObject);
	GLint iInfoLogLength = 0;
	GLint iShaderCompiledStatus = 0;
	char *szInfoLog = NULL;

	glGetShaderiv(gVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(gVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gVertexShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Vertex Shader Compilation Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize(1);
				exit(0);
			}
		}
	}

	//Geometry Shader
	gGeometryShaderObject = glCreateShader(GL_GEOMETRY_SHADER);

	const GLchar *geometryShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"layout(points)in;" \
		"layout(triangle_strip , max_vertices = 4)out;" \
		"in vec3 Color[];" \
		"uniform mat4 MVPMatrix;" \
		"out vec3 final_Color;" \
		"const float size = 0.1;"
		"const vec3 lightDirection = normalize(vec3(0.4,-1.0,0.8));" \
		"const float ambientLight = 0.3;" \

		"void createVertex(vec3 offset , vec3 color)" \
		"{" \
		"vec4 actualOffset = vec4(offset * size, 0.0);" \
		"vec4 worldPosition = gl_in[0].gl_Position + actualOffset;" \
		"gl_Position = MVPMatrix * worldPosition;" \
		"final_Color = color;" \
		"EmitVertex();" \
		"}" \

		"vec3 calculateLight(vec3 faceNormal)" \
		"{" \
		"	float brightness = max(dot(-lightDirection, faceNormal),ambientLight);" \
		"	return (Color[0] * brightness);" \
		"}" \

		"void main(void)" \
		"{" \
		"	vec3 faceNormal = vec3(0.0,0.0,1.0);" \
		"	vec3 color = calculateLight(faceNormal);" \
		"	createVertex(vec3(-1.0,1.0,1.0),color);" \
		"	createVertex(vec3(-1.0,-1.0,1.0),color);" \
		"	createVertex(vec3(1.0,1.0,1.0),color);" \
		"	createVertex(vec3(1.0,-1.0,1.0),color);" \
		"	EndPrimitive();" \

		"	faceNormal = vec3(1.0,0.0,0.0);" \
		"	color = calculateLight(faceNormal);" \
		"	createVertex(vec3(1.0,1.0,1.0),color);" \
		"	createVertex(vec3(1.0,-1.0,1.0),color);" \
		"	createVertex(vec3(1.0,1.0,-1.0),color);" \
		"	createVertex(vec3(1.0,-1.0,-1.0),color);" \
		"	EndPrimitive();" \

		"	faceNormal = vec3(0.0,0.0,-1.0);" \
		"	color = calculateLight(faceNormal);" \
		"	createVertex(vec3(1.0,1.0,-1.0),color);" \
		"	createVertex(vec3(1.0,-1.0,-1.0),color);" \
		"	createVertex(vec3(-1.0,1.0,-1.0),color);" \
		"	createVertex(vec3(-1.0,-1.0,-1.0),color);" \
		"	EndPrimitive();" \

		"	faceNormal = vec3(-1.0,0.0,0.0);" \
		"	color = calculateLight(faceNormal);" \
		"	createVertex(vec3(-1.0,1.0,-1.0),color);" \
		"	createVertex(vec3(-1.0,-1.0,-1.0),color);" \
		"	createVertex(vec3(-1.0,1.0,1.0),color);" \
		"	createVertex(vec3(-1.0,-1.0,1.0),color);" \
		"	EndPrimitive();" \

		"	faceNormal = vec3(0.0,1.0,0.0);" \
		"	color = calculateLight(faceNormal);" \
		"	createVertex(vec3(-1.0,1.0,-1.0),color);" \
		"	createVertex(vec3(-1.0,1.0,1.0),color);" \
		"	createVertex(vec3(1.0,1.0,-1.0),color);" \
		"	createVertex(vec3(1.0,1.0,1.0),color);" \
		"	EndPrimitive();" \

		"	faceNormal = vec3(0.0,-1.0,0.0);" \
		"	color = calculateLight(faceNormal);" \
		"	createVertex(vec3(-1.0,-1.0,1.0),color);" \
		"	createVertex(vec3(-1.0,-1.0,-1.0),color);" \
		"	createVertex(vec3(1.0,-1.0,1.0),color);" \
		"	createVertex(vec3(1.0,-1.0,-1.0),color);" \
		"	EndPrimitive();" \
		"}";

	glShaderSource(gGeometryShaderObject, 1, (const GLchar **)&geometryShaderSourceCode, NULL);

	glCompileShader(gGeometryShaderObject);

	glGetShaderiv(gGeometryShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(gGeometryShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gGeometryShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Geometry Shader Compilation Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize(1);
				exit(0);
			}
		}
	}

	//Fragment Shader
	gFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *fragmentShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"in vec3 final_Color;" \
		"out vec4 FragColor;" \
		"void main(void)"\
		"{"\
		"FragColor = vec4(final_Color,1.0f);"\
		"}";

	glShaderSource(gFragmentShaderObject, 1, (const GLchar **)&fragmentShaderSourceCode, NULL);

	glCompileShader(gFragmentShaderObject);

	glGetShaderiv(gFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(gFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gFragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Fragment Shader Compilation Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize(1);
				exit(0);
			}
		}
	}

	//Shader Program
	gShaderProgramObject = glCreateProgram();

	glAttachShader(gShaderProgramObject, gVertexShaderObject);

	glAttachShader(gShaderProgramObject, gGeometryShaderObject);

	glAttachShader(gShaderProgramObject, gFragmentShaderObject);

	/*glBindAttribLocation(gShaderProgramObject, HAD_ATTRIBUTE_VERTEX, "MCVertex");

	glBindAttribLocation(gShaderProgramObject, HAD_ATTRIBUTE_COLOR, "MColor");

	glBindAttribLocation(gShaderProgramObject, HAD_ATTRIBUTE_VELOCITIES, "Velocity");

	glBindAttribLocation(gShaderProgramObject, HAD_ATTRIBUTE_START_TIME, "StartTime");*/

	glBindAttribLocation(gShaderProgramObject, VERTEX_ARRAY, "MCVertex");

	glBindAttribLocation(gShaderProgramObject, COLOR_ARRAY, "MColor");

	glBindAttribLocation(gShaderProgramObject, VELOCITY_ARRAY, "Velocity");

	glBindAttribLocation(gShaderProgramObject, START_TIME_ARRAY, "StartTime");

	glLinkProgram(gShaderProgramObject);

	GLint iShaderProgramLinkStatus = 0;

	glGetProgramiv(gShaderProgramObject, GL_LINK_STATUS, &iShaderProgramLinkStatus);
	if (iShaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Shader Program Link Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize(1);
				exit(0);
			}
		}
	}

	gbParticleTimeUniform = glGetUniformLocation(gShaderProgramObject, "Time");
	gbPerpectiveProjectionMatrixUniform = glGetUniformLocation(gShaderProgramObject, "MVPMatrix");

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	/*Vbo Vertices*/
	glGenBuffers(1, &vbo_pos);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);

	glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_VERTEX);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/*Vbo Color*/
	glGenBuffers(1, &vbo_color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_color);

	glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_COLOR);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/*Vbo Velocities*/
	glGenBuffers(1, &vbo_velocities);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_velocities);

	glBufferData(GL_ARRAY_BUFFER, sizeof(velocities), velocities, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_VELOCITIES, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_VELOCITIES);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/*Vbo Start Time*/
	glGenBuffers(1, &vbo_start_time);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_start_time);

	glBufferData(GL_ARRAY_BUFFER, sizeof(startTimes), startTimes, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_START_TIME, 1, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_START_TIME);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//glClearDepth(1.0f);
	//glEnable(GL_DEPTH_TEST);
	//glDepthFunc(GL_LEQUAL);
	//glShadeModel(GL_SMOOTH);
	//glHint(GL_PERSPECTIVE_CORRECTION_HINT , GL_NICEST);

	//createPoints(10, 10);


	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerpectiveProjectionMatrix = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
}

void display(void)
{
	glPointSize(2.0f);	
	void draw_Points(void);
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
		
	//Use Shader Program Object
	glUseProgram(gShaderProgramObject);

	mat4 modelMatrix = mat4::identity();
	mat4 viewMatrix = mat4::identity();
	mat4 rotationMatrix = mat4::identity();
	mat4 mvpMatrix = mat4::identity();

	modelMatrix = translate(0.0f, 0.0f, -3.0f);

	viewMatrix = vmath::lookat(vec3(0.0f, 2.0f, 2.0f), vec3(0.0f, 0.0f, -10.0f), vec3(0.0f, 1.0f, 0.0f));

	rotationMatrix = rotate(45.0f,vec3(1.0f,0.0f,0.0f));
	//viewMatrix = viewMatrix * rotationMatrix;

	mvpMatrix = gPerpectiveProjectionMatrix*viewMatrix*modelMatrix;

	glUniformMatrix4fv(gbPerpectiveProjectionMatrixUniform, 1, GL_FALSE, mvpMatrix);
	glUniform1f(gbParticleTimeUniform, Particle_Time);
	glPointSize(2.0f);
	draw_Points();
	/*glBindVertexArray(vao);

	glDrawArrays(GL_POINTS, 0, arrayWidth*arrayHeight);

	glBindVertexArray(0);*/

	glUseProgram(0);

	SwapBuffers(ghdc);
}

void draw_Points()
{
	glPointSize(2.0f);

	glVertexAttribPointer(VERTEX_ARRAY, 3, GL_FLOAT, GL_FALSE, 0, verts);
	glVertexAttribPointer(COLOR_ARRAY, 3, GL_FLOAT, GL_FALSE, 0, colors);
	glVertexAttribPointer(VELOCITY_ARRAY, 3, GL_FLOAT, GL_FALSE, 0, velocities);
	glVertexAttribPointer(START_TIME_ARRAY, 1, GL_FLOAT, GL_FALSE, 0, startTimes);

	glEnableVertexAttribArray(VERTEX_ARRAY);
	glEnableVertexAttribArray(COLOR_ARRAY);
	glEnableVertexAttribArray(VELOCITY_ARRAY);
	glEnableVertexAttribArray(START_TIME_ARRAY);

	glDrawArrays(GL_POINTS, 0, arrayWidth*arrayHeight);

	glDisableVertexAttribArray(VERTEX_ARRAY);
	glDisableVertexAttribArray(COLOR_ARRAY);
	glDisableVertexAttribArray(VELOCITY_ARRAY);
	glDisableVertexAttribArray(START_TIME_ARRAY);
}

void update(void)
{
	Particle_Time += 0.1f;
	fprintf(gpFile, TEXT("Particle Time : %f\n"), Particle_Time);
}

void resize(int width, int height)
{
	if (height == 0)
		height = 1;
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	gPerpectiveProjectionMatrix = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void createPoints(GLint w, GLint h)
{
	GLfloat *vptr, *cptr, *velptr, *stptr;
	GLfloat i, j;

	if (verts != NULL)
		free(verts);

	verts = (GLfloat*)malloc(w * h * 3 * sizeof(float));
	colors = (GLfloat*)malloc(w * h * 3 * sizeof(float));
	velocities = (GLfloat*)malloc(w * h * 3 * sizeof(float));
	startTimes = (GLfloat*)malloc(w * h * sizeof(float));

	vptr = verts;
	cptr = colors;
	velptr = velocities;
	stptr = startTimes;

	for (i = 0.5 / w - 0.5; i < 0.5; i = i + 1.0 / w)
	{
		for (j = 0.5 / h - 0.5; j < 0.5; j = j + 1.0 / h)
		{
			*vptr = 0.0f;
			*(vptr + 1) = 0.0f;
			*(vptr + 2) = i;
			vptr += 3;

			//gAngle = gAngle + 0.1f;

			*cptr = 1.0f;
			*(cptr + 1) = 1.0f;
			*(cptr + 2) = 1.0f;
			cptr += 3;

			/**cptr = ((float)rand() / RAND_MAX)*0.5 + 0.5;
			*(cptr + 1) = ((float)rand() / RAND_MAX)*0.5 + 0.5;
			*(cptr + 2) = ((float)rand() / RAND_MAX)*0.5 + 0.5;
			cptr += 3;*/

			*velptr = (((float)rand() / RAND_MAX))*sin(gAngle);
			*(velptr + 1) = ((float)rand() / RAND_MAX)*10.0 + 3.0;
			*(velptr + 2) = (((float)rand() / RAND_MAX))*cos(gAngle);
			velptr += 3;

			gAngle = gAngle + 0.1f;
			if (gAngle >= 360.0f)
				gAngle = gAngle - 360.0f;

			*stptr = ((float)rand() / RAND_MAX)*100.0;
			stptr++;
		}
	}
	arrayWidth = w;
	arrayHeight = h;
}

void ToggleFullscreen(void)
{
	MONITORINFO mi = { sizeof(MONITORINFO) };
	if (gbFullscreen == false)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
	}

	else
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}
}

void uninitialize(int i_Exit_Flag)
{
	if (gbFullscreen == false)
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}

	//Detach Shader 
	glDetachShader(gShaderProgramObject, gVertexShaderObject);
	glDetachShader(gShaderProgramObject, gGeometryShaderObject);
	glDetachShader(gShaderProgramObject, gFragmentShaderObject);

	//Delete Shader
	glDeleteShader(gVertexShaderObject);
	gVertexShaderObject = 0;

	glDeleteShader(gGeometryShaderObject);
	gGeometryShaderObject = 0;

	glDeleteShader(gFragmentShaderObject);
	gFragmentShaderObject = 0;

	//Delete Program
	glDeleteProgram(gShaderProgramObject);
	gShaderProgramObject = 0;

	//Stray call to glUseProgram(0)
	glUseProgram(0);

	wglMakeCurrent(NULL, NULL);

	if (ghrc != NULL)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}

	if (ghdc != NULL)
	{
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	DestroyWindow(ghwnd);
}