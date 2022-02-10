#include<windows.h>
#include<GL/glew.h> //Wrangler For PP , add additional headers and lib path 
#include<gl/GL.h>
#include<stdio.h>
#include"vmath.h"
#include"KtxLoader.h"

#pragma comment(lib,"glew32.lib")
#pragma comment(lib,"opengl32.lib")

//global namespace

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

void loadVertexShaderTonemapping(void);

void loadFragmentShaderNaive(void);
void loadFragmentShaderAdaptive(void);
void loadFragmentShaderExposure(void);


void createShaderProgramNaive(void);
void createShaderProgramAdaptive(void);
void createShaderProgramExposure(void);

void deleteShaderProgram(GLuint);

GLuint gShaderProgramObject_naive;
GLuint gShaderProgramObject_adaptive;
GLuint gShaderProgramObject_exposure;

GLuint vertexShaderObject_tonemapping;
const GLchar *vertexShaderSourceCode_tonemapping;

GLuint fragmentShaderObject_naive;
const GLchar *fragmentShaderSourceCode_naive;

GLuint fragmentShaderObject_adaptive;
const GLchar *fragmentShaderSourceCode_adaptive;

GLuint fragmentShaderObject_exposure;
const GLchar *fragmentShaderSourceCode_exposure;

GLuint vao_rectangle;
GLuint vbo_position_rectangle;
GLuint vbo_texture;

GLuint ktxTexImage;
GLuint texLut;

GLuint mvpUniform; 
GLuint samplerUniform;
GLuint exposureUniform;

int mode = 0;
float exposure_value = 1.0f;

int gWidth;
int gHeight;
GLfloat rectangleVertices[12];

mat4 perspectiveProjectionMatrix;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szAppName[] = TEXT("HDR Tone mapping");
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
		TEXT("HRD tone mapping"),
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

	//ToggleFullScreen();
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
				//update();
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
		resize(gWidth,gHeight);
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

		case 'f':
		case 'F':
			ToggleFullScreen();
			break;

		case 0x30:
			mode = 0;
			break;
		
		case 0x31:
			mode = 1;
			break;

		case 0x32:
			mode = 2;
			break;

		case 'M':
			mode = (mode + 1) % 3;
			break;

		case VK_UP:
			exposure_value *= 1.1f;
			break;
		
		case VK_DOWN:
			exposure_value /= 1.1f;
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
	else
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
		gbFullScreen = false;

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

	loadVertexShaderTonemapping();
	
	loadFragmentShaderNaive();
	createShaderProgramNaive();

	loadFragmentShaderExposure();
	createShaderProgramExposure();
	exposureUniform = glGetUniformLocation(gShaderProgramObject_exposure,
		"u_exposure");

	loadFragmentShaderAdaptive();
	createShaderProgramAdaptive();

	const GLfloat rectangleVertices[] = {
		1.0f, 1.0f, 0.0f,
		-1.0f,1.0f,0.0f,
		-1.0f,-1.0f,0.0f,
		1.0f,-1.0f,0.0f };
	
	const GLfloat rectangleTexCoord[] = {
		1.0f, 1.0f,
		0.0f,1.0f,
		0.0f,0.0f,
		1.0f,0.0f };
	
	glGenVertexArrays(1, &vao_rectangle);
	glBindVertexArray(vao_rectangle);
	
	//position
	glGenBuffers(1, &vbo_position_rectangle);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_position_rectangle);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(rectangleVertices),
		rectangleVertices,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	//texture
	glGenBuffers(1, &vbo_texture);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_texture);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(rectangleTexCoord),
		rectangleTexCoord,
		GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOORD0,
		2,
		GL_FLOAT,
		GL_FALSE,
		0,
		NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOORD0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);
	

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	glEnable(GL_TEXTURE_2D);
	ktxTexImage = load("treelights_2k.ktx");
	if (!ktxTexImage)
	{
		fprintf(gpFile, "Failled to load image\n");
	}

	//glBindTexture(GL_TEXTURE_2D, ktxTexImage);
	
	/*static const GLfloat exposureLUT[20] = { 11.0f, 6.0f, 3.2f, 2.8f, 2.2f, 1.90f, 1.80f, 1.80f, 1.70f, 1.70f,  1.60f, 1.60f, 1.50f, 1.50f, 1.40f, 1.40f, 1.30f, 1.20f, 1.10f, 1.00f };
	glGenTextures(1, &texLut);
	glBindTexture(GL_TEXTURE_1D, texLut);
	glTexStorage1D(GL_TEXTURE_1D, 1, GL_R32F, 20);
	glTexSubImage1D(GL_TEXTURE_1D, 0, 0, 20, GL_RED, GL_FLOAT, exposureLUT);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_1D, 0);
	*/

	//make identity
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
	//static const GLfloat black[] = { 0.0f, 0.25f, 0.0f, 1.0f };
	//glViewport(0, 0, gWidth, gHeight);
	
	//glClearBufferfv(GL_COLOR, 0, black);

	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	if (mode == 0)
	{
		glUseProgram(gShaderProgramObject_naive);
	}

	if (mode == 1)
	{
		glUseProgram(gShaderProgramObject_exposure);
		glUniform1f(exposureUniform, exposure_value);
	}

	if (mode == 2)
	{	
		glUseProgram(gShaderProgramObject_adaptive);
	}
		
	mat4 modelViewMatrix;
	mat4 modelViewProjectionMatrix;
	mat4 translationMatrix;

	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();
	translationMatrix = mat4::identity();

	translationMatrix = translate(0.0f, 0.0f, -3.0f);

	modelViewMatrix = modelViewMatrix * translationMatrix;
	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(mvpUniform,
		1,
		GL_FALSE,
		modelViewProjectionMatrix);
	
	//glActiveTexture(GL_TEXTURE1);
	//glBindTexture(GL_TEXTURE_2D, texLut);
	//glUniform1i(samplerUniform, 0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, ktxTexImage);
	glUniform1i(samplerUniform, 0);

	glBindVertexArray(vao_rectangle);

	glDrawArrays(GL_TRIANGLE_FAN,
		0,
		4);

	glBindVertexArray(0);

	glUseProgram(0);

	SwapBuffers(ghdc);

}

void update(void)
{
}

void uninitialize(void)
{
	if (texLut)
	{
		glDeleteTextures(1, &texLut);
		texLut = 0;
	}

	if (ktxTexImage)
	{
		glDeleteTextures(1, &ktxTexImage);
		ktxTexImage = 0;
	}

	if (vbo_texture)
	{
		glDeleteBuffers(1, &vbo_texture);
		vbo_texture = 0;
	}

	if (vbo_position_rectangle)
	{
		glDeleteBuffers(1, &vbo_position_rectangle);
		vbo_position_rectangle = 0;
	}

	if (vao_rectangle)
	{
		glDeleteVertexArrays(1, &vao_rectangle);
		vao_rectangle = 0;
	}

	deleteShaderProgram(gShaderProgramObject_naive);
	deleteShaderProgram(gShaderProgramObject_adaptive);
	deleteShaderProgram(gShaderProgramObject_exposure);

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

void loadVertexShaderTonemapping(void)
{
	vertexShaderObject_tonemapping = glCreateShader(GL_VERTEX_SHADER);

	vertexShaderSourceCode_tonemapping =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec2 vTexCoord;" \
		"uniform mat4 u_mvp_matrix;" \
		"out vec2 out_texcoord;" \
		"void main(void)" \
		"{" \
		"	gl_Position = u_mvp_matrix * vPosition;" \
		"	out_texcoord = vTexCoord;" \
		"}";

	glShaderSource(vertexShaderObject_tonemapping,
		1,
		(const GLchar **)&vertexShaderSourceCode_tonemapping,
		NULL);

	glCompileShader(vertexShaderObject_tonemapping);

	GLint iShaderCompileStatus = 0;
	GLint iInfoLogLength = 0;
	GLchar *szInfoLog = NULL;

	glGetShaderiv(vertexShaderObject_tonemapping,
		GL_COMPILE_STATUS,
		&iShaderCompileStatus);

	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(vertexShaderObject_tonemapping,
			GL_INFO_LOG_LENGTH,
			&iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;

				glGetShaderInfoLog(vertexShaderObject_tonemapping,//whose?
					iInfoLogLength,//length?
					&written,//might have not used all, give that much only which have been used in what?
					szInfoLog);//store in what?

				fprintf(gpFile, "\nVertex Shader Tone mapping Compilation Log : %s\n", szInfoLog);

				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}
}

void loadFragmentShaderNaive(void)
{
	fragmentShaderObject_naive = glCreateShader(GL_FRAGMENT_SHADER);

	fragmentShaderSourceCode_naive =
		"#version 450 core" \
		"\n" \
		"in vec2 out_texcoord;" \
		"uniform sampler2D u_sampler;" \
		"out vec4 fragColor;" \
		"void main(void)" \
		"{" \
		"	fragColor = texture(u_sampler, out_texcoord);" \
		"}";

	glShaderSource(fragmentShaderObject_naive,
		1,
		(const GLchar **)&fragmentShaderSourceCode_naive,
		NULL);

	glCompileShader(fragmentShaderObject_naive);

	GLint iShaderCompileStatus = 0;
	GLint iInfoLogLength = 0;
	GLchar *szInfoLog = NULL;

	glGetShaderiv(fragmentShaderObject_naive,
		GL_COMPILE_STATUS,
		&iShaderCompileStatus);

	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(fragmentShaderObject_naive,
			GL_INFO_LOG_LENGTH,
			&iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);

			if (szInfoLog != NULL)
			{
				GLsizei written;

				glGetShaderInfoLog(fragmentShaderObject_naive,
					iInfoLogLength,
					&written,
					szInfoLog);

				fprintf(gpFile, "\nFragment Shader Naive Compilation Log : %s\n", szInfoLog);

				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}
	

}

void loadFragmentShaderAdaptive(void)
{
	fragmentShaderObject_adaptive = glCreateShader(GL_FRAGMENT_SHADER);

	fragmentShaderSourceCode_adaptive =
		"#version 450 core" \
		"\n" \
		"in vec2 out_texcoord;" \
		"uniform sampler2D u_sampler;" \
		"out vec4 fragColor;" \
		"void main(void)" \
		"{" \
		"	int i;" \
		"	float lum[25];" \
		"	vec2 tex_scale = vec2(1.0) / textureSize(u_sampler, 0);" \
		"	for(i = 0; i < 25; i++)" \
		"	{" \
		"		vec2 tc = (2.0 * gl_FragCoord.xy + 3.5 * vec2(i % 5 - 2, i / 5 - 2));" \
		"		vec3 col = texture(u_sampler, tc * tex_scale).rgb;" \
		"		lum[i] = dot(col, vec3(0.3, 0.59, 0.11));" \
		"	}" \
		"	vec3 vColor = texelFetch(u_sampler, 2 * ivec2(gl_FragCoord.xy), 0).rgb;" \
		"	float kernelLuminance = ((1.0 * (lum[0] + lum[4] + lum[20] + lum[24])) +(4.0 * (lum[1] + lum[3] + lum[5] + lum[9] +lum[15] + lum[19] + lum[21] + lum[23])) +(7.0 * (lum[2] + lum[10] + lum[14] + lum[22])) +(16.0 * (lum[6] + lum[8] + lum[16] + lum[18])) +(26.0 * (lum[7] + lum[11] + lum[13] + lum[17])) +(41.0 * lum[12])) / 273.0;" \
		"	float exposure = sqrt(8.0 / (kernelLuminance + 0.25));" \
		"	fragColor.rgb = 1.0 - exp2(-vColor * exposure);" \
		"	fragColor.a = 1.0f;"\
		"}";

	glShaderSource(fragmentShaderObject_adaptive,
		1,
		(const GLchar **)&fragmentShaderSourceCode_adaptive,
		NULL);

	glCompileShader(fragmentShaderObject_adaptive);

	GLint iShaderCompileStatus = 0;
	GLint iInfoLogLength = 0;
	GLchar *szInfoLog = NULL;

	glGetShaderiv(fragmentShaderObject_adaptive,
		GL_COMPILE_STATUS,
		&iShaderCompileStatus);

	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(fragmentShaderObject_adaptive,
			GL_INFO_LOG_LENGTH,
			&iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);

			if (szInfoLog != NULL)
			{
				GLsizei written;

				glGetShaderInfoLog(fragmentShaderObject_adaptive,
					iInfoLogLength,
					&written,
					szInfoLog);

				fprintf(gpFile, "\nFragment Shader Adaptive Compilation Log : %s\n", szInfoLog);

				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}

}

void loadFragmentShaderExposure(void)
{
	fragmentShaderObject_exposure = glCreateShader(GL_FRAGMENT_SHADER);

	fragmentShaderSourceCode_exposure =
		"#version 450 core" \
		"\n" \
		"in vec2 out_texcoord;" \
		"uniform float u_exposure;" \
		"uniform sampler2D u_sampler;" \
		"out vec4 fragColor;" \
		"void main(void)" \
		"{" \
		"	vec4 c = texelFetch(u_sampler, 2 * ivec2(gl_FragCoord.xy), 0);"\
		"	c.rgb = vec3(1.0) - exp(-c.rgb * u_exposure);" \
		"	fragColor = c; " \
		"}";

	glShaderSource(fragmentShaderObject_exposure,
		1,
		(const GLchar **)&fragmentShaderSourceCode_exposure,
		NULL);

	glCompileShader(fragmentShaderObject_exposure);

	GLint iShaderCompileStatus = 0;
	GLint iInfoLogLength = 0;
	GLchar *szInfoLog = NULL;

	glGetShaderiv(fragmentShaderObject_exposure,
		GL_COMPILE_STATUS,
		&iShaderCompileStatus);

	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(fragmentShaderObject_exposure,
			GL_INFO_LOG_LENGTH,
			&iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);

			if (szInfoLog != NULL)
			{
				GLsizei written;

				glGetShaderInfoLog(fragmentShaderObject_exposure,
					iInfoLogLength,
					&written,
					szInfoLog);

				fprintf(gpFile, "\nFragment Shader Exposure Compilation Log : %s\n", szInfoLog);

				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}

}

void createShaderProgramNaive(void)
{
	gShaderProgramObject_naive = glCreateProgram();

	glAttachShader(gShaderProgramObject_naive,
		vertexShaderObject_tonemapping);

	glAttachShader(gShaderProgramObject_naive,
		fragmentShaderObject_naive);

	glBindAttribLocation(gShaderProgramObject_naive,
		AMC_ATTRIBUTE_POSITION,
		"vPosition");

	glBindAttribLocation(gShaderProgramObject_naive,
		AMC_ATTRIBUTE_TEXCOORD0,
		"vTexCoord");

	//Link the shader program
	glLinkProgram(gShaderProgramObject_naive);

	GLint iProgramLinkStatus = 0;
	GLint iInfoLogLength = 0;
	GLchar *szInfoLog = NULL;

	glGetProgramiv(gShaderProgramObject_naive,
		GL_LINK_STATUS,
		&iProgramLinkStatus);

	if (iProgramLinkStatus == GL_FALSE)
	{

		glGetProgramiv(gShaderProgramObject_naive,
			GL_INFO_LOG_LENGTH,
			&iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;

				glGetProgramInfoLog(gShaderProgramObject_naive,
					iInfoLogLength,
					&written,
					szInfoLog);

				fprintf(gpFile, "\nShader Program Naive Linking Log : %s\n", szInfoLog);

				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}

	mvpUniform = glGetUniformLocation(gShaderProgramObject_naive,
		"u_mvp_matrix");

	samplerUniform = glGetUniformLocation(gShaderProgramObject_naive,
		"u_sampler");
	
}

void createShaderProgramExposure(void)
{
	gShaderProgramObject_exposure = glCreateProgram();

	glAttachShader(gShaderProgramObject_exposure,
		vertexShaderObject_tonemapping);

	glAttachShader(gShaderProgramObject_exposure,
		fragmentShaderObject_exposure);

	glBindAttribLocation(gShaderProgramObject_exposure,
		AMC_ATTRIBUTE_POSITION,
		"vPosition");

	glBindAttribLocation(gShaderProgramObject_exposure,
		AMC_ATTRIBUTE_TEXCOORD0,
		"vTexCoord");

	//Link the shader program
	glLinkProgram(gShaderProgramObject_exposure);

	GLint iProgramLinkStatus = 0;
	GLint iInfoLogLength = 0;
	GLchar *szInfoLog = NULL;

	glGetProgramiv(gShaderProgramObject_exposure,
		GL_LINK_STATUS,
		&iProgramLinkStatus);

	if (iProgramLinkStatus == GL_FALSE)
	{

		glGetProgramiv(gShaderProgramObject_exposure,
			GL_INFO_LOG_LENGTH,
			&iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;

				glGetProgramInfoLog(gShaderProgramObject_exposure,
					iInfoLogLength,
					&written,
					szInfoLog);

				fprintf(gpFile, "\nShader Program Exposure Linking Log : %s\n", szInfoLog);

				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}

	mvpUniform = glGetUniformLocation(gShaderProgramObject_exposure,
		"u_mvp_matrix");

	samplerUniform = glGetUniformLocation(gShaderProgramObject_exposure,
		"u_sampler");
}

void createShaderProgramAdaptive(void)
{
	gShaderProgramObject_adaptive = glCreateProgram();

	glAttachShader(gShaderProgramObject_adaptive,
		vertexShaderObject_tonemapping);

	glAttachShader(gShaderProgramObject_adaptive,
		fragmentShaderObject_adaptive);

	glBindAttribLocation(gShaderProgramObject_adaptive,
		AMC_ATTRIBUTE_POSITION,
		"vPosition");

	glBindAttribLocation(gShaderProgramObject_adaptive,
		AMC_ATTRIBUTE_TEXCOORD0,
		"vTexCoord");

	//Link the shader program
	glLinkProgram(gShaderProgramObject_adaptive);

	GLint iProgramLinkStatus = 0;
	GLint iInfoLogLength = 0;
	GLchar *szInfoLog = NULL;

	glGetProgramiv(gShaderProgramObject_adaptive,
		GL_LINK_STATUS,
		&iProgramLinkStatus);

	if (iProgramLinkStatus == GL_FALSE)
	{

		glGetProgramiv(gShaderProgramObject_adaptive,
			GL_INFO_LOG_LENGTH,
			&iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;

				glGetProgramInfoLog(gShaderProgramObject_adaptive,
					iInfoLogLength,
					&written,
					szInfoLog);

				fprintf(gpFile, "\nShader Program Adaptive Linking Log : %s\n", szInfoLog);

				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}

	mvpUniform = glGetUniformLocation(gShaderProgramObject_adaptive,
		"u_mvp_matrix");

	samplerUniform = glGetUniformLocation(gShaderProgramObject_adaptive,
		"u_sampler");

}

void deleteShaderProgram(GLuint ShaderProgramObject)
{
	if (ShaderProgramObject)
	{
		GLsizei shaderCount;
		GLsizei shaderNumber;

		glUseProgram(ShaderProgramObject);

		glGetProgramiv(ShaderProgramObject,
			GL_ATTACHED_SHADERS,
			&shaderCount);

		GLuint *pShaders = (GLuint *)malloc(sizeof(GLuint) * shaderCount);

		if (pShaders)
		{
			glGetAttachedShaders(ShaderProgramObject,
				shaderCount,
				&shaderCount,
				pShaders);

			for (shaderNumber = 0; shaderNumber < shaderCount; shaderNumber++)
			{
				glDetachShader(ShaderProgramObject,
					pShaders[shaderNumber]);

				glDeleteShader(pShaders[shaderNumber]);

				pShaders[shaderNumber] = 0;
			}

			free(pShaders);
		}

		glDeleteProgram(ShaderProgramObject);
		ShaderProgramObject = 0;

		glUseProgram(0);
	}
}
