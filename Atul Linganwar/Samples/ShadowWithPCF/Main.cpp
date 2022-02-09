#include<Windows.h>
#include<stdio.h>
#include<stdlib.h>

#include<gl/glew.h>
#include<gl/GL.h>

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>

#include "Camera.h"
#include "Timer.h"
#include"BasicShapes.h"


#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")

#define WINWIDTH	800
#define WINHEIGHT	600

enum
{
	VDG_ATTRIBUTE_POSITION = 0,
	VDG_ATTRIBUTE_COLOR,
	VDG_ATTRIBUTE_NORMAL,
	VDG_ATTRIBUTE_TEXTURE0,
};

//declaration of callback procedure
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

//global variables
FILE *gpFile = NULL;

HWND ghwnd = NULL;
HDC ghdc = NULL;
HGLRC ghrc = NULL;

DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

bool gbActiveWindow = false;
bool gbEscapeKeyIsPressed = false;
bool gbFullscreen = false;

GLuint gFbo;
GLuint depthBufferObject;

GLuint gVertexShaderObjectForDepth;
GLuint gFragmentShaderObjectForDepth;
GLuint gShaderProgramObjectForDepth;

GLuint gVertexShaderObjectForShadow;
GLuint gFragmentShaderObjectForShadow;
GLuint gShaderProgramObjectForShadow;

GLuint gVao_cube;
GLuint gVao_pyramid;
GLuint gVao_ground;
GLuint gVao_quad;

GLuint gVbo_position;
GLuint gVbo_color;
GLuint gVbo_normal;
GLuint gVbo_texcoord;

GLuint gLightSpaceMatrixUniform;
GLuint gModelMatrixUniform;

GLuint gTextureSamplerUniform;
GLuint gNearPlaneUniform;
GLuint gFarPlaneUniform;

GLuint gModelMatrixUniform1;
GLuint gViewMatrixUniform1;
GLuint gProjectionMatrixUniform1;

GLuint gModelMatrixUniformForShadow;
GLuint gViewMatrixUniformForShadow;
GLuint gProjectionMatrixUniformForShadow;
GLuint gLightSpaceMatrixUniformForShadow;
GLuint gShadowMapUniformSamplerForShadow;
GLuint gLightPositionUniformForShadow;
GLuint gViewPositionUniformForShadow;

glm::mat4 gPerspectiveProjectionMatrix;

GLint gWinWidth;
GLint gWinHeight;

bool gbFirstMouse = true;
float gfLastX;
float gfLastY;

GLfloat xAngle, yAngle, zAngle;

CCamera camera(glm::vec3(0.0f, 0.0f, 12.0f),glm::vec3(0.0f, 1.0f, 0.0f));
Timer timer;

CCone cone;
CSphere sphere;
CRing ring;
CTorus torus;
CCylinder cylinder;

//WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int nCmdShow)
{
	//function prototype
	void initialize(void);
	void display(void);
	void update(void);
	void uninitialize(void);

	//variables declaration
	WNDCLASSEX wndclass;
	MSG msg;
	HWND hwnd;
	TCHAR szClassName[] = TEXT("OpenGLPP");

	bool bDone = false;

	//code
	//log file
	if (fopen_s(&gpFile, "Log.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Log File can not be created"), TEXT("Error"), MB_OK);
		ExitProcess(EXIT_FAILURE);
	}
	else
		fprintf(gpFile, "Log file created successfully\n");

	//initialize wndclass
	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.hInstance = hInstance;
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.lpfnWndProc = WndProc;
	wndclass.lpszClassName = szClassName;
	wndclass.lpszMenuName = NULL;
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

	//register class
	if (!RegisterClassEx(&wndclass))
	{
		fprintf(gpFile, "Error while registering the class\n");
		fclose(gpFile);
		ExitProcess(EXIT_FAILURE);
	}

	int x = GetSystemMetrics(SM_CXSCREEN);
	int y = GetSystemMetrics(SM_CYSCREEN);

	//Create Window
	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szClassName,
		TEXT("Shadow"),
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		(x / 2) - (WINWIDTH / 2),
		(y / 2) - (WINHEIGHT / 2),
		WINWIDTH,
		WINHEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	if (!hwnd)
	{
		fprintf(gpFile, "Error while creating window\n");
		fclose(gpFile);
		ExitProcess(EXIT_FAILURE);
	}

	ghwnd = hwnd;

	ShowWindow(hwnd, nCmdShow);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	//initialize
	initialize();
	timer.InitializeTimer();

	//game loop
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
			//render
			timer.SetDeltaTime();
			timer.CalculateFPS();

			update();
			display();

			if (gbActiveWindow == true)
			{
				if (gbEscapeKeyIsPressed == true)
					bDone = true;
			}
		}
	}

	//uninitialize
	uninitialize();

	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	//function prototype
	void resize(int, int);
	void ToggleFullscreen(void);
	void uninitialize(void);

	//variable declaration
	static WORD xMouse = NULL;
	static WORD yMouse = NULL;

	int mouseX;
	int mouseY;
	DWORD flags;

	GLfloat fXOffset;
	GLfloat fYOffset;

	//code
	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			gbActiveWindow = true;
		else
			gbActiveWindow = false;
		break;

		//case WM_ERASEBKGND:
		//	return(0);

	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		gWinWidth = LOWORD(lParam);
		gWinHeight = HIWORD(lParam);
		break;

	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			if (gbEscapeKeyIsPressed == false)
				gbEscapeKeyIsPressed = true;
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

		case VK_UP:
			camera.ProcessNavigationKeys(CameraMovement::CAMERA_FORWARD, timer.GetDeltaTime());
			break;

		case VK_DOWN:
			camera.ProcessNavigationKeys(CameraMovement::CAMERA_BACKWARD, timer.GetDeltaTime());
			break;

		case VK_LEFT:
			camera.ProcessNavigationKeys(CameraMovement::CAMERA_LEFT, timer.GetDeltaTime());
			break;

		case VK_RIGHT:
			camera.ProcessNavigationKeys(CameraMovement::CAMERA_RIGHT, timer.GetDeltaTime());
			break;

		case VK_PRIOR:
			camera.ProcessNavigationKeys(CameraMovement::CAMERA_UP, timer.GetDeltaTime());
			break;

		case VK_NEXT:
			camera.ProcessNavigationKeys(CameraMovement::CAMERA_DOWN, timer.GetDeltaTime());
			break;

		default:
			break;
		}
		break;

	case WM_MOUSEMOVE:
		mouseX = LOWORD(lParam);
		mouseY = HIWORD(lParam);
		flags = (DWORD)wParam;
		if (gbFirstMouse)
		{
			gfLastX = (GLfloat)mouseX;
			gfLastY = (GLfloat)mouseY;
			gbFirstMouse = GL_FALSE;
		}

		fXOffset = mouseX - gfLastX;
		fYOffset = gfLastY - mouseY;

		gfLastX = (GLfloat)mouseX;
		gfLastY = (GLfloat)mouseY;

		if (flags & MK_LBUTTON)
		{
			camera.ProcessMouseMovement(fXOffset, fYOffset, GL_TRUE);
		}
		break;

	case WM_LBUTTONDOWN:
		break;

	case WM_CLOSE:
		uninitialize();
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	default:
		break;
	}

	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullscreen(void)
{
	//variable declaration
	MONITORINFO mi;

	//code
	if (gbFullscreen == false)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			mi = { sizeof(MONITORINFO) };
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
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);

		ShowCursor(TRUE);
	}
}

//initialize
void initialize(void)
{
	///function prototype
	void uninitialize(void);
	void resize(int, int);

	//variables
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

	//code
	ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

	//Initialize the structure pfd
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

	ghdc = GetDC(ghwnd);

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0)
	{
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == false)
	{
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}
	if (wglMakeCurrent(ghdc, ghrc) == false)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	//GLEW initialization code
	GLenum glew_error = glewInit();
	if (glew_error != GLEW_OK)
	{
		fprintf(gpFile, "Error at glewInit()\n");
		wglDeleteContext(ghrc);
		ghrc = NULL;
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	//Vertex Shader
	gVertexShaderObjectForDepth = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *vertexShaderSourceCode =
		"#version 400 core"		\
		"\n"					\
		"in vec4 vPosition;"	\
		"uniform mat4 lightSpaceMatrix;" \
		"uniform mat4 modelMatrix;" \
		"void main(void)"		\
		"{"						\
		"gl_Position = lightSpaceMatrix * modelMatrix * vPosition;" \
		"}";

	glShaderSource(gVertexShaderObjectForDepth, 1, (const GLchar **)&vertexShaderSourceCode, NULL);

	//compile shader
	glCompileShader(gVertexShaderObjectForDepth);
	GLint iInfoLogLength = 0;
	GLint iShaderCompileStatus = 0;
	char *szInfoLog = NULL;

	glGetShaderiv(gVertexShaderObjectForDepth, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(gVertexShaderObjectForDepth, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gVertexShaderObjectForDepth, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Vertex Shader Compilation Log: %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize();
				ExitProcess(EXIT_FAILURE);
			}
		}
	}

	//Fragment Shader
	gFragmentShaderObjectForDepth = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *fragmentShaderSourceCode =
		"#version 400 core" \
		"\n" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
			/*"FragColor = vec4(1.0f);" \*/
		"}";

	glShaderSource(gFragmentShaderObjectForDepth, 1, (const GLchar **)&fragmentShaderSourceCode, NULL);

	//compile shader
	glCompileShader(gFragmentShaderObjectForDepth);

	glGetShaderiv(gFragmentShaderObjectForDepth, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(gFragmentShaderObjectForDepth, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gFragmentShaderObjectForDepth, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Fragment Shader compilation log: %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize();
				ExitProcess(EXIT_FAILURE);
			}
		}
	}

	//Shader Program
	gShaderProgramObjectForDepth = glCreateProgram();

	//attach shaders
	glAttachShader(gShaderProgramObjectForDepth, gVertexShaderObjectForDepth);
	glAttachShader(gShaderProgramObjectForDepth, gFragmentShaderObjectForDepth);

	glBindAttribLocation(gShaderProgramObjectForDepth, VDG_ATTRIBUTE_POSITION, "vPosition");

	//link shader
	glLinkProgram(gShaderProgramObjectForDepth);
	GLint iShaderProgramLinkStatus = 0;
	glGetProgramiv(gShaderProgramObjectForDepth, GL_LINK_STATUS, &iShaderProgramLinkStatus);
	if (iShaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(gShaderProgramObjectForDepth, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObjectForDepth, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Shader Program Link log: %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize();
				ExitProcess(EXIT_FAILURE);
			}
		}
	}

	//get MVP Uniform Location
	gLightSpaceMatrixUniform = glGetUniformLocation(gShaderProgramObjectForDepth, "lightSpaceMatrix");
	gModelMatrixUniform = glGetUniformLocation(gShaderProgramObjectForDepth, "modelMatrix");

	//SHADOW CALCULATIONS
	gVertexShaderObjectForShadow = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *vertexShaderSourceForShadow =
		"#version 400 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vNormal;" \
		"in vec4 vColor;" \
		"out VS_OUT" \
		"{" \
		"vec3 FragPos;" \
		"vec3 Normals;" \
		"vec4 Colors;" \
		"vec4 FragPosLightSpace;" \
		"}vs_out;" \
		"uniform mat4 u_projectionMatrix;" \
		"uniform mat4 u_viewMatrix;" \
		"uniform mat4 u_modelMatrix;" \
		"uniform mat4 u_lightSpaceMatrix;" \
		"void main(void)" \
		"{" \
		"vs_out.FragPos = vec3(u_modelMatrix * vPosition);" \
		"vs_out.Normals = transpose(inverse(mat3(u_modelMatrix))) * vNormal;" \
		"vs_out.Colors = vColor;" \
		"vs_out.FragPosLightSpace = u_lightSpaceMatrix * vec4(vs_out.FragPos, 1.0);" \
		"gl_Position = u_projectionMatrix * u_viewMatrix * u_modelMatrix * vPosition;" \
		"}";

	glShaderSource(gVertexShaderObjectForShadow, 1, (const GLchar **)&vertexShaderSourceForShadow, NULL);

	glCompileShader(gVertexShaderObjectForShadow);

	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(gVertexShaderObjectForShadow, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(gVertexShaderObjectForShadow, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gVertexShaderObjectForShadow, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Vertex Shader For Shadow Compilation Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize();
				ExitProcess(EXIT_FAILURE);
			}
		}
	}

	//frament shader
	gFragmentShaderObjectForShadow = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *fragmentShaderSourceForShadow =
		"#version 400 core" \
		"\n" \
		"in VS_OUT" \
		"{" \
		"vec3 FragPos;" \
		"vec3 Normals;" \
		"vec4 Colors;" \
		"vec4 FragPosLightSpace;" \
		"}fs_in;" \
		"uniform sampler2D u_shadowMapSampler;" \
		"uniform vec3 u_lightPos;" \
		"uniform vec3 u_viewPos;" \
		"out vec4 FragColor;" \
		"float ShadowCalculation(vec4 fragPosLightSpace, float bias)" \
		"{" \
		/*perform perspective divide*/
		"vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;" \
		/*transform to [0, 1] range*/
		"projCoords = projCoords * 0.5 + 0.5;" \
		/*get closet depth value from light's perspective (using [0, 1] range FragPosLight as coords)*/
		"float closestDepth = texture(u_shadowMapSampler, projCoords.xy).r;" \
		/*get depth of current fragment from light's perspective*/
		"float currentDepth = projCoords.z;" \
		"float shadow = 0.0;" \
		"vec2 texelSize = 1.0 / textureSize(u_shadowMapSampler, 0);" \
		"for(int x = -1; x <= 1; ++x)" \
		"{" \
		"for(int y = -1; y <= 1; ++y)" \
		"{" \
		"float pcfDepth = texture(u_shadowMapSampler, projCoords.xy + vec2(x, y) * texelSize).r;" \
		"shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;" \
		"}" \
		"}" \
		"shadow /= 9.0f;" \
		/*"shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;" \*/
		"if(projCoords.z > 1.0)" \
		"{" \
		"shadow = 0.0;"
		"}" \
		"return(shadow);" \
		"}" \
		"void main(void)" \
		"{" \
		"vec4 color = fs_in.Colors;" \
		"vec3 normal = normalize(fs_in.Normals);" \
		"vec3 lightColor = vec3(1.0, 1.0, 0.3);" \
		"vec3 ambient = 0.15 * color.xyz;" \
		/*diffuse*/
		"vec3 lightDir = normalize(u_lightPos - fs_in.FragPos);" \
		"float diff = max(dot(lightDir, normal), 0.0);" \
		"vec3 diffuse = diff * lightColor;" \
		/*specular*/
		"vec3 viewDir = normalize(u_viewPos - fs_in.FragPos);" \
		"vec3 reflectDir = reflect(-lightDir, normal);" \
		"float spec = 0.0;" \
		"vec3 halfwayDir = normalize(lightDir + viewDir);" \
		"spec = pow(max(dot(normal, halfwayDir), 0.0), 64.0f);" \
		"vec3 specular = spec * lightColor;" \
		/*calculate shadow*/
		"float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);" \
		"float shadow = ShadowCalculation(fs_in.FragPosLightSpace, bias);" \
		"vec3 lighting = color.xyz * (ambient + (1.0 - shadow) * (diffuse + specular));" \
		"FragColor = vec4(lighting, 1.0);" \
		"}";

	glShaderSource(gFragmentShaderObjectForShadow, 1, (const GLchar **)&fragmentShaderSourceForShadow, NULL);

	//compile shader
	glCompileShader(gFragmentShaderObjectForShadow);

	glGetShaderiv(gFragmentShaderObjectForShadow, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(gFragmentShaderObjectForShadow, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gFragmentShaderObjectForShadow, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Fragment Shader For Shadow compilation log: %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize();
				ExitProcess(EXIT_FAILURE);
			}
		}
	}

	//Shader Program
	gShaderProgramObjectForShadow = glCreateProgram();

	//attach shaders
	glAttachShader(gShaderProgramObjectForShadow, gVertexShaderObjectForShadow);
	glAttachShader(gShaderProgramObjectForShadow, gFragmentShaderObjectForShadow);

	glBindAttribLocation(gShaderProgramObjectForShadow, VDG_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(gShaderProgramObjectForShadow, VDG_ATTRIBUTE_NORMAL, "vNormal");
	glBindAttribLocation(gShaderProgramObjectForShadow, VDG_ATTRIBUTE_COLOR, "vColors");

	//link shader
	glLinkProgram(gShaderProgramObjectForShadow);
	iShaderProgramLinkStatus = 0;
	glGetProgramiv(gShaderProgramObjectForShadow, GL_LINK_STATUS, &iShaderProgramLinkStatus);
	if (iShaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(gShaderProgramObjectForShadow, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObjectForShadow, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Shader Program For Shadow Link log: %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize();
				ExitProcess(EXIT_FAILURE);
			}
		}
	}

	gModelMatrixUniformForShadow = glGetUniformLocation(gShaderProgramObjectForShadow, "u_modelMatrix");
	gViewMatrixUniformForShadow = glGetUniformLocation(gShaderProgramObjectForShadow, "u_viewMatrix");
	gProjectionMatrixUniformForShadow = glGetUniformLocation(gShaderProgramObjectForShadow, "u_projectionMatrix");
	gLightSpaceMatrixUniformForShadow = glGetUniformLocation(gShaderProgramObjectForShadow, "u_lightSpaceMatrix");
	gShadowMapUniformSamplerForShadow = glGetUniformLocation(gShaderProgramObjectForShadow, "u_shadowMap");
	gLightPositionUniformForShadow = glGetUniformLocation(gShaderProgramObjectForShadow, "u_lightPos");
	gViewPositionUniformForShadow = glGetUniformLocation(gShaderProgramObjectForShadow, "u_viewPos");

	cone.InitializeCone(2.0f, 5.0f, 20, false);
	sphere.InitializeSphere(2.0f, 13, 26);
	torus.InitializeTorus(2.0f, 5.0f, 13, 26);
	cylinder.InitializeCylinder(2.0f, 5.0f, 20, false, false);
	ring.InitializeRing(2.0f, 10.0f, 20);

	//VAOs AND VBOs
	//attributes initialization
	const GLfloat groundVertices[] =
	{
		10.0f, 0.0f, -10.0f,
		-10.0f, 0.0f, -10.0f,
		-10.0f, 0.0f, 10.0f,
		10.0f, 0.0f, 10.0f
	};

	const GLfloat groundNormals[] =
	{
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f
	};

	const GLfloat groundTexCoords[] =
	{
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	const GLfloat quadVertices[] =
	{
		1.0f, 1.0f, 0.0f,
		-1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f
	};

	const GLfloat quadNormals[] =
	{
		0.0f, 0.0f, -1.0f,
		0.0f, 0.0f, -1.0f,
		0.0f, 0.0f, -1.0f,
		0.0f, 0.0f, -1.0f
	};

	const GLfloat quadTexCoords[] =
	{
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	{
		//Ground
		glGenVertexArrays(1, &gVao_ground);
		glBindVertexArray(gVao_ground);

		glGenBuffers(1, &gVbo_position);
		glBindBuffer(GL_ARRAY_BUFFER, gVbo_position);
		glBufferData(GL_ARRAY_BUFFER, sizeof(groundVertices), groundVertices, GL_STATIC_DRAW);
		glVertexAttribPointer(VDG_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(VDG_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glVertexAttrib3f(VDG_ATTRIBUTE_COLOR, 0.996f, 0.659f, 0.467f);

		glGenBuffers(1, &gVbo_normal);
		glBindBuffer(GL_ARRAY_BUFFER, gVbo_normal);
		glBufferData(GL_ARRAY_BUFFER, sizeof(groundNormals), groundNormals, GL_STATIC_DRAW);
		glVertexAttribPointer(VDG_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(VDG_ATTRIBUTE_NORMAL);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glGenBuffers(1, &gVbo_texcoord);
		glBindBuffer(GL_ARRAY_BUFFER, gVbo_texcoord);
		glBufferData(GL_ARRAY_BUFFER, sizeof(groundTexCoords), groundTexCoords, GL_STATIC_DRAW);
		glVertexAttribPointer(VDG_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(VDG_ATTRIBUTE_TEXTURE0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindVertexArray(0);

		//Quad
		glGenVertexArrays(1, &gVao_quad);
		glBindVertexArray(gVao_quad);

		glGenBuffers(1, &gVbo_position);
		glBindBuffer(GL_ARRAY_BUFFER, gVbo_position);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
		glVertexAttribPointer(VDG_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(VDG_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glGenBuffers(1, &gVbo_normal);
		glBindBuffer(GL_ARRAY_BUFFER, gVbo_normal);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadNormals), quadNormals, GL_STATIC_DRAW);
		glVertexAttribPointer(VDG_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(VDG_ATTRIBUTE_NORMAL);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glGenBuffers(1, &gVbo_texcoord);
		glBindBuffer(GL_ARRAY_BUFFER, gVbo_texcoord);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadTexCoords), quadTexCoords, GL_STATIC_DRAW);
		glVertexAttribPointer(VDG_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(VDG_ATTRIBUTE_TEXTURE0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindVertexArray(0);

		//FRAME BUFFER
		glGenFramebuffers(1, &gFbo);
		glBindFramebuffer(GL_FRAMEBUFFER, gFbo);

		glGenTextures(1, &depthBufferObject);
		glBindTexture(GL_TEXTURE_2D, depthBufferObject);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, WINWIDTH, WINHEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
		glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthBufferObject, 0);
		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_CULL_FACE);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix = glm::mat4(1.0f);

	resize(WINWIDTH, WINHEIGHT);
}

//DISPLAY
void display(void)
{
	GLfloat degToRad(GLfloat degree);

	//code
	GLfloat near_plane = 1.0f, far_plane = 40.0f;
	GLfloat dimension = 50.0f;
	glm::mat4 lightProjection = glm::ortho(-dimension, dimension, -dimension, dimension, near_plane, far_plane);
	glm::mat4 lightView = glm::lookAt(glm::vec3(-2.0f, 15.0f, -1.0f),
		glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f));

	glm::mat4 lightSpaceMatrix = lightProjection * lightView;

	glUseProgram(gShaderProgramObjectForDepth);
	glCullFace(GL_FRONT);
	//First pass
	glViewport(0, 0, WINWIDTH, WINHEIGHT);

	glBindFramebuffer(GL_FRAMEBUFFER, gFbo);
	glClear(GL_DEPTH_BUFFER_BIT);

	glm::mat4 modelMatrix = glm::mat4(1.0f);
	glm::mat4 rotationMatrix = glm::mat4(1.0f);

	modelMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(4.0f, 0.0f, -3.0f));

	rotationMatrix = glm::rotate(glm::mat4(1.0f), degToRad(xAngle), glm::vec3(1.0f, 0.0f, 0.0f));
	modelMatrix *= rotationMatrix;
	rotationMatrix = glm::rotate(glm::mat4(1.0f), degToRad(yAngle), glm::vec3(0.0f, 1.0f, 0.0f));
	modelMatrix *= rotationMatrix;
	rotationMatrix = glm::rotate(glm::mat4(1.0f), degToRad(zAngle), glm::vec3(0.0f, 0.0f, 1.0f));
	modelMatrix *= rotationMatrix;

	glUniformMatrix4fv(gLightSpaceMatrixUniform, 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
	glUniformMatrix4fv(gModelMatrixUniform, 1, GL_FALSE, glm::value_ptr(modelMatrix));

	//sphere.DrawSphere();
	//cone.DrawCone();
	cylinder.DrawCylinder();
	//torus.DrawTorus();
	//disk.DrawDisk();

	modelMatrix = glm::mat4(1.0f);
	rotationMatrix = glm::mat4(1.0f);

	modelMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(-4.0f, 0.0f, -3.0f));

	rotationMatrix = glm::rotate(glm::mat4(1.0f), degToRad(xAngle), glm::vec3(1.0f, 0.0f, 0.0f));
	modelMatrix *= rotationMatrix;
	rotationMatrix = glm::rotate(glm::mat4(1.0f), degToRad(yAngle), glm::vec3(0.0f, 1.0f, 0.0f));
	modelMatrix *= rotationMatrix;
	rotationMatrix = glm::rotate(glm::mat4(1.0f), degToRad(zAngle), glm::vec3(0.0f, 0.0f, 1.0f));
	modelMatrix *= rotationMatrix;

	glUniformMatrix4fv(gLightSpaceMatrixUniform, 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
	glUniformMatrix4fv(gModelMatrixUniform, 1, GL_FALSE, glm::value_ptr(modelMatrix));

	//sphere.DrawSphere();
	cone.DrawCone();
	//cylinder.DrawCylinder();
	//torus.DrawTorus();
	//disk.DrawDisk();

	modelMatrix = glm::mat4(1.0f);
	modelMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -5.0f, -3.0f));

	glUniformMatrix4fv(gLightSpaceMatrixUniform, 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
	glUniformMatrix4fv(gModelMatrixUniform, 1, GL_FALSE, glm::value_ptr(modelMatrix));

	glBindVertexArray(gVao_ground);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glCullFace(GL_BACK);
	glUseProgram(0);
//_--------------------------------------------------------------------------------------------------------------------------------
	//Draw with shadows
	glViewport(0, 0, gWinWidth, gWinHeight);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(gShaderProgramObjectForShadow);

	//TRIANGLE
	glm::mat4 modelMatrixForShadow = glm::mat4(1.0f);
	glm::mat4 viewMatrixForShadow = glm::mat4(1.0f);
	glm::mat4 projectionMatrixForShadow = glm::mat4(1.0f);
	glm::mat4 rotationMatrixForShadow = glm::mat4(1.0f);

	viewMatrixForShadow = camera.GetViewMatrix();
	projectionMatrixForShadow = gPerspectiveProjectionMatrix;

	modelMatrixForShadow = glm::translate(glm::mat4(1.0f), glm::vec3(4.0f, 0.0f, -3.0f));

	rotationMatrixForShadow = glm::rotate(glm::mat4(1.0f), degToRad(xAngle), glm::vec3(1.0f, 0.0f, 0.0f));
	modelMatrixForShadow *= rotationMatrixForShadow;
	rotationMatrixForShadow = glm::rotate(glm::mat4(1.0f), degToRad(yAngle), glm::vec3(0.0f, 1.0f, 0.0f));
	modelMatrixForShadow *= rotationMatrixForShadow;
	rotationMatrixForShadow = glm::rotate(glm::mat4(1.0f), degToRad(zAngle), glm::vec3(0.0f, 0.0f, 1.0f));
	modelMatrixForShadow *= rotationMatrixForShadow;

	glUniformMatrix4fv(gModelMatrixUniformForShadow, 1, GL_FALSE, glm::value_ptr(modelMatrixForShadow));
	glUniformMatrix4fv(gViewMatrixUniformForShadow, 1, GL_FALSE, glm::value_ptr(viewMatrixForShadow));
	glUniformMatrix4fv(gProjectionMatrixUniformForShadow, 1, GL_FALSE, glm::value_ptr(projectionMatrixForShadow));
	glUniformMatrix4fv(gLightSpaceMatrixUniformForShadow, 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));

	glUniform3f(gLightPositionUniformForShadow, -2.0f, 4.0f, -1.0f);
	glUniform3f(gViewPositionUniformForShadow, camera.GetCameraPosition().x, camera.GetCameraPosition().y, camera.GetCameraPosition().z);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, depthBufferObject);
	glUniform1i(gShadowMapUniformSamplerForShadow, 0);

	//sphere.DrawSphere();
	//cone.DrawCone();
	cylinder.DrawCylinder();
	//torus.DrawTorus();
	//disk.DrawDisk();

	modelMatrixForShadow = glm::mat4(1.0f);
	rotationMatrixForShadow = glm::mat4(1.0f);

	modelMatrixForShadow = glm::translate(glm::mat4(1.0f), glm::vec3(-4.0f, 0.0f, -3.0f));

	rotationMatrixForShadow = glm::rotate(glm::mat4(1.0f), degToRad(xAngle), glm::vec3(1.0f, 0.0f, 0.0f));
	modelMatrixForShadow *= rotationMatrixForShadow;
	rotationMatrixForShadow = glm::rotate(glm::mat4(1.0f), degToRad(yAngle), glm::vec3(0.0f, 1.0f, 0.0f));
	modelMatrixForShadow *= rotationMatrixForShadow;
	rotationMatrixForShadow = glm::rotate(glm::mat4(1.0f), degToRad(zAngle), glm::vec3(0.0f, 0.0f, 1.0f));
	modelMatrixForShadow *= rotationMatrixForShadow;

	glUniformMatrix4fv(gModelMatrixUniformForShadow, 1, GL_FALSE, glm::value_ptr(modelMatrixForShadow));
	glUniformMatrix4fv(gViewMatrixUniformForShadow, 1, GL_FALSE, glm::value_ptr(viewMatrixForShadow));
	glUniformMatrix4fv(gProjectionMatrixUniformForShadow, 1, GL_FALSE, glm::value_ptr(projectionMatrixForShadow));
	glUniformMatrix4fv(gLightSpaceMatrixUniformForShadow, 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));

	glUniform3f(gLightPositionUniformForShadow, -2.0f, 4.0f, -1.0f);
	glUniform3f(gViewPositionUniformForShadow, camera.GetCameraPosition().x, camera.GetCameraPosition().y, camera.GetCameraPosition().z);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, depthBufferObject);
	glUniform1i(gShadowMapUniformSamplerForShadow, 0);

	//sphere.DrawSphere();
	cone.DrawCone();
	//cylinder.DrawCylinder();
	//torus.DrawTorus();
	//disk.DrawDisk();

	modelMatrixForShadow = glm::mat4(1.0f);

	modelMatrixForShadow = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -5.0f, -3.0f));

	glUniformMatrix4fv(gModelMatrixUniformForShadow, 1, GL_FALSE, glm::value_ptr(modelMatrixForShadow));
	glUniformMatrix4fv(gViewMatrixUniformForShadow, 1, GL_FALSE, glm::value_ptr(viewMatrixForShadow));
	glUniformMatrix4fv(gProjectionMatrixUniformForShadow, 1, GL_FALSE, glm::value_ptr(projectionMatrixForShadow));
	glUniformMatrix4fv(gLightSpaceMatrixUniformForShadow, 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));

	glUniform3f(gLightPositionUniformForShadow, -2.0f, 4.0f, -1.0f);
	glUniform3f(gViewPositionUniformForShadow, camera.GetCameraPosition().x, camera.GetCameraPosition().y, camera.GetCameraPosition().z);

	glBindVertexArray(gVao_ground);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, depthBufferObject);
	glUniform1i(gShadowMapUniformSamplerForShadow, 0);

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);

	glUseProgram(0);

	SwapBuffers(ghdc);
}

//resize
void resize(int width, int height)
{
	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	gPerspectiveProjectionMatrix = glm::perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

GLfloat degToRad(GLfloat degree)
{
	return((GLfloat)(degree * PI / 180.0f));
}

void update(void)
{
	xAngle = xAngle + 0.05f;
	if (xAngle >= 360.0f)
		xAngle = 0.0f;

	yAngle = yAngle + 0.05f;
	if (yAngle >= 360.0f)
		yAngle = 0.0f;

	zAngle = zAngle + 0.05f;
	if (zAngle >= 360.0f)
		zAngle = 0.0f;
}

//uninitialize
void uninitialize(void)
{
	//code
	if (gbFullscreen == true)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);

		ShowCursor(TRUE);
	}

	if (gVao_pyramid)
	{
		glDeleteVertexArrays(1, &gVao_pyramid);
		gVao_pyramid = 0;
	}

	if (gVao_cube)
	{
		glDeleteVertexArrays(1, &gVao_cube);
		gVao_cube = 0;
	}

	if (gVao_ground)
	{
		glDeleteVertexArrays(1, &gVao_ground);
		gVao_ground = 0;
	}

	if (gVao_quad)
	{
		glDeleteVertexArrays(1, &gVao_quad);
		gVao_quad = 0;
	}

	if (gVbo_position)
	{
		glDeleteBuffers(1, &gVbo_position);
		gVbo_position = 0;
	}

	if (gVbo_color)
	{
		glDeleteBuffers(1, &gVbo_color);
		gVbo_color = 0;
	}

	if (gVbo_normal)
	{
		glDeleteBuffers(1, &gVbo_normal);
		gVbo_normal = 0;
	}

	if (gVbo_texcoord)
	{
		glDeleteBuffers(1, &gVbo_texcoord);
		gVbo_texcoord = 0;
	}

	if (depthBufferObject)
	{
		glDeleteRenderbuffers(1, &depthBufferObject);
		depthBufferObject = 0;
	}

	if (gFbo)
	{
		glDeleteFramebuffers(1, &gFbo);
		gFbo = 0;
	}

	//detach shaders first
	glDetachShader(gShaderProgramObjectForDepth, gVertexShaderObjectForDepth);
	glDetachShader(gShaderProgramObjectForDepth, gFragmentShaderObjectForDepth);

	glDetachShader(gShaderProgramObjectForShadow, gVertexShaderObjectForShadow);
	glDetachShader(gShaderProgramObjectForShadow, gFragmentShaderObjectForShadow);
	//delete shaders
	glDeleteShader(gVertexShaderObjectForDepth);
	gVertexShaderObjectForDepth = 0;
	glDeleteShader(gFragmentShaderObjectForDepth);
	gFragmentShaderObjectForDepth = 0;

	glDeleteShader(gVertexShaderObjectForShadow);
	gVertexShaderObjectForShadow = 0;
	glDeleteShader(gFragmentShaderObjectForShadow);
	gFragmentShaderObjectForShadow = 0;

	//delete shader program object
	glDeleteProgram(gShaderProgramObjectForDepth);
	gShaderProgramObjectForDepth = 0;

	glDeleteProgram(gShaderProgramObjectForShadow);
	gShaderProgramObjectForShadow = 0;

	//unlink shader program
	glUseProgram(0);

	wglMakeCurrent(NULL, NULL);

	wglDeleteContext(ghrc);
	ghrc = NULL;

	ReleaseDC(ghwnd, ghdc);
	ghdc = NULL;

	if (gpFile)
	{
		fprintf(gpFile, "Log file closed successfully\n");
		fclose(gpFile);
		gpFile = NULL;
	}
}