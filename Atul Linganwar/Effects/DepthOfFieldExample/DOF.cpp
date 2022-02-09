#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "DOF.h"

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glew32.lib")
#define FBO_SIZE                2048

HDC ghDC = NULL;
HWND ghWnd = NULL;
HGLRC ghRC = NULL;

FILE* gpFile = NULL;

bool gbIsActivate = false;
bool gbIsFullScreen = false;
bool gbIsEscKeyPressed = false;

DWORD dwStyle = 0;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

GLuint guiVAOCube;

GLuint guiVBOPosition;
GLuint guiVBOColor;

GLuint guiMVMatrixUniform;
GLuint guiProjectionMatrixUniform;

GLuint guiRenderVSO;
GLuint guiRenderFSO;
GLuint guiRenderSPO;

GLuint guiDisplayVSO;
GLuint guiDisplayFSO;
GLuint guiDisplaySPO;
GLuint guiTextureSamplerUniform;

GLuint guiFilterCSO;
GLuint guiFilterSPO;

vmath::mat4 gPerspectiveProjectionMatrix;

GLfloat gRotateAnglePyramid;
GLfloat gRotateAngleCube;

GLuint depth_fbo;
GLuint depth_tex;
GLuint color_tex;
GLuint temp_tex;

GLuint quad_vao;

// WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	MSG msg = { 0 };
	HWND hWnd = NULL;
	bool bDone = false;
	WNDCLASSEX WndClass = { 0 };
	WCHAR wszClassName[] = L"3D Rotation";

	fopen_s(&gpFile, "OGLLog.txt", "w");
	if (NULL == gpFile)
	{
		MessageBox(NULL, L"Error while creating log file", L"Error", MB_OK);
		return EXIT_FAILURE;
	}

	// WndClass
	WndClass.cbSize = sizeof(WNDCLASSEX);
	WndClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	WndClass.lpfnWndProc = WndProc;
	WndClass.lpszClassName = wszClassName;
	WndClass.lpszMenuName = NULL;
	WndClass.hInstance = hInstance;
	WndClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	WndClass.hIcon = (HICON)LoadIcon(NULL, IDI_APPLICATION);
	WndClass.hIconSm = (HICON)LoadIcon(NULL, IDI_APPLICATION);
	WndClass.hCursor = (HCURSOR)LoadCursor(NULL, IDC_ARROW);
	WndClass.cbWndExtra = 0;
	WndClass.cbClsExtra = 0;


	// Register WndClass
	if (!RegisterClassEx(&WndClass))
	{
		fprintf(gpFile, "Error while registering WndClass.\n");
		fclose(gpFile);
		gpFile = NULL;
		return EXIT_FAILURE;
	}

	int iScreenWidth = GetSystemMetrics(SM_CXSCREEN);
	int iScreenHeight = GetSystemMetrics(SM_CYSCREEN);

	// Create Window
	hWnd = CreateWindowEx(
		WS_EX_APPWINDOW,
		wszClassName,
		wszClassName,
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		(iScreenWidth / 2) - (OGL_WINDOW_WIDTH / 2),
		(iScreenHeight / 2) - (OGL_WINDOW_HEIGHT / 2),
		OGL_WINDOW_WIDTH,
		OGL_WINDOW_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL
	);
	if (NULL == hWnd)
	{
		fprintf(gpFile, "Error while creating window.\n");
		fclose(gpFile);
		gpFile = NULL;
		return EXIT_FAILURE;
	}

	ghWnd = hWnd;

	ShowWindow(hWnd, iCmdShow);
	SetForegroundWindow(hWnd);
	SetFocus(hWnd);

	bool bRet = Initialize();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while Initialize().\n");
		fclose(gpFile);
		gpFile = NULL;
		DestroyWindow(hWnd);
		hWnd = NULL;
		return EXIT_FAILURE;
	}

	// Game loop
	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (WM_QUIT == msg.message)
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
			Display();
			Update();

			if (true == gbIsActivate)
			{
				if (true == gbIsEscKeyPressed)
				{
					bDone = true;
				}
			}
		}
	}

	// Uninitialize
	UnInitialize();

	return (int)msg.wParam;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (HIWORD(lParam) == 0)
		{
			gbIsActivate = true;
		}
		else
		{
			gbIsActivate = false;
		}
		break;

	case WM_ERASEBKGND:
		return 0;

	case WM_SIZE:
		Resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			if (false == gbIsEscKeyPressed)
			{
				gbIsEscKeyPressed = true;
			}
			break;

		case 0x46:
			if (false == gbIsFullScreen)
			{
				ToggleFullScreen();
				gbIsFullScreen = true;
			}
			else
			{
				ToggleFullScreen();
				gbIsFullScreen = false;
			}
			break;

		default:
			break;
		}
		break;

	case WM_LBUTTONDOWN:
		break;

	case WM_CLOSE:
		UnInitialize();
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	default:
		break;
	}

	return DefWindowProc(hWnd, iMsg, wParam, lParam);
}

bool Initialize()
{
	int iPixelFormatIndex = 0;
	PIXELFORMATDESCRIPTOR pfd = { 0 };

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

	ghDC = GetDC(ghWnd);
	if (NULL == ghDC)
	{
		fprintf(gpFile, "Error while GetDC().\n");
		return false;
	}

	iPixelFormatIndex = ChoosePixelFormat(ghDC, &pfd);
	if (0 == iPixelFormatIndex)
	{
		fprintf(gpFile, "Error while ChoosePixelFormat().\n");
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
		return false;
	}

	if (false == SetPixelFormat(ghDC, iPixelFormatIndex, &pfd))
	{
		fprintf(gpFile, "Error while SetPixelFormat().\n");
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
		return false;
	}

	ghRC = wglCreateContext(ghDC);
	if (NULL == ghRC)
	{
		fprintf(gpFile, "Error while wglCreateContext().\n");
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
		return false;
	}

	if (false == wglMakeCurrent(ghDC, ghRC))
	{
		fprintf(gpFile, "Error while wglMakeCurrent().\n");
		wglDeleteContext(ghRC);
		ghRC = NULL;
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
		return false;
	}

	GLenum glError = glewInit();
	if (GLEW_OK != glError)
	{
		fprintf(gpFile, "Error while glewInit().\n");
		wglDeleteContext(ghRC);
		ghRC = NULL;
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
		return false;
	}

	// Create vertex shader
	guiRenderVSO = glCreateShader(GL_VERTEX_SHADER);

	const GLchar* glchVertexShaderSource =
		"#version 430 core" \
		"\n" \
		"uniform mat4 mv_matrix;" \
		"uniform mat4 proj_matrix;" \
		"in vec4 position;" \
		"in vec3 normal;" \
		"out VS_OUT" \
		"{" \
		"vec3 N;" \
		"vec3 L;" \
		"vec3 V;" \
		"} vs_out;" \
		"uniform vec3 light_pos = vec3(100.0, 100.0, 100.0);" \
		"void main(void)" \
		"{" \
		"vec4 P = mv_matrix * position;" \
		"vs_out.N = mat3(mv_matrix) * normal;" \
		"vs_out.L = light_pos - P.xyz;" \
		"vs_out.V = -P.xyz;" \
		"gl_Position = proj_matrix * P;" \
		"}";


	glShaderSource(guiRenderVSO, 1, (const GLchar**)&glchVertexShaderSource, NULL);

	glCompileShader(guiRenderVSO);
	GLint gliInfoLogLength = 0;
	GLint gliShaderComileStatus = 0;
	char* pszInfoLog = NULL;

	glGetShaderiv(guiRenderVSO, GL_COMPILE_STATUS, &gliShaderComileStatus);
	if (GL_FALSE == gliShaderComileStatus)
	{
		glGetShaderiv(guiRenderVSO, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				GLsizei bytesWritten = 0;
				glGetShaderInfoLog(guiRenderVSO, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "guiRenderVSO Vertex shader compilation Error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitialize();
				return false;
			}
		}
	}

	// Create fragment shader
	guiRenderFSO = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar* glchFragmentShaderSource =
		"#version 430 core\n" \
		"\n" \
		"out vec4 color;\n" \
		"in VS_OUT\n" \
		"{\n" \
		"vec3 N;\n" \
		"vec3 L;\n" \
		"vec3 V;\n" \
		"} fs_in;\n" \
		"uniform vec3 diffuse_albedo = vec3(0.9, 0.8, 1.0);\n" \
		"uniform vec3 specular_albedo = vec3(0.7);\n" \
		"uniform float specular_power = 300.0;\n" \
		"uniform bool full_shading = true;\n" \
		"void main(void)\n" \
		"{\n" \
		"vec3 N = normalize(fs_in.N);\n" \
		"vec3 L = normalize(fs_in.L);\n" \
		"vec3 V = normalize(fs_in.V);\n" \
		"vec3 R = reflect(-L, N);\n" \
		"vec3 diffuse = max(dot(N, L), 0.0) * diffuse_albedo;\n" \
		"vec3 specular = pow(max(dot(R, V), 0.0), specular_power) * specular_albedo;\n" \
		"color = vec4(diffuse + specular, fs_in.V.z);\n" \
		"}\n";


	glShaderSource(guiRenderFSO, 1, (const GLchar**)&glchFragmentShaderSource, NULL);

	glCompileShader(guiRenderFSO);
	gliInfoLogLength = 0;
	gliShaderComileStatus = 0;
	pszInfoLog = NULL;

	glGetShaderiv(guiRenderFSO, GL_COMPILE_STATUS, &gliShaderComileStatus);
	if (GL_FALSE == gliShaderComileStatus)
	{
		glGetShaderiv(guiRenderFSO, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			GLsizei bytesWritten = 0;
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				glGetShaderInfoLog(guiRenderFSO, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "guiRenderFSO Fragment shader compilation error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitialize();
				return false;
			}
		}
	}

	// Create shader program
	guiRenderSPO = glCreateProgram();

	glAttachShader(guiRenderSPO, guiRenderFSO);
	glAttachShader(guiRenderSPO, guiRenderVSO);

	glBindAttribLocation(guiRenderSPO, OGL_ATTRIBUTE_POSITION, "position");
	glBindAttribLocation(guiRenderSPO, OGL_ATTRIBUTE_NORMAL, "normal");

	glLinkProgram(guiRenderSPO);

	GLint gliProgramLinkStatus = 0;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;

	glGetProgramiv(guiRenderSPO, GL_LINK_STATUS, &gliProgramLinkStatus);
	if (GL_FALSE == gliProgramLinkStatus)
	{
		glGetProgramiv(guiRenderSPO, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			GLsizei bytesWritten = 0;
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				glGetProgramInfoLog(guiRenderSPO, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "guiRenderSPO Shader program link error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitialize();
				return false;
			}
		}
	}

	// get mvp uniform location
	guiMVMatrixUniform = glGetUniformLocation(guiRenderSPO, "mv_matrix");
	guiProjectionMatrixUniform = glGetUniformLocation(guiRenderSPO, "proj_matrix");

	const GLfloat cubeVertices[] =
	{
		//TOP FACE
		1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,

		//BOTTOM FACE
		1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f ,

		//FRONT FACE
		1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,

		//BACK FACE
		1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,

		//RIGHT FACE
		1.0f, 1.0f, -1.0f,
		1.0f, 1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f,

		//LEFT FACE
		-1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f
	};

	const GLfloat cubeNormals[] =
	{
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,

		0.0f, -1.0f, 0.0f,
		0.0f, -1.0f, 0.0f,
		0.0f, -1.0f, 0.0f,
		0.0f, -1.0f, 0.0f,

		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,

		0.0f, 0.0f, -1.0f,
		0.0f, 0.0f, -1.0f,
		0.0f, 0.0f, -1.0f,
		0.0f, 0.0f, -1.0f,

		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,

		-1.0f, 0.0f, 0.0f,
		-1.0f, 0.0f, 0.0f,
		-1.0f, 0.0f, 0.0f,
		-1.0f, 0.0f, 0.0f
	};

	// Square
	glGenVertexArrays(1, &guiVAOCube);
	glBindVertexArray(guiVAOCube);

	glGenBuffers(1, &guiVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, guiVBOPosition);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &guiVBOColor);
	glBindBuffer(GL_ARRAY_BUFFER, guiVBOColor);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cubeNormals), cubeNormals, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//
	// Display SPO
	//
	guiDisplayVSO = glCreateShader(GL_VERTEX_SHADER);

	glchVertexShaderSource = NULL;

	glchVertexShaderSource =
		"#version 430 core\n " \
		"\n" \
		"void main(void)\n" \
		"{\n" \
		"const vec4 vertex[] = vec4[] ( vec4(-1.0, -1.0, 0.5, 1.0),\n" \
		"								vec4(1.0, -1.0, 0.5, 1.0),\n" \
		"								vec4(-1.0, 1.0, 0.5, 1.0),\n" \
		"								vec4(1.0, 1.0, 0.5, 1.0) );\n" \
		"gl_Position = vertex[gl_VertexID];\n" \
		"}";

	glShaderSource(guiDisplayVSO, 1, (const GLchar**)&glchVertexShaderSource, NULL);

	glCompileShader(guiDisplayVSO);
	gliInfoLogLength = 0;
	gliShaderComileStatus = 0;
	pszInfoLog = NULL;

	glGetShaderiv(guiDisplayVSO, GL_COMPILE_STATUS, &gliShaderComileStatus);
	if (GL_FALSE == gliShaderComileStatus)
	{
		glGetShaderiv(guiDisplayVSO, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				GLsizei bytesWritten = 0;
				glGetShaderInfoLog(guiDisplayVSO, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "guiDisplayVSO Vertex shader compilation Error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitialize();
				return false;
			}
		}
	}

	// Create fragment shader
	guiDisplayFSO = glCreateShader(GL_FRAGMENT_SHADER);
	glchFragmentShaderSource = NULL;
	glchFragmentShaderSource =
		"#version 430 core\n" \
		"\n" \
		"uniform sampler2D input_image;\n" \
		"out vec4 color;\n" \
		"uniform float focal_distance = 50.0;\n" \
		"uniform float focal_depth = 30.0;\n" \
		"void main(void)\n" \
		"{\n" \
		"vec2 s = 1.0 / textureSize(input_image, 0);\n" \
		"vec2 C = gl_FragCoord.xy;\n" \
		"vec4 v = texelFetch(input_image, ivec2(gl_FragCoord.xy), 0).rgba;\n" \
		/*M will be the radius of filter kernel*/
		"float m;\n" \
		"if(v.w == 0.0)\n" \
		"{\n" \
		"m = 0.5;\n" \
		"}\n" \
		"else\n" \
		"{\n" \
		"m = abs(v.w - focal_distance);\n" \
		"m = 0.5 + smoothstep(0.0, focal_depth, m) * 7.5;\n" \
		"}\n" \
		"vec2 P0 = vec2(C * 1.0) + vec2(-m, -m);\n" \
		"vec2 P1 = vec2(C * 1.0) + vec2(-m, m);\n" \
		"vec2 P2 = vec2(C * 1.0) + vec2(m, -m);\n" \
		"vec2 P3 = vec2(C * 1.0) + vec2(m, m);\n" \
		"P0 *= s;\n" \
		"P1 *= s;\n" \
		"P2 *= s;\n" \
		"P3 *= s;\n" \
		"vec3 a = textureLod(input_image, P0, 0).rgb;\n" \
		"vec3 b = textureLod(input_image, P1, 0).rgb;\n" \
		"vec3 c = textureLod(input_image, P2, 0).rgb;\n" \
		"vec3 d = textureLod(input_image, P3, 0).rgb;\n" \
		"vec3 f = a - b - c + d;\n" \
		"m *= 2;\n" \
		"f /= float(m * m);\n" \
		"color = vec4(f, 1.0);\n" \
		"}";

	glShaderSource(guiDisplayFSO, 1, (const GLchar**)&glchFragmentShaderSource, NULL);

	glCompileShader(guiDisplayFSO);
	gliInfoLogLength = 0;
	gliShaderComileStatus = 0;
	pszInfoLog = NULL;

	glGetShaderiv(guiDisplayFSO, GL_COMPILE_STATUS, &gliShaderComileStatus);
	if (GL_FALSE == gliShaderComileStatus)
	{
		glGetShaderiv(guiDisplayFSO, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			GLsizei bytesWritten = 0;
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				glGetShaderInfoLog(guiDisplayFSO, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "guiDisplayFSO Fragment shader compilation error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitialize();
				return false;
			}
		}
	}

	// Create shader program
	guiDisplaySPO = glCreateProgram();

	glAttachShader(guiDisplaySPO, guiDisplayVSO);
	glAttachShader(guiDisplaySPO, guiDisplayFSO);

	glLinkProgram(guiDisplaySPO);

	gliProgramLinkStatus = 0;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;

	glGetProgramiv(guiDisplaySPO, GL_LINK_STATUS, &gliProgramLinkStatus);
	if (GL_FALSE == gliProgramLinkStatus)
	{
		glGetProgramiv(guiDisplaySPO, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			GLsizei bytesWritten = 0;
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				glGetProgramInfoLog(guiDisplaySPO, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "guiDisplaySPO Shader program link error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitialize();
				return false;
			}
		}
	}

	// get mvp uniform location
	guiTextureSamplerUniform = glGetUniformLocation(guiDisplaySPO, "input_image");

	//
	// Comppute Shader
	//
	guiFilterCSO = glCreateShader(GL_COMPUTE_SHADER);
	const GLchar* glchComputeShaderSource =
		"#version 430 core\n" \
		"\n" \
		"layout (local_size_x = 1024) in;\n" \
		"shared vec3 shared_data[gl_WorkGroupSize.x * 2];\n" \
		"layout (binding = 0, rgba32f) readonly uniform image2D input_image;\n" \
		"layout (binding = 1, rgba32f) writeonly uniform image2D output_image;\n" \
		"void main(void)\n" \
		"{\n" \
		"uint id = gl_LocalInvocationID.x;\n" \
		"uint rd_id;\n" \
		"uint wr_id;\n" \
		"uint mask;\n" \
		"ivec2 P0 = ivec2(id * 2, gl_WorkGroupID.x);\n" \
		"ivec2 P1 = ivec2(id * 2 + 1, gl_WorkGroupID.x);\n" \
		"const uint steps = uint(log2(gl_WorkGroupSize.x)) + 1;\n" \
		"uint step = 0;\n" \
		"vec4 i0 = imageLoad(input_image, P0);\n" \
		"vec4 i1 = imageLoad(input_image, P1);\n" \
		"shared_data[P0.x] = i0.rgb;\n" \
		"shared_data[P1.x] = i1.rgb;\n" \
		"barrier();\n" \
		"for(step = 0; step < steps; step++)\n" \
		"{\n" \
		"mask = (1 << step) - 1;\n" \
		"rd_id = ((id >> step) << (step + 1)) + mask;\n" \
		"wr_id = rd_id + 1 + (id & mask);\n" \
		"shared_data[wr_id] += shared_data[rd_id];\n" \
		"barrier();\n" \
		"}\n" \
		"imageStore(output_image, P0.yx, vec4(shared_data[P0.x], i0.a));\n" \
		"imageStore(output_image, P1.yx, vec4(shared_data[P1.x], i1.a));\n" \
		"}\n";

	glShaderSource(guiFilterCSO, 1, (const GLchar**)&glchComputeShaderSource, NULL);

	glCompileShader(guiFilterCSO);
	gliInfoLogLength = 0;
	gliShaderComileStatus = 0;
	pszInfoLog = NULL;

	glGetShaderiv(guiFilterCSO, GL_COMPILE_STATUS, &gliShaderComileStatus);
	if (GL_FALSE == gliShaderComileStatus)
	{
		glGetShaderiv(guiFilterCSO, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			GLsizei bytesWritten = 0;
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				glGetShaderInfoLog(guiFilterCSO, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "guiFilterCSO Fragment shader compilation error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitialize();
				return false;
			}
		}
	}

	// Create shader program
	guiFilterSPO = glCreateProgram();

	glAttachShader(guiFilterSPO, guiFilterCSO);

	glLinkProgram(guiFilterSPO);

	gliProgramLinkStatus = 0;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;

	glGetProgramiv(guiFilterSPO, GL_LINK_STATUS, &gliProgramLinkStatus);
	if (GL_FALSE == gliProgramLinkStatus)
	{
		glGetProgramiv(guiFilterSPO, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			GLsizei bytesWritten = 0;
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				glGetProgramInfoLog(guiFilterSPO, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "guiFilterSPO Shader program link error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitialize();
				return false;
			}
		}
	}

	//
	// FB0
	//
	glGenFramebuffers(1, &depth_fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, depth_fbo);

	glGenTextures(1, &depth_tex);
	glBindTexture(GL_TEXTURE_2D, depth_tex);
	glTexStorage2D(GL_TEXTURE_2D, 11, GL_DEPTH_COMPONENT32F, FBO_SIZE, FBO_SIZE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glGenTextures(1, &color_tex);
	glBindTexture(GL_TEXTURE_2D, color_tex);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, FBO_SIZE, FBO_SIZE);

	glGenTextures(1, &temp_tex);
	glBindTexture(GL_TEXTURE_2D, temp_tex);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, FBO_SIZE, FBO_SIZE);

	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depth_tex, 0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color_tex, 0);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glGenVertexArrays(1, &quad_vao);
	glBindVertexArray(quad_vao);

	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glEnable(GL_TEXTURE0);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix = vmath::mat4::identity();

	Resize(OGL_WINDOW_WIDTH, OGL_WINDOW_HEIGHT);

	return true;
}

void ToggleFullScreen()
{
	MONITORINFO mi = { 0 };

	if (false == gbIsFullScreen)
	{
		dwStyle = GetWindowLong(ghWnd, GWL_STYLE);
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			mi.cbSize = sizeof(MONITORINFO);
			if (
				GetWindowPlacement(ghWnd, &wpPrev) &&
				GetMonitorInfo(MonitorFromWindow(ghWnd, MONITORINFOF_PRIMARY), &mi)
				)
			{
				SetWindowLong(ghWnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(
					ghWnd,
					HWND_TOP,
					mi.rcMonitor.left,
					mi.rcMonitor.top,
					mi.rcMonitor.right - mi.rcMonitor.left,
					mi.rcMonitor.bottom - mi.rcMonitor.top,
					SWP_NOZORDER | SWP_FRAMECHANGED
				);
			}
		}
		ShowCursor(FALSE);
	}
	else
	{
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghWnd, &wpPrev);
		SetWindowPos(
			ghWnd,
			HWND_TOP,
			0, 0, 0, 0,
			SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED
		);

		ShowCursor(TRUE);
	}
}

void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	glBindFramebuffer(GL_FRAMEBUFFER, depth_fbo);
	glDrawBuffer(GL_COLOR_ATTACHMENT0);
	glViewport(0, 0, OGL_WINDOW_WIDTH, OGL_WINDOW_HEIGHT);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(guiRenderSPO);

	vmath::mat4 rotationMatrix = vmath::mat4::identity();
	vmath::mat4 modelViewMatrix = vmath::mat4::identity();

	modelViewMatrix = vmath::translate(0.0f, 0.0f, -10.0f);

	rotationMatrix = vmath::mat4::identity();

	rotationMatrix = vmath::rotate(gRotateAngleCube, gRotateAngleCube, gRotateAngleCube);

	modelViewMatrix = modelViewMatrix * rotationMatrix;

	glUniformMatrix4fv(guiMVMatrixUniform, 1, GL_FALSE, modelViewMatrix);
	glUniformMatrix4fv(guiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	glBindVertexArray(guiVAOCube);

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 20, 4);

	glBindVertexArray(0);
	glUseProgram(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0, 0, OGL_WINDOW_WIDTH, OGL_WINDOW_HEIGHT);

	// COMPUTE
	glUseProgram(guiFilterSPO);

	glBindImageTexture(0, color_tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
	glBindImageTexture(1, temp_tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

	glDispatchCompute(OGL_WINDOW_HEIGHT, 1, 1);

	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	glBindImageTexture(0, temp_tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
	glBindImageTexture(1, color_tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

	glDispatchCompute(OGL_WINDOW_WIDTH, 1, 1);

	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	glUseProgram(guiDisplaySPO);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, color_tex);
	glUniform1i(guiTextureSamplerUniform, 0);

	glDisable(GL_DEPTH_TEST);

	glBindVertexArray(quad_vao);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glUseProgram(0);

	SwapBuffers(ghDC);
}

void Update(void)
{
	gRotateAnglePyramid = gRotateAnglePyramid + 0.2f;
	if (gRotateAnglePyramid > 360.0f)
	{
		gRotateAnglePyramid = 0.0f;
	}

	gRotateAngleCube = gRotateAngleCube + 0.2f;
	if (gRotateAngleCube > 360.0f)
	{
		gRotateAngleCube = 0.0f;
	}
}

void Resize(int iWidth, int iHeight)
{
	if (iHeight == 0)
	{
		iHeight = 1;
	}

	glViewport(0, 0, (GLsizei)iWidth, (GLsizei)iHeight);

	gPerspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)iWidth / (GLfloat)iHeight, 0.1f, 100.0f);
}

void UnInitialize()
{
	if (true == gbIsFullScreen)
	{
		dwStyle = GetWindowLong(ghWnd, GWL_STYLE);
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghWnd, &wpPrev);
		SetWindowPos(ghWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED | SWP_NOMOVE);

		ShowCursor(TRUE);
	}

	if (0 != guiVBOPosition)
	{
		glDeleteBuffers(1, &guiVBOPosition);
		guiVBOPosition = 0;
	}

	if (0 != guiVBOColor)
	{
		glDeleteBuffers(1, &guiVBOColor);
		guiVBOColor = 0;
	}

	if (0 != guiVAOCube)
	{
		glDeleteVertexArrays(1, &guiVAOCube);
		guiVAOCube = 0;
	}

	glDetachShader(guiRenderSPO, guiRenderFSO);
	glDetachShader(guiRenderSPO, guiRenderVSO);

	glDeleteShader(guiRenderFSO);
	guiRenderFSO = 0;

	glDeleteShader(guiRenderVSO);
	guiRenderVSO = 0;

	glDeleteProgram(guiRenderSPO);
	guiRenderSPO = 0;

	glUseProgram(0);

	wglMakeCurrent(NULL, NULL);

	wglDeleteContext(ghRC);
	ghRC = NULL;

	ReleaseDC(ghWnd, ghDC);
	ghDC = NULL;

	if (NULL != gpFile)
	{
		fprintf(gpFile, "Exitting successfully.\n");
		fclose(gpFile);
		gpFile = NULL;
	}

	return;
}