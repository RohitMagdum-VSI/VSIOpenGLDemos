#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"
#include "../common/Shapes.h"

#include "3DNoiseExample.h"
#include "Sphere.h"

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "GeometricShapes.lib")

#define GL_CHECK(x) \
    x; \
    { \
        GLenum glError = glGetError(); \
        if(glError != GL_NO_ERROR) { \
            fprintf(gpFile, "glGetError() = %i (0x%.8x) at %s:%i\n", glError, glError, __FILE__, __LINE__); \
            exit(1); \
        } \
    }


#define MAXB 0x100
#define N 0x1000
#define NP 12   // 2^N
#define NM 0xfff

#define s_curve(t) ( t * t * (3. - 2. * t) )
#define lerp(t, a, b) ( a + t * (b - a) )
#define setup(i, b0, b1, r0, r1)\
        t = vec[i] + N;\
        b0 = ((int)t) & BM;\
        b1 = (b0+1) & BM;\
        r0 = t - (int)t;\
        r1 = r0 - 1.;
#define at2(rx, ry) ( rx * q[0] + ry * q[1] )
#define at3(rx, ry, rz) ( rx * q[0] + ry * q[1] + rz * q[2] )

HDC ghDC = NULL;
HWND ghWnd = NULL;
HGLRC ghRC = NULL;

FILE* gpFile = NULL;

bool gbIsActivate = false;
bool gbIsFullScreen = false;
bool gbIsEscKeyPressed = false;

DWORD dwStyle = 0;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

GLuint guiVAO;

GLuint guiVBOPosition;
GLuint guiVBOTexture;
GLuint guiVBONormal;

GLuint guiMVMatrixUniform;
GLuint guiMVPMatrixUniform;
GLuint guiOffsetUniform;
GLuint guiProjectionMatrixUniform;
GLuint guiVeinColorUniform;
GLuint guiMarbalColorUniform;
GLuint guiTextureSamplerUniform;
GLuint guiScaleUniform;
GLuint guiNoiseScaleUniform;
GLuint guiLightIntensityUniform;
GLuint guiSineFactoryUniform;
GLuint guiSineIntensityFactorUniform;

GLfloat gfScale = 1.0f;
GLfloat gfSineFactor = 6.0f;
GLfloat gfNoiseScale = 1.0f;
GLfloat gfLightIntensity = 1.2f;
GLfloat gfSineIntensityFactor = 12.0f;

GLuint guiTextureSmiley;

GLuint guiVertexShaderObject;
GLuint guiFragmentShaderObject;
GLuint guiShaderProgramObject;

GLfloat VeinColor[] = { 0.0f, 0.0f, 0.8f };
GLfloat MarbalColor[] = { 0.8f, 0.8f, 0.8f };

GLfloat gfAngle = 0.0;

vmath::mat4 gPerspectiveProjectionMatrix;

SPHERE cSphere;

int noise3DTexSize = 128;
GLuint noise3DTexName = 0;
GLubyte* noise3DTexPtr;

static int p[MAXB + MAXB + 2];
static double g3[MAXB + MAXB + 2][3];
static double g2[MAXB + MAXB + 2][2];
static double g1[MAXB + MAXB + 2];

bool gbIncrease = true;

int start;
int B;
int BM;

float fOffset = 0.0f;

void SetNoiseFrequency(int frequency)
{
	start = 1;
	B = frequency;
	BM = B - 1;
}

void normalize2(double v[2])
{
	double s;

	s = sqrt(v[0] * v[0] + v[1] * v[1]);
	v[0] = v[0] / s;
	v[1] = v[1] / s;
}

void normalize3(double v[3])
{
	double s;

	s = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	v[0] = v[0] / s;
	v[1] = v[1] / s;
	v[2] = v[2] / s;
}

void initNoise()
{
	int i, j, k;

	srand(30757);
	for (i = 0; i < B; i++)
	{
		p[i] = i;
		g1[i] = (double)((rand() % (B + B)) - B) / B;

		for (j = 0; j < 2; j++)
			g2[i][j] = (double)((rand() % (B + B)) - B) / B;
		normalize2(g2[i]);

		for (j = 0; j < 3; j++)
			g3[i][j] = (double)((rand() % (B + B)) - B) / B;
		normalize3(g3[i]);
	}

	while (--i)
	{
		k = p[i];
		p[i] = p[j = rand() % B];
		p[j] = k;
	}

	for (i = 0; i < B + 2; i++)
	{
		p[B + i] = p[i];
		g1[B + i] = g1[i];
		for (j = 0; j < 2; j++)
			g2[B + i][j] = g2[i][j];
		for (j = 0; j < 3; j++)
			g3[B + i][j] = g3[i][j];
	}
}

double noise2(double vec[2])
{
	int bx0, bx1, by0, by1, b00, b10, b01, b11;
	double rx0, rx1, ry0, ry1, * q, sx, sy, a, b, t, u, v;
	int i, j;

	if (start)
	{
		start = 0;
		initNoise();
	}

	setup(0, bx0, bx1, rx0, rx1);
	setup(1, by0, by1, ry0, ry1);

	i = p[bx0];
	j = p[bx1];

	b00 = p[i + by0];
	b10 = p[j + by0];
	b01 = p[i + by1];
	b11 = p[j + by1];

	sx = s_curve(rx0);
	sy = s_curve(ry0);

	q = g2[b00]; u = at2(rx0, ry0);
	q = g2[b10]; v = at2(rx1, ry0);
	a = lerp(sx, u, v);

	q = g2[b01]; u = at2(rx0, ry1);
	q = g2[b11]; v = at2(rx1, ry1);
	b = lerp(sx, u, v);

	return lerp(sy, a, b);
}

double noise3(double vec[3])
{
	int bx0, bx1, by0, by1, bz0, bz1, b00, b10, b01, b11;
	double rx0, rx1, ry0, ry1, rz0, rz1, * q, sy, sz, a, b, c, d, t, u, v;
	int i, j;

	if (start)
	{
		start = 0;
		initNoise();
	}

	setup(0, bx0, bx1, rx0, rx1);
	setup(1, by0, by1, ry0, ry1);
	setup(2, bz0, bz1, rz0, rz1);

	i = p[bx0];
	j = p[bx1];

	b00 = p[i + by0];
	b10 = p[j + by0];
	b01 = p[i + by1];
	b11 = p[j + by1];

	t = s_curve(rx0);
	sy = s_curve(ry0);
	sz = s_curve(rz0);

	q = g3[b00 + bz0]; u = at3(rx0, ry0, rz0);
	q = g3[b10 + bz0]; v = at3(rx1, ry0, rz0);
	a = lerp(t, u, v);

	q = g3[b01 + bz0]; u = at3(rx0, ry1, rz0);
	q = g3[b11 + bz0]; v = at3(rx1, ry1, rz0);
	b = lerp(t, u, v);

	c = lerp(sy, a, b);

	q = g3[b00 + bz1]; u = at3(rx0, ry0, rz1);
	q = g3[b10 + bz1]; v = at3(rx1, ry0, rz1);
	a = lerp(t, u, v);

	q = g3[b01 + bz1]; u = at3(rx0, ry1, rz1);
	q = g3[b11 + bz1]; v = at3(rx1, ry1, rz1);
	b = lerp(t, u, v);

	d = lerp(sy, a, b);

	return lerp(sz, c, d);
}

// WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	MSG msg = { 0 };
	HWND hWnd = NULL;
	bool bDone = false;
	WNDCLASSEX WndClass = { 0 };
	WCHAR wszClassName[] = L"2D Texture";

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

		case 0x53: // S
			if (GetAsyncKeyState(VK_SHIFT))
				gfScale -= 0.1f;
			else
				gfScale += 0.1f;
			break;

		case 0x4E: // N
			if (GetAsyncKeyState(VK_SHIFT))
				gfNoiseScale -= 0.1f;
			else
				gfNoiseScale += 0.1f;
			break;

		case 0x4C: // L
			if (GetAsyncKeyState(VK_SHIFT))
				gfLightIntensity -= 0.01f;
			else
				gfLightIntensity += 0.01f;
			break;

		case 0x4A: // J
			if (GetAsyncKeyState(VK_SHIFT))
				gfSineFactor -= 0.1f;
			else
				gfSineFactor += 0.1f;
			break;

		case 0x4B: // K
			if (GetAsyncKeyState(VK_SHIFT))
				gfSineIntensityFactor -= 0.1f;
			else
				gfSineIntensityFactor += 0.1f;
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
	guiVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar* glchVertexShaderSource =
		"#version 430 core" \
		"\n" \
		"uniform mat4 MVMatrix;\n" \
		"uniform mat4 MVPMatrix;\n" \
		"in vec4 MCvertex;\n" \
		"in vec3 MCnormal;\n" \
		"out float LightIntensity;\n" \
		"out vec3 MCposition;\n" \
		"void main(void)\n" \
		"{\n" \
		"vec3 ECposition = vec3(MVMatrix * MCvertex);\n" \
		"MCposition = vec3(MCvertex) * 1.2;\n" \
		"LightIntensity = 1.5;\n" \
		"gl_Position = MVPMatrix * MCvertex;\n" \
		"}\n";

	/*const GLchar* glchVertexShaderSource =
		"#version 430 core" \
		"\n" \
		"uniform mat4 MVMatrix;\n" \
		"uniform mat4 ProjectionMatrix;\n" \
		"uniform float Scale;\n" \
		"in vec4 MCvertex;\n" \
		"in vec3 MCnormal;\n" \
		"in vec2 MCtexcoord;\n" \
		"out vec3 MCposition;\n" \
		"void main(void)\n" \
		"{\n" \
		"MCposition = vec3(MCvertex) * Scale;\n" \
		"gl_Position = ProjectionMatrix * MVMatrix * MCvertex;\n" \
		"}\n";*/

	glShaderSource(guiVertexShaderObject, 1, (const GLchar**)&glchVertexShaderSource, NULL);

	glCompileShader(guiVertexShaderObject);
	GLint gliInfoLogLength = 0;
	GLint gliShaderComileStatus = 0;
	char* pszInfoLog = NULL;

	glGetShaderiv(guiVertexShaderObject, GL_COMPILE_STATUS, &gliShaderComileStatus);
	if (GL_FALSE == gliShaderComileStatus)
	{
		glGetShaderiv(guiVertexShaderObject, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				GLsizei bytesWritten = 0;
				glGetShaderInfoLog(guiVertexShaderObject, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "Vertex shader compilation Error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitialize();
				return false;
			}
		}
	}

	// Create fragment shader
	guiFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	//
	// Example
	//
	const GLchar* glchFragmentShaderSource =
		"#version 430 core" \
		"\n" \
		"uniform sampler3D Noise;\n" \
		"uniform vec3 SkyColor = vec3(1.0, 0.0, 0.8);\n" \
		"uniform vec3 CloudColor = vec3(0.8, 0.8, 0.8);\n" \
		"uniform float offset;\n" \
		"in vec3 MCposition;\n" \
		"out vec4 FragColor;\n" \
		"void main(void)\n" \
		"{\n" \
		"vec4 noisevec = texture(Noise, (MCposition) + offset);\n" \
		"float intensity = (noisevec[0] + noisevec[1] + noisevec[2] + noisevec[3] + 0.03125) * 1.5;\n" \
		"vec3 color = mix(SkyColor, CloudColor, intensity) * 1.2;\n" \
		"FragColor = vec4(1.0);\n" \
		"}\n";

	//
	// Sun
	//
	/*const GLchar* glchFragmentShaderSource =
		"#version 430 core" \
		"\n" \
		"uniform sampler3D Noise;\n" \
		"uniform vec3 SkyColor = vec3(0.8, 0.7, 0.0);\n" \
		"uniform vec3 CloudColor = vec3(0.6, 0.1, 0.0);\n" \
		"uniform float Scale = 1.2;\n" \
		"in vec3 MCposition;\n" \
		"out vec4 FragColor;\n" \
		"void main(void)\n" \
		"{\n" \
		"vec4 noisevec = texture(Noise, (Scale * MCposition));\n" \
		"float intensity = abs(noisevec[0] - 0.25) +\n" \
		"					abs(noisevec[1] - 0.125) + \n" \
		"					abs(noisevec[2] - 0.0625) + \n" \
		"					abs(noisevec[3] - 0.03125);\n" \
		"intensity = clamp(intensity * 6.0, 0.0, 1.0);\n" \
		"vec3 color = mix(SkyColor, CloudColor, intensity) * 1.2;\n" \
		"FragColor = vec4(color, 1.0);\n" \
		"}\n";*/

	//
	// Marble
	//
	//const GLchar* glchFragmentShaderSource =
	//	"#version 430 core" \
	//	"\n" \
	//	"uniform sampler3D Noise;\n" \
	//	"uniform vec3 VeinColor;\n" \
	//	"uniform vec3 MarbalColor;\n" \
	//	"uniform float NoiseScale;\n" \
	//	"uniform float LightIntensity;\n" \
	//	"uniform float SineFactor;\n" \
	//	"uniform float SineIntensityFactor;\n" \
	//	"in vec3 MCposition;\n" \
	//	"out vec4 FragColor;\n" \
	//	"void main(void)\n" \
	//	"{\n" \
	//	"vec4 noisevec = texture(Noise, (NoiseScale * MCposition));\n" \
	//	"float intensity = abs(noisevec[0] - 0.25) +\n" \
	//	"					abs(noisevec[1] - 0.125) + \n" \
	//	"					abs(noisevec[2] - 0.0625) + \n" \
	//	"					abs(noisevec[3] - 0.03125);\n" \
	//	"float sineval = sin(MCposition.y * SineFactor + intensity * SineIntensityFactor) * 0.5 + 0.5;\n" \
	//	"vec3 color = mix(VeinColor, MarbalColor, sineval * LightIntensity);\n" \
	//	"FragColor = vec4(color, 1.0);\n" \
	//	"}\n";
	
	//
	// Granite
	//
	/*const GLchar* glchFragmentShaderSource =
		"#version 430 core" \
		"\n" \
		"uniform sampler3D Noise;\n" \
		"uniform vec3 SkyColor = vec3(0.8, 0.7, 0.0);\n" \
		"uniform vec3 CloudColor = vec3(0.6, 0.1, 0.0);\n" \
		"uniform float Scale = 1.2;\n" \
		"uniform float offset;\n" \
		"in vec3 MCposition;\n" \
		"out vec4 FragColor;\n" \
		"void main(void)\n" \
		"{\n" \
		"vec4 noisevec = texture(Noise, (Scale * MCposition));\n" \
		"float intensity = min(1.0, noisevec[3] * 18.0);\n" \
		"float sineval = sin(MCposition.y * 6.0 + intensity * 12.0) * 0.5 + 0.5;\n" \
		"vec3 color = vec3(intensity * 1.2);\n" \
		"FragColor = vec4(color, 1.0);\n" \
		"}\n";*/

	////
	//// Wood
	////
	/*const GLchar * glchFragmentShaderSource = 
		"#version 430 core" \
		"\n" \
		"uniform sampler3D Noise;\n" \
		"uniform vec3 LightWood = vec3(0.9, 0.9, 0.6);\n" \
		"uniform vec3 DarkWood = vec3(0.8, 0.8, 0.5);\n" \
		"uniform float RingFreq = 4.0;\n" \
		"uniform float LightGrains = 0.5;\n" \
		"uniform float DarkGrains = 0.2;\n" \
		"uniform float GrainThreshold = 0.6;\n" \
		"uniform vec3 NoiseScale = vec3(0.2, 0.2, 0.2);\n" \
		"uniform float Noisiness = 0.1;\n" \
		"uniform float GrainScale = 2.0;\n" \
		"float LightIntensity = 1.2;\n" \
		"in vec3 MCposition;\n" \
		"out vec4 FragColor;\n" \
		"void main(void)\n" \
		"{\n" \
		"vec3 noisevec = vec3(texture(Noise, MCposition * NoiseScale) * Noisiness);" \
		"vec3 location = MCposition + noisevec;\n" \
		"float dist = sqrt(location.x * location.x + location.z * location.z);\n" \
		"dist *= RingFreq;\n" \
		"float r = fract(dist + noisevec[0] + noisevec[1] + noisevec[2]) * 2.0;\n" \
		"if(r > 1.0)\n" \
		"{\n" \
		"r = 2.0 - r;\n" \
		"}\n" \
		"vec3 color = mix(LightWood, DarkWood, r);\n" \
		"r = fract((MCposition.x + MCposition.z) * GrainScale + 0.5);\n" \
		"noisevec[2] *= r;\n" \
		"if(r < GrainThreshold)\n" \
		"{\n" \
		"color += LightWood * LightGrains * noisevec[2];\n" \
		"}\n" \
		"else\n" \
		"{\n" \
		"color -= LightWood * DarkGrains * noisevec[2];\n" \
		"}\n" \
		"color *= LightIntensity;\n" \
		"FragColor = vec4(color, 1.0);\n" \
		"}\n";*/

	glShaderSource(guiFragmentShaderObject, 1, (const GLchar**)&glchFragmentShaderSource, NULL);

	glCompileShader(guiFragmentShaderObject);
	gliInfoLogLength = 0;
	gliShaderComileStatus = 0;
	pszInfoLog = NULL;

	glGetShaderiv(guiFragmentShaderObject, GL_COMPILE_STATUS, &gliShaderComileStatus);
	if (GL_FALSE == gliShaderComileStatus)
	{
		glGetShaderiv(guiFragmentShaderObject, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			GLsizei bytesWritten = 0;
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				glGetShaderInfoLog(guiFragmentShaderObject, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "Fragment shader compilation error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitialize();
				return false;
			}
		}
	}

	// Create shader program
	guiShaderProgramObject = glCreateProgram();

	glAttachShader(guiShaderProgramObject, guiVertexShaderObject);
	glAttachShader(guiShaderProgramObject, guiFragmentShaderObject);

	glBindAttribLocation(guiShaderProgramObject, OGL_ATTRIBUTE_POSITION, "MCvertex");
	glBindAttribLocation(guiShaderProgramObject, OGL_ATTRIBUTE_TEXTURE0, "MCtexcoord");
	glBindAttribLocation(guiShaderProgramObject, OGL_ATTRIBUTE_NORMAL, "MCnormal");

	glLinkProgram(guiShaderProgramObject);

	GLint gliProgramLinkStatus = 0;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;

	glGetProgramiv(guiShaderProgramObject, GL_LINK_STATUS, &gliProgramLinkStatus);
	if (GL_FALSE == gliProgramLinkStatus)
	{
		glGetProgramiv(guiShaderProgramObject, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			GLsizei bytesWritten = 0;
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				glGetProgramInfoLog(guiShaderProgramObject, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "Shader program link error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitialize();
				return false;
			}
		}
	}

	// get mvp uniform location
	guiMVMatrixUniform = glGetUniformLocation(guiShaderProgramObject, "MVMatrix");
	guiMVPMatrixUniform = glGetUniformLocation(guiShaderProgramObject, "MVPMatrix");
	guiOffsetUniform = glGetUniformLocation(guiShaderProgramObject, "offset");

	guiVeinColorUniform = glGetUniformLocation(guiShaderProgramObject, "SkyColor");
	guiMarbalColorUniform = glGetUniformLocation(guiShaderProgramObject, "CloudColor");

	/*guiProjectionMatrixUniform = glGetUniformLocation(guiShaderProgramObject, "ProjectionMatrix");
	guiTextureSamplerUniform = glGetUniformLocation(guiShaderProgramObject, "Noise");
	guiVeinColorUniform = GL_CHECK(glGetUniformLocation(guiShaderProgramObject, "VeinColor"));
	guiMarbalColorUniform = glGetUniformLocation(guiShaderProgramObject, "MarbalColor");
	guiScaleUniform = glGetUniformLocation(guiShaderProgramObject, "Scale");
	guiNoiseScaleUniform = glGetUniformLocation(guiShaderProgramObject, "NoiseScale");
	guiLightIntensityUniform = glGetUniformLocation(guiShaderProgramObject, "LightIntensity");
	guiSineFactoryUniform = glGetUniformLocation(guiShaderProgramObject, "SineFactor");
	guiSineIntensityFactorUniform = glGetUniformLocation(guiShaderProgramObject, "SineIntensityFactor");*/

	const GLfloat squareVertices[] =
	{
		1.0f, 1.0f, 0.0f,
		-1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f
	};

	const GLfloat squareNormals[] =
	{
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f
	};

	const GLfloat squareTexcoord[] =
	{
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
	};

	glGenVertexArrays(1, &guiVAO);
	glBindVertexArray(guiVAO);

	glGenBuffers(1, &guiVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, guiVBOPosition);
	glBufferData(GL_ARRAY_BUFFER, sizeof(squareVertices), squareVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &guiVBONormal);
	glBindBuffer(GL_ARRAY_BUFFER, guiVBONormal);
	glBufferData(GL_ARRAY_BUFFER, sizeof(squareNormals), squareNormals, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &guiVBOTexture);
	glBindBuffer(GL_ARRAY_BUFFER, guiVBOTexture);
	glBufferData(GL_ARRAY_BUFFER, sizeof(squareTexcoord), squareTexcoord, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	make3DNoiseTexture();

	glGenTextures(1, &noise3DTexName);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, noise3DTexName);
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, noise3DTexSize, noise3DTexSize, noise3DTexSize, 0, GL_RGBA, GL_UNSIGNED_BYTE, noise3DTexPtr);
	glBindTexture(GL_TEXTURE_3D, 0);

	if (FALSE == InitializeSphere(1.0f, 30, 30, &cSphere))
	{
		fprintf(gpFile, "Error while InitializeSphere().\n");
		UnInitialize();
		return false;
	}

	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glEnable(GL_CULL_FACE);
	glEnable(GL_TEXTURE_2D);

	LoadGLTexture(&guiTextureSmiley, MAKEINTRESOURCE(IDBITMAP_SMILEY));

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

	glUseProgram(guiShaderProgramObject);

	vmath::mat4 modelViewMatrix = vmath::mat4::identity();

	modelViewMatrix = vmath::translate(0.0f, 0.0f, -5.0f);

	if (fOffset <= 0.85 && gbIncrease == true)
	{
		fOffset += 0.0005f;
		if (fOffset == 0.85)
		{
			gbIncrease = false;
		}
	}
	else
	{
		gbIncrease = false;
		fOffset -= 0.0005f;
		if (fOffset <= 0.65)
		{
			gbIncrease = true;
		}
	}
	
	glUniform1f(guiOffsetUniform, fOffset);

	glUniformMatrix4fv(guiMVMatrixUniform, 1, GL_FALSE, modelViewMatrix);
	glUniformMatrix4fv(guiProjectionMatrixUniform, 1,GL_FALSE, gPerspectiveProjectionMatrix);

	//
	// Texture
	//
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, noise3DTexName);
	glUniform1i(guiTextureSamplerUniform, 0);

	//DrawSphere(&cSphere);
	glBindVertexArray(guiVAO);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);

	glUseProgram(0);

	SwapBuffers(ghDC);
}

void Update(void)
{
	gfAngle = gfAngle + 0.2f;
	if (gfAngle > 360.0f)
	{
		gfAngle = 0.0f;
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

int LoadGLTexture(GLuint* texture, TCHAR imageResourceId[])
{
	//local variables
	HBITMAP hBitmap;
	BITMAP bmp;
	int iStatus = FALSE;

	//code
	glGenTextures(1, texture);
	hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL), imageResourceId, IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);
	if (hBitmap)
	{
		iStatus = TRUE;
		GetObject(hBitmap, sizeof(bmp), &bmp);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glBindTexture(GL_TEXTURE_2D, *texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp.bmWidth, bmp.bmHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, bmp.bmBits);

		glGenerateMipmap(GL_TEXTURE_2D);

		DeleteObject(hBitmap);
	}
	return(iStatus);
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

	FreeSphere(&cSphere);

	if (0 != guiVBOPosition)
	{
		glDeleteBuffers(1, &guiVBOPosition);
		guiVBOPosition = 0;
	}

	if (0 != guiVBOTexture)
	{
		glDeleteBuffers(1, &guiVBOTexture);
		guiVBOTexture = 0;
	}

	if (0 != guiVAO)
	{
		glDeleteVertexArrays(1, &guiVAO);
		guiVAO = 0;
	}

	glDetachShader(guiShaderProgramObject, guiFragmentShaderObject);
	glDetachShader(guiShaderProgramObject, guiVertexShaderObject);

	glDeleteShader(guiFragmentShaderObject);
	guiFragmentShaderObject = 0;

	glDeleteShader(guiVertexShaderObject);
	guiVertexShaderObject = 0;

	glDeleteProgram(guiShaderProgramObject);
	guiShaderProgramObject = 0;

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

void make3DNoiseTexture(void)
{
	int f, i, j, k, inc;
	int startFrequency = 4;
	int numOctaves = 4;
	double ni[3];
	double inci, incj, inck;
	int frequency = startFrequency;
	GLubyte* ptr;
	double amp = 0.5;

	noise3DTexPtr = (GLubyte*)malloc(noise3DTexSize * noise3DTexSize * noise3DTexSize * 4);

	if (NULL == noise3DTexPtr)
	{
		fprintf(gpFile, "Error while malloc 3d noise texture\n");
		return;
	}

	for (f = 0, inc = 0; f < numOctaves; ++f, frequency *= 2, ++inc, amp *= 0.5)
	{
		SetNoiseFrequency(frequency);
		ptr = noise3DTexPtr;
		ni[0] = ni[1] = ni[2] = 0;

		inci = 1.0 / (noise3DTexSize / frequency);
		for (i = 0; i < noise3DTexSize; ++i, ni[0] += inci)
		{
			incj = 1.0 / (noise3DTexSize / frequency);

			for (j = 0; j < noise3DTexSize; ++j, ni[1] += incj)
			{
				inck = 1.0 / (noise3DTexSize / frequency);

				for (k = 0; k < noise3DTexSize; ++k, ni[2] += inck, ptr += 4)
				{
					*(ptr + inc) = (GLubyte)(((noise3(ni) + (double)1.0) * amp) * 128.0);
				}
			}
		}
	}
}