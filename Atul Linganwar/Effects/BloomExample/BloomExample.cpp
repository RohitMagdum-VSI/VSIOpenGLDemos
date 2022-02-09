#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"
#include "../common/Shapes.h"
#include "Header.h"
#include "VertexData.h"
#include "ObjectTransformations.h"
#include "Objects.h"

#include "FrameBuffers.h"
#include "BloomExample.h"

#include "HDRSceneShader.h"
#include "HDRFilterShader.h"
#include "HDRResolveShader.h"

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "GeometricShapes.lib")

HDC ghDC = NULL;
HWND ghWnd = NULL;
HGLRC ghRC = NULL;

FILE* gpFile = NULL;

bool gbIsActivate = false;
bool gbIsFullScreen = false;
bool gbIsEscKeyPressed = false;

GLfloat gMouseX = 0.0f;
GLfloat gMouseY = 0.0f;
bool bIsMouseButtonPressed = false;

DWORD dwStyle = 0;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

GLfloat gfAngle;
GLfloat gWindowWidth;
GLfloat gWindowHeight;

vmath::mat4 gPerspectiveProjectionMatrix;
vmath::mat4 gOrthographicProjectionMatrix;

vmath::mat4 gmat4ViewMatrix = vmath::mat4::identity();
vmath::mat4 gmat4ModelMatrix = vmath::mat4::identity();
vmath::mat4 gmat4TranslationMatrix = vmath::mat4::identity();
vmath::mat4 gmat4RotationMatrix = vmath::mat4::identity();
vmath::mat4 gmat4ScaleMatrix = vmath::mat4::identity();

GLfloat gfAmbientLight[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat gfDiffuseLight[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat gfSpecularLight[] = { 0.03f, 0.03f, 0.03f, 1.0f };
GLfloat gfLightPosition[] = { 0.0f, 0.0f, 0.0f, 1.0f };

GLfloat gfAmbientMaterial[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat gfDiffuseMaterial[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat gfSpecularMaterial[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat gfMaterialShininess = 10.0f;

PLANETS Planet;
HDR_SCENE_SHADER HDRSceneShader;
HDR_FILTER_SHADER HDRFilterShader;
HDR_RESOLVE_SHADER HDRResolveShader;

FRAMEBUFFER_OBJECT gFBO;

GLuint guiVAORTT;
GLuint guiVBOPositionRTT;
GLuint guiVBOTextureRTT;

GLuint guiTextureSun;

GLuint      tex_src;
GLuint      tex_lut;

GLuint      render_fbo;
GLuint      filter_fbo[2];

GLuint      tex_scene;
GLuint      tex_brightpass;
GLuint      tex_depth;
GLuint      tex_filter[2];

GLuint      program_render;
GLuint      program_filter;
GLuint      program_resolve;
GLuint      vao;
float       exposure = 1.0f;
int         mode = false;
bool        paused = false;
float       bloom_factor = 1.0f;
bool        show_bloom = true;
bool        show_scene = true;
bool        show_prefilter = false;
float       bloom_thresh_min = 0.8f;
float       bloom_thresh_max = 1.2f;

// WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	MSG msg = { 0 };
	HWND hWnd = NULL;
	bool bDone = false;
	WNDCLASSEX WndClass = { 0 };
	WCHAR wszClassName[] = L"First Scene Rotating Planets";

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
		gWindowWidth = LOWORD(lParam);
		gWindowHeight = HIWORD(lParam);
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

		case 'B':
			show_bloom = !show_bloom;
			break;
		case 'V':
			show_scene = !show_scene;
			break;
		case 'A':
			bloom_factor += 0.1f;
			break;
		case 'Z':
			bloom_factor -= 0.1f;
			break;
		case 'S':
			bloom_thresh_min += 0.1f;
			break;
		case 'X':
			bloom_thresh_min -= 0.1f;
			break;
		case 'D':
			bloom_thresh_max += 0.1f;
			break;
		case 'C':
			bloom_thresh_max -= 0.1f;
			break;
		case 'N':
			show_prefilter = !show_prefilter;
			break;
		case VK_UP:
			exposure *= 1.1f;
			break;
		case VK_DOWN:
			exposure /= 1.1f;
			break;

		default:
			break;
		}
		break;

	case WM_LBUTTONDOWN:
		break;

	case WM_RBUTTONDOWN:
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

	if (false == InitializeHDRSceneShaderProgram(&HDRSceneShader))
	{
		fprintf(gpFile, "Error while InitializeHDRSceneShaderProgram(HDRSceneShader).\n");
		UnInitialize();
		return false;
	}

	if (false == InitializeHDRFilterShaderProgram(&HDRFilterShader))
	{
		fprintf(gpFile, "Error while InitializeHDRFilterShaderProgram(HDRFilterShader).\n");
		UnInitialize();
		return false;
	}

	if (false == InitializeHDRResolveShaderProgram(&HDRResolveShader))
	{
		fprintf(gpFile, "Error while InitializeHDRResolveShaderProgram(HDRResolveShader).\n");
		UnInitialize();
		return false;
	}

	if (FALSE == InitializePlanetsData(2.0f, 30, 30, &Planet))
	{
		fprintf(gpFile, "Error while InitializePlanetsData().\n");
		UnInitialize();
		return false;
	}

	if (false == InitializeFrameBuffer(&gFBO))
	{
		fprintf(gpFile, "Error while InitializeFrameBuffer(gFBO)\n");
		UnInitialize();
		return false;
	}
	
	//QUAD
	glGenVertexArrays(1, &guiVAORTT);
	glBindVertexArray(guiVAORTT);

	glGenBuffers(1, &guiVBOPositionRTT);
	glBindBuffer(GL_ARRAY_BUFFER, guiVBOPositionRTT);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &guiVBOTextureRTT);
	glBindBuffer(GL_ARRAY_BUFFER, guiVBOTextureRTT);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadTexCoords), quadTexCoords, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	static const GLenum buffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };

	// HDR Bloom
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	static const GLfloat exposureLUT[20] = { 11.0f, 6.0f, 3.2f, 2.8f, 2.2f, 1.90f, 1.80f, 1.80f, 1.70f, 1.70f,  1.60f, 1.60f, 1.50f, 1.50f, 1.40f, 1.40f, 1.30f, 1.20f, 1.10f, 1.00f };

	glGenFramebuffers(1, &render_fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, render_fbo);

	glGenTextures(1, &tex_scene);
	glBindTexture(GL_TEXTURE_2D, tex_scene);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, OGL_WINDOW_WIDTH, OGL_WINDOW_HEIGHT);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex_scene, 0);

	glGenTextures(1, &tex_brightpass);
	glBindTexture(GL_TEXTURE_2D, tex_brightpass);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, OGL_WINDOW_WIDTH, OGL_WINDOW_HEIGHT);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, tex_brightpass, 0);

	glGenTextures(1, &tex_depth);
	glBindTexture(GL_TEXTURE_2D, tex_depth);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT32F, OGL_WINDOW_WIDTH, OGL_WINDOW_HEIGHT);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, tex_depth, 0);

	glDrawBuffers(2, buffers);
	
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glGenFramebuffers(2, &filter_fbo[0]);
	glGenTextures(2, &tex_filter[0]);
	for (int i = 0; i < 2; i++)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, filter_fbo[i]);
		glBindTexture(GL_TEXTURE_2D, tex_filter[i]);
		glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, i ? OGL_WINDOW_WIDTH : OGL_WINDOW_HEIGHT, i ? OGL_WINDOW_HEIGHT : OGL_WINDOW_WIDTH);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex_filter[i], 0);
		glDrawBuffers(1, buffers);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glGenTextures(1, &tex_lut);
	glBindTexture(GL_TEXTURE_1D, tex_lut);
	glTexStorage1D(GL_TEXTURE_1D, 1, GL_R32F, 20);
	glTexSubImage1D(GL_TEXTURE_1D, 0, 0, 20, GL_RED, GL_FLOAT, exposureLUT);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable(GL_TEXTURE_2D);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	LoadGLTextures(&guiTextureSun, MAKEINTRESOURCE(ID_BITMAP_SUN));

	gPerspectiveProjectionMatrix = vmath::mat4::identity();
	gOrthographicProjectionMatrix = vmath::mat4::identity();

	//ToggleFullScreen();
	//gbIsFullScreen = true;

	Resize((int)gWindowWidth, (int)gWindowHeight);

	return true;
}

void Display(void)
{
	static const GLfloat black[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	static const GLfloat one = 1.0f;
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBindFramebuffer(GL_FRAMEBUFFER, render_fbo);
	glClearBufferfv(GL_COLOR, 0, black);
	glClearBufferfv(GL_COLOR, 1, black);
	glClearBufferfv(GL_DEPTH, 0, &one);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glUseProgram(HDRSceneShader.ShaderObject.uiShaderProgramObject);

	ClearMatrices();

	gmat4ViewMatrix = vmath::lookat(vmath::vec3(0.0f, 0.0f, 10.0f), vmath::vec3(0.0f, 0.0f, 0.0f), vmath::vec3(0.0f, 1.0f, 0.0f));

	//
	//	Sun
	//
	gmat4RotationMatrix = vmath::mat4::identity();
	gmat4RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f); // after this y becomes -z and z becomes y
	gmat4ModelMatrix = gmat4ModelMatrix * gmat4RotationMatrix;

	glUniformMatrix4fv(HDRSceneShader.uiModelMatrixUniform, 1, GL_FALSE, gmat4ModelMatrix);
	glUniformMatrix4fv(HDRSceneShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
	glUniformMatrix4fv(HDRSceneShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, guiTextureSun);
	glUniform1i(HDRSceneShader.uiTextureSamplerUniform, 0);

	glUniform3fv(HDRSceneShader.uiLaUniform, 1, gfAmbientLight);
	glUniform3fv(HDRSceneShader.uiLdUniform, 1, gfDiffuseLight);
	glUniform3fv(HDRSceneShader.uiLsUniform, 1, gfSpecularLight);
	glUniform4fv(HDRSceneShader.uiLightPositionUniform, 1, gfLightPosition);

	glUniform3fv(HDRSceneShader.uiKaUniform, 1, gfAmbientMaterial);
	glUniform3fv(HDRSceneShader.uiKdUniform, 1, gfDiffuseMaterial);
	glUniform3fv(HDRSceneShader.uiKsUniform, 1, gfSpecularMaterial);
	glUniform1f(HDRSceneShader.uiMaterialShininessUniform, gfMaterialShininess);

	glUniform1f(HDRSceneShader.uiBloomThreshMaxUniform, bloom_thresh_max);
	glUniform1f(HDRSceneShader.uiBloomThreshMinUniform, bloom_thresh_min);

	glBindVertexArray(Planet.Sphere.uiVAO);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Planet.Sphere.uiVBOElements);
	glDrawElements(GL_TRIANGLES, Planet.Sphere.SphereData.uiIndicesCount, GL_UNSIGNED_INT, 0);

	glBindVertexArray(0);

	glDisable(GL_DEPTH_TEST);

	glUseProgram(0);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// Filter
	glUseProgram(HDRFilterShader.ShaderObject.uiShaderProgramObject);

	glBindVertexArray(vao);

	glBindFramebuffer(GL_FRAMEBUFFER, filter_fbo[0]);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex_brightpass);
	glUniform1i(HDRFilterShader.uiTextureHDRImageUniform, 0);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, filter_fbo[1]);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex_filter[0]);
	glUniform1i(HDRFilterShader.uiTextureHDRImageUniform, 0);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//glUseProgram(0);
	//
	//glBindVertexArray(0);

	//// Resolve
	//glUseProgram(HDRResolveShader.ShaderObject.uiShaderProgramObject);

	//glUniform1f(HDRResolveShader.uiExposureUniform, exposure);
	//if (show_prefilter)
	//{
	//	glUniform1f(HDRResolveShader.uiBloomFactorUniform, 0.0f);
	//	glUniform1f(HDRResolveShader.uiSceneFactorUniform, 1.0f);
	//}
	//else
	//{
	//	glUniform1f(HDRResolveShader.uiBloomFactorUniform, show_bloom ? bloom_factor : 0.0f);
	//	glUniform1f(HDRResolveShader.uiSceneFactorUniform, show_scene ? 1.0f : 0.0f);
	//}
	//glBindVertexArray(vao);

	//glActiveTexture(GL_TEXTURE1);
	//glBindTexture(GL_TEXTURE_2D, tex_filter[1]);
	//glUniform1i(HDRResolveShader.uiTextureHDRImageUniform, 0);

	//glActiveTexture(GL_TEXTURE0);
	//glBindTexture(GL_TEXTURE_2D, show_prefilter ? tex_brightpass : tex_scene);
	//glUniform1i(HDRResolveShader.uiTextureBloomImageUniform, 0);

	//glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	//glBindVertexArray(0);

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

	return;
}

int LoadGLTextures(GLuint* texture, TCHAR imageResourceId[])
{
	HBITMAP hBitmap;
	BITMAP bmp;
	int iStatus = FALSE;

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

		glTexImage2D(
			GL_TEXTURE_2D,
			0,
			GL_RGB,
			bmp.bmWidth,
			bmp.bmHeight,
			0,
			GL_BGR,
			GL_UNSIGNED_BYTE,
			bmp.bmBits
		);

		glGenerateMipmap(GL_TEXTURE_2D);

		DeleteObject(hBitmap);
	}

	return iStatus;
}

void Resize(int iWidth, int iHeight)
{
	if (iHeight == 0)
	{
		iHeight = 1;
	}

	glViewport(0, 0, (GLsizei)iWidth, (GLsizei)iHeight);

	if (iWidth < iHeight)
	{
		gOrthographicProjectionMatrix = vmath::ortho(-ORTHO, ORTHO, (-ORTHO * ((GLfloat)iHeight / (GLfloat)iWidth)), (ORTHO * ((GLfloat)iHeight / (GLfloat)iWidth)), -ORTHO, ORTHO);
	}
	else
	{
		gOrthographicProjectionMatrix = vmath::ortho((-ORTHO * ((GLfloat)iWidth / (GLfloat)iHeight)), (ORTHO * ((GLfloat)iWidth / (GLfloat)iHeight)), -ORTHO, ORTHO, -ORTHO, ORTHO);
	}

	gPerspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)iWidth / (GLfloat)iHeight, 0.1f, 100000.0f);
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
		// ShowCursor(FALSE);
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

	UnInitializeHDRSceneShaderProgram(&HDRSceneShader);
	UnInitializeHDRFilterShaderProgram(&HDRFilterShader);
	UnInitializeHDRResolveShaderProgram(&HDRResolveShader);

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

void ClearMatrices()
{
	gmat4ModelMatrix = vmath::mat4::identity();
	gmat4TranslationMatrix = vmath::mat4::identity();
	gmat4RotationMatrix = vmath::mat4::identity();
	gmat4ScaleMatrix = vmath::mat4::identity();
}