#include <windows.h>
#include <GL/glew.h> 
#include <gl/GL.h>
#include <stdio.h>
#include <math.h>
#include <strsafe.h>

#include "vmath.h"
#include "Sphere.h"

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "Sphere.lib")

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

GLfloat gWidth;
GLfloat gHeight;

void ToggleFullScreen(void);
int initialize(void);
void resize(int, int);
void display(void);
void update(void);
void uninitialize(void);

GLuint vao_sphere;
GLuint vbo_position_sphere; 
GLuint vbo_normal_sphere;
GLuint vbo_element_sphere;

GLuint viewUniform;
GLuint modelUniform;
GLuint projectionUniform;

GLuint laUniform;
GLuint ldUniform;
GLuint lsUniform;
GLuint lightPositionUniform;

GLuint kaUniform;
GLuint kdUniform;
GLuint ksUniform;
GLuint materialShininessUniform;

mat4 perspectiveProjectionMatrix;

bool gbLight = false;
GLfloat angleRotation = 0.0f;

float sphere_vertices[1146];
float sphere_normals[1146];
float sphere_texture[746];
unsigned short sphere_elements[2280];
int gNumVertices = 0;
int gNumElements = 0;

//frame buffer
GLuint render_fbo;
GLuint tex_scene;
GLuint tex_brightpass;
GLuint tex_depth;
GLuint filter_fbo[2];
GLuint tex_filter[2];
GLuint tex_lut;

//light
float LightAmbient[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
float LightDiffuse[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float LightSpecular[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
float LightPosition[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

float MaterialAmbient[4];
float MaterialDiffuse[4];
float MaterialSpecular[4];
float MaterialShininess;
float ambient;

void sphereInitialize(void);

void programRender(void);
void programFilter(void);
void programResolve(void);
void passUniforms(void);

//
GLuint gShaderProgramObject_render;
GLuint gShaderProgramObject_filter;
GLuint gShaderProgramObject_resolve;

GLuint vao_rectangle;
GLuint vbo_position_rectangle;
GLuint vbo_texture_rectangle;

float exposure = 1.0f;
int mode = 0;
float bloom_factor = 1.5f;
float scene_factor = 1.0f;
float bloom_thresh_min = 0.8f;
float bloom_thresh_max = 1.2f;

bool show_bloom = true;
bool show_scene = true;
bool show_prefilter = false;

GLuint bloomThreshMaxUniform;
GLuint bloomThreshMinUniform;
GLuint samplerUniform;

GLuint bloomSamplerUniform;
GLuint exposureUniform;
GLuint bloomFactorUniform;
GLuint sceneFactorUniform;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szAppName[] = TEXT("MY WINDOW");
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
		TEXT("HDR BLOOM - Jayshree"),
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
			if (gbActiveWindow == true)
			{
				update();
			}
			display();
		}
	}

	return ((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	TCHAR str[255];

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

		resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_ERASEBKGND:
		return (0);
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
			if (gbFullScreen == false)
			{
				ToggleFullScreen();
				gbFullScreen = true;
			}
			else
			{
				ToggleFullScreen();
				gbFullScreen = false;
			}
			break;

		case 'l':
		case 'L':
			if (gbLight == false)
			{
				gbLight = true;
			}
			else
			{
				gbLight = false;
			}
			break;

		case 'b':
		case 'B':
			show_bloom = !show_bloom;
			break;

		case 'v':
		case 'V':
			show_scene = !show_scene;
			break;
		
		case 'a':
		case 'A':
			bloom_factor += 0.1f;
			//wsprintf(str, TEXT("Bloom Factor : %f"), bloom_factor);
			_snwprintf_s(str, sizeof(str), TEXT("Bloom Factor : %f"), bloom_factor);
			SetWindowText(hwnd, str);
			break;

		case 'z':
		case 'Z':
			bloom_factor -= 0.1f;
			_snwprintf_s(str, sizeof(str), TEXT("Bloom Factor : %f"), bloom_factor);
			SetWindowText(hwnd, str);
			break;

		case 's':
		case 'S':
			bloom_thresh_min += 0.1f;
			_snwprintf_s(str, sizeof(str), TEXT("Bloom Min Threshold : %f"), bloom_thresh_min);
			SetWindowText(hwnd, str);
			break;

		case 'x':
		case 'X':
			bloom_thresh_min -= 0.1f;
			_snwprintf_s(str, sizeof(str), TEXT("Bloom Min Threshold : %f"), bloom_thresh_min);
			SetWindowText(hwnd, str);
			break;

		case 'd':
		case 'D':
			bloom_thresh_max += 0.1f;
			_snwprintf_s(str, sizeof(str), TEXT("Bloom Max Threshold : %f"), bloom_thresh_max);
			SetWindowText(hwnd, str);
			break;

		case 'c':
		case 'C':
			bloom_thresh_max -= 0.1f;
			_snwprintf_s(str, sizeof(str), TEXT("Bloom Max Threshold : %f"), bloom_thresh_max);
			SetWindowText(hwnd, str);
			break;
		
		case 'n':
		case 'N':
			show_prefilter = !show_prefilter;
			break;

		case VK_UP:
			exposure *= 1.1f;
			_snwprintf_s(str, sizeof(str), TEXT("Exposure : %f"), exposure);
			SetWindowText(hwnd, str);
			break;

		case VK_DOWN:
			exposure /= 1.1f;
			_snwprintf_s(str, sizeof(str), TEXT("Exposure : %f"), exposure);
			SetWindowText(hwnd, str);
			break;
		}
		break;

	case WM_DESTROY:
		uninitialize();
		PostQuitMessage(0);
		break;
	}

	return (DefWindowProc(hwnd, iMsg, wParam, lParam));
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
			SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);

		ShowCursor(TRUE);
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
		return (-1);
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		return (-2);
	}

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		return (-3);
	}

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		return (-4);
	}

	result = glewInit();
	if (result != GLEW_OK)
	{
		fprintf(gpFile, "\nglewInit() failed...\n");
		uninitialize();
		DestroyWindow(ghWnd);
	}

	static const GLfloat exposureLUT[20] = { 11.0f, 6.0f, 3.2f, 2.8f, 2.2f, 1.90f, 1.80f, 1.80f, 1.70f, 1.70f,  1.60f, 1.60f, 1.50f, 1.50f, 1.40f, 1.40f, 1.30f, 1.20f, 1.10f, 1.00f };
	int i;
	static const GLenum buffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
	
	programRender();
	programFilter();
	programResolve();
	passUniforms();
	
	const GLfloat rectangleTexCoord[] = {
		1.0f, 1.0f,
		0.0f,1.0f,
		0.0f,0.0f,
		1.0f,0.0f };
	
	//create vao rectangle(vertex array object)
	glGenVertexArrays(1, &vao_rectangle);

	//Bind vao
	glBindVertexArray(vao_rectangle);

	//--------------------texture------------------
	//generate vertex buffers
	glGenBuffers(1, &vbo_texture_rectangle);

	//bind buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo_texture_rectangle);

	//transfer vertex data(CPU) to GPU buffer
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(rectangleTexCoord),
		rectangleTexCoord,
		GL_STATIC_DRAW);

	//attach or map attribute pointer to vbo's buffer
	glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOORD0,
		2,
		GL_FLOAT,
		GL_FALSE,
		0,
		NULL);

	//enable vertex attribute array
	glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOORD0);

	//unbind vbo
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//unbind vao
	glBindVertexArray(0);

	glGenFramebuffers(1, &render_fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, render_fbo);

	glGenTextures(1, &tex_scene);
	glBindTexture(GL_TEXTURE_2D, tex_scene);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, 2048, 2048);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex_scene, 0);

	glGenTextures(1, &tex_brightpass);
	glBindTexture(GL_TEXTURE_2D, tex_brightpass);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, 2048, 2048);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, tex_brightpass, 0);

	glGenTextures(1, &tex_depth);
	glBindTexture(GL_TEXTURE_2D, tex_depth);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT32F, 2048, 2048);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, tex_depth, 0);

	glDrawBuffers(2, buffers);

	glGenFramebuffers(2, &filter_fbo[0]);
	glGenTextures(2, &tex_filter[0]);
	for (i = 0; i < 2; i++)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, filter_fbo[i]);
		glBindTexture(GL_TEXTURE_2D, tex_filter[i]);
		glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, i ? 2048 : 2048, i ? 2048 : 2048);
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

	sphereInitialize();
	
	ambient = 0.002f * (1.5f * 8);	

	float fi = 3.14159267f * (float)7/ 8.0f;
	MaterialAmbient[0] = ambient * 0.025f;
	MaterialAmbient[1] = ambient * 0.025f;
	MaterialAmbient[2] = ambient * 0.025f;
	MaterialAmbient[3] = 0.0f;

	MaterialDiffuse[0] = sinf(fi) * 0.5f + 0.5f;
	MaterialDiffuse[1] = sinf(fi + 1.345f) * 0.5f + 0.5f;
	MaterialDiffuse[2] = sinf(fi + 2.567f) * 0.5f + 0.5f;
	MaterialDiffuse[3] = 1.0f;

	MaterialSpecular[0] = 2.8f;
	MaterialSpecular[1] = 2.8f;
	MaterialSpecular[2] = 2.9f;
	MaterialSpecular[3] = 1.0f;

	MaterialShininess = 30.0f;

	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

	//glDisable(GL_CULL_FACE);
	glEnable(GL_CULL_FACE);

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);

	//glDepthFunc(GL_LEQUAL);
	glDepthFunc(GL_LESS);

	perspectiveProjectionMatrix = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);

	return (0);
}

void resize(int width, int height)
{
	if (height == 0)
	{
		height = 1;
	}

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	perspectiveProjectionMatrix = perspective(50.0f, (GLfloat)width / (GLfloat)height, 1.0f, 1000.0f);
}

void display(void)
{
	mat4 viewMatrix;
	mat4 modelMatrix;
	mat4 ProjectionMatrix;
	mat4 translationMatrix;
	mat4 rotationMatrix;

	static const GLfloat black[] = {0.0f, 0.0f, 0.0f, 1.0f};
	static const GLfloat one = 1.0f;

	glViewport(0, 0, (GLsizei)gWidth, (GLsizei)gHeight);

	glBindFramebuffer(GL_FRAMEBUFFER, render_fbo);
	glClearBufferfv(GL_COLOR, 0, black);
	glClearBufferfv(GL_COLOR, 1, black);
	glClearBufferfv(GL_DEPTH, 0, &one);

	glUseProgram(gShaderProgramObject_render);

	viewMatrix = mat4::identity();
	modelMatrix = mat4::identity();
	ProjectionMatrix = mat4::identity();
	translationMatrix = mat4::identity();
	rotationMatrix = mat4::identity();

	translationMatrix = translate(0.0f, 0.0f, -3.0f);
	rotationMatrix = rotate(angleRotation, 0.0f, 1.0f, 0.0f);

	modelMatrix =  translationMatrix * rotationMatrix;
	ProjectionMatrix = perspectiveProjectionMatrix;

	glUniformMatrix4fv(viewUniform, 1, GL_FALSE, viewMatrix); 
	glUniformMatrix4fv(modelUniform, 1, GL_FALSE, modelMatrix); 
	glUniformMatrix4fv(projectionUniform, 1, GL_FALSE, ProjectionMatrix);

	glUniform3fv(laUniform, 1, LightAmbient);
	glUniform3fv(ldUniform, 1, LightDiffuse);
	glUniform3fv(lsUniform, 1, LightSpecular);
	glUniform3fv(kaUniform, 1, MaterialAmbient);
	glUniform3fv(kdUniform, 1, MaterialDiffuse);
	glUniform3fv(ksUniform, 1, MaterialSpecular);
	glUniform1f(materialShininessUniform, MaterialShininess);
	glUniform4fv(lightPositionUniform, 1, LightPosition);
	
	glUniform1f(bloomThreshMinUniform, bloom_thresh_min);
	glUniform1f(bloomThreshMaxUniform, bloom_thresh_max);

	//glCullFace(GL_BACK);

	glBindVertexArray(vao_sphere);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_element_sphere);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindVertexArray(0);

	//glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);

	glUseProgram(gShaderProgramObject_filter);

	glBindVertexArray(vao_rectangle);
	glBindFramebuffer(GL_FRAMEBUFFER, filter_fbo[0]);
	glBindTexture(GL_TEXTURE_2D, tex_brightpass);
	glViewport(0, 0, (GLsizei)gHeight, (GLsizei)gWidth);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	glBindFramebuffer(GL_FRAMEBUFFER, filter_fbo[1]);
	glBindTexture(GL_TEXTURE_2D, tex_filter[0]);
	glViewport(0, 0, (GLsizei)gWidth, (GLsizei)gHeight);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	glUseProgram(gShaderProgramObject_resolve);
	glUniformMatrix4fv(projectionUniform, 1, GL_FALSE, ProjectionMatrix);
	glUniform1f(exposureUniform, exposure);
	if (show_prefilter)
	{
		glUniform1f(bloomFactorUniform, 0.0f);
		glUniform1f(sceneFactorUniform, 1.0f);
	}
	else
	{
		glUniform1f(bloomFactorUniform, show_bloom ? bloom_factor : 0.0f);
		glUniform1f(sceneFactorUniform, show_scene ? 1.0f : 0.0f);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, tex_filter[1]);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, show_prefilter ? tex_brightpass : tex_scene);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	
	//glUseProgram(0);

	SwapBuffers(ghdc);
}

void update(void)
{
	angleRotation = angleRotation + 0.1f;
	if (angleRotation > 360.0f)
		angleRotation = 0.0f;
}

void uninitialize(void)
{
	if (vbo_element_sphere)
	{
		glDeleteBuffers(1, &vbo_element_sphere);
		vbo_element_sphere = 0;
	}

	if (vbo_normal_sphere)
	{
		glDeleteBuffers(1, &vbo_normal_sphere);
		vbo_normal_sphere = 0;
	}

	if (vbo_position_sphere)
	{
		glDeleteBuffers(1, &vbo_position_sphere);
		vbo_position_sphere = 0;
	}

	if (vao_sphere)
	{
		glDeleteVertexArrays(1, &vao_sphere);
		vao_sphere = 0;
	}
	
	if (gShaderProgramObject_render)
	{
		GLsizei shaderCount;
		GLsizei shaderNumber;

		glUseProgram(gShaderProgramObject_render);
				
		glGetProgramiv(gShaderProgramObject_render,
			GL_ATTACHED_SHADERS,
			&shaderCount);

		GLuint *pShaders = (GLuint *)malloc(sizeof(GLuint) * shaderCount);

		if (pShaders)
		{
		
			glGetAttachedShaders(gShaderProgramObject_render,
				shaderCount,
				&shaderCount,
				pShaders);

			for (shaderNumber = 0; shaderNumber < shaderCount; shaderNumber++)
			{
				glDetachShader(gShaderProgramObject_render,
					pShaders[shaderNumber]);

				glDeleteShader(pShaders[shaderNumber]);

				pShaders[shaderNumber] = 0;
			}

			free(pShaders);
		}

		glDeleteProgram(gShaderProgramObject_render);
		gShaderProgramObject_render = 0;

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

void sphereInitialize(void)
{
	getSphereVertexData(sphere_vertices,
		sphere_normals,
		sphere_texture,
		sphere_elements);

	gNumVertices = getNumberOfSphereVertices();
	gNumElements = getNumberOfSphereElements();

	glGenVertexArrays(1, &vao_sphere);
	glBindVertexArray(vao_sphere);
	glGenBuffers(1, &vbo_position_sphere);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_position_sphere);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(sphere_vertices),
		sphere_vertices,
		GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &vbo_normal_sphere);

	glBindBuffer(GL_ARRAY_BUFFER, vbo_normal_sphere);

	glBufferData(GL_ARRAY_BUFFER,
		sizeof(sphere_normals),
		sphere_normals,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &vbo_element_sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_element_sphere);

	glBufferData(GL_ELEMENT_ARRAY_BUFFER,
		sizeof(sphere_elements),
		sphere_elements,
		GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);


}


void programRender(void)
{
	GLuint vertexShaderObject;
	GLuint fragmentShaderObject;

	//***************** 1. VERTEX SHADER ************************************
	vertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *vertexShaderSourceCode =
		"#version 450 core"
		"\n"
		"in vec4 vPosition;"
		"in vec3 vNormal;"
		"uniform mat4 u_view_matrix;"
		"uniform mat4 u_model_matrix;"
		"uniform mat4 u_projection_matrix;"
		"uniform vec4 u_light_position;"
		"out vec3 tNormal;"
		"out vec3 light_direction;"
		"out vec3 viewer_vector;"
		"void main(void)"
		"{"
		"vec4 eyeCoords = u_view_matrix * u_model_matrix * vPosition;"
		"tNormal = mat3(u_view_matrix * u_model_matrix) * vNormal;"
		"light_direction = vec3(u_light_position - eyeCoords);"
		"viewer_vector = vec3(-eyeCoords.xyz);"
		"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;"
		"}";

	glShaderSource(vertexShaderObject,
		1,
		(const GLchar **)&vertexShaderSourceCode,
		NULL);


	glCompileShader(vertexShaderObject);

	GLint iShaderCompileStatus = 0;
	GLint iInfoLogLength = 0;
	GLchar *szInfoLog = NULL;

	glGetShaderiv(vertexShaderObject,
		GL_COMPILE_STATUS,
		&iShaderCompileStatus);

	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(vertexShaderObject,
			GL_INFO_LOG_LENGTH,
			&iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;

				glGetShaderInfoLog(vertexShaderObject,
					iInfoLogLength,
					&written,
					szInfoLog);

				fprintf(gpFile, "\nVertex Shader (program render) Compilation Log : %s\n", szInfoLog);

				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}

	//************************** 2. FRAGMENT SHADER ********************************
	fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *fragmentShaderSourceCode =
		"#version 450 core"
		"\n"
		"in vec3 tNormal;"
		"in vec3 light_direction;"
		"in vec3 viewer_vector;"
		"uniform float u_bloom_thresh_min;"
		"uniform float u_bloom_thresh_max;"
		"uniform vec3 u_la;"
		"uniform vec3 u_ld;"
		"uniform vec3 u_ls;"
		"uniform vec3 u_ka;"
		"uniform vec3 u_kd;"
		"uniform vec3 u_ks;"
		"uniform float u_material_shininess;"
		"vec3 phong_ads_light;"
		"out vec4 fragColor0;"
		"out vec4 fragColor1;"
		"void main(void)"
		"{"
		"	vec3 normalized_tNormal = normalize(tNormal);"
		"	vec3 normalized_light_direction = normalize(light_direction);"
		"	float tNorm_Dot_LightDirection = max(dot(normalized_light_direction, normalized_tNormal), 0.0);"
		"	vec3 reflection_vector = reflect(-normalized_light_direction, normalized_tNormal);"
		"	vec3 normalized_viewer_vector = normalize(viewer_vector);"
		"	vec3 ambient = u_la * u_ka;"
		"	vec3 diffuse = u_ld * u_kd * tNorm_Dot_LightDirection;"
		"	vec3 specular = u_ls * u_ks * pow(max(dot(reflection_vector,normalized_viewer_vector), 0.0), u_material_shininess);"
		"	phong_ads_light = ambient + diffuse + specular;"
		"	fragColor0 = vec4(phong_ads_light, 1.0);" \
		"	float Y = dot(phong_ads_light, vec3(0.299, 0.587, 0.144));"	\
		"	phong_ads_light = phong_ads_light * 4.0 * smoothstep(u_bloom_thresh_min, u_bloom_thresh_max, Y);"
		"	fragColor1 = vec4(phong_ads_light,1.0);"
		"}";


	glShaderSource(fragmentShaderObject,
		1,
		(const GLchar **)&fragmentShaderSourceCode,
		NULL);


	glCompileShader(fragmentShaderObject);
	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(fragmentShaderObject,
		GL_COMPILE_STATUS,
		&iShaderCompileStatus);

	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(fragmentShaderObject,
			GL_INFO_LOG_LENGTH,
			&iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);

			if (szInfoLog != NULL)
			{
				GLsizei written;

				glGetShaderInfoLog(fragmentShaderObject,
					iInfoLogLength,
					&written,
					szInfoLog);

				fprintf(gpFile, "\nFragment Shader (program render)Compilation Log : %s\n", szInfoLog);

				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}

	gShaderProgramObject_render = glCreateProgram();

	glAttachShader(gShaderProgramObject_render,
		vertexShaderObject);

	glAttachShader(gShaderProgramObject_render,
		fragmentShaderObject);

	glBindAttribLocation(gShaderProgramObject_render,
		AMC_ATTRIBUTE_POSITION,
		"vPosition");

	glBindAttribLocation(gShaderProgramObject_render,
		AMC_ATTRIBUTE_NORMAL,
		"vNormal");

	glLinkProgram(gShaderProgramObject_render);

	GLint iProgramLinkStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;


	glGetProgramiv(gShaderProgramObject_render,
		GL_LINK_STATUS,
		&iProgramLinkStatus);

	if (iProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(gShaderProgramObject_render,
			GL_INFO_LOG_LENGTH,
			&iInfoLogLength);

		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;

				glGetProgramInfoLog(gShaderProgramObject_render,
					iInfoLogLength,
					&written,
					szInfoLog);

				fprintf(gpFile, "\nShader Program Render Linking Log : %s\n", szInfoLog);


				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}
}

void programFilter(void)
{
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
		"in vec2 vTexCoord;" \
		"out vec2 out_texcoord;" \
		"void main(void)"\
		"{" \
		"const vec4 vertices[] = vec4[](vec4(-1.0, -1.0, 0.0, 1.0),vec4(1.0, -1.0, 0.0, 1.0),vec4(-1.0, 1.0, 0.0, 1.0),vec4(1.0, 1.0, 0.0, 1.0));" \
		"gl_Position = vertices[gl_VertexID];" \
		"out_texcoord = vTexCoord;" \
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

				fprintf(gpFile, "\nVertex Shader filter Compilation Log : %s\n", szInfoLog);

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
		"in vec2 out_texcoord;" \
		"uniform sampler2D u_sampler;" \
		"out vec4 fragColor;" \
		"const float weights[] = float[](0.0024499299678342,0.0043538453346397,0.0073599963704157,0.0118349786570722,0.0181026699707781,0.0263392293891488,0.0364543006660986,0.0479932050577658,0.0601029809166942,0.0715974486241365,0.0811305381519717,0.0874493212267511,0.0896631113333857,0.0874493212267511,0.0811305381519717,0.0715974486241365,0.0601029809166942,0.0479932050577658,0.0364543006660986,0.0263392293891488,0.0181026699707781,0.0118349786570722,0.0073599963704157,0.0043538453346397,0.0024499299678342);" \
		"void main(void)" \
		"{" \
		"vec4 c = vec4(0.0);" \
		"ivec2 P = ivec2(gl_FragCoord.yx) - ivec2(0, weights.length() >> 1);" \
		"int i;" \
		"for(i =0; i < weights.length(); i++)" \
		"{"
		"	c += texelFetch(u_sampler, P * ivec2(0, i), 0) * weights[i];" \
		"}" \
		"fragColor = c; " \
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

				fprintf(gpFile, "\nFragment Shader filter Compilation Log : %s\n", szInfoLog);

				//free the memory
				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}


	//create shader program object
	gShaderProgramObject_filter = glCreateProgram();

	//Attach vertex shader to shader program
	glAttachShader(gShaderProgramObject_filter,//to whom?
		vertexShaderObject);//what to attach?

	//Attach fragment shader to shader program
	glAttachShader(gShaderProgramObject_filter,
		fragmentShaderObject);

	//Pre-Linking binding to vertex attribute
	//glBindAttribLocation(gShaderProgramObject_filter,
		//AMC_ATTRIBUTE_POSITION,
		//"vPosition");

	glBindAttribLocation(gShaderProgramObject_filter,
		AMC_ATTRIBUTE_TEXCOORD0,
		"vTexCoord");

	//Link the shader program
	glLinkProgram(gShaderProgramObject_filter);//link to whom?


	//Error checking for linking
	GLint iProgramLinkStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	//Step 1 : Call glGetShaderiv() to get comiple status of particular shader
	glGetProgramiv(gShaderProgramObject_filter, // whose?
		GL_LINK_STATUS,//what to get?
		&iProgramLinkStatus);//in what?

	//Step 2 : Check shader compile status for GL_FALSE
	if (iProgramLinkStatus == GL_FALSE)
	{
		//Step 3 : If GL_FALSE , call glGetShaderiv() again , but this time to get info log length
		glGetProgramiv(gShaderProgramObject_filter,
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

				glGetProgramInfoLog(gShaderProgramObject_filter,//whose?
					iInfoLogLength,//length?
					&written,//might have not used all, give that much only which have been used in what?
					szInfoLog);//store in what?

				fprintf(gpFile, "\nShader Program Filter Linking Log : %s\n", szInfoLog);

				//free the memory
				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}

}

void programResolve(void)
{
	GLuint vertexShaderObject;
	GLuint fragmentShaderObject;

	//***************** 1. VERTEX SHADER ************************************ 
	vertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	//Write vertex shader code
	const GLchar *vertexShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"in vec2 vTexCoord;" \
		"out vec2 out_texcoord;" \
		"void main(void)"\
		"{" \
		"const vec4 vertices[] = vec4[](vec4(-1.0, -1.0, 0.0, 1.0),vec4(1.0, -1.0, 0.0, 1.0),vec4(-1.0, 1.0, 0.0, 1.0),vec4(1.0, 1.0, 0.0, 1.0));" \
		"gl_Position = vertices[gl_VertexID];" \
		"out_texcoord = vTexCoord;" \
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

				fprintf(gpFile, "\nVertex Shader resolve Compilation Log : %s\n", szInfoLog);

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
		"in vec2 out_texcoord;" \
		"uniform sampler2D u_sampler;" \
		"uniform sampler2D u_sampler_bloom;" \
		"uniform float u_exposure;" \
		"uniform float u_bloom_factor;" \
		"uniform float u_scene_factor;" \
		"out vec4 fragColor;" \
		"void main(void)" \
		"{" \
		"vec4 c = vec4(0.0);" \
		"c += texelFetch(u_sampler, ivec2(gl_FragCoord.xy), 0) * u_scene_factor;" \
		"c += texelFetch(u_sampler_bloom, ivec2(gl_FragCoord.xy), 0) * u_bloom_factor;" \
		"c.rgb = vec3(1.0) - exp(-c.rgb * u_exposure);"
		"fragColor = c; " \
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

				fprintf(gpFile, "\nFragment Shader resolve Compilation Log : %s\n", szInfoLog);

				//free the memory
				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}


	//create shader program object
	gShaderProgramObject_resolve = glCreateProgram();

	//Attach vertex shader to shader program
	glAttachShader(gShaderProgramObject_resolve,//to whom?
		vertexShaderObject);//what to attach?

	//Attach fragment shader to shader program
	glAttachShader(gShaderProgramObject_resolve,
		fragmentShaderObject);

	//Pre-Linking binding to vertex attribute
	//glBindAttribLocation(gShaderProgramObject_resolve,
		//AMC_ATTRIBUTE_POSITION,
		//"vPosition");

	glBindAttribLocation(gShaderProgramObject_resolve,
		AMC_ATTRIBUTE_TEXCOORD0,
		"vTexCoord");

	//Link the shader program
	glLinkProgram(gShaderProgramObject_resolve);//link to whom?


	//Error checking for linking
	GLint iProgramLinkStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	//Step 1 : Call glGetShaderiv() to get comiple status of particular shader
	glGetProgramiv(gShaderProgramObject_resolve, // whose?
		GL_LINK_STATUS,//what to get?
		&iProgramLinkStatus);//in what?

	//Step 2 : Check shader compile status for GL_FALSE
	if (iProgramLinkStatus == GL_FALSE)
	{
		//Step 3 : If GL_FALSE , call glGetShaderiv() again , but this time to get info log length
		glGetProgramiv(gShaderProgramObject_resolve,
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

				glGetProgramInfoLog(gShaderProgramObject_resolve,//whose?
					iInfoLogLength,//length?
					&written,//might have not used all, give that much only which have been used in what?
					szInfoLog);//store in what?

				fprintf(gpFile, "\nShader Program resolve Linking Log : %s\n", szInfoLog);

				//free the memory
				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}

}

void passUniforms(void)
{
	viewUniform = glGetUniformLocation(gShaderProgramObject_render,
		"u_view_matrix");

	modelUniform = glGetUniformLocation(gShaderProgramObject_render,
		"u_model_matrix");

	projectionUniform = glGetUniformLocation(gShaderProgramObject_render,
		"u_projection_matrix");

	//lKeyPressedUniform = glGetUniformLocation(gShaderProgramObject_render,
		//"u_lKeyPressed");

	laUniform = glGetUniformLocation(gShaderProgramObject_render,
		"u_la");

	ldUniform = glGetUniformLocation(gShaderProgramObject_render,
		"u_ld");

	lsUniform = glGetUniformLocation(gShaderProgramObject_render,
		"u_ls");

	lightPositionUniform = glGetUniformLocation(gShaderProgramObject_render,
		"u_light_position");

	kaUniform = glGetUniformLocation(gShaderProgramObject_render,
		"u_ka");

	kdUniform = glGetUniformLocation(gShaderProgramObject_render,
		"u_kd");

	ksUniform = glGetUniformLocation(gShaderProgramObject_render,
		"u_ks");

	materialShininessUniform = glGetUniformLocation(gShaderProgramObject_render,
		"u_material_shininess");

	bloomThreshMaxUniform = glGetUniformLocation(gShaderProgramObject_render,
		"u_bloom_thresh_max");

	bloomThreshMinUniform = glGetUniformLocation(gShaderProgramObject_render,
		"u_bloom_thresh_min");

	samplerUniform = glGetUniformLocation(gShaderProgramObject_filter,
		"u_sampler");
		
	samplerUniform = glGetUniformLocation(gShaderProgramObject_resolve,
		"u_sampler");

	bloomSamplerUniform = glGetUniformLocation(gShaderProgramObject_resolve,
		"u_sampler_bloom");

	exposureUniform = glGetUniformLocation(gShaderProgramObject_resolve,
		"u_exposure");

	bloomFactorUniform = glGetUniformLocation(gShaderProgramObject_resolve,
		"u_bloom_factor");

	sceneFactorUniform = glGetUniformLocation(gShaderProgramObject_resolve,
		"u_scene_factor");
}
