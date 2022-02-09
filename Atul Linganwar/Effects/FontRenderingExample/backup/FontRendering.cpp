#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include <ft2build.h>
#include FT_FREETYPE_H

#include <freetype/fttypes.h>
#include <freetype/fterrors.h>

#include "../common/vmath.h"
#include "FontRendering.h"

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "freetype.lib")

HDC ghDC = NULL;
HWND ghWnd = NULL;
HGLRC ghRC = NULL;

FILE* gpFile = NULL;

bool gbIsActivate = false;
bool gbIsFullScreen = false;
bool gbIsEscKeyPressed = false;

DWORD dwStyle = 0;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

GLuint guiVBO;
GLuint guiAttribCoord;
GLuint guiFontColorUniform;
GLuint guiTextureSamplerUniform;

GLuint guiVertexShaderObject;
GLuint guiFragmentShaderObject;
GLuint guiShaderProgramObject;

vmath::mat4 gPerspectiveProjectionMatrix;
vmath::mat4 gOrthoProjectionMatrix;

FT_Library gFT;
FT_Face gFace;

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

	// Initialize FreeType library
	if (FT_Init_FreeType(&gFT)) 
	{
		fprintf(gpFile, "Error while initializing freetype library\n");
		wglDeleteContext(ghRC);
		ghRC = NULL;
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
		return false;
	}
	
	// Load some font
	if (FT_New_Face(gFT, "arial.ttf", 0, &gFace))
	{
		fprintf(gpFile, "Error while initializing FT_New_Face()\n");
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
		"in vec4 coord;" \
		"out vec2 texpos;" \
		"void main(void)" \
		"{" \
		"gl_Position = vec4(coord.xy, 0, 1);" \
		"texpos = coord.zw;" \
		"}";

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

	const GLchar* glchFragmentShaderSource =
		"#version 430 core" \
		"\n" \
		"in vec2 texpos;" \
		"out vec4 FragColor;" \
		"uniform sampler2D u_tex;" \
		"uniform vec4 u_color;" \
		"void main(void)" \
		"{" \
		"FragColor = vec4(1, 1, 1, texture2D(u_tex, texpos).a) * u_color;" \
		"}";

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

	//glBindAttribLocation(guiShaderProgramObject, OGL_ATTRIBUTE_POSITION, "coord");
	guiAttribCoord = glGetAttribLocation(guiShaderProgramObject, "coord");

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
	guiFontColorUniform = glGetUniformLocation(guiShaderProgramObject, "u_color");
	guiTextureSamplerUniform = glGetUniformLocation(guiShaderProgramObject, "u_tex");

	glGenBuffers(1, &guiVBO);

	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	//glEnable(GL_CULL_FACE);
	//glEnable(GL_TEXTURE_2D);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gOrthoProjectionMatrix = vmath::mat4::identity();
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
	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(guiShaderProgramObject);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	GLfloat black[4] = { 0, 0, 0, 1 };

	FT_Set_Pixel_Sizes(gFace, 0, 48);
	glUniform4fv(guiFontColorUniform, 1, black);

	float sx = 2.0 / OGL_WINDOW_WIDTH;
	float sy = 2.0 / OGL_WINDOW_HEIGHT;

	render_text("Hello World", -1 + 8 * sx, 1 - 50 * sy, sx, sy);

	glUseProgram(0);

	SwapBuffers(ghDC);
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
		gOrthoProjectionMatrix = vmath::ortho(-100.0f, 100.0f, (-100.0f * ((GLfloat)iHeight / (GLfloat)iWidth)), (100.0f * ((GLfloat)iHeight / (GLfloat)iWidth)), -100.0f, 100.0f);
	}
	else
	{
		gOrthoProjectionMatrix = vmath::ortho((-100.0f * ((GLfloat)iWidth / (GLfloat)iHeight)), (100.0f * ((GLfloat)iWidth / (GLfloat)iHeight)), -100.0f, 100.0f, -100.0f, 100.0f);
	}

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

void render_text(const char* text, float x, float y, float sx, float sy)
{
	const char* pch;
	GLuint uiTexture = 0;
	FT_GlyphSlot g = gFace->glyph;

	glActiveTexture(GL_TEXTURE0);

	glGenTextures(1, &uiTexture);
	glBindTexture(GL_TEXTURE_2D, uiTexture);
	glUniform1i(guiTextureSamplerUniform, 0);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	// Clamping to edges is important to prevent objects when scaling
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	// Linear filtering usually looks for text
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glEnableVertexAttribArray(guiAttribCoord);
	glBindBuffer(GL_ARRAY_BUFFER, guiVBO);
	glVertexAttribPointer(guiAttribCoord, 4, GL_FLOAT, GL_FALSE, 0, 0);

	int i = 0;

	for (pch = text; *pch; pch++)
	{
		i = i + 3;
		FT_Error error = FT_Load_Char(gFace, *pch, FT_LOAD_RENDER);
		if (FT_Err_Ok != error)
		{
			fprintf(gpFile, "FT Error : %s\n", FT_Error_String(error));
			continue;
		}
		
		glTexImage2D(
			GL_TEXTURE_2D,
			0,
			GL_ALPHA,
			gFace->glyph->bitmap.width,
			gFace->glyph->bitmap.rows,
			0,
			GL_ALPHA,
			GL_UNSIGNED_BYTE,
			gFace->glyph->bitmap.buffer
		);
		glGenerateMipmap(GL_TEXTURE_2D);

		/*float xpos = x + gFace->glyph->bitmap_left * sx;
		float ypos = -y - gFace->glyph->bitmap_top * sy;
		float w = gFace->glyph->bitmap.width * sx;
		float h = gFace->glyph->bitmap.rows * sy;

		GLfloat box[4][4] =
		{
			{xpos, -ypos, 0, 0},
			{xpos + w, -ypos, 1, 0},
			{xpos, -ypos - h, 0, 1},
			{xpos + w, -ypos - h, 1, 1},
		};*/

		GLfloat box[4][4] =
		{
			{1 + i, 1, 1, 1},
			{-1 + i, 1, 0, 1},
			{-1+i, -1, 0, 0},
			{1+i, -1, 1, 0},
		};

		float x2 = x + g->bitmap_left * sx;
		float y2 = -y - g->bitmap_top * sy;
		float w = g->bitmap.width * sx;
		float h = g->bitmap.rows * sy;

		/*GLfloat box[4][4] = {
			{x2,     -y2    , 0, 0},
			{x2 + w, -y2    , 1, 0},
			{x2,     -y2 - h, 0, 1},
			{x2 + w, -y2 - h, 1, 1},
		};*/

		glBufferData(GL_ARRAY_BUFFER, sizeof(box), box, GL_DYNAMIC_DRAW);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

		x += (gFace->glyph->advance.x / 64) * sx;
		y += (gFace->glyph->advance.y / 64) * sy;
	}
	
	glDisableVertexAttribArray(OGL_ATTRIBUTE_POSITION);

	if (0 != uiTexture)
	{
		glDeleteTextures(1, &uiTexture);
		uiTexture = 0;
	}

	return;
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

	if (0 != guiVBO)
	{
		glDeleteBuffers(1, &guiVBO);
		guiVBO = 0;
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