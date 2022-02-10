#include "Common.h"

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

#define ORTHO_SIZE 100.0f

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "GeometricShapes.lib")

HWND ghwnd;
HDC ghdc;
HGLRC ghrc;

FILE *gpFile = NULL;
bool gbActiveWindow = false;
bool gbEscapeKeyIsPressed = false;
bool gbFullScreen = false;
DWORD dwStyle = 0;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

mat4 gOrthographicProjectionMatrix;
mat4 gPerspectiveProjectionMatrix;
mat4 gViewMatrix = mat4::identity();

// Objects
PLANET gPlanet;
ORBIT gOrbit;

// Shader objects
PLANET_SHADER gPlanetShader;
COLOR_SHADER gColorShader;
PICKING_SHADER gPickingShader;

SATURN_RING gSaturnRing;

PICKING_DATA gPickedObjectData;

GLuint gPlanetTextures[15];

STAR_FIELD gStarField;

extern FLOAT gXRot;
extern FLOAT gYRot;
extern FLOAT gZRot;

extern FRAME_BUFFER gFrameBuffer;

FLOAT gWindowWidth = 0;
FLOAT gWindowHeight = 0;

FLOAT gLeftMouseButtonX = 0.0f;
FLOAT gLeftMouseButtonY = 0.0f;

static double gCurrentTime = 0;
 
BOOL gProcessPicking = FALSE;

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCommandLine, int iCmdShow)
{
	void initialize(void);
	void uninitialize(void);
	void display(void);
	void update(void);

	WNDCLASSEX wndClass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("OGL");
	bool bDone = false;

	if (fopen_s(&gpFile, "Log.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Log File can not be Created\nExitting...."), TEXT("Error"), MB_OK | MB_TOPMOST | MB_ICONSTOP);
		exit(0);
	}
	fprintf(gpFile, "Log File is successfully opened. \n");

	int width = 0, height = 0;
	width = GetSystemMetrics(SM_CXSCREEN);
	height = GetSystemMetrics(SM_CXSCREEN);

	wndClass.cbSize = sizeof(WNDCLASSEX);
	wndClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndClass.cbClsExtra = 0;
	wndClass.cbWndExtra = 0;
	wndClass.hInstance = hInstance;
	wndClass.lpfnWndProc = WndProc;
	wndClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndClass.lpszClassName = szClassName;
	wndClass.lpszMenuName = NULL;
	wndClass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

	RegisterClassEx(&wndClass);

	hwnd = CreateWindowEx(
		WS_EX_APPWINDOW,
		szClassName,
		TEXT("OGL : SolarSystem"),
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		(width - WIN_WIDTH) / 2,
		(height - WIN_HEIGHT) / 2,
		WIN_WIDTH,
		WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL
	);

	ghwnd = hwnd;

	initialize();

	ShowWindow(hwnd, SW_SHOW);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

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
				if (gbEscapeKeyIsPressed == true)
					bDone = true;
			}

			update();
			display();
		}
	}

	uninitialize();
	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	void resize(int, int);
	void ToggleFullScreen(void);
	void uninitialize(void);

	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			gbActiveWindow = true;
		else
			gbActiveWindow = false;
		break;
	case WM_ERASEBKGND:
		return 0;
		break;
	case WM_SIZE:
		gWindowWidth = LOWORD(lParam);
		gWindowHeight = HIWORD(lParam);
		resize(LOWORD(lParam), HIWORD(lParam));
		break;
	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			if (gbEscapeKeyIsPressed == false)
				gbEscapeKeyIsPressed = true;
			break;
		case 0x46:
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
		case VK_LEFT:
			gYRot -= 0.1f;
			break;
		case VK_RIGHT:
			gYRot += 0.1f;
			break;
		case VK_UP:
			gXRot -= 10.1f;
			break;
		case VK_DOWN:
			gXRot += 10.1f;
			break;
		default:
			break;
		}
		break;
	case WM_MOUSEWHEEL:
		gXRot -= (GET_WHEEL_DELTA_WPARAM(wParam) * 5.0f);
		break;
	case WM_LBUTTONDOWN:
		if (!gProcessPicking) 
		{
			gLeftMouseButtonX = LOWORD(lParam);
			gLeftMouseButtonY = HIWORD(lParam);
			gProcessPicking = TRUE;
		}
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

	return (DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullScreen(void)
{
	MONITORINFO mi;

	if (gbFullScreen == false)
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
		//ShowCursor(FALSE);
	}

	else
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);

		//ShowCursor(TRUE);
	}
}

void initialize(void)
{
	void resize(int, int);
	void uninitialize(void);
	int LoadGLTextures(GLuint *, TCHAR[]);

	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

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

	GLenum glew_error = glewInit();
	if (glew_error != GLEW_OK)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	//
	//	Initialize FrameBuffers
	//
	InitFrameBuffer(gFrameBuffer);

	//
	//	PLANETS
	//
	InitPlanetShaders(gPlanetShader);	
	InitPlanet(gPlanet, DEFAULT_PLANET_RADIUS, FALSE);

	LoadGLTextures(&gPlanetTextures[PLANETS_AND_SATELLITES::SPHERE_MAP], MAKEINTRESOURCE(ID_STARRY_SKY));
	LoadGLTextures(&gPlanetTextures[PLANETS_AND_SATELLITES::SUN], MAKEINTRESOURCE(ID_PLANET_SUN));
	LoadGLTextures(&gPlanetTextures[PLANETS_AND_SATELLITES::MERCURY], MAKEINTRESOURCE(ID_PLANET_MECURY));
	LoadGLTextures(&gPlanetTextures[PLANETS_AND_SATELLITES::VENUS], MAKEINTRESOURCE(ID_PLANET_VENUS));
	LoadGLTextures(&gPlanetTextures[PLANETS_AND_SATELLITES::EARTH], MAKEINTRESOURCE(ID_PLANET_EARTH));
	LoadGLTextures(&gPlanetTextures[PLANETS_AND_SATELLITES::MARS], MAKEINTRESOURCE(ID_PLANET_MARS));
	LoadGLTextures(&gPlanetTextures[PLANETS_AND_SATELLITES::JUPITER], MAKEINTRESOURCE(ID_PLANET_JUPITER));
	LoadGLTextures(&gPlanetTextures[PLANETS_AND_SATELLITES::SATURN], MAKEINTRESOURCE(ID_PLANET_SATURN));
	LoadGLTextures(&gPlanetTextures[PLANETS_AND_SATELLITES::SATURN_RING_ID], MAKEINTRESOURCE(ID_SATURN_RING));
	LoadGLTextures(&gPlanetTextures[PLANETS_AND_SATELLITES::URANUS], MAKEINTRESOURCE(ID_PLANET_URANUS));
	LoadGLTextures(&gPlanetTextures[PLANETS_AND_SATELLITES::NEPTUNE], MAKEINTRESOURCE(ID_PLANET_NEPTUNE));
	LoadGLTextures(&gPlanetTextures[PLANETS_AND_SATELLITES::PLUTO], MAKEINTRESOURCE(ID_PLANET_PLUTO));
	
	//
	//	Star Field
	//
	InitStarField(gStarField, gWindowWidth, gWindowHeight);

	//
	//	Orbit
	//
	InitColorShaders(gColorShader);
	InitOrbit(gOrbit, DEFAULT_ELLIPSE_MAJOR_RADIUS, DEFAULT_ELLIPSE_MINOR_RADIUS);

	//
	//	Picking Shader
	//
	InitPickingShader(gPickingShader);

	//
	//	Initilize Saturn Ring
	//
	InitSaturnRing(gSaturnRing, SATURN_RING_INNER_RADIUS, SATURN_RING_OUTER_RADIUS);

	//
	//	Clear Color
	//
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glShadeModel(GL_SMOOTH);

	//
	//	Depth
	//
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glEnable(GL_TEXTURE_2D);

	gPerspectiveProjectionMatrix = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);

	ToggleFullScreen();
	gbFullScreen = TRUE;
}

int LoadGLTextures(GLuint *texture, TCHAR imageResourceId[])
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
			GL_BGR_EXT,
			GL_UNSIGNED_BYTE,
			bmp.bmBits
		);

		glGenerateMipmap(GL_TEXTURE_2D);

		glBindTexture(GL_TEXTURE_2D, 0);

		DeleteObject(hBitmap);
	}

	return iStatus;
}

void resize(int width, int height)
{
	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	ResizeFrameBuffer(gFrameBuffer);
	
	if (width < height)
		gOrthographicProjectionMatrix = ortho(-ORTHO_SIZE, ORTHO_SIZE, (-ORTHO_SIZE * ((GLfloat)height / (GLfloat)width)), (ORTHO_SIZE * ((GLfloat)height / (GLfloat)width)), -ORTHO_SIZE, ORTHO_SIZE);
	else 
		gOrthographicProjectionMatrix = ortho((-ORTHO_SIZE * ((GLfloat)width / (GLfloat)height)), (ORTHO_SIZE * ((GLfloat)width / (GLfloat)height)), -ORTHO_SIZE, ORTHO_SIZE,  -ORTHO_SIZE, ORTHO_SIZE);

	gPerspectiveProjectionMatrix = perspective(50.0f, ((GLfloat)width / (GLfloat)height), 0.1f, 100000.0f);
}

void update()
{
	gCurrentTime += 0.00001f;

	UpdatePlanetsAngle();
}

void display(void)
{
	double starFieldSpeed = gCurrentTime * 100;
	float fPlanetScale = 0;

	GLfloat black[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	GLfloat one[] = { 1.0f };

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	mat4 modelViewMatrix = mat4::identity();
	mat4 modelViewProjectionMatrix = mat4::identity();
	mat4 translationMatrix = mat4::identity();
	mat4 rotationMatrix = mat4::identity();
	mat4 scaleMatrix = mat4::identity();

	gViewMatrix = vmath::lookat(
		vec3(0.0f, 20000.0f, 30000.0f + gXRot),
		vec3(0.0f, 8000.0f, 0.0f), 
		vec3(0.0f, 1.0f, 0.0f)
	);

	//
	//	Draw StarField in FBO
	//
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gFrameBuffer.fbo);
	glDisable(GL_DEPTH_TEST);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		DrawStarField(starFieldSpeed, gStarField);
	glEnable(GL_DEPTH_TEST);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//
	//	Drawing sphere map
	//
	fPlanetScale = GetPlanetScale(PLANETS_AND_SATELLITES::SPHERE_MAP);
	scaleMatrix = scale(fPlanetScale, fPlanetScale, fPlanetScale);

	rotationMatrix = rotate(90.0f, 0.0f, 0.0f, 1.0f);
	rotationMatrix *= rotate(270.0f, 0.0f, 1.0f, 0.0f);
	
	modelViewMatrix = translate(0.0f, 0.0f, 0.0f) * rotationMatrix * scaleMatrix;

	glDisable(GL_CULL_FACE);
	DrawPlanet(gStarField.SphereMap, gPlanetShader, modelViewMatrix, gFrameBuffer.colorTexture);
	glEnable(GL_CULL_FACE);

	//
	//	Check if mouse clicked , then process picking
	//
	if (gProcessPicking)
	{
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gFrameBuffer.fbo);
		DrawAllPlanetsPicking();
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		GetPickedObject();
		
		gProcessPicking = FALSE;
	}

	if (IsObjectPicked())
	{
		DrawPickedObject();
	}

	//
	//	Planets
	//
	DrawAllPlanets();

	//
	//	Orbits
	//
	DrawAllOrbits();

	SwapBuffers(ghdc);
}

void uninitialize(void)
{
	//UNINITIALIZATION CODE
	if (gbFullScreen == true)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);

		ShowCursor(TRUE);
	}

	CleanupPlanetShader(gPlanetShader);
	CleanupPlanet(gPlanet);
	CleanupColorShader(gColorShader);
	CleanupOrbit(gOrbit);
	CleanupStarField(gStarField);
	CleanupFrameBuffer(gFrameBuffer);

	wglMakeCurrent(NULL, NULL);

	wglDeleteContext(ghrc);
	ghrc = NULL;

	ReleaseDC(ghwnd, ghdc);
	ghdc = NULL;

	DestroyWindow(ghwnd);
	ghwnd = NULL;

	if (gpFile)
	{
		fprintf(gpFile, "Log File is Successfully closed.");
		fclose(gpFile);
		gpFile = NULL;
	}
}
