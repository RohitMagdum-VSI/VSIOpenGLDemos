#include<windows.h>
#include<gl/GL.h>
#include<gl/GLU.h>
#include<stdio.h>

#pragma comment(lib,"User32.lib")
#pragma comment(lib,"GDI32.lib")
#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"glu32.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

HWND ghwnd;
HDC ghdc;
HGLRC ghrc;
FILE *gpFile;
WINDOWPLACEMENT wpPrev;
DWORD dwStyle;

bool gbFullscreen = false;
bool gbActiveWindow = false;
bool gbIsEscapeKeyPressed = false;
bool gbIsLKeyPressed = false;
bool gbLight_Rotate_Flag = false;

GLfloat light_ambient[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat light_diffuse[] = { 1.0f,1.0f,1.0f,1.0f };
GLfloat light_specular[] = { 1.0f,1.0f,1.0f,1.0f };
GLfloat light_position[] = { 0.0f,0.0f,1.0f,0.0f };

GLfloat material_ambient_1[] = { 0.0215f,0.1745f,0.0215f,1.0f };
GLfloat material_diffuse_1[] = { 0.07568f,0.61424f,0.07568f,1.0f };
GLfloat material_specular_1[] = { 0.633f,0.727811f,0.633f,1.0f };
GLfloat material_shininess_1 = 0.6f * 128.0f;

GLfloat material_ambient_2[] = { 0.135f,0.2225f,0.1575f,1.0f };
GLfloat material_diffuse_2[] = { 0.54f,0.89f,0.63f,1.0f };
GLfloat material_specular_2[] = { 0.316228f,0.316228f,0.316228f,1.0f };
GLfloat material_shininess_2 = 0.1f * 128.0f;

GLfloat material_ambient_3[] = { 0.05375f,0.05f,0.06625f,1.0f };
GLfloat material_diffuse_3[] = { 0.18275f,0.17f,0.22525f,1.0f };
GLfloat material_specular_3[] = { 0.332741f,0.328634f,0.346435f,1.0f };
GLfloat material_shininess_3 = 0.3f * 128.0f;

GLfloat material_ambient_4[] = { 0.25f,0.20725f,0.20725f,1.0f };
GLfloat material_diffuse_4[] = { 1.0f,0.829f,0.829f,1.0f };
GLfloat material_specular_4[] = { 0.296648f,0.296648f,0.296648f,1.0f };
GLfloat material_shininess_4 = 0.088f * 128.0f;

GLfloat material_ambient_5[] = { 0.1745f,0.01175f,0.01175f,1.0f };
GLfloat material_diffuse_5[] = { 0.61424f,0.04136f,0.04136f,1.0f };
GLfloat material_specular_5[] = { 0.727811f,0.626959f,0.626959f,1.0f };
GLfloat material_shininess_5 = 0.6f * 128.0f;

GLfloat material_ambient_6[] = { 0.1f,0.18725f,0.1745f,1.0f };
GLfloat material_diffuse_6[] = { 0.396f,0.74151f,0.69102f,1.0f };
GLfloat material_specular_6[] = { 0.297254f,0.30829f,0.306678f,1.0f };
GLfloat material_shininess_6 = 0.1f * 128.0f;

GLfloat material_ambient_7[] = { 0.329412f,0.223529f,0.027451f,1.0f };
GLfloat material_diffuse_7[] = { 0.780392f,0.568627f,0.113725f,1.0f };
GLfloat material_specular_7[] = { 0.992157f,0.941176f,0.807843f,1.0f };
GLfloat material_shininess_7 = 0.21794872f * 128.0f;

GLfloat material_ambient_8[] = { 0.2125f,0.1275f,0.054f,1.0f };
GLfloat material_diffuse_8[] = { 0.714f,0.4284f,0.18144f,1.0f };
GLfloat material_specular_8[] = { 0.393548f,0.271906f,0.166721f,1.0f };
GLfloat material_shininess_8 = 0.2f * 128.0f;

GLfloat material_ambient_9[] = { 0.25f,0.25f,0.25f,1.0f };
GLfloat material_diffuse_9[] = { 0.4f,0.4f,0.4f,1.0f };
GLfloat material_specular_9[] = { 0.774597f,0.774597f,0.774597f,1.0f };
GLfloat material_shininess_9 = 0.6f * 128.0f;

GLfloat material_ambient_10[] = { 0.19125f,0.0735f,0.0225f,1.0f };
GLfloat material_diffuse_10[] = { 0.7038f,0.27048f,0.0828f,1.0f };
GLfloat material_specular_10[] = { 0.256777f,0.137622f,0.086014f,1.0f };
GLfloat material_shininess_10 = 0.1f * 128.0f;

GLfloat material_ambient_11[] = { 0.24725f,0.1995f,0.0745f,1.0f };
GLfloat material_diffuse_11[] = { 0.75164f,0.60648f,0.22648f,1.0f };
GLfloat material_specular_11[] = { 0.628281f,0.555802f,0.366065f,1.0f };
GLfloat material_shininess_11 = 0.4f * 128.0f;

GLfloat material_ambient_12[] = { 0.19225f,0.19225f,0.19225f,1.0f };
GLfloat material_diffuse_12[] = { 0.50754f,0.50754f,0.50754f,1.0f };
GLfloat material_specular_12[] = { 0.508273f,0.508273f,0.508273f,1.0f };
GLfloat material_shininess_12 = 0.4f * 128.0f;

GLfloat material_ambient_13[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat material_diffuse_13[] = { 0.01f,0.01f,0.01f,1.0f };
GLfloat material_specular_13[] = { 0.5f,0.5f,0.5f,1.0f };
GLfloat material_shininess_13 = 0.25f * 128.0f;

GLfloat material_ambient_14[] = { 0.0f,0.1f,0.06f,1.0f };
GLfloat material_diffuse_14[] = { 0.0f,0.50980392f,0.50980392f,1.0f };
GLfloat material_specular_14[] = { 0.50196078f,0.50196078f,0.50196078f,1.0f };
GLfloat material_shininess_14 = 0.25f * 128.0f;

GLfloat material_ambient_15[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat material_diffuse_15[] = { 0.1f,0.35f,0.1f,1.0f };
GLfloat material_specular_15[] = { 0.45f,0.55f,0.45f,1.0f };
GLfloat material_shininess_15 = 0.25f * 128.0f;

GLfloat material_ambient_16[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat material_diffuse_16[] = { 0.5f,0.0f,0.0f,1.0f };
GLfloat material_specular_16[] = { 0.7f,0.6f,0.6f,1.0f };
GLfloat material_shininess_16 = 0.25f * 128.0f;

GLfloat material_ambient_17[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat material_diffuse_17[] = { 0.55f,0.55f,0.55f,1.0f };
GLfloat material_specular_17[] = { 0.70f,0.70f,0.70f,1.0f };
GLfloat material_shininess_17 = 0.25f * 128.0f;

GLfloat material_ambient_18[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat material_diffuse_18[] = { 0.5f,0.5f,0.0f,1.0f };
GLfloat material_specular_18[] = { 0.6f,0.6f,0.5f,1.0f };
GLfloat material_shininess_18 = 0.25f * 128.0f;

GLfloat material_ambient_19[] = { 0.02f,0.02f,0.02f,1.0f };
GLfloat material_diffuse_19[] = { 0.1f,0.1f,0.1f,1.0f };
GLfloat material_specular_19[] = { 0.4f,0.4f,0.4f,1.0f };
GLfloat material_shininess_19 = 0.078125f * 128.0f;

GLfloat material_ambient_20[] = { 0.0f,0.05f,0.05f,1.0f };
GLfloat material_diffuse_20[] = { 0.4f,0.5f,0.5f,1.0f };
GLfloat material_specular_20[] = { 0.04f,0.7f,0.7f,1.0f };
GLfloat material_shininess_20 = 0.078125f * 128.0f;

GLfloat material_ambient_21[] = { 0.0f,0.05f,0.0f,1.0f };
GLfloat material_diffuse_21[] = { 0.4f,0.5f,0.4f,1.0f };
GLfloat material_specular_21[] = { 0.04f,0.7f,0.04f,1.0f };
GLfloat material_shininess_21 = 0.078125f * 128.0f;

GLfloat material_ambient_22[] = { 0.05f,0.0f,0.0f,1.0f };
GLfloat material_diffuse_22[] = { 0.5f,0.4f,0.4f,1.0f };
GLfloat material_specular_22[] = { 0.7f,0.04f,0.04f,1.0f };
GLfloat material_shininess_22 = 0.078125f * 128.0f;

GLfloat material_ambient_23[] = { 0.05f,0.05f,0.05f,1.0f };
GLfloat material_diffuse_23[] = { 0.5f,0.5f,0.5f,1.0f };
GLfloat material_specular_23[] = { 0.7f,0.7f,0.7f,1.0f };
GLfloat material_shininess_23 = 0.078125f * 128.0f;

GLfloat material_ambient_24[] = { 0.05f,0.05f,0.0f,1.0f };
GLfloat material_diffuse_24[] = { 0.5f,0.5f,0.4f,1.0f };
GLfloat material_specular_24[] = { 0.7f,0.7f,0.04f,1.0f };
GLfloat material_shininess_24 = 0.078125f * 128.0f;

GLfloat gAngle = 0.0f;
GLfloat gfRotate_X_Value = 0.0f;
GLfloat gfRotate_Y_Value = 0.0f;
GLfloat gfRotate_Z_Value = 0.0f;

GLUquadric *quadric = NULL;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	void uninitialize(int);
	void initialize(void);
	void display(void);
	void update(void);

	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("My OpenGL");
	bool bDone = false;

	if (fopen_s(&gpFile, "Log.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Cannot Create File!!!"), TEXT("ERROR"), MB_OK);
		exit(EXIT_FAILURE);
	}
	else
		fprintf(gpFile, "Log File Is Created...\n");

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

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, TEXT("OpenGL"), WS_OVERLAPPEDWINDOW|WS_CLIPCHILDREN|WS_CLIPSIBLINGS|WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);
	if (hwnd == NULL)
	{
		fprintf(gpFile, "CreateWindow() Failed");
		uninitialize(1);
	}

	ghwnd = hwnd;

	ShowWindow(ghwnd,iCmdShow);
	SetForegroundWindow(ghwnd);
	SetFocus(ghwnd);

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
	void uninitialize(void);

	//code
	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			gbActiveWindow = true;
		else
			gbActiveWindow = false;
		break;
	case WM_ERASEBKGND:
		return(0);
	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;
	case WM_KEYDOWN:
		switch (LOWORD(wParam))
		{
		case VK_ESCAPE:
			gbIsEscapeKeyPressed = true;
			break;
		}
		break;
	case WM_CHAR:
		switch(wParam)
		{
		case 'L':
		case 'l':
			if (gbIsLKeyPressed == false)
			{
				glEnable(GL_LIGHTING);
				gbIsLKeyPressed = true;
			}
			else
			{
				glDisable(GL_LIGHTING);
				gbIsLKeyPressed = false;
			}
			break;
		case 'F':
		case 'f':
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

		case 'X':
		case 'x':
			gfRotate_X_Value = 1.0f;
			gfRotate_Y_Value = 0.0f;
			gfRotate_Z_Value = 0.0f;
			gbLight_Rotate_Flag = true;
			break;

		case 'Y':
		case 'y':
			gfRotate_X_Value = 0.0f;
			gfRotate_Y_Value = 1.0f;
			gfRotate_Z_Value = 0.0f;
			gbLight_Rotate_Flag = true;
			break;

		case 'Z':
		case 'z':
			gfRotate_X_Value = 0.0f;
			gfRotate_Y_Value = 0.0f;
			gfRotate_Z_Value = 1.0f;
			gbLight_Rotate_Flag = true;
			break;

		default:
			gbLight_Rotate_Flag = false;
			light_position[0] = 0.0f;
			light_position[1] = 0.0f;
			light_position[2] = 0.0f;
			light_position[3] = 0.0f;
			break;
		}
		break;
	case WM_LBUTTONDOWN:
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void initialize(void)
{
	void uninitialize(int);
	void resize(int, int);
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
		fprintf(gpFile, "GetDC() Failed");
		uninitialize(1);
	}

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0)
	{
		fprintf(gpFile, "ChoosePixelFormat() Failed");
		uninitialize(1);
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		fprintf(gpFile, "SetPixelFormat() Failed");
		uninitialize(1);
	}

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		fprintf(gpFile, "wglCreateContext() Failed");
		uninitialize(1);
	}

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		fprintf(gpFile, "wglMakeCurrent() Failed");
		uninitialize(1);
	}

	glClearColor(0.1f, 0.1f, 0.1f, 0.0f);

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT , GL_NICEST);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	//glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular);
	//glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess);
	glEnable(GL_LIGHT0);

	resize(WIN_WIDTH, WIN_HEIGHT);
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(0.0f, 0.0f, -0.1f);
	glPushMatrix();

	if (gbLight_Rotate_Flag == true)
	{
		glPushMatrix();
		glTranslatef(0.0f, 0.0f, -1.0f);
		glRotatef(gAngle, gfRotate_X_Value, gfRotate_Y_Value, gfRotate_Z_Value);
		if (gfRotate_Y_Value == 1.0f)
		{
			light_position[0] = gAngle;
			light_position[1] = 0.0f;
		}
		else
		{
			light_position[0] = 0.0f;
			light_position[1] = gAngle;
		}
		glLightfv(GL_LIGHT0, GL_POSITION, light_position);
		glPopMatrix();
	}
	
	glPushMatrix();
	//Polygon 1
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_1);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_1);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_1);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_1);

	glTranslatef(-3.0f, 2.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 2
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_2);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_2);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_2);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_2);

	glTranslatef(-3.0f, 1.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 3
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_3);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_3);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_3);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_3);

	glTranslatef(-3.0f, 0.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 4
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_4);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_4);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_4);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_4);

	glTranslatef(-3.0f, -0.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 5
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_5);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_5);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_5);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_5);

	glTranslatef(-3.0f, -1.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 6
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_6);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_6);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_6);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_6);

	glTranslatef(-3.0f, -2.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 7
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_7);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_7);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_7);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_7);

	glTranslatef(-1.0f, 2.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 8
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_8);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_8);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_8);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_8);

	glTranslatef(-1.0f, 1.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 9
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_9);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_9);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_9);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_9);

	glTranslatef(-1.0f, 0.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 10
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_10);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_10);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_10);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_10);

	glTranslatef(-1.0f, -0.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 11
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_11);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_11);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_11);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_11);

	glTranslatef(-1.0f, -1.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 12
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_12);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_12);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_12);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_12);

	glTranslatef(-1.0f, -2.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 13
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_13);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_13);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_13);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_13);

	glTranslatef(1.0f, 2.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 14
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_14);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_14);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_14);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_14);

	glTranslatef(1.0f, 1.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 15
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_15);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_15);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_15);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_15);

	glTranslatef(1.0f, 0.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 16
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_16);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_16);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_16);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_16);

	glTranslatef(1.0f, -0.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 17
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_17);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_17);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_17);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_17);

	glTranslatef(1.0f, -1.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 18
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_18);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_18);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_18);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_18);

	glTranslatef(1.0f, -2.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 19
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_19);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_19);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_19);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_19);

	glTranslatef(3.0f, 2.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 20
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_20);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_20);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_20);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_20);

	glTranslatef(3.0f, 1.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 21
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_21);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_21);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_21);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_21);

	glTranslatef(3.0f, 0.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 22
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_22);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_22);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_22);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_22);

	glTranslatef(3.0f, -0.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 23
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_23);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_23);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_23);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_23);

	glTranslatef(3.0f, -1.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	//Polygon 24
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient_24);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse_24);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular_24);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess_24);

	glTranslatef(3.0f, -2.5f, -10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();

	gluSphere(quadric, 0.4, 100, 100);

	glPopMatrix(); 
	glPopMatrix();

	SwapBuffers(ghdc);
}

void resize(int width, int height)
{
	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void update(void)
{
	/*if (gAngle >= 360.0f)
		gAngle = 0.0f;*/
	gAngle = gAngle + 1.0f;
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
				SetWindowPos(ghwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.left, SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
	}

	else
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}
}

void uninitialize(int iExit_Flag)
{
	if(gbFullscreen==true)
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}

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

	if (iExit_Flag == 1)
		fprintf(gpFile, "Log file closed due to some error!!!\n");
	else if (iExit_Flag == 0)
		fprintf(gpFile, "Log file closed Successfully...\n");

	DestroyWindow(ghwnd);
}
