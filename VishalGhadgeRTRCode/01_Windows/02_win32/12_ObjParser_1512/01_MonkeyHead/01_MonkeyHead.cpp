#include <Windows.h>
#include <gl\GL.h>
#include <gl\GLU.h>

//	c++ headers.
#include <vector>

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")

#define WIN_WIDTH	800
#define WIN_HEIGHT	600

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

VOID resize(int, int);

//
//	Global variable.
//
HWND g_hWnd;
HDC g_hdc;
HGLRC g_hRC;

DWORD g_dwStyle;
WINDOWPLACEMENT g_WindowPlacementPrev = { sizeof(WINDOWPLACEMENT) };

bool g_boFullScreen = false;
bool g_boActiveWindow = false;
bool g_boEscapeKeyPressed = false;

GLfloat g_arrLightAmbient[] = { 1.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrLightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_arrLightSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
//GLfloat g_arrLightPosition[] = { 0.0f, 1.0f, 0.0f, 1.0f };
GLfloat g_arrLightPosition[] = { 1.0f, 1.0f, 1.0f, 0.0f };

GLfloat g_arrMaterialSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_arrMaterialShininess[] = { 50.0f };

BOOLEAN g_bEnableLight = FALSE;

#define	CLASS_NAME		TEXT("Object : Monkey Head")

///////////////////////////////////////////////////////////////////////////
//+	Symbol constant for obj parser

#define OBJ_FILE_NAME	"MonkeyHead.obj"
#define LOG_FILE_NAME	"MonkeyHead.log"
#define BUFFER_SIZE		256
#define S_EQUAL			0	//	Return value of strcmp when strings are equal.

#define NUM_POINTS_COORDS	3	//	Number of point coordinates
#define NUM_TEXTURE_COORD	2	//	No of texture coordinates
#define NUM_NORMAL_COORD	3	//	No of normal coordinates
#define NUM_FACE_TOKENS		3	//	Minimum number of face entries in face data

#define MONKEYHEAD_TRANSLATE_X	0.0f
#define MONKEYHEAD_TRANSLATE_Y	-0.0f
#define MONKEYHEAD_TRANSLATE_Z	-5.0f

#define MONKEYHEAD_SCALE_FACTOR_X	1.5f
#define MONKEYHEAD_SCALE_FACTOR_Y	1.5f
#define MONKEYHEAD_SCALE_FACTOR_Z	1.5f

GLfloat g_rotate = 360.0f;
#define UPDATE_ANGLE	0.1f;

//	Vector of vector of floats to hold vertext data.
std::vector <std::vector<float>> g_vecVertices;

//	Vector of vector of floats to hold texture data.
std::vector<std::vector<float>> g_vecTexture;

//	Vector of vector of floats to hold normal data.
std::vector<std::vector<float>> g_vecNormals;

//	Vector of vector of int to hold index data in g_vecVertices.
std::vector<std::vector<int>> g_vecFaceTriangles;
std::vector<std::vector<int>> g_vecFaceTexture;
std::vector<std::vector<int>> g_vecFaceNormals;

//	Handle to mesh file
FILE *g_fpMeshFile = NULL;

//	Handle to log file
FILE *g_fpLogFile = NULL;

//	Hold line in file.
char g_Line[BUFFER_SIZE];

//-	Symbol constant for obj parser
///////////////////////////////////////////////////////////////////////////

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	VOID initialize();
	VOID display();
	VOID uninitialize();
	void update();

	MSG Msg;
	int x, y;
	HWND hWnd;
	int iMaxWidth;
	int iMaxHeight;
	WNDCLASSEX WndClass;
	bool boDone = false;
	TCHAR szClassName[] = CLASS_NAME;

	//
	//	Initialize members of window class.
	//
	WndClass.cbSize = sizeof(WNDCLASSEX);
	WndClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;		//	Change:Added CS_OWNDC.
	WndClass.cbClsExtra = 0;
	WndClass.cbWndExtra = 0;
	WndClass.hInstance = hInstance;
	WndClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	WndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	WndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	WndClass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	WndClass.lpfnWndProc = WndProc;
	WndClass.lpszClassName = szClassName;
	WndClass.lpszMenuName = NULL;

	//
	//	Register class.
	//
	RegisterClassEx(&WndClass);

	iMaxWidth = GetSystemMetrics(SM_CXFULLSCREEN);
	iMaxHeight = GetSystemMetrics(SM_CYFULLSCREEN);

	x = (iMaxWidth - WIN_WIDTH) / 2;
	y = (iMaxHeight - WIN_HEIGHT) / 2;

	//
	//	Create Window.
	//
	hWnd = CreateWindowEx(
		WS_EX_APPWINDOW,	//	Change: New member get added for CreateWindowEx API.
		szClassName,
		CLASS_NAME,
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,		//	Change: Added styles -WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE
		x,
		y,
		WIN_WIDTH,
		WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL
		);
	if (NULL == hWnd)
	{
		return 0;
	}

	g_hWnd = hWnd;

	initialize();

	ShowWindow(hWnd, SW_SHOW);
	SetForegroundWindow(hWnd);
	SetFocus(hWnd);

	//
	//	Message loop.
	//
	while (false == boDone)
	{
		if (PeekMessage(&Msg, NULL, 0, 0, PM_REMOVE))
		{
			if (WM_QUIT == Msg.message)
			{
				boDone = true;
			}
			else
			{
				TranslateMessage(&Msg);
				DispatchMessage(&Msg);
			}
		}
		else
		{
			if (true == g_boActiveWindow)
			{
				if (true == g_boEscapeKeyPressed)
				{
					boDone = true;
				}
				update();
				display();
			}
		}
	}

	uninitialize();

	return((int)Msg.wParam);
}


LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	VOID ToggleFullScreen();
	void InitLight();

	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (0 == HIWORD(wParam))
		{
			g_boActiveWindow = true;
		}
		else
		{
			g_boActiveWindow = false;
		}
		break;


		//case WM_ERASEBKGND:
		//return(0);

	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;


	case WM_CHAR:
		switch (wParam)
		{
		case VK_ESCAPE:
			g_boEscapeKeyPressed = true;
			break;

		case 'f':
		case 'F':
			if (false == g_boFullScreen)
			{
				ToggleFullScreen();
				g_boFullScreen = true;
			}
			else
			{
				ToggleFullScreen();
				g_boFullScreen = false;
			}
			break;

		case 'l':
		case 'L':
			if (FALSE == g_bEnableLight)
			{
				glEnable(GL_LIGHTING);
				g_bEnableLight = TRUE;
			}
			else
			{
				glDisable(GL_LIGHTING);
				g_bEnableLight = FALSE;
			}
			break;

		default:
			break;
		}
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	default:
		break;
	}

	return (DefWindowProc(hWnd, iMsg, wParam, lParam));
}


VOID ToggleFullScreen()
{
	MONITORINFO MonitorInfo;

	if (false == g_boFullScreen)
	{
		g_dwStyle = GetWindowLong(g_hWnd, GWL_STYLE);

		if (g_dwStyle & WS_OVERLAPPEDWINDOW)
		{
			MonitorInfo = { sizeof(MonitorInfo) };

			if (GetWindowPlacement(g_hWnd, &g_WindowPlacementPrev) && GetMonitorInfo(MonitorFromWindow(g_hWnd, MONITORINFOF_PRIMARY), &MonitorInfo))
			{
				SetWindowLong(g_hWnd, GWL_STYLE, g_dwStyle & (~WS_OVERLAPPEDWINDOW));
				SetWindowPos(
					g_hWnd,
					HWND_TOP,
					MonitorInfo.rcMonitor.left,
					MonitorInfo.rcMonitor.top,
					MonitorInfo.rcMonitor.right - MonitorInfo.rcMonitor.left,
					MonitorInfo.rcMonitor.bottom - MonitorInfo.rcMonitor.top,
					SWP_NOZORDER | SWP_FRAMECHANGED
					);
			}
		}
		ShowCursor(FALSE);
	}
	else
	{
		SetWindowLong(g_hWnd, GWL_STYLE, g_dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(g_hWnd, &g_WindowPlacementPrev);
		SetWindowPos(g_hWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);

		ShowCursor(TRUE);
	}
}

VOID initialize()
{
	void uninitialize();
	void InitLight();
	BOOLEAN LoadMeshData();

	HDC hDC;
	BOOLEAN bRes;
	int iPixelFormatIndex;
	PIXELFORMATDESCRIPTOR pfd;

	fopen_s(&g_fpLogFile, LOG_FILE_NAME, "w");
	if (NULL == g_fpLogFile)
	{
		uninitialize();
		return;
	}

	ZeroMemory(&pfd, sizeof(pfd));

	//
	//	Init Pixel format descriptor structure.
	//
	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32;	//	Change 1: for 3d

	g_hdc = GetDC(g_hWnd);

	hDC = GetDC(g_hWnd);

	ReleaseDC(g_hWnd, hDC);

	iPixelFormatIndex = ChoosePixelFormat(g_hdc, &pfd);
	if (0 == iPixelFormatIndex)
	{
		ReleaseDC(g_hWnd, g_hdc);
		g_hdc = NULL;
	}

	if (FALSE == SetPixelFormat(g_hdc, iPixelFormatIndex, &pfd))
	{
		ReleaseDC(g_hWnd, g_hdc);
		g_hdc = NULL;
	}

	g_hRC = wglCreateContext(g_hdc);
	if (NULL == g_hRC)
	{
		ReleaseDC(g_hWnd, g_hdc);
		g_hdc = NULL;
	}

	if (FALSE == wglMakeCurrent(g_hdc, g_hRC))
	{
		wglDeleteContext(g_hRC);
		g_hRC = NULL;
		ReleaseDC(g_hWnd, g_hdc);
		g_hdc = NULL;
	}

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	//+	Change 2 For 3D
	glClearDepth(1.0f);

	glEnable(GL_DEPTH_TEST);

	glDepthFunc(GL_LEQUAL);

	//
	//	Optional.
	//
	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	//-	Change 2 For 3D

	//+	Change For Light
	InitLight();

	//	Read mesh file and load global vectors with appropriate data.
	bRes = LoadMeshData();
	if (FALSE == bRes)
	{
		fprintf(g_fpLogFile, "LoadMeshData() failed.");
		uninitialize();
		return;
	}

	//
	//	Resize.
	//
	resize(WIN_WIDTH, WIN_HEIGHT);
}

void InitLight()
{
	glLightfv(GL_LIGHT0, GL_AMBIENT, g_arrLightAmbient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, g_arrLightDiffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, g_arrLightSpecular);
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);

	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterialSpecular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterialShininess);

	glEnable(GL_LIGHT0);
}


BOOLEAN LoadMeshData()
{
	//void uninitialize();

	//	Open mesh file, name of mesh file can be parameterized.
	fopen_s(&g_fpMeshFile, OBJ_FILE_NAME, "r");
	if (NULL == g_fpMeshFile)
	{
		fprintf(g_fpLogFile, "fopen failed(mesh file)");
		return FALSE;
	}

	//+Seperator string.
	//	String holding space seperator for strtok()
	char *szSepSpace = " ";

	//	String holding forward slash seperator for strtok()
	char *szSepSlash = "/";
	//-Seperator string.

	//+	Token pointers

	//	Character pointer for holding first word in a line.
	char *pszFirstToken = NULL;
	//	Character pointer for holding next word seperated by specified seperator to strtok
	char *pszToken = NULL;
	//-	Token pointers

	//
	//	Array of character pointers to hold string of face entries.
	//	face entries can be variable. In some files they are three
	//	and in some files they are four.
	//
	char *parrFaceTokens[NUM_FACE_TOKENS];
	//	Number of non-null tokens in above vector.
	int iNumOfFaceTokens;

	//	Character pointer for holding string associated with vertex index.
	char *pszTokenIndexVertex = NULL;

	//	Character pointer for holding string associated with texture index.
	char *pszTokenIndexTexture = NULL;

	//	Character pointer for holding string associated with Normal index.
	char *pszTokenIndexNormal = NULL;

	//	While there is a line in a file.
	while (fgets(g_Line, BUFFER_SIZE, g_fpMeshFile))
	{
		//	Bind line to a seperator and get first token
		pszFirstToken = strtok(g_Line, szSepSpace);

		//	If first token indicates vertex data.
		if (strcmp(pszFirstToken, "v") == S_EQUAL)
		{
			//	Create vector of NUM_POINTS_COORDS number of floats to hold coordinates.
			std::vector<float> vecPointCoordinates(NUM_POINTS_COORDS);

			//	Do following NUM_POINTS_COORDS times
			//	1. Get next token
			//	2. feed it to atof to get floating point number out of it
			//	3. add the floating point genearted to vector.
			//	End of loop.
			//	4. At the end of loop, vector is constructed, add it to global vector of vector of floats.i.e. g_vecVertices
			for (int i = 0; i < NUM_POINTS_COORDS; i++)
			{
				vecPointCoordinates[i] = atof(strtok(NULL, szSepSpace));
			}
			g_vecVertices.push_back(vecPointCoordinates);
		}
		//	If first token indicates texture data.
		else if (strcmp(pszFirstToken, "vt") == S_EQUAL)
		{
			std::vector<float> vecTextureCoordinates(NUM_TEXTURE_COORD);

			//	Do following NUM_TEXTURE_COORD times
			//	1. Get next token
			//	2. feed it to atof to get floating point number out of it
			//	3. add the floating point genearted to vector.
			//	End of loop.
			//	4. At the end of loop, vector is constructed, add it to global vector of vector of floats. i.e. g_vecTexture
			for (int i = 0; i < NUM_TEXTURE_COORD; i++)
			{
				vecTextureCoordinates[i] = atof(strtok(NULL, szSepSpace));
			}
			g_vecTexture.push_back(vecTextureCoordinates);
		}
		//	If first token indicates normal data.
		else if (strcmp(pszFirstToken, "vn") == S_EQUAL)
		{
			std::vector<float> vecNormalCoordinates(NUM_NORMAL_COORD);

			//	Do following NUM_NORMAL_COORD times
			//	1. Get next token
			//	2. feed it to atof to get floating point number out of it
			//	3. add the floating point genearted to vector.
			//	End of loop.
			//	4. At the end of loop, vector is constructed, add it to global vector of vector of floats. i.e. g_vecNormals
			for (int i = 0; i < NUM_NORMAL_COORD; i++)
			{
				vecNormalCoordinates[i] = atof(strtok(NULL, szSepSpace));
			}
			g_vecNormals.push_back(vecNormalCoordinates);
		}
		//	If first token indicates face data.
		else if (strcmp(pszFirstToken, "f") == S_EQUAL)
		{
			//	Define three vector of integers with length 3 to hold indices of
			//	triangles positional co-ordinates, texture coordinates, and normal coordinates
			//	in g_vecVertices, g_vecTexture, g_vecNormals respectively.
			std::vector<int> vecVertexIndicesTriangle(3);
			std::vector<int> vecVertexIndicesTexture(3);
			std::vector<int> vecVertexIndicesNormal(3);

			//Initialize all char pointers in face tokens to NULL.
			memset(parrFaceTokens, 0, NUM_FACE_TOKENS);

			//	Extract three fields of information in face parrFaceTokens
			//	and increment iNumOfFaceTokens accordingly.
			iNumOfFaceTokens = 0;
			while (pszToken = strtok(NULL, szSepSpace))
			{
				if (strlen(pszToken) < 3)
				{
					fprintf(g_fpLogFile, "Invalid token count in face data (%s)", g_Line);
					break;
				}
				parrFaceTokens[iNumOfFaceTokens] = pszToken;
				iNumOfFaceTokens++;
			}

			//	Every face data entry is going to have minimum three fields
			//	therefore, construct a triangle out of it with
			//	1. Triangle coordinate data
			//	2. Texture coordinate data
			//	3. Normal coordinate data
			//	4. Put the data in vecVertexIndicesTriangle, vecVertexIndicesTexture, vecVertexIndicesNormal,
			//		Vectors will be constructed at the end of the loop.
			for (int i = 0; i != NUM_FACE_TOKENS; i++)
			{
				//61/1/1 65/2/1 49/3/1 
				vecVertexIndicesTriangle[i] = atoi(strtok(parrFaceTokens[i], szSepSlash));	//61 65 49 
				vecVertexIndicesTexture[i] = atoi(strtok(NULL, szSepSlash));	//1 2 3 
				vecVertexIndicesNormal[i] = atoi(strtok(NULL, szSepSlash));	//1 1 1 
			}
			//	Add constructed vectors to global vectors.
			g_vecFaceTriangles.push_back(vecVertexIndicesTriangle);
			g_vecFaceTexture.push_back(vecVertexIndicesTexture);
			g_vecFaceNormals.push_back(vecVertexIndicesNormal);
		}

		//	initialize g_Line buffer to NULL.
		memset(g_Line, '\0', BUFFER_SIZE);
	}

	//	Close mesh file and file pointer NULL.
	fclose(g_fpMeshFile);
	g_fpMeshFile = NULL;

	//	Log vertex, texture and face data.
	fprintf(g_fpLogFile, "Vertices : %u, Texture : %u, Normals : %u, Triangle faces : %u\n", g_vecVertices.size(), g_vecTexture.size(), g_vecNormals.size(), g_vecFaceTriangles.size());

	return TRUE;
}


void update()
{
	g_rotate = g_rotate + UPDATE_ANGLE;
	if (g_rotate >= 360.0f)
	{
		g_rotate = 0.0f;
	}
}


VOID display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	//	Change 3 for 3D

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//
	//	Model transformation.
	//
	glTranslatef(0.0f, 0.0f, -3.0f);
	glRotatef(g_rotate, 0.0f, 1.0f, 0.0f);

	//	Keep couter-clockwise winding of vertices of geometry.
	glFrontFace(GL_CCW);

	//	Set polygon mode mentioning front and back faces and GL_LINE
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	//	For every face index maintained in triangular form in g_vecFaceTriangles
	//	Do following
	//	1. Set geometry primitive to GL_TRIANGLES
	//	2. Extract triangles from outer loop index.
	//	3. For every point of traingle
	//		a. Calculate the index in g_vecVertices
	//		b. Calculate x, y , z co-ordinates of point.
	//		c. Send to glVertex3f
	//	Note : In step (a), We have to substract g_vecFaceTriangles[i][j] by one because
	//			in mesh file, indexing starts from 1 and in case of arrays/vectors
	//			indexing starts from 0.
	for (int i = 0; i != g_vecFaceTriangles.size(); ++i)
	{
		glBegin(GL_TRIANGLES);
		for (int j = 0; j != g_vecFaceTriangles[i].size(); j++)
		{
			int vi = g_vecFaceTriangles[i][j] - 1;
			int ni = g_vecFaceNormals[i][j] - 1;
			glNormal3f(g_vecNormals[ni][0], g_vecNormals[ni][1], g_vecNormals[ni][2]);
			glVertex3f(g_vecVertices[vi][0], g_vecVertices[vi][1], g_vecVertices[vi][2]);
		}
		glEnd();
	}

	SwapBuffers(g_hdc);
}


VOID resize(int iWidth, int iHeight)
{
	if (0 == iHeight)
	{
		iHeight = 1;
	}

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	//
	//	znear and zfar must positive.
	//
	if (iWidth <= iHeight)
	{
		gluPerspective(45, (GLfloat)iHeight / (GLfloat)iWidth, 0.1f, 100.0f);
	}
	else
	{
		gluPerspective(45, (GLfloat)iWidth / (GLfloat)iHeight, 0.1f, 100.0f);
	}

	glViewport(0, 0, iWidth, iHeight);
}

VOID uninitialize()
{
	if (true == g_boFullScreen)
	{
		SetWindowLong(g_hWnd, GWL_STYLE, g_dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(g_hWnd, &g_WindowPlacementPrev);
		SetWindowPos(g_hWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);

		ShowCursor(TRUE);
	}

	wglMakeCurrent(NULL, NULL);
	wglDeleteContext(g_hRC);
	g_hRC = NULL;

	ReleaseDC(g_hWnd, g_hdc);
	g_hdc = NULL;

	DestroyWindow(g_hWnd);
	g_hWnd = NULL;

	if (g_fpLogFile)
	{
		fclose(g_fpLogFile);
		g_fpLogFile = NULL;
	}
}
