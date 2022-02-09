#include<windows.h>
#include<gl/GL.h>
#include<gl/GLU.h>
#include<stdio.h>
#include<stdlib.h>

//C++ Header
#include<vector>

#pragma comment(lib,"User32.lib")
#pragma comment(lib,"GDI32.lib")
#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"glu32.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

#define NR_POINT_COORDS			3	//Number of point coordinates
#define NR_TEXTURE_COORDS		2	//Number of texture coordinates
#define NR_NORMAL_COORDS		3	//Number of normal coordinates
#define NR_FACE_TOKENS			3	//Minimum number of entries in face data

#define MONKEY_X_TRANSLATE		0.0f	//X-translation of monkeyhead
#define MONKEY_Y_TRANSLATE		0.0f	//Y-translation of monkeyhead
#define MONKEY_Z_TRANSLATE		-8.0f	//Z-translation of monkeyhead

#define MONKEY_X_SCALE_FACTOR	1.5f	//X-scale factor of monkeyhead
#define MONKEY_Y_SCALE_FACTOR	1.5f	//Y-scale factor of monkeyhead
#define MONKEY_Z_SCALE_FACTOR	1.5f	//Z-scale factor of monkeyhead

#define START_ANGLE_POS			0.0f	//Marks beginning angle position of rotation
#define END_ANGLE_POS			360.0f	//Marks terminating angle position rotation
#define MONKEY_ANGLE_INCREMENT	0.1f	//Increment angle for monkeyhead

#define BUFFER_SIZE				256
#define S_EQUAL					0

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

FILE *gpFile;
HWND ghwnd;
HDC ghdc;
HGLRC ghrc;
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
bool gbActiveWindow = false;
bool gbFullscreen = false;
bool gbIsEscapeKeyPressed = false;
bool gbPolygonModeFlag = false;

GLfloat g_rotate;

//Vector of vector of floats to hold vertex data
std::vector<std::vector<float>> g_vertices;

//Vector of vector of floats to hold texture data
std::vector<std::vector<float>> g_texture;

//Vector of vector of floats to hold normal data
std::vector<std::vector<float>> g_normals;

//Vector of vector of int to hold index data in g_vertices
std::vector<std::vector<int>> g_face_tri,g_face_texture,g_face_normals;

//Handle to mesh file
FILE *g_fp_meshfile = NULL;

//Handle a Log file
FILE *g_fp_logfile = NULL;

//Hold line in a file
char line[BUFFER_SIZE];

GLfloat light_ambient[] = { 1.0f,1.0f,1.0f,1.0f };
GLfloat light_diffuse[] = { 1.0f,1.0f,1.0f,1.0f };
GLfloat light_specular[] = { 1.0f,0.0f,0.0f,1.0f };
GLfloat light_position[] = { 0.0f,0.0f,1.0f,0.0f };

GLfloat material_specular[] = { 1.0f,1.0f,1.0f,1.0f };
GLfloat material_shininess[] = { 50.0f,50.0f,50.0f,50.0f };

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	void initialize(void);
	void display(void);
	void update(void);
	void uninitialize(int);
	void ToggleFullscreen(void);

	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("My App");
	bool bDone = false;

	if (fopen_s(&gpFile, "Log.txt", "w") != NULL)
	{
		MessageBox(NULL, TEXT("Cannot Create Log File !!!"), TEXT("Error"), MB_OK);
		exit(EXIT_FAILURE);
	}
	else
		fprintf(gpFile, "Log File Created Successfully...\n");

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

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, TEXT("Monkey Head"), WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE, 100, 100, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);
	if (hwnd == NULL)
	{
		fprintf(gpFile, "Cannot Create Window...\n");
		uninitialize(1);
	}

	ghwnd = hwnd;

	ShowWindow(hwnd, iCmdShow);
	SetFocus(hwnd);
	SetForegroundWindow(hwnd);

	initialize();

	ToggleFullscreen();

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
	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			gbActiveWindow = true;
		else
			gbActiveWindow = false;
		break;
	case WM_CREATE:
		break;
	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;
	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			gbIsEscapeKeyPressed = true;
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

		case 0x57:
			if (gbPolygonModeFlag == false)
			{
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
				gbPolygonModeFlag = true;
			}
			else
			{
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
				gbPolygonModeFlag = false;
			}
			break;
		}
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void initialize(void)
{
	void resize(int, int);
	void uninitialize(int);
	void LoadMeshData(void);
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

	g_fp_logfile = fopen("MONKEYHEADLOADER.LOG", "w");
	if (!g_fp_logfile)
		uninitialize(1);

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
		fprintf(gpFile, "GetDC() Failed.\n");
		uninitialize(1);
	}

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0)
	{
		fprintf(gpFile, "ChoosePixelFormat() Failed.\n");
		uninitialize(1);
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		fprintf(gpFile, "SetPixelFormat() Failed.\n");
		uninitialize(1);
	}

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		fprintf(gpFile, "wglCreateContext() Failed.\n");
		uninitialize(1);
	}

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		fprintf(gpFile, "wglMakeCurrent() Failed");
		uninitialize(1);
	}

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT , GL_NICEST);
	glFrontFace(GL_CCW);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	glEnable(GL_LIGHTING);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glMaterialfv(GL_FRONT, GL_SPECULAR, material_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, material_shininess);
	glEnable(GL_LIGHT0);
	

	//Read mesh file and load global vectors with appropriate data
	LoadMeshData();

	resize(WIN_WIDTH, WIN_HEIGHT);
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(MONKEY_X_TRANSLATE, MONKEY_Y_TRANSLATE, MONKEY_Z_TRANSLATE);
	glRotatef(g_rotate, 0.0f, 1.0f, 0.0f);
	glScalef(MONKEY_X_SCALE_FACTOR, MONKEY_Y_SCALE_FACTOR, MONKEY_Z_SCALE_FACTOR);
	//glColor3f(1.0f, 0.0f, 0.0f);
	for (int i = 0; i != g_face_tri.size(); ++i)
	{
		glBegin(GL_TRIANGLES);
		for (int j = 0; j != g_face_tri[i].size(); j++)
		{
			int vi = g_face_tri[i][j] - 1;
			int ni = g_face_normals[i][j] - 1;
			glNormal3f(g_normals[ni][0], g_normals[ni][1], g_normals[ni][2]);
			glVertex3f(g_vertices[vi][0], g_vertices[vi][1], g_vertices[vi][2]);
		}
		glEnd();
	}
	
	SwapBuffers(ghdc);
}

void LoadMeshData(void)
{
	void uninitialize(int);

	//Open Mesh file, name of mesh file can be parameterized
	g_fp_meshfile = fopen("MonkeyHead.OBJ", "r");
	if (!g_fp_meshfile)
		uninitialize(1);

	//Separator strings
	//String holding space separator for strtok
	char *sep_space = " ";
	//String holding forward slash separator for strtok
	char *sep_fslash = "/";

	//Token pointers
	//Character pointer for holding first word in a line
	char *first_token = NULL;
	//Character pointer for holding next word separated by
	//Specified separator to strtok
	char *token = NULL;

	//Array of character pointers to hold strings of face entries
	//Face entries can be variable. In some files they are three and in some files they are four.
	char *face_tokens[NR_FACE_TOKENS];
	//Number of non-null tokens in the above vector
	int nr_tokens;

	//Character pointer holding string associated with
	//vertex index
	char *token_vertex_index = NULL;
	//Character pointer holding string associated with
	//texture index
	char *token_texture_index = NULL;
	//Character pointer holding string associated with
	//normal index
	char *token_normal_index = NULL;

	//While there is a line in file
	while (fgets(line, BUFFER_SIZE, g_fp_meshfile) != NULL)
	{
		//Bind line to a separator and get first token
		first_token = strtok(line, sep_space);

		//If first token indicates vertex data
		if (strcmp(first_token, "v") == S_EQUAL)
		{
			/*Create a vector of NR_POINT_COORDS number of floats
			to hold coordinates*/
			std::vector<float>vec_point_coord(NR_POINT_COORDS);

			//Do following NR_POINTS_COORDS time
			//S1. Get next token
			//S2. Feed it to atof to get floating point number out of it
			//S3. And the floating point number generated to vector
			//End of loop
			/*S4. At the end of loop vector is constructed, add it to
			global vector of vector of floats, named g_vertices*/
			for (int i = 0; i != NR_POINT_COORDS; i++)
				vec_point_coord[i] = atof(strtok(NULL, sep_space));	//S1, S2, S3
			g_vertices.push_back(vec_point_coord);
		}

		//If first token indicates texture data
		else if (strcmp(first_token, "vt") == S_EQUAL)
		{
			/*Create a vector of NR_TEXTURE_COORDS number of floats
			to hold coordinates*/
			std::vector<float>vec_texture_coord(NR_TEXTURE_COORDS);

			//Do following NR_TEXTURE_COORDS time
			//S1. Get next token
			//S2. Feed it to atof to get floating point number out of it
			//S3. And the floating point number generated to vector
			//End of loop
			/*S4. At the end of loop vector is constructed, add it to
			global vector of vector of floats, named g_texture*/
			for (int i = 0; i != NR_TEXTURE_COORDS; i++)
				vec_texture_coord[i] = atof(strtok(NULL, sep_space));	//S1, S2, S3
			g_texture.push_back(vec_texture_coord);
		}

		//If first token indicates normal data
		else if (strcmp(first_token, "vn") == S_EQUAL)
		{
			/*Create a vector of NR_NORMAL_COORDS number of floats
			to hold coordinates*/
			std::vector<float>vec_normal_coord(NR_NORMAL_COORDS);

			//Do following NR_NORMAL_COORDS time
			//S1. Get next token
			//S2. Feed it to atof to get floating point number out of it
			//S3. And the floating point number generated to vector
			//End of loop
			/*S4. At the end of loop vector is constructed, add it to
			global vector of vector of floats, named g_normal*/
			for (int i = 0; i != NR_NORMAL_COORDS; i++)
				vec_normal_coord[i] = atof(strtok(NULL, sep_space));	//S1, S2, S3
			g_normals.push_back(vec_normal_coord);
		}

		//If first token indicates face data
		else if (strcmp(first_token, "f") == S_EQUAL)
		{
			/*Define three vector of integers with length 3 to hold indices of
			triangle's positional coordinates, texture coordinates, and normal
			coordinates in g_vertices,g_textures and g_normals resp*/
			std::vector<int> triangle_vertex_indices(3), texture_vertex_indices(3), normal_vertex_indices(3);

			//Initialize all char pointers in face_tokens to NULL
			memset((void*)face_tokens, 0, NR_FACE_TOKENS);

			//Extract three fields of information in face_tokens
			//and increment nr_tokens accordingly
			nr_tokens = 0;
			while (token = strtok(NULL, sep_space))
			{
				if (strlen(token) < 3)
					break;
				face_tokens[nr_tokens] = token;
				nr_tokens++;
			}

			for (int i = 0; i != NR_FACE_TOKENS; ++i)
			{
				token_vertex_index = strtok(face_tokens[i], sep_fslash);
				token_texture_index = strtok(NULL, sep_fslash);
				token_normal_index = strtok(NULL, sep_fslash);
				triangle_vertex_indices[i] = atoi(token_vertex_index);
				texture_vertex_indices[i] = atoi(token_texture_index);
				normal_vertex_indices[i] = atoi(token_normal_index);
			}

			//Add constructed vectors to global face vectors
			g_face_tri.push_back(triangle_vertex_indices);
			g_face_texture.push_back(texture_vertex_indices);
			g_face_normals.push_back(normal_vertex_indices);
		}

		//Initialize line buffer to NULL
		memset((void*)line, (int)'\0', BUFFER_SIZE);
	}

	//Close meshfile and make file pointer NULL
	fclose(g_fp_meshfile);
	g_fp_meshfile = NULL;

	//Log Vertex, Texture and face data in log file.
	fprintf(g_fp_logfile, "g_vertices : %llu g_texture : %llu g_normals:%llu g_face_tri : %llu\n", g_vertices.size(), g_texture.size(), g_normals.size(), g_face_tri.size());
}

void update(void)
{
	g_rotate = g_rotate + MONKEY_ANGLE_INCREMENT;
	if (g_rotate >= END_ANGLE_POS)
		g_rotate = START_ANGLE_POS;
}

void resize(int width, int height)
{
	if (height == 0)
		height = 1;
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.1f,100.0f);
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
				SetWindowPos(ghwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
	}

	else
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}
}

void uninitialize(int i_Exit_Flag)
{
	if (gbFullscreen == false)
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
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

	if (g_fp_logfile)
	{
		fclose(g_fp_logfile);
		g_fp_logfile = NULL;
	}

	if (i_Exit_Flag == 0)
		fprintf(gpFile, "Log File Closed Successfully...\n");
	else
		fprintf(gpFile, "Log File Closed Erroniously...\n");

	fclose(gpFile);
	gpFile = NULL;

	DestroyWindow(ghwnd);
}