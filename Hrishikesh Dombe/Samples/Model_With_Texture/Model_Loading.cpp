#include<windows.h>
#include<windowsx.h>
#include<C:\glew\include\GL\glew.h>
#include<gl/GL.h>
#include<stdio.h>
#include"vmath.h"
#include<vector>
#include<stdlib.h>
#include"Camera_2.h"
#include"Read_Mtl.h"
#include"Obj_Loader1.h"
#include"Arrange_Material.h"

#pragma comment(lib,"User32.lib")
#pragma comment(lib,"GDI32.lib")
#pragma comment(lib,"C:\\glew\\lib\\Release\\x64\\glew32.lib")
#pragma comment(lib,"opengl32.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

//using namespace vmath;

enum
{
	HAD_ATTRIBUTE_POSITION = 0,
	HAD_ATTRIBUTE_COLOR,
	HAD_ATTRIBUTE_NORMAL,
	HAD_ATTRIBUTE_TEXTURE0,
};

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

#define MODEL_X_TRANSLATE		0.0f	//X-translation of MODEL
#define MODEL_Y_TRANSLATE		-1.0f	//Y-translation of MODEL
#define MODEL_Z_TRANSLATE		-20.0f	//Z-translation of MODEL

#define MODEL_X_SCALE_FACTOR	1.5f	//X-scale factor of MODEL
#define MODEL_Y_SCALE_FACTOR	1.5f	//Y-scale factor of MODEL
#define MODEL_Z_SCALE_FACTOR	1.5f	//Z-scale factor of MODEL

#define START_ANGLE_POS			0.0f	//Marks beginning angle position of rotation
#define END_ANGLE_POS			45.0f	//Marks terminating angle position rotation
#define MODEL_ANGLE_INCREMENT	1.2f	//Increment angle for MODEL

FILE *gpFile;
HWND ghwnd;
HDC ghdc;
HGLRC ghrc;
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
bool gbActiveWindow = false;
bool gbFullscreen = false;
bool gbIsEscapeKeyPressed = false;

GLuint gVertexShaderObject;
GLuint gFragmentShaderObject;
GLuint gShaderProgramObject;

GLuint gVao_Project_Model, gVao_Helicopter;
GLuint gVbo_Position, gVbo_Normal, gVbo_Texture;

GLuint gModelMatrixUniform, gViewMatrixUniform, gProjectionMatrixUniform, gMVPUniform;
GLuint gLKeyPressedUniform;

GLuint gLaUniform, gLdUniform, gLsUniform;
GLuint gLightPositionUniform;

GLuint gKaUniform, gKdUniform, gKsUniform, gAlphaUniform;
GLuint gMaterialShininessUniform;
GLuint gTextureSamplerUniform, gTextureActiveUniform, gTexture;

GLfloat gAngle_Cube;

glm::mat4 gPerspectiveProjectionMatrix;

bool gbAnimate = false;
bool gbLight = false;
bool gbIsAKeyPressed = false;
bool gbIsLKeyPressed = false;

GLfloat lightAmbient[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat lightDiffuse[] = { 1.0f,1.0f,1.0f,1.0f };
GLfloat lightSpecular[] = { 1.0f,1.0f,1.0f,1.0f };
GLfloat lightPosition[] = { 0.0f,20.0f,20.0f,0.0f };

GLfloat materialAmbient[] = { 0.25f,0.25f,0.25f,1.0f };
GLfloat materialDiffuse[] = { 0.4f,0.4f,0.4f,1.0f };
GLfloat materialSpecular[] = { 0.774597f,0.774597f,0.774597f,1.0f };
GLfloat materialShininess = 0.6f * 128.0f;

GLfloat g_rotate, g_rotate_2nd_turn = 45.0f, g_rotate_3rd_turn = 0.0f, g_rotate_4th_turn = -45.0f;

std::vector<float> gv_vertices_1, gv_textures_1, gv_normals_1;
std::vector<float> gv_vertices_2, gv_textures_2, gv_normals_2;
std::vector<int> gv_face_tri_1, gv_face_textures_1, gv_face_normals_1;

int count_of_vertices_car_1;

std::vector<material>arr_material;
std::vector<Mesh_Data>mesh_data;
std::vector<material>arr_material2;
std::vector<Mesh_Data>mesh_data2;

char MtlLib[256];
char MtlLib2[256];

FRAG_Camera2::Camera Scene3_Camera;
bool gbFirstMouse = true;

float lastX = WIN_WIDTH / 2.0f;
float lastY = WIN_HEIGHT / 2.0f;
float midX = lastX;
float midY = lastY;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	void initialize(void);
	void ToggleFullscreen(void);
	void display(void);
	void update(void);
	void uninitialize(int);
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

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, TEXT("OpenGLPP : 3D Rotation"), WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE, 100, 100, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);
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

	MessageBox(hwnd, TEXT("After init"), TEXT("MSG"), MB_OK);

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
				//if (gbAnimate == true)
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
	int xPos;
	int yPos;
	void MouseMovement(double xPos, double yPos);
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

	case WM_MOUSEMOVE:
		xPos = GET_X_LPARAM(lParam);
		yPos = GET_Y_LPARAM(lParam);
		MouseMovement((double)xPos, (double)yPos);
		break;

	case WM_KEYDOWN:
		if (GetAsyncKeyState(VK_DOWN))
		{
			Scene3_Camera.ProcessKeyboard(FRAG_Camera2::BACKWARD, 0.0f);
		}
		else if (GetAsyncKeyState(VK_UP))
		{
			Scene3_Camera.ProcessKeyboard(FRAG_Camera2::FORWARD, 0.0f);
		}
		else if (GetAsyncKeyState(VK_LEFT))
		{
			Scene3_Camera.ProcessKeyboard(FRAG_Camera2::LEFT, 0.0f);
		}
		else if (GetAsyncKeyState(VK_RIGHT))
		{
			Scene3_Camera.ProcessKeyboard(FRAG_Camera2::RIGHT, 0.0f);
		}
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

		case VK_TAB:
			if (gbIsAKeyPressed == false)
			{
				gbAnimate = true;
				gbIsAKeyPressed = true;
			}
			else
			{
				gbAnimate = false;
				gbIsAKeyPressed = false;
			}
			break;

		case 0x4C:
			if (gbIsLKeyPressed == false)
			{
				gbLight = true;
				gbIsLKeyPressed = true;
			}
			else
			{
				gbLight = false;
				gbIsLKeyPressed = false;
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

void MouseMovement(double xPos, double yPos)
{
	if (gbFirstMouse)
	{
		lastX = (GLfloat)xPos;
		lastY = (GLfloat)yPos;
		gbFirstMouse = false;
	}

	GLfloat xOffset = (GLfloat)xPos - (GLfloat)lastX;
	GLfloat yOffset = (GLfloat)lastY - (GLfloat)yPos;
	lastX = (GLfloat)xPos;
	lastY = (GLfloat)yPos;

	Scene3_Camera.ProcessMouseMovement(xOffset, yOffset);
	if (lastX != (float)midX || lastY != (float)midY)
		SetCursorPos(midX, midY);

	lastX = (float)midX;
	lastY = (float)midY;
}

void initialize(void)
{
	void resize(int, int);
	void uninitialize(int);
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

	GLenum glew_error = glewInit();
	if (glew_error != GLEW_OK)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	//Project Model
	MessageBox(ghwnd, TEXT("Before LoadMeshData"), TEXT("MSG"), MB_OK);
	LoadMeshData("Final_Project_Model.obj", gv_vertices_1, gv_textures_1, gv_normals_1, mesh_data, MtlLib);
	MessageBox(ghwnd, TEXT("After LoadMeshData 1"), TEXT("MSG"), MB_OK);
	LoadMaterialData(MtlLib, arr_material);
	MessageBox(ghwnd, TEXT("After LoadMaterialData"), TEXT("MSG"), MB_OK);
	Rearrange_Material_Data(mesh_data, arr_material);
	MessageBox(ghwnd, TEXT("After Rearrange_Material_Data "), TEXT("MSG"), MB_OK);

	//Helicopter
	MessageBox(ghwnd, TEXT("Before LoadMeshData"), TEXT("MSG"), MB_OK);
	LoadMeshData("Helicopter/Helicopter_Blades.obj", gv_vertices_2, gv_textures_2, gv_normals_2, mesh_data2, MtlLib2);
	MessageBox(ghwnd, TEXT("After LoadMeshData 1"), TEXT("MSG"), MB_OK);
	LoadMaterialData(MtlLib2, arr_material2);
	MessageBox(ghwnd, TEXT("After LoadMaterialData"), TEXT("MSG"), MB_OK);
	Rearrange_Material_Data(mesh_data2, arr_material2);
	MessageBox(ghwnd, TEXT("After Rearrange_Material_Data "), TEXT("MSG"), MB_OK);

	//Vertex Shader
	gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *vertexShaderSourceCode =
		"#version 450" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vNormal;" \
		"in vec2 vTexture0_Coord;" \
		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"uniform int u_lighting_enabled;" \
		"uniform vec4 u_light_position;" \
		"out vec3 transformed_normals;" \
		"out vec3 light_direction;" \
		"out vec3 viewer_vector;" \
		"out vec2 out_texture0_coord;" \
		"void main(void)" \
		"{" \
		"if(u_lighting_enabled==1)" \
		"{" \
		"vec4 eye_coordinates = u_view_matrix*u_model_matrix*vPosition;" \
		"transformed_normals = mat3(u_view_matrix*u_model_matrix)*vNormal;" \
		"light_direction = vec3(u_light_position)-eye_coordinates.xyz;" \
		"viewer_vector = -eye_coordinates.xyz;" \
		"}" \
		"gl_Position = u_projection_matrix*u_view_matrix*u_model_matrix*vPosition;" \
		"out_texture0_coord = vTexture0_Coord;" \
		"}";

	glShaderSource(gVertexShaderObject, 1, (const GLchar **)&vertexShaderSourceCode, NULL);

	glCompileShader(gVertexShaderObject);
	GLint iInfoLogLength = 0;
	GLint iShaderCompiledStatus = 0;
	char *szInfoLog = NULL;

	glGetShaderiv(gVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(gVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gVertexShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Vertex Shader Compilation Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize(1);
				exit(0);
			}
		}
	}

	//Fragment Shader
	gFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *fragmentShaderSourceCode =
		"#version 450" \
		"\n" \
		"in vec3 transformed_normals;" \
		"in vec3 light_direction;" \
		"in vec3 viewer_vector;" \
		"in vec2 out_texture0_coord;" \
		"out vec4 FragColor;" \
		"uniform vec3 u_La;" \
		"uniform vec3 u_Ld;" \
		"uniform vec3 u_Ls;" \
		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_material_shininess;" \
		"uniform int u_lighting_enabled;" \
		"uniform float u_alpha;" \
		"uniform sampler2D u_texture0_sampler;"\
		"uniform int u_is_texture;" \
		"vec4 Final_Texture;" \
		"vec4 Temp_Output;" \
		"void main(void)" \
		"{" \
		"vec3 phong_ads_color;" \
		"if(u_lighting_enabled == 1)" \
		"{" \
		"vec3 normalized_transformed_normals = normalize(transformed_normals);" \
		"vec3 normalized_light_direction = normalize(light_direction);" \
		"vec3 normalized_viewer_vector = normalize(viewer_vector);" \
		"vec3 ambient = u_La * u_Ka;" \
		"float tn_dot_ld = max(dot(normalized_transformed_normals,normalized_light_direction),0.0);" \
		"vec3 diffuse = u_Ld * u_Kd * tn_dot_ld;" \
		"vec3 reflection_vector = reflect(-normalized_light_direction,normalized_transformed_normals);" \
		"vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector,normalized_viewer_vector),0.0),u_material_shininess);" \
		"phong_ads_color = ambient + diffuse + specular;" \
		"}" \
		"else" \
		"{" \
		"phong_ads_color = vec3(1.0f,1.0f,1.0f);" \
		"}" \
		"if(u_is_texture == 1)" \
		"{" \
		"Final_Texture = texture(u_texture0_sampler,out_texture0_coord);" \
		"Temp_Output = vec4(phong_ads_color,u_alpha) * Final_Texture;" \
		"FragColor = Temp_Output;" \
		"}" \
		"else" \
		"{" \
		"FragColor = vec4(phong_ads_color,u_alpha);" \
		"}" \
		"}";

	glShaderSource(gFragmentShaderObject, 1, (const GLchar **)&fragmentShaderSourceCode, NULL);

	glCompileShader(gFragmentShaderObject);

	glGetShaderiv(gFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(gFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gFragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Fragment Shader Compilation Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize(1);
				exit(0);
			}
		}
	}

	//Shader Program
	gShaderProgramObject = glCreateProgram();

	glAttachShader(gShaderProgramObject, gVertexShaderObject);

	glAttachShader(gShaderProgramObject, gFragmentShaderObject);

	glBindAttribLocation(gShaderProgramObject, HAD_ATTRIBUTE_POSITION, "vPosition");

	glBindAttribLocation(gShaderProgramObject, HAD_ATTRIBUTE_NORMAL, "vNormal");

	glBindAttribLocation(gShaderProgramObject, HAD_ATTRIBUTE_TEXTURE0, "vTexture0_Coord");

	glLinkProgram(gShaderProgramObject);

	GLint iShaderProgramLinkStatus = 0;

	glGetProgramiv(gShaderProgramObject, GL_LINK_STATUS, &iShaderProgramLinkStatus);
	if (iShaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Shader Program Link Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize(1);
				exit(0);
			}
		}
	}

	gModelMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_model_matrix");
	gViewMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_view_matrix");
	gProjectionMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_projection_matrix");

	gLKeyPressedUniform = glGetUniformLocation(gShaderProgramObject, "u_lighting_enabled");

	gLaUniform = glGetUniformLocation(gShaderProgramObject, "u_La");
	gLdUniform = glGetUniformLocation(gShaderProgramObject, "u_Ld");
	gLsUniform = glGetUniformLocation(gShaderProgramObject, "u_Ls");

	gLightPositionUniform = glGetUniformLocation(gShaderProgramObject, "u_light_position");

	gKaUniform = glGetUniformLocation(gShaderProgramObject, "u_Ka");
	gKdUniform = glGetUniformLocation(gShaderProgramObject, "u_Kd");
	gKsUniform = glGetUniformLocation(gShaderProgramObject, "u_Ks");
	gAlphaUniform = glGetUniformLocation(gShaderProgramObject, "u_alpha");

	gMaterialShininessUniform = glGetUniformLocation(gShaderProgramObject, "u_material_shininess");

	gTextureSamplerUniform = glGetUniformLocation(gShaderProgramObject, "u_texture0_sampler");

	gTextureActiveUniform = glGetUniformLocation(gShaderProgramObject, "u_is_texture");

	/*****************VAO For Cube*****************/
	glGenVertexArrays(1, &gVao_Project_Model);
	glBindVertexArray(gVao_Project_Model);

	/*****************Cube Position****************/
	glGenBuffers(1, &gVbo_Position);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Position);
	glBufferData(GL_ARRAY_BUFFER, gv_vertices_1.size() * sizeof(float), &gv_vertices_1[0], GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/*******************Texture******************/
	glGenBuffers(1, &gVbo_Texture);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Texture);
	glBufferData(GL_ARRAY_BUFFER, gv_textures_1.size() * sizeof(float), &gv_textures_1[0], GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_TEXTURE0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/*****************Cube Color****************/
	glGenBuffers(1, &gVbo_Normal);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Normal);
	glBufferData(GL_ARRAY_BUFFER, gv_normals_1.size() * sizeof(float), &gv_normals_1[0], GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	/*****************VAO For Helicopter*****************/
	glGenVertexArrays(1, &gVao_Helicopter);
	glBindVertexArray(gVao_Helicopter);

	/*****************Cube Position****************/
	glGenBuffers(1, &gVbo_Position);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Position);
	glBufferData(GL_ARRAY_BUFFER, gv_vertices_2.size() * sizeof(float), &gv_vertices_2[0], GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/*******************Texture******************/
	glGenBuffers(1, &gVbo_Texture);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Texture);
	glBufferData(GL_ARRAY_BUFFER, gv_textures_2.size() * sizeof(float), &gv_textures_2[0], GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_TEXTURE0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/*****************Cube Color****************/
	glGenBuffers(1, &gVbo_Normal);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Normal);
	glBufferData(GL_ARRAY_BUFFER, gv_normals_2.size() * sizeof(float), &gv_normals_2[0], GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_TEXTURE_2D);

	glClearColor(0.75f, 0.75f, 0.75f, 0.0f);

	gPerspectiveProjectionMatrix = glm::mat4(1.0f);

	resize(WIN_WIDTH, WIN_HEIGHT);
}

void display(void)
{
	//MessageBox(NULL, TEXT("IN DISPLAY1"), TEXT("MSG"), MB_OK);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//Use Shader Program Object
	glUseProgram(gShaderProgramObject);

	if (gbLight == true)
	{
		glUniform1i(gLKeyPressedUniform, 1);

		glUniform3fv(gLaUniform, 1, lightAmbient);
		glUniform3fv(gLdUniform, 1, lightDiffuse);
		glUniform3fv(gLsUniform, 1, lightSpecular);
		glUniform4fv(gLightPositionUniform, 1, lightPosition);

		glUniform3fv(gKaUniform, 1, materialAmbient);
		glUniform3fv(gKdUniform, 1, materialDiffuse);
		glUniform3fv(gKsUniform, 1, materialSpecular);
		glUniform1f(gMaterialShininessUniform, materialShininess);
		glUniform1f(gAlphaUniform, 1.0f);
	}
	else
	{
		glUniform1i(gLKeyPressedUniform, 0);
	}

	glm::mat4 modelMatrix = glm::mat4(1.0f);
	glm::mat4 viewMatrix = glm::mat4(1.0f);
	glm::mat4 scaleMatrix = glm::mat4(1.0f);
	glm::mat4 rotationMatrix = glm::mat4(1.0f);
	glm::mat4 translationMatrix = glm::mat4(1.0f);

	viewMatrix = Scene3_Camera.GetViewMatrix();

	modelMatrix = glm::translate(modelMatrix, glm::vec3(MODEL_X_TRANSLATE, MODEL_Y_TRANSLATE-0.1f, MODEL_Z_TRANSLATE));

	//rotationMatrix = glm::rotate(rotationMatrix, glm::radians(g_rotate), glm::vec3(1.0f, 0.0f, 0.0f));
	//modelMatrix = modelMatrix*rotationMatrix;

	rotationMatrix = glm::rotate(rotationMatrix, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	modelMatrix = modelMatrix*rotationMatrix;

	//rotationMatrix = glm::rotate(rotationMatrix,glm::radians(g_rotate), glm::vec3(0.0f, 0.0f, 1.0f));
	//modelMatrix = modelMatrix*rotationMatrix;

	glUniformMatrix4fv(gModelMatrixUniform, 1, GL_FALSE, glm::value_ptr(modelMatrix));

	glUniformMatrix4fv(gViewMatrixUniform, 1, GL_FALSE, glm::value_ptr(viewMatrix));

	glUniformMatrix4fv(gProjectionMatrixUniform, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));

	//glUniformMatrix4fv(gMVPUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix*viewMatrix*modelMatrix);

	glBindVertexArray(gVao_Project_Model);

	for (int i = 0; i < mesh_data.size(); i++)
	{
		if (gbLight == true)
		{
			glUniform3fv(gKaUniform, 1, arr_material[mesh_data[i].material_index].Ka);
			glUniform3fv(gKdUniform, 1, arr_material[mesh_data[i].material_index].Kd);
			glUniform3fv(gKsUniform, 1, arr_material[mesh_data[i].material_index].Ks);
			glUniform1f(gMaterialShininessUniform, materialShininess);
			glUniform1f(gAlphaUniform, arr_material[mesh_data[i].material_index].d);

			if (arr_material[mesh_data[i].material_index].ismap_Kd == true)
			{
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, arr_material[mesh_data[i].material_index].gTexture);
				glUniform1i(gTextureSamplerUniform, 0);
				glUniform1i(gTextureActiveUniform, 1);
			}
			else
				glUniform1i(gTextureActiveUniform, 0);
		}

		glDrawArrays(GL_TRIANGLES, mesh_data[i].vertex_Index, mesh_data[i].vertex_Count);
	}

	glBindVertexArray(0);

	//Helicopter
	modelMatrix = glm::mat4(1.0f);
	viewMatrix = glm::mat4(1.0f);
	scaleMatrix = glm::mat4(1.0f);
	rotationMatrix = glm::mat4(1.0f);
	translationMatrix = glm::mat4(1.0f);

	viewMatrix = Scene3_Camera.GetViewMatrix();

	modelMatrix = glm::translate(modelMatrix, glm::vec3(MODEL_X_TRANSLATE, MODEL_Y_TRANSLATE + 1.0f, MODEL_Z_TRANSLATE));

	//rotationMatrix = glm::rotate(rotationMatrix, glm::radians(g_rotate), glm::vec3(1.0f, 0.0f, 0.0f));
	//modelMatrix = modelMatrix*rotationMatrix;

	rotationMatrix = glm::rotate(rotationMatrix, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	modelMatrix = modelMatrix*rotationMatrix;

	//rotationMatrix = glm::rotate(rotationMatrix,glm::radians(g_rotate), glm::vec3(0.0f, 0.0f, 1.0f));
	//modelMatrix = modelMatrix*rotationMatrix;

	glUniformMatrix4fv(gModelMatrixUniform, 1, GL_FALSE, glm::value_ptr(modelMatrix));

	glUniformMatrix4fv(gViewMatrixUniform, 1, GL_FALSE, glm::value_ptr(viewMatrix));

	glUniformMatrix4fv(gProjectionMatrixUniform, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));

	//glUniformMatrix4fv(gMVPUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix*viewMatrix*modelMatrix);

	glBindVertexArray(gVao_Helicopter);

	for (int i = 0; i < mesh_data2.size(); i++)
	{
		if (gbLight == true)
		{
			glUniform3fv(gKaUniform, 1, arr_material2[mesh_data2[i].material_index].Ka);
			glUniform3fv(gKdUniform, 1, arr_material2[mesh_data2[i].material_index].Kd);
			glUniform3fv(gKsUniform, 1, arr_material2[mesh_data2[i].material_index].Ks);
			glUniform1f(gMaterialShininessUniform, materialShininess);
			glUniform1f(gAlphaUniform, arr_material2[mesh_data2[i].material_index].d);

			if (arr_material2[mesh_data2[i].material_index].ismap_Kd == true)
			{
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, arr_material2[mesh_data2[i].material_index].gTexture);
				glUniform1i(gTextureSamplerUniform, 0);
				glUniform1i(gTextureActiveUniform, 1);
			}
			else
				glUniform1i(gTextureActiveUniform, 0);
		}

		glDrawArrays(GL_TRIANGLES, mesh_data2[i].vertex_Index, mesh_data2[i].vertex_Count);
	}

	glBindVertexArray(0);

	glUseProgram(0);

	SwapBuffers(ghdc);
}

void update(void)
{

}

void resize(int width, int height)
{
	if (height == 0)
		height = 1;
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	gPerspectiveProjectionMatrix = glm::perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100000.0f);
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

	if (gVao_Project_Model)
	{
		glDeleteVertexArrays(1, &gVao_Project_Model);
		gVao_Project_Model = 0;
	}

	if (gVao_Helicopter)
	{
		glDeleteVertexArrays(1, &gVao_Helicopter);
		gVao_Helicopter = 0;
	}

	if (gVbo_Position)
	{
		glDeleteBuffers(1, &gVbo_Position);
		gVbo_Position = 0;
	}

	if (gVbo_Normal)
	{
		glDeleteBuffers(1, &gVbo_Normal);
		gVbo_Normal = 0;
	}

	//Detach Shader 
	glDetachShader(gShaderProgramObject, gVertexShaderObject);
	glDetachShader(gShaderProgramObject, gFragmentShaderObject);

	//Delete Shader
	glDeleteShader(gVertexShaderObject);
	gVertexShaderObject = 0;

	glDeleteShader(gFragmentShaderObject);
	gFragmentShaderObject = 0;

	//Delete Program
	glDeleteProgram(gShaderProgramObject);
	gShaderProgramObject = 0;

	//Stray call to glUseProgram(0)
	glUseProgram(0);

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

	if (i_Exit_Flag == 0)
	{
		fprintf(gpFile, "Log File Closed Successfully");
	}
	else if (i_Exit_Flag == 1)
	{
		fprintf(gpFile, "Log File Closed Erroniously");
	}

	fclose(gpFile);
	gpFile = NULL;

	DestroyWindow(ghwnd);
}