#include<windows.h>
#include<C:\glew\include\GL\glew.h>
#include<gl/GL.h>
#include<stdio.h>
#include<math.h>
#include "glm/glm.hpp" 
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
//#include"vmath.h"
#include"Sphere.h"

#pragma comment(lib,"User32.lib")
#pragma comment(lib,"GDI32.lib")
#pragma comment(lib,"C:\\glew\\lib\\Release\\Win32\\glew32.lib")
#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"Sphere.lib")

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

FILE *gpFile;
HWND ghwnd;
HDC ghdc;
HGLRC ghrc;
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
bool gbActiveWindow = false;
bool gbFullscreen = false;
bool gbIsEscapeKeyPressed = false;

struct PointLight
{
	GLfloat lightAmbient[4] = { 0.2f,0.2f,0.2f,1.0f };
	GLfloat lightDiffuse[4] = { 0.5f,0.5f,0.5f,1.0f };
	GLfloat lightSpecular[4] = { 1.0f,1.0f,1.0f,1.0f };
	GLfloat lightPosition[4] = { 2.5f,0.3f,-4.3f,1.0f };
	GLfloat gConstant = 1.0f;
	GLfloat gLinear = 1.419999f;
	GLfloat gQuadratic = 0.090000f;
	GLuint gLaUniform;
	GLuint gLdUniform;
	GLuint gLsUniform;
	GLuint gConstantUniform;
	GLuint gLinearUniform;
	GLuint gQuadraticUniform;
	GLuint gLightPositionUniform;
}pointLight;

struct SpotLight
{
	GLfloat lightAmbient[4] = { 0.1f,0.1f,0.1f,1.0f };
	GLfloat lightDiffuse[4] = { 0.8f,0.8f,0.8f,1.0f };
	GLfloat lightSpecular[4] = { 1.0f,1.0f,1.0f,1.0f };
	GLfloat lightDirection[3] = { 1.0f, 0.0f, -1.0f };
	GLfloat lightDirection1[3] = { 0.0f, 0.0f, -1.0f };
	GLfloat lightDirection2[3] = { -1.0f, 0.0f, -1.0f };
	GLfloat lightPosition[4] = { -3.5f,-0.4f,-1.0f,1.0f };
	GLfloat lightPosition1[4] = { 0.0f,-0.4f,-1.0f,1.0f };
	GLfloat lightPosition2[4] = { 3.5f,-0.4f,-1.0f,1.0f };
	GLfloat gConstant = 1.0f;
	GLfloat gLinear = 0.09f;
	GLfloat gQuadratic = 0.032f;
	GLfloat cutoff_angle = 18.5f;
	GLfloat outer_cutoff_angle = 23.5f;
	GLuint gLaUniform;
	GLuint gLdUniform;
	GLuint gLsUniform;
	GLuint gLightPositionUniform;
	GLuint gLightPosition1Uniform;
	GLuint gLightPosition2Uniform;
	GLuint gLightDirectionUniform;
	GLuint gLightDirection1Uniform;
	GLuint gLightDirection2Uniform;
	GLuint gConstantUniform;
	GLuint gLinearUniform;
	GLuint gQuadraticUniform;
	GLuint gCutOffUniform;
	GLuint gOuterCutOffUniform;
}spotLight;

struct MaterialProperties
{
	GLfloat materialAmbient[4] = { 0.0f,0.0f,0.0f,1.0f };
	GLfloat materialDiffuse[4] = { 1.0f,1.0f,1.0f,1.0f };
	GLfloat materialSpecular[4] = { 1.0f,1.0f,1.0f,1.0f };
	GLfloat materialShininess = 50.0f;
	GLuint gKaUniform;
	GLuint gKdUniform;
	GLuint gKsUniform;
	GLuint gMaterialShininessUniform;
}materialProperties;

GLuint gVertexShaderObject;
GLuint gFragmentShaderObject;
GLuint gShaderProgramObject;

GLuint gVao_Sphere, gVao_Square, gVao_Cube, gVao_Pyramid;
GLuint gVbo_Position, gVbo_Normal, gVbo_Elements;

GLuint gModelMatrixUniform, gViewMatrixUniform, gProjectionMatrixUniform;
GLuint gLKeyPressedUniform;

GLuint gViewPositionUniform;

GLfloat gAngle_Sphere;

GLfloat lightAmbient[] = { 0.1f,0.1f,0.1f,1.0f };
GLfloat lightDiffuse[] = { 0.8f,0.8f,0.8f,1.0f };
GLfloat lightSpecular[] = { 1.0f,1.0f,1.0f,1.0f };
GLfloat lightPosition[] = { 0.0f, 1.6f, -3.9f};

GLfloat materialAmbient[] = { 0.0f,0.0f,0.0f,1.0f };
GLfloat materialDiffuse[] = { 1.0f,1.0f,1.0f,1.0f };
GLfloat materialSpecular[] = { 1.0f,1.0f,1.0f,1.0f };
GLfloat materialShininess = 50.0f;

GLfloat sphere_vertices[1146];
GLfloat sphere_normals[1146];
GLfloat sphere_textures[764];
unsigned short sphere_elements[2280];
unsigned int gNumVertices, gNumElements;

glm::mat4 gPerspectiveProjectionMatrix;
glm::vec3 gViewPosition = glm::vec3(0.0f, 0.0f, 3.0f);


bool gbLight = false;
bool gbIsLKeyPressed = false;

int light_position_counter = 1;

GLfloat cutoff_angle = 12.5f;
GLfloat outer_cutoff_angle = 17.5f;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	void initialize(void);
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
				//update();
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
		/*case 0x41://A
			spotLight.lightPosition[0] = spotLight.lightPosition[0] - 0.1f;
			break;

		case 0x44://D
			spotLight.lightPosition[0] = spotLight.lightPosition[0] + 0.1f;
			break;

		case 0x51://Q
			spotLight.lightPosition[1] = spotLight.lightPosition[1] + 0.1f;
			break;

		case 0x45://E
			spotLight.lightPosition[1] = spotLight.lightPosition[1] - 0.1f;
			break;

		case 0x53://S
			spotLight.lightPosition[2] = spotLight.lightPosition[2] + 0.1f;
			break;

		case 0x57://W
			spotLight.lightPosition[2] = spotLight.lightPosition[2] - 0.1f;
			break;*/

		case 0x50://P
			fprintf(gpFile, "Angle :%f :%f\n", spotLight.cutoff_angle, spotLight.outer_cutoff_angle);
			break;
			
		case VK_ADD:
			spotLight.cutoff_angle = spotLight.cutoff_angle + 1.0f;
			spotLight.outer_cutoff_angle = spotLight.outer_cutoff_angle + 1.0f;
			break;

		case VK_SUBTRACT:
			spotLight.cutoff_angle = spotLight.cutoff_angle - 1.0f;
			spotLight.outer_cutoff_angle = spotLight.outer_cutoff_angle - 1.0f;
			break;

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

	getSphereVertexData(sphere_vertices, sphere_normals, sphere_textures, sphere_elements);

	gNumVertices = getNumberOfSphereVertices();
	gNumElements = getNumberOfSphereElements();
	//Vertex Shader
	gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *vertexShaderSourceCode =
		"#version 450" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vNormal;" \
		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"uniform int u_lighting_enabled;" \
		"out vec3 transformed_normals;" \
		"out vec3 FragPos;" \
		"void main(void)" \
		"{" \
		"if(u_lighting_enabled==1)" \
		"{" \
		"transformed_normals = mat3(u_view_matrix*u_model_matrix)*vNormal;" \
		"FragPos = vec3(u_model_matrix * vPosition);" \
		"}" \
		"gl_Position = u_projection_matrix*u_view_matrix*u_model_matrix*vPosition;" \
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
		"in vec3 FragPos;" \

		"struct PointLight" \
		"{" \
		"vec3 u_La;" \
		"vec3 u_Ld;" \
		"vec3 u_Ls;" \
		"float u_constant;" \
		"float u_linear;" \
		"float u_quadratic;" \
		"vec3 u_light_position;" \
		"};" \

		"struct SpotLight" \
		"{" \
		"vec3 u_La;" \
		"vec3 u_Ld;" \
		"vec3 u_Ls;" \
		"float constant;" \
		"float linear;" \
		"float quadratic;" \
		"vec3 u_light_position;" \
		"vec3 u_light_position1;" \
		"vec3 u_light_position2;" \
		"vec3 u_light_direction;" \
		"vec3 u_light_direction1;" \
		"vec3 u_light_direction2;" \
		"float u_cutOff;" \
		"float u_outerCutOff;" \
		"};" \

		"struct Material" \
		"{" \
		"float u_material_shininess;" \
		"vec3 u_Ka;" \
		"vec3 u_Kd;" \
		"vec3 u_Ks;" \
		"};" \

		"uniform int u_lighting_enabled;" \
		"uniform vec3 u_viewPos;" \
		"uniform PointLight pointlight;" \
		"uniform SpotLight spotlight;" \
		"uniform Material material;" \

		"out vec4 FragColor;" \

		"vec3 CalculatePointLight(vec3 lightpos)" \
		"{" \
		"vec3 ambient = pointlight.u_La * material.u_Ka;" \
		"vec3 normalized_transformed_normals = normalize(transformed_normals);" \
		"vec3 normalized_light_direction = normalize(lightpos);" \
		"float tn_dot_ld = max(dot(normalized_transformed_normals,normalized_light_direction),0.0);" \
		"vec3 diffuse = pointlight.u_Ld * material.u_Kd * tn_dot_ld;" \
		"vec3 view_direction = normalize(u_viewPos - FragPos);" \
		"vec3 reflection_vector = reflect(-normalized_light_direction,normalized_transformed_normals);" \
		"vec3 specular = pointlight.u_Ls * material.u_Ks * pow(max(dot(reflection_vector,view_direction),0.0),material.u_material_shininess);" \
		"float distance = length(lightpos-FragPos);" \
		"float attenuation = 1.0 / (pointlight.u_constant + pointlight.u_linear * distance + pointlight.u_quadratic * (distance * distance));" \
		/*"ambient = ambient * attenuation;" \*/
		"diffuse = diffuse * attenuation;" \
		"specular = specular * attenuation;" \
		"return(ambient + diffuse + specular);" \
		"}" \

		"vec3 CalculateSpotLight(vec3 lightPos,vec3 dir)" \
		"{" \
		"vec3 ambient = spotlight.u_La * material.u_Ka;" \

		"vec3 normalized_transformed_normals = normalize(transformed_normals);" \
		"vec3 normalized_light_direction = normalize(lightPos - FragPos);" \
		"float tn_dot_ld = max(dot(normalized_transformed_normals,normalized_light_direction),0.0);" \
		"vec3 diffuse = spotlight.u_Ld * material.u_Kd * tn_dot_ld;" \

		"vec3 view_direction = normalize(u_viewPos - FragPos);" \
		"vec3 reflection_vector = reflect(-normalized_light_direction,normalized_transformed_normals);" \
		"vec3 specular = spotlight.u_Ls * material.u_Ks * pow(max(dot(reflection_vector,view_direction),0.0),material.u_material_shininess);" \

		"float theta = dot(normalized_light_direction,normalize(-dir));" \
		"float epsilon = (spotlight.u_cutOff - spotlight.u_outerCutOff);" \
		"float intensity = clamp((theta - spotlight.u_outerCutOff) / epsilon, 0.0, 1.0);" \
		"diffuse = diffuse * intensity;" \
		"specular = specular * intensity;" \

		"float distance = length(lightPos-FragPos);" \
		"float attenuation = 1.0 / (spotlight.constant + spotlight.linear * distance + spotlight.quadratic * (distance * distance));" \
		"ambient = ambient * attenuation;" \
		"diffuse = diffuse * attenuation;" \
		"specular = specular * attenuation;" \
		"return(ambient + diffuse + specular);" \
		"}" \

		"void main(void)" \
		"{" \
		"vec3 phong_ads_color;" \
		"if(u_lighting_enabled == 1)" \
		"{" \
		/*"phong_ads_color = CalculatePointLight(u_light_position);" \*/
		//"phong_ads_color += CalculatePointLight(pointlight.u_light_position);" 
		/*"phong_ads_color += CalculateSpotLight();" \*/
		"phong_ads_color += CalculateSpotLight(spotlight.u_light_position,spotlight.u_light_direction);" \
		"phong_ads_color += CalculateSpotLight(spotlight.u_light_position1,spotlight.u_light_direction1);" \
		"phong_ads_color += CalculateSpotLight(spotlight.u_light_position2,spotlight.u_light_direction2);" \
		"}" \
		"else if(u_lighting_enabled == 0)" \
		"{" \
		"phong_ads_color = vec3(0.0f,0.0f,0.0f);" \
		"}" \
		"else" \
		"{" \
		"phong_ads_color = vec3(1.0f,1.0f,1.0f);" \
		"}" \
		"FragColor = vec4(phong_ads_color,1.0);" \
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
	//Point Light Uniform
	pointLight.gLaUniform = glGetUniformLocation(gShaderProgramObject, "pointlight.u_La");
	pointLight.gLdUniform = glGetUniformLocation(gShaderProgramObject, "pointlight.u_Ld");
	pointLight.gLsUniform = glGetUniformLocation(gShaderProgramObject, "pointlight.u_Ls");

	pointLight.gLightPositionUniform = glGetUniformLocation(gShaderProgramObject, "pointlight.u_light_position");
	pointLight.gConstantUniform = glGetUniformLocation(gShaderProgramObject, "pointlight.u_constant");
	pointLight.gLinearUniform = glGetUniformLocation(gShaderProgramObject, "pointlight.u_linear");
	pointLight.gQuadraticUniform = glGetUniformLocation(gShaderProgramObject, "pointlight.u_quadratic");

	materialProperties.gKaUniform = glGetUniformLocation(gShaderProgramObject, "material.u_Ka");
	materialProperties.gKdUniform = glGetUniformLocation(gShaderProgramObject, "material.u_Kd");
	materialProperties.gKsUniform = glGetUniformLocation(gShaderProgramObject, "material.u_Ks");

	materialProperties.gMaterialShininessUniform = glGetUniformLocation(gShaderProgramObject, "material.u_material_shininess");

	gViewPositionUniform = glGetUniformLocation(gShaderProgramObject, "u_viewPos");

	//Spot Light Uniform
	spotLight.gLaUniform = glGetUniformLocation(gShaderProgramObject, "spotlight.u_La");
	spotLight.gLdUniform = glGetUniformLocation(gShaderProgramObject, "spotlight.u_Ld");
	spotLight.gLsUniform = glGetUniformLocation(gShaderProgramObject, "spotlight.u_Ls");
	spotLight.gLightPositionUniform = glGetUniformLocation(gShaderProgramObject, "spotlight.u_light_position");
	spotLight.gLightPosition1Uniform = glGetUniformLocation(gShaderProgramObject, "spotlight.u_light_position1");
	spotLight.gLightPosition2Uniform = glGetUniformLocation(gShaderProgramObject, "spotlight.u_light_position2");

	spotLight.gConstantUniform = glGetUniformLocation(gShaderProgramObject, "spotlight.constant");
	spotLight.gLinearUniform = glGetUniformLocation(gShaderProgramObject, "spotlight.linear");
	spotLight.gQuadraticUniform = glGetUniformLocation(gShaderProgramObject, "spotlight.quadratic");

	spotLight.gLightDirectionUniform = glGetUniformLocation(gShaderProgramObject, "spotlight.u_light_direction");
	spotLight.gLightDirection1Uniform = glGetUniformLocation(gShaderProgramObject, "spotlight.u_light_direction1");
	spotLight.gLightDirection2Uniform = glGetUniformLocation(gShaderProgramObject, "spotlight.u_light_direction2");
	spotLight.gCutOffUniform = glGetUniformLocation(gShaderProgramObject, "spotlight.u_cutOff");
	spotLight.gOuterCutOffUniform = glGetUniformLocation(gShaderProgramObject, "spotlight.u_outerCutOff");

	/*****************VAO For Sphere*****************/
	glGenVertexArrays(1, &gVao_Sphere);
	glBindVertexArray(gVao_Sphere);

	/*****************Sphere Position****************/
	glGenBuffers(1, &gVbo_Position);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_vertices), sphere_vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/*****************Sphere Normals****************/
	glGenBuffers(1, &gVbo_Normal);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Normal);
	glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_normals), sphere_normals, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &gVbo_Elements);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(sphere_elements), sphere_elements, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	const GLfloat square_vertices[] = 
	{
		1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,1.0f,
		1.0f,-1.0f,1.0f
	};

	const GLfloat square_normals[] = 
	{
		0.0f,1.0f,0.0f,
		0.0f,1.0f,0.0f,
		0.0f,1.0f,0.0f,
		0.0f,1.0f,0.0f
	};

	/*************Square****************/
	glGenVertexArrays(1, &gVao_Square);
	glBindVertexArray(gVao_Square);

	/**************Square Position************/
	glGenBuffers(1, &gVbo_Position);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Position);

	glBufferData(GL_ARRAY_BUFFER, sizeof(square_vertices), square_vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/***************Square Normals*************/
	glGenBuffers(1, &gVbo_Normal);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Normal);
	
	glBufferData(GL_ARRAY_BUFFER, sizeof(square_normals), square_normals, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	const GLfloat cubeVertices[] =
	{
		1.0f,1.0f,1.0f,
		-1.0f,1.0f,1.0f,
		-1.0f,-1.0f,1.0f,
		1.0f,-1.0f,1.0f,

		1.0f,1.0f,-1.0f,
		1.0f,1.0f,1.0f,
		1.0f,-1.0f,1.0f,
		1.0f,-1.0f,-1.0f,

		-1.0f,1.0f,-1.0f,
		1.0f,1.0f,-1.0f,
		1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,-1.0f,

		-1.0f,1.0f,1.0f,
		-1.0f,1.0f,-1.0f,
		-1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,1.0f,

		1.0f,1.0f,-1.0f,
		-1.0f,1.0f,-1.0f,
		-1.0f,1.0f,1.0f,
		1.0f,1.0f,1.0f,

		1.0f,-1.0f,1.0f,
		-1.0f,-1.0f,1.0f,
		-1.0f,-1.0f,-1.0f,
		1.0f,-1.0f,-1.0f
	};

	const GLfloat cubeNormal[] =
	{
		0.0f,0.0f,1.0f,
		0.0f,0.0f,1.0f,
		0.0f,0.0f,1.0f,
		0.0f,0.0f,1.0f,

		1.0f,0.0f,0.0f,
		1.0f,0.0f,0.0f,
		1.0f,0.0f,0.0f,
		1.0f,0.0f,0.0f,

		0.0f,0.0f,-1.0f,
		0.0f,0.0f,-1.0f,
		0.0f,0.0f,-1.0f,
		0.0f,0.0f,-1.0f,

		-1.0f,0.0f,0.0f,
		-1.0f,0.0f,0.0f,
		-1.0f,0.0f,0.0f,
		-1.0f,0.0f,0.0f,

		0.0f,1.0f,0.0f,
		0.0f,1.0f,0.0f,
		0.0f,1.0f,0.0f,
		0.0f,1.0f,0.0f,

		0.0f,-1.0f,0.0f,
		0.0f,-1.0f,0.0f,
		0.0f,-1.0f,0.0f,
		0.0f,-1.0f,0.0f
	};

	/****************Cube************/
	glGenVertexArrays(1, &gVao_Cube);
	glBindVertexArray(gVao_Cube);

	/******************Cube Vertices*****************/
	glGenBuffers(1, &gVbo_Position);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Position);

	glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_POSITION);

	glBindVertexArray(0);

	/*****************Cube Normal****************/
	glGenBuffers(1, &gVbo_Normal);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Normal);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cubeNormal), cubeNormal, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	GLfloat pyramidVertices[] =
	{
		0.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,

		0.0f, 1.0f, 0.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f,

		0.0f,1.0f,0.0f,
		1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,-1.0f,

		0.0f,1.0f,0.0f,
		-1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,1.0f
	};

	GLfloat pyramidNormal[] =
	{
		0.0f, 0.447214f, 0.894428f,
		0.0f, 0.447214f, 0.894428f,
		0.0f, 0.447214f, 0.894428f,

		0.894428f, 0.447214f, 0.0f,
		0.894428f, 0.447214f, 0.0f,
		0.894428f, 0.447214f, 0.0f,

		0.0f, 0.447214f, -0.894428f,
		0.0f, 0.447214f, -0.894428f,
		0.0f, 0.447214f, -0.894428f,

		-0.894428f, 0.447214f, 0.0f,
		-0.894428f, 0.447214f, 0.0f,
		-0.894428f, 0.447214f, 0.0f
	};

	/*****************Pyramid Position****************/
	glGenBuffers(1, &gVbo_Position);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(pyramidVertices), pyramidVertices, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/*****************Pyramid Lights****************/
	glGenBuffers(1, &gVbo_Normal);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Normal);
	glBufferData(GL_ARRAY_BUFFER, sizeof(pyramidNormal), pyramidNormal, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glShadeModel(GL_SMOOTH);
	//glHint(GL_PERSPECTIVE_CORRECTION_HINT , GL_NICEST);
	//glEnable(GL_CULL_FACE);
	
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix = glm::mat4(1.0f);

	resize(WIN_WIDTH, WIN_HEIGHT);
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	
	//Use Shader Program Object
	glUseProgram(gShaderProgramObject);

	if (gbLight == true)
	{
		glUniform1i(gLKeyPressedUniform, 1);//1 For Material and Light Calculation

		//Point Light
		glUniform3fv(pointLight.gLaUniform, 1, pointLight.lightAmbient);
		glUniform3fv(pointLight.gLdUniform, 1, pointLight.lightDiffuse);
		glUniform3fv(pointLight.gLsUniform, 1, pointLight.lightSpecular);
		glUniform3fv(pointLight.gLightPositionUniform, 1, pointLight.lightPosition);
		glUniform1f(pointLight.gConstantUniform, pointLight.gConstant);
		glUniform1f(pointLight.gLinearUniform, pointLight.gLinear);
		glUniform1f(pointLight.gQuadraticUniform, pointLight.gQuadratic);

		materialProperties.materialAmbient[0] = 0.24725f;
		materialProperties.materialAmbient[1] = 0.1995f;
		materialProperties.materialAmbient[2] = 0.0745f;
		materialProperties.materialAmbient[3] = 1.0f;
		materialProperties.materialDiffuse[0] = 0.75164f;
		materialProperties.materialDiffuse[1] = 0.60648f;
		materialProperties.materialDiffuse[2] = 0.22648f;
		materialProperties.materialDiffuse[3] = 1.0f;
		materialProperties.materialSpecular[0] = 0.628281f;
		materialProperties.materialSpecular[1] = 0.555802f;
		materialProperties.materialSpecular[2] = 0.366065f;
		materialProperties.materialSpecular[3] = 1.0f;
		materialProperties.materialShininess = 0.4f*128.0f;

		//Material
		glUniform3fv(materialProperties.gKaUniform, 1, materialProperties.materialAmbient);
		glUniform3fv(materialProperties.gKdUniform, 1, materialProperties.materialDiffuse);
		glUniform3fv(materialProperties.gKsUniform, 1, materialProperties.materialSpecular);
		glUniform1f(materialProperties.gMaterialShininessUniform, materialProperties.materialShininess);
		glUniform3fv(gViewPositionUniform, 1, glm::value_ptr(gViewPosition));

		//Spot Light
		glUniform3fv(spotLight.gLaUniform, 1, spotLight.lightAmbient);
		glUniform3fv(spotLight.gLdUniform, 1, spotLight.lightDiffuse);
		glUniform3fv(spotLight.gLsUniform, 1, spotLight.lightSpecular);
		glUniform3fv(spotLight.gLightPositionUniform, 1, spotLight.lightPosition);
	//	glUniform3fv(spotLight.gLightPosition1Uniform, 1, spotLight.lightPosition1);
		glUniform3fv(spotLight.gLightPosition2Uniform, 1, spotLight.lightPosition2);
		glUniform1f(spotLight.gConstantUniform, spotLight.gConstant);
		glUniform1f(spotLight.gLinearUniform, spotLight.gLinear);
		glUniform1f(spotLight.gQuadraticUniform, spotLight.gQuadratic);
		glUniform3fv(spotLight.gLightDirectionUniform, 1, spotLight.lightDirection);
		//glUniform3fv(spotLight.gLightDirection1Uniform, 1, spotLight.lightDirection1);
		glUniform3fv(spotLight.gLightDirection2Uniform, 1, spotLight.lightDirection2);
		glUniform1f(spotLight.gCutOffUniform, cos(glm::radians(spotLight.cutoff_angle)));
		glUniform1f(spotLight.gOuterCutOffUniform, cos(glm::radians(spotLight.outer_cutoff_angle)));

	}
	else
	{
		glUniform1i(gLKeyPressedUniform, 0);
	}

	glm::mat4 modelMatrix = glm::mat4(1.0f);
	glm::mat4 viewMatrix = glm::mat4(1.0f);
	glm::mat4 scaleMatrix = glm::mat4(1.0f);
	glm::mat4 rotationMatrix = glm::mat4(1.0f);

	modelMatrix = glm::translate(modelMatrix,glm::vec3(0.0f, -0.55f, -4.0f));

	glUniformMatrix4fv(gModelMatrixUniform, 1, GL_FALSE, glm::value_ptr(modelMatrix));

	glUniformMatrix4fv(gViewMatrixUniform, 1, GL_FALSE, glm::value_ptr(viewMatrix));

	glUniformMatrix4fv(gProjectionMatrixUniform, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);

	glBindVertexArray(0);

	/************Square***********/

	modelMatrix = glm::mat4(1.0f);
	viewMatrix = glm::mat4(1.0f);
	scaleMatrix = glm::mat4(1.0f);
	rotationMatrix = glm::mat4(1.0f);

	modelMatrix = glm::translate(modelMatrix, glm::vec3(0.0f, 0.0f, -5.0f));

	scaleMatrix = glm::scale(scaleMatrix, glm::vec3(5.0f, 1.0f, 5.0f));
	modelMatrix = modelMatrix * scaleMatrix;

	glUniformMatrix4fv(gModelMatrixUniform, 1, GL_FALSE, glm::value_ptr(modelMatrix));

	glUniformMatrix4fv(gViewMatrixUniform, 1, GL_FALSE, glm::value_ptr(viewMatrix));

	glUniformMatrix4fv(gProjectionMatrixUniform, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));

	if (gbLight == true)
	{
		materialProperties.materialAmbient[0] = 0.0f;
		materialProperties.materialAmbient[1] = 0.05f;
		materialProperties.materialAmbient[2] = 0.05f;
		materialProperties.materialAmbient[3] = 1.0f;
		materialProperties.materialDiffuse[0] = 0.4f;
		materialProperties.materialDiffuse[1] = 0.5f;
		materialProperties.materialDiffuse[2] = 0.5f;
		materialProperties.materialDiffuse[3] = 1.0f;
		materialProperties.materialSpecular[0] = 0.04f;
		materialProperties.materialSpecular[1] = 0.7f;
		materialProperties.materialSpecular[2] = 0.7f;
		materialProperties.materialSpecular[3] = 1.0f;
		materialProperties.materialShininess = 0.078125f*128.0f;

		glUniform1i(gLKeyPressedUniform, 1);
		glUniform3fv(materialProperties.gKaUniform, 1, materialProperties.materialAmbient);
		glUniform3fv(materialProperties.gKdUniform, 1, materialProperties.materialDiffuse);
		glUniform3fv(materialProperties.gKsUniform, 1, materialProperties.materialSpecular);
		glUniform1f(materialProperties.gMaterialShininessUniform, materialProperties.materialShininess);
	}
	else
	{
		glUniform1i(gLKeyPressedUniform, 0);
	}

	glBindVertexArray(gVao_Square);

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	glBindVertexArray(0);

	/*******************Cube**********************/
	modelMatrix = glm::mat4(1.0f);
	viewMatrix = glm::mat4(1.0f);
	scaleMatrix = glm::mat4(1.0f);
	rotationMatrix = glm::mat4(1.0f);

	modelMatrix = glm::translate(modelMatrix, glm::vec3(1.7f, -0.45f, -4.0f));

	scaleMatrix = glm::scale(scaleMatrix, glm::vec3(0.5f, 0.5f, 0.5f));
	modelMatrix = modelMatrix * scaleMatrix;

	glUniformMatrix4fv(gModelMatrixUniform, 1, GL_FALSE, glm::value_ptr(modelMatrix));

	glUniformMatrix4fv(gViewMatrixUniform, 1, GL_FALSE, glm::value_ptr(viewMatrix));

	glUniformMatrix4fv(gProjectionMatrixUniform, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));

	if (gbLight == true)
	{
		materialProperties.materialAmbient[0] = 0.0f;
		materialProperties.materialAmbient[1] = 0.0f;
		materialProperties.materialAmbient[2] = 0.0f;
		materialProperties.materialAmbient[3] = 1.0f;
		materialProperties.materialDiffuse[0] = 0.1f;
		materialProperties.materialDiffuse[1] = 0.35f;
		materialProperties.materialDiffuse[2] = 0.1f;
		materialProperties.materialDiffuse[3] = 1.0f;
		materialProperties.materialSpecular[0] = 0.45f;
		materialProperties.materialSpecular[1] = 0.55f;
		materialProperties.materialSpecular[2] = 0.45f;
		materialProperties.materialSpecular[3] = 1.0f;
		materialProperties.materialShininess = 0.25f*128.0f;

		glUniform1i(gLKeyPressedUniform, 1);
		glUniform3fv(materialProperties.gKaUniform, 1, materialProperties.materialAmbient);
		glUniform3fv(materialProperties.gKdUniform, 1, materialProperties.materialDiffuse);
		glUniform3fv(materialProperties.gKsUniform, 1, materialProperties.materialSpecular);
		glUniform1f(materialProperties.gMaterialShininessUniform, materialProperties.materialShininess);
	}
	else
	{
		glUniform1i(gLKeyPressedUniform, 0);
	}

	glBindVertexArray(gVao_Cube);

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 20, 4);

	glBindVertexArray(0);

	/*******************Pyramid**********************/
	modelMatrix = glm::mat4(1.0f);
	viewMatrix = glm::mat4(1.0f);
	scaleMatrix = glm::mat4(1.0f);
	rotationMatrix = glm::mat4(1.0f);

	modelMatrix = glm::translate(modelMatrix, glm::vec3(-1.5f, -0.4f, -4.0f));

	scaleMatrix = glm::scale(scaleMatrix, glm::vec3(0.5f, 0.5f, 0.5f));
	modelMatrix = modelMatrix * scaleMatrix;

	glUniformMatrix4fv(gModelMatrixUniform, 1, GL_FALSE, glm::value_ptr(modelMatrix));

	glUniformMatrix4fv(gViewMatrixUniform, 1, GL_FALSE, glm::value_ptr(viewMatrix));

	glUniformMatrix4fv(gProjectionMatrixUniform, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));

	if (gbLight == true)
	{
		materialProperties.materialAmbient[0] = 0.0f;
		materialProperties.materialAmbient[1] = 0.0f;
		materialProperties.materialAmbient[2] = 0.0f;
		materialProperties.materialAmbient[3] = 1.0f;
		materialProperties.materialDiffuse[0] = 0.5f;
		materialProperties.materialDiffuse[1] = 0.0f;
		materialProperties.materialDiffuse[2] = 0.0f;
		materialProperties.materialDiffuse[3] = 1.0f;
		materialProperties.materialSpecular[0] = 0.7f;
		materialProperties.materialSpecular[1] = 0.6f;
		materialProperties.materialSpecular[2] = 0.6f;
		materialProperties.materialSpecular[3] = 1.0f;
		materialProperties.materialShininess = 0.25f*128.0f;

		glUniform1i(gLKeyPressedUniform, 1);
		glUniform3fv(materialProperties.gKaUniform, 1, materialProperties.materialAmbient);
		glUniform3fv(materialProperties.gKdUniform, 1, materialProperties.materialDiffuse);
		glUniform3fv(materialProperties.gKsUniform, 1, materialProperties.materialSpecular);
		glUniform1f(materialProperties.gMaterialShininessUniform, materialProperties.materialShininess);
	}
	else
	{
		glUniform1i(gLKeyPressedUniform, 0);
	}

	glBindVertexArray(gVao_Pyramid);

	glDrawArrays(GL_TRIANGLE_FAN, 0, 12);

	glBindVertexArray(0);

	glUseProgram(0);

	SwapBuffers(ghdc);
}

void update(void)
{
	gAngle_Sphere = gAngle_Sphere + 1.0f;
	if (gAngle_Sphere >= 360.0f)
		gAngle_Sphere = 0.0f;
}

void resize(int width, int height)
{
	if (height == 0)
		height = 1;
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	gPerspectiveProjectionMatrix = glm::perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
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

	if (gVao_Sphere)
	{
		glDeleteVertexArrays(1, &gVao_Sphere);
		gVao_Sphere = 0;
	}

	if (gVao_Square)
	{
		glDeleteVertexArrays(1, &gVao_Square);
		gVao_Square = 0;
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

	if (gVbo_Elements)
	{
		glDeleteBuffers(1, &gVbo_Elements);
		gVbo_Elements = 0;
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