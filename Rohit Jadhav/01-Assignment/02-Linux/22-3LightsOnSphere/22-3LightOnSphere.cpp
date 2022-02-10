#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<memory.h>

#include<X11/Xlib.h>
#include<X11/Xutil.h>
#include<X11/XKBlib.h>
#include<X11/keysym.h>

#include<GL/glew.h>
#include<GL/gl.h>
#include<GL/glx.h>

#include"vmath.h"
#include"Sphere.h"

using namespace std;
using namespace vmath;

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

enum {
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_TEXCOORD0,
	AMC_ATTRIBUTE_NORMAL
};


//For Window
Display *gpDisplay_RRJ = NULL;
Colormap gColormap_RRJ;
Window gWindow_RRJ;
XVisualInfo *gpXVisualInfo_RRJ = NULL;

//For FullScreen
bool bIsFullScreen_RRJ = false;

//For Error
FILE *gbFile_RRJ = NULL;

//For OpenGL
GLXContext gGLXContext_RRJ;
GLXFBConfig gGLXFBConfig_RRJ;
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
glXCreateContextAttribsARBProc glXCreateContextAttribsARB_RRJ = NULL;

//For Shader
GLuint shaderProgramObject_PV_RRJ;
GLuint shaderProgramObject_PF_RRJ;

//For Projection
mat4 perspectiveProjectionMatrix_RRJ;


//For Uniform Per Vertex Lighting
GLuint modelMatrixUniform_PV_RRJ;
GLuint viewMatrixUniform_PV_RRJ;
GLuint projectionMatrixUniform_PV_RRJ;

GLuint red_LaUniform_PV_RRJ;
GLuint red_LdUniform_PV_RRJ;
GLuint red_LsUniform_PV_RRJ;
GLuint red_lightPositionUniform_PV_RRJ;

GLuint green_LaUniform_PV_RRJ;
GLuint green_LdUniform_PV_RRJ;
GLuint green_LsUniform_PV_RRJ;
GLuint green_lightPositionUniform_PV_RRJ;

GLuint blue_LaUniform_PV_RRJ;
GLuint blue_LdUniform_PV_RRJ;
GLuint blue_LsUniform_PV_RRJ;
GLuint blue_lightPositionUniform_PV_RRJ;

GLuint KaUniform_PV_RRJ;
GLuint KdUniform_PV_RRJ;
GLuint KsUniform_PV_RRJ;
GLuint shininessUniform_PV_RRJ;
GLuint LKeyPressUniform_PV_RRJ;



//For Uniform Per Fragment Lighting
GLuint modelMatrixUniform_PF_RRJ;
GLuint viewMatrixUniform_PF_RRJ;
GLuint projectionMatrixUniform_PF_RRJ;

GLuint red_LaUniform_PF_RRJ;
GLuint red_LdUniform_PF_RRJ;
GLuint red_LsUniform_PF_RRJ;
GLuint red_lightPositionUniform_PF_RRJ;

GLuint green_LaUniform_PF_RRJ;
GLuint green_LdUniform_PF_RRJ;
GLuint green_LsUniform_PF_RRJ;
GLuint green_lightPositionUniform_PF_RRJ;

GLuint blue_LaUniform_PF_RRJ;
GLuint blue_LdUniform_PF_RRJ;
GLuint blue_LsUniform_PF_RRJ;
GLuint blue_lightPositionUniform_PF_RRJ;

GLuint KaUniform_PF_RRJ;
GLuint KdUniform_PF_RRJ;
GLuint KsUniform_PF_RRJ;
GLuint shininessUniform_PF_RRJ;
GLuint LKeyPressUniform_PF_RRJ;


//For Light
int iWhichLight_RRJ = 1;
bool bLight_RRJ = false;

struct LIGHTS{
	GLfloat lightAmbient_RRJ[4];
	GLfloat lightDiffuse_RRJ[4];
	GLfloat lightSpecular_RRJ[4];
	GLfloat lightPosition_RRJ[4];
};

struct LIGHTS lights_RRJ[3];

GLfloat angle_red_RRJ = 0.0f;
GLfloat angle_green_RRJ = 0.0f;
GLfloat angle_blue_RRJ = 0.0f;



//For Material
GLfloat materialAmbient_RRJ[] = {0.0f, 0.0f, 0.0f, 0.0f};
GLfloat materialDiffuse_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
GLfloat materialSpecular_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
GLfloat materialShininess_RRJ = 128.0f;

//For Sphere
GLuint vao_Sphere_RRJ;
GLuint vbo_Sphere_Position_RRJ;
GLuint vbo_Sphere_Normal_RRJ;
GLuint vbo_Sphere_Element_RRJ;
float sphere_vertices_RRJ[1146];
float sphere_normals_RRJ[1146];
float sphere_textures_RRJ[764];
unsigned short sphere_elements_RRJ[2280];
unsigned int gNumVertices_RRJ;
unsigned int gNumElements_RRJ;


int main(void){
	
	void CreateWindow(void);
	void initialize(void);
	void ToggleFullScreen(void);
	void resize(int, int);
	void display(void);
	void uninitialize(void);
	void update(void);

	int winWidth_RRJ = WIN_WIDTH;
	int winHeight_RRJ = WIN_HEIGHT;


	gbFile_RRJ = fopen("Log.txt", "w");
	if(gbFile_RRJ == NULL){
		printf("Log  Creation Failed!!\n");
		uninitialize();
		exit(1);
	}
	else
		fprintf(gbFile_RRJ, "Log Created!!\n");


	CreateWindow();
	initialize();
	ToggleFullScreen();

	//For Event Loop
	XEvent event_RRJ;
	KeySym keysym_RRJ;
	bool bDone_RRJ = false;

	while(bDone_RRJ == false){
		while(XPending(gpDisplay_RRJ)){
	
			XNextEvent(gpDisplay_RRJ, &event_RRJ);
			switch(event_RRJ.type){
				case MapNotify:
					break;
				case Expose:
					break;
				case MotionNotify:
					break;
				case DestroyNotify:
					break;

				case ConfigureNotify:
					winWidth_RRJ = event_RRJ.xconfigure.width;
					winHeight_RRJ = event_RRJ.xconfigure.height;
					resize(winWidth_RRJ, winHeight_RRJ);
					break;

				case KeyPress:
					keysym_RRJ = XkbKeycodeToKeysym(gpDisplay_RRJ, event_RRJ.xkey.keycode, 0, 0);
					switch(keysym_RRJ){
						case XK_Escape:
							if(bIsFullScreen_RRJ == false){
								ToggleFullScreen();
								bIsFullScreen_RRJ = true;
							}
							else{
								ToggleFullScreen();
								bIsFullScreen_RRJ = false;
							}
							break;
							
							
						
						case XK_F:
						case XK_f:
							iWhichLight_RRJ = 2;
							break;

						case XK_L:
						case XK_l:
							if(bLight_RRJ == false)
								bLight_RRJ = true;
							else
								bLight_RRJ = false;
							break;
							
							
						case XK_V:
						case XK_v:
							iWhichLight_RRJ = 1;
							break;
							
						case XK_Q:
						case XK_q:
							bDone_RRJ = true;
							break;

						default:
							break;
					}
					break;

				case ButtonPress:
					switch(event_RRJ.xbutton.button){
						case 1:
							break;
						case 2:
							break;
						case 3:
							break;
						default:
							break;
					}
					break;

				case 33:
					bDone_RRJ = true;
					break;

				default:
					break;
			}

		}
		update();
		display();
	}

	uninitialize();
	return(0);
}

void ToggleFullScreen(void){
	
	Atom wm_state_RRJ;
	Atom fullscreen_RRJ;
	XEvent xev_RRJ = {0};

	wm_state_RRJ = XInternAtom(gpDisplay_RRJ, "_NET_WM_STATE", False);
	memset(&xev_RRJ, 0, sizeof(XEvent));

	xev_RRJ.type = ClientMessage;
	xev_RRJ.xclient.window = gWindow_RRJ;
	xev_RRJ.xclient.message_type = wm_state_RRJ;
	xev_RRJ.xclient.format = 32;
	xev_RRJ.xclient.data.l[0] = bIsFullScreen_RRJ ? 0 : 1;
	
	fullscreen_RRJ = XInternAtom(gpDisplay_RRJ, "_NET_WM_STATE_FULLSCREEN", False);
	xev_RRJ.xclient.data.l[1] = fullscreen_RRJ;

	XSendEvent(gpDisplay_RRJ,
		RootWindow(gpDisplay_RRJ, gpXVisualInfo_RRJ->screen),
		False,
		StructureNotifyMask,
		&xev_RRJ);
}

void CreateWindow(void){

	void uninitialize(void);

	XSetWindowAttributes winAttrib_RRJ;
	int defaultScreen_RRJ;
	int styleMask_RRJ;

	GLXFBConfig *pGLXFBConfig_RRJ = NULL;
	GLXFBConfig bestFBConfig_RRJ;
	int iNumberOfFBConfig_RRJ = 0;
	XVisualInfo *pTempXVisualInfo_RRJ = NULL;


	static int frameBufferAttributes_RRJ[] = {
		GLX_X_RENDERABLE, True,
		GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
		GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
		GLX_RENDER_TYPE, GLX_RGBA_BIT,
		GLX_RED_SIZE, 8,
		GLX_GREEN_SIZE, 8,
		GLX_BLUE_SIZE, 8,
		GLX_ALPHA_SIZE, 8,
		GLX_DEPTH_SIZE, 24,
		GLX_DOUBLEBUFFER, True,
		None
	};

	gpDisplay_RRJ = XOpenDisplay(NULL);
	if(gpDisplay_RRJ == NULL){
		fprintf(gbFile_RRJ, "XOpenDisplay() Failed!!\n");
		uninitialize();
		exit(1);
	}

	defaultScreen_RRJ = XDefaultScreen(gpDisplay_RRJ);

	pGLXFBConfig_RRJ = glXChooseFBConfig(gpDisplay_RRJ, defaultScreen_RRJ, frameBufferAttributes_RRJ, &iNumberOfFBConfig_RRJ);
	if(pGLXFBConfig_RRJ == NULL){
		fprintf(gbFile_RRJ, "glXChooseFBConfig() Failed!!\n");
		uninitialize();
		exit(1);
	}

	
	int bestFrameBufferConfig_RRJ = -1;
	int bestNumberOfSamples_RRJ = -1;
	int worstFrameBufferConfig_RRJ = -1;
	int worstNumberOfSamples_RRJ = -1;


	fprintf(gbFile_RRJ, "FBConfig: %d\n", iNumberOfFBConfig_RRJ);

	for(int i = 0; i < iNumberOfFBConfig_RRJ; i++){
		pTempXVisualInfo_RRJ = glXGetVisualFromFBConfig(gpDisplay_RRJ, pGLXFBConfig_RRJ[i]);
		if(pTempXVisualInfo_RRJ){
			int samples, sampleBuffers;

			glXGetFBConfigAttrib(gpDisplay_RRJ, pGLXFBConfig_RRJ[i], GLX_SAMPLES, &samples);
			glXGetFBConfigAttrib(gpDisplay_RRJ, pGLXFBConfig_RRJ[i], GLX_SAMPLE_BUFFERS, &sampleBuffers);

			if(bestFrameBufferConfig_RRJ < 0 || sampleBuffers && samples > bestNumberOfSamples_RRJ){
				bestFrameBufferConfig_RRJ = i;
				bestNumberOfSamples_RRJ = samples;
			}

			if(worstFrameBufferConfig_RRJ < 0 || sampleBuffers && samples < worstNumberOfSamples_RRJ){
				worstFrameBufferConfig_RRJ = i;
				worstNumberOfSamples_RRJ = samples;
			}
		}
		XFree(pTempXVisualInfo_RRJ);
		pTempXVisualInfo_RRJ = NULL;
	}

	bestFBConfig_RRJ = pGLXFBConfig_RRJ[bestFrameBufferConfig_RRJ];
	gGLXFBConfig_RRJ = bestFBConfig_RRJ;
	XFree(pGLXFBConfig_RRJ);
	pGLXFBConfig_RRJ = NULL;

	gpXVisualInfo_RRJ = glXGetVisualFromFBConfig(gpDisplay_RRJ, bestFBConfig_RRJ);
	if(gpXVisualInfo_RRJ == NULL){
		fprintf(gbFile_RRJ, "glXGetVisualFromFBConfig() Failed!!\n");
		uninitialize();
		exit(1);
	}

	winAttrib_RRJ.border_pixel = 0;
	winAttrib_RRJ.border_pixmap = 0;
	winAttrib_RRJ.background_pixel = BlackPixel(gpDisplay_RRJ, defaultScreen_RRJ);
	winAttrib_RRJ.background_pixmap = 0;
	winAttrib_RRJ.colormap = XCreateColormap(gpDisplay_RRJ,
				RootWindow(gpDisplay_RRJ, gpXVisualInfo_RRJ->screen),
				gpXVisualInfo_RRJ->visual,
				AllocNone);
	gColormap_RRJ = winAttrib_RRJ.colormap;
	winAttrib_RRJ.event_mask = ExposureMask | VisibilityChangeMask | PointerMotionMask |
				KeyPressMask | ButtonPressMask | StructureNotifyMask;

	styleMask_RRJ = CWBorderPixel | CWBackPixel | CWEventMask | CWColormap;

	gWindow_RRJ = XCreateWindow(gpDisplay_RRJ,
			RootWindow(gpDisplay_RRJ, gpXVisualInfo_RRJ->screen),
			0, 0,
			WIN_WIDTH, WIN_HEIGHT,
			0,
			gpXVisualInfo_RRJ->depth,
			InputOutput,
			gpXVisualInfo_RRJ->visual,
			styleMask_RRJ,
			&winAttrib_RRJ);

	if(!gWindow_RRJ){
		fprintf(gbFile_RRJ, "XCreateWindow() Failed!!\n");
		uninitialize();
		exit(1);
	}
	
	XStoreName(gpDisplay_RRJ, gWindow_RRJ, "22-3LightsOnSphere");

	Atom windowManagerDelete = XInternAtom(gpDisplay_RRJ, "WM_DELETE_WINDOW", True);
	XSetWMProtocols(gpDisplay_RRJ, gWindow_RRJ, &windowManagerDelete, 1);

	XMapWindow(gpDisplay_RRJ, gWindow_RRJ);
}

void initialize(void){
	
	void uninitialize(void);
	void resize(int, int);
	void fill_LightsData(void);
	
	
	//Shader Object;
	GLuint vertexShaderObject_PV_RRJ;
	GLuint fragmentShaderObject_PV_RRJ;

	GLuint vertexShaderObject_PF_RRJ;
	GLuint fragmentShaderObject_PF_RRJ;
	
	
	glXCreateContextAttribsARB_RRJ = (glXCreateContextAttribsARBProc)glXGetProcAddressARB((GLubyte*)"glXCreateContextAttribsARB");
	if(glXCreateContextAttribsARB_RRJ == NULL){
		fprintf(gbFile_RRJ, "glXGetProcAddressARB() Failed!!\n");
		uninitialize();
		exit(1);
	}

	const int attributes_RRJ[] = {
		GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
		GLX_CONTEXT_MINOR_VERSION_ARB, 5,
		GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
		None
	};

	gGLXContext_RRJ = glXCreateContextAttribsARB_RRJ(gpDisplay_RRJ, gGLXFBConfig_RRJ, NULL, True, attributes_RRJ);
	if(gGLXContext_RRJ == NULL){
		fprintf(gbFile_RRJ, "glXCreateContextAttribsARB_RRJ() Failed!!\n");
		fprintf(gbFile_RRJ, "Getting Context give by System!!\n");

		const int attribs_RRJ[] = {
			GLX_CONTEXT_MAJOR_VERSION_ARB, 1,
			GLX_CONTEXT_MINOR_VERSION_ARB, 0,
			None
		};

		gGLXContext_RRJ = glXCreateContextAttribsARB_RRJ(gpDisplay_RRJ, gGLXFBConfig_RRJ, NULL, True, attribs_RRJ);
	}

	if(!glXIsDirect(gpDisplay_RRJ, gGLXContext_RRJ))
		fprintf(gbFile_RRJ, "S/W Context!!\n");
	else
		fprintf(gbFile_RRJ, "H/W Context!!\n");

	glXMakeCurrent(gpDisplay_RRJ, gWindow_RRJ, gGLXContext_RRJ);



	GLenum result_RRJ;
	result_RRJ = glewInit();
	if(result_RRJ != GLEW_OK){
		fprintf(gbFile_RRJ, "glewInit() Failed!!\n");
		uninitialize();
		exit(1);
	}


	/********** Vertex Shader Per Vertex *********/
	vertexShaderObject_PV_RRJ = glCreateShader(GL_VERTEX_SHADER);
	const char *szVertexShaderCode_PV_RRJ =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vNormals;" \
		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \

		"uniform vec3 u_Red_La;" \
		"uniform vec3 u_Red_Ld;" \
		"uniform vec3 u_Red_Ls;" \
		"uniform vec4 u_Red_light_position;" \

		"uniform vec3 u_Green_La;" \
		"uniform vec3 u_Green_Ld;" \
		"uniform vec3 u_Green_Ls;" \
		"uniform vec4 u_Green_light_position;" \

		"uniform vec3 u_Blue_La;" \
		"uniform vec3 u_Blue_Ld;" \
		"uniform vec3 u_Blue_Ls;" \
		"uniform vec4 u_Blue_light_position;" \

		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_shininess;" \
		"uniform int u_L_keypress;" \
		"out vec3 phongLight;" \
		"void main(void)" \
		"{" \
		"if(u_L_keypress == 1){" \
		"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" \

		"vec3 RedSource = normalize(vec3(u_Red_light_position - eyeCoordinate));" \
		"vec3 GreenSource = normalize(vec3(u_Green_light_position - eyeCoordinate));" \
		"vec3 BlueSource = normalize(vec3(u_Blue_light_position - eyeCoordinate));" \

		"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" \
		"vec3 Normal = normalize(vec3(normalMatrix * vNormals));" \

		"float SRed_Dot_N = max(dot(RedSource, Normal), 0.0);" \
		"float SGreen_Dot_N = max(dot(GreenSource, Normal), 0.0);" \
		"float SBlue_Dot_N = max(dot(BlueSource, Normal), 0.0);" \

		"vec3 RedReflection = reflect(-RedSource, Normal);" \
		"vec3 GreenReflection = reflect(-GreenSource, Normal);" \
		"vec3 BlueReflection = reflect(-BlueSource, Normal);" \

		"vec3 Viewer = normalize(vec3(-eyeCoordinate.xyz));" \


		"float RRed_Dot_V = max(dot(RedReflection, Viewer), 0.0);" \
		"vec3 ambientRed = u_Red_La * u_Ka;" \
		"vec3 diffuseRed = u_Red_Ld * u_Kd * SRed_Dot_N;" \
		"vec3 specularRed = u_Red_Ls * u_Ks * max(pow(RRed_Dot_V, u_shininess), 0.0);" \
		"vec3 Red = ambientRed + diffuseRed + specularRed;" \


		"float RGreen_Dot_V = max(dot(GreenReflection, Viewer), 0.0);" \
		"vec3 ambientGreen = u_Green_La * u_Ka;" \
		"vec3 diffuseGreen = u_Green_Ld * u_Kd * SGreen_Dot_N;" \
		"vec3 specularGreen = u_Green_Ls * u_Ks * max(pow(RGreen_Dot_V, u_shininess), 0.0);" \
		"vec3 Green = ambientGreen + diffuseGreen + specularGreen;" \


		"float RBlue_Dot_V = max(dot(BlueReflection, Viewer), 0.0);" \
		"vec3 ambientBlue = u_Blue_La * u_Ka;" \
		"vec3 diffuseBlue = u_Blue_Ld * u_Kd * SBlue_Dot_N;" \
		"vec3 specularBlue = u_Blue_Ls * u_Ks * max(pow(RBlue_Dot_V, u_shininess), 0.0);" \
		"vec3 Blue = ambientBlue + diffuseBlue + specularBlue;" \

		"phongLight = Red + Green + Blue;" \


		"}" \
		"else{" \
		"phongLight = vec3(1.0, 1.0, 1.0);" \
		"}" \
		"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
		"}";

	glShaderSource(vertexShaderObject_PV_RRJ, 1, (const GLchar**)&szVertexShaderCode_PV_RRJ, NULL);

	glCompileShader(vertexShaderObject_PV_RRJ);

	GLint iShaderCompileStatus_RRJ;
	GLint iInfoLogLength_RRJ;
	GLchar *szInfoLog_RRJ = NULL;
	glGetShaderiv(vertexShaderObject_PV_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(vertexShaderObject_PV_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(vertexShaderObject_PV_RRJ, iInfoLogLength_RRJ,
					&written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Per Vertex Lighting Vertex Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(0);
			}
		}
	}


	/********** Fragment Shader Per Vertex *********/
	fragmentShaderObject_PV_RRJ = glCreateShader(GL_FRAGMENT_SHADER);
	const char* szFragmentShaderCode_PV_RRJ =
		"#version 450 core" \
		"\n" \
		"in vec3 phongLight;" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
		"FragColor = vec4(phongLight, 1.0);" \
		"}";


	glShaderSource(fragmentShaderObject_PV_RRJ, 1,
		(const GLchar**)&szFragmentShaderCode_PV_RRJ, NULL);

	glCompileShader(fragmentShaderObject_PV_RRJ);

	iShaderCompileStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(fragmentShaderObject_PV_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(fragmentShaderObject_PV_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject_PV_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Per Vertex Lighting Fragment Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(0);
			}
		}
	}


	/********* Program Object For Per Vertex Lighting **********/
	shaderProgramObject_PV_RRJ = glCreateProgram();

	glAttachShader(shaderProgramObject_PV_RRJ, vertexShaderObject_PV_RRJ);
	glAttachShader(shaderProgramObject_PV_RRJ, fragmentShaderObject_PV_RRJ);

	glBindAttribLocation(shaderProgramObject_PV_RRJ, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(shaderProgramObject_PV_RRJ, AMC_ATTRIBUTE_NORMAL, "vNormals");

	glLinkProgram(shaderProgramObject_PV_RRJ);

	GLint iProgramLinkingStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;
	glGetProgramiv(shaderProgramObject_PV_RRJ, GL_LINK_STATUS, &iProgramLinkingStatus_RRJ);
	if (iProgramLinkingStatus_RRJ == GL_FALSE) {
		glGetProgramiv(shaderProgramObject_PV_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetProgramInfoLog(shaderProgramObject_PV_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Shader Program Object Linking Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(0);
			}
		}
	}

	modelMatrixUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_model_matrix");
	viewMatrixUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_view_matrix");
	projectionMatrixUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_projection_matrix");
	
	red_LaUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Red_La");
	red_LdUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Red_Ld");
	red_LsUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Red_Ls");
	red_lightPositionUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Red_light_position");

	green_LaUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Green_La");
	green_LdUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Green_Ld");
	green_LsUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Green_Ls");
	green_lightPositionUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Green_light_position");

	blue_LaUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Blue_La");
	blue_LdUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Blue_Ld");
	blue_LsUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Blue_Ls");
	blue_lightPositionUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Blue_light_position");


	KaUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ka");
	KdUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Kd");
	KsUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ks");
	shininessUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_shininess");
	LKeyPressUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_L_keypress");


	fprintf(gbFile_RRJ, "%d/%d/%d\n", red_LdUniform_PV_RRJ, green_lightPositionUniform_PV_RRJ, blue_lightPositionUniform_PV_RRJ);





	/********** Vertex Shader Per Fragment Lighting *********/
	vertexShaderObject_PF_RRJ = glCreateShader(GL_VERTEX_SHADER);
	const char *szVertexShaderCode_PF_RRJ =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vNormals;" \

		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \

		"uniform vec4 u_Red_light_position;" \
		"uniform vec4 u_Green_light_position;" \
		"uniform vec4 u_Blue_light_position;" \

		"out vec3 lightDirectionRed_VS;" \
		"out vec3 lightDirectionGreen_VS;" \
		"out vec3 lightDirectionBlue_VS;" \

		"out vec3 Normal_VS;" \
		"out vec3 Viewer_VS;" \

		"void main(void)" \
		"{"
		"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" \
		"lightDirectionRed_VS = vec3(u_Red_light_position - eyeCoordinate);" \
		"lightDirectionGreen_VS = vec3(u_Green_light_position - eyeCoordinate);" \
		"lightDirectionBlue_VS = vec3(u_Blue_light_position - eyeCoordinate);" \
		
		"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" \
		"Normal_VS = vec3(normalMatrix * vNormals);" \
		"Viewer_VS = vec3(-eyeCoordinate);" \
		"gl_Position =	u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
		"}";


	glShaderSource(vertexShaderObject_PF_RRJ, 1, (const GLchar**)&szVertexShaderCode_PF_RRJ, NULL);

	glCompileShader(vertexShaderObject_PF_RRJ);


	glGetShaderiv(vertexShaderObject_PF_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(vertexShaderObject_PF_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(vertexShaderObject_PF_RRJ, iInfoLogLength_RRJ,
					&written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Per Fragment Lighting Vertex Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(0);
			}
		}
	}


	/********** Fragment Shader Per Fragment *********/
	fragmentShaderObject_PF_RRJ = glCreateShader(GL_FRAGMENT_SHADER);
	const char* szFragmentShaderCode_PF_RRJ =
		"#version 450 core" \
		"\n" \

		"in vec3 lightDirectionRed_VS;" \
		"in vec3 lightDirectionGreen_VS;" \
		"in vec3 lightDirectionBlue_VS;" \

		"in vec3 Normal_VS;" \
		"in vec3 Viewer_VS;" \

		"uniform vec3 u_Red_La;" \
		"uniform vec3 u_Red_Ld;" \
		"uniform vec3 u_Red_Ls;" \

		"uniform vec3 u_Green_La;" \
		"uniform vec3 u_Green_Ld;" \
		"uniform vec3 u_Green_Ls;" \

		"uniform vec3 u_Blue_La;" \
		"uniform vec3 u_Blue_Ld;" \
		"uniform vec3 u_Blue_Ls;" \


		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_shininess;" \
		"uniform int u_L_keypress;" \


		"out vec4 FragColor;" \


		"void main(void)" \
		"{" \
		"vec3 phongLight;" \
		"if(u_L_keypress == 1){" \
		"vec3 RedLightDirection = normalize(lightDirectionRed_VS);" \
		"vec3 GreenLightDirection = normalize(lightDirectionGreen_VS);" \
		"vec3 BlueLightDirection = normalize(lightDirectionBlue_VS);" \
		
		
		"vec3 Normal = normalize(Normal_VS);" \

		"float LRed_Dot_N = max(dot(RedLightDirection, Normal), 0.0);" \
		"float LGreen_Dot_N = max(dot(GreenLightDirection, Normal), 0.0);" \
		"float LBlue_Dot_N = max(dot(BlueLightDirection, Normal), 0.0);" \
		
		
		"vec3 RedReflection = reflect(-RedLightDirection, Normal);" \
		"vec3 GreenReflection = reflect(-GreenLightDirection, Normal);" \
		"vec3 BlueReflection = reflect(-BlueLightDirection, Normal);" \
		
		
		"vec3 Viewer = normalize(Viewer_VS);" \


		"float RRed_Dot_V = max(dot(RedReflection, Viewer), 0.0);" \
		"float RGreen_Dot_V = max(dot(GreenReflection, Viewer), 0.0);" \
		"float RBlue_Dot_V = max(dot(BlueReflection, Viewer), 0.0);" \



		"vec3 ambientRed = u_Red_La * u_Ka;" \
		"vec3 diffuseRed = u_Red_Ld * u_Kd * LRed_Dot_N;" \
		"vec3 specularRed = u_Red_Ls * u_Ks * pow(RRed_Dot_V, u_shininess);" \
		"vec3 Red = ambientRed + diffuseRed + specularRed;" \


		"vec3 ambientGreen = u_Green_La * u_Ka;" \
		"vec3 diffuseGreen = u_Green_Ld * u_Kd * LGreen_Dot_N;" \
		"vec3 specularGreen = u_Green_Ls * u_Ks * pow(RGreen_Dot_V, u_shininess);" \
		"vec3 Green = ambientGreen + diffuseGreen + specularGreen;" \

		"vec3 ambientBlue = u_Blue_La * u_Ka;" \
		"vec3 diffuseBlue = u_Blue_Ld * u_Kd * LBlue_Dot_N;" \
		"vec3 specularBlue = u_Blue_Ls * u_Ks * pow(RBlue_Dot_V, u_shininess);" \
		"vec3 Blue = ambientBlue + diffuseBlue + specularBlue;" \

		"phongLight = Red + Green + Blue;" \

		"}" \
		"else{" \
		"phongLight = vec3(1.0, 1.0, 1.0);" \
		"}" \
		"FragColor = vec4(phongLight, 0.0);" \
		"}";


	glShaderSource(fragmentShaderObject_PF_RRJ, 1,
		(const GLchar**)&szFragmentShaderCode_PF_RRJ, NULL);

	glCompileShader(fragmentShaderObject_PF_RRJ);

	iShaderCompileStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(fragmentShaderObject_PF_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(fragmentShaderObject_PF_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject_PF_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Per Fragment Lighting Fragment Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(0);
			}
		}
	}


	/********* Program Object For Per Fragment Lighting **********/
	shaderProgramObject_PF_RRJ = glCreateProgram();

	glAttachShader(shaderProgramObject_PF_RRJ, vertexShaderObject_PF_RRJ);
	glAttachShader(shaderProgramObject_PF_RRJ, fragmentShaderObject_PF_RRJ);

	glBindAttribLocation(shaderProgramObject_PF_RRJ, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(shaderProgramObject_PF_RRJ, AMC_ATTRIBUTE_NORMAL, "vNormals");

	glLinkProgram(shaderProgramObject_PF_RRJ);


	glGetProgramiv(shaderProgramObject_PF_RRJ, GL_LINK_STATUS, &iProgramLinkingStatus_RRJ);
	if (iProgramLinkingStatus_RRJ == GL_FALSE) {
		glGetProgramiv(shaderProgramObject_PF_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetProgramInfoLog(shaderProgramObject_PF_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Shader Program Object Linking Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(0);
			}
		}
	}

	modelMatrixUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_model_matrix");
	viewMatrixUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_view_matrix");
	projectionMatrixUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_projection_matrix");
	
	red_LaUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Red_La");
	red_LdUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Red_Ld");
	red_LsUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Red_Ls");
	red_lightPositionUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Red_light_position");

	green_LaUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Green_La");
	green_LdUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Green_Ld");
	green_LsUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Green_Ls");
	green_lightPositionUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Green_light_position");

	blue_LaUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Blue_La");
	blue_LdUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Blue_Ld");
	blue_LsUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Blue_Ls");
	blue_lightPositionUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Blue_light_position");

	KaUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Ka");
	KdUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Kd");
	KsUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Ks");
	shininessUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_shininess");
	LKeyPressUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_L_keypress");






	/********** Position, Normal and Elements **********/
	getSphereVertexData(sphere_vertices_RRJ, sphere_normals_RRJ, sphere_textures_RRJ, sphere_elements_RRJ);
	gNumVertices_RRJ = getNumberOfSphereVertices();
	gNumElements_RRJ = getNumberOfSphereElements();



	/********** Sphere Vao **********/
	glGenVertexArrays(1, &vao_Sphere_RRJ);
	glBindVertexArray(vao_Sphere_RRJ);

	/********** Position **********/
	glGenBuffers(1, &vbo_Sphere_Position_RRJ);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Sphere_Position_RRJ);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(sphere_vertices_RRJ),
		sphere_vertices_RRJ,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/********** Normals **********/
	glGenBuffers(1, &vbo_Sphere_Normal_RRJ);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Sphere_Normal_RRJ);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(sphere_normals_RRJ),
		sphere_normals_RRJ,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	/********** Element Vbo **********/
	glGenBuffers(1, &vbo_Sphere_Element_RRJ);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(sphere_elements_RRJ), sphere_elements_RRJ, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


	glBindVertexArray(0);
	
	fill_LightsData();
	

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	perspectiveProjectionMatrix_RRJ = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);

}


void fill_LightsData(void){

	//Red Light

	lights_RRJ[0].lightAmbient_RRJ[0] = 0.0f;
	lights_RRJ[0].lightAmbient_RRJ[1] = 0.0f;
	lights_RRJ[0].lightAmbient_RRJ[2] = 0.0f;
	lights_RRJ[0].lightAmbient_RRJ[3] = 0.0f;
	
	lights_RRJ[0].lightDiffuse_RRJ[0] = 1.0f;
	lights_RRJ[0].lightDiffuse_RRJ[1] = 0.0f;
	lights_RRJ[0].lightDiffuse_RRJ[2] = 0.0f;
	lights_RRJ[0].lightDiffuse_RRJ[3] = 1.0f;
	
	lights_RRJ[0].lightSpecular_RRJ[0] = 1.0f;
	lights_RRJ[0].lightSpecular_RRJ[1] = 0.0f;
	lights_RRJ[0].lightSpecular_RRJ[2] = 0.0f;
	lights_RRJ[0].lightSpecular_RRJ[3] = 1.0f;
	
	lights_RRJ[0].lightPosition_RRJ[0] = 0.0f;
	lights_RRJ[0].lightPosition_RRJ[1] = 0.0f;
	lights_RRJ[0].lightPosition_RRJ[2] = 0.0f;
	lights_RRJ[0].lightPosition_RRJ[3] = 1.0f;
	
	//Green Light
	lights_RRJ[1].lightAmbient_RRJ[0] = 0.0f;
	lights_RRJ[1].lightAmbient_RRJ[1] = 0.0f;
	lights_RRJ[1].lightAmbient_RRJ[2] = 0.0f;
	lights_RRJ[1].lightAmbient_RRJ[3] = 0.0f;
	
	lights_RRJ[1].lightDiffuse_RRJ[0] = 0.0f;
	lights_RRJ[1].lightDiffuse_RRJ[1] = 1.0f;
	lights_RRJ[1].lightDiffuse_RRJ[2] = 0.0f;
	lights_RRJ[1].lightDiffuse_RRJ[3] = 1.0f;
	
	lights_RRJ[1].lightSpecular_RRJ[0] = 0.0f;
	lights_RRJ[1].lightSpecular_RRJ[1] = 1.0f;
	lights_RRJ[1].lightSpecular_RRJ[2] = 0.0f;
	lights_RRJ[1].lightSpecular_RRJ[3] = 1.0f;
	
	lights_RRJ[1].lightPosition_RRJ[0] = 0.0f;
	lights_RRJ[1].lightPosition_RRJ[1] = 0.0f;
	lights_RRJ[1].lightPosition_RRJ[2] = 0.0f;
	lights_RRJ[1].lightPosition_RRJ[3] = 1.0f;
	
	
	//Blue Light
	lights_RRJ[2].lightAmbient_RRJ[0] = 0.0f;
	lights_RRJ[2].lightAmbient_RRJ[1] = 0.0f;
	lights_RRJ[2].lightAmbient_RRJ[2] = 0.0f;
	lights_RRJ[2].lightAmbient_RRJ[3] = 0.0f;
	
	lights_RRJ[2].lightDiffuse_RRJ[0] = 0.0f;
	lights_RRJ[2].lightDiffuse_RRJ[1] = 0.0f;
	lights_RRJ[2].lightDiffuse_RRJ[2] = 1.0f;
	lights_RRJ[2].lightDiffuse_RRJ[3] = 1.0f;
	
	lights_RRJ[2].lightSpecular_RRJ[0] = 0.0f;
	lights_RRJ[2].lightSpecular_RRJ[1] = 0.0f;
	lights_RRJ[2].lightSpecular_RRJ[2] = 1.0f;
	lights_RRJ[2].lightSpecular_RRJ[3] = 1.0f;
	
	lights_RRJ[2].lightPosition_RRJ[0] = 0.0f;
	lights_RRJ[2].lightPosition_RRJ[1] = 0.0f;
	lights_RRJ[2].lightPosition_RRJ[2] = 0.0f;
	lights_RRJ[2].lightPosition_RRJ[3] = 1.0f;
			
}


void uninitialize(void) {

	GLXContext currentContext_RRJ = glXGetCurrentContext();

	if (vbo_Sphere_Element_RRJ) {
		glDeleteBuffers(1, &vbo_Sphere_Element_RRJ);
		vbo_Sphere_Element_RRJ = 0;
	}

	if (vbo_Sphere_Normal_RRJ) {
		glDeleteBuffers(1, &vbo_Sphere_Normal_RRJ);
		vbo_Sphere_Normal_RRJ = 0;
	}

	if (vbo_Sphere_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Sphere_Position_RRJ);
		vbo_Sphere_Position_RRJ = 0;
	}

	if (vao_Sphere_RRJ) {
		glDeleteVertexArrays(1, &vao_Sphere_RRJ);
		vao_Sphere_RRJ = 0;
	}


	GLsizei ShaderCount_RRJ;
	GLsizei ShaderNumber_RRJ;

	if (shaderProgramObject_PV_RRJ) {
		glUseProgram(shaderProgramObject_PV_RRJ);

		glGetProgramiv(shaderProgramObject_PV_RRJ, GL_ATTACHED_SHADERS, &ShaderCount_RRJ);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount_RRJ);
		if (pShader) {
			glGetAttachedShaders(shaderProgramObject_PV_RRJ, ShaderCount_RRJ, &ShaderCount_RRJ, pShader);
			for (ShaderNumber_RRJ = 0; ShaderNumber_RRJ < ShaderCount_RRJ; ShaderNumber_RRJ++) {
				glDetachShader(shaderProgramObject_PV_RRJ, pShader[ShaderNumber_RRJ]);
				glDeleteShader(pShader[ShaderNumber_RRJ]);
				pShader[ShaderNumber_RRJ] = 0;
			}
			free(pShader);
			pShader = NULL;
		}
		glDeleteProgram(shaderProgramObject_PV_RRJ);
		shaderProgramObject_PV_RRJ = 0;
		glUseProgram(0);
	}


	ShaderCount_RRJ = 0;
	ShaderNumber_RRJ = 0;
	if (shaderProgramObject_PF_RRJ) {
		glUseProgram(shaderProgramObject_PF_RRJ);

		glGetProgramiv(shaderProgramObject_PF_RRJ, GL_ATTACHED_SHADERS, &ShaderCount_RRJ);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount_RRJ);
		if (pShader) {
			glGetAttachedShaders(shaderProgramObject_PF_RRJ, ShaderCount_RRJ, &ShaderCount_RRJ, pShader);
			for (ShaderNumber_RRJ = 0; ShaderNumber_RRJ < ShaderCount_RRJ; ShaderNumber_RRJ++) {
				glDetachShader(shaderProgramObject_PF_RRJ, pShader[ShaderNumber_RRJ]);
				glDeleteShader(pShader[ShaderNumber_RRJ]);
				pShader[ShaderNumber_RRJ] = 0;
			}
			free(pShader);
			pShader = NULL;
		}
		glDeleteProgram(shaderProgramObject_PF_RRJ);
		shaderProgramObject_PF_RRJ = 0;
		glUseProgram(0);
	}
	

	if(currentContext_RRJ != NULL && currentContext_RRJ == gGLXContext_RRJ)
		glXMakeCurrent(gpDisplay_RRJ, 0, 0);
		
	if(glXCreateContextAttribsARB_RRJ)
		glXCreateContextAttribsARB_RRJ = NULL;
	
	
	if(gGLXContext_RRJ)
		glXDestroyContext(gpDisplay_RRJ, gGLXContext_RRJ);
		
	if(gGLXFBConfig_RRJ)
		gGLXFBConfig_RRJ = 0;
		
	if(gWindow_RRJ)
		XDestroyWindow(gpDisplay_RRJ, gWindow_RRJ);
		
	if(gColormap_RRJ)
		XFreeColormap(gpDisplay_RRJ, gColormap_RRJ);
		
	
	if(gpXVisualInfo_RRJ){
		XFree(gpXVisualInfo_RRJ);
		gpXVisualInfo_RRJ = NULL;
	}
	
	if(gpDisplay_RRJ){
		XCloseDisplay(gpDisplay_RRJ);
		gpDisplay_RRJ = NULL;
	}
	
	if(gbFile_RRJ){
		fprintf(gbFile_RRJ, "Log Close!!\n");
		fclose(gbFile_RRJ);
		gbFile_RRJ = NULL;
	}
	
}


void resize(int width, int height) {
	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	perspectiveProjectionMatrix_RRJ = mat4::identity();
	perspectiveProjectionMatrix_RRJ = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void display(void) {

	mat4 translateMatrix_RRJ;
	mat4 modelMatrix_RRJ;
	mat4 viewMatrix_RRJ;
	
	void rotateRedLight(GLfloat);
	void rotateGreenLight(GLfloat);
	void rotateBlueLight(GLfloat);
	
	

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	translateMatrix_RRJ = mat4::identity();
	modelMatrix_RRJ = mat4::identity();
	viewMatrix_RRJ = mat4::identity();

	translateMatrix_RRJ = translate(0.0f, 0.0f, -1.50f);
	modelMatrix_RRJ = modelMatrix_RRJ * translateMatrix_RRJ;


	if (iWhichLight_RRJ == 1){
		glUseProgram(shaderProgramObject_PV_RRJ);
		
		glUniformMatrix4fv(modelMatrixUniform_PV_RRJ, 1, GL_FALSE, modelMatrix_RRJ);
		glUniformMatrix4fv(viewMatrixUniform_PV_RRJ, 1, GL_FALSE, viewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, perspectiveProjectionMatrix_RRJ);
			
			
		if (bLight_RRJ == true) {


			rotateRedLight(angle_red_RRJ);
			rotateGreenLight(angle_green_RRJ);
			rotateBlueLight(angle_blue_RRJ);
			
			glUniform1i(LKeyPressUniform_PV_RRJ, 1);

			glUniform3fv(red_LaUniform_PV_RRJ, 1, lights_RRJ[0].lightAmbient_RRJ);
			glUniform3fv(red_LdUniform_PV_RRJ, 1, lights_RRJ[0].lightDiffuse_RRJ);
			glUniform3fv(red_LsUniform_PV_RRJ, 1, lights_RRJ[0].lightSpecular_RRJ);
			glUniform4fv(red_lightPositionUniform_PV_RRJ, 1, lights_RRJ[0].lightPosition_RRJ);

			glUniform3fv(green_LaUniform_PV_RRJ, 1, lights_RRJ[1].lightAmbient_RRJ);
			glUniform3fv(green_LdUniform_PV_RRJ, 1, lights_RRJ[1].lightDiffuse_RRJ);
			glUniform3fv(green_LsUniform_PV_RRJ, 1, lights_RRJ[1].lightSpecular_RRJ);
			glUniform4fv(green_lightPositionUniform_PV_RRJ, 1, lights_RRJ[1].lightPosition_RRJ);

			glUniform3fv(blue_LaUniform_PV_RRJ, 1, lights_RRJ[2].lightAmbient_RRJ);
			glUniform3fv(blue_LdUniform_PV_RRJ, 1, lights_RRJ[2].lightDiffuse_RRJ);
			glUniform3fv(blue_LsUniform_PV_RRJ, 1, lights_RRJ[2].lightSpecular_RRJ);
			glUniform4fv(blue_lightPositionUniform_PV_RRJ, 1, lights_RRJ[2].lightPosition_RRJ);


			glUniform3fv(KaUniform_PV_RRJ, 1, materialAmbient_RRJ);
			glUniform3fv(KdUniform_PV_RRJ, 1, materialDiffuse_RRJ);
			glUniform3fv(KsUniform_PV_RRJ, 1, materialSpecular_RRJ);
			glUniform1f(shininessUniform_PV_RRJ, materialShininess_RRJ);
		}
		else
			glUniform1i(LKeyPressUniform_PV_RRJ, 0);			
				
	}
	else{
		glUseProgram(shaderProgramObject_PF_RRJ);
		
		glUniformMatrix4fv(modelMatrixUniform_PF_RRJ, 1, GL_FALSE, modelMatrix_RRJ);
		glUniformMatrix4fv(viewMatrixUniform_PF_RRJ, 1, GL_FALSE, viewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, perspectiveProjectionMatrix_RRJ);
			
			
		if (bLight_RRJ == true) {
		
			rotateRedLight(angle_red_RRJ);
			rotateGreenLight(angle_green_RRJ);
			rotateBlueLight(angle_blue_RRJ);

	//fprintf(gbFile_RRJ, "%f/%f/%f/%f\n", lights_RRJ[0].lightPosition_RRJ[0], lights_RRJ[0].lightPosition_RRJ[1], lights_RRJ[0].lightPosition_RRJ[2], lights_RRJ[0].lightPosition_RRJ[3]);
			glUniform1i(LKeyPressUniform_PF_RRJ, 1);
			glUniform3fv(red_LaUniform_PF_RRJ, 1, lights_RRJ[0].lightAmbient_RRJ);
			glUniform3fv(red_LdUniform_PF_RRJ, 1, lights_RRJ[0].lightDiffuse_RRJ);
			glUniform3fv(red_LsUniform_PF_RRJ, 1, lights_RRJ[0].lightSpecular_RRJ);
			glUniform4fv(red_lightPositionUniform_PF_RRJ, 1, lights_RRJ[0].lightPosition_RRJ);

			glUniform3fv(green_LaUniform_PF_RRJ, 1, lights_RRJ[1].lightAmbient_RRJ);
			glUniform3fv(green_LdUniform_PF_RRJ, 1, lights_RRJ[1].lightDiffuse_RRJ);
			glUniform3fv(green_LsUniform_PF_RRJ, 1, lights_RRJ[1].lightSpecular_RRJ);
			glUniform4fv(green_lightPositionUniform_PF_RRJ, 1, lights_RRJ[1].lightPosition_RRJ);

			glUniform3fv(blue_LaUniform_PF_RRJ, 1, lights_RRJ[2].lightAmbient_RRJ);
			glUniform3fv(blue_LdUniform_PF_RRJ, 1, lights_RRJ[2].lightDiffuse_RRJ);
			glUniform3fv(blue_LsUniform_PF_RRJ, 1, lights_RRJ[2].lightSpecular_RRJ);
			glUniform4fv(blue_lightPositionUniform_PF_RRJ, 1, lights_RRJ[2].lightPosition_RRJ);
			
		
			glUniform3fv(KaUniform_PF_RRJ, 1, materialAmbient_RRJ);
			glUniform3fv(KdUniform_PF_RRJ, 1, materialDiffuse_RRJ);
			glUniform3fv(KsUniform_PF_RRJ, 1, materialSpecular_RRJ);
			glUniform1f(shininessUniform_PF_RRJ, materialShininess_RRJ);
		}
		else
			glUniform1i(LKeyPressUniform_PF_RRJ, 0);	
			
	}
	
	glBindVertexArray(vao_Sphere_RRJ);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ);
	glDrawElements(GL_TRIANGLES, gNumElements_RRJ, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	glUseProgram(0);
	
	glXSwapBuffers(gpDisplay_RRJ, gWindow_RRJ);
	
}


void rotateRedLight(float angle) {
	//fprintf(gbFile_RRJ, "In Red!!\n");
	lights_RRJ[0].lightPosition_RRJ[1] = (float)(5.0f * sin(angle));
	lights_RRJ[0].lightPosition_RRJ[2] = (float)(5.0f * cos(angle));

	lights_RRJ[0].lightDiffuse_RRJ[1] = (float)(0.5f * sin(angle));

	//fprintf(gbFile_RRJ, "%f/%f/%f/%f\n", lights_RRJ[0].lightPosition_RRJ[0], lights_RRJ[0].lightPosition_RRJ[1], lights_RRJ[0].lightPosition_RRJ[2], lights_RRJ[0].lightPosition_RRJ[3]);
}

void rotateGreenLight(float angle) {

	lights_RRJ[1].lightPosition_RRJ[0] = (float)(5.0f * sin(angle)); 
	lights_RRJ[1].lightPosition_RRJ[2] = (float)(5.0f * cos(angle)); 

}

void rotateBlueLight(float angle) {

	lights_RRJ[2].lightPosition_RRJ[0] = (float)(5.0f * cos(angle)); 
	lights_RRJ[2].lightPosition_RRJ[1] = (float)(5.0f * sin(angle));
}




void update(void) {

	//fprintf(gbFile_RRJ, "in update()\n");
	
	angle_red_RRJ = angle_red_RRJ + .020f;
	angle_green_RRJ = angle_green_RRJ + .02f;
	angle_blue_RRJ = angle_blue_RRJ + .02f;

	if (angle_red_RRJ > 360.0f)
		angle_red_RRJ = 0.0f;

	if (angle_green_RRJ > 360.0f)
		angle_green_RRJ = 0.0f;

	if (angle_blue_RRJ > 360.0f)
		angle_blue_RRJ = 0.0f;
 }


