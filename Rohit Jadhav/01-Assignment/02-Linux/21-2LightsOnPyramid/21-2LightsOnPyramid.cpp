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
GLuint LaRed_Uniform_PV_RRJ;
GLuint LdRed_Uniform_PV_RRJ;
GLuint LsRed_Uniform_PV_RRJ;
GLuint lightPositionRed_Uniform_PV_RRJ;
GLuint LaBlue_Uniform_PV_RRJ;
GLuint LdBlue_Uniform_PV_RRJ;
GLuint LsBlue_Uniform_PV_RRJ;
GLuint lightPositionBlue_Uniform_PV_RRJ;
GLuint KaUniform_PV_RRJ;
GLuint KdUniform_PV_RRJ;
GLuint KsUniform_PV_RRJ;
GLuint shininessUniform_PV_RRJ;
GLuint LKeyPressUniform_PV_RRJ;

//For Uniform Per Fragment Lighting
GLuint modelMatrixUniform_PF_RRJ;
GLuint viewMatrixUniform_PF_RRJ;
GLuint projectionMatrixUniform_PF_RRJ;
GLuint LaRed_Uniform_PF_RRJ;
GLuint LdRed_Uniform_PF_RRJ;
GLuint LsRed_Uniform_PF_RRJ;
GLuint lightPositionRed_Uniform_PF_RRJ;
GLuint LaBlue_Uniform_PF_RRJ;
GLuint LdBlue_Uniform_PF_RRJ;
GLuint LsBlue_Uniform_PF_RRJ;
GLuint lightPositionBlue_Uniform_PF_RRJ;
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

struct LIGHTS lights_RRJ[2];


//For Material
GLfloat materialAmbient_RRJ[] = {0.0f, 0.0f, 0.0f, 0.0f};
GLfloat materialDiffuse_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
GLfloat materialSpecular_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
GLfloat materialShininess_RRJ = 128.0f;


//For Pyramid
GLuint vao_Pyramid_RRJ;
GLuint vbo_Pyramid_Position_RRJ;
GLuint vbo_Pyramid_Normal_RRJ;
GLfloat angle_Pyramid_RRJ = 0.0f;


int main(void){
	
	void CreateWindow(void);
	void initialize(void);
	void ToggleFullScreen(void);
	void resize(int, int);
	void display(void);
	void uninitialize(void);


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
	
	XStoreName(gpDisplay_RRJ, gWindow_RRJ, "21-2LightsOnPyramid");

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
		"uniform vec3 u_La_Red;" \
		"uniform vec3 u_Ld_Red;" \
		"uniform vec3 u_Ls_Red;" \
		"uniform vec4 u_light_position_Red;" \
		"uniform vec3 u_La_Blue;" \
		"uniform vec3 u_Ld_Blue;" \
		"uniform vec3 u_Ls_Blue;" \
		"uniform vec4 u_light_position_Blue;" \
		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_shininess;" \
		"uniform int u_L_keypress;" \
		"out vec3 phongLight;"
		"void main(void)" \
		"{" \
		"vec3 phongRed_Light;" \
		"vec3 phongBlue_Light;" \
		"if(u_L_keypress == 1){" \
		"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" \

		"vec3 Source_Red = normalize(vec3(u_light_position_Red - eyeCoordinate));" \
		"vec3 Source_Blue = normalize(vec3(u_light_position_Blue - eyeCoordinate));" \

		"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" \
		"vec3 Normal = normalize(normalMatrix * vNormals);" \

		"float SRed_Dot_N = max(dot(Source_Red, Normal), 0.0);" \
		"float SBlue_Dot_N = max(dot(Source_Blue, Normal), 0.0);" \

		"vec3 Reflection_Red = reflect(-Source_Red, Normal);" \
		"vec3 Reflection_Blue = reflect(-Source_Blue, Normal);" \

		"vec3 Viewer = normalize(vec3(-eyeCoordinate.xyz));" \

		"float RRed_Dot_V = max(dot(Reflection_Red, Viewer), 0.0);" \
		"float RBlue_Dot_V = max(dot(Reflection_Blue, Viewer), 0.0);" \

		"vec3 ambient_Red = u_La_Red * u_Ka;" \
		"vec3 diffuse_Red = u_Ld_Red * u_Kd * SRed_Dot_N;" \
		"vec3 specular_Red = u_Ls_Red * u_Ks * pow(RRed_Dot_V, u_shininess);" \
		"phongRed_Light = ambient_Red + diffuse_Red + specular_Red;" \

		"vec3 ambient_Blue = u_La_Blue * u_Ka;" \
		"vec3 diffuse_Blue = u_Ld_Blue * u_Kd * SBlue_Dot_N;" \
		"vec3 specular_Blue = u_Ls_Blue * u_Ks * pow(RBlue_Dot_V, u_shininess);" \
		"phongBlue_Light = ambient_Blue + diffuse_Blue + specular_Blue;" \

		"phongLight = phongRed_Light + phongBlue_Light;" \

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
		"FragColor = vec4(phongLight, 0.0);" \
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

	LaRed_Uniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_La_Red");
	LdRed_Uniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ld_Red");
	LsRed_Uniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ls_Red");
	lightPositionRed_Uniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_light_position_Red");

	LaBlue_Uniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_La_Blue");
	LdBlue_Uniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ld_Blue");
	LsBlue_Uniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ls_Blue");
	lightPositionBlue_Uniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_light_position_Blue");

	KaUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ka");
	KdUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Kd");
	KsUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ks");
	shininessUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_shininess");
	LKeyPressUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_L_keypress");








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
		"uniform vec4 u_light_position_Red;" \
		"uniform vec4 u_light_position_Blue;" \
		"out vec3 lightDirectionRed_VS;" \
		"out vec3 lightDirectionBlue_VS;" \
		"out vec3 Normal_VS;" \
		"out vec3 Viewer_VS;" \
		"void main(void)" \
		"{"
		"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" \
		"lightDirectionRed_VS = vec3(u_light_position_Red - eyeCoordinate);" \
		"lightDirectionBlue_VS = vec3(u_light_position_Blue - eyeCoordinate);" \
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
		"in vec3 lightDirectionBlue_VS;" \
		"in vec3 Normal_VS;" \
		"in vec3 Viewer_VS;" \
		"uniform vec3 u_La_Red;" \
		"uniform vec3 u_Ld_Red;" \
		"uniform vec3 u_Ls_Red;" \

		"uniform vec3 u_La_Blue;" \
		"uniform vec3 u_Ld_Blue;" \
		"uniform vec3 u_Ls_Blue;" \

		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \

		"uniform float u_shininess;" \
		"uniform int u_L_keypress;" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
		"vec3 phongRed_Light;" \
		"vec3 phongBlue_Light;" \
		"vec3 phongLight;" \
		"if(u_L_keypress == 1){" \
		"vec3 LightDirection_Red = normalize(lightDirectionRed_VS);" \
		"vec3 LightDirection_Blue = normalize(lightDirectionBlue_VS);" \

		"vec3 Normal = normalize(Normal_VS);" \
		"float LRed_Dot_N = max(dot(LightDirection_Red, Normal), 0.0);" \
		"float LBlue_Dot_N = max(dot(LightDirection_Blue, Normal), 0.0);" \

		"vec3 ReflectionRed = reflect(-LightDirection_Red, Normal);" \
		"vec3 ReflectionBlue = reflect(-LightDirection_Blue, Normal);" \

		"vec3 Viewer = normalize(Viewer_VS);" \

		"float RRed_Dot_V = max(dot(ReflectionRed, Viewer), 0.0);" \
		"float RBlue_Dot_V = max(dot(ReflectionBlue, Viewer), 0.0);" \

		"vec3 ambient_Red = u_La_Red * u_Ka;" \
		"vec3 diffuse_Red = u_Ld_Red * u_Kd * LRed_Dot_N;" \
		"vec3 specular_Red = u_Ls_Red * u_Ks * pow(RRed_Dot_V, u_shininess);" \
		"phongRed_Light = ambient_Red + diffuse_Red + specular_Red;" \

		"vec3 ambient_Blue = u_La_Blue * u_Ka;" \
		"vec3 diffuse_Blue = u_Ld_Blue * u_Kd * LBlue_Dot_N;" \
		"vec3 specular_Blue = u_Ls_Blue * u_Ks * pow(RBlue_Dot_V, u_shininess);" \
		"phongBlue_Light = ambient_Blue + diffuse_Blue + specular_Blue;" \

		"phongLight = phongRed_Light + phongBlue_Light;" \

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

	LaRed_Uniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_La_Red");
	LdRed_Uniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Ld_Red");
	LsRed_Uniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Ls_Red");
	lightPositionRed_Uniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_light_position_Red");

	LaBlue_Uniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_La_Blue");
	LdBlue_Uniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Ld_Blue");
	LsBlue_Uniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Ls_Blue");
	lightPositionBlue_Uniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_light_position_Blue");

	KaUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Ka");
	KdUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Kd");
	KsUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Ks");
	shininessUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_shininess");
	LKeyPressUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_L_keypress");

	
	
	/********** Positions **********/
	GLfloat Pyramid_Vertices_RRJ[] = {
		//Face
		0.0f, 1.0f, 0.0f,
		-1.0, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		//Right
		0.0f, 1.0f, 0.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f,
		//Back
		0.0f, 1.0f, 0.0f,
		1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		//Left
		0.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f
	};

	GLfloat Pyramid_Normals_RRJ[] = {
		//Face
		0.0f, 0.447214f, 0.894427f,
		0.0f, 0.447214f, 0.894427f,
		0.0f, 0.447214f, 0.894427f,


		//Right
		0.894427f, 0.447214f, 0.0f,
		0.894427f, 0.447214f, 0.0f,
		0.894427f, 0.447214f, 0.0f,


		//Back
		0.0f, 0.447214f, -0.894427f,
		0.0f, 0.447214f, -0.894427f,
		0.0f, 0.447214f, -0.894427f,

		//Left
		-0.894427f, 0.447214f, 0.0f,
		-0.894427f, 0.447214f, 0.0f,
		-0.894427f, 0.447214f, 0.0f,
	};



	/********* Vao Pyramid **********/
	glGenVertexArrays(1, &vao_Pyramid_RRJ);
	glBindVertexArray(vao_Pyramid_RRJ);

		/********** Position *********/
		glGenBuffers(1, &vbo_Pyramid_Position_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Pyramid_Position_RRJ);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(Pyramid_Vertices_RRJ),
			Pyramid_Vertices_RRJ,
			GL_STATIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
			3,
			GL_FLOAT,
			GL_FALSE,
			0, NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** Normals **********/
		glGenBuffers(1, &vbo_Pyramid_Normal_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Pyramid_Normal_RRJ);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(Pyramid_Normals_RRJ),
			Pyramid_Normals_RRJ,
			GL_STATIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL,
			3,
			GL_FLOAT,
			GL_FALSE,
			0, NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);
	
	
	//Fill Light Data!
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
	
	lights_RRJ[0].lightPosition_RRJ[0] = -2.0f;
	lights_RRJ[0].lightPosition_RRJ[1] = 0.0f;
	lights_RRJ[0].lightPosition_RRJ[2] = 0.0f;
	lights_RRJ[0].lightPosition_RRJ[3] = 1.0f;
	
	
	//Blue Light
	lights_RRJ[1].lightAmbient_RRJ[0] = 0.0f;
	lights_RRJ[1].lightAmbient_RRJ[1] = 0.0f;
	lights_RRJ[1].lightAmbient_RRJ[2] = 0.0f;
	lights_RRJ[1].lightAmbient_RRJ[3] = 0.0f;
	
	lights_RRJ[1].lightDiffuse_RRJ[0] = 0.0f;
	lights_RRJ[1].lightDiffuse_RRJ[1] = 0.0f;
	lights_RRJ[1].lightDiffuse_RRJ[2] = 1.0f;
	lights_RRJ[1].lightDiffuse_RRJ[3] = 1.0f;
	
	lights_RRJ[1].lightSpecular_RRJ[0] = 0.0f;
	lights_RRJ[1].lightSpecular_RRJ[1] = 0.0f;
	lights_RRJ[1].lightSpecular_RRJ[2] = 1.0f;
	lights_RRJ[1].lightSpecular_RRJ[3] = 1.0f;
	
	lights_RRJ[1].lightPosition_RRJ[0] = 2.0f;
	lights_RRJ[1].lightPosition_RRJ[1] = 0.0f;
	lights_RRJ[1].lightPosition_RRJ[2] = 0.0f;
	lights_RRJ[1].lightPosition_RRJ[3] = 1.0f;
		
}

void uninitialize(void) {

	GLXContext currentContext_RRJ = glXGetCurrentContext();

	if (vbo_Pyramid_Normal_RRJ) {
		glDeleteBuffers(1, &vbo_Pyramid_Normal_RRJ);
		vbo_Pyramid_Normal_RRJ = 0;
	}

	if (vbo_Pyramid_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Pyramid_Position_RRJ);
		vbo_Pyramid_Position_RRJ = 0;
	}

	if (vao_Pyramid_RRJ) {
		glDeleteVertexArrays(1, &vao_Pyramid_RRJ);
		vao_Pyramid_RRJ = 0;
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
	mat4 rotateMatrix_RRJ;
	mat4 modelMatrix_RRJ;
	mat4 viewMatrix_RRJ;
	static GLfloat angle_RRJ = 0.0f;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	translateMatrix_RRJ = mat4::identity();
	rotateMatrix_RRJ = mat4::identity();
	modelMatrix_RRJ = mat4::identity();
	viewMatrix_RRJ = mat4::identity();

	translateMatrix_RRJ = translate(0.0f, 0.0f, -4.0f);
	rotateMatrix_RRJ = rotate(angle_RRJ, 0.0f, 1.0f, 0.0f);
	modelMatrix_RRJ = modelMatrix_RRJ * translateMatrix_RRJ * rotateMatrix_RRJ;


	if (iWhichLight_RRJ == 1){
		glUseProgram(shaderProgramObject_PV_RRJ);
		
		glUniformMatrix4fv(modelMatrixUniform_PV_RRJ, 1, GL_FALSE, modelMatrix_RRJ);
		glUniformMatrix4fv(viewMatrixUniform_PV_RRJ, 1, GL_FALSE, viewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, perspectiveProjectionMatrix_RRJ);
			
			
		if (bLight_RRJ == true) {

			glUniform1i(LKeyPressUniform_PV_RRJ, 1);
			
			glUniform3fv(LaRed_Uniform_PV_RRJ, 1, lights_RRJ[0].lightAmbient_RRJ);
			glUniform3fv(LdRed_Uniform_PV_RRJ, 1, lights_RRJ[0].lightDiffuse_RRJ);
			glUniform3fv(LsRed_Uniform_PV_RRJ, 1, lights_RRJ[0].lightSpecular_RRJ);
			glUniform4fv(lightPositionRed_Uniform_PV_RRJ, 1, lights_RRJ[0].lightPosition_RRJ);

			glUniform3fv(LaBlue_Uniform_PV_RRJ, 1, lights_RRJ[1].lightAmbient_RRJ);
			glUniform3fv(LdBlue_Uniform_PV_RRJ, 1, lights_RRJ[1].lightDiffuse_RRJ);
			glUniform3fv(LsBlue_Uniform_PV_RRJ, 1, lights_RRJ[1].lightSpecular_RRJ);
			glUniform4fv(lightPositionBlue_Uniform_PV_RRJ, 1, lights_RRJ[1].lightPosition_RRJ);

			glUniform3fv(KaUniform_PV_RRJ, 1, materialAmbient_RRJ);
			glUniform3fv(KdUniform_PV_RRJ, 1, materialDiffuse_RRJ);
			glUniform3fv(KsUniform_PV_RRJ, 1, materialSpecular_RRJ);
			glUniform1f(shininessUniform_PV_RRJ, materialShininess_RRJ);
		}
		else
			glUniform1i(LKeyPressUniform_PV_RRJ, 0);			
			
	}else{
		glUseProgram(shaderProgramObject_PF_RRJ);
		
		glUniformMatrix4fv(modelMatrixUniform_PF_RRJ, 1, GL_FALSE, modelMatrix_RRJ);
		glUniformMatrix4fv(viewMatrixUniform_PF_RRJ, 1, GL_FALSE, viewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, perspectiveProjectionMatrix_RRJ);
			
			
		if (bLight_RRJ == true) {

			glUniform1i(LKeyPressUniform_PF_RRJ, 1);
			glUniform3fv(LaRed_Uniform_PF_RRJ, 1, lights_RRJ[0].lightAmbient_RRJ);
			glUniform3fv(LdRed_Uniform_PF_RRJ, 1, lights_RRJ[0].lightDiffuse_RRJ);
			glUniform3fv(LsRed_Uniform_PF_RRJ, 1, lights_RRJ[0].lightSpecular_RRJ);
			glUniform4fv(lightPositionRed_Uniform_PF_RRJ, 1, lights_RRJ[0].lightPosition_RRJ);

			glUniform3fv(LaBlue_Uniform_PF_RRJ, 1, lights_RRJ[1].lightAmbient_RRJ);
			glUniform3fv(LdBlue_Uniform_PF_RRJ, 1, lights_RRJ[1].lightDiffuse_RRJ);
			glUniform3fv(LsBlue_Uniform_PF_RRJ, 1, lights_RRJ[1].lightSpecular_RRJ);
			glUniform4fv(lightPositionBlue_Uniform_PF_RRJ, 1, lights_RRJ[1].lightPosition_RRJ);

			glUniform3fv(KaUniform_PF_RRJ, 1, materialAmbient_RRJ);
			glUniform3fv(KdUniform_PF_RRJ, 1, materialDiffuse_RRJ);
			glUniform3fv(KsUniform_PF_RRJ, 1, materialSpecular_RRJ);
			glUniform1f(shininessUniform_PF_RRJ, materialShininess_RRJ);
		}
		else
			glUniform1i(LKeyPressUniform_PF_RRJ, 0);	
	}
	

	glBindVertexArray(vao_Pyramid_RRJ);
		glDrawArrays(GL_TRIANGLES, 0, 12);
	glBindVertexArray(0);
	angle_RRJ += 0.5f;

	glUseProgram(0);
	glXSwapBuffers(gpDisplay_RRJ, gWindow_RRJ);
	
}




