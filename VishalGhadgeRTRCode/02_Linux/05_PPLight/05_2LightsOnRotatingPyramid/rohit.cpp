// headers
#include <iostream>
#include <stdio.h> // for FILE I/O
#include <stdlib.h> // for exit()
#include <memory.h> // for memset()

// headers for XServer
#include <X11/Xlib.h> // analogos to window.h
#include <X11/Xutil.h> // for visuals
#include <X11/XKBlib.h> // XkbKeycodeToKeysym()
#include <X11/keysym.h> // for 'Keysym'

#include <GL/glew.h> // for GLSL exentensions IMPORTANT : This Line Should Be Before #include <GL/gl.h>
#include <GL/gl.h>
#include <GL/glx.h> // for 'glx' functions

#include "vmath.h"

#define WIN_WIDTH 800
#define WIN_HEIGHT 600
#define SIZE 2

using namespace vmath;

enum
{
	VDG_ATTRIBUTE_POSITION = 0,
	VDG_ATTRIBUTE_COLOR,
	VDG_ATTRIBUTE_NORMAL,
	VDG_ATTRIBUTE_TEXTURE0,
};

// global variable declarations

FILE *gpFile = NULL;

Display *gpDisplay = NULL;
XVisualInfo *gpXVisualInfo = NULL;
Colormap gColormap;
Window gWindow;
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display *, GLXFBConfig, GLXContext, Bool, const int *);
glXCreateContextAttribsARBProc glXCreateContextAttribsARB = NULL;
GLXFBConfig gGLXFBConfig;
GLXContext gGLXContext; // parallel to HGLRC

bool gbFullscreen = false;

GLuint gVertexShaderObject;
GLuint gFragmentShaderObject;
GLuint gShaderProgramObject;

GLuint gVao_pyramid;
GLuint gVbo_pyramid_position;
GLuint gVbo_pyramid_normal;

GLuint model_matrix_uniform, view_matrix_uniform, projection_matrix_uniform;

GLuint gLdUniform, gKdUniform, gLightPositionUniform;

GLuint L_KeyPressed_uniform;

GLuint La_uniform[SIZE];
GLuint Ld_uniform[SIZE];
GLuint Ls_uniform[SIZE];
GLuint light_position_uniform[SIZE];

GLuint Ka_uniform;
GLuint Kd_uniform;
GLuint Ks_uniform;
GLuint material_shininess_uniform;

mat4 gPerspectiveProjectionMatrix;

bool gbLight;

GLfloat light0Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat light0Diffuse[] = { 1.0f, 0.0f, 0.0f, 1.0f }; // Decides Colour of Light : Red Light
GLfloat light0Specular[] = { 1.0f, 0.0f, 0.0f, 1.0f }; // Decides Highlight of Light : Red Highlight
GLfloat light0Position[] = { -200.0f, 100.0f, 100.0f, 1.0f }; // light from Left side

GLfloat light1Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat light1Diffuse[] = { 0.0f, 0.0f, 1.0f, 1.0f }; // Decides Colour of Light : Blue Light
GLfloat light1Specular[] = { 0.0f, 0.0f, 1.0f, 1.0f }; // Decides Highlight of Light : Blue Highlight
GLfloat light1Position[] = { 200.0f, 100.0f, 100.0f, 1.0f }; // light from Right side

GLfloat materialAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat materialDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialShininess = 50.0f;

float gAnglePyramid = 0.0f;

// entry-point function
int main(int argc, char *argv[])
{
	// function prototype
	void CreateWindow(void);
	void ToggleFullscreen(void);
	void initialize(void);
	void resize(int, int);
	void display(void);
	void update(void);
	void uninitialize(void);
	
	// variable declarations
	static bool bIsLKeyPressed = false;

	// code
	// create log file
	gpFile = fopen("Log.txt", "w");
	if (!gpFile) {
		printf("Log File Can Not Be Created. Exitting Now ...\n");
		exit(0);
	} else {
		fprintf(gpFile, "Log File is Successfully Opened.\n");
	}

	// create the window
	CreateWindow();

	// initialize
	initialize();

	// message loop

	// variable declarations
	XEvent event; // parallel to 'MSG' structure
	KeySym keySym;
	int winWidth;
	int winHeight;
	bool bDone = false;

	while (bDone == false) {
		while (XPending(gpDisplay)) {
			XNextEvent(gpDisplay, &event); // parallel to GetMessage()
			switch (event.type) { // paralle to 'iMsg'
				case MapNotify: // parallel to WM_CREATE
					break;
				case KeyPress: // parallel to WM_KEYDOWN
					keySym = XkbKeycodeToKeysym(gpDisplay, event.xkey.keycode, 0, 0);
					switch (keySym) {
						case XK_Escape:
							bDone = true;
							break;
						case XK_F:
						case XK_f:
							if (gbFullscreen == false) {
								ToggleFullscreen();
								gbFullscreen = true;
							} else {
								ToggleFullscreen();
								gbFullscreen = false;
							}
							break;
						case XK_L:
						case XK_l:
							if (bIsLKeyPressed == false)
							{
								gbLight = true;
								bIsLKeyPressed = true;
							}
							else
							{
								gbLight = false;
								bIsLKeyPressed = false;
							}
							break;
						default:
							break;
					}
					break;
				case ButtonPress:
					switch (event.xbutton.button) {
						case 1: // Left Button
							break;
						case 2: // Middle Button
							break;
						case 3: // Right Button
							break;
						default:
							break;
					}
					break;
				case MotionNotify: // parallel to WM_MOUSEMOVE
					break;
				case ConfigureNotify: // parallel to WM_SIZE
					winWidth = event.xconfigure.width;
					winHeight = event.xconfigure.height;
					resize(winWidth, winHeight);
					break;
				case Expose: // parallel to WM_PAINT
					break;
				case DestroyNotify:
					break;
				case 33: // close button, system menu - > close
					bDone = true;
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

void CreateWindow(void)
{
	// function prototype
	void uninitialize(void);

	// variable declarations
	XSetWindowAttributes winAttribs;
	GLXFBConfig *pGLXFBConfigs = NULL;
	GLXFBConfig bestGLXFBConfig;
	XVisualInfo *pTempXViualInfo = NULL;
	int iNumFBConfig = 0;
	int styleMask;
	int i;

	static int frameBufferAttributes[] = {
		GLX_X_RENDERABLE, True,
		GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
		GLX_RENDER_TYPE, GLX_RGBA_BIT,
		GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
		GLX_RED_SIZE, 8,
		GLX_GREEN_SIZE, 8,
		GLX_BLUE_SIZE, 8,
		GLX_ALPHA_SIZE, 8,
		GLX_DEPTH_SIZE, 24,
		GLX_STENCIL_SIZE, 8,
		GLX_DOUBLEBUFFER, True,
		None // array must be terminated by 0
	};

	// code
	gpDisplay = XOpenDisplay(NULL);
	if (!gpDisplay) {
		printf("ERROR : Unable To Obtain X Display.\n");
		fprintf(gpFile, "ERROR : Unable To Obtain X Display.\n");
		uninitialize();
		exit(1);
	}
	
	// get a new framebuffer config that meets our attrib requirements
	pGLXFBConfigs = glXChooseFBConfig(gpDisplay, DefaultScreen(gpDisplay), frameBufferAttributes, &iNumFBConfig);
	if (!pGLXFBConfigs) {
		printf("Failed To Get Valid Framebuffer Config. Exitting Now ...\n");
		fprintf(gpFile, "Failed To Get Valid Framebuffer Config. Exitting Now ...\n");
		uninitialize();
		exit(1);
	}
	printf("%d Matching FB Configs Found.\n", iNumFBConfig);

	// pick that FB config/visual with the most samples per pixel
	int bestFramebufferconfig = -1;
	int worstFramebufferconfig = -1;
	int bestNumberOfSamples = -1;
	int worstNumberOfSamples = 999;
	for (i = 0; i < iNumFBConfig; i++) {
		pTempXViualInfo = glXGetVisualFromFBConfig(gpDisplay, pGLXFBConfigs[i]);
		if (pTempXViualInfo) {
			int sampleBuffer, samples;
			glXGetFBConfigAttrib(gpDisplay, pGLXFBConfigs[i], GLX_SAMPLE_BUFFERS, &sampleBuffer);
			glXGetFBConfigAttrib(gpDisplay, pGLXFBConfigs[i], GLX_SAMPLES, &samples);
			printf("Matching Framebuffer Config=%d : Visual ID=0x%lu : SAMPLE BUFFERS=%d : SAMPLES=%d\n", i, pTempXViualInfo->visualid, sampleBuffer, samples);
			if (bestFramebufferconfig < 0 || sampleBuffer && samples > bestNumberOfSamples) {
				bestFramebufferconfig = i;
				bestNumberOfSamples = samples;
			}
			if (worstFramebufferconfig < 0 || !sampleBuffer || samples < worstNumberOfSamples) {
				worstFramebufferconfig = i;
				worstNumberOfSamples = samples;
			}
		}
		XFree(pTempXViualInfo);
	}
	bestGLXFBConfig = pGLXFBConfigs[bestFramebufferconfig];
	// set global GLXFBConfig
	gGLXFBConfig = bestGLXFBConfig;

	// be sure to free FBConfig list allocated by glXChooseFBConfig()
	XFree(pGLXFBConfigs);

	gpXVisualInfo = glXGetVisualFromFBConfig(gpDisplay, bestGLXFBConfig);
	printf("Chosen Visual ID=0x%lu\n", gpXVisualInfo->visualid);

	// setting window's attributes
	winAttribs.border_pixel = 0;
	winAttribs.background_pixmap = 0;
	winAttribs.colormap = XCreateColormap(gpDisplay, RootWindow(gpDisplay, gpXVisualInfo->screen), // you can give defaultScreen as well
		gpXVisualInfo->visual, AllocNone); // for 'movable' memory allocation
	winAttribs.event_mask = StructureNotifyMask | KeyPressMask  | ButtonPressMask | ExposureMask | VisibilityChangeMask | PointerMotionMask;

	styleMask = CWBorderPixel | CWEventMask | CWColormap;

	gColormap = winAttribs.colormap;
	gWindow = XCreateWindow(gpDisplay,
		RootWindow(gpDisplay, gpXVisualInfo->screen),
		0,
		0,
		WIN_WIDTH,
		WIN_HEIGHT,
		0, // border width
		gpXVisualInfo->depth, // depth of visual (depth for Colormap)
		InputOutput, // class(type) of your window
		gpXVisualInfo->visual,
		styleMask,
		&winAttribs);
	if (!gWindow) {
		printf("Failure In Window Creation.\n");
		fprintf(gpFile, "Failure In Window Creation.\n");
		uninitialize();
		exit(1);
	}

	XStoreName(gpDisplay, gWindow, "OpenGL PP - Two Lights On Rotating Pyramid");

	Atom windowManagerDelete = XInternAtom(gpDisplay, "WM_WINDOW_DELETE", True);
	XSetWMProtocols(gpDisplay, gWindow, &windowManagerDelete, 1);

	XMapWindow(gpDisplay, gWindow);
}

void ToggleFullscreen(void)
{
	// variable declarations
	Atom wm_state;
	Atom fullscreen;
	XEvent xev = {0};

	// Code
	wm_state = XInternAtom(gpDisplay, "_NET_WM_STATE", False);

	memset(&xev, 0, sizeof(xev));
	xev.type = ClientMessage;
	xev.xclient.window = gWindow;
	xev.xclient.message_type = wm_state;
	xev.xclient.format = 32;
	xev.xclient.data.l[0] = gbFullscreen ? 0 : 1;

	fullscreen = XInternAtom(gpDisplay, "_NET_WM_STATE_FULLSCREEN", False);
	xev.xclient.data.l[1] = fullscreen;

	XSendEvent(gpDisplay,
		RootWindow(gpDisplay, gpXVisualInfo->screen),
		False,
		StructureNotifyMask,
		&xev);
}

void initialize(void)
{
	// function prototype
	void uninitialize(void);
	void resize(int, int);

	// variable declaration
	GLint num;

	// code
	// create new GL context 4.5 for rendering
	glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)glXGetProcAddressARB((GLubyte *)"glXCreateContextAttribsARB");
	if (!glXCreateContextAttribsARB) {
		fprintf(gpFile, "Failed to get glXCreateContextAttribsARB address\n");
	} else {
		printf("Got glXCreateContextAttribsARB address\n");
	}

	GLint attribs[] = {
		GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
		GLX_CONTEXT_MINOR_VERSION_ARB, 1,
		//GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB,
		GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
		0 // array must be terminated by 0
	};

	gGLXContext = glXCreateContextAttribsARB(gpDisplay, gGLXFBConfig, 0, True, attribs);

	if (!gGLXContext) {// fallback to safe old style 2.x context
		// When a context version below 3.0 is requested, implementation will return
		// the newest context version compatible with OpenGL ersiond less than version 3.0
		GLint attribs[] = {
			GLX_CONTEXT_MAJOR_VERSION_ARB, 1,
			GLX_CONTEXT_MINOR_VERSION_ARB, 0,
			0 // array must be terminated by 0
		};
		printf("Failed To Create GLX 4.5 context. Hence Using Old-Style GLX Context\n");
		fprintf(gpFile, "Failed To Create GLX 4.5 context. Hence Using Old-Style GLX Context\n");

		gGLXContext = glXCreateContextAttribsARB(gpDisplay, gGLXFBConfig, 0, True, attribs);
		/* where GL_TRUE is for H/W Rendering
		 * GL_FALSE is for S/W rendering
		 * 3rd param : Sharable context for multimonitor NULL means non sharable
		 */
	} else {// successfully create 4.1 context
		printf("OpenGL Context 3.1 Is Created.\n");
	}

	// verifying that context is a direct context
	if (!glXIsDirect(gpDisplay, gGLXContext)) {
		printf("Indirect GLX Rendering Context Obtained\n");
	} else {
		printf("Direct GLX Rendering Context Obtained\n");
	}

	 glXMakeCurrent(gpDisplay, gWindow, gGLXContext); // same as wglMakeCurrent

	 // GLEW Initialization Code For GLSL (IMPORTANT : It Must Be Here.
	 // Means After Creating Context But Before Using Any OpenGL Function)
	 GLenum glew_error = glewInit();
	 if (glew_error != GLEW_OK) {
		 GLXContext currentGLXContext;
		 currentGLXContext = glXGetCurrentContext();

		 // This code is for sharable context
		 if (currentGLXContext != NULL && currentGLXContext == gGLXContext) {
			 glXMakeCurrent(gpDisplay, 0, 0);
		 }

		 if (gGLXContext) {
			 glXDestroyContext(gpDisplay, gGLXContext); // same as wglDeleteContext
		 }
	 }

	 /* get OpenGL version */
	 fprintf(gpFile, "------------------------------------------------------------------\n");
	 fprintf(gpFile, "OpenGL version : %s\n", glGetString(GL_VERSION));
	 fprintf(gpFile, "------------------------------------------------------------------\n");
	 /* get GLSL version */
	 fprintf(gpFile, "GLSL version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	 fprintf(gpFile, "------------------------------------------------------------------\n");

	 // *** VERTEX SHADER ***
	 // create shader
	 gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	 // provide source code to shader
	 const GLchar *vertexShaderSourceCode = 
	 	"#version 330"\
		"\n"\
		"in vec4 vPosition;"\
		"in vec3 vNormal;"\
		"uniform mat4 u_model_matrix;"\
		"uniform mat4 u_view_matrix;"\
		"uniform mat4 u_projection_matrix;"\
		"uniform int u_lighting_enabled;"\
		"uniform vec3 u_La[2];"\
		"uniform vec3 u_Ld[2];"\
		"uniform vec3 u_Ls[2];"\
		"uniform vec4 u_light_position[2];"\
		"uniform vec3 u_Ka;"\
		"uniform vec3 u_Kd;"\
		"uniform vec3 u_Ks;"\
		"uniform float u_material_shininess;"\
		"out vec3 phong_ads_color;"\
	 	"void main(void)"\
		"{"\
		"if (u_lighting_enabled == 1)"\
		"{"\
		"vec4 eye_coordinates = u_view_matrix * u_model_matrix * vPosition;"\
		"vec3 transformed_normals = normalize(mat3(u_view_matrix * u_model_matrix) * vNormal);"\
		"vec3 viewer_vector = normalize(-eye_coordinates.xyz);"\
		"phong_ads_color = vec3(0.0, 0.0, 0.0);"\
		"for (int i = 0; i < 2; i++)"\
		"{"\
		"vec3 light_direction = normalize(vec3(u_light_position[i]) - eye_coordinates.xyz);"\
		"float tn_dot_ld = max(dot(transformed_normals, light_direction), 0.0);"\
		"vec3 ambient = u_La[i] * u_Ka;"\
		"vec3 diffuse = u_Ld[i] * u_Kd * tn_dot_ld;"\
		"vec3 reflection_vector = reflect(-light_direction, transformed_normals);"\
		"vec3 specular = u_Ls[i] * u_Ks * pow(max(dot(reflection_vector, viewer_vector), 0.0), u_material_shininess);"\
		"phong_ads_color += ambient + diffuse + specular;"\
		"}"\
		"}"\
		"else"\
		"{"\
		"phong_ads_color = vec3(1.0, 1.0, 1.0);"\
		"}"\
		"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;"\
		"}";

	 glShaderSource(gVertexShaderObject, 1, (const GLchar **)&vertexShaderSourceCode, NULL);

	 // compile shader
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
				 uninitialize();
				 exit(0);
			 }
		 }
	 }
	 
	 // *** FRAGMENT SHADER ***
	 // create shader
	 gFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	 // provide source code to shader
	 const GLchar *fragmentShaderSourceCode = 
	 	"#version 330"\
		"\n"\
		"in vec3 phong_ads_color;"\
		"out vec4 FragColor;"\
		"void main(void)"\
		"{"\
		"FragColor = vec4(phong_ads_color, 1.0);"\
		"}";
		
	 glShaderSource(gFragmentShaderObject, 1, (const GLchar **)&fragmentShaderSourceCode, NULL);

	 // compile shader
	 glCompileShader(gFragmentShaderObject);

	 // *** SHADER PROGRAM ***
	 // create
	 gShaderProgramObject = glCreateProgram();
	 iInfoLogLength = 0;
	 iShaderCompiledStatus = 0;
	 szInfoLog = NULL;
	 glGetShaderiv(gFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	 if (iShaderCompiledStatus == GL_FALSE)
	 {
		 glGetShaderiv(gFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		 if (iInfoLogLength > 0)
		 {
			 szInfoLog = (char *)malloc(iInfoLogLength);
			 if (szInfoLog != NULL)
			 {
				 GLsizei written;
				 glGetShaderInfoLog(gFragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				 fprintf(gpFile, "Fragment Shader Compilation Log : %s\n", szInfoLog);
				 free(szInfoLog);
				 uninitialize();
				 exit(0);
			 }
		 }
	 }

	 // attach vertex shader to shader program
	 glAttachShader(gShaderProgramObject, gVertexShaderObject);
	 
	 // attach fragment shader to shader program
	 glAttachShader(gShaderProgramObject, gFragmentShaderObject);

	 // pre-link binding of shader program objeect with vertex shader position attribute
	 glBindAttribLocation(gShaderProgramObject, VDG_ATTRIBUTE_POSITION, "vPosition");
	 
	 // pre-link binding of shader program objeect with vertex shader normal attribute
	 glBindAttribLocation(gShaderProgramObject, VDG_ATTRIBUTE_NORMAL, "vNormal");

	 // link shader
	 glLinkProgram(gShaderProgramObject);
	 GLint iShaderProgramLinkStatus = 0;
	 iInfoLogLength = 0;
	 szInfoLog = NULL;
	 glGetProgramiv(gShaderProgramObject, GL_LINK_STATUS, &iShaderProgramLinkStatus);
	 if (iShaderProgramLinkStatus == GL_FALSE)
	 {
		 glGetProgramiv(gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		 if (iInfoLogLength > 0)
		 {
			 GLsizei written;
			 glGetProgramInfoLog(gShaderProgramObject, iInfoLogLength, &written, szInfoLog);
			 fprintf(gpFile, "Shader Program Link Log : %s\n", szInfoLog);
			 free(szInfoLog);
			 uninitialize();
			 exit(0);
		 }
	 }
	
	// get uniform locations
	model_matrix_uniform = glGetUniformLocation(gShaderProgramObject, "u_model_matrix");
	view_matrix_uniform = glGetUniformLocation(gShaderProgramObject, "u_view_matrix");
	projection_matrix_uniform = glGetUniformLocation(gShaderProgramObject, "u_projection_matrix");

	// L/l key is pressed or not
	L_KeyPressed_uniform = glGetUniformLocation(gShaderProgramObject, "u_lighting_enabled");

	// Light0
	// ambient color intensity of light0
	La_uniform[0] = glGetUniformLocation(gShaderProgramObject, "u_La[0]");
	// diffuse color intensity of light0
	Ld_uniform[0] = glGetUniformLocation(gShaderProgramObject, "u_Ld[0]");
	// specular color intensity of light0
	Ls_uniform[0] = glGetUniformLocation(gShaderProgramObject, "u_Ls[0]");
	// position of light0
	light_position_uniform[0] = glGetUniformLocation(gShaderProgramObject, "u_light_position[0]");

	// Light1
	// ambient color intensity of light1
	La_uniform[1] = glGetUniformLocation(gShaderProgramObject, "u_La[1]");
	// diffuse color intensity of light1
	Ld_uniform[1] = glGetUniformLocation(gShaderProgramObject, "u_Ld[1]");
	// specular color intensity of light1
	Ls_uniform[1] = glGetUniformLocation(gShaderProgramObject, "u_Ls[1]");
	// position of light1
	light_position_uniform[1] = glGetUniformLocation(gShaderProgramObject, "u_light1_position[1]");

	// ambient reflective color intensity of material
	Ka_uniform = glGetUniformLocation(gShaderProgramObject, "u_Ka");
	// diffuse reflective color intensity of material
	Kd_uniform = glGetUniformLocation(gShaderProgramObject, "u_Kd");
	// specular reflective color intensity of material
	Ks_uniform = glGetUniformLocation(gShaderProgramObject, "u_Ks");
	// shininess of material (value is conventionally between 1 to 200)
	material_shininess_uniform = glGetUniformLocation(gShaderProgramObject, "u_material_shininess");

	// *** vertices, colors, shader attribs, vbo, vao initializations ***
	const GLfloat pyramidVertices[] =
	{
		// FRONT FACE
		0.0f, 1.0f, 0.0f, // apex
		-1.0f, -1.0f, 1.0f, // left-bottom of front face
		1.0f, -1.0f, 1.0f, // right-bottom of front face

		// RIGHT FACE
		0.0f, 1.0f, 0.0f, // apex
		1.0f, -1.0f, 1.0f, // left-bottom of right face
		1.0f, -1.0f, -1.0f, // right-bottom of right face

		// BACK FACE
		0.0f, 1.0f, 0.0f, // apex
		1.0f, -1.0f, -1.0f, // left-bottom of back face
		-1.0f, -1.0f, -1.0f, // right-bottom of back face

		// LEFT FACE
		0.0f, 1.0f, 0.0f, // apex
		-1.0f, -1.0f, -1.0f, // left-bottom of left face
		-1.0f, -1.0f, 1.0f, // right-bottom of left face
	};

	const GLfloat pyramidNormals[] =
	{
		// FRONT FACE
		0.0f, 0.447214f, 0.894427f,
		0.0f, 0.447214f, 0.894427f,
		0.0f, 0.447214f, 0.894427f,

		// RIGHT FACE
		0.894427f, 0.447214f, 0.0f,
		0.894427f, 0.447214f, 0.0f,
		0.894427f, 0.447214f, 0.0f,

		// BACK FACE
		0.0f, 0.447214f, -0.894427f,
		0.0f, 0.447214f, -0.894427f,
		0.0f, 0.447214f, -0.894427f,

		// LEFT FACE
		-0.894427f, 0.447214f, 0.0f,
		-0.894427f, 0.447214f, 0.0f,
		-0.894427f, 0.447214f, 0.0f,
	};

	// PYRAMID
	// vao
	glGenVertexArrays(1, &gVao_pyramid); // recorder
	glBindVertexArray(gVao_pyramid);

	// Vbo for position
	glGenBuffers(1, &gVbo_pyramid_position);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_pyramid_position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(pyramidVertices), pyramidVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(VDG_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(VDG_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind with Vbo_position

	// Vbo for normal
	glGenBuffers(1, &gVbo_pyramid_normal);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_pyramid_normal);
	glBufferData(GL_ARRAY_BUFFER, sizeof(pyramidNormals), pyramidNormals, GL_STATIC_DRAW);
	glVertexAttribPointer(VDG_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(VDG_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind with Vbo_normal

	glBindVertexArray(0); // Unbind with Vao

	 // code
	 glShadeModel(GL_SMOOTH);
	 // set-up depth buffer
	 glClearDepth(1.0f);
	 // enable depth testing
	 glEnable(GL_DEPTH_TEST);
	 // depth test to do
	 glDepthFunc(GL_LEQUAL);
	 // set really nice perspective calculations?
	 glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	 // We will always cull back faces for better performance
	 //glEnable(GL_CULL_FACE);
	 
	 // set background color to which it will display even if it will empty.
	 glClearColor(0.0f, 0.0f, 0.0f, 0.0f); // black

	 // set perspectiveMatrix to identity matrix
	 gPerspectiveProjectionMatrix = mat4::identity();

	 gbLight = false;

	 resize(WIN_WIDTH, WIN_HEIGHT); // warm up call
}

void display(void)
{
	// code
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// start using OpenGL program object
	glUseProgram(gShaderProgramObject);
	
	// start using OpenGL program object
	glUseProgram(gShaderProgramObject);

	if (gbLight == true)
	{
		// set 'u_lighting_enabled' uniform
		glUniform1i(L_KeyPressed_uniform, 1);

		// setting light0's properties
		glUniform3fv(La_uniform[0], 1, light0Ambient);
		glUniform3fv(Ld_uniform[0], 1, light0Diffuse);
		glUniform3fv(Ls_uniform[0], 1, light0Specular);
		glUniform4fv(light_position_uniform[0], 1, light0Position);

		// setting light1's properties
		glUniform3fv(La_uniform[1], 1, light1Ambient);
		glUniform3fv(Ld_uniform[1], 1, light1Diffuse);
		glUniform3fv(Ls_uniform[1], 1, light1Specular);
		glUniform4fv(light_position_uniform[1], 1, light1Position);

		// setting material's properties
		glUniform3fv(Ka_uniform, 1, materialAmbient);
		glUniform3fv(Kd_uniform, 1, materialDiffuse);
		glUniform3fv(Ks_uniform, 1, materialSpecular);
		glUniform1f(material_shininess_uniform, materialShininess);
	}
	else
	{
		// reset 'u_lighting_enabled' uniform
		glUniform1i(L_KeyPressed_uniform, 0);

	}

	// OpenGL Drawing
	// set all matrices to identity
	mat4 modelMatrix = mat4::identity();
	mat4 viewMatrix = mat4::identity();
	mat4 rotationMatrix = mat4::identity();

	modelMatrix = translate(0.0f, 0.0f, -5.0f);

	rotationMatrix = rotate(gAnglePyramid, 0.0f, 1.0f, 0.0f); // Y axis rotation

	modelMatrix = modelMatrix * rotationMatrix;

	glUniformMatrix4fv(model_matrix_uniform, 1, GL_FALSE, modelMatrix); // 1 for how many matrices
	glUniformMatrix4fv(view_matrix_uniform, 1, GL_FALSE, viewMatrix); // 1 for how many matrices
	glUniformMatrix4fv(projection_matrix_uniform, 1, GL_FALSE, gPerspectiveProjectionMatrix); // 1 for how many matrices
	
	// *** bind vao ***
	glBindVertexArray(gVao_pyramid);

	// *** draw, either by glDrawTriangles() or glDrawArrays() or glDrawElements() ***
	glDrawArrays(GL_TRIANGLES, 0, 12); // 12 (each with its x,y,z) vertices in pyramidVertices array

	// *** unbind vao ***
	glBindVertexArray(0);
	// stop using OpenGL program object
	glUseProgram(0);

	glXSwapBuffers(gpDisplay, gWindow);
}

void resize(int width, int height)
{
	// code
	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	gPerspectiveProjectionMatrix = perspective(45.0f, (GLfloat)width/(GLfloat)height, 0.1f, 100.0f);
}

void update(void)
{
	gAnglePyramid = gAnglePyramid + 0.3f;
	if (gAnglePyramid > 360.0f)
		gAnglePyramid = gAnglePyramid - 360.0f;
}

void uninitialize(void)
{
	// destroy vbo_position
	if (gVbo_pyramid_position)
	{
		glDeleteBuffers(1, &gVbo_pyramid_position);
		gVbo_pyramid_position = 0;
	}

	// destroy vbo_normal
	if (gVbo_pyramid_normal)
	{
		glDeleteBuffers(1, &gVbo_pyramid_normal);
		gVbo_pyramid_normal = 0;
	}

	// destroy vao
	if (gVao_pyramid)
	{
		glDeleteVertexArrays(1, &gVao_pyramid);
		gVao_pyramid = 0;
	}

	// detach vertex shader from shader program object
	glDetachShader(gShaderProgramObject, gVertexShaderObject);
	// detach fragment shader from shader program object
	glDetachShader(gShaderProgramObject, gFragmentShaderObject);

	// delete vertex shader object
	glDeleteShader(gVertexShaderObject);
	gVertexShaderObject = 0;
	
	// delete fragment shader object
	glDeleteShader(gFragmentShaderObject);
	gFragmentShaderObject = 0;

	// delete shader program object
	glDeleteProgram(gShaderProgramObject);
	gShaderProgramObject = 0;

	// unlink shader program
	glUseProgram(0);

	GLXContext currentGLXContext;
	currentGLXContext = glXGetCurrentContext();

	// This code is for sharable context
	if (currentGLXContext != NULL && currentGLXContext == gGLXContext) {
		glXMakeCurrent(gpDisplay, 0, 0);
	}

	if (gGLXContext) {
		glXDestroyContext(gpDisplay, gGLXContext); // same as wglDeleteContext
	}
		
	if (gWindow) {
		XDestroyWindow(gpDisplay, gWindow);
	}

	if (gColormap) {
		XFreeColormap(gpDisplay, gColormap);
	}

	if (gpXVisualInfo) {
		free(gpXVisualInfo);
		gpXVisualInfo = NULL;
	}

	if (gpDisplay) {
		XCloseDisplay(gpDisplay);
		gpDisplay = NULL;
	}

	if (gpFile) {
		fprintf(gpFile, "Log File Is Successfully Closed.\n");
		fclose(gpFile);
		gpFile = NULL;
	}
}

