#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<memory.h>
#define _USE_MATH_DEFINES 1
#include<math.h>


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

enum {
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOORD0
};


#define WIN_WIDTH 800
#define WIN_HEIGHT 600

//For FullScreen
bool bIsFullScreen = false;

//For OpenGL
Display *gpDisplay = NULL;
XVisualInfo *gpXVisualInfo = NULL;
Colormap gColormap;
Window gWindow;

//For OpenGL
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
glXCreateContextAttribsARBProc glXCreateContextAttribsARB = NULL;
GLXFBConfig gGLXFBConfig;
GLXContext gGLXContext;

//For Uniform
GLuint mvpUniform;

//For Perspective
mat4 perspectiveProjectionMatrix;

//For I
GLuint vao_I;
GLuint vbo_I_Position;
GLuint vbo_I_Color;

//For N
GLuint vao_N;
GLuint vbo_N_Position;
GLuint vbo_N_Color;

//For D
GLuint vao_D;
GLuint vbo_D_Position;
GLuint vbo_D_Color;

//For A
GLuint vao_A;
GLuint vbo_A_Position;
GLuint vbo_A_Color;

//For Flag
GLuint vao_Flag;
GLuint vbo_Flag_Position;
GLuint vbo_Flag_Color;


//For Program
GLuint gShaderProgramObject;

//For Error
FILE *gbFile = NULL;


int main(void){
	
	void CreateWindow(void);
	void initialize(void);
	void resize(int, int);
	void ToggleFullScreen(void);
	void display(void);
	void uninitialize(void);

	int winWidth = WIN_WIDTH;
	int winHeight = WIN_HEIGHT;
	bool bDone = false;
	XEvent event;
	KeySym keysym;


	gbFile = fopen("Log.txt", "w");
	if(gbFile == NULL){
		printf("Log Creation Failed!!\n");
		uninitialize();
		exit(1);
	}	
	else
		fprintf(gbFile, "Log Created!!\n");


	CreateWindow();

	ToggleFullScreen();

	initialize();

	while(bDone == false){
		while(XPending(gpDisplay)){
		
			XNextEvent(gpDisplay, &event);
			switch(event.type){
				case MapNotify:
					break;
				case Expose:
					break;
				case MotionNotify:
					break;
				case DestroyNotify:
					break;
				case ConfigureNotify:
					winWidth = event.xconfigure.width;
					winHeight = event.xconfigure.height;
					resize(winWidth, winHeight);
					break;
					
				case KeyPress:
					keysym = XkbKeycodeToKeysym(gpDisplay, event.xkey.keycode, 0, 0);
					switch(keysym){
						case XK_Escape:
							bDone = true;
							break;

						case XK_F:
						case XK_f:
							if(bIsFullScreen == false){
								ToggleFullScreen();
								bIsFullScreen = true;
							}
							else{
								ToggleFullScreen();
								bIsFullScreen = false;
							}
							break;

						default:
							break;
					}
					break;

				case ButtonPress:
					switch(event.xbutton.button){
						case 1:
							printf("Left Button!!\n");
							break;
						case 2:
							printf("Middle Button!!\n");
							break;
						case 3:
							printf("Right Button!!\n");
							break;
						default:
							break;
					}
					break;

				case 33:
					bDone = true;
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

void CreateWindow(void){
		
	void uninitialize(void);

	int defaultScreen;
	XSetWindowAttributes winAttribs;
	int styleMask;
	
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
		GLX_DOUBLEBUFFER, True,
		None
	};

	GLXFBConfig *pGLXFBConfig = NULL;
	int iNumberOfFBConfig;
	XVisualInfo *pTempXVisualInfo = NULL;
	GLXFBConfig bestFBConfig;



	gpDisplay = XOpenDisplay(NULL);
	if(gpDisplay == NULL){
		printf("XOpenDisplay() Failed!!\n");
		uninitialize();
		exit(1);
	}

	defaultScreen = XDefaultScreen(gpDisplay);


	pGLXFBConfig = glXChooseFBConfig(gpDisplay, defaultScreen, frameBufferAttributes, &iNumberOfFBConfig);
	fprintf(gbFile, "Match: %d\n", iNumberOfFBConfig);

	int bestFrameBufferConfig = -1;
	int bestSamples = -1;
       	int worstFrameBufferConfig = -1;
	int worstSamples = -1;

	for(int i = 0; i < iNumberOfFBConfig; i++){
		pTempXVisualInfo = glXGetVisualFromFBConfig(gpDisplay, pGLXFBConfig[i]);
		if(pTempXVisualInfo){
			
			GLint samples, sampleBuffers;

			glXGetFBConfigAttrib(gpDisplay, pGLXFBConfig[i], GLX_SAMPLES, &samples);
			glXGetFBConfigAttrib(gpDisplay, pGLXFBConfig[i], GLX_SAMPLE_BUFFERS, &sampleBuffers);

			if(bestFrameBufferConfig < 0 || sampleBuffers && samples > bestSamples){
				bestFrameBufferConfig = i;
				bestSamples = samples;
			}

			if(worstFrameBufferConfig < 0 || sampleBuffers && samples < worstSamples){
				worstFrameBufferConfig = i;
				worstSamples = samples;
			}
		}
		XFree(pTempXVisualInfo);
		pTempXVisualInfo = NULL;
	}

	bestFBConfig = pGLXFBConfig[bestFrameBufferConfig];
	gGLXFBConfig = bestFBConfig;
	XFree(pGLXFBConfig);
	pGLXFBConfig = NULL;

	gpXVisualInfo = glXGetVisualFromFBConfig(gpDisplay, bestFBConfig);
	if(gpXVisualInfo == NULL){
		fprintf(gbFile, "glXGetVisualFromFBConfig() Failed!!\n");
		uninitialize();
		exit(1);
	}	



	winAttribs.border_pixel = 0;
	winAttribs.border_pixmap = 0;
	winAttribs.background_pixel = BlackPixel(gpDisplay, defaultScreen);
	winAttribs.background_pixmap = 0;
	winAttribs.colormap = XCreateColormap(gpDisplay, 
					RootWindow(gpDisplay, gpXVisualInfo->screen),
					gpXVisualInfo->visual,
					AllocNone);
	gColormap = winAttribs.colormap;
	winAttribs.event_mask = ExposureMask | VisibilityChangeMask | PointerMotionMask | 
				KeyPressMask | ButtonPressMask | StructureNotifyMask;

	styleMask = CWBorderPixel | CWBackPixel | CWEventMask | CWColormap;

	gWindow = XCreateWindow(gpDisplay,
			RootWindow(gpDisplay, gpXVisualInfo->screen),
			0, 0,
			WIN_WIDTH, WIN_HEIGHT,
			0,
			gpXVisualInfo->depth,
			InputOutput,
			gpXVisualInfo->visual,
			styleMask,
			&winAttribs);

	if(!gWindow){
		printf("XCreateWindow() Failed!!\n");
		uninitialize();
		exit(1);
	}

	XStoreName(gpDisplay, gWindow, "14-StaticIndia");

	Atom windowManagerDelete = XInternAtom(gpDisplay, "WM_DELETE_WINDOW", True);
	XSetWMProtocols(gpDisplay, gWindow, &windowManagerDelete, 1);

	XMapWindow(gpDisplay, gWindow);
}

void ToggleFullScreen(void){

	Atom wm_state;
	Atom fullscreen;
	XEvent xev = {0};

	wm_state = XInternAtom(gpDisplay, "_NET_WM_STATE", False);

	memset(&xev, 0, sizeof(XEvent));

	xev.type = ClientMessage;
	xev.xclient.window = gWindow;
	xev.xclient.message_type = wm_state;
	xev.xclient.format = 32;
	xev.xclient.data.l[0] = bIsFullScreen ? 0 : 1;

	fullscreen = XInternAtom(gpDisplay, "_NET_WM_STATE_FULLSCREEN", False);
	xev.xclient.data.l[1] = fullscreen;

	XSendEvent(gpDisplay,
		RootWindow(gpDisplay, gpXVisualInfo->screen),
		False,
		StructureNotifyMask,
		&xev);
}

void initialize(void){
	
	void resize(int, int);
	void uninitialize(void);
	void Calculation(GLfloat[]);
	void FillCircle_Position(GLfloat[], GLfloat[]);
	
	GLenum Result;
	GLuint vertexShaderObject;
	GLuint fragmentShaderObject;

	
	glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)glXGetProcAddressARB((GLubyte*)"glXCreateContextAttribsARB");
	if(glXCreateContextAttribsARB == NULL){
		fprintf(gbFile, "glXGetProcAddressARB() Failed!!\n");
		uninitialize();
		exit(1);
	}

	const int Attributes[] = {
		GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
		GLX_CONTEXT_MINOR_VERSION_ARB, 5,
		GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
		None
	};

	gGLXContext = glXCreateContextAttribsARB(gpDisplay, gGLXFBConfig, NULL, True, Attributes);
	if(gGLXContext == NULL){
		fprintf(gbFile, "glXCreateContextAttribsARB() Failed For 4.5 Version!!\n");
		fprintf(gbFile, "Getting Best From System!!\n");

		const int Attribs[] = {
			GLX_CONTEXT_MAJOR_VERSION_ARB, 1,
			GLX_CONTEXT_MINOR_VERSION_ARB, 0,
			None
		};

		gGLXContext = glXCreateContextAttribsARB(gpDisplay, gGLXFBConfig, NULL, True, Attribs);
	}

	if(!glXIsDirect(gpDisplay, gGLXContext))
		fprintf(gbFile, "S/W Context!!\n");
	else
		fprintf(gbFile, "H/W Context!!\n");


	glXMakeCurrent(gpDisplay, gWindow, gGLXContext);


	Result = glewInit();
	if(Result != GLEW_OK){
		printf("glewInit() Failed!!\n");
		uninitialize();
		exit(1);
	}
	
	/********** Vertex Shader **********/
	vertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const char *vertexShaderSourceCode = 
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec4 vColor;" \
		"out vec4 outColor;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void)" \
		"{" \
			"gl_Position = u_mvp_matrix * vPosition;" \
			"outColor = vColor;" \
		"}";

	glShaderSource(vertexShaderObject, 1,
		(const char**)&vertexShaderSourceCode, NULL);

	glCompileShader(vertexShaderObject);

	int iShaderCompileStatus;
	int iInfoLogLength;
	char *szInfoLog = NULL;

	glGetShaderiv(vertexShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if(iShaderCompileStatus == GL_FALSE){
		glGetShaderiv(vertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if(iInfoLogLength > 0){
			szInfoLog = (char*)malloc(sizeof(char) * iInfoLogLength);
			if(szInfoLog){
				GLsizei written;
				glGetShaderInfoLog(vertexShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gbFile, "Vertex Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				exit(1);
			}
		}

	}


	/********** Fragment Shader **********/
	fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const char *fragmentShaderSourceCode = 
		"#version 450 core" \
		"\n" \ 
		"in vec4 outColor;" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
			"FragColor = outColor;" \
		"}";

	glShaderSource(fragmentShaderObject, 1,
		(const char**)&fragmentShaderSourceCode, NULL);

	glCompileShader(fragmentShaderObject);

	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(fragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if(iShaderCompileStatus == GL_FALSE){
		glGetShaderiv(fragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if(iInfoLogLength > 0){
			szInfoLog = (char*)malloc(sizeof(char) * iInfoLogLength);
			if(szInfoLog){
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gbFile, "Fragment Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				exit(1);
			}
		}
	}


	/********** Program Object **********/
	gShaderProgramObject = glCreateProgram();

	glAttachShader(gShaderProgramObject, vertexShaderObject);
	glAttachShader(gShaderProgramObject, fragmentShaderObject);

	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_COLOR, "vColor");

	glLinkProgram(gShaderProgramObject);

	int iProgramLinkStatus;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetProgramiv(gShaderProgramObject, GL_LINK_STATUS, &iProgramLinkStatus);
	if(iProgramLinkStatus == GL_FALSE){
		glGetProgramiv(gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if(iInfoLogLength > 0){
			szInfoLog = (char*)malloc(sizeof(char) * iInfoLogLength);
			if(szInfoLog){
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gbFile, "Shader Program Linking Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				exit(1);
			}
		}
	}

	mvpUniform = glGetUniformLocation(gShaderProgramObject, "u_mvp_matrix");

	/********** Position And Color **********/
	GLfloat I_Position[] = {
		-0.3f, 1.0f, 0.0f,
		0.3f, 1.0f, 0.0f,
		
		0.0f, 1.0f, 0.0f,
		0.0f, -1.0f, 0.0f,
		
		-0.3f, -1.0f, 0.0f,
		0.3f, -1.0f, 0.0f,
		None
	};
	
	
	GLfloat I_Color[] = {
		1.0f, 0.6f, 0.2f,	
		1.0f, 0.6f, 0.2f,
		
		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f,
		
		0.0705f, 0.533f, 0.0274f,
		0.0705f, 0.533f, 0.0274f,
		None
	};
	
	GLfloat N_Position[] = {
		0.0f, 1.06f, 0.0f,
		0.0f, -1.06f, 0.0f,
		
		0.75f, 1.06f, 0.0f,
		0.75f, -1.06f, 0.0f,
		
		0.0f, 1.06f, 0.0f,
		0.75f, -1.06f, 0.0f,
		None
	};
	
	
	GLfloat N_Color[] = {
		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f,
	
		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f,
		
		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f,
		None
	};
		
	GLfloat D_Position[] = {
		0.0f, 1.0f, 0.0f,
		0.0f, -1.0f, 0.0f,
		
		-0.1f, 1.0f, 0.0f,
		0.6f, 1.0f, 0.0f,
		
		-0.1f, -1.0f, 0.0f,
		0.6f, -1.0f, 0.0f,
		
		0.6f, 1.0f, 0.0f,
		0.6f, -1.0f, 0.0f,
		None
	};
	
	GLfloat D_Color[] = {
		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f,
	
		1.0f, 0.6f, 0.2f,
		1.0f, 0.6f, 0.2f,
		
		0.0705f, 0.533f, 0.0274f,
		0.0705f, 0.533f, 0.0274f,
		
		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f,
		None
	};
	
	
	GLfloat A_Position[] = {
		0.0f, 1.06f, 0.0f,
		-0.5f, -1.06f, 0.0f,
		
		0.0f, 1.06f, 0.0f,
		0.5f, -1.06f, 0.0f,
		None
	};
		

	GLfloat A_Color[] = {
		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f,
	
		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f,
		None	
	};


	GLfloat Flag_Position[] = {
		-0.207, 0.06f, 0.0f,
		0.207f, 0.06f, 0.0f,
		
		-0.218f, 0.0f, 0.0f,
		0.219f, 0.0f, 0.0f,
		
		-0.235, -0.06f, 0.0f,
		0.235, -0.06f, 0.0f,
		None			
	};
	
	
	GLfloat Flag_Color[] = {
		1.0f, 0.6f, 0.2f,
		1.0f, 0.6f, 0.2f,
	
		1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,
		
		0.0705f, 0.533f, 0.0274f,
		0.0705f, 0.533f, 0.0274f,
		None		
	};


	/********** I **********/
	glGenVertexArrays(1, &vao_I);
	glBindVertexArray(vao_I);
	
		/********** Position **********/
		glGenBuffers(1, &vbo_I_Position);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_I_Position);
		glBufferData(GL_ARRAY_BUFFER, 
				sizeof(I_Position),
				I_Position,
				GL_STATIC_DRAW);
				
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
					3,
					GL_FLOAT,
					GL_FALSE,
					0, NULL);
					
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
		
		/********** Color **********/
		glGenBuffers(1, &vbo_I_Color);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_I_Color);
		glBufferData(GL_ARRAY_BUFFER,
				sizeof(I_Color),
				I_Color,
				GL_STATIC_DRAW);
			
		glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
					3,
					GL_FLOAT,
					GL_FALSE,
					0, NULL);
					
		glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
	glBindVertexArray(0);
	
	
	
	/********** N **********/
	glGenVertexArrays(1, &vao_N);
	glBindVertexArray(vao_N);
	
		/********** Position **********/
		glGenBuffers(1, &vbo_N_Position);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_N_Position);
		glBufferData(GL_ARRAY_BUFFER, 
				sizeof(N_Position),
				N_Position,
				GL_STATIC_DRAW);
				
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
					3,
					GL_FLOAT,
					GL_FALSE,
					0, NULL);
					
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
		
		/********** Color **********/
		glGenBuffers(1, &vbo_N_Color);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_N_Color);
		glBufferData(GL_ARRAY_BUFFER,
				sizeof(N_Color),
				N_Color,
				GL_STATIC_DRAW);
			
		glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
					3,
					GL_FLOAT,
					GL_FALSE,
					0, NULL);
					
		glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
	glBindVertexArray(0);
	
	
	
	/********** D **********/
	glGenVertexArrays(1, &vao_D);
	glBindVertexArray(vao_D);
	
		/********** Position **********/
		glGenBuffers(1, &vbo_D_Position);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_D_Position);
		glBufferData(GL_ARRAY_BUFFER, 
				sizeof(D_Position),
				D_Position,
				GL_STATIC_DRAW);
				
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
					3,
					GL_FLOAT,
					GL_FALSE,
					0, NULL);
					
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
		
		/********** Color **********/
		glGenBuffers(1, &vbo_D_Color);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_D_Color);
		glBufferData(GL_ARRAY_BUFFER,
				sizeof(D_Color),
				D_Color,
				GL_STATIC_DRAW);
			
		glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
					3,
					GL_FLOAT,
					GL_FALSE,
					0, NULL);
					
		glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
	glBindVertexArray(0);
	
	
	
	/********** A **********/
	glGenVertexArrays(1, &vao_A);
	glBindVertexArray(vao_A);
	
		/********** Position **********/
		glGenBuffers(1, &vbo_A_Position);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_A_Position);
		glBufferData(GL_ARRAY_BUFFER, 
				sizeof(A_Position),
				A_Position,
				GL_STATIC_DRAW);
				
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
					3,
					GL_FLOAT,
					GL_FALSE,
					0, NULL);
					
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
		
		/********** Color **********/
		glGenBuffers(1, &vbo_A_Color);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_A_Color);
		glBufferData(GL_ARRAY_BUFFER,
				sizeof(A_Color),
				A_Color,
				GL_STATIC_DRAW);
			
		glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
					3,
					GL_FLOAT,
					GL_FALSE,
					0, NULL);
					
		glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
	glBindVertexArray(0);
	
	
	/********** Flag **********/
	glGenVertexArrays(1, &vao_Flag);
	glBindVertexArray(vao_Flag);
	
		/********** Position **********/
		glGenBuffers(1, &vbo_Flag_Position);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Flag_Position);
		glBufferData(GL_ARRAY_BUFFER, 
				sizeof(Flag_Position),
				Flag_Position,
				GL_STATIC_DRAW);
				
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
					3,
					GL_FLOAT,
					GL_FALSE,
					0, NULL);
					
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
		
		/********** Color **********/
		glGenBuffers(1, &vbo_Flag_Color);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Flag_Color);
		glBufferData(GL_ARRAY_BUFFER,
				sizeof(Flag_Color),
				Flag_Color,
				GL_STATIC_DRAW);
			
		glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
					3,
					GL_FLOAT,
					GL_FALSE,
					0, NULL);
					
		glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
	glBindVertexArray(0);

	
	

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);
	

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	resize(WIN_WIDTH, WIN_HEIGHT);
}

void uninitialize(void){
	
	GLXContext currentContext = glXGetCurrentContext();

	//Flag
	if(vbo_Flag_Color){
		glDeleteBuffers(1, &vbo_Flag_Color);
		vbo_Flag_Color = 0;
	}
	
	if(vbo_Flag_Position){
		glDeleteBuffers(1, &vbo_Flag_Position);
		vbo_Flag_Position = 0;
	}
	
	if(vao_Flag){
		glDeleteVertexArrays(1, &vao_Flag);
		vao_Flag = 0;
	}


	//A
	if(vbo_A_Color){
		glDeleteBuffers(1, &vbo_A_Color);
		vbo_A_Color = 0;
	}
	
	if(vbo_A_Position){
		glDeleteBuffers(1, &vbo_A_Position);
		vbo_A_Position = 0;
	}

	if(vao_A){
		glDeleteVertexArrays(1, &vao_A);
		vao_A = 0;
	}

	//D
	if(vbo_D_Color){
		glDeleteBuffers(1, &vbo_D_Color);
		vbo_D_Color = 0;
	}
	
	if(vbo_D_Position){
		glDeleteBuffers(1, &vbo_D_Position);
		vbo_D_Position = 0;
	}

	if(vao_D){
		glDeleteVertexArrays(1, &vao_D);
		vao_D = 0;
	}
	
	//N
	if(vbo_N_Color){
		glDeleteBuffers(1, &vbo_N_Color);
		vbo_N_Color = 0;
	}
	
	if(vbo_N_Position){
		glDeleteBuffers(1, &vbo_N_Position);
		vbo_N_Position = 0;
	}

	if(vao_N){
		glDeleteVertexArrays(1, &vao_N);
		vao_N = 0;
	}
	
	//I
	if(vbo_I_Color){
		glDeleteBuffers(1, &vbo_I_Color);
		vbo_I_Color = 0;
	}
	
	if(vbo_I_Position){
		glDeleteBuffers(1, &vbo_I_Position);
		vbo_I_Position = 0;
	}

	if(vao_I){
		glDeleteVertexArrays(1, &vao_I);
		vao_I = 0;
	}


	if(gShaderProgramObject){
		
		GLint iShaderNo = 0;
		GLint iShaderCount = 0;

		glUseProgram(gShaderProgramObject);

			glGetProgramiv(gShaderProgramObject, GL_ATTACHED_SHADERS, &iShaderCount);
			GLuint *pShaders = (GLuint*)malloc(sizeof(GLuint) * iShaderCount);

			if(pShaders){
				
				glGetAttachedShaders(gShaderProgramObject, iShaderCount, &iShaderCount, pShaders);

				for(iShaderNo = 0; iShaderNo < iShaderCount; iShaderNo++){
					glDetachShader(gShaderProgramObject, pShaders[iShaderNo]);
					glDeleteShader(pShaders[iShaderNo]);
					pShaders[iShaderNo] = 0;
				}
				free(pShaders);
				pShaders = NULL;
			}

		glUseProgram(0);
		glDeleteProgram(gShaderProgramObject);
		gShaderProgramObject = 0;
	}
	
	if(currentContext != NULL && currentContext == gGLXContext){
		glXMakeCurrent(gpDisplay, 0, 0);
	}

	if(gGLXContext){
		glXDestroyContext(gpDisplay, gGLXContext);
	}

	if(gWindow){
		XDestroyWindow(gpDisplay, gWindow);
	}

	if(gColormap){
		XFreeColormap(gpDisplay, gColormap);
	}

	if(gpXVisualInfo){
		free(gpXVisualInfo);
		gpXVisualInfo = NULL;
	}

	if(gpDisplay){
		XCloseDisplay(gpDisplay);
		gpDisplay = NULL;
	}
}

void resize(int width, int height){

	if(height == 0){
		height = 1;
	}

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	perspectiveProjectionMatrix = mat4::identity();
	perspectiveProjectionMatrix = perspective(45.0f,
					(GLfloat)width / (GLfloat)height,
					0.1f,
					100.0f);
}



mat4 translateMatrix;
mat4 modelViewMatrix;
mat4 modelViewProjectionMatrix;


void display(void){

	void My_I(GLfloat, GLfloat, GLfloat, GLfloat);
	void My_N(GLfloat, GLfloat, GLfloat, GLfloat);
	void My_D(GLfloat, GLfloat, GLfloat, GLfloat);
	void My_A(GLfloat, GLfloat, GLfloat, GLfloat);
	void My_Flag(GLfloat, GLfloat, GLfloat, GLfloat);


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	
	glUseProgram(gShaderProgramObject);
		
		//I
		My_I(-2.0f, 0.0f, -8.0f, 20.0f);
		
		//N
		My_N(-1.35f, 0.0f, -8.0f, 20.0f);
		
		//D
		My_D(-0.15f, 0.0f, -8.0f, 20.0f);
		
		//I
		My_I(1.02f, 0.0f, -8.0f, 20.0f);
		
		//A
		My_A(2.0f, 0.0f, -8.0f, 20.0f);		
		
		//Flag
		My_Flag(2.0f, 0.0f, -8.0f, 20.0f);
		
	glUseProgram(0);
		
	glXSwapBuffers(gpDisplay, gWindow);
}

void My_I(GLfloat x, GLfloat y, GLfloat z, GLfloat fWidth){
	
	translateMatrix = mat4::identity();
	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();
	
	translateMatrix = translate(x, y, z);
	modelViewMatrix = modelViewMatrix * translateMatrix;
	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;
	
	glUniformMatrix4fv(mvpUniform,
				1, 
				GL_FALSE,
			modelViewProjectionMatrix);
			
	glLineWidth(fWidth);
	glBindVertexArray(vao_I);
		glDrawArrays(GL_LINES, 0, 6 * 3);
	glBindVertexArray(0);		
}

void My_N(GLfloat x, GLfloat y, GLfloat z, GLfloat fWidth){
	
	translateMatrix = mat4::identity();
	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();
	
	translateMatrix = translate(x, y, z);
	modelViewMatrix = modelViewMatrix * translateMatrix;
	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;
	
	glUniformMatrix4fv(mvpUniform,
				1, 
				GL_FALSE,
			modelViewProjectionMatrix);
			
	glLineWidth(fWidth);		
	glBindVertexArray(vao_N);
		glDrawArrays(GL_LINES, 0, 6 * 3);
	glBindVertexArray(0);		
}


void My_D(GLfloat x, GLfloat y, GLfloat z, GLfloat fWidth){
	
	translateMatrix = mat4::identity();
	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();
	
	translateMatrix = translate(x, y, z);
	modelViewMatrix = modelViewMatrix * translateMatrix;
	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;
	
	glUniformMatrix4fv(mvpUniform,
				1, 
				GL_FALSE,
			modelViewProjectionMatrix);
			
	glLineWidth(fWidth);		
	glBindVertexArray(vao_D);
		glDrawArrays(GL_LINES, 0, 8 * 3);
	glBindVertexArray(0);		
}

void My_A(GLfloat x, GLfloat y, GLfloat z, GLfloat fWidth){
	
	translateMatrix = mat4::identity();
	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();
	
	translateMatrix = translate(x, y, z);
	modelViewMatrix = modelViewMatrix * translateMatrix;
	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;
	
	glUniformMatrix4fv(mvpUniform,
				1, 
				GL_FALSE,
			modelViewProjectionMatrix);
			
	glLineWidth(fWidth);		
	glBindVertexArray(vao_A);
		glDrawArrays(GL_LINES, 0, 4 * 3);
	glBindVertexArray(0);		
}



void My_Flag(GLfloat x, GLfloat y, GLfloat z, GLfloat fWidth){
	
	translateMatrix = mat4::identity();
	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();
	
	translateMatrix = translate(x, y, z);
	modelViewMatrix = modelViewMatrix * translateMatrix;
	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;
	
	glUniformMatrix4fv(mvpUniform,
				1, 
				GL_FALSE,
			modelViewProjectionMatrix);
			
	glLineWidth(fWidth);		
	glBindVertexArray(vao_Flag);
		glDrawArrays(GL_LINES, 0, 6 * 3);
	glBindVertexArray(0);		
}


