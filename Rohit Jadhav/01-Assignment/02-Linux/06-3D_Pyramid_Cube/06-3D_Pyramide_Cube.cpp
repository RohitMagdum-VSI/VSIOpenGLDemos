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
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOORD0
};

//For FullScreen
bool bIsFullScreen = false;

//For Window
Display *gpDisplay = NULL;
XVisualInfo *gpXVisualInfo = NULL;
Colormap gColormap;
Window gWindow;

//For OpenGL
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
glXCreateContextAttribsARBProc glXCreateContextAttribsARB = NULL;
GLXFBConfig gGLXFBConfig;
GLXContext gGLXContext;

//For Pyramid
GLuint vao_Pyramid;
GLuint vbo_Pyramid_Position;
GLuint vbo_Pyramid_Color;
GLfloat angle_Pyramid = 0.0f;

//For Cube
GLuint vao_Cube;
GLuint vbo_Cube_Position;
GLuint vbo_Cube_Color;
GLfloat angle_Cube = 360.0f;

//For Matrix;
GLuint mvpUniform;
mat4 perspectiveProjectionMatrix;

//For ShaderProgram
GLuint gShaderProgramObject;

//For Error
FILE *gbFile = NULL;

int main(void){
	
	void CreateWindow(void);
	void initialize(void);
	void ToggleFullScreen(void);
	void resize(int, int);
	void update(void);
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

void CreateWindow(void){
	
	void uninitialize(void);

	int defaultScreen;
	XSetWindowAttributes winAttribs;
	int styleMask;

	static int frameBufferAttribs[] = {
		GLX_X_RENDERABLE, True,
		GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
		GLX_RENDER_TYPE, GLX_RGBA_BIT,
		GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
		GLX_RED_SIZE, 8,
		GLX_GREEN_SIZE, 8,
		GLX_BLUE_SIZE, 8,
		GLX_ALPHA_SIZE, 8,
		GLX_DEPTH_SIZE, 24,
		None
	};

	GLXFBConfig bestFBConfig;
	GLXFBConfig *pGLXFBConfig = NULL;
	XVisualInfo *pTempXVisualInfo = NULL;
	int iNumberOfFBConfig;

	gpDisplay = XOpenDisplay(NULL);
	if(gpDisplay == NULL){
		fprintf(gbFile, "XOpenDisplay() Failed!!\n");
		uninitialize();
		exit(1);
	}

	defaultScreen = XDefaultScreen(gpDisplay);

	pGLXFBConfig = glXChooseFBConfig(gpDisplay, defaultScreen, frameBufferAttribs, &iNumberOfFBConfig);
	fprintf(gbFile, "Matched: %d\n", iNumberOfFBConfig);

	int bestFrameBufferConfig = -1;
	int bestNumberOfSamples = -1;
	int worstFrameBufferConfig = -1;
	int worstNumberOfSamples = -1;

	for(int i = 0; i < iNumberOfFBConfig; i++){
		pTempXVisualInfo = glXGetVisualFromFBConfig(gpDisplay, pGLXFBConfig[i]);
		if(pTempXVisualInfo){
			int samples, sampleBuffers;

			glXGetFBConfigAttrib(gpDisplay, pGLXFBConfig[i], GLX_SAMPLE_BUFFERS, &sampleBuffers);
			glXGetFBConfigAttrib(gpDisplay, pGLXFBConfig[i], GLX_SAMPLES, &samples);

			if(bestFrameBufferConfig < 0 || sampleBuffers && samples > bestNumberOfSamples){
				bestFrameBufferConfig = i;
				bestNumberOfSamples = samples;
			}

			if(worstFrameBufferConfig < 0 || sampleBuffers && samples < worstNumberOfSamples){
				worstFrameBufferConfig = i;
				worstNumberOfSamples = samples;
			}
		}	
		XFree(pTempXVisualInfo);
		pTempXVisualInfo = NULL;
	}

	bestFBConfig = pGLXFBConfig[bestFrameBufferConfig];
	gGLXFBConfig = bestFBConfig;
	XFree(pGLXFBConfig);

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

	styleMask = CWBackPixel | CWBorderPixel | CWEventMask | CWColormap;

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
		fprintf(gbFile, "XCreateWindow() Failed!!\n");
		uninitialize();
		exit(1);
	}

	XStoreName(gpDisplay, gWindow, "06-3D_Pyramid_And_Cube_Rotation");

	Atom windowManagerDelete = XInternAtom(gpDisplay, "WM_DELETE_WINDOW", True);
	XSetWMProtocols(gpDisplay, gWindow, &windowManagerDelete, 1);

	XMapWindow(gpDisplay, gWindow);
}

void ToggleFullScreen(void){
	
	Atom wm_state;
	Atom fullscreen;
	XEvent xev;

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
	
	void uninitialize(void);
	void resize(int, int);


	GLuint vertexShaderObject;
	GLuint fragmentShaderObject;
	GLenum Result;
	
	/********** Context **********/
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
		fprintf(gbFile, "glXCreateContextAttribsARB() Failed!!\n");
		fprintf(gbFile, "Geting Context From System\n");

		const int Attribs[] = {
			GLX_CONTEXT_MAJOR_VERSION_ARB, 1,
			GLX_CONTEXT_MINOR_VERSION_ARB, 0,
			None
		};

		gGLXContext = glXCreateContextAttribsARB(gpDisplay, gGLXFBConfig, NULL, True, Attribs);
	}

	if(!glXIsDirect(gpDisplay, gGLXContext))
		fprintf(gbFile, "Software Context!!\n");
	else
		fprintf(gbFile, "Hardware Context!!\n");

	glXMakeCurrent(gpDisplay, gWindow, gGLXContext);



	Result = glewInit();
	if(Result != GLEW_OK){
		fprintf(gbFile, "glewInit() Failed!!\n");
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
		"out vec4 OutColor;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void)" \
		"{" \
			"gl_Position = u_mvp_matrix * vPosition;" \ 
			"OutColor = vColor;" \
		"}";

	glShaderSource(vertexShaderObject, 1,
		(const char**)&vertexShaderSourceCode, NULL);

	glCompileShader(vertexShaderObject);

	int iShaderCompileStatus = 0;
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
			fprintf(gbFile, "VS: Memory Allocation Error for szInfoLog cha Malloc!!\n");
			uninitialize();
			exit(1);
		}
	}


	/********* Fragment Shader **********/
	fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const char *fragmentShaderSourceCode = 
		"#version 450 core" \
		"\n" \
		"in vec4 OutColor;" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
			"FragColor = OutColor;" \
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
			szInfoLog =(char*)malloc(sizeof(char) * iInfoLogLength); 
			if(szInfoLog){
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gbFile, "Fragment Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				exit(1);

			}
			fprintf(gbFile, "FS: Memory Allocation Error szInfoLog's Malloc!!\n");
			uninitialize();
			exit(0);
		}
	}


	/********** Program Object **********/
	gShaderProgramObject = glCreateProgram();

	glAttachShader(gShaderProgramObject, vertexShaderObject);
	glAttachShader(gShaderProgramObject, fragmentShaderObject);

	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_COLOR, "vColor");

	glLinkProgram(gShaderProgramObject);

	int iProgramLinkStatus = 0;
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
				fprintf(gbFile, "Shader Program Link Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				exit(1);
			}
			fprintf(gbFile, "SP: Malloc error: %s\n", szInfoLog);
			uninitialize();
			exit(1);
		}
	}


	mvpUniform = glGetUniformLocation(gShaderProgramObject, "u_mvp_matrix");


	/********** Positions **********/
	GLfloat Pyramid_Vertices[] = {
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
		-1.0f, -1.0f, 1.0f,
		None
	};

	GLfloat Cube_Vertices[] = {
		//Top
		1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,
		//Bottom
		1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		//Front
		1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		//Back
		1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,
		//Right
		1.0f, 1.0f, -1.0f,
		1.0f, 1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f,
		//Left
		-1.0f, 1.0f, 1.0f, 
		-1.0f, 1.0f, -1.0f, 
		-1.0f, -1.0f, -1.0f, 
		-1.0f, -1.0f, 1.0f,
	       	None	
	};


	/********** Color **********/
	GLfloat Pyramid_Color[] = {
		//Front
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f,
		//Right
		1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 1.0f, 0.0f,
		//Back
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f,
		//Left
		1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 1.0f, 0.0f,
		None
	};

	GLfloat Cube_Color[] = {
		//Top
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		//Bottom
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		//Front
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		//Back
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		//Right
		0.0f, 1.0f, 1.0f,
		0.0f, 1.0f, 1.0f,
		0.0f, 1.0f, 1.0f,
		0.0f, 1.0f, 1.0f,
		//Left
		1.0f, 0.0f, 1.0f,
		1.0f, 0.0f, 1.0f,
		1.0f, 0.0f, 1.0f,
		1.0f, 0.0f, 1.0f,
		None
	};



	/********** Pyramid Vao **********/
	glGenVertexArrays(1, &vao_Pyramid);
	glBindVertexArray(vao_Pyramid);
		
		/********** Position **********/
		glGenBuffers(1, &vbo_Pyramid_Position);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Pyramid_Position);
		glBufferData(GL_ARRAY_BUFFER,
				sizeof(Pyramid_Vertices),
				Pyramid_Vertices,
				GL_STATIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
					3,
					GL_FLOAT,
					GL_FALSE,
					0, NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/********** Color **********/
		glGenBuffers(1, &vbo_Pyramid_Color);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Pyramid_Color);
		glBufferData(GL_ARRAY_BUFFER, 
				sizeof(Pyramid_Color),
				Pyramid_Color,
				GL_STATIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
					3,
					GL_FLOAT,
					GL_FALSE,
					0, NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);


	/********** Cube Vao **********/
	glGenVertexArrays(1, &vao_Cube);
	glBindVertexArray(vao_Cube);

		/********** Position **********/
		glGenBuffers(1, &vbo_Cube_Position);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Cube_Position);
		glBufferData(GL_ARRAY_BUFFER, 
				sizeof(Cube_Vertices),
				Cube_Vertices,
				GL_STATIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
					3,
					GL_FLOAT,
					GL_FALSE,
					0, NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/********** Color **********/
		glGenBuffers(1, &vbo_Cube_Color);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Cube_Color);
		glBufferData(GL_ARRAY_BUFFER,
				sizeof(Cube_Color),
				Cube_Color,
				GL_STATIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
					3,
					GL_FLOAT,
					GL_FALSE,
					0, NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


	glBindVertexArray(vao_Cube);


	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	glDisable(GL_CULL_FACE);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	//perspectiveProjectionMatrix = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
}

void uninitialize(void){
	
	GLXContext currentContext = glXGetCurrentContext();

	if(vbo_Cube_Color){
		glDeleteBuffers(1, &vbo_Cube_Color);
		vbo_Cube_Color = 0;
	}

	if(vbo_Cube_Position){
		glDeleteBuffers(1, &vbo_Cube_Position);
		vbo_Cube_Position = 0;
	}

	if(vao_Cube){
		glDeleteVertexArrays(1, &vao_Cube);
		vao_Cube = 0;
	}

	if(vbo_Pyramid_Color){
		glDeleteBuffers(1, &vbo_Pyramid_Color);
		vbo_Pyramid_Color = 0;
	}
	
	if(vbo_Pyramid_Position){
		glDeleteBuffers(1, &vbo_Pyramid_Position);
		vbo_Pyramid_Position = 0;
	}

	if(vao_Pyramid){
		glDeleteVertexArrays(1, &vao_Pyramid);
		vao_Pyramid = 0;
	}

	if(gShaderProgramObject){
	
		glUseProgram(gShaderProgramObject);

			int iShaderNo;
			int iShaderCount;	
			GLuint *pShaders = NULL;

			glGetProgramiv(gShaderProgramObject, GL_ATTACHED_SHADERS, &iShaderCount);
			pShaders = (GLuint*)malloc(sizeof(GLuint) * iShaderCount);
			if(pShaders){
				glGetAttachedShaders(gShaderProgramObject, iShaderCount, &iShaderCount, pShaders);

				for(iShaderNo = 0; iShaderNo < iShaderCount; iShaderNo++){
					glDetachShader(gShaderProgramObject, pShaders[iShaderNo]);
					glDeleteShader(pShaders[iShaderNo]);
					pShaders[iShaderNo] = 0;
				}
				free(pShaders);
				pShaders = 0;
			}


		glUseProgram(0);
		glDeleteProgram(gShaderProgramObject);
		gShaderProgramObject = 0;

	}


	if(currentContext != NULL && currentContext == gGLXContext)
		glXMakeCurrent(gpDisplay, 0, 0);

	if(gGLXContext)
		glXDestroyContext(gpDisplay, gGLXContext);

	if(gWindow)
		XDestroyWindow(gpDisplay, gWindow);

	if(gColormap)
		XFreeColormap(gpDisplay, gColormap);

	if(gpXVisualInfo){
		XFree(gpXVisualInfo);
		gpXVisualInfo = NULL;
	}

	if(gpDisplay){
		XCloseDisplay(gpDisplay);
		gpDisplay = NULL;
	}

	if(gbFile){
		fprintf(gbFile, "Log Close!!\n");
		fclose(gbFile);
		gbFile = NULL;
	}
}

void resize(int width, int height){
	
	if(height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	perspectiveProjectionMatrix = mat4::identity();
	perspectiveProjectionMatrix = perspective(45.0f,
						(GLfloat)width / (GLfloat)height,
						0.1f,
						100.0f);
}

void display(void){
	
	mat4 translateMatrix;
	mat4 rotateMatrix;
	mat4 modelViewMatrix;
	mat4 modelViewProjectionMatrix;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(gShaderProgramObject);

		/********** Pyramid **********/
		translateMatrix = mat4::identity();
		rotateMatrix = mat4::identity();
		modelViewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();

		translateMatrix = translate(-2.0f, 0.0f, -6.0f);
		rotateMatrix = rotate(angle_Pyramid, 0.0f, 1.0f, 0.0f);
		modelViewMatrix = modelViewMatrix * translateMatrix * rotateMatrix;
		modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

		glUniformMatrix4fv(mvpUniform,
					1,
					GL_FALSE,
					modelViewProjectionMatrix);

		glBindVertexArray(vao_Pyramid);
			glDrawArrays(GL_TRIANGLES,
					0,
					12);

		glBindVertexArray(0);


		/********** Cube **********/
		translateMatrix = mat4::identity();
		rotateMatrix = mat4::identity();
		modelViewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();

		translateMatrix = translate(2.0f, 0.0f, -6.0f);
		rotateMatrix = rotate(angle_Cube, angle_Cube, angle_Cube);
		modelViewMatrix = modelViewMatrix * translateMatrix * rotateMatrix;
		modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

		glUniformMatrix4fv(mvpUniform,
					1,
					GL_FALSE,
					modelViewProjectionMatrix);

		glBindVertexArray(vao_Cube);
			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
			glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
			glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
			glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
			glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
			glDrawArrays(GL_TRIANGLE_FAN, 20, 4);

		glBindVertexArray(0);

	glUseProgram(0);

	glXSwapBuffers(gpDisplay, gWindow);
}

void update(void){
	
	angle_Pyramid = angle_Pyramid + 0.1f;
	angle_Cube = angle_Cube - 0.1f;
	if(angle_Pyramid > 360.0f)
		angle_Pyramid = 0.0f;

	if(angle_Cube < 0.0f)
		angle_Cube = 360.0f;
}



