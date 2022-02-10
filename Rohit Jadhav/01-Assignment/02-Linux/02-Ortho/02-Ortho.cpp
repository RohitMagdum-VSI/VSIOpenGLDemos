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

#define WIN_WIDTH 800
#define WIN_HEIGHT  600

using namespace std;
using namespace vmath;

//Attributes For Programmable Pipeline
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
typedef GLXContext (*GLXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);

GLXCreateContextAttribsARBProc GLXCreateContextAttribsARB = NULL;

GLXFBConfig gGLXFBConfig;

GLXContext gGLXContext;

//For Programmable Pipeline
GLuint gShaderProgramObject;
GLuint vao_Triangle;
GLuint vbo_Position;
GLuint mvpUniform;
mat4 orthographicProjectionMatrix;

//For Error
FILE *gbFile = NULL;

int main(void){
	

	void CreateWindow(void);
	void ToggleFullScreen(void);
	void initialize(void);
	void resize(int, int);
	void display(void);
	void uninitialize(void);

	int winWidth = WIN_WIDTH;
	int winHeight = WIN_HEIGHT;
	bool bDone = false;
	XEvent event;
	KeySym keysym;

	gbFile = fopen("Log.txt", "w");
	if(!gbFile){
		printf("Error Creating Log file!!\n");
		exit(1);
	}

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

	//For FBConfig
	GLXFBConfig *pGLXFBConfig = NULL;
	GLXFBConfig bestGLXFBConfig;
	XVisualInfo *pTempXVisualInfo = NULL;
	int iNumberOfFBConfig = 0;

	static int frameBufferAttributes[] = 
	{
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
		GLX_STENCIL_SIZE, 8,
		None
	};

	gpDisplay = XOpenDisplay(NULL);
	if(gpDisplay == NULL){
		fprintf(gbFile, "XOpenDisplay() Failed!!\n");
		uninitialize();
		exit(1);
	}

	defaultScreen = XDefaultScreen(gpDisplay);

	//Getting Visual From GLXFBConfig
	
	pGLXFBConfig = glXChooseFBConfig(gpDisplay, defaultScreen, frameBufferAttributes, &iNumberOfFBConfig);
	fprintf(gbFile, "%d Matching GLXFBConfig\n", iNumberOfFBConfig);

	int bestFrameBufferConfig = -1;
	int bestNumberOfSamples = -1;
	int worstFrameBufferConfig = -1;
	int worstNumberOfSamples = -1;

	for(int i = 0; i < iNumberOfFBConfig; i++){
		int sampleBuffers, samples;

		pTempXVisualInfo = glXGetVisualFromFBConfig(gpDisplay, pGLXFBConfig[i]);

		if(pTempXVisualInfo){
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
	}

	bestGLXFBConfig = pGLXFBConfig[bestFrameBufferConfig];

	gGLXFBConfig = bestGLXFBConfig;

	XFree(pGLXFBConfig);

	gpXVisualInfo = glXGetVisualFromFBConfig(gpDisplay, bestGLXFBConfig);
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
			gpXVisualInfo->screen,
			InputOutput,
			gpXVisualInfo->visual,
			styleMask,
			&winAttribs);

	if(!gWindow){
		fprintf(gbFile, "XCreateWindow() Failed!!\n");
		uninitialize();
		exit(1);
	}

	XStoreName(gpDisplay, gWindow, "02-Ortho");

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
		
	GLenum Result;
	GLuint vertexShaderObject;
	GLuint fragmentShaderObject;

	const int Attribs[] = {
		GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
		GLX_CONTEXT_MINOR_VERSION_ARB, 5,
		GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
		None
	};


	//Creating Context
	GLXCreateContextAttribsARB = (GLXCreateContextAttribsARBProc)glXGetProcAddressARB((GLubyte*)"glXCreateContextAttribsARB");

	if(GLXCreateContextAttribsARB == NULL){
		fprintf(gbFile, "glXGetProcAdderssARB() Failed!!\n");
		uninitialize();
		exit(1);
	}

	gGLXContext = GLXCreateContextAttribsARB(gpDisplay, gGLXFBConfig, NULL, True, Attribs);

	if(!gGLXContext){
		const int pAttribs[] = {
			GLX_CONTEXT_MAJOR_VERSION_ARB, 1,
			GLX_CONTEXT_MINOR_VERSION_ARB, 0,
			None
		};

		GLXCreateContextAttribsARB(gpDisplay, gGLXFBConfig, NULL, True, pAttribs);
	}	

	if(!glXIsDirect(gpDisplay, gGLXContext)){
		fprintf(gbFile, "Not H/W Rendering Context\n");
	}
	else{
		fprintf(gbFile, "H/W Rendering Context\n");
	} 

	glXMakeCurrent(gpDisplay, gWindow, gGLXContext);


	Result = glewInit();
	if(Result != GLEW_OK){
		fprintf(gbFile, "glewInit() Failed!!\n");
		uninitialize();
		exit(1);
	}


	//VS
	vertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *vertexShaderSourceCode = 
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"uniform mat4 u_mvp_matrix;" \ 
		"void main(void)" \
		"{" \ 
		"gl_Position = u_mvp_matrix * vPosition;" \ 
		"}";

	glShaderSource(vertexShaderObject, 1, (const GLchar**)&vertexShaderSourceCode, NULL);

	glCompileShader(vertexShaderObject);

	int iShaderCompileStatus = 0;
	int iInfoLogLength = 0;
	char *szInfoLog = NULL;

	glGetShaderiv(vertexShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if(iShaderCompileStatus == GL_FALSE){
		glGetShaderiv(vertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if(iInfoLogLength > 0){
			szInfoLog = (char*)malloc(sizeof(char) * iInfoLogLength);
			if(szInfoLog != NULL){
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

	//FS
	fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *fragmentShaderSourceCode = 
		"#version 450 core" \ 
		"\n" \
		"out vec4 FragColor;" \ 
		"void main()" \ 
		"{" \ 
			"FragColor = vec4(1.0, 1.0, 1.0, 1.0);" \ 
		"}";

	glShaderSource(fragmentShaderObject, 1, (const GLchar**)&fragmentShaderSourceCode, NULL);

	glCompileShader(fragmentShaderObject);

	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(fragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if(iShaderCompileStatus == GL_FALSE){
		glGetShaderiv(fragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if(iInfoLogLength > 0){
			szInfoLog = (char*)malloc(sizeof(char) * iInfoLogLength);
			if(szInfoLog != NULL){
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


	//SP
	gShaderProgramObject = glCreateProgram();

	glAttachShader(gShaderProgramObject, vertexShaderObject);
	glAttachShader(gShaderProgramObject, fragmentShaderObject);

	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_POSITION, "vPosition");

	glLinkProgram(gShaderProgramObject);

	int iLinkStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	
	glGetProgramiv(gShaderProgramObject, GL_LINK_STATUS, &iLinkStatus);
	if(iLinkStatus == GL_FALSE){
		glGetProgramiv(gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if(iInfoLogLength > 0){
			szInfoLog = (char*)malloc(sizeof(char) * iInfoLogLength);
			if(szInfoLog != NULL){
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

	const GLfloat Tri_Vertex[] = {	0.0f, 50.0f, 0.0f,
					-50.0f, -50.0f, 0.0f,
					50.0f, -50.0f, 0.0f};

	glGenVertexArrays(1, &vao_Triangle);
	glBindVertexArray(vao_Triangle);

		glGenBuffers(1, &vbo_Position);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Position);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Tri_Vertex), Tri_Vertex, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	glBindVertexArray(0);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	orthographicProjectionMatrix = mat4::identity();

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	resize(WIN_WIDTH, WIN_HEIGHT);
}

void uninitialize(void){

	GLsizei shaderCount;
	GLsizei shaderNumber;

	GLXContext currentContext = glXGetCurrentContext();

	if(gShaderProgramObject){
		glUseProgram(gShaderProgramObject);
		glGetProgramiv(gShaderProgramObject, GL_ATTACHED_SHADERS, &shaderCount);
		GLuint *pShaders = (GLuint*)malloc(sizeof(GLuint) * shaderCount);
		if(pShaders){
			glGetAttachedShaders(gShaderProgramObject, shaderCount, &shaderCount, pShaders);
			for(shaderNumber = 0; shaderNumber < shaderCount; shaderNumber++){
				glDetachShader(gShaderProgramObject, pShaders[shaderNumber]);
				glDeleteShader(pShaders[shaderNumber]);
				pShaders[shaderNumber] = 0;
			}
			free(pShaders);
			pShaders = NULL;	
		}
		glDeleteProgram(gShaderProgramObject);
		gShaderProgramObject = 0;
		glUseProgram(0);
	}

	if(vbo_Position){
		glDeleteBuffers(1, &vbo_Position);
		vbo_Position = 0;
	}

	if(vao_Triangle){
		glDeleteVertexArrays(1, &vao_Triangle);
		vao_Triangle = 0;
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

	if(gGLXFBConfig){
		//XFree(gGLXFBConfig);
	}

	if(gbFile){
		fprintf(gbFile, "Log Close!!\n");
		fclose(gbFile);
		gbFile = NULL;
	}

	if(gpDisplay){
		XCloseDisplay(gpDisplay);
		gpDisplay = NULL;
	}
}

void resize(int width, int height){
	if(height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	if(width <= height){
		orthographicProjectionMatrix = ortho(-100.0f, 100.0f,
						(-100.0f * (GLfloat)height / (GLfloat)width),
						(100.0f * (GLfloat)height / (GLfloat)width),
						-100.0f, 100.0f);
	}
	else{
		orthographicProjectionMatrix = ortho((-100.0f * (GLfloat)width /(GLfloat)height),
						(100.0f * (GLfloat)width / (GLfloat)height),
						-100.0f, 100.0f,
						-100.0f, 100.0f);
	}
}

void display(void){
	
	mat4 modelViewMatrix = mat4::identity();
	mat4 modelViewProjectionMatrix = mat4::identity();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(gShaderProgramObject);

	modelViewProjectionMatrix = orthographicProjectionMatrix * modelViewMatrix;
	glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

	glBindVertexArray(vao_Triangle);

		glDrawArrays(GL_TRIANGLES, 0, 3);

	glBindVertexArray(0);
	glUseProgram(0);

	glXSwapBuffers(gpDisplay, gWindow);
}

