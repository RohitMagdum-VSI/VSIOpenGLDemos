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
bool bIsFullScreen_RRJ = false;

//For Window
Display *gpDisplay_RRJ = NULL;
XVisualInfo *gpXVisualInfo_RRJ = NULL;
Colormap gColormap_RRJ;
Window gWindow_RRJ;

//For OpenGL
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
glXCreateContextAttribsARBProc glXCreateContextAttribsARB_RRJ = NULL;
GLXFBConfig gGLXFBConfig_RRJ;
GLXContext gGLXContext_RRJ;

//For Triangle
GLuint vao_Tri_RRJ;
GLuint vbo_Tri_Position_RRJ;
GLuint vbo_Tri_Color_RRJ;

//For Viewport
int iViewPort_RRJ = 0;
GLsizei viewPortWidth_RRJ = 0.0f;
GLsizei viewPortHeight_RRJ = 0.0f;

//For Uniform
GLuint mvpUniform_RRJ;

//For Matrix;
mat4 gPerspectiveProjectionMatrix_RRJ;

//For ShaderProgram
GLuint gShaderProgramObject_RRJ;

//For Error
FILE *gbFile_RRJ = NULL;


//For Keys
char Key[26];


int main(void){
	
	void CreateWindow(void);
	void initialize(void);
	void ToggleFullScreen(void);
	void resize(int, int);
	void display(void);
	void uninitialize(void);

	int winWidth_RRJ = WIN_WIDTH;
	int winHeight_RRJ = WIN_HEIGHT;
	bool bDone_RRJ = false;
	XEvent event_RRJ;
	KeySym keysym_RRJ;

	gbFile_RRJ = fopen("Log.txt", "w");
	if(gbFile_RRJ == NULL){
		printf("Log Creation Failed!!\n");
		uninitialize();
		exit(1);
	}
	else
		fprintf(gbFile_RRJ, "Log Created!!\n");


	CreateWindow();
	ToggleFullScreen();
	initialize();


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
							bDone_RRJ = true;
							break;

						case XK_F:
						case XK_f:
							if(bIsFullScreen_RRJ == false){
								ToggleFullScreen();
								bIsFullScreen_RRJ = true;
							}
							else{
								ToggleFullScreen();
								bIsFullScreen_RRJ = false;
							}
							break;

						default:
							break;
					}
					
					XLookupString(&event_RRJ.xkey, Key, sizeof(Key), NULL, NULL);
					switch(Key[0]){
						case '0':
							iViewPort_RRJ = 0;
							break;
						case '1':
							iViewPort_RRJ = 1;
							break;
						case '2':
							iViewPort_RRJ = 2;
							break;
						case '3':
							iViewPort_RRJ = 3;
							break;
						case '4':
							iViewPort_RRJ = 4;
							break;
						case '5':
							iViewPort_RRJ = 5;
							break;
						case '6':
							iViewPort_RRJ = 6;
							break;
						case '7':
							iViewPort_RRJ = 7;
							break;
						case '8':
							iViewPort_RRJ = 8;
							break;
						case '9':
							iViewPort_RRJ = 9;
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

void CreateWindow(void){
	
	void uninitialize(void);

	int defaultScreen_RRJ;
	XSetWindowAttributes winAttribs_RRJ;
	int styleMask_RRJ;

	static int frameBufferAttribs_RRJ[] = {
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

	GLXFBConfig bestFBConfig_RRJ;
	GLXFBConfig *pGLXFBConfig_RRJ = NULL;
	XVisualInfo *pTempXVisualInfo_RRJ = NULL;
	int iNumberOfFBConfig_RRJ;

	gpDisplay_RRJ = XOpenDisplay(NULL);
	if(gpDisplay_RRJ == NULL){
		fprintf(gbFile_RRJ, "XOpenDisplay() Failed!!\n");
		uninitialize();
		exit(1);
	}

	defaultScreen_RRJ = XDefaultScreen(gpDisplay_RRJ);

	pGLXFBConfig_RRJ = glXChooseFBConfig(gpDisplay_RRJ, defaultScreen_RRJ, frameBufferAttribs_RRJ, &iNumberOfFBConfig_RRJ);
	fprintf(gbFile_RRJ, "Matched: %d\n", iNumberOfFBConfig_RRJ);

	int bestFrameBufferConfig_RRJ = -1;
	int bestNumberOfSamples_RRJ = -1;
	int worstFrameBufferConfig_RRJ = -1;
	int worstNumberOfSamples_RRJ = -1;

	for(int i = 0; i < iNumberOfFBConfig_RRJ; i++){
		pTempXVisualInfo_RRJ = glXGetVisualFromFBConfig(gpDisplay_RRJ, pGLXFBConfig_RRJ[i]);
		if(pTempXVisualInfo_RRJ){
			int samples, sampleBuffers;

			glXGetFBConfigAttrib(gpDisplay_RRJ, pGLXFBConfig_RRJ[i], GLX_SAMPLE_BUFFERS, &sampleBuffers);
			glXGetFBConfigAttrib(gpDisplay_RRJ, pGLXFBConfig_RRJ[i], GLX_SAMPLES, &samples);

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

	gpXVisualInfo_RRJ = glXGetVisualFromFBConfig(gpDisplay_RRJ, bestFBConfig_RRJ);
	if(gpXVisualInfo_RRJ == NULL){
		fprintf(gbFile_RRJ, "glXGetVisualFromFBConfig() Failed!!\n");
		uninitialize();
		exit(1);
	}

	winAttribs_RRJ.border_pixel = 0;
	winAttribs_RRJ.border_pixmap = 0;
	winAttribs_RRJ.background_pixel = BlackPixel(gpDisplay_RRJ, defaultScreen_RRJ);
	winAttribs_RRJ.background_pixmap = 0;
	winAttribs_RRJ.colormap = XCreateColormap(gpDisplay_RRJ,
					RootWindow(gpDisplay_RRJ, gpXVisualInfo_RRJ->screen),
					gpXVisualInfo_RRJ->visual,
					AllocNone);
	gColormap_RRJ = winAttribs_RRJ.colormap;
	winAttribs_RRJ.event_mask = ExposureMask | VisibilityChangeMask | PointerMotionMask |
				KeyPressMask | ButtonPressMask | StructureNotifyMask;

	styleMask_RRJ = CWBackPixel | CWBorderPixel | CWEventMask | CWColormap;

	gWindow_RRJ = XCreateWindow(gpDisplay_RRJ,
			RootWindow(gpDisplay_RRJ, gpXVisualInfo_RRJ->screen),
			0, 0, 
			WIN_WIDTH, WIN_HEIGHT,
			0,
			gpXVisualInfo_RRJ->depth,
			InputOutput,
			gpXVisualInfo_RRJ->visual,
			styleMask_RRJ,
			&winAttribs_RRJ);

	if(!gWindow_RRJ){
		fprintf(gbFile_RRJ, "XCreateWindow() Failed!!\n");
		uninitialize();
		exit(1);
	}

	XStoreName(gpDisplay_RRJ, gWindow_RRJ, "32-Multi-Viewport");

	Atom windowManagerDelete = XInternAtom(gpDisplay_RRJ, "WM_DELETE_WINDOW", True);
	XSetWMProtocols(gpDisplay_RRJ, gWindow_RRJ, &windowManagerDelete, 1);

	XMapWindow(gpDisplay_RRJ, gWindow_RRJ);
}

void ToggleFullScreen(void){
	
	Atom wm_state_RRJ;
	Atom fullscreen_RRJ;
	XEvent xev_RRJ;

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

void initialize(void){
	
	void uninitialize(void);
	void resize(int, int);


	GLuint vertexShaderObject_RRJ;
	GLuint fragmentShaderObject_RRJ;
	GLenum Result_RRJ;
	
	/********** Context **********/
	glXCreateContextAttribsARB_RRJ = (glXCreateContextAttribsARBProc)glXGetProcAddressARB((GLubyte*)"glXCreateContextAttribsARB");
	if(glXCreateContextAttribsARB_RRJ == NULL){
		fprintf(gbFile_RRJ, "glXGetProcAddressARB() Failed!!\n");
		uninitialize();
		exit(1);
	}

	const int Attributes_RRJ[] = {
		GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
		GLX_CONTEXT_MINOR_VERSION_ARB, 5,
		GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
		None
	};

	gGLXContext_RRJ = glXCreateContextAttribsARB_RRJ(gpDisplay_RRJ, gGLXFBConfig_RRJ, NULL, True, Attributes_RRJ);
	if(gGLXContext_RRJ == NULL){
		fprintf(gbFile_RRJ, "glXCreateContextAttribsARB_RRJ() Failed!!\n");
		fprintf(gbFile_RRJ, "Geting Context From System\n");

		const int Attribs_RRJ[] = {
			GLX_CONTEXT_MAJOR_VERSION_ARB, 1,
			GLX_CONTEXT_MINOR_VERSION_ARB, 0,
			None
		};

		gGLXContext_RRJ = glXCreateContextAttribsARB_RRJ(gpDisplay_RRJ, gGLXFBConfig_RRJ, NULL, True, Attribs_RRJ);
	}

	if(!glXIsDirect(gpDisplay_RRJ, gGLXContext_RRJ))
		fprintf(gbFile_RRJ, "Software Context!!\n");
	else
		fprintf(gbFile_RRJ, "Hardware Context!!\n");

	glXMakeCurrent(gpDisplay_RRJ, gWindow_RRJ, gGLXContext_RRJ);



	Result_RRJ = glewInit();
	if(Result_RRJ != GLEW_OK){
		fprintf(gbFile_RRJ, "glewInit() Failed!!\n");
		uninitialize();
		exit(1);
	}

	
	/********** Vertex Shader **********/
	vertexShaderObject_RRJ = glCreateShader(GL_VERTEX_SHADER);

	const char *vertexShaderSourceCode_RRJ = 
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

	glShaderSource(vertexShaderObject_RRJ, 1,
		(const char**)&vertexShaderSourceCode_RRJ, NULL);

	glCompileShader(vertexShaderObject_RRJ);

	int iShaderCompileStatus_RRJ = 0;
	int iInfoLogLength_RRJ;
	char *szInfoLog_RRJ = NULL;

	glGetShaderiv(vertexShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if(iShaderCompileStatus_RRJ == GL_FALSE){
		glGetShaderiv(vertexShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if(iInfoLogLength_RRJ > 0){
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if(szInfoLog_RRJ){
				GLsizei written;
				glGetShaderInfoLog(vertexShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Vertex Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(1);
			}
			fprintf(gbFile_RRJ, "VS: Memory Allocation Error for szInfoLog_RRJ cha Malloc!!\n");
			uninitialize();
			exit(1);
		}
	}


	/********* Fragment Shader **********/
	fragmentShaderObject_RRJ = glCreateShader(GL_FRAGMENT_SHADER);

	const char *fragmentShaderSourceCode_RRJ = 
		"#version 450 core" \
		"\n" \
		"in vec4 outColor;" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
			"FragColor = outColor;" \
		"}";

	glShaderSource(fragmentShaderObject_RRJ, 1,
		(const char**)&fragmentShaderSourceCode_RRJ, NULL);

	glCompileShader(fragmentShaderObject_RRJ);

	iShaderCompileStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(fragmentShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if(iShaderCompileStatus_RRJ == GL_FALSE){
		glGetShaderiv(fragmentShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if(iInfoLogLength_RRJ > 0){
			szInfoLog_RRJ =(char*)malloc(sizeof(char) * iInfoLogLength_RRJ); 
			if(szInfoLog_RRJ){
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Fragment Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(1);

			}
			fprintf(gbFile_RRJ, "FS: Memory Allocation Error szInfoLog_RRJ's Malloc!!\n");
			uninitialize();
			exit(0);
		}
	}


	/********** Program Object **********/
	gShaderProgramObject_RRJ = glCreateProgram();

	glAttachShader(gShaderProgramObject_RRJ, vertexShaderObject_RRJ);
	glAttachShader(gShaderProgramObject_RRJ, fragmentShaderObject_RRJ);

	glBindAttribLocation(gShaderProgramObject_RRJ, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(gShaderProgramObject_RRJ, AMC_ATTRIBUTE_COLOR, "vColor");

	glLinkProgram(gShaderProgramObject_RRJ);

	int iProgramLinkStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetProgramiv(gShaderProgramObject_RRJ, GL_LINK_STATUS, &iProgramLinkStatus_RRJ);
	if(iProgramLinkStatus_RRJ == GL_FALSE){
		glGetProgramiv(gShaderProgramObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if(iInfoLogLength_RRJ > 0){
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if(szInfoLog_RRJ){
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Shader Program Link Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(1);
			}
			fprintf(gbFile_RRJ, "SP: Malloc error: %s\n", szInfoLog_RRJ);
			uninitialize();
			exit(1);
		}
	}


	mvpUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_mvp_matrix");

	/********** Positions **********/
	GLfloat Tri_Vertices_RRJ[] = {
		0.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f
	};

	GLfloat Tri_Color_RRJ[] = {
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f,
	};




	/********** Triangle Vao **********/
	glGenVertexArrays(1, &vao_Tri_RRJ);
	glBindVertexArray(vao_Tri_RRJ);

		/********** Position **********/
		glGenBuffers(1, &vbo_Tri_Position_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Tri_Position_RRJ);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(Tri_Vertices_RRJ),
			Tri_Vertices_RRJ,
			GL_STATIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
			3,
			GL_FLOAT,
			GL_FALSE,
			0, NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/******************** Color *******************/
		glGenBuffers(1, &vbo_Tri_Color_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Tri_Color_RRJ);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(Tri_Color_RRJ),
			Tri_Color_RRJ,
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

	glDisable(GL_CULL_FACE);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	//perspectiveProjectionMatrix_RRJ = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
}

void uninitialize(void){
	
	GLXContext currentContext_RRJ = glXGetCurrentContext();

	if (vbo_Tri_Color_RRJ) {
		glDeleteBuffers(1, &vbo_Tri_Color_RRJ);
		vbo_Tri_Color_RRJ = 0;
	}

	if (vbo_Tri_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Tri_Position_RRJ);
		vbo_Tri_Position_RRJ = 0;
	}

	if (vao_Tri_RRJ) {
		glDeleteVertexArrays(1, &vao_Tri_RRJ);
		vao_Tri_RRJ = 0;
	}

	if(gShaderProgramObject_RRJ){
	
		glUseProgram(gShaderProgramObject_RRJ);

			int iShaderNo_RRJ;
			int iShaderCount_RRJ;	
			GLuint *pShaders_RRJ = NULL;

			glGetProgramiv(gShaderProgramObject_RRJ, GL_ATTACHED_SHADERS, &iShaderCount_RRJ);
			pShaders_RRJ = (GLuint*)malloc(sizeof(GLuint) * iShaderCount_RRJ);
			if(pShaders_RRJ){
				glGetAttachedShaders(gShaderProgramObject_RRJ, iShaderCount_RRJ, &iShaderCount_RRJ, pShaders_RRJ);

				for(iShaderNo_RRJ = 0; iShaderNo_RRJ < iShaderCount_RRJ; iShaderNo_RRJ++){
					glDetachShader(gShaderProgramObject_RRJ, pShaders_RRJ[iShaderNo_RRJ]);
					glDeleteShader(pShaders_RRJ[iShaderNo_RRJ]);
					pShaders_RRJ[iShaderNo_RRJ] = 0;
				}
				free(pShaders_RRJ);
				pShaders_RRJ = 0;
			}


		glUseProgram(0);
		glDeleteProgram(gShaderProgramObject_RRJ);
		gShaderProgramObject_RRJ = 0;

	}


	if(currentContext_RRJ != NULL && currentContext_RRJ == gGLXContext_RRJ)
		glXMakeCurrent(gpDisplay_RRJ, 0, 0);

	if(gGLXContext_RRJ)
		glXDestroyContext(gpDisplay_RRJ, gGLXContext_RRJ);

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

void resize(int width, int height){
	
	if(height == 0)
		height = 1;

	viewPortWidth_RRJ = (GLsizei)width;
	viewPortHeight_RRJ = (GLsizei)height;

	
	gPerspectiveProjectionMatrix_RRJ = mat4::identity();
	gPerspectiveProjectionMatrix_RRJ = perspective(45.0f,
						(GLfloat)width / (GLfloat)height,
						0.1f,
						100.0f);
}

void display(void) {


	mat4 translateMatrix_RRJ;
	mat4 scaleMatrix_RRJ;
	mat4 rotateMatrix_RRJ;
	mat4 modelViewMatrix_RRJ;
	mat4 modelViewProjectionMatrix_RRJ;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	glUseProgram(gShaderProgramObject_RRJ);

	/********** Triangle **********/
	translateMatrix_RRJ = mat4::identity();
	modelViewMatrix_RRJ = mat4::identity();
	modelViewProjectionMatrix_RRJ = mat4::identity();

	translateMatrix_RRJ = translate(0.0f, 0.0f, -4.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
	modelViewProjectionMatrix_RRJ = gPerspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;
	glUniformMatrix4fv(mvpUniform_RRJ,
		1,
		GL_FALSE,
		modelViewProjectionMatrix_RRJ);


	if (iViewPort_RRJ == 0)
		glViewport(0, 0, (GLsizei)viewPortWidth_RRJ, (GLsizei)viewPortHeight_RRJ);
	else if (iViewPort_RRJ == 1)
		glViewport(0, 0, ((GLsizei)viewPortWidth_RRJ) / (GLsizei)2, ((GLsizei)viewPortHeight_RRJ) / (GLsizei)2);
	else if (iViewPort_RRJ == 2)
		glViewport((GLsizei)viewPortWidth_RRJ / 2, 0, (GLsizei)viewPortWidth_RRJ / 2, (GLsizei)viewPortHeight_RRJ / 2);
	else if (iViewPort_RRJ == 3)
		glViewport((GLsizei)viewPortWidth_RRJ / 2, (GLsizei)viewPortHeight_RRJ / 2, (GLsizei)viewPortWidth_RRJ / 2, (GLsizei)viewPortHeight_RRJ / 2);
	else if (iViewPort_RRJ == 4)
		glViewport(0, (GLsizei)viewPortHeight_RRJ / 2, (GLsizei)viewPortWidth_RRJ / 2, (GLsizei)viewPortHeight_RRJ / 2);
	else if (iViewPort_RRJ == 5)
		glViewport(0, 0, (GLsizei)viewPortWidth_RRJ / 2, (GLsizei)viewPortHeight_RRJ);
	else if (iViewPort_RRJ == 6)
		glViewport((GLsizei)viewPortWidth_RRJ / 2, 0, (GLsizei)viewPortWidth_RRJ / 2, (GLsizei)viewPortHeight_RRJ);
	else if (iViewPort_RRJ == 7)
		glViewport(0, (GLsizei)viewPortHeight_RRJ / 2, (GLsizei)viewPortWidth_RRJ, (GLsizei)viewPortHeight_RRJ / 2);
	else if (iViewPort_RRJ == 8)
		glViewport(0, 0, (GLsizei)viewPortWidth_RRJ, (GLsizei)viewPortHeight_RRJ / 2);
	else if (iViewPort_RRJ == 9)
		glViewport((GLsizei)viewPortWidth_RRJ / 4, (GLsizei)viewPortHeight_RRJ / 4, (GLsizei)viewPortWidth_RRJ / 2, (GLsizei)viewPortHeight_RRJ / 2);


	glBindVertexArray(vao_Tri_RRJ);
	glDrawArrays(GL_TRIANGLES, 0, 3);

	glBindVertexArray(0);

	glUseProgram(0);

	glXSwapBuffers(gpDisplay_RRJ, gWindow_RRJ);
}





