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

//For OpenGL;
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
glXCreateContextAttribsARBProc glXCreateContextAttribsARB = NULL;
GLXFBConfig gGLXFBConfig;
GLXContext gGLXContext;

//For Uniform
GLuint mvpUniform;
GLuint samplerUniform;

//For Matrix
mat4 perspectiveProjectionMatrix;

//For ProgramObject
GLuint gShaderProgramObject;

//For Rectangle
GLuint vao_Rect;
GLuint vbo_Rect_Position;
GLuint vbo_Rect_TexCoord;

//For CheckerBoard
const int CHECK_IMAGE_HEIGHT = 64;
const int CHECK_IMAGE_WIDTH = 64;
GLubyte CheckImageData[CHECK_IMAGE_HEIGHT][CHECK_IMAGE_WIDTH][4];
GLuint texImage;

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
	if(gbFile == NULL){
		printf("Log Creation Failed!!\n");
		uninitialize();
		exit(1);
	}
	else
		fprintf(gbFile, "Log Created!!\n");


	CreateWindow();
	initialize();
	ToggleFullScreen();

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
	int iNumberOfFBConfig = 0;

	gpDisplay = XOpenDisplay(NULL);
	if(gpDisplay == NULL){
		fprintf(gbFile, "XOpenDisplay() Failed!!\n");
		uninitialize();
		exit(1);
	}

	defaultScreen = XDefaultScreen(gpDisplay);

	pGLXFBConfig = glXChooseFBConfig(gpDisplay, defaultScreen, frameBufferAttribs, &iNumberOfFBConfig);
	fprintf(gbFile, "Match: %d\n", iNumberOfFBConfig);

	int bestFrameBufferConfig = -1;
	int bestNoOfSamples = -1;
	int worstFrameBufferConfig = -1;
	int worstNoOfSamples = -1;

	for(int i = 0; i < iNumberOfFBConfig; i++){
		pTempXVisualInfo = glXGetVisualFromFBConfig(gpDisplay, pGLXFBConfig[i]);

		if(pTempXVisualInfo){
			GLsizei samples, sampleBuffers;

			glXGetFBConfigAttrib(gpDisplay, pGLXFBConfig[i], GLX_SAMPLES, &samples);
			glXGetFBConfigAttrib(gpDisplay, pGLXFBConfig[i], GLX_SAMPLE_BUFFERS, &sampleBuffers);

			if(bestFrameBufferConfig < 0 || sampleBuffers && samples > bestNoOfSamples){
				bestFrameBufferConfig = i;
				bestNoOfSamples = samples;
			}

			if(worstFrameBufferConfig < 0 || sampleBuffers && samples < worstNoOfSamples){
				worstFrameBufferConfig = i;
				worstNoOfSamples = samples;
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

	XStoreName(gpDisplay, gWindow, "10-CheckerBoard");

	Atom windowManagerDelete = XInternAtom(gpDisplay, "WM_DELETE_WINDOW", True);
	XSetWMProtocols(gpDisplay, gWindow, &windowManagerDelete, 1);

	XMapWindow(gpDisplay, gWindow);
}

void ToggleFullScreen(void){
	
	XEvent xev;
	Atom wm_state;
	Atom fullscreen;

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
	GLuint LoadTexture(void);


	GLuint vertexShaderObject;
	GLuint fragmentShaderObject;
	GLenum Result;


	glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)glXGetProcAddressARB((GLubyte*)"glXCreateContextAttribsARB");
	if(glXCreateContextAttribsARB == NULL){
		fprintf(gbFile, "glXGetProcAddress() Failed!!\n");
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
		fprintf(gbFile, "Getting Best From System\n");

		const int Attribs[] = {
			GLX_CONTEXT_MAJOR_VERSION_ARB, 1,
			GLX_CONTEXT_MINOR_VERSION_ARB, 0,
			None
		};

		gGLXContext = glXCreateContextAttribsARB(gpDisplay, gGLXFBConfig, NULL, True, Attribs);
	}

	if(!glXIsDirect(gpDisplay, gGLXContext))
		fprintf(gbFile, "S/W Context\n");
	else
		fprintf(gbFile, "H/W Context !!\n");

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
		"in vec2 vTexCoord;" \
		"out vec2 outTexCoord;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void)" \ 
		"{" \
			"gl_Position = u_mvp_matrix * vPosition;" \
			"outTexCoord = vTexCoord;" \
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
		"in vec2 outTexCoord;" \
		"out vec4 FragColor;" \
		"uniform sampler2D u_sampler;" \ 
		"void main(void)" \ 
		"{" \ 
			"FragColor = texture(u_sampler, outTexCoord);" \
		"}";

	glShaderSource(fragmentShaderObject, 1,
		(const char**)&fragmentShaderSourceCode, NULL);

	glCompileShader(fragmentShaderObject);

	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(fragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if(fragmentShaderObject == GL_FALSE){
		glGetShaderiv(fragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if(iInfoLogLength > 0){
			szInfoLog = (char*)malloc(sizeof(char) * iInfoLogLength);
			if(szInfoLog){
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gbFile, "Fragment Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
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
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_TEXCOORD0, "vTexCoord");

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
				fprintf(gbFile, "Shader Program Link Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				exit(1);
			}
		}
	}


	mvpUniform = glGetUniformLocation(gShaderProgramObject, "u_mvp_matrix");
	samplerUniform = glGetUniformLocation(gShaderProgramObject, "u_sampler");


	/********** TexCoord **********/
	GLfloat Rect_TexCoord[] = {
		0.0f, 0.0f,
		0.0f, 1.0f,
		1.0f, 1.0f,
		1.0f, 0.0f,
		None
	};

	GLfloat Rect_Pos[] = {
		-1.0f, -1.0f, 0.0f,
		-1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		None
	};


	/********** Vao Rectangle **********/
	glGenVertexArrays(1, &vao_Rect);
	glBindVertexArray(vao_Rect);

		/********** Position **********/
		glGenBuffers(1, &vbo_Rect_Position);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Position);
		glBufferData(GL_ARRAY_BUFFER, 
				3 * 4 * sizeof(GLfloat),
				NULL,
				GL_DYNAMIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
				3,
				GL_FLOAT,
				GL_FALSE,
				0, NULL);
		
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

		glBindBuffer(GL_ARRAY_BUFFER, 0);



		/********** Texture **********/
		glGenBuffers(1, &vbo_Rect_TexCoord);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_TexCoord);
		glBufferData(GL_ARRAY_BUFFER,
				sizeof(Rect_TexCoord),
				Rect_TexCoord,
				GL_STATIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOORD0,
				2,
				GL_FLOAT,
				GL_FALSE,
				0, NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOORD0);

		glBindBuffer(GL_ARRAY_BUFFER, 0);


	glBindVertexArray(0);



	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	glEnable(GL_TEXTURE_2D);
	texImage = LoadTexture();
	fprintf(gbFile, "texImage: %d\n", texImage);

	perspectiveProjectionMatrix = mat4::identity();

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	resize(WIN_WIDTH, WIN_HEIGHT);
}

void uninitialize(void){
	
	GLXContext currentContext = glXGetCurrentContext();

	GLint iShaderNo;
	GLint iShaderCount;

	if(vbo_Rect_TexCoord){
		glDeleteBuffers(1, &vbo_Rect_TexCoord);
		vbo_Rect_TexCoord = 0;	
	}

	if(vbo_Rect_Position){
		glDeleteBuffers(1, &vbo_Rect_Position);
		vbo_Rect_Position = 0;
	}

	if(vao_Rect){
		glDeleteVertexArrays(1, &vao_Rect);
		vao_Rect = 0;
	}

	if(gShaderProgramObject){
		
		glUseProgram(gShaderProgramObject);

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
			}
			free(pShaders);

		glUseProgram(0);
		glDeleteProgram(gShaderProgramObject);
		gShaderProgramObject = 0;
	}

	if(currentContext != NULL && currentContext == gGLXContext){
		glXMakeCurrent(gpDisplay, 0, 0);
	}

	if(gGLXContext){
		glXDestroyContext(gpDisplay, gGLXContext);
		gGLXContext = 0;
	}

	if(gWindow){
		XDestroyWindow(gpDisplay, gWindow);
		gWindow = 0;
	}

	if(gpXVisualInfo){
		XFree(gpXVisualInfo);
		gpXVisualInfo = NULL;
	}

	if(gGLXFBConfig){
		//XFree(&gGLXFBConfig);
		gGLXFBConfig = 0;
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

GLuint LoadTexture(void){
	
	GLuint texture;
	void MakeCheckImage(void);


	MakeCheckImage();


	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

	glGenBuffers(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D,
		      	0,	
			GL_RGBA,
			CHECK_IMAGE_WIDTH, CHECK_IMAGE_HEIGHT, 0,
			GL_RGBA,
			GL_UNSIGNED_BYTE, 
			CheckImageData);


	glBindTexture(GL_TEXTURE_2D, 0);

	return(texture);
}



void MakeCheckImage(void){
	
	GLint c;

	for(int i = 0; i < CHECK_IMAGE_HEIGHT; i++){
		for(int j = 0; j < CHECK_IMAGE_WIDTH; j++){

			c = ((( i & 0x8 ) == 0) ^ ((j & 0x8) == 0) )* 255;
			CheckImageData[i][j][0] = (GLubyte)c;
			CheckImageData[i][j][1] = (GLubyte)c;
			CheckImageData[i][j][2] = (GLubyte)c;
			CheckImageData[i][j][3] = (GLubyte)255;
		}
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
	mat4 modelViewMatrix;
	mat4 modelViewProjectionMatrix;
	
	GLfloat CheckerBoard_Position[3 * 4];

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glUseProgram(gShaderProgramObject);

		
		for(int i = 1; i <= 2; i++){
			if(i == 1){
			
				CheckerBoard_Position[0] = -2.0f;		
				CheckerBoard_Position[1] = -1.0f;		
				CheckerBoard_Position[2] = 0.0f;				

				CheckerBoard_Position[3] = -2.0f; 
				CheckerBoard_Position[4] = 1.0f; 
				CheckerBoard_Position[5] = 0.0f; 

				CheckerBoard_Position[6] = 0.0f;
				CheckerBoard_Position[7] = 1.0f;
				CheckerBoard_Position[8] = 0.0f;		

				CheckerBoard_Position[9] = 0.0f;
				CheckerBoard_Position[10] = -1.0f;
				CheckerBoard_Position[11] = 0.0f;

		
			}
			else if(i == 2){

				CheckerBoard_Position[0] = 1.0f;			
				CheckerBoard_Position[1] = -1.0f;			
				CheckerBoard_Position[2] = 0.0f;				

				CheckerBoard_Position[3] = 1.0f; 
				CheckerBoard_Position[4] = 1.0f; 
				CheckerBoard_Position[5] = 0.0f; 

				CheckerBoard_Position[6] = 2.41421f;
				CheckerBoard_Position[7] = 1.0f;
				CheckerBoard_Position[8] = -1.41421f;	

				CheckerBoard_Position[9] = 2.41421f;
				CheckerBoard_Position[10] = -1.0f;
				CheckerBoard_Position[11] = -1.41421f;
			}



			translateMatrix = mat4::identity();
			modelViewMatrix = mat4::identity();
			modelViewProjectionMatrix = mat4::identity();

			translateMatrix = translate(0.0f, 0.0f, -3.0f);
			modelViewMatrix = modelViewMatrix * translateMatrix;
			modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;
	
			glUniformMatrix4fv(mvpUniform,
					1,
					GL_FALSE,
					modelViewProjectionMatrix);	

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, texImage);
			glUniform1i(samplerUniform, 0);

			glBindVertexArray(vao_Rect);
		
				glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Position);
	
					glBufferData(GL_ARRAY_BUFFER,
							sizeof(CheckerBoard_Position),
							CheckerBoard_Position,
							GL_DYNAMIC_DRAW);

					glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

				glBindBuffer(GL_ARRAY_BUFFER, 0);	

			glBindVertexArray(0);
		}		


			/*translateMatrix = mat4::identity();
			modelViewMatrix = mat4::identity();
			modelViewProjectionMatrix = mat4::identity();

			translateMatrix = translate(0.0f, 0.0f, -3.0f);
			modelViewMatrix = modelViewMatrix * translateMatrix;
			modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;
	
			glUniformMatrix4fv(mvpUniform,
					1,
					GL_FALSE,
					modelViewProjectionMatrix);	

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, texImage);
			glUniform1i(samplerUniform, 0);

			glBindVertexArray(vao_Rect);
		
				//glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Position);
	
					//glBufferData(GL_ARRAY_BUFFER,
							//sizeof(CheckerBoard_Position),
							//CheckerBoard_Position,
							//GL_DYNAMIC_DRAW);

					glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

				//glBindBuffer(GL_ARRAY_BUFFER, 0);	

			glBindVertexArray(0);*/
		



	glUseProgram(0);

	glXSwapBuffers(gpDisplay, gWindow);

}



