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


enum {
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOORD0,
};

#define WIN_WIDTH 800
#define WIN_HEIGHT 600


//For FullScreen
bool bIsFullScreen_RRJ = false;
Display *gpDisplay_RRJ = NULL;
XVisualInfo *gpXVisualInfo_RRJ = NULL;
Colormap gColormap_RRJ;
Window gWindow_RRJ;


//For Context
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
glXCreateContextAttribsARBProc glXCreateContextAttribsARB_RRJ = NULL;
GLXContext gGLXContext_RRJ;

//For FBConfig
GLXFBConfig gGLXFBConfig_RRJ;

//For Error
FILE *gbFile_RRJ = NULL;


//For Shader
GLuint vertexShaderObject_RRJ;
GLuint geometryShaderObject_RRJ;
GLuint fragmentShaderObject_RRJ;
GLuint shaderProgramObject_RRJ;

GLuint vao_Tri_RRJ;
GLuint vbo_Tri_Position_RRJ;

//For Uniform
GLuint mvpUniform_RRJ;


mat4 perspectiveProjectionMatrix_RRJ;


int main(void){
	
	void CreateWindow(void);
	void initialize(void);
	void ToggleFullScreen(void);
	void uninitialize(void);
	void display(void);
	void resize(int, int);


	int winWidth_RRJ = WIN_WIDTH;
	int winHeight_RRJ = WIN_HEIGHT;
	bool bDone_RRJ = false;

	gbFile_RRJ = fopen("Log.txt", "w");
	if(gbFile_RRJ == NULL){
		printf("ERROR: Log Creation Failed!!\n");
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: Log Created!!\n");


	CreateWindow();
	initialize();

	//Event Loop
	XEvent event_RRJ;
	KeySym keysym_RRJ;
	char keys_RRJ[26];

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
			}
		}
		display();
	}

	uninitialize();
	return(0);
}



void CreateWindow(void){

	void uninitialize(void);

	XSetWindowAttributes winAttribs_RRJ;
	int defaultScreen_RRJ;
	int styleMask_RRJ;


	GLXFBConfig *pGLXFBConfig_RRJ = NULL;
	GLXFBConfig bestFBConfig_RRJ;
	XVisualInfo *pTempXVisualInfo_RRJ = NULL;
	int iNumberOfFBConfig_RRJ = 0;


	static int frameBufferAttributes_RRJ[] = {
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
		None
	};


	gpDisplay_RRJ = XOpenDisplay(NULL);
	if(gpDisplay_RRJ == NULL){
		fprintf(gbFile_RRJ, "ERROR: XOpenDisplay() Failed!!\n");
		uninitialize();
		exit(0);	
	}


	defaultScreen_RRJ = XDefaultScreen(gpDisplay_RRJ);

	
	
	pGLXFBConfig_RRJ = glXChooseFBConfig(gpDisplay_RRJ, defaultScreen_RRJ, frameBufferAttributes_RRJ, &iNumberOfFBConfig_RRJ);
	fprintf(gbFile_RRJ, "SUCCESS: Totoal GLXFBConfig: %d\n", iNumberOfFBConfig_RRJ);

	int bestFrameBufferConfig_RRJ = -1;
	int bestNoOfSamples_RRJ = -1;
	int worstFrameBufferConfig_RRJ = -1;
	int worstNoOfSamples_RRJ = -1;

	for(int i = 0; i < iNumberOfFBConfig_RRJ; i++){
	
		pTempXVisualInfo_RRJ = glXGetVisualFromFBConfig(gpDisplay_RRJ, pGLXFBConfig_RRJ[i]);

		if(pTempXVisualInfo_RRJ){
			
			int samples, sampleBuffers;
				
			glXGetFBConfigAttrib(gpDisplay_RRJ, pGLXFBConfig_RRJ[i], GLX_SAMPLE_BUFFERS, &sampleBuffers);
			glXGetFBConfigAttrib(gpDisplay_RRJ, pGLXFBConfig_RRJ[i], GLX_SAMPLES, &samples);

			if(bestFrameBufferConfig_RRJ < 0 || sampleBuffers && samples >  bestNoOfSamples_RRJ){
				bestNoOfSamples_RRJ = samples;
				bestFrameBufferConfig_RRJ = i;
			}

			if(worstFrameBufferConfig_RRJ < 0 || sampleBuffers && samples < worstNoOfSamples_RRJ){
				worstNoOfSamples_RRJ = samples;
			       	worstFrameBufferConfig_RRJ = i;	
			}
		}

		XFree(pTempXVisualInfo_RRJ);
	}

	bestFBConfig_RRJ = pGLXFBConfig_RRJ[bestFrameBufferConfig_RRJ];

	gGLXFBConfig_RRJ = bestFBConfig_RRJ;

	XFree(pGLXFBConfig_RRJ);

	gpXVisualInfo_RRJ = glXGetVisualFromFBConfig(gpDisplay_RRJ, bestFBConfig_RRJ);


	winAttribs_RRJ.border_pixel = 0;
	winAttribs_RRJ.border_pixmap = 0;
	winAttribs_RRJ.background_pixel = BlackPixel(gpDisplay_RRJ, defaultScreen_RRJ);
	winAttribs_RRJ.background_pixmap = 0;
	winAttribs_RRJ.colormap = XCreateColormap(gpDisplay_RRJ,
					RootWindow(gpDisplay_RRJ, gpXVisualInfo_RRJ->screen),
					gpXVisualInfo_RRJ->visual,
					AllocNone);
	gColormap_RRJ = winAttribs_RRJ.colormap;
	winAttribs_RRJ.event_mask = ExposureMask | VisibilityChangeMask | ButtonPressMask |
					KeyPressMask | PointerMotionMask | StructureNotifyMask;

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
			&winAttribs_RRJ);

	if(!gWindow_RRJ){
		fprintf(gbFile_RRJ, "ERROR: XCreateWindow() Failed!!\n");
		uninitialize();
		exit(0);
	}


	XStoreName(gpDisplay_RRJ, gWindow_RRJ, "Rohit_R_Jadhav-PP-XWindows-GeometryShader");

	Atom windowManagerDelete_RRJ = XInternAtom(gpDisplay_RRJ, "WM_DELETE_WINDOW", True);
	XSetWMProtocols(gpDisplay_RRJ, gWindow_RRJ, &windowManagerDelete_RRJ, 1);

	XMapWindow(gpDisplay_RRJ, gWindow_RRJ);

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




void initialize(void){

	
	void uninitialize(void);
	void resize(int, int);

	//Shader Object;
	GLint iVertexShaderObject_RRJ;
	GLint iFragmentShaderObject_RRJ;


	glXCreateContextAttribsARB_RRJ = (glXCreateContextAttribsARBProc)glXGetProcAddressARB((GLubyte*)"glXCreateContextAttribsARB");
	if(glXCreateContextAttribsARB_RRJ == NULL){
		fprintf(gbFile_RRJ, "ERROR: GetProcAddressARB() Failed!!\n");
		uninitialize();
		exit(0);
	}


	const int attribs[] = {
		GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
		GLX_CONTEXT_MINOR_VERSION_ARB, 5,
		GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
		None
	};

	gGLXContext_RRJ = glXCreateContextAttribsARB_RRJ(gpDisplay_RRJ, gGLXFBConfig_RRJ, NULL, True, attribs);
	if(!gGLXContext_RRJ){
		const int attributes[] = {
			GLX_CONTEXT_MAJOR_VERSION_ARB, 1,
			GLX_CONTEXT_MINOR_VERSION_ARB, 0,
			None
		};

		gGLXContext_RRJ = glXCreateContextAttribsARB_RRJ(gpDisplay_RRJ, gGLXFBConfig_RRJ, NULL, True, attributes);
	}

	if(!glXIsDirect(gpDisplay_RRJ, gGLXContext_RRJ)){
		fprintf(gbFile_RRJ, "ERROR: Software Rendering Context!!\n");
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: Hardware Rendering Context!!\n");

	glXMakeCurrent(gpDisplay_RRJ, gWindow_RRJ, gGLXContext_RRJ);




	GLenum result_RRJ;
	result_RRJ = glewInit();
	if(result_RRJ != GLEW_OK){
		fprintf(gbFile_RRJ, "glewInit() Failed!!\n");
		uninitialize();
		exit(1);
	}



	/********** VERTEX SHADER **********/
	vertexShaderObject_RRJ = glCreateShader(GL_VERTEX_SHADER);
	const char *vertexShaderSourceCode_RRJ =
		"#version 450" \
		"\n" \
		"in vec4 vPosition;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void) {" \
		"gl_Position = u_mvp_matrix * vPosition;" \
		"}";

	glShaderSource(vertexShaderObject_RRJ, 1, (const char**)&vertexShaderSourceCode_RRJ, NULL);
	glCompileShader(vertexShaderObject_RRJ);

	int iInfoLogLength_RRJ;
	int iShaderCompileStatus_RRJ;
	char *szInfoLog_RRJ = NULL;

	glGetShaderiv(vertexShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(vertexShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ) {
				GLsizei written;

				glGetShaderInfoLog(vertexShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "VERTEX SHADER ERROR: \n %s", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(0);
			}
		}
	}


	/********** GEOMETRY SHADER **********/
	geometryShaderObject_RRJ = glCreateShader(GL_GEOMETRY_SHADER);
	const char *geometryShaderSourceCode_RRJ =
		"#version 450" \
		"\n" \
		"layout(triangles)in;" \
		"layout(triangle_strip, max_vertices = 9)out;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void) {" \
			"for(int vertex = 0; vertex < 3; vertex++){" \
		
				"gl_Position = u_mvp_matrix * (gl_in[vertex].gl_Position + vec4(0.0, 1.0, 0.0, 0.0));" \
				"EmitVertex();" \
					
				"gl_Position = u_mvp_matrix * (gl_in[vertex].gl_Position + vec4(-1.0, -1.0, 0.0, 0.0));" \
				"EmitVertex();" \
			
				"gl_Position = u_mvp_matrix * (gl_in[vertex].gl_Position + vec4(1.0, -1.0, 0.0, 0.0));" \
				"EmitVertex();" \

				"EndPrimitive();" \
			"}" \
		"}";


	glShaderSource(geometryShaderObject_RRJ, 1,
		(const char**)&geometryShaderSourceCode_RRJ, NULL);

	glCompileShader(geometryShaderObject_RRJ);

	iInfoLogLength_RRJ = 0;
	iShaderCompileStatus_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(geometryShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(geometryShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ) {
				GLsizei written;
				glGetShaderInfoLog(geometryShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "GEOMETRY SHADER ERROR: \n %s", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(0);
			}
		}
	}
		
	



	/********** FRAGMENT SHADER **********/
	fragmentShaderObject_RRJ = glCreateShader(GL_FRAGMENT_SHADER);
	const char *fragmentShaderSourceCode_RRJ =
		"#version 450" \
		"\n" \
		"out vec4 FragColor;" \
		"void main(void) {" \
		"FragColor = vec4(1.0, 1.0, 1.0, 1.0);" \
		"}";


	glShaderSource(fragmentShaderObject_RRJ, 1,
		(const char**)&fragmentShaderSourceCode_RRJ, NULL);

	glCompileShader(fragmentShaderObject_RRJ);

	iInfoLogLength_RRJ = 0;
	iShaderCompileStatus_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(fragmentShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(fragmentShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ) {
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "FRAGMENT SHADER ERROR: \n %s", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(0);
			}
		}
	}


	/********** SHADER PROGRAM **********/
	shaderProgramObject_RRJ = glCreateProgram();

	glAttachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
	glAttachShader(shaderProgramObject_RRJ, geometryShaderObject_RRJ);
	glAttachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);

	glBindAttribLocation(shaderProgramObject_RRJ, AMC_ATTRIBUTE_POSITION, "vPosition");

	glLinkProgram(shaderProgramObject_RRJ);

	int iProgramLinkStatus_RRJ;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;
	glGetProgramiv(shaderProgramObject_RRJ, GL_LINK_STATUS, &iProgramLinkStatus_RRJ);
	if (iProgramLinkStatus_RRJ == GL_FALSE) {
		glGetProgramiv(shaderProgramObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ) {
				GLsizei written;
				glGetProgramInfoLog(shaderProgramObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "SHADER PROGRAM ERROR: %s", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(0);
			}
		}
	}


	mvpUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_mvp_matrix");
	



	/********** LINE COORDINATES **********/
	float tri_Vertices_RRJ[] = {
		0.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
	};

	glGenVertexArrays(1, &vao_Tri_RRJ);
	glBindVertexArray(vao_Tri_RRJ);

	/********** Position **********/
	glGenBuffers(1, &vbo_Tri_Position_RRJ);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Tri_Position_RRJ);
	glBufferData(GL_ARRAY_BUFFER, sizeof(tri_Vertices_RRJ), tri_Vertices_RRJ, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);


	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	
	resize(WIN_WIDTH, WIN_HEIGHT);

}




void uninitialize(void){

	if (vbo_Tri_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Tri_Position_RRJ);
		vbo_Tri_Position_RRJ = 0;
	}

	if (vao_Tri_RRJ) {
		glDeleteVertexArrays(1, &vao_Tri_RRJ);
		vao_Tri_RRJ = 0;
	}

	if (shaderProgramObject_RRJ) {
		glUseProgram(shaderProgramObject_RRJ);

		/*GLint shaderCount_RRJ;
		GLint shaderNo_RRJ;

		glGetShaderiv(shaderProgramObject_RRJ, GL_ATTACHED_SHADERS, &shaderCount_RRJ);
		fprintf(gbFile_RRJ, "INFO: ShaderCount: %d\n", shaderCount_RRJ);
		GLuint *pShaders = (GLuint*)malloc(sizeof(GLuint*) * shaderCount_RRJ);
		if (pShaders) {
			glGetAttachedShaders(shaderProgramObject_RRJ, shaderCount_RRJ, &shaderCount_RRJ, pShaders);
			for (shaderNo_RRJ = 0; shaderNo_RRJ < shaderCount_RRJ; shaderNo_RRJ++) {
				glDetachShader(shaderProgramObject_RRJ, pShaders[shaderNo_RRJ]);
				glDeleteShader(pShaders[shaderNo_RRJ]);
				pShaders[shaderNo_RRJ] = 0;
			}
			free(pShaders);
			pShaders = NULL;
		}*/


		if (fragmentShaderObject_RRJ) {
			glDetachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);
			glDeleteShader(fragmentShaderObject_RRJ);
			fragmentShaderObject_RRJ = 0;
		}

		if (geometryShaderObject_RRJ) {
			glDetachShader(shaderProgramObject_RRJ, geometryShaderObject_RRJ);
			glDeleteShader(geometryShaderObject_RRJ);
			geometryShaderObject_RRJ = 0;
		}

		if (vertexShaderObject_RRJ) {
			glDetachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
			glDeleteShader(vertexShaderObject_RRJ);
			vertexShaderObject_RRJ = 0;
		}


		glUseProgram(0);
		glDeleteProgram(shaderProgramObject_RRJ);
		shaderProgramObject_RRJ = 0;
	}


	

	if(gWindow_RRJ){
		XDestroyWindow(gpDisplay_RRJ, gWindow_RRJ);
		gWindow_RRJ = 0;
	}

	if(gColormap_RRJ){
		XFreeColormap(gpDisplay_RRJ, gColormap_RRJ);
		gColormap_RRJ = 0;
	}

	if(gpXVisualInfo_RRJ){
		XFree(gpXVisualInfo_RRJ);
		gpXVisualInfo_RRJ = 0;
	}


	if(gpDisplay_RRJ){
		XCloseDisplay(gpDisplay_RRJ);
		gpDisplay_RRJ = NULL;
	}

	if(gbFile_RRJ){
		fprintf(gbFile_RRJ, "SUCCESS: Log Close!!\n");
		fprintf(gbFile_RRJ, "SUCCESS: End!\n");
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

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	mat4 modelViewMatrix_RRJ;
	mat4 modelViewProjectionMatrix_RRJ;


	glUseProgram(shaderProgramObject_RRJ);

	modelViewMatrix_RRJ = mat4::identity();
	modelViewProjectionMatrix_RRJ = mat4::identity();

	modelViewMatrix_RRJ = translate(0.0f, 0.00f, -4.0f);
	modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

	glUniformMatrix4fv(mvpUniform_RRJ, 1, GL_FALSE, modelViewProjectionMatrix_RRJ);
	


	glBindVertexArray(vao_Tri_RRJ);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	glBindVertexArray(0);


	glUseProgram(0);

	glXSwapBuffers(gpDisplay_RRJ, gWindow_RRJ);
}

