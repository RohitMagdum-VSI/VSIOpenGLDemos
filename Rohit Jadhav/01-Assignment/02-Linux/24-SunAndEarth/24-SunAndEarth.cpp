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


//For Stack
struct STACK {
	mat4 ModelViewMatrix;
	struct STACK *next;
	struct STACK *prev;
};

typedef struct STACK ModelViewStack;
ModelViewStack *TopNode_RRJ = NULL;
int MaxTop_RRJ = 32;
int iTop_RRJ = -1;


//For Planet
int year_RRJ;
int day_RRJ;


//For Shader Program Object;
GLint gShaderProgramObject_RRJ;

//For Perspective Matric
mat4 gPerspectiveProjectionMatrix_RRJ;

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
GLfloat angle_Sphere = 0.0f;

//For Uniform
GLuint mvUniform_RRJ;
GLuint projectionUniform_RRJ;



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

					XLookupString(&event_RRJ.xkey,keys_RRJ, sizeof(keys_RRJ), NULL, NULL);
					switch(keys_RRJ[0]){
						case 'Y':
							year_RRJ = (year_RRJ + 3) % 360;
							break;


						case 'y':
							year_RRJ = (year_RRJ - 3) % 360;
							break;

						case 'D':
							day_RRJ = (day_RRJ + 6) % 360;
							break;

						case 'd':
							day_RRJ = (day_RRJ - 6) % 360;
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


	XStoreName(gpDisplay_RRJ, gWindow_RRJ, "Rohit_R_Jadhav-PP-XWindows-SunAndEarth");

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




	/********** Vertex Shader **********/
	iVertexShaderObject_RRJ = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *szVertexShaderSourceCode_RRJ =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vColor;" \
		"out vec3 outColor;" \
		"uniform mat4 u_mv_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"void main(void) {" \
			"outColor = vColor;" \
			"gl_Position = u_projection_matrix * u_mv_matrix * vPosition;" \
		"}";

	glShaderSource(iVertexShaderObject_RRJ, 1,
		(const GLchar**)&szVertexShaderSourceCode_RRJ, NULL);

	glCompileShader(iVertexShaderObject_RRJ);

	GLint iShaderCompileStatus_RRJ;
	GLint iInfoLogLength_RRJ;
	GLchar *szInfoLog_RRJ = NULL;
	glGetShaderiv(iVertexShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(iVertexShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(iVertexShaderObject_RRJ, iInfoLogLength_RRJ,
					&written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Vertex Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(0);
			}
		}
	}

	/********** Fragment Shader **********/
	iFragmentShaderObject_RRJ = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *szFragmentShaderSourceCode_RRJ =
		"#version 450 core" \
		"\n" \
		"in vec3 outColor;" \
		"out vec4 FragColor;" \
		"void main(void) {" \
			"FragColor = vec4(outColor, 1.0);" \
		"}";

	glShaderSource(iFragmentShaderObject_RRJ, 1,
		(const GLchar**)&szFragmentShaderSourceCode_RRJ, NULL);

	glCompileShader(iFragmentShaderObject_RRJ);

	iShaderCompileStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(iFragmentShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(iFragmentShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(iFragmentShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Fragment Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(0);
			}
		}
	}

	/********** Shader Program Object **********/
	gShaderProgramObject_RRJ = glCreateProgram();

	glAttachShader(gShaderProgramObject_RRJ, iVertexShaderObject_RRJ);
	glAttachShader(gShaderProgramObject_RRJ, iFragmentShaderObject_RRJ);

	glBindAttribLocation(gShaderProgramObject_RRJ, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(gShaderProgramObject_RRJ, AMC_ATTRIBUTE_COLOR, "vColor");
	
	glLinkProgram(gShaderProgramObject_RRJ);

	GLint iProgramLinkingStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;
	glGetProgramiv(gShaderProgramObject_RRJ, GL_LINK_STATUS, &iProgramLinkingStatus_RRJ);
	if (iProgramLinkingStatus_RRJ == GL_FALSE) {
		glGetProgramiv(gShaderProgramObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Shader Program Object Linking Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(0);
			}
		}
	}

	mvUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_mv_matrix");
	projectionUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_projection_matrix");
	


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

	
	/********** Color **********/
	glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);

	
	/********** Element Vbo **********/
	glGenBuffers(1, &vbo_Sphere_Element_RRJ);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(sphere_elements_RRJ), sphere_elements_RRJ, GL_STATIC_DRAW);
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


	glBindVertexArray(0);


	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix_RRJ = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);

}




void uninitialize(void){

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

	if (gShaderProgramObject_RRJ) {
		glUseProgram(gShaderProgramObject_RRJ);

		glGetProgramiv(gShaderProgramObject_RRJ, GL_ATTACHED_SHADERS, &ShaderCount_RRJ);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount_RRJ);
		if (pShader) {
			glGetAttachedShaders(gShaderProgramObject_RRJ, ShaderCount_RRJ, &ShaderCount_RRJ, pShader);
			for (ShaderNumber_RRJ = 0; ShaderNumber_RRJ < ShaderCount_RRJ; ShaderNumber_RRJ++) {
				glDetachShader(gShaderProgramObject_RRJ, pShader[ShaderNumber_RRJ]);
				glDeleteShader(pShader[ShaderNumber_RRJ]);
				pShader[ShaderNumber_RRJ] = 0;
			}
			free(pShader);
			pShader = NULL;
		}
		glDeleteProgram(gShaderProgramObject_RRJ);
		gShaderProgramObject_RRJ = 0;
		glUseProgram(0);
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

	gPerspectiveProjectionMatrix_RRJ = mat4::identity();
	gPerspectiveProjectionMatrix_RRJ = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}


void display(void) {

	void my_glPushMatrix(mat4);
	mat4 my_glPopMatrix();


	mat4 modelMatrix_RRJ;
	mat4 viewMatrix_RRJ;
	mat4 modelViewMatrix_RRJ;


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(gShaderProgramObject_RRJ);



	/********** Sphere **********/
	modelMatrix_RRJ = mat4::identity();
	viewMatrix_RRJ = mat4::identity();
	modelViewMatrix_RRJ = mat4::identity();


	viewMatrix_RRJ =  lookat(vec3(0.0f, 0.0f, 3.0f),
						vec3(0.0f, 0.0f, 0.0f),
						vec3(0.0f, 1.0f, 0.0));

	modelViewMatrix_RRJ = viewMatrix_RRJ * modelMatrix_RRJ;


	//Sun
	my_glPushMatrix(modelViewMatrix_RRJ);

	
	glUniformMatrix4fv(mvUniform_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
	glUniformMatrix4fv(projectionUniform_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);

	glBindVertexArray(vao_Sphere_RRJ);
	glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 0.0f);
	glDrawElements(GL_TRIANGLES, gNumElements_RRJ, GL_UNSIGNED_SHORT, 0);
	glBindVertexArray(0);

	
	//Earth
	modelViewMatrix_RRJ = my_glPopMatrix();
	
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate((GLfloat)year_RRJ, 0.0f, 1.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(1.50f, 0.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * scale(0.4f, 0.4f, 0.4f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate((GLfloat)day_RRJ, 0.0f, 1.0f, 0.0f);

	my_glPushMatrix(modelViewMatrix_RRJ);
	
	glUniformMatrix4fv(mvUniform_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
	glUniformMatrix4fv(projectionUniform_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);

	glBindVertexArray(vao_Sphere_RRJ);
	glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 0.0f, 0.0f, 0.5f);
	glDrawElements(GL_LINES, gNumElements_RRJ, GL_UNSIGNED_SHORT, 0);
	glBindVertexArray(0);

	my_glPopMatrix();


	glUseProgram(0);

	glXSwapBuffers(gpDisplay_RRJ, gWindow_RRJ);
}


void my_glPushMatrix(mat4 matrix) {	

	void uninitialize(void);

	ModelViewStack *temp_RRJ = (ModelViewStack*)malloc(sizeof(ModelViewStack));
	if (temp_RRJ == NULL) {
		fprintf(gbFile_RRJ, "ERROR: Malloc Failed!!\n");
		uninitialize();
		exit(0);
	}
	else {

		temp_RRJ->ModelViewMatrix = matrix;
		temp_RRJ->next = NULL;

		if (TopNode_RRJ == NULL) {
			TopNode_RRJ = temp_RRJ;
			TopNode_RRJ->prev = NULL;
			fprintf(gbFile_RRJ, "Node Added!!\n");
		}
		else {
			TopNode_RRJ->next = temp_RRJ;
			temp_RRJ->prev = TopNode_RRJ;
			TopNode_RRJ = temp_RRJ;
			fprintf(gbFile_RRJ, "Node Added!!\n");
		}
	}

	if (iTop_RRJ > MaxTop_RRJ) {
		fprintf(gbFile_RRJ, "ERROR: Stack Overflow!!\n");
		uninitialize();
		exit(0);
	}
	

	
}

mat4 my_glPopMatrix(void) {
	
	void uninitialize(void);

	ModelViewStack *temp_RRJ = TopNode_RRJ;
	mat4 matrix_RRJ;
	if (temp_RRJ->prev != NULL) {
		TopNode_RRJ = temp_RRJ->prev;
		temp_RRJ->next = NULL;
		temp_RRJ->prev = NULL;
		matrix_RRJ = temp_RRJ->ModelViewMatrix;
		fprintf(gbFile_RRJ, "Node Delete!!\n");
		free(temp_RRJ);
	}
	else {
		temp_RRJ->next = NULL;
		temp_RRJ->prev = NULL;
		matrix_RRJ = temp_RRJ->ModelViewMatrix;
		fprintf(gbFile_RRJ, "Node Delete!!\n");
		free(temp_RRJ);
		TopNode_RRJ = NULL;
	}
	return(matrix_RRJ);
	
}
