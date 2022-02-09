//headers
#include <iostream>
#include <stdio.h> //for printf()
#include <stdlib.h> //for exit()
#include <memory.h> //for memset()

//headers for XServer
#include <X11/Xlib.h> //analogous to windows.h
#include <X11/Xutil.h> //for visuals
#include <X11/XKBlib.h> //XkbKeycodeToKeysym()
#include <X11/keysym.h> //for 'Keysym'

#include <GL/glew.h> //for programable
#include <GL/gl.h>
#include <GL/glx.h> //for 'glx' functions

#include <SOIL/SOIL.h>

#include "../../include/vmath.h"

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

using namespace vmath;

#define WIN_TITLE	"PP: 3D Texture"

//global variable declarations
FILE *g_fpLogFile = NULL;

Display *gpDisplay=NULL;
XVisualInfo *gpXVisualInfo=NULL;
Colormap gColormap;
Window gWindow;
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
glXCreateContextAttribsARBProc glXCreateContextAttribsARB=NULL;
GLXFBConfig gGLXFBConfig;
GLXContext gGLXContext; //parallel to HGLRC

bool gbFullscreen = false;

GLfloat g_glfAnglePyramid = 0.0f;
GLfloat g_glfAngleCube = 0.0f;

enum
{
	RTR_ATTRIBUTE_POSITION = 0,
	RTR_ATTRIBUTE_COLOR,
	RTR_ATTRIBUTE_NORMAL,
	RTR_ATTRIBUTE_TEXTURE0
};

GLuint g_gluiShaderObjectVertex;
GLuint g_gluiShaderObjectFragment;
GLuint g_gluiShaderObjectProgram;

GLuint g_gluiVAOPyramid;
GLuint g_gluiVAOCube;
GLuint g_gluiVBOPosition;
GLuint g_gluiVBOTexture;

GLint g_gliMVPUniform;
GLint g_gliTextureSamplerUniform;

GLuint g_gluiTextureStone;
GLuint g_gluiTextureKundali;

mat4 g_matPerspectiveProjection;

//entry-point function
//int main(int argc, char *argv[])
int main()
{
	//function prototype
	void CreateWindow(void);
	void ToggleFullscreen(void);
	void initialize(void);
	void resize(int,int);
	void display(void);
	void update(void);
	void uninitialize(void);
	
	//code
	// create log file
	g_fpLogFile=fopen("Log.txt", "w");
	if (g_fpLogFile==NULL)
	{
		printf("Log File Can Not Be Created. EXitting Now ...\n");
		exit(0);
	}
	else
	{
		fprintf(g_fpLogFile, "Log File Is Successfully Opened.\n");
	}
	
	// create the window
	CreateWindow();
	
	//initialize()
	initialize();
	
	//Message Loop

	//variable declarations
	XEvent event; //parallel to 'MSG' structure
	KeySym keySym;
	int winWidth;
	int winHeight;
	bool bDone=false;
	
	while(bDone==false)
	{
		while(XPending(gpDisplay))
		{
			XNextEvent(gpDisplay,&event); //parallel to GetMessage()
			switch(event.type) //parallel to 'iMsg'
			{
				case MapNotify: //parallel to WM_CREATE
					break;
				case KeyPress: //parallel to WM_KEYDOWN
					keySym=XkbKeycodeToKeysym(gpDisplay,event.xkey.keycode,0,0);
					switch(keySym)
					{
						case XK_Escape:
							bDone=true;
							break;
						case XK_F:
						case XK_f:
							if(gbFullscreen==false)
							{
								ToggleFullscreen();
								gbFullscreen=true;
							}
							else
							{
								ToggleFullscreen();
								gbFullscreen=false;
							}
							break;
						default:
							break;
					}
					break;
				case ButtonPress:
					switch(event.xbutton.button)
					{
						case 1: //Left Button
							break;
						case 2: //Middle Button
							break;
						case 3: //Right Button
							break;
						default: 
							break;
					}
					break;
				case MotionNotify: //parallel to WM_MOUSEMOVE
					break;
				case ConfigureNotify: //parallel to WM_SIZE
					winWidth=event.xconfigure.width;
					winHeight=event.xconfigure.height;
					resize(winWidth,winHeight);
					break;
				case Expose: //parallel to WM_PAINT
					break;
				case DestroyNotify:
					break;
				case 33: //close button, system menu -> close
					bDone=true;
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
	//function prototype
	void uninitialize(void);
	
	//variable declarations
	XSetWindowAttributes winAttribs;
	GLXFBConfig *pGLXFBConfigs=NULL;
	GLXFBConfig bestGLXFBConfig;
	XVisualInfo *pTempXVisualInfo=NULL;
	int iNumFBConfigs=0;
	int styleMask;
	int i;
	
	static int frameBufferAttributes[]={
		GLX_X_RENDERABLE,True,
		GLX_DRAWABLE_TYPE,GLX_WINDOW_BIT,
		GLX_RENDER_TYPE,GLX_RGBA_BIT,
		GLX_X_VISUAL_TYPE,GLX_TRUE_COLOR,
		GLX_RED_SIZE,8,
		GLX_GREEN_SIZE,8,
		GLX_BLUE_SIZE,8,
		GLX_ALPHA_SIZE,8,
		GLX_DEPTH_SIZE,24,
		GLX_STENCIL_SIZE,8,
		GLX_DOUBLEBUFFER,True,
		GLX_SAMPLE_BUFFERS,1,
		GLX_SAMPLES,4,
		None}; // array must be terminated by 0
	
	printf("==>CreateWindow\n");	
	
	//code
	gpDisplay=XOpenDisplay(NULL);
	if(gpDisplay==NULL)
	{
		printf("ERROR : Unable To Obtain X Display.\n");
		uninitialize();
		exit(1);
	}
	
	// get a new framebuffer config that meets our attrib requirements
	pGLXFBConfigs=glXChooseFBConfig(gpDisplay,DefaultScreen(gpDisplay),frameBufferAttributes,&iNumFBConfigs);
	if(pGLXFBConfigs==NULL)
	{
		printf( "Failed To Get Valid Framebuffer Config. Exitting Now ...\n");
		uninitialize();
		exit(1);
	}
	printf("%d Matching FB Configs Found.\n",iNumFBConfigs);
	
	// pick that FB config/visual with the most samples per pixel
	int bestFramebufferconfig=-1,worstFramebufferConfig=-1,bestNumberOfSamples=-1,worstNumberOfSamples=999;
	for(i=0;i<iNumFBConfigs;i++)
	{
		pTempXVisualInfo=glXGetVisualFromFBConfig(gpDisplay,pGLXFBConfigs[i]);
		if(pTempXVisualInfo)
		{
			int sampleBuffer,samples;
			glXGetFBConfigAttrib(gpDisplay,pGLXFBConfigs[i],GLX_SAMPLE_BUFFERS,&sampleBuffer);
			glXGetFBConfigAttrib(gpDisplay,pGLXFBConfigs[i],GLX_SAMPLES,&samples);
			printf("Matching Framebuffer Config=%d : Visual ID=0x%lu : SAMPLE_BUFFERS=%d : SAMPLES=%d\n",i,pTempXVisualInfo->visualid,sampleBuffer,samples);
			if(bestFramebufferconfig < 0 || sampleBuffer && samples > bestNumberOfSamples)
			{
				bestFramebufferconfig=i;
				bestNumberOfSamples=samples;
			}
			if( worstFramebufferConfig < 0 || !sampleBuffer || samples < worstNumberOfSamples)
			{
				worstFramebufferConfig=i;
			    worstNumberOfSamples=samples;
			}
		}
		XFree(pTempXVisualInfo);
	}
	bestGLXFBConfig = pGLXFBConfigs[bestFramebufferconfig];
	// set global GLXFBConfig
	gGLXFBConfig=bestGLXFBConfig;
	
	// be sure to free FBConfig list allocated by glXChooseFBConfig()
	XFree(pGLXFBConfigs);
	
	gpXVisualInfo=glXGetVisualFromFBConfig(gpDisplay,bestGLXFBConfig);
	printf("Chosen Visual ID=0x%lu\n",gpXVisualInfo->visualid );
	
	//setting window's attributes
	winAttribs.border_pixel=0;
	winAttribs.background_pixmap=0;
	winAttribs.colormap=XCreateColormap(gpDisplay,
										RootWindow(gpDisplay,gpXVisualInfo->screen), //you can give defaultScreen as well
										gpXVisualInfo->visual,
										AllocNone); //for 'movable' memory allocation
										
	winAttribs.event_mask=StructureNotifyMask | KeyPressMask | ButtonPressMask |
						  ExposureMask | VisibilityChangeMask | PointerMotionMask;
	
	styleMask=CWBorderPixel | CWEventMask | CWColormap;
	gColormap=winAttribs.colormap;										           
	
	gWindow=XCreateWindow(gpDisplay,
						  RootWindow(gpDisplay,gpXVisualInfo->screen),
						  0,
						  0,
						  WIN_WIDTH,
						  WIN_HEIGHT,
						  0, //border width
						  gpXVisualInfo->depth, //depth of visual (depth for Colormap)          
						  InputOutput, //class(type) of your window
						  gpXVisualInfo->visual,
						  styleMask,
						  &winAttribs);
	if(!gWindow)
	{
		printf("Failure In Window Creation.\n");
		uninitialize();
		exit(1);
	}
	
	XStoreName(gpDisplay,gWindow,WIN_TITLE);
	
	Atom windowManagerDelete=XInternAtom(gpDisplay,"WM_WINDOW_DELETE",True);
	XSetWMProtocols(gpDisplay,gWindow,&windowManagerDelete,1);
	
	XMapWindow(gpDisplay,gWindow);
	
	printf("<==CreateWindow\n");	
}

void ToggleFullscreen(void)
{
	//code
	Atom wm_state=XInternAtom(gpDisplay,"_NET_WM_STATE",False); //normal window state
	
	XEvent event;
	memset(&event,0,sizeof(XEvent));
	
	event.type=ClientMessage;
	event.xclient.window=gWindow;
	event.xclient.message_type=wm_state;
	event.xclient.format=32; //32-bit
	event.xclient.data.l[0]=gbFullscreen ? 0 : 1;

	Atom fullscreen=XInternAtom(gpDisplay,"_NET_WM_STATE_FULLSCREEN",False);	
	event.xclient.data.l[1]=fullscreen;
	
	//parallel to SendMessage()
	XSendEvent(gpDisplay,
			   RootWindow(gpDisplay,gpXVisualInfo->screen),
			   False, //do not send this message to Sibling windows
			   StructureNotifyMask, //resizing mask (event_mask)
			   &event);	
}

void initialize(void)
{
	// function declarations
	void uninitialize(void);
	void resize(int,int);
	void LoadGLTextures(GLuint *texture, const char *path);
	int iMajor;
	int iMinor;
	int iNumOfSupportedExtension;
	int i;
	
	printf("==>Initialize\n");	
	
	//code
	// create a new GL context 4.5 for rendering
	glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)glXGetProcAddressARB((GLubyte *)"glXCreateContextAttribsARB");
	
	iMajor = 4;
	iMinor = 3;
	GLint attribs[] = {
		GLX_CONTEXT_MAJOR_VERSION_ARB,iMajor,
		GLX_CONTEXT_MINOR_VERSION_ARB,iMinor,
		//GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB,
		GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
		0 }; // array must be terminated by 0
		
	gGLXContext = glXCreateContextAttribsARB(gpDisplay,gGLXFBConfig,0,True,attribs);

	if(!gGLXContext) // fallback to safe old style 2.x context
	{
		// When a context version below 3.0 is requested, implementations will return 
		// the newest context version compatible with OpenGL versions less than version 3.0.
		GLint attribs[] = {
			GLX_CONTEXT_MAJOR_VERSION_ARB,1,
			GLX_CONTEXT_MINOR_VERSION_ARB,0,
			0 }; // array must be terminated by 0
		printf("Failed To Create GLX 4.5 context. Hence Using Old-Style GLX Context\n");
		gGLXContext = glXCreateContextAttribsARB(gpDisplay,gGLXFBConfig,0,True,attribs);
	}
	else // successfully created 4.1 context
	{
		printf("OpenGL Context %d.%d Is Created.\n", iMajor, iMinor);
	}
	
	// verifying that context is a direct context
	if(!glXIsDirect(gpDisplay,gGLXContext))
	{
		printf("Indirect GLX Rendering Context Obtained\n");
	}
	else
	{
		printf("Direct GLX Rendering Context Obtained\n" );
	}
	
	glXMakeCurrent(gpDisplay,gWindow,gGLXContext);
	
	GLenum glewError = glewInit();
	if (GLEW_OK != glewError)
	{
		uninitialize();
		fprintf(g_fpLogFile, "glewInit() failed, Error :%d", glewError);
		return;
	}

	////////////////////////////////////////////////////////////////////
	//+	Shader code

	////////////////////////////////////////////////////////////////////
	//+	Vertex shader.

	//	Create shader.
	g_gluiShaderObjectVertex = glCreateShader(GL_VERTEX_SHADER);

	//	Provide source code.
	const GLchar *szVertexShaderSourceCode =
		"#version 430 core"							\
		"\n"										\
		"in vec4 vPosition;"						\
		"in vec2 vTexture0Coord;"					\
		"out vec2 out_texture0_coord;"				\
		"uniform mat4 u_mvp_matrix;"				\
		"void main(void)"							\
		"{"											\
		"gl_Position = u_mvp_matrix * vPosition;"	\
		"out_texture0_coord = vTexture0Coord;"		\
		"}";

	glShaderSource(g_gluiShaderObjectVertex, 1, &szVertexShaderSourceCode, NULL);

	//	Compile shader.
	glCompileShader(g_gluiShaderObjectVertex);

	GLint gliCompileStatus;
	GLint gliInfoLogLength;
	char *pszInfoLog = NULL;
	GLsizei glsiWritten;
	glGetShaderiv(g_gluiShaderObjectVertex, GL_COMPILE_STATUS, &gliCompileStatus);
	if (GL_FALSE == gliCompileStatus)
	{
		glGetShaderiv(g_gluiShaderObjectVertex, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength <= 0)
		{
			fprintf(g_fpLogFile, "GL_INFO_LOG_LENGTH is less than 0.");
			uninitialize();
			exit(0);
		}

		pszInfoLog = (char*)malloc(gliInfoLogLength);
		if (NULL == pszInfoLog)
		{
			fprintf(g_fpLogFile, "malloc failed.");
			uninitialize();
			exit(0);
		}

		glGetShaderInfoLog(g_gluiShaderObjectVertex, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//-	Vertex shader.
	////////////////////////////////////////////////////////////////////
	printf("==>Vertex shader\n");	

	////////////////////////////////////////////////////////////////////
	//+	Fragment shader.

	//	Create shader.
	g_gluiShaderObjectFragment = glCreateShader(GL_FRAGMENT_SHADER);

	//	Provide source code.
	const GLchar *szFragmentShaderSourceCode =
		"#version 430 core"							\
		"\n"										\
		"in vec2 out_texture0_coord;"				\
		"out vec4 vFragColor;"						\
		"uniform sampler2D u_texture0_sampler;"		\		
		"void main(void)"							\
		"{"											\
		"vFragColor = texture(u_texture0_sampler, out_texture0_coord);"					\
		"}";

	glShaderSource(g_gluiShaderObjectFragment, 1, &szFragmentShaderSourceCode, NULL);

	//	Compile shader.
	glCompileShader(g_gluiShaderObjectFragment);

	gliCompileStatus = 0;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;
	glsiWritten = 0;
	glGetShaderiv(g_gluiShaderObjectFragment, GL_COMPILE_STATUS, &gliCompileStatus);
	if (GL_FALSE == gliCompileStatus)
	{
		glGetShaderiv(g_gluiShaderObjectFragment, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength <= 0)
		{
			fprintf(g_fpLogFile, "Fragment : GL_INFO_LOG_LENGTH is less than 0.");
			uninitialize();
			exit(0);
		}

		pszInfoLog = (char*)malloc(gliInfoLogLength);
		if (NULL == pszInfoLog)
		{
			fprintf(g_fpLogFile, "Fragment : malloc failed.");
			uninitialize();
			exit(0);
		}

		glGetShaderInfoLog(g_gluiShaderObjectFragment, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//-	Fragment shader.
	////////////////////////////////////////////////////////////////////
	printf("==>Fragment shader\n");	
	////////////////////////////////////////////////////////////////////
	//+	Shader program.

	//	Create.
	g_gluiShaderObjectProgram = glCreateProgram();

	//	Attach vertex shader to shader program.
	glAttachShader(g_gluiShaderObjectProgram, g_gluiShaderObjectVertex);

	//	Attach Fragment shader to shader program.
	glAttachShader(g_gluiShaderObjectProgram, g_gluiShaderObjectFragment);

	//	IMP:Pre-link binding of shader program object with vertex shader position attribute.
	glBindAttribLocation(g_gluiShaderObjectProgram, RTR_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(g_gluiShaderObjectProgram, RTR_ATTRIBUTE_TEXTURE0, "vTexture0Coord");

	//	Link shader.
	glLinkProgram(g_gluiShaderObjectProgram);

	GLint gliLinkStatus;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;
	glsiWritten = 0;
	glGetShaderiv(g_gluiShaderObjectProgram, GL_LINK_STATUS, &gliLinkStatus);
	if (GL_FALSE == gliLinkStatus)
	{
		glGetProgramiv(g_gluiShaderObjectProgram, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength <= 0)
		{
			fprintf(g_fpLogFile, "Link : GL_INFO_LOG_LENGTH is less than 0.");
			uninitialize();
			exit(0);
		}

		pszInfoLog = (char*)malloc(gliInfoLogLength);
		if (NULL == pszInfoLog)
		{
			fprintf(g_fpLogFile, "Link : malloc failed.");
			uninitialize();
			exit(0);
		}

		glGetProgramInfoLog(g_gluiShaderObjectProgram, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//-	Shader program.
	////////////////////////////////////////////////////////////////////
	printf("==>shader program\n");	
	//-	Shader code
	////////////////////////////////////////////////////////////////////

	//
	//	The actual locations assigned to uniform variables are not known until the program object is linked successfully.
	//	After a program object has been linked successfully, the index values for uniform variables remain fixed until the next link command occurs.
	//
	g_gliMVPUniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_mvp_matrix");
	if (-1 == g_gliMVPUniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation failed.");
		uninitialize();
		exit(0);
	}
	
	g_gliTextureSamplerUniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_texture0_sampler");
	if (-1 == g_gliMVPUniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_texture0_sampler) failed.");
		uninitialize();
		exit(0);
	}

	////////////////////////////////////////////////////////////////////
	//+	Vertices,color, shader attribute, vbo,vao initialization.

	const GLfloat glfarrPyramidVertices[] =
	{
		//	Front face
		0.0f, 1.0f, 0.0f,	//	apex
		-1.0f, -1.0f, 1.0f,	//	left_bottom
		1.0f, -1.0f, 1.0f,	//	right_bottom
		//	Right face
		0.0f, 1.0f, 0.0f,	//	apex
		1.0f, -1.0f, 1.0f,	//	left_bottom
		1.0f, -1.0f, -1.0f,	//	right_bottom
		//	Back face
		0.0f, 1.0f, 0.0f,	//	apex
		1.0f, -1.0f, -1.0f,	//	left_bottom
		-1.0f, -1.0f, -1.0f,	//	right_bottom
		//	Left face
		0.0f, 1.0f, 0.0f,	//	apex
		-1.0f, -1.0f, -1.0f,	//	left_bottom
		-1.0f, -1.0f, 1.0f,	//	right_bottom

	};

	const GLfloat glfarrPyramidTextCoord[] =
	{
		0.5f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,

		0.5f, 1.0f,
		1.0f, 0.0f,
		0.0f, 0.0f,

		0.5f, 1.0f,
		1.0f, 0.0f,
		0.0f, 0.0f,

		0.5f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
	};
	
	
	const GLfloat glfarrCubeVertices[] =
	{
		//	Front face
		1.0f, 1.0f, 1.0f,	//	left top
		1.0f, -1.0f, 1.0f,	//	left bottom
		-1.0f, -1.0f, 1.0f,	//	Right bottom
		-1.0f, 1.0f, 1.0f,	//	Right top

		//	Right face
		1.0f, 1.0f, -1.0f,	//	left top
		1.0f, 1.0f, 1.0f,	//	left bottom
		1.0f, -1.0f, 1.0f,	//	Right bottom
		1.0f, -1.0f, -1.0f,	//	Right top

		//	Top face
		1.0f, 1.0f, -1.0f,	//	left top
		-1.0f, 1.0f, -1.0f,	//	left bottom
		-1.0f, 1.0f, 1.0f,	//	Right bottom
		1.0f, 1.0f, 1.0f,	//	Right top

		//	Front face
		1.0f, 1.0f, -1.0f,	//	left top
		1.0f, -1.0f, -1.0f,	//	left bottom
		-1.0f, -1.0f, -1.0f,	//	Right bottom
		-1.0f, 1.0f, -1.0f,	//	Right top

		//	Right face
		-1.0f, 1.0f, -1.0f,	//	left top
		-1.0f, 1.0f, 1.0f,	//	left bottom
		-1.0f, -1.0f, 1.0f,	//	Right bottom
		-1.0f, -1.0f, -1.0f,	//	Right top

		//	Top face
		1.0f, -1.0f, -1.0f,	//	left top
		-1.0f, -1.0f, -1.0f,	//	left bottom
		-1.0f, -1.0f, 1.0f,	//	Right bottom
		1.0f, -1.0f, 1.0f,	//	Right top
	};

	const GLfloat glfarrCubeTextCoord[] =
	{
		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f,

		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f,

		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f,

		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f,

		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f,

		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f,

	};

	////////////////////////////////////////////////////////////////////
	//+	Pyramid VAO

	glGenVertexArrays(1, &g_gluiVAOPyramid);
	glBindVertexArray(g_gluiVAOPyramid);

	////////////////////////////////////////////////////////////////////
	//+ Vertex position
	glGenBuffers(1, &g_gluiVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, g_gluiVBOPosition);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glfarrPyramidVertices), glfarrPyramidVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(RTR_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(RTR_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//- Vertex position
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+ Vertex Texture
	glGenBuffers(1, &g_gluiVBOTexture);
	glBindBuffer(GL_ARRAY_BUFFER, g_gluiVBOTexture);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glfarrPyramidTextCoord), glfarrPyramidTextCoord, GL_STATIC_DRAW);
	glVertexAttribPointer(RTR_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(RTR_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//- Vertex Texture
	////////////////////////////////////////////////////////////////////

	glBindVertexArray(0);

	//-	Pyramid VAO
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+	Cube VAO
	glGenVertexArrays(1, &g_gluiVAOCube);
	glBindVertexArray(g_gluiVAOCube);

	////////////////////////////////////////////////////////////////////
	//+ Vertex position
	glGenBuffers(1, &g_gluiVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, g_gluiVBOPosition);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glfarrCubeVertices), glfarrCubeVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(RTR_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(RTR_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//- Vertex position
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+ Vertex Texture
	glGenBuffers(1, &g_gluiVBOTexture);
	glBindBuffer(GL_ARRAY_BUFFER, g_gluiVBOTexture);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glfarrCubeTextCoord), glfarrCubeTextCoord, GL_STATIC_DRAW);
	glVertexAttribPointer(RTR_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(RTR_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//- Vertex Texture
	////////////////////////////////////////////////////////////////////

	glBindVertexArray(0);
	//-	Cube VAO
	////////////////////////////////////////////////////////////////////

	//-	Vertices,color, shader attribute, vbo,vao initialization.
	////////////////////////////////////////////////////////////////////	
	//code
	glShadeModel(GL_SMOOTH);
	// set-up depth buffer
	glClearDepth(1.0f);
	// enable depth testing
	glEnable(GL_DEPTH_TEST);
	// depth test to do
	glDepthFunc(GL_LEQUAL);
	// set really nice percpective calculations ?
	glHint(GL_PERSPECTIVE_CORRECTION_HINT,GL_NICEST);
	// We will always cull back faces for better performance
	//glEnable(GL_CULL_FACE);

	// set background clearing color
	glClearColor(0.0f,0.0f,0.0f,0.0f); // blue
	
	glEnable(GL_TEXTURE_2D); // enable texture mapping
	LoadGLTextures(&g_gluiTextureStone, "Stone.bmp");
	LoadGLTextures(&g_gluiTextureKundali, "Kundali.bmp");
	
	// resize
	resize(WIN_WIDTH, WIN_HEIGHT);
	printf("<==Initialize\n");	
}


void LoadGLTextures(GLuint *texture, const char *path)
{
	int width, height;
	unsigned char *imageData = NULL;

	// code
	imageData = SOIL_load_image(path, &width, &height, 0, SOIL_LOAD_RGB);
	if (imageData)
	 {
		glGenTextures(1, texture); // 1 image
		glBindTexture(GL_TEXTURE_2D ,*texture); // bind texture
		glPixelStorei(GL_UNPACK_ALIGNMENT, 4); // pixel storage mode (word aligment/4 bytes)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		// generate mipmapped texture (3 bytes, width, height & data from bmp)
		//gluBuild2DMipmaps(GL_TEXTURE_2D, 3, width, height, GL_RGB, GL_UNSIGNED_BYTE, (void *)imageData);
		glTexImage2D(
					GL_TEXTURE_2D,
					0,	//	Bitmap level of depth(0 for all)
					GL_RGB,//3,	// type of image format which is use by open GL.
					width,
					height,
					0,	//	Border width (let the implementation decide)
					GL_RGB,
					GL_UNSIGNED_BYTE,	// Type of last parameter.
					(void *)imageData
					);

		glGenerateMipmap(GL_TEXTURE_2D);

		SOIL_free_image_data(imageData); // free the imageData
	}
}


void resize(int iWidth,int iHeight)
{
    //code
	if(iHeight==0)
		iHeight=1;
		
	//	perspective(float fovy, float aspect, float n, float f)
	if (iWidth <= iHeight)
	{
		g_matPerspectiveProjection = perspective(45, (GLfloat)iHeight / (GLfloat)iWidth, 0.1f, 100.0f);
	}
	else
	{
		g_matPerspectiveProjection = perspective(45, (GLfloat)iWidth / (GLfloat)iHeight, 0.1f, 100.0f);
	}
		
	glViewport(0,0,(GLsizei)iWidth,(GLsizei)iHeight);
	
}

void display(void)
{
	mat4 matModelView;
	mat4 matModelViewProjection;

	//code
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);	//	Change 3 for 3D

	//	Start using opengl program.
	glUseProgram(g_gluiShaderObjectProgram);

	////////////////////////////////////////////////////////////////////
	//+	Draw Pyramid.

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModelView = mat4::identity();
	matModelViewProjection = mat4::identity();	//	Good practice to initialize to identity matrix though it will change in next call.

	matModelView = translate(-2.0f, 0.0f, -7.0f);
	
	matModelView = matModelView * rotate(g_glfAnglePyramid, 0.0f, 1.0f, 0.0f);
	
	//	Multiply the modelview and orthographic projection matrix to get modelviewprojection matrix.
	//	Order is very important.
	matModelViewProjection = g_matPerspectiveProjection * matModelView;

	//
	//	Pass above modelviewprojection matrix to the vertex shader in 'u_mvp_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gliMVPUniform, 1, GL_FALSE, matModelViewProjection);
	
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, g_gluiTextureStone);
	glUniform1i(g_gliTextureSamplerUniform, 0);//	0th sampler enable as we have only 1 taxture sampler in fragment shader.

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOPyramid);

	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glDrawArrays(GL_TRIANGLES, 0, 12); //	3 - each with its x,y,z vertices in triangle vertices array.

	//	Unbind 'VAO'
	glBindVertexArray(0);
	
	//-	Draw Pyramid.
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+	Draw Cube.

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModelView = mat4::identity();
	matModelViewProjection = mat4::identity();	//	Good practice to initialize to identity matrix though it will change in next call.

	matModelView = translate(2.0f, 0.0f, -7.0f);
	matModelView = matModelView * rotate(g_glfAngleCube, g_glfAngleCube, g_glfAngleCube);
	
	//	Multiply the modelview and orthographic projection matrix to get modelviewprojection matrix.
	//	Order is very important.
	matModelViewProjection = g_matPerspectiveProjection * matModelView;

	//
	//	Pass above modelviewprojection matrix to the vertex shader in 'u_mvp_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gliMVPUniform, 1, GL_FALSE, matModelViewProjection);
	
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, g_gluiTextureKundali);
	glUniform1i(g_gliTextureSamplerUniform, 0);//	0th sampler enable as we have only 1 taxture sampler in fragment shader.

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOCube);

	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 20, 4);

	//	Unbind 'VAO'
	glBindVertexArray(0);
	
	//-	Draw Triangle.
	////////////////////////////////////////////////////////////////////
	
	//	Stop using opengl program.
	glUseProgram(0);

	glXSwapBuffers(gpDisplay,gWindow);
}

void update(void)
{
	g_glfAngleCube = g_glfAngleCube + 0.5f;

	if (g_glfAngleCube >= 360)
	{
		g_glfAngleCube = 0.0f;
	}

	g_glfAnglePyramid = g_glfAnglePyramid + 0.5f;

	if (g_glfAnglePyramid >= 360)
	{
		g_glfAnglePyramid = 0.0f;
	}
}

void uninitialize(void)
{
	if (g_gluiVBOPosition)
	{
		glDeleteBuffers(1, &g_gluiVBOPosition);
		g_gluiVBOPosition = 0;
	}
	
	if (g_gluiVBOTexture)
	{
		glDeleteBuffers(1, &g_gluiVBOTexture);
		g_gluiVBOTexture = 0;
	}

	if (g_gluiVAOCube)
	{
		glDeleteVertexArrays(1, &g_gluiVAOCube);
		g_gluiVAOCube = 0;
	}

	if (g_gluiVAOPyramid)
	{
		glDeleteVertexArrays(1, &g_gluiVAOPyramid);
		g_gluiVAOPyramid = 0;
	}
	
	if (g_gluiTextureStone)
	{
		glDeleteTextures(1, &g_gluiTextureStone);
		g_gluiTextureStone = 0;
	}

	if (g_gluiTextureKundali)
	{
		glDeleteTextures(1, &g_gluiTextureKundali);
		g_gluiTextureKundali = 0;
	}
	
	if (g_gluiShaderObjectVertex)
	{
		glDetachShader(g_gluiShaderObjectProgram, g_gluiShaderObjectVertex);
		glDeleteShader(g_gluiShaderObjectVertex);
		g_gluiShaderObjectVertex = 0;
	}

	if (g_gluiShaderObjectFragment)
	{
		glDetachShader(g_gluiShaderObjectProgram, g_gluiShaderObjectFragment);
		glDeleteShader(g_gluiShaderObjectFragment);
		g_gluiShaderObjectFragment = 0;
	}

	//
	//	Unlink shader program
	//	This will be useful when detach multiple shaders in loop.
	//	1.glUseProgram(Shader_Program_Object)
	//	2.Get Attach shader list
	//	3.Detach i loop.
	//	4.glUseProgram(0)
	//
	glUseProgram(0);

	if (g_gluiShaderObjectProgram)
	{
		glDeleteProgram(g_gluiShaderObjectProgram);
		g_gluiShaderObjectProgram = 0;
	}
	
	//code
	// Releasing OpenGL related and XWindow related objects 	
	GLXContext currentContext=glXGetCurrentContext();
	if(currentContext!=NULL && currentContext==gGLXContext)
	{
		glXMakeCurrent(gpDisplay,0,0);
	}
	
	if(gGLXContext)
	{
		glXDestroyContext(gpDisplay,gGLXContext);
	}
	
	if(gWindow)
	{
		XDestroyWindow(gpDisplay,gWindow);
	}
	
	if(gColormap)
	{
		XFreeColormap(gpDisplay,gColormap);
	}
	
	if(gpXVisualInfo)
	{
		free(gpXVisualInfo);
		gpXVisualInfo=NULL;
	}
	
	if(gpDisplay)
	{
		XCloseDisplay(gpDisplay);
		gpDisplay=NULL;
	}

	if (g_fpLogFile)
	{
		fprintf(g_fpLogFile, "Log File Is Successfully Closed.\n");
		fclose(g_fpLogFile);
		g_fpLogFile = NULL;
	}
}
