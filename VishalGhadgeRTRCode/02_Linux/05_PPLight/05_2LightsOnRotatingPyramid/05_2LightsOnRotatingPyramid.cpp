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

#include "vmath.h"
#define WIN_WIDTH 800
#define WIN_HEIGHT 600

using namespace vmath;

#define WIN_TITLE	"Single Light On Sphere PerVertex And Per Fragment"

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
GLuint g_gluiVBOPosition;
GLuint g_gluiVBONormal;

/////////////////////////////////////////////////////////////////
//+Uniforms.
GLuint g_gluiModelMat4Uniform;
GLuint g_gluiViewMat4Uniform;
GLuint g_gluiProjectionMat4Uniform;

GLuint g_gluiKeyPressedUniform;

GLuint g_gluiLaRVec3Uniform;	//	light ambient
GLuint g_gluiLdRVec3Uniform;	//	light diffuse
GLuint g_gluiLsRVec3Uniform;	//	light specular
GLuint g_gluiLightPositionRVec4Uniform;

GLuint g_gluiLaLVec3Uniform;	//	light ambient
GLuint g_gluiLdLVec3Uniform;	//	light diffuse
GLuint g_gluiLsLVec3Uniform;	//	light specular
GLuint g_gluiLightPositionLVec4Uniform;

GLuint g_gluiKaVec3Uniform;//	Material ambient
GLuint g_gluiKdVec3Uniform;//	Material diffuse
GLuint g_gluiKsVec3Uniform;//	Material specular
GLuint g_gluiMaterialShininessUniform;
//-Uniforms.
/////////////////////////////////////////////////////////////////

//
//	Light R == Red Light = Right Side light.
//
GLfloat g_glfarrLightRAmbient[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Decides general light
GLfloat g_glfarrLightRDiffuse[] = { 1.0f, 0.0f, 0.0f, 0.0f };	//	Decides color of light
GLfloat g_glfarrLightRSpecular[] = { 1.0f, 0.0f, 0.0f, 0.0f };	//	Decides height of light
GLfloat g_glfarrLightRPosition[] = { 2.0f, 1.0f, 1.0f, 0.0f };	//	Decides position of light

//
//	Light L == Blue Light = Left Side light.
//
GLfloat g_glfarrLightLAmbient[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Decides general light
GLfloat g_glfarrLightLDiffuse[] = { 0.0f, 0.0f, 1.0f, 0.0f };	//	Decides color of light
GLfloat g_glfarrLightLSpecular[] = { 0.0f, 0.0f, 1.0f, 0.0f };	//	Decides height of light
GLfloat g_glfarrLightLPosition[] = { -2.0f, 1.0f, 1.0f, 0.0f };	//	Decides position of light

GLfloat g_glfarrMaterialAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_glfarrMaterialDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfarrMaterialSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfMaterialShininess = 50.0f;

mat4 g_matPerspectiveProjection;

bool g_bAnimate = false;
bool g_bLight = false;
GLfloat g_glfAngle = 0.0f;

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

						case XK_a:
						case XK_A:
							if (false == g_bAnimate)
							{
								g_bAnimate = true;
							}
							else
							{
								g_bAnimate = false;
							}
							break;

						case XK_l:
						case XK_L:
							if (false == g_bLight)
							{
								g_bLight = true;
							}
							else
							{
								g_bLight = false;
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

	printf("==>Vertex Shader\n" );
	//	Create shader.
	g_gluiShaderObjectVertex = glCreateShader(GL_VERTEX_SHADER);

	//	Provide source code.
	const GLchar *szVertexShaderSourceCode =
		"#version 430 core"							\
		"\n"										\
		"in vec4 vPosition;"						\
		"in vec3 vNormal;"							\
		"uniform mat4 u_model_matrix;"	\
		"uniform mat4 u_view_matrix;"	\
		"uniform mat4 u_projection_matrix;"	\
		"uniform int u_L_key_pressed;"			\
		"uniform vec3 u_LaR;	"				\
		"uniform vec3 u_LdR;	"				\
		"uniform vec3 u_LsR;	"				\
		"uniform vec4 u_light_positionR;"		\
		"uniform vec3 u_LaL;	"				\
		"uniform vec3 u_LdL;	"				\
		"uniform vec3 u_LsL;	"				\
		"uniform vec4 u_light_positionL;"		\
		"uniform vec3 u_Ka;"					\
		"uniform vec3 u_Kd;"					\
		"uniform vec3 u_Ks;"					\
		"uniform float u_material_shininess;"		\
		"out vec3 out_phong_ads_color;"			\
		"void main(void)"							\
		"{"											\
		"vec4 matNotUsed = u_model_matrix *  vPosition;"											\
		"mat4 matModelView = u_view_matrix;"											\
		"if (1 == u_L_key_pressed)"										\
		"{"											\
		/*"vec4 eyeCoordinates = u_view_matrix * u_model_matrix * vPosition;"											\*/
		"vec4 eyeCoordinates = matModelView *  vPosition;"											\
		"vec3 tnorm = normalize(mat3(matModelView) * vNormal);"											\
		"vec3 light_directionL = normalize(vec3(u_light_positionL - eyeCoordinates));"											\
		"float tn_dot_ldL = max(dot(tnorm, light_directionL), 0.0);"											\
		"vec3 ambientL = u_LaL * u_Ka;"											\
		"vec3 diffuseL = u_LdL * u_Kd * tn_dot_ldL;"											\
		"vec3 reflection_vectorL = reflect(-light_directionL, tnorm);"											\
		"vec3 viewer_vectorL = normalize(-eyeCoordinates.xyz);"											\
		"vec3 specularL = u_LsL * u_Ks * pow(max(dot(reflection_vectorL, viewer_vectorL), 0.0), u_material_shininess);"											\
		"out_phong_ads_color = ambientL + diffuseL + specularL;"											\
		"vec3 light_directionR = normalize(vec3(u_light_positionR - eyeCoordinates));"											\
		"float tn_dot_ldR = max(dot(tnorm, light_directionR), 0.0);"											\
		"vec3 ambientR = u_LaR * u_Ka;"											\
		"vec3 diffuseR = u_LdR * u_Kd * tn_dot_ldR;"											\
		"vec3 reflection_vectorR = reflect(-light_directionR, tnorm);"											\
		"vec3 viewer_vectorR = normalize(-eyeCoordinates.xyz);"											\
		"vec3 specularR = u_LsR * u_Ks * pow(max(dot(reflection_vectorR, viewer_vectorR), 0.0), u_material_shininess);"											\
		/*"out_phong_ads_color = ambientR + diffuseR + specularR;"											\*/
		"out_phong_ads_color = ambientL + ambientR + diffuseL+ diffuseR + specularL+ specularR;"											\
		"}"											\
		"else"											\
		"{"											\
		"out_phong_ads_color = vec3(1.0,1.0,1.0);"											\
		"}"											\
		"gl_Position = u_projection_matrix * matModelView * vPosition;"	\
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
	printf("<==Vertex Shader\n" );
	//-	Vertex shader.
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+	Fragment shader.

	printf("==>Fragment Shader\n" );
	//	Create shader.
	g_gluiShaderObjectFragment = glCreateShader(GL_FRAGMENT_SHADER);

	//	Provide source code.
	const GLchar *szFragmentShaderSourceCode =
		"#version 430 core"							\
		"\n"										\
		"in vec3 out_phong_ads_color;"				\
		"out vec4 vFragColor;"						\
		"void main(void)"							\
		"{"											\
		"vFragColor = vec4(out_phong_ads_color, 1.0);"					\
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
	printf("<== Fragment Shader\n" );	
	//-	Fragment shader.
	////////////////////////////////////////////////////////////////////

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

	glBindAttribLocation(g_gluiShaderObjectProgram, RTR_ATTRIBUTE_NORMAL, "vNormal");

	//	Link shader.
	glLinkProgram(g_gluiShaderObjectProgram);

	GLint gliLinkStatus;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;
	glsiWritten = 0;
	glGetProgramiv(g_gluiShaderObjectProgram, GL_LINK_STATUS, &gliLinkStatus);
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
	
	printf("<== Shader Program\n" );	
	//-	Shader program.
	////////////////////////////////////////////////////////////////////

	//-	Shader code
	////////////////////////////////////////////////////////////////////

	//
	//	The actual locations assigned to uniform variables are not known until the program object is linked successfully.
	//	After a program object has been linked successfully, the index values for uniform variables remain fixed until the next link command occurs.
	//
	g_gluiModelMat4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_model_matrix");
	if (-1 == g_gluiModelMat4Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_model_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiViewMat4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_view_matrix");
	if (-1 == g_gluiViewMat4Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_view_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiProjectionMat4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_projection_matrix");
	if (-1 == g_gluiProjectionMat4Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_projection_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiKeyPressedUniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_L_key_pressed");
	if (-1 == g_gluiKeyPressedUniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_L_key_pressed) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLaRVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_LaR");
	if (-1 == g_gluiLaRVec3Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_La) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLdRVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_LdR");
	if (-1 == g_gluiLdRVec3Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LD) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLsRVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_LsR");
	if (-1 == g_gluiLsRVec3Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Ls) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLightPositionRVec4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_light_positionR");
	if (-1 == g_gluiLightPositionRVec4Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_light_position) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLaLVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_LaL");
	if (-1 == g_gluiLaLVec3Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_La) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLdLVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_LdL");
	if (-1 == g_gluiLdLVec3Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LD) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLsLVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_LsL");
	if (-1 == g_gluiLsLVec3Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Ls) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLightPositionLVec4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_light_positionL");
	if (-1 == g_gluiLightPositionLVec4Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_light_position) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiKaVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Ka");
	if (-1 == g_gluiKaVec3Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Ka) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiKdVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Kd");
	if (-1 == g_gluiKdVec3Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Kd) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiKsVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Ks");
	if (-1 == g_gluiKsVec3Uniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Ks) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiMaterialShininessUniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_material_shininess");
	if (-1 == g_gluiMaterialShininessUniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_material_shininess) failed.");
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

	const GLfloat glfarrPyramidNormals[] =
	{
		//	Front face
		0.0f, 0.447214f, 0.894427f,	//	apex
		0.0f, 0.447214f, 0.894427f,	//	left_bottom
		0.0f, 0.447214f, 0.894427f,	//	right_bottom

		//	Right face
		0.894427f, 0.447214f, 0.0f,	//	apex
		0.894427f, 0.447214f, 0.0f,	//	left_bottom
		0.894427f, 0.447214f, 0.0f,	//	right_bottom

		//	Back face
		0.0f, 0.447214f, -0.894427f,	//	apex
		0.0f, 0.447214f, -0.894427f,	//	left_bottom
		0.0f, 0.447214f, -0.894427f,	//	right_bottom

		//	Left face
		-0.894427f, 0.447214f, 0.0f,	//	apex
		-0.894427f, 0.447214f, 0.0f,	//	left_bottom
		-0.894427f, 0.447214f, 0.0f,	//	right_bottom
	};

	glGenVertexArrays(1, &g_gluiVAOPyramid);	//	It is like recorder.
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
	//+ Vertex Normal
	glGenBuffers(1, &g_gluiVBONormal);
	glBindBuffer(GL_ARRAY_BUFFER, g_gluiVBONormal);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glfarrPyramidNormals), glfarrPyramidNormals, GL_STATIC_DRAW);

	glVertexAttribPointer(RTR_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(RTR_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//- Vertex Normal
	////////////////////////////////////////////////////////////////////

	glBindVertexArray(0);

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
	
	// resize
	resize(WIN_WIDTH, WIN_HEIGHT);
	printf("<==Initialize\n");	
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
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);	//	Change 3 for 3D

	//	Start using opengl program.
	glUseProgram(g_gluiShaderObjectProgram);

	if (true == g_bLight)
	{
		glUniform1i(g_gluiKeyPressedUniform, 1);

		glUniform3fv(g_gluiLaRVec3Uniform, 1, g_glfarrLightRAmbient);	//	Ambient
		glUniform3fv(g_gluiLdRVec3Uniform, 1, g_glfarrLightRDiffuse);	//	Diffuse
		glUniform3fv(g_gluiLsRVec3Uniform, 1, g_glfarrLightRSpecular);	//	Specular
		glUniform4fv(g_gluiLightPositionRVec4Uniform, 1, g_glfarrLightRPosition);

		glUniform3fv(g_gluiLaLVec3Uniform, 1, g_glfarrLightLAmbient);	//	Ambient
		glUniform3fv(g_gluiLdLVec3Uniform, 1, g_glfarrLightLDiffuse);	//	Diffuse
		glUniform3fv(g_gluiLsLVec3Uniform, 1, g_glfarrLightLSpecular);	//	Specular
		glUniform4fv(g_gluiLightPositionLVec4Uniform, 1, g_glfarrLightLPosition);

		glUniform3fv(g_gluiKaVec3Uniform, 1, g_glfarrMaterialAmbient);
		glUniform3fv(g_gluiKdVec3Uniform, 1, g_glfarrMaterialDiffuse);
		glUniform3fv(g_gluiKsVec3Uniform, 1, g_glfarrMaterialSpecular);
		glUniform1f(g_gluiMaterialShininessUniform, g_glfMaterialShininess);
	}
	else
	{
		glUniform1i(g_gluiKeyPressedUniform, 0);
	}

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	mat4 matModel = mat4::identity();
	mat4 matView = mat4::identity();
	mat4 matRotation = mat4::identity();	//	Good practice to initialize to identity matrix though it will change in next call.

	matModel = translate(0.0f, 0.0f, -5.0f);
	matRotation = rotate(g_glfAngle, 0.0f, 1.0f, 0.0f);
	matView = matModel * matRotation;

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOPyramid);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glDrawArrays(GL_TRIANGLES,0, 12);
	//	Unbind 'VAO'
	glBindVertexArray(0);

	//	Stop using opengl program.
	glUseProgram(0);

	glXSwapBuffers(gpDisplay,gWindow);
}

void update(void)
{
	g_glfAngle = g_glfAngle + 0.1f;

	if (g_glfAngle >= 360)
	{
		g_glfAngle = 0.0f;
	}
}

void uninitialize(void)
{
	if (g_gluiVBONormal)
	{
		glDeleteBuffers(1, &g_gluiVBONormal);
		g_gluiVBONormal = 0;
	}

	if (g_gluiVBOPosition)
	{
		glDeleteBuffers(1, &g_gluiVBOPosition);
		g_gluiVBOPosition = 0;
	}

	if (g_gluiVAOPyramid)
	{
		glDeleteVertexArrays(1, &g_gluiVAOPyramid);
		g_gluiVAOPyramid = 0;
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
