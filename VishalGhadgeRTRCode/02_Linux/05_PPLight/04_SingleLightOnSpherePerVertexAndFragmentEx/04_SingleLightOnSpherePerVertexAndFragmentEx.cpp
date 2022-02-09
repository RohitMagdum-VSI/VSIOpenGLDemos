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
#include "Sphere.h"

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


float g_farrSphereVertices[1146];
float g_farrSphereNormals[1146];
float g_farrSphereTextures[764];
unsigned short g_uiarrSphereElements[2280];
GLuint g_gluiNumVertices;
GLuint g_gluiNumElements;

GLuint g_gluiShaderObjectVertexPerVertexLight;
GLuint g_gluiShaderObjectFragmentPerVertexLight;
GLuint g_gluiShaderObjectProgramPerVertexLight;

GLuint g_gluiShaderObjectVertexPerFragmentLight;
GLuint g_gluiShaderObjectFragmentPerFragmentLight;
GLuint g_gluiShaderObjectProgramPerFragmentLight;

GLuint g_gluiVAOSphere;
GLuint g_gluiVBOPosition;
GLuint g_gluiVBONormal;
GLuint g_gluiVBOElement;

/////////////////////////////////////////////////////////////////
//+Uniforms.

//	0 th uniform for Per Vertex Light
//	1 th uniform for Per Fragment Light
#define UNIFORM_INDEX_PER_VERTEX	0
#define UNIFORM_INDEX_PER_FRAGMENT	1

GLuint g_gluiModelMat4Uniform[2];
GLuint g_gluiViewMat4Uniform[2];
GLuint g_gluiProjectionMat4Uniform[2];

GLuint g_gluiLKeyPressedUniform[2];
GLuint g_gluiSKeyPressedUniform[2];

GLuint g_gluiLaVec3Uniform[2];	//	light ambient
GLuint g_gluiLdVec3Uniform[2];	//	light diffuse
GLuint g_gluiLsVec3Uniform[2];	//	light specular
GLuint g_gluiLightPositionVec4Uniform[2];

GLuint g_gluiKaVec3Uniform[2];//	Material ambient
GLuint g_gluiKdVec3Uniform[2];//	Material diffuse
GLuint g_gluiKsVec3Uniform[2];//	Material specular
GLuint g_gluiMaterialShininessUniform[2];
//-Uniforms.
/////////////////////////////////////////////////////////////////

GLfloat g_glfarrLightAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_glfarrLightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfarrLightSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfarrLightPosition[] = { 100.0f, 100.0f, 100.0f, 1.0f };

GLfloat g_glfarrMaterialAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_glfarrMaterialDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfarrMaterialSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfMaterialShininess = 50.0f;

mat4 g_matPerspectiveProjection;

bool g_bAnimate = false;
bool g_bLight = false;
int g_iLightType = 1;	//	1 for vertex light else fragment light.

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

						case XK_s:
						case XK_S:
							if (1 == g_iLightType)
							{
								g_iLightType = 2;
							}
							else
							{
								g_iLightType = 1;
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
	//+	Vertex shader Per vertex Light.

	fprintf(g_fpLogFile, "==>Vertex Shader-Per vertex light.");

	//	Create shader.
	g_gluiShaderObjectVertexPerVertexLight = glCreateShader(GL_VERTEX_SHADER);

	//	Provide source code.
	//	Provide source code.
	const GLchar *szVertexShaderSourceCodePerVertexLight =
		"#version 430 core"							\
		"\n"										\
		"in vec4 vPosition;"						\
		"in vec3 vNormal;"							\
		"uniform mat4 u_model_matrix;"	\
		"uniform mat4 u_view_matrix;"	\
		"uniform mat4 u_projection_matrix;"	\
		"uniform int u_L_key_pressed;"			\
		"uniform vec3 u_La;	"				\
		"uniform vec3 u_Ld;	"				\
		"uniform vec3 u_Ls;	"				\
		"uniform vec4 u_light_position;"		\
		"uniform vec3 u_Ka;"					\
		"uniform vec3 u_Kd;"					\
		"uniform vec3 u_Ks;"					\
		"uniform float u_material_shininess;"		\
		"out vec3 out_phong_ads_color;"			\
		"void main(void)"							\
		"{"											\
			"if (1 == u_L_key_pressed)"										\
			"{"											\
				"vec4 eyeCoordinates = u_view_matrix * u_model_matrix * vPosition;"											\
				"vec3 tnorm = normalize(mat3(u_view_matrix * u_model_matrix) * vNormal);"											\
				"vec3 light_direction = normalize(vec3(u_light_position - eyeCoordinates));"											\
				"float tn_dot_ld = max(dot(tnorm, light_direction), 0.0);"											\
				"vec3 ambient = u_La * u_Ka;"											\
				"vec3 diffuse = u_Ld * u_Kd * tn_dot_ld;"											\
				"vec3 reflection_vector = reflect(-light_direction, tnorm);"											\
				"vec3 viewer_vector = normalize(-eyeCoordinates.xyz);"											\
				"vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector, viewer_vector), 0.0), u_material_shininess);"											\
				"out_phong_ads_color = ambient + diffuse + specular;"											\
			"}"											\
			"else"											\
			"{"											\
				"out_phong_ads_color = vec3(1.0,1.0,1.0);"											\
			"}"											\
		"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;"	\
		"}";

	glShaderSource(g_gluiShaderObjectVertexPerVertexLight, 1, &szVertexShaderSourceCodePerVertexLight, NULL);

	//	Compile shader.
	glCompileShader(g_gluiShaderObjectVertexPerVertexLight);

	GLint gliCompileStatus;
	GLint gliInfoLogLength;
	char *pszInfoLog = NULL;
	GLsizei glsiWritten;
	glGetShaderiv(g_gluiShaderObjectVertexPerVertexLight, GL_COMPILE_STATUS, &gliCompileStatus);
	if (GL_FALSE == gliCompileStatus)
	{
		glGetShaderiv(g_gluiShaderObjectVertexPerVertexLight, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
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

		glGetShaderInfoLog(g_gluiShaderObjectVertexPerVertexLight, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//-	Vertex shader Per vertex Light.
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+	Vertex shader Per Fragment Light.

	fprintf(g_fpLogFile, "==>Vertex Shader-Per Fragment light.");

	//	Create shader.
	g_gluiShaderObjectVertexPerFragmentLight = glCreateShader(GL_VERTEX_SHADER);

	//	Provide source code.
	const GLchar *szVertexShaderSourceCodePerFragmentLight =
		"#version 430 core"							\
		"\n"										\
		"in vec4 vPosition;"						\
		"in vec3 vNormal;"							\
		"uniform mat4 u_model_matrix;"	\
		"uniform mat4 u_view_matrix;"	\
		"uniform mat4 u_projection_matrix;"	\
		"uniform vec4 u_light_position;"		\
		"uniform int u_L_key_pressed;"			\
		"out vec3 transformed_normals;"			\
		"out vec3 light_direction;"			\
		"out vec3 viewer_vector;"			\
		"void main(void)"							\
		"{"											\
			"if (1 == u_L_key_pressed)"										\
			"{"											\
				"vec4 eyeCoordinates = u_view_matrix * u_model_matrix * vPosition;"											\
				"transformed_normals = mat3(u_view_matrix * u_model_matrix) * vNormal;"											\
				"light_direction = vec3(u_light_position) - eyeCoordinates.xyz;"											\
				"viewer_vector = -eyeCoordinates.xyz;"											\
			"}"											\
			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;"	\
		"}";

	glShaderSource(g_gluiShaderObjectVertexPerFragmentLight, 1, &szVertexShaderSourceCodePerFragmentLight, NULL);

	//	Compile shader.
	glCompileShader(g_gluiShaderObjectVertexPerFragmentLight);

	gliCompileStatus = 0;
	gliInfoLogLength = 0;
	glsiWritten = 0;
	glGetShaderiv(g_gluiShaderObjectVertexPerFragmentLight, GL_COMPILE_STATUS, &gliCompileStatus);
	if (GL_FALSE == gliCompileStatus)
	{
		glGetShaderiv(g_gluiShaderObjectVertexPerFragmentLight, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
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

		glGetShaderInfoLog(g_gluiShaderObjectVertexPerFragmentLight, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//-	Vertex shader Per Fragment Light.
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+	Fragment shader - Per Vertex Light.

	fprintf(g_fpLogFile, "==>Fragment Shader- Per vertex Light.");

	//	Create shader.
	g_gluiShaderObjectFragmentPerVertexLight = glCreateShader(GL_FRAGMENT_SHADER);

	//	Provide source code.
	const GLchar *szFragmentShaderSourceCodePerVertexLight =
		"#version 430 core"							\
		"\n"										\
		"in vec3 out_phong_ads_color;"				\
		"out vec4 vFragColor;"						\
		"void main(void)"							\
		"{"											\
		"vFragColor = vec4(out_phong_ads_color, 1.0);"					\
		"}";

	glShaderSource(g_gluiShaderObjectFragmentPerVertexLight, 1, &szFragmentShaderSourceCodePerVertexLight, NULL);

	//	Compile shader.
	glCompileShader(g_gluiShaderObjectFragmentPerVertexLight);

	gliCompileStatus = 0;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;
	glsiWritten = 0;
	glGetShaderiv(g_gluiShaderObjectFragmentPerVertexLight, GL_COMPILE_STATUS, &gliCompileStatus);
	if (GL_FALSE == gliCompileStatus)
	{
		glGetShaderiv(g_gluiShaderObjectFragmentPerVertexLight, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
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

		glGetShaderInfoLog(g_gluiShaderObjectFragmentPerVertexLight, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//-	Fragment shader - Per Vertex Light.
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+	Fragment shader - Per Fragment Light.

	fprintf(g_fpLogFile, "==>Fragment Shader- Per Fragment Light.");

	//	Create shader.
	g_gluiShaderObjectFragmentPerFragmentLight = glCreateShader(GL_FRAGMENT_SHADER);

	//	Provide source code.
	const GLchar *szFragmentShaderSourceCodePerFragmentLight =
		"#version 430 core"							\
		"\n"										\
		"in vec3 transformed_normals;"			\
		"in vec3 light_direction;"			\
		"in vec3 viewer_vector;"			\
		"uniform vec3 u_La;	"				\
		"uniform vec3 u_Ld;	"				\
		"uniform vec3 u_Ls;	"				\
		"uniform vec3 u_Ka;"					\
		"uniform vec3 u_Kd;"					\
		"uniform vec3 u_Ks;"					\
		"uniform float u_material_shininess;"		\
		"uniform int u_L_key_pressed;"			\
		"out vec4 vFragColor;"						\
		"void main(void)"							\
		"{"											\
			"vec3 phong_ads_color;"					\
			"if (1 == u_L_key_pressed)"										\
			"{"											\
				"vec3 normalized_transformed_normals = normalize(transformed_normals);"											\
				"vec3 normalized_light_direction = normalize(light_direction);"											\
				"vec3 normalized_viewer_vector = normalize(viewer_vector);"											\
				"vec3 ambient = u_La * u_Ka;"											\
				"float tn_dot_ld = max(dot(normalized_transformed_normals, normalized_light_direction), 0.0);"											\
				"vec3 diffuse = u_Ld * u_Kd * tn_dot_ld;"											\
				"vec3 reflection_vector = reflect(-normalized_light_direction, normalized_transformed_normals);"											\
				"vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector, normalized_viewer_vector), 0.0), u_material_shininess);"											\
				"phong_ads_color = ambient + diffuse + specular;"											\
			"}"											\
			"else"											\
			"{"											\
			"	phong_ads_color = vec3(1.0,1.0,1.0);"											\
			"}"											\
			"vFragColor = vec4(phong_ads_color, 1.0);"					\
		"}";

	glShaderSource(g_gluiShaderObjectFragmentPerFragmentLight, 1, &szFragmentShaderSourceCodePerFragmentLight, NULL);

	//	Compile shader.
	glCompileShader(g_gluiShaderObjectFragmentPerFragmentLight);

	gliCompileStatus = 0;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;
	glsiWritten = 0;
	glGetShaderiv(g_gluiShaderObjectFragmentPerFragmentLight, GL_COMPILE_STATUS, &gliCompileStatus);
	if (GL_FALSE == gliCompileStatus)
	{
		glGetShaderiv(g_gluiShaderObjectFragmentPerFragmentLight, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
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

		glGetShaderInfoLog(g_gluiShaderObjectFragmentPerFragmentLight, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//-	Fragment shader - Per Fragment Light.
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+	Shader program - Per Vertex Light.

	//	Create.
	g_gluiShaderObjectProgramPerVertexLight = glCreateProgram();

	//	Attach vertex shader to shader program.
	glAttachShader(g_gluiShaderObjectProgramPerVertexLight, g_gluiShaderObjectVertexPerVertexLight);

	//	Attach Fragment shader to shader program.
	glAttachShader(g_gluiShaderObjectProgramPerVertexLight, g_gluiShaderObjectFragmentPerVertexLight);

	//	IMP:Pre-link binding of shader program object with vertex shader position attribute.
	glBindAttribLocation(g_gluiShaderObjectProgramPerVertexLight, RTR_ATTRIBUTE_POSITION, "vPosition");

	glBindAttribLocation(g_gluiShaderObjectProgramPerVertexLight, RTR_ATTRIBUTE_NORMAL, "vNormal");

	//	Link shader.
	glLinkProgram(g_gluiShaderObjectProgramPerVertexLight);

	GLint gliLinkStatus;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;
	glsiWritten = 0;
	glGetProgramiv(g_gluiShaderObjectProgramPerVertexLight, GL_LINK_STATUS, &gliLinkStatus);
	if (GL_FALSE == gliLinkStatus)
	{
		glGetProgramiv(g_gluiShaderObjectProgramPerVertexLight, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
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

		glGetProgramInfoLog(g_gluiShaderObjectProgramPerVertexLight, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//-	Shader program - Per Vertex Light.
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+	Shader program - Per Fragment Light.

	//	Create.
	g_gluiShaderObjectProgramPerFragmentLight = glCreateProgram();

	//	Attach vertex shader to shader program.
	glAttachShader(g_gluiShaderObjectProgramPerFragmentLight, g_gluiShaderObjectVertexPerFragmentLight);

	//	Attach Fragment shader to shader program.
	glAttachShader(g_gluiShaderObjectProgramPerFragmentLight, g_gluiShaderObjectFragmentPerFragmentLight);

	//	IMP:Pre-link binding of shader program object with vertex shader position attribute.
	glBindAttribLocation(g_gluiShaderObjectProgramPerFragmentLight, RTR_ATTRIBUTE_POSITION, "vPosition");

	glBindAttribLocation(g_gluiShaderObjectProgramPerFragmentLight, RTR_ATTRIBUTE_NORMAL, "vNormal");

	//	Link shader.
	glLinkProgram(g_gluiShaderObjectProgramPerFragmentLight);

	gliLinkStatus = 0;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;
	glsiWritten = 0;
	glGetProgramiv(g_gluiShaderObjectProgramPerFragmentLight, GL_LINK_STATUS, &gliLinkStatus);
	if (GL_FALSE == gliLinkStatus)
	{
		glGetProgramiv(g_gluiShaderObjectProgramPerFragmentLight, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
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

		glGetProgramInfoLog(g_gluiShaderObjectProgramPerFragmentLight, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//-	Shader program - Per Fragment Light.
	////////////////////////////////////////////////////////////////////

	//-	Shader code
	////////////////////////////////////////////////////////////////////

	//
	//	The actual locations assigned to uniform variables are not known until the program object is linked successfully.
	//	After a program object has been linked successfully, the index values for uniform variables remain fixed until the next link command occurs.
	//

	//+	Per Vertex uniform variables.
	g_gluiModelMat4Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_model_matrix");
	if (-1 == g_gluiModelMat4Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_model_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiViewMat4Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_view_matrix");
	if (-1 == g_gluiViewMat4Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_view_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiProjectionMat4Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_projection_matrix");
	if (-1 == g_gluiProjectionMat4Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_projection_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLKeyPressedUniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_L_key_pressed");
	if (-1 == g_gluiLKeyPressedUniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_L_key_pressed) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLaVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_La");
	if (-1 == g_gluiLaVec3Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_La) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLdVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_Ld");
	if (-1 == g_gluiLdVec3Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LD) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLsVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_Ls");
	if (-1 == g_gluiLsVec3Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Ls) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLightPositionVec4Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_light_position");
	if (-1 == g_gluiLightPositionVec4Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_light_position) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiKaVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_Ka");
	if (-1 == g_gluiKaVec3Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Ka) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiKdVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_Kd");
	if (-1 == g_gluiKdVec3Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Kd) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiKsVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_Ks");
	if (-1 == g_gluiKsVec3Uniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Ks) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiMaterialShininessUniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_material_shininess");
	if (-1 == g_gluiMaterialShininessUniform[UNIFORM_INDEX_PER_VERTEX])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_material_shininess) failed.");
		uninitialize();
		exit(0);
	}
	//-	Per Vertex uniform variables.

	//+	Per Fragment uniform variables.
	g_gluiModelMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_model_matrix");
	if (-1 == g_gluiModelMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_model_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiViewMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_view_matrix");
	if (-1 == g_gluiViewMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_view_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiProjectionMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_projection_matrix");
	if (-1 == g_gluiProjectionMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_projection_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLKeyPressedUniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_L_key_pressed");
	if (-1 == g_gluiLKeyPressedUniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_L_key_pressed) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLaVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_La");
	if (-1 == g_gluiLaVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_La) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLdVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_Ld");
	if (-1 == g_gluiLdVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_LD) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLsVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_Ls");
	if (-1 == g_gluiLsVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Ls) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiLightPositionVec4Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_light_position");
	if (-1 == g_gluiLightPositionVec4Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_light_position) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiKaVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_Ka");
	if (-1 == g_gluiKaVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Ka) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiKdVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_Kd");
	if (-1 == g_gluiKdVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Kd) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiKsVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_Ks");
	if (-1 == g_gluiKsVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_Ks) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiMaterialShininessUniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_material_shininess");
	if (-1 == g_gluiMaterialShininessUniform[UNIFORM_INDEX_PER_FRAGMENT])
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_material_shininess) failed.");
		uninitialize();
		exit(0);
	}
	//-	Per Fragment uniform variables.

	////////////////////////////////////////////////////////////////////
	//+	Vertices,color, shader attribute, vbo,vao initialization.

	getSphereVertexData(g_farrSphereVertices, g_farrSphereNormals, g_farrSphereTextures, g_uiarrSphereElements);
	g_gluiNumVertices = getNumberOfSphereVertices();
	g_gluiNumElements = getNumberOfSphereElements();

	glGenVertexArrays(1, &g_gluiVAOSphere);	//	It is like recorder.
	glBindVertexArray(g_gluiVAOSphere);

	////////////////////////////////////////////////////////////////////
	//+ Vertex position
	glGenBuffers(1, &g_gluiVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, g_gluiVBOPosition);

	glBufferData(GL_ARRAY_BUFFER, sizeof(g_farrSphereVertices), g_farrSphereVertices, GL_STATIC_DRAW);

	glVertexAttribPointer(RTR_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(RTR_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//- Vertex position
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+ Vertex Normal
	glGenBuffers(1, &g_gluiVBONormal);
	glBindBuffer(GL_ARRAY_BUFFER, g_gluiVBONormal);

	glBufferData(GL_ARRAY_BUFFER, sizeof(g_farrSphereNormals), g_farrSphereNormals, GL_STATIC_DRAW);

	glVertexAttribPointer(RTR_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(RTR_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//- Vertex Normal
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+ Vertex Element
	glGenBuffers(1, &g_gluiVBOElement);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(g_uiarrSphereElements), g_uiarrSphereElements, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	//- Vertex Element
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
	int index;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);	//	Change 3 for 3D

	//	Start using opengl program.
	if (1 == g_iLightType)
	{
		glUseProgram(g_gluiShaderObjectProgramPerVertexLight);
		index = UNIFORM_INDEX_PER_VERTEX;
	}
	else
	{
		glUseProgram(g_gluiShaderObjectProgramPerFragmentLight);
		index = UNIFORM_INDEX_PER_FRAGMENT;
	}

	if (true == g_bLight)
	{
		glUniform1i(g_gluiLKeyPressedUniform[index], 1);

		glUniform3fv(g_gluiLaVec3Uniform[index], 1, g_glfarrLightAmbient);	//	Ambient
		glUniform3fv(g_gluiLdVec3Uniform[index], 1, g_glfarrLightDiffuse);	//	Diffuse
		glUniform3fv(g_gluiLsVec3Uniform[index], 1, g_glfarrLightSpecular);	//	Specular
		glUniform4fv(g_gluiLightPositionVec4Uniform[index], 1, g_glfarrLightPosition);

		glUniform3fv(g_gluiKaVec3Uniform[index], 1, g_glfarrMaterialAmbient);
		glUniform3fv(g_gluiKdVec3Uniform[index], 1, g_glfarrMaterialDiffuse);
		glUniform3fv(g_gluiKsVec3Uniform[index], 1, g_glfarrMaterialSpecular);
		glUniform1f(g_gluiMaterialShininessUniform[index], g_glfMaterialShininess);
	}
	else
	{
		glUniform1i(g_gluiLKeyPressedUniform[index], 0);
	}

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	mat4 matModel = mat4::identity();
	mat4 matView = mat4::identity();

	matModel = translate(0.0f, 0.0f, -3.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform[index], 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform[index], 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform[index], 1, GL_FALSE, g_matPerspectiveProjection);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOSphere);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gluiVBOElement);
	glDrawElements(GL_TRIANGLES, g_gluiNumElements, GL_UNSIGNED_SHORT, 0);
	//	Unbind 'VAO'
	glBindVertexArray(0);

	//	Stop using opengl program.
	glUseProgram(0);

	glXSwapBuffers(gpDisplay,gWindow);
}

void update(void)
{
	
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

	if (g_gluiVBOElement)
	{
		glDeleteBuffers(1, &g_gluiVBOElement);
		g_gluiVBOElement = 0;
	}

	if (g_gluiVAOSphere)
	{
		glDeleteVertexArrays(1, &g_gluiVAOSphere);
		g_gluiVAOSphere = 0;
	}

	if (g_gluiShaderObjectVertexPerVertexLight)
	{
		glDetachShader(g_gluiShaderObjectProgramPerVertexLight, g_gluiShaderObjectVertexPerVertexLight);
		glDeleteShader(g_gluiShaderObjectVertexPerVertexLight);
		g_gluiShaderObjectVertexPerVertexLight = 0;
	}

	if (g_gluiShaderObjectVertexPerFragmentLight)
	{
		glDetachShader(g_gluiShaderObjectProgramPerFragmentLight, g_gluiShaderObjectVertexPerFragmentLight);
		glDeleteShader(g_gluiShaderObjectVertexPerFragmentLight);
		g_gluiShaderObjectVertexPerFragmentLight = 0;
	}

	if (g_gluiShaderObjectFragmentPerVertexLight)
	{
		glDetachShader(g_gluiShaderObjectProgramPerVertexLight, g_gluiShaderObjectFragmentPerVertexLight);
		glDeleteShader(g_gluiShaderObjectFragmentPerVertexLight);
		g_gluiShaderObjectFragmentPerVertexLight = 0;
	}

	if (g_gluiShaderObjectFragmentPerFragmentLight)
	{
		glDetachShader(g_gluiShaderObjectProgramPerFragmentLight, g_gluiShaderObjectFragmentPerFragmentLight);
		glDeleteShader(g_gluiShaderObjectFragmentPerFragmentLight);
		g_gluiShaderObjectFragmentPerFragmentLight = 0;
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

	if (g_gluiShaderObjectProgramPerVertexLight)
	{
		glDeleteProgram(g_gluiShaderObjectProgramPerVertexLight);
		g_gluiShaderObjectProgramPerVertexLight = 0;
	}

	if (g_gluiShaderObjectProgramPerFragmentLight)
	{
		glDeleteProgram(g_gluiShaderObjectProgramPerFragmentLight);
		g_gluiShaderObjectProgramPerFragmentLight = 0;
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
