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
#include<assert.h>
#include<SOIL/SOIL.h>


using namespace vmath;

#define WIN_WIDTH 800
#define WIN_HEIGHT 600


enum {
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_TEXCOORD0,
	AMC_ATTRIBUTE_NORMAL
};



//For Window
Display *gpDisplay_RRJ = NULL;
Colormap gColormap_RRJ;
Window gWindow_RRJ;
XVisualInfo *gpXVisualInfo_RRJ = NULL;

//For FullScreen
bool bIsFullScreen_RRJ = false;

//For Error
FILE *gbFile_RRJ = NULL;

//For OpenGL
GLXContext gGLXContext_RRJ;
GLXFBConfig gGLXFBConfig_RRJ;
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
glXCreateContextAttribsARBProc glXCreateContextAttribsARB_RRJ = NULL;

//For Shader
GLuint gShaderProgramObject_RRJ;

//For Projection
mat4 gPerspectiveProjectionMatrix_RRJ;


//For Model
GLuint vao_Model_RRJ;
GLuint vbo_Model_Position_RRJ;
GLuint vbo_Model_Normal_RRJ;
GLuint vbo_Model_Texcoord_RRJ;


//For Texture
GLuint texture_Factory_RRJ;
GLuint samplerUniform_RRJ;



//For Uniform
GLuint modelMatrixUniform_RRJ;
GLuint viewMatrixUniform_RRJ;
GLuint projectionMatrixUniform_RRJ;
GLuint La_Uniform_RRJ;
GLuint Ld_Uniform_RRJ;
GLuint Ls_Uniform_RRJ;
GLuint lightPositionUniform_RRJ;
GLuint Ka_Uniform_RRJ;
GLuint Kd_Uniform_RRJ;
GLuint Ks_Uniform_RRJ;
GLuint materialShininessUniform_RRJ;
GLuint LKeyPressUniform_RRJ;


//For Lights
bool bLights_RRJ = false;
GLfloat lightAmbient_RRJ[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat lightDiffuse_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat lightSpecular_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat lightPosition_RRJ[] = { 100.0f, 100.0f, 100.0f, 1.0f };

//For Material
GLfloat materialAmbient_RRJ[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat materialDiffuse_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialSpecular_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialShininess_RRJ = 128.0f;


//Model Loading
struct VecFloat {

	float *pData;
	int iSize;
};


#define RRJ_SUCCESS 1
#define RRJ_ERROR 0


struct VecFloat *pVecFloat_Model_Vertices = NULL;
struct VecFloat *pVecFloat_Model_Normals = NULL;
struct VecFloat *pVecFloat_Model_Texcoord = NULL;

struct VecFloat *pVecFloat_Model_Sorted_Vertices = NULL;
struct VecFloat *pVecFloat_Model_Sorted_Normals = NULL;
struct VecFloat *pVecFloat_Model_Sorted_Texcoord = NULL;

struct VecFloat *pVecFloat_Model_Elements = NULL;

int PushBackVecFloat(struct VecFloat*, float);
void ShowVecFloat(struct VecFloat*);
struct VecFloat* CreateVecFloat(void);
int DestroyVecFloat(struct VecFloat*);


FILE *gbFile_Model = NULL;
FILE *gbFile_Vertices = NULL;
FILE *gbFile_Normals = NULL;
FILE *gbFile_TexCoord = NULL;
FILE *gbFile_FaceIndices = NULL;



int main(void){
	
	void CreateWindow(void);
	void initialize(void);
	void ToggleFullScreen(void);
	void resize(int, int);
	void display(void);
	void uninitialize(void);


	int winWidth_RRJ = WIN_WIDTH;
	int winHeight_RRJ = WIN_HEIGHT;


	gbFile_RRJ = fopen("Log.txt", "w");
	if(gbFile_RRJ == NULL){
		printf("Log  Creation Failed!!\n");
		uninitialize();
		exit(1);
	}
	else
		fprintf(gbFile_RRJ, "Log Created!!\n");



	gbFile_Model = fopen("factory.txt", "r");
	if (gbFile_Model == NULL) {
		printf("ERROR: Main(): Model File fopen() Failed!!\n");
		exit(0);
	}

	gbFile_Vertices = fopen("Vertices.txt", "w");
	if (gbFile_Vertices == NULL) {
		printf("ERROR:  Vertices.txt Creation Failed!!\n");
		exit(0);
	}

	gbFile_Normals = fopen("Normals.txt", "w");
	if (gbFile_Normals == NULL) {
		printf("ERROR: Normal.txt Failed!!\n");
		exit(0);
	}

	gbFile_TexCoord = fopen("Texcoord.txt", "w");
	if (gbFile_TexCoord == NULL) {
		printf("ERROR: Texcoord.txt Failed!!\n");
		exit(0);
	}

	gbFile_FaceIndices = fopen("Face.txt", "w");
	if (gbFile_FaceIndices == NULL) {
		printf("ERROR: Face.txt Failed!!\n");
		exit(0);
	}



	CreateWindow();
	initialize();
	ToggleFullScreen();

	//For Event Loop
	XEvent event_RRJ;
	KeySym keysym_RRJ;
	bool bDone_RRJ = false;

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

						case XK_L:
						case XK_l:
							if(bLights_RRJ == false)
								bLights_RRJ = true;
							else
								bLights_RRJ = false;
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

				default:
					break;
			}

		}
		display();
	}

	uninitialize();
	return(0);
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

void CreateWindow(void){

	void uninitialize(void);

	XSetWindowAttributes winAttrib_RRJ;
	int defaultScreen_RRJ;
	int styleMask_RRJ;

	GLXFBConfig *pGLXFBConfig_RRJ = NULL;
	GLXFBConfig bestFBConfig_RRJ;
	int iNumberOfFBConfig_RRJ = 0;
	XVisualInfo *pTempXVisualInfo_RRJ = NULL;


	static int frameBufferAttributes_RRJ[] = {
		GLX_X_RENDERABLE, True,
		GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
		GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
		GLX_RENDER_TYPE, GLX_RGBA_BIT,
		GLX_RED_SIZE, 8,
		GLX_GREEN_SIZE, 8,
		GLX_BLUE_SIZE, 8,
		GLX_ALPHA_SIZE, 8,
		GLX_DEPTH_SIZE, 24,
		GLX_DOUBLEBUFFER, True,
		None
	};

	gpDisplay_RRJ = XOpenDisplay(NULL);
	if(gpDisplay_RRJ == NULL){
		fprintf(gbFile_RRJ, "XOpenDisplay() Failed!!\n");
		uninitialize();
		exit(1);
	}

	defaultScreen_RRJ = XDefaultScreen(gpDisplay_RRJ);

	pGLXFBConfig_RRJ = glXChooseFBConfig(gpDisplay_RRJ, defaultScreen_RRJ, frameBufferAttributes_RRJ, &iNumberOfFBConfig_RRJ);
	if(pGLXFBConfig_RRJ == NULL){
		fprintf(gbFile_RRJ, "glXChooseFBConfig() Failed!!\n");
		uninitialize();
		exit(1);
	}

	
	int bestFrameBufferConfig_RRJ = -1;
	int bestNumberOfSamples_RRJ = -1;
	int worstFrameBufferConfig_RRJ = -1;
	int worstNumberOfSamples_RRJ = -1;


	fprintf(gbFile_RRJ, "FBConfig: %d\n", iNumberOfFBConfig_RRJ);

	for(int i = 0; i < iNumberOfFBConfig_RRJ; i++){
		pTempXVisualInfo_RRJ = glXGetVisualFromFBConfig(gpDisplay_RRJ, pGLXFBConfig_RRJ[i]);
		if(pTempXVisualInfo_RRJ){
			int samples, sampleBuffers;

			glXGetFBConfigAttrib(gpDisplay_RRJ, pGLXFBConfig_RRJ[i], GLX_SAMPLES, &samples);
			glXGetFBConfigAttrib(gpDisplay_RRJ, pGLXFBConfig_RRJ[i], GLX_SAMPLE_BUFFERS, &sampleBuffers);

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
	pGLXFBConfig_RRJ = NULL;

	gpXVisualInfo_RRJ = glXGetVisualFromFBConfig(gpDisplay_RRJ, bestFBConfig_RRJ);
	if(gpXVisualInfo_RRJ == NULL){
		fprintf(gbFile_RRJ, "glXGetVisualFromFBConfig() Failed!!\n");
		uninitialize();
		exit(1);
	}

	winAttrib_RRJ.border_pixel = 0;
	winAttrib_RRJ.border_pixmap = 0;
	winAttrib_RRJ.background_pixel = BlackPixel(gpDisplay_RRJ, defaultScreen_RRJ);
	winAttrib_RRJ.background_pixmap = 0;
	winAttrib_RRJ.colormap = XCreateColormap(gpDisplay_RRJ,
				RootWindow(gpDisplay_RRJ, gpXVisualInfo_RRJ->screen),
				gpXVisualInfo_RRJ->visual,
				AllocNone);
	gColormap_RRJ = winAttrib_RRJ.colormap;
	winAttrib_RRJ.event_mask = ExposureMask | VisibilityChangeMask | PointerMotionMask |
				KeyPressMask | ButtonPressMask | StructureNotifyMask;

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
			&winAttrib_RRJ);

	if(!gWindow_RRJ){
		fprintf(gbFile_RRJ, "XCreateWindow() Failed!!\n");
		uninitialize();
		exit(1);
	}
	
	XStoreName(gpDisplay_RRJ, gWindow_RRJ, "31-ModelLoading_WithTexture");

	Atom windowManagerDelete = XInternAtom(gpDisplay_RRJ, "WM_DELETE_WINDOW", True);
	XSetWMProtocols(gpDisplay_RRJ, gWindow_RRJ, &windowManagerDelete, 1);

	XMapWindow(gpDisplay_RRJ, gWindow_RRJ);
}

void initialize(void){
	
	void uninitialize(void);
	void resize(int, int);
	void LoadModel(void);
	GLuint LoadTexture(const char*);

	
	//Shader Object;
	GLint iVertexShaderObject_RRJ;
	GLint iFragmentShaderObject_RRJ;
	
	
	glXCreateContextAttribsARB_RRJ = (glXCreateContextAttribsARBProc)glXGetProcAddressARB((GLubyte*)"glXCreateContextAttribsARB");
	if(glXCreateContextAttribsARB_RRJ == NULL){
		fprintf(gbFile_RRJ, "glXGetProcAddressARB() Failed!!\n");
		uninitialize();
		exit(1);
	}

	const int attributes_RRJ[] = {
		GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
		GLX_CONTEXT_MINOR_VERSION_ARB, 5,
		GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
		None
	};

	gGLXContext_RRJ = glXCreateContextAttribsARB_RRJ(gpDisplay_RRJ, gGLXFBConfig_RRJ, NULL, True, attributes_RRJ);
	if(gGLXContext_RRJ == NULL){
		fprintf(gbFile_RRJ, "glXCreateContextAttribsARB_RRJ() Failed!!\n");
		fprintf(gbFile_RRJ, "Getting Context give by System!!\n");

		const int attribs_RRJ[] = {
			GLX_CONTEXT_MAJOR_VERSION_ARB, 1,
			GLX_CONTEXT_MINOR_VERSION_ARB, 0,
			None
		};

		gGLXContext_RRJ = glXCreateContextAttribsARB_RRJ(gpDisplay_RRJ, gGLXFBConfig_RRJ, NULL, True, attribs_RRJ);
	}

	if(!glXIsDirect(gpDisplay_RRJ, gGLXContext_RRJ))
		fprintf(gbFile_RRJ, "S/W Context!!\n");
	else
		fprintf(gbFile_RRJ, "H/W Context!!\n");

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
		"in vec3 vNormal;" \
		"in vec2 vTex;" \
		"out vec2 outTex;" \

		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"uniform vec4 u_light_position;" \
		"out vec3 viewer_vector_VS;" \
		"out vec3 tNorm_VS;" \
		"out vec3 lightDirection_VS;" \
		"void main(void)" \
		"{" \
			"vec4 eye_coordinate = u_view_matrix * u_model_matrix * vPosition;" \
			"viewer_vector_VS = vec3(-eye_coordinate);" \
			"tNorm_VS = mat3(u_view_matrix * u_model_matrix) * vNormal;" \
			"lightDirection_VS = vec3(u_light_position - eye_coordinate);" \
			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
			"outTex = vTex;" \
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

		"in vec2 outTex;" \
		"uniform sampler2D u_sampler;" \

		"in vec3 viewer_vector_VS;" \
		"in vec3 tNorm_VS;" \
		"in vec3 lightDirection_VS;" \
		"out vec4 FragColor;" \
		"uniform vec3 u_La;" \
		"uniform vec3 u_Ld;" \
		"uniform vec3 u_Ls;" \
		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_shininess;" \
		"uniform int u_LKeyPress;" \
		"void main(void)" \
		"{" \
			"if(u_LKeyPress == 1){" \
				"vec3 normalize_viewer_vector = normalize(viewer_vector_VS);" \
				"vec3 normalize_tNorm = normalize(tNorm_VS);" \
				"vec3 normalize_lightDirection = normalize(lightDirection_VS);" \
				"vec3 reflection_vector = reflect(-normalize_lightDirection, normalize_tNorm);" \
				"float s_dot_n = max(dot(normalize_lightDirection, normalize_tNorm), 0.0);" \
				"float r_dot_v = max(dot(reflection_vector, normalize_viewer_vector), 0.0);" \
				"vec3 ambient = u_La * u_Ka;" \
				"vec3 diffuse = u_Ld * u_Kd * s_dot_n;" \
				"vec3 specular = u_Ls * u_Ks * pow(r_dot_v, u_shininess);" \
				"vec3 Phong_ADS_Light = ambient + diffuse + specular;" \

				"vec4 tex = texture(u_sampler, outTex);" \

				"FragColor = tex * vec4(Phong_ADS_Light, 1.0);" \
			"}" \
			"else{" \
				"FragColor = vec4(1.0, 1.0, 1.0, 1.0);" \
			"}" \
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
	glBindAttribLocation(gShaderProgramObject_RRJ, AMC_ATTRIBUTE_NORMAL, "vNormal");
	glBindAttribLocation(gShaderProgramObject_RRJ, AMC_ATTRIBUTE_TEXCOORD0, "vTex");

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

	modelMatrixUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_model_matrix");
	viewMatrixUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_view_matrix");
	projectionMatrixUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_projection_matrix");
	La_Uniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_La");
	Ld_Uniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_Ld");
	Ls_Uniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_Ls");
	lightPositionUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_light_position");
	Ka_Uniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_Ka");
	Kd_Uniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_Kd");
	Ks_Uniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_Ks");
	materialShininessUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_shininess");
	LKeyPressUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_LKeyPress");

	samplerUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_sampler");


	/********** Position, Normal and Elements **********/
	pVecFloat_Model_Vertices = CreateVecFloat();
	pVecFloat_Model_Normals = CreateVecFloat();
	pVecFloat_Model_Texcoord = CreateVecFloat();

	pVecFloat_Model_Elements = CreateVecFloat();

	pVecFloat_Model_Sorted_Vertices = CreateVecFloat();
	pVecFloat_Model_Sorted_Normals = CreateVecFloat();
	pVecFloat_Model_Sorted_Texcoord = CreateVecFloat();


	LoadModel();

	fprintf(gbFile_RRJ, "Size: %ld\n", pVecFloat_Model_Vertices->iSize * sizeof(float));


	/********** Model Vao **********/
	glGenVertexArrays(1, &vao_Model_RRJ);
	glBindVertexArray(vao_Model_RRJ);

		/********** Position **********/
		glGenBuffers(1, &vbo_Model_Position_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Model_Position_RRJ);
		glBufferData(GL_ARRAY_BUFFER,
			pVecFloat_Model_Sorted_Vertices->iSize * sizeof(float),
			pVecFloat_Model_Sorted_Vertices->pData,
			GL_STATIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
			3,
			GL_FLOAT,
			GL_FALSE,
			0, NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** Texture **********/
		glGenBuffers(1, &vbo_Model_Texcoord_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Model_Texcoord_RRJ);
		glBufferData(GL_ARRAY_BUFFER, 
			sizeof(float) * pVecFloat_Model_Sorted_Texcoord->iSize,
			pVecFloat_Model_Sorted_Texcoord->pData,
			GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOORD0,
			2,
			GL_FLOAT,
			GL_FALSE,
			0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOORD0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** Normals **********/
		glGenBuffers(1, &vbo_Model_Normal_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Model_Normal_RRJ);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(float) * pVecFloat_Model_Sorted_Normals->iSize,
			pVecFloat_Model_Sorted_Normals->pData,
			GL_STATIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL,
			3,
			GL_FLOAT,
			GL_FALSE,
			0, NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


	glBindVertexArray(0);


	texture_Factory_RRJ = LoadTexture("factory.png");



	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix_RRJ = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);

}




GLuint LoadTexture(const char *path){

	GLuint texture;
	int imageWidth;
	int imageHeight;
	unsigned char *imageData = NULL;

	imageData = SOIL_load_image(path, &imageWidth, &imageHeight, 0, SOIL_LOAD_RGBA);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

	glTexImage2D(GL_TEXTURE_2D, 
			0,		//Mipmap Level
			GL_RGBA,		//Internal Image Format
			imageWidth, imageHeight,
			0,
			GL_RGBA,		//External Image Format
			GL_UNSIGNED_BYTE,
			imageData);

	glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	SOIL_free_image_data(imageData);

	return(texture);
}






void uninitialize(void) {

	GLXContext currentContext_RRJ = glXGetCurrentContext();


	if(texture_Factory_RRJ){
		glDeleteTextures(1, &texture_Factory_RRJ);
		texture_Factory_RRJ = 0;
	}


	if (pVecFloat_Model_Sorted_Texcoord) {
		DestroyVecFloat(pVecFloat_Model_Sorted_Texcoord);
		pVecFloat_Model_Sorted_Texcoord = NULL;
	}

	if (pVecFloat_Model_Sorted_Normals) {
		DestroyVecFloat(pVecFloat_Model_Sorted_Normals);
		pVecFloat_Model_Sorted_Normals = NULL;
	}


	if (pVecFloat_Model_Sorted_Vertices) {
		DestroyVecFloat(pVecFloat_Model_Sorted_Vertices);
		pVecFloat_Model_Sorted_Vertices = NULL;
	}


	if (pVecFloat_Model_Normals) {
		DestroyVecFloat(pVecFloat_Model_Normals);
		pVecFloat_Model_Normals = NULL;
	}

	if (pVecFloat_Model_Texcoord) {
		DestroyVecFloat(pVecFloat_Model_Texcoord);
		pVecFloat_Model_Texcoord = NULL;
	}

	if (pVecFloat_Model_Vertices) {
		DestroyVecFloat(pVecFloat_Model_Vertices);
		pVecFloat_Model_Vertices = NULL;
	}


	if (pVecFloat_Model_Elements) {
		DestroyVecFloat(pVecFloat_Model_Elements);
		pVecFloat_Model_Elements = NULL;
	}



	if (vbo_Model_Normal_RRJ) {
		glDeleteBuffers(1, &vbo_Model_Normal_RRJ);
		vbo_Model_Normal_RRJ = 0;
	}

	if (vbo_Model_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Model_Position_RRJ);
		vbo_Model_Position_RRJ = 0;
	}

	if (vao_Model_RRJ) {
		glDeleteVertexArrays(1, &vao_Model_RRJ);
		vao_Model_RRJ = 0;
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
		glUseProgram(0);
		glDeleteProgram(gShaderProgramObject_RRJ);
		gShaderProgramObject_RRJ = 0;

	}
	

	if(currentContext_RRJ != NULL && currentContext_RRJ == gGLXContext_RRJ)
		glXMakeCurrent(gpDisplay_RRJ, 0, 0);
		
	if(glXCreateContextAttribsARB_RRJ)
		glXCreateContextAttribsARB_RRJ = NULL;
	
	
	if(gGLXContext_RRJ)
		glXDestroyContext(gpDisplay_RRJ, gGLXContext_RRJ);
		
	if(gGLXFBConfig_RRJ)
		gGLXFBConfig_RRJ = 0;
		
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


void resize(int width, int height) {
	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	gPerspectiveProjectionMatrix_RRJ = mat4::identity();
	gPerspectiveProjectionMatrix_RRJ = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}


void display(void) {

	mat4 translateMatrix_RRJ;
	mat4 scaleMatrix_RRJ;
	mat4 rotateMatrix_RRJ;
	mat4 modelMatrix_RRJ;
	mat4 viewMatrix_RRJ;

	static GLfloat angle_Model_RRJ = 0.0f;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(gShaderProgramObject_RRJ);



	/********** Model **********/
	translateMatrix_RRJ = mat4::identity();
	scaleMatrix_RRJ = mat4::identity();
	rotateMatrix_RRJ = mat4::identity();
	modelMatrix_RRJ = mat4::identity();
	viewMatrix_RRJ = mat4::identity();


	translateMatrix_RRJ = translate(0.0f, -1.0f, -6.0f);
	rotateMatrix_RRJ = rotate(0.0f, angle_Model_RRJ, 0.0f);
	scaleMatrix_RRJ = scale(0.01f, 0.01f, 0.01f);
	modelMatrix_RRJ = modelMatrix_RRJ * translateMatrix_RRJ * scaleMatrix_RRJ * rotateMatrix_RRJ;

	glUniformMatrix4fv(modelMatrixUniform_RRJ, 1, GL_FALSE, modelMatrix_RRJ);
	glUniformMatrix4fv(viewMatrixUniform_RRJ, 1, GL_FALSE, viewMatrix_RRJ);
	glUniformMatrix4fv(projectionMatrixUniform_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);

	if (bLights_RRJ == true) {
		glUniform1i(LKeyPressUniform_RRJ, 1);


		glUniform3fv(La_Uniform_RRJ, 1, lightAmbient_RRJ);
		glUniform3fv(Ld_Uniform_RRJ, 1, lightDiffuse_RRJ);
		glUniform3fv(Ls_Uniform_RRJ, 1, lightSpecular_RRJ);
		glUniform4fv(lightPositionUniform_RRJ, 1, lightPosition_RRJ);

		glUniform3fv(Ka_Uniform_RRJ, 1, materialAmbient_RRJ);
		glUniform3fv(Kd_Uniform_RRJ, 1, materialDiffuse_RRJ);
		glUniform3fv(Ks_Uniform_RRJ, 1, materialSpecular_RRJ);
		glUniform1f(materialShininessUniform_RRJ, materialShininess_RRJ);
	}
	else
		glUniform1i(LKeyPressUniform_RRJ, 0);


	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture_Factory_RRJ);
	glUniform1i(samplerUniform_RRJ, 0);


	glBindVertexArray(vao_Model_RRJ);
		glDrawArrays(GL_TRIANGLES, 0, pVecFloat_Model_Sorted_Vertices->iSize / 3);
	glBindVertexArray(0);

	glUseProgram(0);

	angle_Model_RRJ += 0.8f;

	glXSwapBuffers(gpDisplay_RRJ, gWindow_RRJ);
}



struct VecFloat* CreateVecFloat(void) {

	struct VecFloat *pTemp = NULL;

	pTemp = (struct VecFloat*)malloc(sizeof(struct VecFloat));
	if (pTemp == NULL) {
		printf("ERROR: CreateVecInt(): Malloc() Failed!\n");
		exit(0);
	}

	memset((void*)pTemp, 0, sizeof(struct VecFloat));

	return(pTemp);
}


int PushBackVecFloat(struct VecFloat *pVec, float data) {

	pVec->pData = (float*)realloc(pVec->pData, sizeof(struct VecFloat) * (pVec->iSize + 1));

	assert(pVec->pData);

	pVec->iSize = pVec->iSize + 1;
	pVec->pData[pVec->iSize - 1] = data;
	//fprintf(gbFile_RRJ, "iSize: %d   iData: %f\n", pVec->iSize, pVec->pData[pVec->iSize - 1]);

	return(RRJ_SUCCESS);
}


void ShowVecFloat(struct VecFloat *pVec) {

	for (int i = 0; i < pVec->iSize; i++)
		fprintf(gbFile_RRJ, "P[%d]: %f\t", i, pVec->pData[i]);
}


int DestroyVecFloat(struct VecFloat *pVec) {


	free(pVec->pData);
	pVec->pData = NULL;
	pVec->iSize = 0;
	free(pVec);
	pVec = NULL;

	return(RRJ_SUCCESS);
}



void LoadModel(void) {

	char buffer[1024];
	char *firstToken = NULL;
	char *My_Strtok(char*, char);
	const char *space = " ";
	char *cContext = NULL;


	while (fgets(buffer, 1024, gbFile_Model) != NULL) {

		firstToken = strtok(buffer, space);

		if (strcmp(firstToken, "v") == 0) {
			//Vertices
			float x, y, z;
			x = (float)atof(strtok(NULL, space));
			y = (float)atof(strtok(NULL, space));
			z = (float)atof(strtok(NULL, space));

			fprintf(gbFile_Vertices, "%f/%f/%f\n", x, y, z);

			PushBackVecFloat(pVecFloat_Model_Vertices, x);
			//fprintf(gbFile_Vertices, "\n\nSrt: %f\n", pVecFloat_Model_Vertices->pData[0]);
			PushBackVecFloat(pVecFloat_Model_Vertices, y);
			PushBackVecFloat(pVecFloat_Model_Vertices, z);

		}
		else if (strcmp(firstToken, "vt") == 0) {
			//Texture

			float u, v;
			u = (float)atof(strtok(NULL, space));
			v = (float)atof(strtok(NULL, space));

			fprintf(gbFile_TexCoord, "%f/%f\n", u, v);
			PushBackVecFloat(pVecFloat_Model_Texcoord, u);
			PushBackVecFloat(pVecFloat_Model_Texcoord, 1.0f - v);
		}
		else if (strcmp(firstToken, "vn") == 0) {
			//Normals

			float x, y, z;
			x = (float)atof(strtok(NULL, space));
			y = (float)atof(strtok(NULL, space));
			z = (float)atof(strtok(NULL, space));

			fprintf(gbFile_Normals, "%f/%f/%f\n", x, y, z);
			PushBackVecFloat(pVecFloat_Model_Normals, x);
			PushBackVecFloat(pVecFloat_Model_Normals, y);
			PushBackVecFloat(pVecFloat_Model_Normals, z);

		}
		else if (strcmp(firstToken, "f") == 0) {
			//Faces


			for (int i = 0; i < 3; i++) {

				char *faces = strtok(NULL, space);
				int v, vt, vn;
				v = atoi(My_Strtok(faces, '/')) - 1;
				vt = atoi(My_Strtok(faces, '/')) - 1;
				vn = atoi(My_Strtok(faces, '/')) - 1;

				float x, y, z;

				//Sorted Vertices
				x = pVecFloat_Model_Vertices->pData[(v * 3) + 0];
				y = pVecFloat_Model_Vertices->pData[(v * 3) + 1];
				z = pVecFloat_Model_Vertices->pData[(v * 3) + 2];

				PushBackVecFloat(pVecFloat_Model_Sorted_Vertices, x);
				PushBackVecFloat(pVecFloat_Model_Sorted_Vertices, y);
				PushBackVecFloat(pVecFloat_Model_Sorted_Vertices, z);


				//Sorted Normals
				x = pVecFloat_Model_Normals->pData[(vn * 3) + 0];
				y = pVecFloat_Model_Normals->pData[(vn * 3) + 1];
				z = pVecFloat_Model_Normals->pData[(vn * 3) + 2];

				PushBackVecFloat(pVecFloat_Model_Sorted_Normals, x);
				PushBackVecFloat(pVecFloat_Model_Sorted_Normals, y);
				PushBackVecFloat(pVecFloat_Model_Sorted_Normals, z);


				//Sorted Texcoord;
				x = pVecFloat_Model_Texcoord->pData[(vt * 2) + 0];
				y = pVecFloat_Model_Texcoord->pData[(vt * 2) + 1];

				PushBackVecFloat(pVecFloat_Model_Sorted_Texcoord, x);
				PushBackVecFloat(pVecFloat_Model_Sorted_Texcoord, y);



				//Face Elements
				PushBackVecFloat(pVecFloat_Model_Elements, v);

				fprintf(gbFile_FaceIndices, "%d/ %d/ %d     ", v, vt, vn);
			}
			fprintf(gbFile_FaceIndices, "\n");


		}


	}


}

char gBuffer[128];

char* My_Strtok(char* str, char delimiter) {

	static int  i = 0;
	int  j = 0;
	char c;


	while ((c = str[i]) != delimiter && c != '\0') {
		gBuffer[j] = c;
		j = j + 1;
		i = i + 1;
	}

	gBuffer[j] = '\0';


	if (c == '\0') {
		i = 0;
	}
	else
		i = i + 1;


	return(gBuffer);
}



