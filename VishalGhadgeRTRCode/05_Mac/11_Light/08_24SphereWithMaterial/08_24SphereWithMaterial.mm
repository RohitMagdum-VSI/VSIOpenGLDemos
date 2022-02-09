//Headers
#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>

#import <QuartzCore/CVDisplayLink.h>		//	To link with with core videos display.
#import <OpenGL/gl3.h>		//	Core profile
#import <OpenGL/gl3ext.h>	//	Opengl extensions.
#import "vmath.h"

//	'C' style global function decleration
CVReturn MyDisplayLinkCallback(CVDisplayLinkRef, const CVTimeStamp*, const CVTimeStamp*, CVOptionFlags, CVOptionFlags*, void*);

//	Global variables.
FILE *g_fpLogFile = NULL;

enum
{
	RTR_ATTRIBUTE_POSITION = 0,
	RTR_ATTRIBUTE_COLOR,
	RTR_ATTRIBUTE_NORMAL,
	RTR_ATTRIBUTE_TEXTURE0
};

//
//	Light R == Red Light
//
GLfloat g_glfarrLightAmbient[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfarrLightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfarrLightSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfarrLightPosition[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Give position runtime.

///////////////////////////////////////////////////////////////////////////
//+Material
//
//	Materail 00
//
GLfloat g_arrMaterial00Ambient[] = { 0.0215f, 0.1745f, 0.0215f, 1.0f };
GLfloat g_arrMaterial00Diffuse[] = { 0.07568f, 0.61424f, 0.07568f, 1.0f };
GLfloat g_arrMaterial00Specular[] = { 0.633f, 0.727811f, 0.633f, 1.0f };
GLfloat g_Material00Shininess = 0.6f * 128.0f;

//
//	Materail 10
//
GLfloat g_arrMaterial10Ambient[] = { 0.135f, 0.2225f, 0.1575f, 1.0f };
GLfloat g_arrMaterial10Diffuse[] = { 0.54f, 0.89f, 0.63f, 1.0f };
GLfloat g_arrMaterial10Specular[] = { 0.316228f, 0.316228f, 0.316228f, 1.0f };
GLfloat g_Material10Shininess = 0.1f * 128.0f;

//
//	Materail 20
//
GLfloat g_arrMaterial20Ambient[] = { 0.05375f, 0.05f, 0.06625f, 1.0f };
GLfloat g_arrMaterial20Diffuse[] = { 0.18275f, 0.17f, 0.22525f, 1.0f };
GLfloat g_arrMaterial20Specular[] = { 0.332741f, 0.328634f, 0.346435f, 1.0f };
GLfloat g_Material20Shininess = 0.3f * 128.0f;

//
//	Materail 30
//
GLfloat g_arrMaterial30Ambient[] = { 0.25f, 0.20725f, 0.20725f, 1.0f };
GLfloat g_arrMaterial30Diffuse[] = { 1.0f, 0.829f, 0.829f, 1.0f };
GLfloat g_arrMaterial30Specular[] = { 0.296648f, 0.296648f, 0.296648f, 1.0f };
GLfloat g_Material30Shininess = 0.088f * 128.0f;

//
//	Materail 40
//
GLfloat g_arrMaterial40Ambient[] = { 0.1745f, 0.01175f, 0.01175f, 1.0f };
GLfloat g_arrMaterial40Diffuse[] = { 0.61424f, 0.04136f, 0.04136f, 1.0f };
GLfloat g_arrMaterial40Specular[] = { 0.727811f, 0.626959f, 0.626959f, 1.0f };
GLfloat g_Material40Shininess = 0.6f * 128.0f;

//
//	Materail 50
//
GLfloat g_arrMaterial50Ambient[] = { 0.1f, 0.18725f, 0.1745f, 1.0f };
GLfloat g_arrMaterial50Diffuse[] = { 0.396f, 0.74151f, 0.69102f, 1.0f };
GLfloat g_arrMaterial50Specular[] = { 0.297254f, 0.30829f, 0.306678f, 1.0f };
GLfloat g_Material50Shininess = 0.1f * 128.0f;

//
//	Materail 01
//
GLfloat g_arrMaterial01Ambient[] = { 0.329412f, 0.223529f, 0.027451f, 1.0f };
GLfloat g_arrMaterial01Diffuse[] = { 0.780392f, 0.568627f, 0.113725f, 1.0f };
GLfloat g_arrMaterial01Specular[] = { 0.992157f, 0.941176f, 0.807843f, 1.0f };
GLfloat g_Material01Shininess = 0.21794872f * 128.0f;

//
//	Materail 11
//
GLfloat g_arrMaterial11Ambient[] = { 0.2125f, 0.1275f, 0.054f, 1.0f };
GLfloat g_arrMaterial11Diffuse[] = { 0.714f, 0.4284f, 0.18144f, 1.0f };
GLfloat g_arrMaterial11Specular[] = { 0.393548f, 0.271906f, 0.166721f, 1.0f };
GLfloat g_Material11Shininess = 0.2f * 128.0f;

//
//	Materail 21
//
GLfloat g_arrMaterial21Ambient[] = { 0.25f, 0.25f, 0.25f, 1.0f };
GLfloat g_arrMaterial21Diffuse[] = { 0.4f, 0.4f, 0.4f, 1.0f };
GLfloat g_arrMaterial21Specular[] = { 0.774597f, 0.774597f, 0.774597f, 1.0f };
GLfloat g_Material21Shininess = 0.6f * 128.0f;

//
//	Materail 31
//
GLfloat g_arrMaterial31Ambient[] = { 0.19125f, 0.0735f, 0.0225f, 1.0f };
GLfloat g_arrMaterial31Diffuse[] = { 0.7038f, 0.27048f, 0.0828f, 1.0f };
GLfloat g_arrMaterial31Specular[] = { 0.256777f, 0.137622f, 0.296648f, 1.0f };
GLfloat g_Material31Shininess = 0.1f * 128.0f;

//
//	Materail 41
//
GLfloat g_arrMaterial41Ambient[] = { 0.24725f, 0.1995f, 0.0745f, 1.0f };
GLfloat g_arrMaterial41Diffuse[] = { 0.75164f, 0.60648f, 0.22648f, 1.0f };
GLfloat g_arrMaterial41Specular[] = { 0.628281f, 0.555802f, 0.366065f, 1.0f };
GLfloat g_Material41Shininess = 0.4f * 128.0f;

//
//	Materail 51
//
GLfloat g_arrMaterial51Ambient[] = { 0.19225f, 0.19225f, 0.19225f, 1.0f };
GLfloat g_arrMaterial51Diffuse[] = { 0.50754f, 0.50754f, 0.50754f, 1.0f };
GLfloat g_arrMaterial51Specular[] = { 0.508273f, 0.508273f, 0.508273f, 1.0f };
GLfloat g_Material51Shininess = 0.4f * 128.0f;

//
//	Materail 02
//
GLfloat g_arrMaterial02Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial02Diffuse[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial02Specular[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_Material02Shininess = 0.25f * 128.0f;

//
//	Materail 12
//
GLfloat g_arrMaterial12Ambient[] = { 0.0f, 0.1f, 0.06f, 1.0f };
GLfloat g_arrMaterial12Diffuse[] = { 0.0f, 0.50980392f, 0.50980392f, 1.0f };
GLfloat g_arrMaterial12Specular[] = { 0.50980392f, 0.50980392f, 0.50980392f, 1.0f };
GLfloat g_Material12Shininess = 0.25f * 128.0f;

//
//	Materail 22
//
GLfloat g_arrMaterial22Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial22Diffuse[] = { 0.1f, 0.35f, 0.1f, 1.0f };
GLfloat g_arrMaterial22Specular[] = { 0.45f, 0.45f, 0.45f, 1.0f };
GLfloat g_Material22Shininess = 0.25f * 128.0f;

//
//	Materail 32
//
GLfloat g_arrMaterial32Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial32Diffuse[] = { 0.5f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial32Specular[] = { 0.7f, 0.6f, 0.6f, 1.0f };
GLfloat g_Material32Shininess = 0.25f * 128.0f;

//
//	Materail 42
//
GLfloat g_arrMaterial42Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial42Diffuse[] = { 0.55f, 0.55f, 0.55f, 1.0f };
GLfloat g_arrMaterial42Specular[] = { 0.70f, 0.70f, 0.70f, 1.0f };
GLfloat g_Material42Shininess = 0.25f * 128.0f;

//
//	Materail 52
//
GLfloat g_arrMaterial52Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial52Diffuse[] = { 0.5f, 0.5f, 0.0f, 1.0f };
GLfloat g_arrMaterial52Specular[] = { 0.60f, 0.60f, 0.50f, 1.0f };
GLfloat g_Material52Shininess = 0.25f * 128.0f;

//
//	Materail 03
//
GLfloat g_arrMaterial03Ambient[] = { 0.02f, 0.02f, 0.02f, 1.0f };
GLfloat g_arrMaterial03Diffuse[] = { 0.01f, 0.01f, 0.01f, 1.0f };
GLfloat g_arrMaterial03Specular[] = { 0.4f, 0.4f, 0.4f, 1.0f };
GLfloat g_Material03Shininess = 0.078125f * 128.0f;

//
//	Materail 13
//
GLfloat g_arrMaterial13Ambient[] = { 0.0f, 0.05f, 0.05f, 1.0f };
GLfloat g_arrMaterial13Diffuse[] = { 0.4f, 0.5f, 0.5f, 1.0f };
GLfloat g_arrMaterial13Specular[] = { 0.04f, 0.7f, 0.7f, 1.0f };
GLfloat g_Material13Shininess = 0.078125f * 128.0f;

//
//	Materail 23
//
GLfloat g_arrMaterial23Ambient[] = { 0.0f, 0.05f, 0.0f, 1.0f };
GLfloat g_arrMaterial23Diffuse[] = { 0.4f, 0.5f, 0.4f, 1.0f };
GLfloat g_arrMaterial23Specular[] = { 0.04f, 0.7f, 0.04f, 1.0f };
GLfloat g_Material23Shininess = 0.078125f * 128.0f;

//
//	Materail 33
//
GLfloat g_arrMaterial33Ambient[] = { 0.05f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial33Diffuse[] = { 0.5f, 0.4f, 0.4f, 1.0f };
GLfloat g_arrMaterial33Specular[] = { 0.7f, 0.04f, 0.04f, 1.0f };
GLfloat g_Material33Shininess = 0.078125f * 128.0f;

//
//	Materail 43
//
GLfloat g_arrMaterial43Ambient[] = { 0.05f, 0.05f, 0.05f, 1.0f };
GLfloat g_arrMaterial43Diffuse[] = { 0.5f, 0.5f, 0.5f, 1.0f };
GLfloat g_arrMaterial43Specular[] = { 0.7f, 0.7f, 0.7f, 1.0f };
GLfloat g_Material43Shininess = 0.78125f * 128.0f;

//
//	Materail 53
//
GLfloat g_arrMaterial53Ambient[] = { 0.05f, 0.05f, 0.0f, 1.0f };
GLfloat g_arrMaterial53Diffuse[] = { 0.5f, 0.5f, 0.4f, 1.0f };
GLfloat g_arrMaterial53Specular[] = { 0.7f, 0.7f, 0.04f, 1.0f };
GLfloat g_Material53Shininess = 0.078125f * 128.0f;

//-Material
///////////////////////////////////////////////////////////////////////////

bool g_bAnimate = false;
bool g_bLight = false;
int g_iLightType = 1;	//	1 for vertex light else fragment light.

GLfloat g_fAngleLight = 0.0f;

char chAnimationAxis = 'x';
GLfloat g_fRotateX = 1.0f;
GLfloat g_fRotateY = 0.0f;
GLfloat g_fRotateZ = 0.0f;

int g_iCurrentWidth;
int g_iCurrentHeight;

//	interface decleration
@interface AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@end

@interface GLView : NSOpenGLView	//	Makes your application CGL.
@end

//	Entry point function
int main(int argc, char *argv[])
{
	NSAutoreleasePool *pPool = [[NSAutoreleasePool alloc]init];
	
	NSApp = [NSApplication sharedApplication];
	
	[NSApp setDelegate:[[AppDelegate alloc]init]];
	
	[NSApp run];
	
	[pPool release];
	
	return 0;
}

//	interface implementation
@implementation AppDelegate
{
	@private
			NSWindow *window;
			GLView *glView;
}

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification
{
	//	Log file
	NSBundle *mainBundle = [NSBundle mainBundle];
	NSString *appDirName = [mainBundle bundlePath];
	NSString *parentDirPath = [appDirName stringByDeletingLastPathComponent];
	NSString *logFileNameWithPath = [NSString stringWithFormat:@"%@/Log.txt", parentDirPath];
	const char *pszLogFileNameWithPath = [logFileNameWithPath cStringUsingEncoding:NSASCIIStringEncoding];
	g_fpLogFile = fopen(pszLogFileNameWithPath, "w");
	if (NULL == g_fpLogFile)
	{
		printf("Can not create log file");
		[self release];
		[NSApp terminate:self];
	}
	
	fprintf(g_fpLogFile,"PRogram is started successfully\n");
	
	NSRect win_rect;
	
	win_rect = NSMakeRect(0.0,0.0,800.0,600.0);
	
	//	Create simple window
	window = [[NSWindow alloc] initWithContentRect:win_rect
	styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
	| NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
	backing:NSBackingStoreBuffered defer:NO];
	
	[window setTitle:@"mac OS Window"];
	[window center];
	
	glView = [[GLView alloc]initWithFrame:win_rect];
	
	[window setContentView:glView];
	[window setDelegate:self];
	[window makeKeyAndOrderFront:self];
}

- (void)applicationWillTerminate:(NSNotification *)notification	//	Same as WmDestroy/WmClose
{
	//	Code
	fprintf(g_fpLogFile,"Program is terminated successfully\n");
	if (g_fpLogFile)
	{
		fclose(g_fpLogFile);
		g_fpLogFile = NULL;
	}
}

- (void)windowWillClose:(NSNotification*)notification
{
	//	Code
	[NSApp terminate:self];
}

- (void) dealloc
{
	//	Code
	[glView release];
	
	[window release];
	
	[super dealloc];
}
@end	//	implementation of AppDelegate

@implementation GLView
{
	@private
		CVDisplayLinkRef displayLink;

		GLuint vao;
		GLuint vbo_position;
		GLuint vbo_normal;
		GLuint vbo_texture;
		GLuint vbo_index;

		unsigned short *elements;
		float *verts;
		float *norms;
		float *texCoords;

		unsigned int numElements;
		unsigned int maxElements;
		unsigned int numVertices;

		GLuint g_gluiShaderObjectVertex;
		GLuint g_gluiShaderObjectFragment;
		GLuint g_gluiShaderObjectProgram;


		GLuint g_gluiVAOSphere;
		GLuint g_gluiVBOPosition;
		GLuint g_gluiVBONormal;
		GLuint g_gluiVBOElement;

		/////////////////////////////////////////////////////////////////
		//+Uniforms.
		GLuint g_gluiModelMat4Uniform;
		GLuint g_gluiViewMat4Uniform;
		GLuint g_gluiProjectionMat4Uniform;
		GLuint g_gluiRotationMat4Uniform;

		GLuint g_gluiLKeyPressedUniform;

		GLuint g_gluiLaVec3Uniform;	//	light ambient
		GLuint g_gluiLdVec3Uniform;	//	light diffuse
		GLuint g_gluiLsVec3Uniform;	//	light specular
		GLuint g_gluiLightPositionVec4Uniform;

		GLuint g_gluiKaVec3Uniform;//	Material ambient
		GLuint g_gluiKdVec3Uniform;//	Material diffuse
		GLuint g_gluiKsVec3Uniform;//	Material specular
		GLuint g_gluiMaterialShininessUniform;
		//-Uniforms.
		/////////////////////////////////////////////////////////////////

		vmath::mat4 g_matPerspectiveProjection;

		bool g_bAnimate;
		bool g_bLight;
}

-(id) initWithFrame:(NSRect)frame;
{
	self = [super initWithFrame:frame];
	
	if (!self)
	{
		return(self);
	}
	
	[[self window]setContentView:self];
	
	NSOpenGLPixelFormatAttribute attrs[] = 
	{
		//	Must specify the 4.1 core profile to use openGL 4.1
		NSOpenGLPFAOpenGLProfile,
		NSOpenGLProfileVersion4_1Core,
		NSOpenGLPFAScreenMask, CGDisplayIDToOpenGLDisplayMask(kCGDirectMainDisplay),
		NSOpenGLPFANoRecovery,
		NSOpenGLPFAAccelerated,
		NSOpenGLPFAColorSize, 24,
		NSOpenGLPFADepthSize, 24,
		NSOpenGLPFAAlphaSize, 8,
		NSOpenGLPFADoubleBuffer,
		0};
	
	NSOpenGLPixelFormat *pixelFormat = [[[NSOpenGLPixelFormat alloc]initWithAttributes:attrs] autorelease];	//	Using autorelease, release local allocated OpenGL context automatically.
	
	if (nil == pixelFormat)
	{
		fprintf(g_fpLogFile, "No valid OpenGL pixelFormat is available, Exitting...");
		[self release];
		[NSApp terminate:self];
	}

	NSOpenGLContext *glContext = [[[NSOpenGLContext alloc]initWithFormat:pixelFormat shareContext:nil] autorelease];
	
	[self setPixelFormat:pixelFormat];
	
	[self setOpenGLContext:glContext];	// It automatically releases the older context, if present, and sets the newer one.
 	
	return(self);
}

-(CVReturn)getFrameForTime:(const CVTimeStamp*)pOutputTime
{
	NSAutoreleasePool *pool = [[NSAutoreleasePool alloc]init];
	
	[self drawView];
	
	[pool release];
	
	return(kCVReturnSuccess);
}

-(void)prepareOpenGL
{
	fprintf(g_fpLogFile, "OpenGL version : %s \n", glGetString(GL_VERSION));
	fprintf(g_fpLogFile, "GLSL version : %s \n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	
	g_bAnimate = false;
	g_bLight = false;

	[[self openGLContext]makeCurrentContext];
	
	GLint swapInt = 1;
	[[self openGLContext]setValues:&swapInt forParameter:NSOpenGLCPSwapInterval];
	
	////////////////////////////////////////////////////////////////////
	//+	Shader code

	////////////////////////////////////////////////////////////////////
	//+	Vertex shader.

	fprintf(g_fpLogFile, "==>Vertex Shader.");

	//	Create shader.
	g_gluiShaderObjectVertex = glCreateShader(GL_VERTEX_SHADER);

	//	Provide source code.
	const GLchar *szVertexShaderSourceCode =
		"#version 410 core"							\
		"\n"										\
		"in vec4 vPosition;"						\
		"in vec3 vNormal;"							\
		"uniform mat4 u_model_matrix;"	\
		"uniform mat4 u_view_matrix;"	\
		"uniform mat4 u_projection_matrix;"	\
		"uniform mat4 u_rotation_matrix;"	\
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
		"transformed_normals = normalize(mat3(u_view_matrix * u_model_matrix) * vNormal);"											\
		"vec4 rotated_light_position = u_rotation_matrix * u_light_position;"											\
		"light_direction = normalize(vec3(rotated_light_position) - eyeCoordinates.xyz);"											\
		"viewer_vector = normalize(-eyeCoordinates.xyz);"											\

		"transformed_normals = normalize(transformed_normals);"											\
		"light_direction = normalize(light_direction);"											\
		"viewer_vector = normalize(viewer_vector);"											\
		"}"											\
		"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;"	\
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
			[self release];
			[NSApp terminate:self];
		}

		pszInfoLog = (char*)malloc(gliInfoLogLength);
		if (NULL == pszInfoLog)
		{
			fprintf(g_fpLogFile, "malloc failed.");
			[self release];
			[NSApp terminate:self];
		}

		glGetShaderInfoLog(g_gluiShaderObjectVertex, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		[self release];
		[NSApp terminate:self];
	}
	//-	Vertex shader.
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+	Fragment shader.

	fprintf(g_fpLogFile, "==>Fragment Shader.");

	//	Create shader.
	g_gluiShaderObjectFragment = glCreateShader(GL_FRAGMENT_SHADER);

	//	Provide source code.
	const GLchar *szFragmentShaderSourceCode =
		"#version 410 core"							\
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
		"float tn_dot_ld = max(dot(normalized_transformed_normals, normalized_light_direction), 0.0);"											\
		"vec3 ambient = u_La * u_Ka;"											\
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
			[self release];
			[NSApp terminate:self];
		}

		pszInfoLog = (char*)malloc(gliInfoLogLength);
		if (NULL == pszInfoLog)
		{
			fprintf(g_fpLogFile, "Fragment : malloc failed.");
			[self release];
			[NSApp terminate:self];
		}

		glGetShaderInfoLog(g_gluiShaderObjectFragment, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		[self release];
		[NSApp terminate:self];
	}
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
	glGetShaderiv(g_gluiShaderObjectProgram, GL_LINK_STATUS, &gliLinkStatus);
	if (GL_FALSE == gliLinkStatus)
	{
		glGetProgramiv(g_gluiShaderObjectProgram, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength <= 0)
		{
			fprintf(g_fpLogFile, "Link : GL_INFO_LOG_LENGTH is less than 0.");
			[self release];
			[NSApp terminate:self];
		}

		pszInfoLog = (char*)malloc(gliInfoLogLength);
		if (NULL == pszInfoLog)
		{
			fprintf(g_fpLogFile, "Link : malloc failed.");
			[self release];
			[NSApp terminate:self];
		}

		glGetProgramInfoLog(g_gluiShaderObjectProgram, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		[self release];
		[NSApp terminate:self];
	}
	//-	Shader program.
	////////////////////////////////////////////////////////////////////

	//-	Shader code
	////////////////////////////////////////////////////////////////////

	//
	//	The actual locations assigned to uniform variables are not known until the program object is linked successfully.
	//	After a program object has been linked successfully, the index values for uniform variables remain fixed until the next link command occurs.
	//
	g_gluiModelMat4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_model_matrix");

	g_gluiViewMat4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_view_matrix");

	g_gluiProjectionMat4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_projection_matrix");

	g_gluiRotationMat4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_rotation_matrix");

	g_gluiLKeyPressedUniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_L_key_pressed");

	//	Red Light
	g_gluiLaVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_La");

	g_gluiLdVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Ld");

	g_gluiLsVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Ls");
	
	g_gluiLightPositionVec4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_light_position");

	//	Light Material
	g_gluiKaVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Ka");

	g_gluiKdVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Kd");

	g_gluiKsVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Ks");

	g_gluiMaterialShininessUniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_material_shininess");
	//-	Per Fragment uniform variables.

	// *** vertices, colors, shader attribs, vbo, vao initializations ***
	[self makeSphere:2.0f :30 :30];

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	//+	Change 2 For 3D
	glClearDepth(1.0f);

	glEnable(GL_DEPTH_TEST);

	glDepthFunc(GL_LEQUAL);

	//
	//	We will always cull back faces for better performance.
	//	We will this in case of 3-D rotation/graphics.
	//
	//glEnable(GL_CULL_FACE);

	//-	Change 2 For 3D

	//	See orthographic projection matrix to identity.
	g_matPerspectiveProjection = vmath::mat4::identity();
	
	CVDisplayLinkCreateWithActiveCGDisplays(&displayLink);
	CVDisplayLinkSetOutputCallback(displayLink, &MyDisplayLinkCallback, self);	//	It creates new thread for rendering
	
	CGLContextObj cglContext = (CGLContextObj)[[self openGLContext]CGLContextObj];	//	Typecast requires to work on bit .m and .mm
	CGLPixelFormatObj cglPixelFormat = (CGLPixelFormatObj)[[self pixelFormat]CGLPixelFormatObj];
	CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext(displayLink, cglContext, cglPixelFormat);
	CVDisplayLinkStart(displayLink);	//	Start  thread which created previously.
}

-(void)reshape
{
	CGLLockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);
	
	NSRect rect = [self bounds];
	
	GLfloat width = rect.size.width;
	GLfloat height = rect.size.height;
	
	if (height == 0)
	{
		height = 1;
	}
	
	g_iCurrentHeight = height;
	g_iCurrentWidth = width;
	
	glViewport(0,0,(GLsizei)width,(GLsizei)height);
	
	//	perspective(float fovy, float aspect, float n, float f)
	g_matPerspectiveProjection = vmath::perspective(45, (GLfloat)(width/4) / (GLfloat)(height/6), 0.1f, 100.0f);

	CGLUnlockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);
}


-(void)drawRect:(NSRect)dirtyRect
{
	[self drawView];
}

-(void)drawView
{
	[[self openGLContext]makeCurrentContext];
	
	CGLLockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);	//	Change 3 for 3D

	//	Start using opengl program.
	glUseProgram(g_gluiShaderObjectProgram);

	if (true == g_bLight)
	{
		glUniform1i(g_gluiLKeyPressedUniform, 1);

		glUniform3fv(g_gluiLaVec3Uniform, 1, g_glfarrLightAmbient);	//	Ambient
		glUniform3fv(g_gluiLdVec3Uniform, 1, g_glfarrLightDiffuse);	//	Diffuse
		glUniform3fv(g_gluiLsVec3Uniform, 1, g_glfarrLightSpecular);	//	Specular
	}
	else
	{
		glUniform1i(g_gluiLKeyPressedUniform, 0);
	}

	float fHeightMulti = 0.05f;
	float fWidthMulti = 0.07f;
	
	//
	//	First column,
	//
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere50];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere40];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere30];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere20];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere10];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere00];

	//
	//	Second column.
	//
	fWidthMulti = fWidthMulti + 0.20f;
	fHeightMulti = 0.05f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere51];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere41];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere31];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere21];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere11];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere01];

	//
	//	Third column.
	//
	fWidthMulti = fWidthMulti + 0.20f;
	fHeightMulti = 0.05f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere52];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere42];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere32];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere22];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere12];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere02];

	//
	//	Fourth column.
	//
	fWidthMulti = fWidthMulti + 0.20f;
	fHeightMulti = 0.05f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere53];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere43];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere33];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere23];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere13];

	fHeightMulti = fHeightMulti + 0.15f;
	glViewport((GLint)(g_iCurrentWidth * fWidthMulti), (GLint)(g_iCurrentHeight * fHeightMulti), g_iCurrentWidth / 4, g_iCurrentHeight / 6);
	[self Sphere03];

	//	Stop using opengl program.
	glUseProgram(0);

	[self updateGL];
	
	CGLFlushDrawable((CGLContextObj)[[self openGLContext]CGLContextObj]);
	CGLUnlockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);
}

/////X0

-(void)Sphere00
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial00Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial00Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial00Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material00Shininess);

	[self drawSphere];
}

-(void)Sphere10
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial10Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial10Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial10Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material10Shininess);

	[self drawSphere];
}

-(void)Sphere20
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial20Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial20Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial20Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material20Shininess);

	[self drawSphere];
}

-(void)Sphere30
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial30Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial30Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial30Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material30Shininess);

	[self drawSphere];
}

-(void)Sphere40
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial40Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial40Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial40Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material40Shininess);

	[self drawSphere];
}

-(void)Sphere50
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial50Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial50Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial50Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material50Shininess);

	[self drawSphere];
}

/////X1

-(void)Sphere01
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial01Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial01Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial01Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material01Shininess);

	[self drawSphere];
}

-(void)Sphere11
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial11Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial11Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial11Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material11Shininess);

	[self drawSphere];
}

-(void)Sphere21
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial21Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial21Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial21Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material21Shininess);

	[self drawSphere];
}

-(void)Sphere31
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial31Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial31Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial31Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material31Shininess);

	[self drawSphere];
}

-(void)Sphere41
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial41Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial41Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial41Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material41Shininess);

	[self drawSphere];
}

-(void)Sphere51
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial51Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial51Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial51Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material51Shininess);

	[self drawSphere];
}

/////X1

-(void)Sphere02
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial02Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial02Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial02Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material02Shininess);

	[self drawSphere];
}

-(void)Sphere12
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial12Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial12Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial12Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material12Shininess);

	[self drawSphere];
}

-(void)Sphere22
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial22Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial22Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial22Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material22Shininess);

	[self drawSphere];
}

-(void)Sphere32
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial32Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial32Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial32Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material32Shininess);

	[self drawSphere];
}

-(void)Sphere42
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial42Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial42Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial42Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material42Shininess);

	[self drawSphere];
}

-(void)Sphere52
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial52Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial52Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial52Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material52Shininess);

	[self drawSphere];
}

/////X3

-(void)Sphere03
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial03Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial03Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial03Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material03Shininess);

	[self drawSphere];
}

-(void)Sphere13
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial13Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial13Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial13Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material13Shininess);

	[self drawSphere];
}

-(void)Sphere23
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial23Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial23Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial23Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material23Shininess);

	[self drawSphere];
}

-(void)Sphere33
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial33Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial33Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial33Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material33Shininess);

	[self drawSphere];
}

-(void)Sphere43
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial43Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial43Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial43Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material43Shininess);

	[self drawSphere];
}

-(void)Sphere53
{
	vmath::mat4 matModel;
	vmath::mat4 matView;
	vmath::mat4 matRotation;

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	matModel = vmath::mat4::identity();
	matView = vmath::mat4::identity();
	matRotation = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	if ('x' == chAnimationAxis || 'X' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 1.0f, 0.0f, 0.0f);		//	X-axis rotation
		g_glfarrLightPosition[1] = g_fAngleLight;
		g_glfarrLightPosition[0] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('y' == chAnimationAxis || 'Y' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 1.0f, 0.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else if ('z' == chAnimationAxis || 'Z' == chAnimationAxis)
	{
		matRotation = vmath::rotate(g_fAngleLight, 0.0f, 0.0f, 1.0f);		//	Y-axis rotation
		g_glfarrLightPosition[0] = g_fAngleLight;
		g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);
		glUniformMatrix4fv(g_gluiRotationMat4Uniform, 1, GL_FALSE, matRotation);
	}
	else
	{
		g_glfarrLightPosition[0] = g_glfarrLightPosition[1] = g_glfarrLightPosition[2] = 0.0f;
		g_glfarrLightPosition[2] = 1.0f;
	}

	glUniform3fv(g_gluiKaVec3Uniform, 1, g_arrMaterial53Ambient);
	glUniform3fv(g_gluiKdVec3Uniform, 1, g_arrMaterial53Diffuse);
	glUniform3fv(g_gluiKsVec3Uniform, 1, g_arrMaterial53Specular);
	glUniform1f(g_gluiMaterialShininessUniform, g_Material53Shininess);

	[self drawSphere];
}

-(void)updateGL
{
	g_fAngleLight = g_fAngleLight + 0.1f;
	if (g_fAngleLight >= 720)
		//if (g_fAngleLight >= 360)
	{
		//	Fixme: Search proper way to avoid hitch in light animation.
		g_fAngleLight = 360.0f;
	}
}

-(BOOL)acceptsFirstResponder
{
	//	Code
	[[self window]makeFirstResponder:self];
	return(YES);
}

-(void)keyDown:(NSEvent *)theEvent
{
	int key = (int)[[theEvent characters]characterAtIndex:0];

	switch(key)
	{
		case 27:	//	Esc key
				[self release];
				[NSApp terminate:self];
				break;
		case 'F':
		case 'f':
				[[self window]toggleFullScreen:self];
				break;
		case 'L':
		case 'l':
				if (false == g_bLight)
				{
					g_bLight = true;
				}
				else
				{
					g_bLight = false;
				}
				break;
		case 'A':
		case 'a':
				if (false == g_bAnimate)
				{
					g_bAnimate = true;
				}
				else
				{
					g_bAnimate = false;
				}
				break;
		case 'X':
		case 'x':
		case 'y':
		case 'Y':
		case 'z':
		case 'Z':
			chAnimationAxis = key;
			break;
				
		default:
				break;
	}
}

-(void)mouseDown:(NSEvent *)theEvent
{
	[self setNeedsDisplay:YES];	//	RePainting
}

-(void)mouseDragged:(NSEvent *)theEvent
{
	//	Code
}

-(void)rightMouseDown:(NSEvent *)theEvent
{
	[self setNeedsDisplay:YES];	//	RePainting
}

-(void) dealloc
{
	CVDisplayLinkStop(displayLink);
	CVDisplayLinkRelease(displayLink);

		// destroy vao
	if (vao)
	{
		glDeleteVertexArrays(1, &vao);
		vao = 0;
	}

	// destroy vbo_position
	if (vbo_position)
	{
		glDeleteBuffers(1, &vbo_position);
		vbo_position = 0;
	}
	
	// destroy vbo_normal
	if (vbo_normal)
	{
		glDeleteBuffers(1, &vbo_normal);
		vbo_normal = 0;
	}
	
	// destroy vbo_texture
	if (vbo_texture)
	{
		glDeleteBuffers(1, &vbo_texture);
		vbo_texture = 0;
	}
	
	// destroy vbo_index
	if (vbo_index)
	{
		glDeleteBuffers(1, &vbo_index);
		vbo_index = 0;
	}
	
	[self cleanupMeshData];

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
	
	[super dealloc];
}


//	Sphere
-(void)allocate:(int)numIndices
{
	// code
	// first cleanup, if not initially empty
	// [self cleanupMeshData];
	
	maxElements = numIndices;
	numElements = 0;
	numVertices = 0;
	
	int iNumIndices = numIndices/3;
	
	elements = (unsigned short *)malloc(iNumIndices * 3 * sizeof(unsigned short)); // 3 is x,y,z and 2 is sizeof short
	verts = (float *)malloc(iNumIndices * 3 * sizeof(float)); // 3 is x,y,z and 4 is sizeof float
	norms = (float *)malloc(iNumIndices * 3 * sizeof(float)); // 3 is x,y,z and 4 is sizeof float
	texCoords = (float *)malloc(iNumIndices * 2 * sizeof(float)); // 2 is s,t and 4 is sizeof float
}

// Add 3 vertices, 3 normal and 2 texcoords i.e. one triangle to the geometry.
// This searches the current list for identical vertices (exactly or nearly) and
// if one is found, it is added to the index array.
// if not, it is added to both the index array and the vertex array.
-(void)addTriangle:(float **)single_vertex :(float **)single_normal :(float **)single_texture
{
        const float diff = 0.00001f;
        int i, j;

        // code
        // normals should be of unit length
        [self normalizeVector:single_normal[0]];
        [self normalizeVector:single_normal[1]];
        [self normalizeVector:single_normal[2]];
        
        for (i = 0; i < 3; i++)
        {
            for (j = 0; j < numVertices; j++) //for the first ever iteration of 'j', numVertices will be 0 because of it's initialization in the parameterized constructor
            {
                if ([self isFoundIdentical:verts[j * 3] :single_vertex[i][0] :diff] &&
                    [self isFoundIdentical:verts[(j * 3) + 1] :single_vertex[i][1] :diff] &&
                    [self isFoundIdentical:verts[(j * 3) + 2] :single_vertex[i][2] :diff] &&
                    
                    [self isFoundIdentical:norms[j * 3] :single_normal[i][0] :diff] &&
                    [self isFoundIdentical:norms[(j * 3) + 1] :single_normal[i][1] :diff] &&
                    [self isFoundIdentical:norms[(j * 3) + 2] :single_normal[i][2] :diff] &&
                    
                    [self isFoundIdentical:texCoords[j * 2] :single_texture[i][0] :diff] &&
                    [self isFoundIdentical:texCoords[(j * 2) + 1] :single_texture[i][1] :diff])
                {
                    elements[numElements] = (short)j;
                    numElements++;
                    break;
                }
            }
            
            //If the single vertex, normal and texture do not match with the given, then add the corressponding triangle to the end of the list
            if (j == numVertices && numVertices < maxElements && numElements < maxElements)
            {
                verts[numVertices * 3] = single_vertex[i][0];
                verts[(numVertices * 3) + 1] = single_vertex[i][1];
                verts[(numVertices * 3) + 2] = single_vertex[i][2];

                norms[numVertices * 3] = single_normal[i][0];
                norms[(numVertices * 3) + 1] = single_normal[i][1];
                norms[(numVertices * 3) + 2] = single_normal[i][2];
                
                texCoords[numVertices * 2] = single_texture[i][0];
                texCoords[(numVertices * 2) + 1] = single_texture[i][1];
                
                elements[numElements] = (short)numVertices; //adding the index to the end of the list of elements/indices
                numElements++; //incrementing the 'end' of the list
                numVertices++; //incrementing coun of vertices
            }
        }
}

-(void)prepareToDraw
{
	// vao
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

        // vbo for position
	glGenBuffers(1, &vbo_position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
	glBufferData(GL_ARRAY_BUFFER, (maxElements * 3 * sizeof(float) / 3), verts, GL_STATIC_DRAW);

	glVertexAttribPointer(RTR_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(RTR_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind with vbo_position
        
        // vbo for normals
	glGenBuffers(1, &vbo_normal);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
	glBufferData(GL_ARRAY_BUFFER, (maxElements * 3 * sizeof(float) / 3), norms, GL_STATIC_DRAW);

	glVertexAttribPointer(RTR_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(RTR_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind with vbo_normal
        
        // vbo for texture
	glGenBuffers(1, &vbo_texture);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_texture);
	glBufferData(GL_ARRAY_BUFFER, (maxElements * 2 * sizeof(float) / 3), texCoords, GL_STATIC_DRAW);

	glVertexAttribPointer(RTR_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(RTR_ATTRIBUTE_TEXTURE0);

	glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind with vbo_texture
        
        // vbo for index
	glGenBuffers(1, &vbo_index);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_index);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, (maxElements * 3 * sizeof(unsigned short) / 3), elements, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); // Unbind with vbo_index
        
	glBindVertexArray(0); // Unbind with vao
        
        // after sending data to GPU, now we can free our arrays
        // [self cleanupMeshData];
}

-(void)drawSphere
{
        // code
        // bind vao
	glBindVertexArray(vao);

        // draw
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_index);
	glDrawElements(GL_TRIANGLES, numElements, GL_UNSIGNED_SHORT, 0);

        // unbind vao
	glBindVertexArray(0); // Unbind with vao
}

-(int)getIndexCount
{
        // code
        return(numElements);
}

-(int)getVertexCount
{
        // code
        return(numVertices);
}

-(void)normalizeVector:(float *)v
{
        // code
        
        // square the vector length
        float squaredVectorLength = (v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2]);
        
        // get square root of above 'squared vector length'
        float squareRootOfSquaredVectorLength = (float)sqrt(squaredVectorLength);
        
        // scale the vector with 1/squareRootOfSquaredVectorLength
        v[0] = v[0] * 1.0f/squareRootOfSquaredVectorLength;
        v[1] = v[1] * 1.0f/squareRootOfSquaredVectorLength;
        v[2] = v[2] * 1.0f/squareRootOfSquaredVectorLength;
}

-(bool)isFoundIdentical:(float)val1 :(float)val2 :(float)diff
{
        // code
        if(fabs(val1 - val2) < diff)
            return(true);
        else
            return(false);
}

-(void)cleanupMeshData
{
        // code
        if(elements != NULL)
        {
	    free(elements);
            elements = NULL;
        }
        
        if(verts != NULL)
        {
	    free(verts);
            verts = NULL;
        }
        
        if(norms != NULL)
        {
	    free(norms);
            norms = NULL;
        }
        
        if(texCoords != NULL)
        {
	    free(texCoords);
            texCoords = NULL;
        }
}

-(void)releaseMemory:(float **)vertex :(float **)normal :(float **)texture
{
        for(int a = 0; a < 4; a++)
	{
		free(vertex[a]);
		free(normal[a]);
		free(texture[a]);
	}
	free(vertex);
	free(normal);
	free(texture);
}

-(void)makeSphere:(float)fRadius :(int)iSlices :(int)iStacks
{
    const float VDG_PI = 3.14159265358979323846;

    // code
    float drho = (float)VDG_PI / (float)iStacks;
    float dtheta = 2.0 * (float)VDG_PI / (float)iSlices;
    float ds = 1.0 / (float)(iSlices);
    float dt = 1.0 / (float)(iStacks);
    float t = 1.0;
    float s = 0.0;
    int i = 0;
    int j = 0;
    
    [self allocate:iSlices * iStacks * 6];
    
    for (i = 0; i < iStacks; i++)
    {
        float rho = (float)(i * drho);
        float srho = (float)(sin(rho));
        float crho = (float)(cos(rho));
        float srhodrho = (float)(sin(rho + drho));
        float crhodrho = (float)(cos(rho + drho));
        
        // Many sources of OpenGL sphere drawing code uses a triangle fan
        // for the caps of the sphere. This however introduces texturing
        // artifacts at the poles on some OpenGL implementations
        s = 0.0;
        
        // initialization of three 2-D arrays, two are 4 x 3 and one is 4 x 2
        float **vertex = (float **)malloc(sizeof(float *) * 4); // 4 rows
        for(int a = 0; a < 4; a++)
            vertex[a]= (float *)malloc(sizeof(float) * 3); // 3 columns
        float **normal = (float **)malloc(sizeof(float *) * 4); // 4 rows
        for(int a = 0;a < 4;a++)
            normal[a]= (float *)malloc(sizeof(float) * 3); // 3 columns
        float **texture = (float **)malloc(sizeof(float *) * 4); // 4 rows
        for(int a = 0;a < 4;a++)
            texture[a]= (float *)malloc(sizeof(float) * 2); // 2 columns

        for ( j = 0; j < iSlices; j++)
        {
            float theta = (j == iSlices) ? 0.0 : j * dtheta;
            float stheta = (float)(-sin(theta));
            float ctheta = (float)(cos(theta));
            
            float x = stheta * srho;
            float y = ctheta * srho;
            float z = crho;
           
            texture[0][0] = s;
            texture[0][1] = t;
            normal[0][0] = x;
            normal[0][1] = y;
            normal[0][2] = z;
            vertex[0][0] = x * fRadius;
            vertex[0][1] = y * fRadius;
            vertex[0][2] = z * fRadius;
            
            x = stheta * srhodrho;
            y = ctheta * srhodrho;
            z = crhodrho;
            
            texture[1][0] = s;
            texture[1][1] = t - dt;
            normal[1][0] = x;
            normal[1][1] = y;
            normal[1][2] = z;
            vertex[1][0] = x * fRadius;
            vertex[1][1] = y * fRadius;
            vertex[1][2] = z * fRadius;
            
            theta = ((j+1) == iSlices) ? 0.0 : (j+1) * dtheta;
            stheta = (float)(-sin(theta));
            ctheta = (float)(cos(theta));
            
            x = stheta * srho;
            y = ctheta * srho;
            z = crho;
            
            s += ds;
            texture[2][0] = s;
            texture[2][1] = t;
            normal[2][0] = x;
            normal[2][1] = y;
            normal[2][2] = z;
            vertex[2][0] = x * fRadius;
            vertex[2][1] = y * fRadius;
            vertex[2][2] = z * fRadius;
            
            x = stheta * srhodrho;
            y = ctheta * srhodrho;
            z = crhodrho;
            
            texture[3][0] = s;
            texture[3][1] = t - dt;
            normal[3][0] = x;
            normal[3][1] = y;
            normal[3][2] = z;
            vertex[3][0] = x * fRadius;
            vertex[3][1] = y * fRadius;
            vertex[3][2] = z * fRadius;
		
            [self addTriangle:vertex :normal :texture];
            
            // Rearrange for next triangle
            vertex[0][0]=vertex[1][0];
            vertex[0][1]=vertex[1][1];
            vertex[0][2]=vertex[1][2];
            normal[0][0]=normal[1][0];
            normal[0][1]=normal[1][1];
            normal[0][2]=normal[1][2];
            texture[0][0]=texture[1][0];
            texture[0][1]=texture[1][1];
            
            vertex[1][0]=vertex[3][0];
            vertex[1][1]=vertex[3][1];
            vertex[1][2]=vertex[3][2];
            normal[1][0]=normal[3][0];
            normal[1][1]=normal[3][1];
            normal[1][2]=normal[3][2];
            texture[1][0]=texture[3][0];
            texture[1][1]=texture[3][1];
            
            [self addTriangle:vertex :normal :texture];
        }
        t -= dt;
	[self releaseMemory:vertex :normal :texture];
    }

    [self prepareToDraw];
}
@end

CVReturn MyDisplayLinkCallback(
							CVDisplayLinkRef displayLink,
							const CVTimeStamp* pNow,
							const CVTimeStamp* pOutputTime,
							CVOptionFlags flagsIn,
							CVOptionFlags* pFlagsOut,
							void* pDisplayLinkContext
							)
{
	CVReturn result = [(GLView*)pDisplayLinkContext getFrameForTime:pOutputTime];
	return(result);
}
