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

GLfloat g_glfarrLightAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_glfarrLightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfarrLightSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfarrLightPosition[] = { 100.0f, 100.0f, 100.0f, 1.0f };

GLfloat g_glfarrMaterialAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_glfarrMaterialDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfarrMaterialSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfMaterialShininess = 50.0f;

int g_iLightType = 1;	//	1 for vertex light else fragment light.
int UNIFORM_INDEX_PER_VERTEX = 0;
int UNIFORM_INDEX_PER_FRAGMENT = 1;

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
		//#define UNIFORM_INDEX_PER_VERTEX	0
		//#define UNIFORM_INDEX_PER_FRAGMENT	1

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
	//+	Vertex shader Per vertex Light.

	fprintf(g_fpLogFile, "==>Vertex Shader-Per vertex light.");

	//	Create shader.
	g_gluiShaderObjectVertexPerVertexLight = glCreateShader(GL_VERTEX_SHADER);

	//	Provide source code.
	//	Provide source code.
	const GLchar *szVertexShaderSourceCodePerVertexLight =
		"#version 410 core"							\
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

		glGetShaderInfoLog(g_gluiShaderObjectVertexPerVertexLight, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		[self release];
		[NSApp terminate:self];
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
		"#version 410 core"							\
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

		glGetShaderInfoLog(g_gluiShaderObjectVertexPerFragmentLight, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		[self release];
		[NSApp terminate:self];
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
		"#version 410 core"							\
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

		glGetShaderInfoLog(g_gluiShaderObjectFragmentPerVertexLight, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		[self release];
		[NSApp terminate:self];
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

		glGetShaderInfoLog(g_gluiShaderObjectFragmentPerFragmentLight, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		[self release];
		[NSApp terminate:self];
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

		glGetProgramInfoLog(g_gluiShaderObjectProgramPerVertexLight, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		[self release];
		[NSApp terminate:self];
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

		glGetProgramInfoLog(g_gluiShaderObjectProgramPerFragmentLight, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		[self release];
		[NSApp terminate:self];
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

	g_gluiViewMat4Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_view_matrix");

	g_gluiProjectionMat4Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_projection_matrix");

	g_gluiLKeyPressedUniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_L_key_pressed");

	g_gluiLaVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_La");
	
	g_gluiLdVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_Ld");
	
	g_gluiLsVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_Ls");

	g_gluiLightPositionVec4Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_light_position");

	g_gluiKaVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_Ka");

	g_gluiKdVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_Kd");
	
	g_gluiKsVec3Uniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_Ks");

	g_gluiMaterialShininessUniform[UNIFORM_INDEX_PER_VERTEX] = glGetUniformLocation(g_gluiShaderObjectProgramPerVertexLight, "u_material_shininess");
	//-	Per Vertex uniform variables.

	//+	Per Fragment uniform variables.
	g_gluiModelMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_model_matrix");

	g_gluiViewMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_view_matrix");

	g_gluiProjectionMat4Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_projection_matrix");

	g_gluiLKeyPressedUniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_L_key_pressed");

	g_gluiLaVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_La");

	g_gluiLdVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_Ld");

	g_gluiLsVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_Ls");

	g_gluiLightPositionVec4Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_light_position");

	g_gluiKaVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_Ka");

	g_gluiKdVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_Kd");

	g_gluiKsVec3Uniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_Ks");

	g_gluiMaterialShininessUniform[UNIFORM_INDEX_PER_FRAGMENT] = glGetUniformLocation(g_gluiShaderObjectProgramPerFragmentLight, "u_material_shininess");
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
	
	glViewport(0,0,(GLsizei)width,(GLsizei)height);
	
		//	perspective(float fovy, float aspect, float n, float f)
	g_matPerspectiveProjection = vmath::perspective(45, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);

	
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
	vmath::mat4 matModel = vmath::mat4::identity();
	vmath::mat4 matView = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform[index], 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform[index], 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform[index], 1, GL_FALSE, g_matPerspectiveProjection);

	[self drawSphere];

	//	Stop using opengl program.
	glUseProgram(0);

	CGLFlushDrawable((CGLContextObj)[[self openGLContext]CGLContextObj]);
	CGLUnlockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);
}


-(void)updateGL
{
	
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

		case 's':
		case 'S':
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
