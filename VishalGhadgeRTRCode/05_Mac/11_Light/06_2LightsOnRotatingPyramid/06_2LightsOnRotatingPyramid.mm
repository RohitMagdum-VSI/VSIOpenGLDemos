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

GLfloat g_glfAngle = 0.0f;
bool g_bAnimate = false;
bool g_bLight = false;

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

		vmath::mat4 g_matPerspectiveProjection;
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
	
	[[self openGLContext]makeCurrentContext];
	
	GLint swapInt = 1;
	[[self openGLContext]setValues:&swapInt forParameter:NSOpenGLCPSwapInterval];
	
	////////////////////////////////////////////////////////////////////
	//+	Shader code

	////////////////////////////////////////////////////////////////////
	//+	Vertex shader.

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

	//	Create shader.
	g_gluiShaderObjectFragment = glCreateShader(GL_FRAGMENT_SHADER);

	//	Provide source code.
	const GLchar *szFragmentShaderSourceCode =
		"#version 410 core"							\
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

	g_gluiKeyPressedUniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_L_key_pressed");

	g_gluiLaRVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_LaR");

	g_gluiLdRVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_LdR");

	g_gluiLsRVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_LsR");

	g_gluiLightPositionRVec4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_light_positionR");

	g_gluiLaLVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_LaL");
	
	g_gluiLdLVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_LdL");
	
	g_gluiLsLVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_LsL");

	g_gluiLightPositionLVec4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_light_positionL");

	g_gluiKaVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Ka");

	g_gluiKdVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Kd");
	
	g_gluiKsVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Ks");
	
	g_gluiMaterialShininessUniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_material_shininess");

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
	vmath::mat4 matModel = vmath::mat4::identity();
	vmath::mat4 matView = vmath::mat4::identity();
	vmath::mat4 matRotation = vmath::mat4::identity();	//	Good practice to initialize to identity matrix though it will change in next call.

	matModel = vmath::translate(0.0f, 0.0f, -5.0f);
	matRotation = vmath::rotate(g_glfAngle, 0.0f, 1.0f, 0.0f);
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
	
	[self updateGL];
	
	CGLFlushDrawable((CGLContextObj)[[self openGLContext]CGLContextObj]);
	
	CGLUnlockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);
}


-(void)updateGL
{
	g_glfAngle = g_glfAngle + 0.1f;

	if (g_glfAngle >= 360)
	{
		g_glfAngle = 0.0f;
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

		case 'a':
		case 'A':
			if (false == g_bAnimate)
			{
				g_bAnimate = true;
			}
			else
			{
				g_bAnimate = false;
			}
			break;

		case 'l':
		case 'L':
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
	
	[super dealloc];
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
