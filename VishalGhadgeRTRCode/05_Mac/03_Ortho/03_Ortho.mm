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

		GLuint g_gluiVAO;
		GLuint g_gluiVBO;
		GLint g_gliMVPUniform;

		vmath::mat4 g_matOrthoGraphicProjection;
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
		"uniform mat4 u_mvp_matrix;"				\
		"void main(void)"							\
		"{"											\
		"gl_Position = u_mvp_matrix * vPosition;"	\
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
		"out vec4 vFragColor;"						\
		"void main(void)"							\
		"{"											\
		"vFragColor = vec4(1.0,1.0,1.0,1.0);"	\
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
	g_gliMVPUniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_mvp_matrix");
	if (-1 == g_gliMVPUniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation failed.");
		[self release];
		[NSApp terminate:self];
	}

	////////////////////////////////////////////////////////////////////
	//+	Vertices,color, shader attribute, vbo,vao initialization.

	const GLfloat glfarrTriangleVertices[] =
	{
		0.0f, 50.0f, 0.0f,	//	apex
		-50.0f, -50.0f, 0.0f,	//	left_bottom
		50.0f, -50.0f, 0.0f,	//	right_bottom
	};

	glGenVertexArrays(1, &g_gluiVAO);	//	It is like recorder.
	glBindVertexArray(g_gluiVAO);

	glGenBuffers(1, &g_gluiVBO);
	glBindBuffer(GL_ARRAY_BUFFER, g_gluiVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glfarrTriangleVertices), glfarrTriangleVertices, GL_STATIC_DRAW);

	glVertexAttribPointer(RTR_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(RTR_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//-	Vertices,color, shader attribute, vbo,vao initialization.
	////////////////////////////////////////////////////////////////////

	glClearColor(0.0f, 0.0f, 1.0f, 0.0f);

	//+	Change 2 For 3D
	glClearDepth(1.0f);

	glEnable(GL_DEPTH_TEST);

	glDepthFunc(GL_LEQUAL);

	//
	//	Optional.
	//
	//glShadeModel(GL_SMOOTH);
	//glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	//
	//	We will always cull back faces for better performance.
	//	We will this in case of 3-D rotation/graphics.
	//
	glEnable(GL_CULL_FACE);

	//-	Change 2 For 3D

	//	See orthographic projection matrix to identity.
	g_matOrthoGraphicProjection = vmath::mat4::identity();
	
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
	
	//	glOrtho(left,right,bottom,top,near.far)
	if (width <= height)
	{
		g_matOrthoGraphicProjection = vmath::ortho(-100.0f, 100.0f, -100.0f * (height/width), 100.0f * (height/width), -100.0f, 100.0f);
	}
	else
	{
		g_matOrthoGraphicProjection = vmath::ortho(-100.0f, 100.0f, -100.0f * (width/height), 100.0f * (width/height), -100.0f, 100.0f);
	}
	
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

	//	OpenGl drawing.
	//	Set model-view and modelviewprojection matrix to identity.
	vmath::mat4 matModelView = vmath::mat4::identity();
	vmath::mat4 matModelViewProjection = vmath::mat4::identity();	//	Good practice to initialize to identity matrix though it will change in next call.

	//	Multiply the modelview and orthographic projection matrix to get modelviewprojection matrix.
	//	Order is very important.
	matModelViewProjection = g_matOrthoGraphicProjection * matModelView;

	//
	//	Pass above modelviewprojection matrix to the vertex shader in 'u_mvp_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gliMVPUniform, 1, GL_FALSE, matModelViewProjection);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAO);

	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glDrawArrays(GL_TRIANGLES, 0, 3); //	3 - each with its x,y,z vertices in triangle vertices array.

	//	Unbind 'VAO'
	glBindVertexArray(0);

	//	Stop using opengl program.
	glUseProgram(0);
	
	CGLFlushDrawable((CGLContextObj)[[self openGLContext]CGLContextObj]);
	
	CGLUnlockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);
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

	if (g_gluiVBO)
	{
		glDeleteBuffers(1, &g_gluiVBO);
		g_gluiVBO = 0;
	}

	if (g_gluiVAO)
	{
		glDeleteVertexArrays(1, &g_gluiVAO);
		g_gluiVAO = 0;
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