#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>
#import <QuartzCore/CVDisplayLink.h>
#import <OpenGL/gl3.h>
#import <OpenGL/gl3ext.h>
#import "vmath.h"

enum
{
	HAD_ATTRIBUTE_VERTEX = 0,
	HAD_ATTRIBUTE_COLOR,
	HAD_ATTRIBUTE_NORMAL,
	HAD_ATTRIBUTE_TEXTURE0,
};

// 'C' style global function declarations
CVReturn MyDisplayLinkCallback(CVDisplayLinkRef,const CVTimeStamp *,const CVTimeStamp *,CVOptionFlags,CVOptionFlags *,void *);

//Global Declarations
FILE *gpFile = NULL;

//Interface Declarations
@interface AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@end

@interface GLView : NSOpenGLView
@end

//Entry-point function
int main(int argc, const char * argv[])
{
	NSAutoreleasePool *pPool=[[NSAutoreleasePool alloc]init];

	NSApp=[NSApplication sharedApplication];

	[NSApp setDelegate:[[AppDelegate alloc]init]];

	[NSApp run];

	[pPool release];

	return(0);
}

//Interface Implementations
@implementation AppDelegate
{
@private
	NSWindow *window;
	GLView *glView;
}

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification
{
	//Creating the required path and creating Log File
	NSBundle *mainBundle=[NSBundle mainBundle];
	NSString *appDirName=[mainBundle bundlePath];
	NSString *parentDirPath=[appDirName stringByDeletingLastPathComponent];
	NSString *logFileNameWithPath=[NSString stringWithFormat:@"%@/Log.txt",parentDirPath];
	const char *pszLogFileNameWithPath=[logFileNameWithPath cStringUsingEncoding:NSASCIIStringEncoding];
	gpFile=fopen(pszLogFileNameWithPath,"w");
	if(gpFile==NULL)
	{
		printf("Cannot Create Log File.\nExitting ...\n");
		[self release];
		[NSApp terminate:self];
	}
	fprintf(gpFile,"Program Is Started Successfully\n");

	//Window
	NSRect win_rect;
	win_rect=NSMakeRect(0.0,0.0,800.0,600.0);

	//Create Simple Window
	window =[[NSWindow alloc]initWithContentRect:win_rect 
							 styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable 
							 backing:NSBackingStoreBuffered 
							 defer:NO];

	[window setTitle:@"MacOS : 3D Rotation"];
	[window center];

	glView=[[GLView alloc]initWithFrame:win_rect];

	[window setContentView:glView];
	[window setDelegate:self];
	[window makeKeyAndOrderFront:self];
}

- (void)applicationWillTerminate:(NSNotification *)notification
{
	fprintf(gpFile,"Program Is Terminated Successfully\n");

	if(gpFile)
	{
		fclose(gpFile);
		gpFile=NULL;	
	}
}

- (void)windowWillClose:(NSNotification *)notification
{
	[NSApp terminate:self];
}

- (void)dealloc
{
	[glView release];

	[window release];

	[super dealloc];
}
@end

@implementation GLView
{
@private
	CVDisplayLinkRef displayLink;

	GLuint vertexShaderObject;
	GLuint fragmentShaderObject;
	GLuint shaderProgramObject;

	GLuint vao_pyramid,vao_cube;
	GLuint vbo_Position,vbo_Color;
	GLuint mvpUniform;

	vmath::mat4 perspectiveProjectionMatrix;

	GLfloat gAngle;
}

-(id)initWithFrame:(NSRect)frame;
{
	self=[super initWithFrame:frame];

	if(self)
	{
		[[self window]setContentView:self];

		NSOpenGLPixelFormatAttribute attribs[]=
		{
			NSOpenGLPFAOpenGLProfile,NSOpenGLProfileVersion4_1Core,
			NSOpenGLPFAScreenMask,CGDisplayIDToOpenGLDisplayMask(kCGDirectMainDisplay),
			NSOpenGLPFANoRecovery,
			NSOpenGLPFAAccelerated,
			NSOpenGLPFAColorSize,24,
			NSOpenGLPFADepthSize,24,
			NSOpenGLPFAAlphaSize,8,
			NSOpenGLPFADoubleBuffer,
			0
		};

		NSOpenGLPixelFormat *pixelFormat=[[[NSOpenGLPixelFormat alloc]initWithAttributes:attribs] autorelease];

		if(pixelFormat==nil)
		{
			fprintf(gpFile,"No Valid OpenGL Pixel Format Is Available. Exitting...\n");
			[self release];
			[NSApp terminate:self];
		}

		NSOpenGLContext *glContext=[[[NSOpenGLContext alloc]initWithFormat:pixelFormat shareContext:nil]autorelease];

		[self setPixelFormat:pixelFormat];

		[self setOpenGLContext:glContext];
	}
	return(self);
}

-(CVReturn)getFrameForTime:(const CVTimeStamp *)pOutputTime
{
	NSAutoreleasePool *pool=[[NSAutoreleasePool alloc]init];

	[self drawView];

	[pool release];
	return(kCVReturnSuccess);
}

-(void)prepareOpenGL
{
	fprintf(gpFile,"OpenGL Version : %s\n",glGetString(GL_VERSION));
	fprintf(gpFile,"GLSL Version   : %s\n",glGetString(GL_SHADING_LANGUAGE_VERSION));

	[[self openGLContext]makeCurrentContext];

	GLint swapInt=1;
	[[self openGLContext]setValues:&swapInt forParameter:NSOpenGLCPSwapInterval];

	//Create Shader
	//Vertex Shader
	vertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *vertexShaderSourceCode=
	"#version 410" \
	"\n" \
	"in vec4 vPosition;" \
	"in vec4 vColor;" \
	"uniform mat4 u_mvp_matrix;" \
	"out vec4 out_color;" \
	"void main(void)" \
	"{" \
	"gl_Position = u_mvp_matrix * vPosition;" \
	"out_color = vColor;" \
	"}";

	glShaderSource(vertexShaderObject, 1, (const GLchar **)&vertexShaderSourceCode,NULL);

	glCompileShader(vertexShaderObject);
	GLint iInfoLogLength = 0;
	GLint iShaderCompiledStatus = 0;
	char *szInfoLog = NULL;
	glGetShaderiv(vertexShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if(iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(vertexShaderObject,GL_INFO_LOG_LENGTH,&iInfoLogLength);
		if(iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if(szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(vertexShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile,"Vertex Shader Compilation Log : %s\n",szInfoLog);
				free(szInfoLog);
				[self release];
				[NSApp terminate:self];
			}
		}
	}

	//Fragment Shader
	iInfoLogLength = 0;
	iShaderCompiledStatus = 0;
	szInfoLog = NULL;

	fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *fragmentShaderSourceCode =
	"#version 410" \
	"\n" \
	"in vec4 out_color;" \
	"out vec4 FragColor;" \
	"void main(void)" \
	"{" \
	"FragColor = out_color;" \
	"}";

	glShaderSource(fragmentShaderObject, 1, (const GLchar **)&fragmentShaderSourceCode,NULL);

	glCompileShader(fragmentShaderObject);
	glGetShaderiv(fragmentShaderObject,GL_COMPILE_STATUS,&iShaderCompiledStatus);
	if(iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(fragmentShaderObject,GL_INFO_LOG_LENGTH,&iInfoLogLength);
		if(iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if(szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject,iInfoLogLength,&written,szInfoLog);
				fprintf(gpFile,"Fragment Shader Compilation Log : %s\n",szInfoLog);
				free(szInfoLog);
				[self release];
				[NSApp terminate:self];
			}
		}
	}

	shaderProgramObject = glCreateProgram();

	glAttachShader(shaderProgramObject,vertexShaderObject);

	glAttachShader(shaderProgramObject,fragmentShaderObject);

	glBindAttribLocation(shaderProgramObject, HAD_ATTRIBUTE_VERTEX, "vPosition");

	glBindAttribLocation(shaderProgramObject, HAD_ATTRIBUTE_COLOR, "vColor");

	glLinkProgram(shaderProgramObject);
	GLint iShaderProgramLinkStatus = 0;
	glGetProgramiv(shaderProgramObject,GL_LINK_STATUS,&iShaderProgramLinkStatus);
	if(iShaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(shaderProgramObject,GL_INFO_LOG_LENGTH,&iInfoLogLength);
		if(iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if(szInfoLog != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(shaderProgramObject,iInfoLogLength,&written,szInfoLog);
				fprintf(gpFile,"Shader Program Link Log : %s\n",szInfoLog);
				free(szInfoLog);
				[self release];
				[NSApp terminate:self];
			}
		}
	}

	mvpUniform = glGetUniformLocation(shaderProgramObject,"u_mvp_matrix");

	const GLfloat pyramidVertices[]=
	{
		0.0f,1.0f,0.0f,
		-1.0f,-1.0f,1.0f,
		1.0f,-1.0f,1.0f,

		0.0f,1.0f,0.0f,
		1.0f,-1.0f,1.0f,
		1.0f,-1.0f,-1.0f,

		0.0f,1.0f,0.0f,
		1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,-1.0f,

		0.0f,1.0f,0.0f,
		-1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,1.0f
	};

	const GLfloat pyramidColor[]=
	{
		1.0f,0.0f,0.0f,
		0.0f,1.0f,0.0f,
		0.0f,0.0f,1.0f,

		1.0f,0.0f,0.0f,
		0.0f,0.0f,1.0f,
		0.0f,1.0f,0.0f,

		1.0f,0.0f,0.0f,
		0.0f,1.0f,0.0f,
		0.0f,0.0f,1.0f,

		1.0f,0.0f,0.0f,
		0.0f,0.0f,1.0f,
		0.0f,1.0f,0.0f
	};

	const GLfloat cubeVertices[]=
	{
		1.0f,1.0f,1.0f,
		-1.0f,1.0f,1.0f,
		-1.0f,-1.0f,1.0f,
		1.0f,-1.0f,1.0f,

		1.0f,1.0f,-1.0f,
		1.0f,1.0f,1.0f,
		1.0f,-1.0f,1.0f,
		1.0f,-1.0f,-1.0f,

		-1.0f,1.0f,-1.0f,
		1.0f,1.0f,-1.0f,
		1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,-1.0f,

		-1.0f,1.0f,1.0f,
		-1.0f,1.0f,-1.0f,
		-1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,1.0f,

		1.0f,1.0f,-1.0f,
		-1.0f,1.0f,-1.0f,
		-1.0f,1.0f,1.0f,
		1.0f,1.0f,1.0f,

		1.0f,-1.0f,1.0f,
		-1.0f,-1.0f,1.0f,
		-1.0f,-1.0f,-1.0f,
		1.0f,-1.0f,-1.0f
	};

	const GLfloat cubeColor[] =
	{
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,

		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,

		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,

		1.0f, 0.0f, 1.0f,
		1.0f, 0.0f, 1.0f,
		1.0f, 0.0f, 1.0f,
		1.0f, 0.0f, 1.0f,

		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,

		0.0f, 1.0f, 1.0f,
		0.0f, 1.0f, 1.0f,
		0.0f, 1.0f, 1.0f,
		0.0f, 1.0f, 1.0f
	};

	glGenVertexArrays(1,&vao_pyramid);
	glBindVertexArray(vao_pyramid);

	glGenBuffers(1,&vbo_Position);
	glBindBuffer(GL_ARRAY_BUFFER,vbo_Position);
	glBufferData(GL_ARRAY_BUFFER,sizeof(pyramidVertices),pyramidVertices,GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_VERTEX,3,GL_FLOAT,GL_FALSE,0,NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_VERTEX);

	glBindBuffer(GL_ARRAY_BUFFER,0);

	glGenBuffers(1,&vbo_Color);
	glBindBuffer(GL_ARRAY_BUFFER,vbo_Color);
	glBufferData(GL_ARRAY_BUFFER,sizeof(pyramidColor),pyramidColor,GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_COLOR,3,GL_FLOAT,GL_FALSE,0,NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_COLOR);

	glBindBuffer(GL_ARRAY_BUFFER,0);
	glBindVertexArray(0);

	glGenVertexArrays(1,&vao_cube);
	glBindVertexArray(vao_cube);

	glGenBuffers(1,&vbo_Position);
	glBindBuffer(GL_ARRAY_BUFFER,vbo_Position);
	glBufferData(GL_ARRAY_BUFFER,sizeof(cubeVertices),cubeVertices,GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_VERTEX,3,GL_FLOAT,GL_FALSE,0,NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_VERTEX);

	glBindBuffer(GL_ARRAY_BUFFER,0);

	glGenBuffers(1,&vbo_Color);
	glBindBuffer(GL_ARRAY_BUFFER,vbo_Color);
	glBufferData(GL_ARRAY_BUFFER,sizeof(cubeColor),cubeColor,GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_COLOR,3,GL_FLOAT,GL_FALSE,0,NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_COLOR);

	glBindBuffer(GL_ARRAY_BUFFER,0);

	glBindVertexArray(0);

	glClearDepth(1.0f);

	glEnable(GL_DEPTH_TEST);

	glDepthFunc(GL_LEQUAL);

	glClearColor(1.0f,1.0f,1.0f,0.0f);

	perspectiveProjectionMatrix = vmath::mat4::identity();

	CVDisplayLinkCreateWithActiveCGDisplays(&displayLink);
	CVDisplayLinkSetOutputCallback(displayLink,&MyDisplayLinkCallback,self);
	CGLContextObj cglContext = (CGLContextObj)[[self openGLContext]CGLContextObj];
	CGLPixelFormatObj cglPixelFormat = (CGLPixelFormatObj)[[self pixelFormat]CGLPixelFormatObj];
	CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext(displayLink,cglContext,cglPixelFormat);
	CVDisplayLinkStart(displayLink);
}

-(void)reshape
{
	CGLLockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);

	NSRect rect = [self bounds];

	GLfloat width=rect.size.width;
	GLfloat height=rect.size.height;

	if(height == 0)
		height=1;

	glViewport(0,0,(GLsizei)width,(GLsizei)height);

	perspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);

	CGLUnlockContext((CGLContextObj)[[self openGLContext] CGLContextObj]);	
}

- (void)drawRect:(NSRect)dirtyRect
{
	[self drawView];
}

- (void)drawView
{
	[[self openGLContext]makeCurrentContext];

	CGLLockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(shaderProgramObject);

	vmath::mat4 modelViewMatrix = vmath::mat4::identity();
	vmath::mat4 modelViewProjectionMatrix = vmath::mat4::identity();
	vmath::mat4 rotationMatrix = vmath::mat4::identity();
	vmath::mat4 scaleMatrix  = vmath::mat4::identity();

	modelViewMatrix = vmath::translate(-1.5f,0.0f,-6.0f);

	rotationMatrix = vmath::rotate(gAngle,0.0f,1.0f,0.0f);

	modelViewMatrix = modelViewMatrix * rotationMatrix;

	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(mvpUniform,1,GL_FALSE,modelViewProjectionMatrix);

	glBindVertexArray(vao_pyramid);

	glDrawArrays(GL_TRIANGLES,0,12);

	glBindVertexArray(0);

	modelViewMatrix = vmath::mat4::identity();

	modelViewProjectionMatrix = vmath::mat4::identity();

	rotationMatrix = vmath::mat4::identity();

	modelViewMatrix = vmath::translate(1.5f,0.0f,-6.0f);

	rotationMatrix = vmath::rotate(gAngle,gAngle,gAngle);

	scaleMatrix = vmath::scale(0.75f,0.75f,0.75f);

	modelViewMatrix = modelViewMatrix * scaleMatrix;

	modelViewMatrix = modelViewMatrix * rotationMatrix;

	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(mvpUniform,1,GL_FALSE,modelViewProjectionMatrix);

	glBindVertexArray(vao_cube);

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 20, 4);

	glBindVertexArray(0);

	glUseProgram(0);

	CGLFlushDrawable((CGLContextObj)[[self openGLContext]CGLContextObj]);
	CGLUnlockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);

	gAngle = gAngle + 0.8f;
	if(gAngle >= 360.0f)
		gAngle = gAngle - 360.0f;		
}

-(BOOL)acceptsFirstResponder
{
	[[self window]makeFirstResponder:self];

	return(YES);
}

-(void)keyDown:(NSEvent *)theEvent
{
	int key=(int)[[theEvent characters]characterAtIndex:0];
	switch(key)
	{
		case 27:
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

}

-(void)mouseDragged:(NSEvent *)theEvent
{

}

-(void)rightMouseDown:(NSEvent *)theEvent
{

}

- (void) dealloc
{
	if(vao_pyramid)
	{
		glDeleteVertexArrays(1,&vao_pyramid);
		vao_pyramid=0;
	}

	if(vao_cube)
	{
		glDeleteVertexArrays(1,&vao_cube);
		vao_cube=0;
	}

	if(vbo_Position)
	{
		glDeleteBuffers(1,&vbo_Position);
		vbo_Position=0;
	}

	if(vbo_Color)
	{
		glDeleteBuffers(1,&vbo_Color);
		vbo_Color=0;
	}

	if(vertexShaderObject)
	{
		glDetachShader(shaderProgramObject,vertexShaderObject);
		glDeleteShader(vertexShaderObject);
		vertexShaderObject=0;
	}

	if(fragmentShaderObject)
	{
		glDetachShader(shaderProgramObject,fragmentShaderObject);
		glDeleteShader(fragmentShaderObject);
		fragmentShaderObject=0;
	}
	
	if(shaderProgramObject)
	{
		glDeleteProgram(shaderProgramObject);
		shaderProgramObject=0;
	}

	CVDisplayLinkStop(displayLink);
	CVDisplayLinkRelease(displayLink);

	[super dealloc];
}
@end

CVReturn MyDisplayLinkCallback(CVDisplayLinkRef displayLink,const CVTimeStamp *pNow,const CVTimeStamp *pOutputTime,CVOptionFlags flagsIn,
								CVOptionFlags *pFlagsOut,void *pDisplayLinkContext)
{
	CVReturn result=[(GLView *)pDisplayLinkContext getFrameForTime:pOutputTime];
	return(result);
}