#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>

#import <QuartzCore/CVDisplayLink.h>

#import <OpenGL/gl3.h>
#import <OpenGL/gl3ext.h>

#import "vmath.h"


CVReturn MyDisplayLinkCallback(CVDisplayLinkRef, const CVTimeStamp*, const CVTimeStamp*, CVOptionFlags, CVOptionFlags*, void*);


enum {
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOORD0,
};


FILE *gbFile = NULL;


@interface AppDelegate:NSObject < NSApplicationDelegate, NSWindowDelegate>
@end


@interface GLView:NSOpenGLView
@end


int main(int argc, const char *argv[]){
	
	NSAutoreleasePool *pPool = [[NSAutoreleasePool alloc] init];

	NSApp = [NSApplication sharedApplication];

	[NSApp setDelegate:[[AppDelegate alloc] init]];

	[NSApp run];

	[pPool release];

	return(0);
}


/********** AppDelegate **********/

@implementation AppDelegate
{
	@private
		NSWindow *window;
		GLView *glView;
}


-(void)applicationDidFinishLaunching:(NSNotification*)aNotification{
	

	NSBundle *mainBundle = [NSBundle mainBundle];
	NSString *appDirName = [mainBundle bundlePath];
	NSString *parentDirPath = [appDirName stringByDeletingLastPathComponent];
	NSString *logFileNameWithPath = [NSString stringWithFormat: @"%@/Log.txt", parentDirPath];
	const char *logFileName = [logFileNameWithPath cStringUsingEncoding:NSASCIIStringEncoding];
	
	gbFile = fopen(logFileName, "w");
	if(gbFile == NULL){
		printf("Log Creation Failed!!\n");
		[self release];
		[NSApp terminate:self];
	}
	else
		fprintf(gbFile, "Log Created!!\n");

	NSRect win_rect;
	win_rect = NSMakeRect(0.0, 0.0, 800.0, 600.0);

	window = [[NSWindow alloc] initWithContentRect: win_rect styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable| NSWindowStyleMaskMiniaturizable |
        NSWindowStyleMaskResizable
        backing: NSBackingStoreBuffered
        defer: NO];

	[window setTitle: @"03-Ortho"];
	[window center];

	glView = [[GLView alloc] initWithFrame: win_rect];

	[window setContentView: glView];
	[window setDelegate: self];
	[window makeKeyAndOrderFront: self];
}


-(void)applicationWillTerminate:(NSNotification*)notification {
	fprintf(gbFile, "Program is Terminate SuccessFully!!\n");

	if(gbFile){
        fprintf(gbFile, "Log Is Close!!\n");
		fclose(gbFile);
		gbFile = NULL;
	}
}


-(void)windowWillClose:(NSNotification*)notification {
	[NSApp terminate: self];
}


-(void) dealloc {
	[glView release];

	[window release];

	[super dealloc];
}

@end


/********** GLView **********/
@implementation GLView
{
	@private
		CVDisplayLinkRef displayLink;

		GLuint vertexShaderObject;
		GLuint fragmentShaderObject;
		GLuint shaderProgramObject;

		GLuint vao_Triangle;
		GLuint vbo_Triangle_Position;

		GLuint mvpUniform;

		vmath::mat4 orthographicProjectionMatrix;

}


-(id)initWithFrame:(NSRect)frame {
	
	self = [super initWithFrame: frame];

	if(self){
		
		[[self window] setContentView: self];


		NSOpenGLPixelFormatAttribute attribs[] = {
			NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion4_1Core,

			NSOpenGLPFAScreenMask, CGDisplayIDToOpenGLDisplayMask(kCGDirectMainDisplay),

			NSOpenGLPFAAccelerated,

			NSOpenGLPFANoRecovery,

			NSOpenGLPFAColorSize, 24,
			NSOpenGLPFADepthSize, 24,
			NSOpenGLPFAAlphaSize, 8,
			NSOpenGLPFADoubleBuffer,
			0
		};

		NSOpenGLPixelFormat *pixelFormat = [[[NSOpenGLPixelFormat alloc] initWithAttributes: attribs] autorelease];

		if(pixelFormat == nil){
			fprintf(gbFile, "No Valid OpenGL PixelFormat !!\n");
			[self release];
			[NSApp terminate:self];
		}

		NSOpenGLContext *glContext = [[[NSOpenGLContext alloc] initWithFormat: pixelFormat shareContext: nil]autorelease];

		[self setPixelFormat: pixelFormat];

		[self setOpenGLContext: glContext];
	}
	return(self);
}



-(CVReturn)getFrameForTime: (const CVTimeStamp*)pOutputTime {
	
	NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

	//Display
	[self drawView];

	[pool release];

	return(kCVReturnSuccess);

}

-(void) prepareOpenGL {
	
	fprintf(gbFile, "OpenGL Version : %s\n", glGetString(GL_VERSION));
	fprintf(gbFile, "OpenGL Shading Language Version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	[[self openGLContext] makeCurrentContext];

	GLint swapInt = 1;

	[[self openGLContext]setValues: &swapInt forParameter: NSOpenGLCPSwapInterval];
    

	/********** Vertex Shader **********/
	vertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
	const GLchar *vertexShaderSourceCode = 
		"#version 410" \
		"\n" \
		"in vec4 vPosition;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void) {" \
			"gl_Position = u_mvp_matrix * vPosition;" \
		"}";

	glShaderSource(vertexShaderObject, 1, (const GLchar**)&vertexShaderSourceCode, NULL);

	glCompileShader(vertexShaderObject);

	GLint shaderCompileStatus = 0;
	GLint iInfoLogLength = 0;
	char *szInfoLog = NULL;

	glGetShaderiv(vertexShaderObject, GL_COMPILE_STATUS, &shaderCompileStatus);
	if(shaderCompileStatus == GL_FALSE){
		glGetShaderiv(vertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if(iInfoLogLength > 0){
			szInfoLog = (char*)malloc(sizeof(char) * iInfoLogLength);
			if(szInfoLog){
				GLsizei written;
				glGetShaderInfoLog(vertexShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gbFile, "Vertex Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				[self release];
				[NSApp terminate: self];
			}
		}
	}



	/********** Fragment Shader **********/
	fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);
	const GLchar *fragmentShaderSourceCode = 
		"#version 410" \
		"\n" \
		"out vec4 FragColor;" \
		"void main(void) {" \
			"FragColor = vec4(1.0, 1.0, 0.0, 1.0);" \
		"}";

	glShaderSource(fragmentShaderObject, 1, (const GLchar**)&fragmentShaderSourceCode, NULL);
	glCompileShader(fragmentShaderObject);

	shaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(fragmentShaderObject, GL_COMPILE_STATUS, &shaderCompileStatus);
	if(shaderCompileStatus == GL_FALSE){
		glGetShaderiv(fragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if(iInfoLogLength > 0){
			szInfoLog = (char*)malloc(sizeof(char) * iInfoLogLength);
			if(szInfoLog){
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gbFile, "Fragment Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				[self release];
				[NSApp terminate: self];
			}
		}
	}


	/********** Shader Program **********/
	shaderProgramObject = glCreateProgram();

	glAttachShader(shaderProgramObject, vertexShaderObject);
	glAttachShader(shaderProgramObject, fragmentShaderObject);

	glBindAttribLocation(shaderProgramObject, AMC_ATTRIBUTE_POSITION, "vPosition");

	glLinkProgram(shaderProgramObject);

	GLint iProgramLinkStatus;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetProgramiv(shaderProgramObject, GL_LINK_STATUS, &iProgramLinkStatus);
	if(iProgramLinkStatus == GL_FALSE){
		glGetProgramiv(shaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if(iInfoLogLength > 0){
			szInfoLog = (char*)malloc(sizeof(char) * iInfoLogLength);
			if(szInfoLog){
				GLsizei written;
				glGetProgramInfoLog(shaderProgramObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gbFile, "Shader Program Linking Error: %s\n", szInfoLog);
				szInfoLog = NULL;
				[self release];
				[NSApp terminate: self];
			}
		}
	}


	mvpUniform = glGetUniformLocation(shaderProgramObject, "u_mvp_matrix");


	/********** Triangle Position, vao, vbo **********/
	GLfloat tri_Position[] = {
		0.0f, 50.0f, 0.0,
		-50.0f, -50.0f, 0.0f,
		50.0f, -50.0f, 0.0f,
	};

	
	
	glGenVertexArrays(1, &vao_Triangle);
	glBindVertexArray(vao_Triangle);

		/********** Position **********/
		glGenBuffers(1, &vbo_Triangle_Position);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Triangle_Position);
		glBufferData(GL_ARRAY_BUFFER, sizeof(tri_Position), tri_Position, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);


	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	glClearColor(0.0f, 0.0f, 1.0f, 0.0f);

	orthographicProjectionMatrix = vmath::mat4::identity();

	CVDisplayLinkCreateWithActiveCGDisplays(&displayLink);
	CVDisplayLinkSetOutputCallback(displayLink, &MyDisplayLinkCallback, self);
	CGLContextObj cglContext = (CGLContextObj)[[self openGLContext]CGLContextObj];
	CGLPixelFormatObj cglPixelFormat = (CGLPixelFormatObj)[[self pixelFormat]CGLPixelFormatObj];
	CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext(displayLink, cglContext, cglPixelFormat);
	CVDisplayLinkStart(displayLink);
}


-(void)reshape {
	
	CGLLockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);

	NSRect rect = [self bounds];
	GLfloat width = rect.size.width;
	GLfloat height = rect.size.height;

	if(height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	if(width >= height){
		orthographicProjectionMatrix = vmath::ortho((-100.0f * (width / height)),
								(100.0f * (width / height)),
								-100.0f, 100.0f,
								-100.0f, 100.0f
								);
	}
	else{
		orthographicProjectionMatrix = vmath::ortho(-100.0f, 100.0f,
								(-100.0f * (height / width)),
								(100.0f * (height / width)),
								-100.0f, 100.0f);
	}


	CGLUnlockContext((CGLContextObj)[[self openGLContext] CGLContextObj]);
}


-(void)drawRect:(NSRect)rect {
	[self drawView];
}


-(void) drawView {
	
	[[self openGLContext]makeCurrentContext];

	CGLLockContext((CGLContextObj) [[self openGLContext] CGLContextObj]);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(shaderProgramObject);

	vmath::mat4 modelViewMatrix = vmath::mat4::identity();
	vmath::mat4 modelViewProjectionMatrix = vmath::mat4::identity();

	modelViewProjectionMatrix = orthographicProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(mvpUniform, 1, false, modelViewProjectionMatrix);

	glBindVertexArray(vao_Triangle);
		glDrawArrays(GL_TRIANGLES, 0, 3);
	glBindVertexArray(0);

	glUseProgram(0);


	CGLFlushDrawable((CGLContextObj) [[self openGLContext] CGLContextObj]);
	CGLUnlockContext((CGLContextObj) [[self openGLContext] CGLContextObj]);
}


-(BOOL) acceptsFirstResponder {
	[[self window]makeFirstResponder: self];
	return(YES);
}

-(void) keyDown: (NSEvent*) event {

	int key = (int)[[event characters]characterAtIndex: 0];
	switch(key){
		case 27:
			[self release];
			[NSApp terminate: self];
			break;

		case 'F':
		case 'f':
			[[self window]toggleFullScreen: self];
			break;

		default:
			break;
	}
}

-(void) mouseDown: (NSEvent*) event{
	
}

-(void) mouseDragged: (NSEvent*) event{

}

-(void) rightMouseDown: (NSEvent*) event{

}

-(void) dealloc {
	
	if(vbo_Triangle_Position){
		glDeleteBuffers(1, &vbo_Triangle_Position);
		vbo_Triangle_Position = 0;
	}

	if(vao_Triangle){
		glDeleteVertexArrays(1, &vao_Triangle);
		vao_Triangle = 0;
	}

	
	GLsizei shaderCount = 0;
	GLuint shaderNo = 0;

	glUseProgram(shaderProgramObject);

	glGetProgramiv(shaderProgramObject, GL_ATTACHED_SHADERS, &shaderCount);
	fprintf(gbFile, "Shader Count: %d\n",  shaderCount);
	GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * shaderCount);
	if(pShader){

		glGetAttachedShaders(shaderProgramObject, shaderCount, &shaderCount, pShader);
		for(shaderNo = 0; shaderNo < shaderCount; shaderNo++){
			glDetachShader(shaderProgramObject, pShader[shaderNo]);
			glDeleteShader(pShader[shaderNo]);
			pShader[shaderNo] = 0;
		}
		free(pShader);
		pShader = NULL;
	}

	glUseProgram(0);
	glDeleteProgram(shaderProgramObject);
	shaderProgramObject = 0;


	CVDisplayLinkStop(displayLink);
	CVDisplayLinkRelease(displayLink);

	[super dealloc];
}

@end 

CVReturn MyDisplayLinkCallback(CVDisplayLinkRef displayLink, const CVTimeStamp *pNow, const CVTimeStamp *pOutputTime, CVOptionFlags flagsIn, CVOptionFlags *pFlagsOut, void *pDisplayLinkContext){

	CVReturn result = [(GLView*)pDisplayLinkContext getFrameForTime: pOutputTime];
	return(result);
}



