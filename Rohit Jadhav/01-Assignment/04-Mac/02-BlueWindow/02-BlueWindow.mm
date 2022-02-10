#import<Foundation/Foundation.h>
#import<Cocoa/Cocoa.h>

#import<QuartzCore/CVDisplayLink.h>

#import<OpenGL/gl3.h>
#import<OpenGL/gl3ext.h>



CVReturn MyDisplayLinkCallback(CVDisplayLinkRef, const CVTimeStamp*, const CVTimeStamp*, CVOptionFlags, CVOptionFlags*, void*);


//For Error
FILE *gbFile = NULL;

@interface AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@end

@interface GLView : NSOpenGLView
@end


int main(int argc, const char* argv[]){

	NSAutoreleasePool *pPool = [[NSAutoreleasePool alloc]init];

	NSApp=[NSApplication sharedApplication];

	[NSApp setDelegate: [[AppDelegate alloc]init]];

	[NSApp run];

	[pPool release];

	return(0);
}



/******************** AppDelgate ********************/
@implementation AppDelegate{
	@private
		NSWindow *window;
		GLView *glView;
}


//WM_CREATE
-(void)applicationDidFinishLaunching:(NSNotification*)notification{


	NSBundle *mainBundle = [NSBundle mainBundle];
	NSString *appDirName = [mainBundle bundlePath];
	NSString *parentDirPath = [appDirName stringByDeletingLastPathComponent];
	NSString *logFileNameWithPath = [NSString stringWithFormat:@"%@/Log.txt", parentDirPath];
	const char *pzLogFileNameWithPath = [logFileNameWithPath cStringUsingEncoding:NSASCIIStringEncoding];
	gbFile = fopen(pzLogFileNameWithPath, "w");
	if(gbFile == NULL){
		printf("Log Creation Failed!!\n");
		[self release];
		[NSApp terminate:self];
	}
	fprintf(gbFile, "Log Created!!\n");
   // fclose(gbFile);

	NSRect win_rect;
	win_rect = NSMakeRect(0.0, 0.0, 800.0, 600.0);

	window = [[NSWindow alloc]initWithContentRect:win_rect styleMask:NSWindowStyleMaskClosable | NSWindowStyleMaskTitled | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable backing:NSBackingStoreBuffered defer:NO];

	[window setTitle:@"02-BlueScreen-RohitRJadhav"];
	[window center];

	glView = [[GLView alloc]initWithFrame:win_rect];

	[window setContentView:glView];
	[window setDelegate:self];
	[window makeKeyAndOrderFront:self];
}

//WM_DESTROY
-(void)applicationWillTerminate:(NSNotification*)notification{

	fprintf(gbFile, "Program is Terminated Successfully!!\n");

	if(gbFile){
		fprintf(gbFile, "Log Close!!\n");
		fclose(gbFile);
		gbFile = NULL;
	}
}


//WM_CLOSE
-(void)windowWillClose:(NSNotification*)notification{
	[NSApp terminate:self];
}

-(void)dealloc{
    [glView release];
    
    [window release];
    
    [super dealloc];
}


@end



/******************** GLView ********************/
@implementation GLView{
	@private 
		CVDisplayLinkRef displayLink;
}

-(id)initWithFrame:(NSRect)frame{

	self = [super initWithFrame:frame];

	if(self){

		[[self window]setContentView:self];

		NSOpenGLPixelFormatAttribute attribute[] = {
			NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion4_1Core,
			NSOpenGLPFAScreenMask, CGDisplayIDToOpenGLDisplayMask(kCGDirectMainDisplay),
			NSOpenGLPFANoRecovery,
			NSOpenGLPFAAccelerated,
			NSOpenGLPFAColorSize, 24,
			NSOpenGLPFADepthSize, 24,
			NSOpenGLPFAAlphaSize, 8,
			NSOpenGLPFADoubleBuffer, 
			0
		};

		NSOpenGLPixelFormat *pixelFormat = [[[NSOpenGLPixelFormat alloc]initWithAttributes:attribute] autorelease];
	
		if(pixelFormat == nil){
			fprintf(gbFile, "InValid OpenGL Pixel Format!!\n");
			[self release];
			[NSApp terminate:self];
		}

		NSOpenGLContext *glContext = [[[NSOpenGLContext alloc]initWithFormat:pixelFormat shareContext:nil]autorelease];

		[self setPixelFormat:pixelFormat];

		[self setOpenGLContext:glContext];
	}
	return(self);
}

-(CVReturn)getFrameForTime:(const CVTimeStamp*)pOutputTime{

	NSAutoreleasePool *pool = [[NSAutoreleasePool alloc]init];

	[self drawView];

	[pool release];
	return(kCVReturnSuccess);
}

-(void)prepareOpenGL{

	fprintf(gbFile, "OpenGL Verision: %s\n", glGetString(GL_VERSION));
	fprintf(gbFile, "OpenGLSL Version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	[[self openGLContext]makeCurrentContext];

	GLuint swapInterval = 1; //1 Frame

	glClearColor(0.0f, 0.0f, 0.8f, 0.0f);



	CVDisplayLinkCreateWithActiveCGDisplays(&displayLink);
	CVDisplayLinkSetOutputCallback(displayLink, &MyDisplayLinkCallback, self);
	CGLContextObj cglContext = (CGLContextObj)[[self openGLContext]CGLContextObj];
	CGLPixelFormatObj cglPixelFormat = (CGLPixelFormatObj)[[self pixelFormat]CGLPixelFormatObj];
	CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext(displayLink, cglContext, cglPixelFormat);
	CVDisplayLinkStart(displayLink);
}


-(void)reshape{

	CGLLockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);

	NSRect rect = [self bounds];

	GLfloat width = rect.size.width;
	GLfloat height = rect.size.height;

	if(height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	CGLUnlockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);
}

-(void)drawRect:(NSRect)rect{

	[self drawView];

}

-(void)drawView{

	[[self openGLContext]makeCurrentContext];

	CGLLockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	CGLFlushDrawable((CGLContextObj)[[self openGLContext]CGLContextObj]);
	CGLUnlockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);
}

-(BOOL)acceptsFirstResponder{
	[[self window]makeFirstResponder:self];
	return(YES);
}

-(void)keyDown:(NSEvent*)event{
	int key = (int)[[event characters]characterAtIndex:0];
	switch(key){
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

-(void)mouseDown:(NSEvent*)event{

}

-(void)mouseDragged:(NSEvent*)event{

}

-(void)rightMouseDown:(NSEvent*)event{

}

-(void)dealloc{
	CVDisplayLinkStop(displayLink);
	CVDisplayLinkRelease(displayLink);

	[super dealloc];
}
@end

CVReturn MyDisplayLinkCallback(CVDisplayLinkRef displayLink, const CVTimeStamp *pNow, const CVTimeStamp *pOutputTime, CVOptionFlags flagsIn, CVOptionFlags *pFlagout, void *pDisplayLinkContext){

	CVReturn result = [(GLView*)pDisplayLinkContext getFrameForTime:pOutputTime];
	return(result);
}

