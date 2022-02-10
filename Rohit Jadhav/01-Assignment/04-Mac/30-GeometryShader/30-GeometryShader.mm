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


FILE *gbFile_RRJ = NULL;


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
    
    gbFile_RRJ = fopen(logFileName, "w");
    if(gbFile_RRJ == NULL){
        printf("Log Creation Failed!!\n");
        [self release];
        [NSApp terminate:self];
    }
    else
        fprintf(gbFile_RRJ, "Log Created!!\n");

    NSRect win_rect;
    win_rect = NSMakeRect(0.0, 0.0, 800.0, 600.0);

    window = [[NSWindow alloc] initWithContentRect: win_rect styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable| NSWindowStyleMaskMiniaturizable |
        NSWindowStyleMaskResizable
        backing: NSBackingStoreBuffered
        defer: NO];

    [window setTitle: @"Rohit_R_Jadhav-Mac-30-GeometryShader"];
    [window center];

    glView = [[GLView alloc] initWithFrame: win_rect];

    [window setContentView: glView];
    [window setDelegate: self];
    [window makeKeyAndOrderFront: self];
}


-(void)applicationWillTerminate:(NSNotification*)notification {
    fprintf(gbFile_RRJ, "Program is Terminate SuccessFully!!\n");

    if(gbFile_RRJ){
        fprintf(gbFile_RRJ, "Log Is Close!!\n");
        fclose(gbFile_RRJ);
        gbFile_RRJ = NULL;
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

       //For Shader
	GLuint vertexShaderObject_RRJ;
	GLuint geometryShaderObject_RRJ;
	GLuint fragmentShaderObject_RRJ;
	GLuint shaderProgramObject_RRJ;

	GLuint vao_Tri_RRJ;
	GLuint vbo_Tri_Position_RRJ;

	//For Uniform
	GLuint mvpUniform_RRJ;

        vmath::mat4 perspectiveProjectionMatrix;

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
            fprintf(gbFile_RRJ, "No Valid OpenGL PixelFormat !!\n");
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
    
    fprintf(gbFile_RRJ, "OpenGL Version : %s\n", glGetString(GL_VERSION));
    fprintf(gbFile_RRJ, "OpenGL Shading Language Version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

    [[self openGLContext] makeCurrentContext];

    GLint swapInt = 1;

    [[self openGLContext]setValues: &swapInt forParameter: NSOpenGLCPSwapInterval];
    

   /********** VERTEX SHADER **********/
	vertexShaderObject_RRJ = glCreateShader(GL_VERTEX_SHADER);
	const char *vertexShaderSourceCode_RRJ =
		"#version 410" \
		"\n" \
		"in vec4 vPosition;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void) {" \
		"gl_Position = u_mvp_matrix * vPosition;" \
		"}";

	glShaderSource(vertexShaderObject_RRJ, 1, (const char**)&vertexShaderSourceCode_RRJ, NULL);
	glCompileShader(vertexShaderObject_RRJ);

	int iInfoLogLength_RRJ;
	int iShaderCompileStatus_RRJ;
	char *szInfoLog_RRJ = NULL;

	glGetShaderiv(vertexShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(vertexShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ) {
				GLsizei written;

				glGetShaderInfoLog(vertexShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "VERTEX SHADER ERROR: \n %s", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				[self release];
				[NSApp terminate:self];
			}
		}
	}


	/********** GEOMETRY SHADER **********/
	geometryShaderObject_RRJ = glCreateShader(GL_GEOMETRY_SHADER);
	const char *geometryShaderSourceCode_RRJ =
		"#version 410" \
		"\n" \
		"layout(triangles)in;" \
		"layout(triangle_strip, max_vertices = 9)out;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void) {" \
			"for(int vertex = 0; vertex < 3; vertex++){" \
		
				"gl_Position = u_mvp_matrix * (gl_in[vertex].gl_Position + vec4(0.0, 1.0, 0.0, 0.0));" \
				"EmitVertex();" \
					
				"gl_Position = u_mvp_matrix * (gl_in[vertex].gl_Position + vec4(-1.0, -1.0, 0.0, 0.0));" \
				"EmitVertex();" \
			
				"gl_Position = u_mvp_matrix * (gl_in[vertex].gl_Position + vec4(1.0, -1.0, 0.0, 0.0));" \
				"EmitVertex();" \

				"EndPrimitive();" \
			"}" \
		"}";


	glShaderSource(geometryShaderObject_RRJ, 1,
		(const char**)&geometryShaderSourceCode_RRJ, NULL);

	glCompileShader(geometryShaderObject_RRJ);

	iInfoLogLength_RRJ = 0;
	iShaderCompileStatus_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(geometryShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(geometryShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ) {
				GLsizei written;
				glGetShaderInfoLog(geometryShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "GEOMETRY SHADER ERROR: \n %s", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				[self release];
				[NSApp terminate:self];
			}
		}
	}
		
	



	/********** FRAGMENT SHADER **********/
	fragmentShaderObject_RRJ = glCreateShader(GL_FRAGMENT_SHADER);
	const char *fragmentShaderSourceCode_RRJ =
		"#version 410" \
		"\n" \
		"out vec4 FragColor;" \
		"void main(void) {" \
		"FragColor = vec4(1.0, 1.0, 1.0, 1.0);" \
		"}";


	glShaderSource(fragmentShaderObject_RRJ, 1,
		(const char**)&fragmentShaderSourceCode_RRJ, NULL);

	glCompileShader(fragmentShaderObject_RRJ);

	iInfoLogLength_RRJ = 0;
	iShaderCompileStatus_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(fragmentShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(fragmentShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ) {
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "FRAGMENT SHADER ERROR: \n %s", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				[self release];
				[NSApp terminate:self];
			}
		}
	}


	/********** SHADER PROGRAM **********/
	shaderProgramObject_RRJ = glCreateProgram();

	glAttachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
	glAttachShader(shaderProgramObject_RRJ, geometryShaderObject_RRJ);
	glAttachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);

	glBindAttribLocation(shaderProgramObject_RRJ, AMC_ATTRIBUTE_POSITION, "vPosition");

	glLinkProgram(shaderProgramObject_RRJ);

	int iProgramLinkStatus_RRJ;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;
	glGetProgramiv(shaderProgramObject_RRJ, GL_LINK_STATUS, &iProgramLinkStatus_RRJ);
	if (iProgramLinkStatus_RRJ == GL_FALSE) {
		glGetProgramiv(shaderProgramObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ) {
				GLsizei written;
				glGetProgramInfoLog(shaderProgramObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "SHADER PROGRAM ERROR: %s", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				[self release];
				[NSApp terminate:self];
			}
		}
	}


	mvpUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_mvp_matrix");
	



	/********** LINE COORDINATES **********/
	float tri_Vertices_RRJ[] = {
		0.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
	};

	glGenVertexArrays(1, &vao_Tri_RRJ);
	glBindVertexArray(vao_Tri_RRJ);

	/********** Position **********/
	glGenBuffers(1, &vbo_Tri_Position_RRJ);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Tri_Position_RRJ);
	glBufferData(GL_ARRAY_BUFFER, sizeof(tri_Vertices_RRJ), tri_Vertices_RRJ, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);


    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glClearDepth(1.0f);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    perspectiveProjectionMatrix = vmath::mat4::identity();

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

    perspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);

    CGLUnlockContext((CGLContextObj)[[self openGLContext] CGLContextObj]);
}


-(void)drawRect:(NSRect)rect {
    [self drawView];
}


-(void) drawView {
    
    [[self openGLContext]makeCurrentContext];

    CGLLockContext((CGLContextObj) [[self openGLContext] CGLContextObj]);

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	vmath::mat4 modelViewMatrix_RRJ;
	vmath::mat4 modelViewProjectionMatrix_RRJ;


	glUseProgram(shaderProgramObject_RRJ);

	modelViewMatrix_RRJ = vmath::mat4::identity();
	modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

	modelViewMatrix_RRJ = vmath::translate(0.0f, 0.00f, -4.0f);
	modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix * modelViewMatrix_RRJ;

	glUniformMatrix4fv(mvpUniform_RRJ, 1, GL_FALSE, modelViewProjectionMatrix_RRJ);
	

	glBindVertexArray(vao_Tri_RRJ);
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
    
   	if (vbo_Tri_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Tri_Position_RRJ);
		vbo_Tri_Position_RRJ = 0;
	}

	if (vao_Tri_RRJ) {
		glDeleteVertexArrays(1, &vao_Tri_RRJ);
		vao_Tri_RRJ = 0;
	}

	if (shaderProgramObject_RRJ) {
		glUseProgram(shaderProgramObject_RRJ);

		/*GLint shaderCount_RRJ;
		GLint shaderNo_RRJ;

		glGetShaderiv(shaderProgramObject_RRJ, GL_ATTACHED_SHADERS, &shaderCount_RRJ);
		fprintf(gbFile_RRJ, "INFO: ShaderCount: %d\n", shaderCount_RRJ);
		GLuint *pShaders = (GLuint*)malloc(sizeof(GLuint*) * shaderCount_RRJ);
		if (pShaders) {
			glGetAttachedShaders(shaderProgramObject_RRJ, shaderCount_RRJ, &shaderCount_RRJ, pShaders);
			for (shaderNo_RRJ = 0; shaderNo_RRJ < shaderCount_RRJ; shaderNo_RRJ++) {
				glDetachShader(shaderProgramObject_RRJ, pShaders[shaderNo_RRJ]);
				glDeleteShader(pShaders[shaderNo_RRJ]);
				pShaders[shaderNo_RRJ] = 0;
			}
			free(pShaders);
			pShaders = NULL;
		}*/


		if (fragmentShaderObject_RRJ) {
			glDetachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);
			glDeleteShader(fragmentShaderObject_RRJ);
			fragmentShaderObject_RRJ = 0;
		}

		if (geometryShaderObject_RRJ) {
			glDetachShader(shaderProgramObject_RRJ, geometryShaderObject_RRJ);
			glDeleteShader(geometryShaderObject_RRJ);
			geometryShaderObject_RRJ = 0;
		}

		if (vertexShaderObject_RRJ) {
			glDetachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
			glDeleteShader(vertexShaderObject_RRJ);
			vertexShaderObject_RRJ = 0;
		}


		glUseProgram(0);
		glDeleteProgram(shaderProgramObject_RRJ);
		shaderProgramObject_RRJ = 0;
	}

    
    GLsizei shaderCount = 0;
    GLuint shaderNo = 0;

    glUseProgram(shaderProgramObject_RRJ);

    glGetProgramiv(shaderProgramObject_RRJ, GL_ATTACHED_SHADERS, &shaderCount);
    fprintf(gbFile_RRJ, "Shader Count: %d\n",  shaderCount);
    GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * shaderCount);
    if(pShader){

        glGetAttachedShaders(shaderProgramObject_RRJ, shaderCount, &shaderCount, pShader);
        for(shaderNo = 0; shaderNo < shaderCount; shaderNo++){
            glDetachShader(shaderProgramObject_RRJ, pShader[shaderNo]);
            glDeleteShader(pShader[shaderNo]);
            pShader[shaderNo] = 0;
        }
        free(pShader);
        pShader = NULL;
    }

    glUseProgram(0);
    glDeleteProgram(shaderProgramObject_RRJ);
    shaderProgramObject_RRJ = 0;


    CVDisplayLinkStop(displayLink);
    CVDisplayLinkRelease(displayLink);

    [super dealloc];
}

@end

CVReturn MyDisplayLinkCallback(CVDisplayLinkRef displayLink, const CVTimeStamp *pNow, const CVTimeStamp *pOutputTime, CVOptionFlags flagsIn, CVOptionFlags *pFlagsOut, void *pDisplayLinkContext){

    CVReturn result = [(GLView*)pDisplayLinkContext getFrameForTime: pOutputTime];
    return(result);
}




