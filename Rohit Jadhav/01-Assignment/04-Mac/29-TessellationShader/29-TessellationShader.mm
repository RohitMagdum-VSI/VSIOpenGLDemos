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

    [window setTitle: @"Rohit_R_Jadhav-Mac-29-TessellationShader"];
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
        GLuint tessellationControlShaderObject_RRJ;
        GLuint tessellationEvaluationShaderObject_RRJ;
        GLuint fragmentShaderObject_RRJ;
        GLuint shaderProgramObject_RRJ;

        GLuint vao_Lines_RRJ;
        GLuint vbo_Lines_Position_RRJ;

        //For Uniform
        GLuint mvpUniform_RRJ;
        GLuint numberOfSegmentsUniform_RRJ;
        GLuint numberOfStripsUniform_RRJ;
        GLuint lineColorUniform_RRJ;

        GLuint numberOfLineSegments_RRJ;
    
        GLint iColor_RRJ;

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
		"in vec2 vPosition;" \
		"void main(void) {" \
		"gl_Position = vec4(vPosition, 0.0, 1.0);" \
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




	/********** TESSELLATION CONTROL SHADER **********/
	tessellationControlShaderObject_RRJ = glCreateShader(GL_TESS_CONTROL_SHADER);
	const char* tessellationControlShaderSourceCode_RRJ =
		"#version 410" \
		"\n" \
		"layout(vertices=4)out;" \
		"uniform int u_numberOfSegments;" \
		"uniform int u_numberOfStrips;" \
		"void main(void) {" \
		"gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;" \
		"gl_TessLevelOuter[0] = float(u_numberOfStrips);" \
		"gl_TessLevelOuter[1] = float(u_numberOfSegments);" \
		"}";


	glShaderSource(tessellationControlShaderObject_RRJ, 1,
		(const char**)&tessellationControlShaderSourceCode_RRJ, NULL);

	glCompileShader(tessellationControlShaderObject_RRJ);

	iShaderCompileStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;
	glGetShaderiv(tessellationControlShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(tessellationControlShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ) {
				GLsizei written;
				glGetShaderInfoLog(tessellationControlShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "TESSELLATION CONTROL SHADER ERROR: \n %s", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				[self release];
				[NSApp terminate:self];
			}
		}
	}



	/********** TESSELLATION EVALUATION SHADER **********/
	tessellationEvaluationShaderObject_RRJ = glCreateShader(GL_TESS_EVALUATION_SHADER);
	const char *tessellationEvaluationShaderSourceCode_RRJ =
		"#version 410" \
		"\n" \
		"layout(isolines)in;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void) {" \

			"float u = gl_TessCoord.x;" \

			"vec3 p0 = gl_in[0].gl_Position.xyz;" \
			"vec3 p1 = gl_in[1].gl_Position.xyz;" \
			"vec3 p2 = gl_in[2].gl_Position.xyz;" \
			"vec3 p3 = gl_in[3].gl_Position.xyz;" \

			"float b0 = (1.0 - u) * (1.0 - u) * (1.0 - u);" \
			"float b1 = 3.0 * u * (1.0 - u) * (1.0 - u);" \
			"float b2 = 3.0 * u * u * (1.0 - u);" \
			"float b3 = u * u * u;" \

			"vec3 p = p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3;" \
			"gl_Position = u_mvp_matrix * vec4(p, 1.0);" \
		"}";	



	glShaderSource(tessellationEvaluationShaderObject_RRJ, 1,
		(const char**)&tessellationEvaluationShaderSourceCode_RRJ, NULL);

	glCompileShader(tessellationEvaluationShaderObject_RRJ);

	iShaderCompileStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(tessellationEvaluationShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(tessellationEvaluationShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ) {
				GLsizei written;
				glGetShaderInfoLog(tessellationEvaluationShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "TESSELLATION EVALUATION SHADER ERROR: \n %s", szInfoLog_RRJ);
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
		"uniform vec4 u_lineColor;" \
		"out vec4 FragColor;" \
		"void main(void) {" \
			"FragColor = u_lineColor;" \
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
	glAttachShader(shaderProgramObject_RRJ, tessellationControlShaderObject_RRJ);
	glAttachShader(shaderProgramObject_RRJ, tessellationEvaluationShaderObject_RRJ);
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
	numberOfSegmentsUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_numberOfSegments");
	numberOfStripsUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_numberOfStrips");
	lineColorUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_lineColor");



	/********** LINE COORDINATES **********/
	float lines_Vertices_RRJ[] = { 
		-1.0f, -1.0f, 
		-0.5f, 1.0f, 
		0.5f, -1.0f, 
		1.0f, 1.0f
	};

	glGenVertexArrays(1, &vao_Lines_RRJ);
	glBindVertexArray(vao_Lines_RRJ);

		/********** Position **********/
		glGenBuffers(1, &vbo_Lines_Position_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Lines_Position_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(lines_Vertices_RRJ), lines_Vertices_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);


    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glClearDepth(1.0f);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    numberOfLineSegments_RRJ = 1;
    iColor_RRJ = 1;
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

	modelViewMatrix_RRJ = vmath::translate(0.0f, 0.00f, -3.0f);
	modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix * modelViewMatrix_RRJ;

	glUniformMatrix4fv(mvpUniform_RRJ, 1, GL_FALSE, modelViewProjectionMatrix_RRJ);
	glUniform1i(numberOfSegmentsUniform_RRJ, numberOfLineSegments_RRJ);
	glUniform1i(numberOfStripsUniform_RRJ, 1);
    
    if(iColor_RRJ == 1)
        glUniform4fv(lineColorUniform_RRJ, 1, vmath::vec4(1.0f, 1.0f, 0.0f, 1.0f));
    else if(iColor_RRJ == 2)
        glUniform4fv(lineColorUniform_RRJ, 1, vmath::vec4(0.0f, 1.0f, 0.0f, 1.0f));

	glBindVertexArray(vao_Lines_RRJ);
	glPatchParameteri(GL_PATCH_VERTICES, 4);
	glDrawArrays(GL_PATCHES, 0, 4);
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
            
        case 63232:
            numberOfLineSegments_RRJ++;
            if(numberOfLineSegments_RRJ > 14){
                iColor_RRJ = 2;
                numberOfLineSegments_RRJ = 14;
            }
            break;
            
        case 63233:
            numberOfLineSegments_RRJ--;
            if(numberOfLineSegments_RRJ < 1){
                iColor_RRJ = 1;
                numberOfLineSegments_RRJ = 1;
            }
            break;

        default:
            break;
    }
    NSString *titleName_RRJ = [NSString stringWithFormat: @"Rohit_R_Jadhav-Mac-29-TessellationShader-[Segments: %d]", numberOfLineSegments_RRJ];
    
    [[self window]setTitle: titleName_RRJ];
   
}

-(void) mouseDown: (NSEvent*) event{
    
}

-(void) mouseDragged: (NSEvent*) event{

}

-(void) rightMouseDown: (NSEvent*) event{

}

-(void) dealloc {
    
    if (vbo_Lines_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Lines_Position_RRJ);
		vbo_Lines_Position_RRJ = 0;
	}

	if (vao_Lines_RRJ) {
		glDeleteVertexArrays(1, &vao_Lines_RRJ);
		vao_Lines_RRJ = 0;
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

		if (tessellationEvaluationShaderObject_RRJ) {
			glDetachShader(shaderProgramObject_RRJ, tessellationEvaluationShaderObject_RRJ);
			glDeleteShader(tessellationEvaluationShaderObject_RRJ);
			tessellationEvaluationShaderObject_RRJ = 0;
		}

		if (tessellationControlShaderObject_RRJ) {
			glDetachShader(shaderProgramObject_RRJ, tessellationControlShaderObject_RRJ);
			glDeleteShader(tessellationControlShaderObject_RRJ);
			tessellationControlShaderObject_RRJ = 0;
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



    CVDisplayLinkStop(displayLink);
    CVDisplayLinkRelease(displayLink);

    [super dealloc];
}

@end

CVReturn MyDisplayLinkCallback(CVDisplayLinkRef displayLink, const CVTimeStamp *pNow, const CVTimeStamp *pOutputTime, CVOptionFlags flagsIn, CVOptionFlags *pFlagsOut, void *pDisplayLinkContext){

    CVReturn result = [(GLView*)pDisplayLinkContext getFrameForTime: pOutputTime];
    return(result);
}




