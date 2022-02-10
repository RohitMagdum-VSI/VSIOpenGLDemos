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


@interface AppDelegate_RRJ:NSObject < NSApplicationDelegate, NSWindowDelegate>
@end


@interface GLView_RRJ:NSOpenGLView
@end


int main(int argc, const char *argv[]){
    
    NSAutoreleasePool *pPool_RRJ = [[NSAutoreleasePool alloc] init];

    NSApp = [NSApplication sharedApplication];

    [NSApp setDelegate:[[AppDelegate_RRJ alloc] init]];

    [NSApp run];

    [pPool_RRJ release];

    return(0);
}


/********** AppDelegate_RRJ **********/

@implementation AppDelegate_RRJ
{
    @private
        NSWindow *window_RRJ;
        GLView_RRJ *glView_RRJ;
}


-(void)applicationDidFinishLaunching:(NSNotification*)aNotification{
    

    NSBundle *mainBundle_RRJ = [NSBundle mainBundle];
    NSString *appDirName_RRJ = [mainBundle_RRJ bundlePath];
    NSString *parentDirPath_RRJ = [appDirName_RRJ stringByDeletingLastPathComponent];
    NSString *logFileNameWithPath_RRJ = [NSString stringWithFormat: @"%@/Log.txt", parentDirPath_RRJ];
    const char *logFileName_RRJ = [logFileNameWithPath_RRJ cStringUsingEncoding:NSASCIIStringEncoding];
    
    gbFile_RRJ = fopen(logFileName_RRJ, "w");
    if(gbFile_RRJ == NULL){
        printf("Log Creation Failed!!\n");
        [self release];
        [NSApp terminate:self];
    }
    else
        fprintf(gbFile_RRJ, "Log Created!!\n");

    NSRect win_rect_RRJ;
    win_rect_RRJ = NSMakeRect(0.0, 0.0, 800.0, 600.0);

    window_RRJ = [[NSWindow alloc] initWithContentRect: win_rect_RRJ styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable| NSWindowStyleMaskMiniaturizable |
        NSWindowStyleMaskResizable
        backing: NSBackingStoreBuffered
        defer: NO];

    [window_RRJ setTitle: @"Rohit_R_Jadhav-Mac-16-CheckerBoard"];
    [window_RRJ center];

    glView_RRJ = [[GLView_RRJ alloc] initWithFrame: win_rect_RRJ];

    [window_RRJ setContentView: glView_RRJ];
    [window_RRJ setDelegate: self];
    [window_RRJ makeKeyAndOrderFront: self];
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
    [glView_RRJ release];

    [window_RRJ release];

    [super dealloc];
}

@end





//For CheckerBoard
const int CHECK_IMAGE_HEIGHT = 64;
const int CHECK_IMAGE_WIDTH = 64;
GLubyte CheckImageData_RRJ[CHECK_IMAGE_HEIGHT][CHECK_IMAGE_WIDTH][4];





/********** GLView_RRJ **********/
@implementation GLView_RRJ
{
    @private
        CVDisplayLinkRef displayLink_RRJ;

        GLuint vertexShaderObject_RRJ;
        GLuint fragmentShaderObject_RRJ;
        GLuint shaderProgramObject_RRJ;

        GLuint vao_Rect_RRJ;
        GLuint vbo_Rect_Position_RRJ;
        GLuint vbo_Rect_Texcoord_RRJ;

        GLuint mvpUniform_RRJ;
        GLuint samplerUniform_RRJ;

        GLuint texture_CheckerBoard_RRJ;


        vmath::mat4 perspectiveProjectionMatrix_RRJ;

}


-(id)initWithFrame:(NSRect)frame {
    
    self = [super initWithFrame: frame];

    if(self){
        
        [[self window] setContentView: self];


        NSOpenGLPixelFormatAttribute attribs_RRJ[] = {
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

        NSOpenGLPixelFormat *pixelFormat_RRJ = [[[NSOpenGLPixelFormat alloc] initWithAttributes: attribs_RRJ] autorelease];

        if(pixelFormat_RRJ == nil){
            fprintf(gbFile_RRJ, "No Valid OpenGL PixelFormat !!\n");
            [self release];
            [NSApp terminate:self];
        }

        NSOpenGLContext *glContext_RRJ = [[[NSOpenGLContext alloc] initWithFormat: pixelFormat_RRJ shareContext: nil]autorelease];

        [self setPixelFormat: pixelFormat_RRJ];

        [self setOpenGLContext: glContext_RRJ];
    }
    return(self);
}



-(CVReturn)getFrameForTime: (const CVTimeStamp*)pOutputTime {
    
    NSAutoreleasePool *pool_RRJ = [[NSAutoreleasePool alloc] init];

    //Display
    [self drawView];

    [pool_RRJ release];

    return(kCVReturnSuccess);

}

-(void) prepareOpenGL {
    
    fprintf(gbFile_RRJ, "OpenGL Version : %s\n", glGetString(GL_VERSION));
    fprintf(gbFile_RRJ, "OpenGL Shading Language Version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

    [[self openGLContext] makeCurrentContext];

    GLint swapInt_RRJ = 1;

    [[self openGLContext]setValues: &swapInt_RRJ forParameter: NSOpenGLCPSwapInterval];
    

    /********** Vertex Shader **********/
    vertexShaderObject_RRJ = glCreateShader(GL_VERTEX_SHADER);
    const GLchar *vertexShaderSourceCode_RRJ =
        "#version 410" \
        "\n" \
        "in vec4 vPosition;" \
        "in vec2 vTex;" \
        "out vec2 outTex;" \
        "uniform mat4 u_mvp_matrix;" \
        "void main(void) {" \
            "gl_Position = u_mvp_matrix * vPosition;" \
            "outTex = vTex;" \
        "}";

    glShaderSource(vertexShaderObject_RRJ, 1, (const GLchar**)&vertexShaderSourceCode_RRJ, NULL);

    glCompileShader(vertexShaderObject_RRJ);

    GLint shaderCompileStatus_RRJ = 0;
    GLint iInfoLogLength_RRJ = 0;
    char *szInfoLog_RRJ = NULL;

    glGetShaderiv(vertexShaderObject_RRJ, GL_COMPILE_STATUS, &shaderCompileStatus_RRJ);
    if(shaderCompileStatus_RRJ == GL_FALSE){
        glGetShaderiv(vertexShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
        if(iInfoLogLength_RRJ > 0){
            szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
            if(szInfoLog_RRJ){
                GLsizei written;
                glGetShaderInfoLog(vertexShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
                fprintf(gbFile_RRJ, "Vertex Shader Compilation Error: %s\n", szInfoLog_RRJ);
                free(szInfoLog_RRJ);
                szInfoLog_RRJ = NULL;
                [self release];
                [NSApp terminate: self];
            }
        }
    }



    /********** Fragment Shader **********/
    fragmentShaderObject_RRJ = glCreateShader(GL_FRAGMENT_SHADER);
    const GLchar *fragmentShaderSourceCode_RRJ =
        "#version 410" \
        "\n" \
        "in vec2 outTex;" \
        "uniform sampler2D u_sampler;"
        "out vec4 FragColor;" \
        "void main(void) {" \
            "vec4 tex = texture(u_sampler, outTex);"
            "FragColor = tex;" \
        "}";

    glShaderSource(fragmentShaderObject_RRJ, 1, (const GLchar**)&fragmentShaderSourceCode_RRJ, NULL);
    glCompileShader(fragmentShaderObject_RRJ);

    shaderCompileStatus_RRJ = 0;
    iInfoLogLength_RRJ = 0;
    szInfoLog_RRJ = NULL;

    glGetShaderiv(fragmentShaderObject_RRJ, GL_COMPILE_STATUS, &shaderCompileStatus_RRJ);
    if(shaderCompileStatus_RRJ == GL_FALSE){
        glGetShaderiv(fragmentShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
        if(iInfoLogLength_RRJ > 0){
            szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
            if(szInfoLog_RRJ){
                GLsizei written;
                glGetShaderInfoLog(fragmentShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
                fprintf(gbFile_RRJ, "Fragment Shader Compilation Error: %s\n", szInfoLog_RRJ);
                free(szInfoLog_RRJ);
                szInfoLog_RRJ = NULL;
                [self release];
                [NSApp terminate: self];
            }
        }
    }


    /********** Shader Program **********/
    shaderProgramObject_RRJ = glCreateProgram();

    glAttachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
    glAttachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);

    glBindAttribLocation(shaderProgramObject_RRJ, AMC_ATTRIBUTE_POSITION, "vPosition");
    glBindAttribLocation(shaderProgramObject_RRJ, AMC_ATTRIBUTE_TEXCOORD0, "vTex");

    glLinkProgram(shaderProgramObject_RRJ);

    GLint iProgramLinkStatus_RRJ;
    iInfoLogLength_RRJ = 0;
    szInfoLog_RRJ = NULL;

    glGetProgramiv(shaderProgramObject_RRJ, GL_LINK_STATUS, &iProgramLinkStatus_RRJ);
    if(iProgramLinkStatus_RRJ == GL_FALSE){
        glGetProgramiv(shaderProgramObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
        if(iInfoLogLength_RRJ > 0){
            szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
            if(szInfoLog_RRJ){
                GLsizei written;
                glGetProgramInfoLog(shaderProgramObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
                fprintf(gbFile_RRJ, "Shader Program Linking Error: %s\n", szInfoLog_RRJ);
                szInfoLog_RRJ = NULL;
                [self release];
                [NSApp terminate: self];
            }
        }
    }


    mvpUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_mvp_matrix");
    samplerUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_sampler");

    /********** Triangle Position, vao, vbo **********/
    GLfloat rectangle_Texcoord_RRJ[] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f
    };

   

    /********* Vao For Rectangle *********/
	glGenVertexArrays(1, &vao_Rect_RRJ);
	glBindVertexArray(vao_Rect_RRJ);
		
		/********** Position *********/
		glGenBuffers(1, &vbo_Rect_Position_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Position_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 3 * 4, NULL, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
    
        /********** Texcoord *********/
        glGenBuffers(1, &vbo_Rect_Texcoord_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Texcoord_RRJ);
        glBufferData(GL_ARRAY_BUFFER, sizeof(rectangle_Texcoord_RRJ), rectangle_Texcoord_RRJ, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOORD0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOORD0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);


    texture_CheckerBoard_RRJ = [self loadTextureFromBMP];

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glClearDepth(1.0f);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    perspectiveProjectionMatrix_RRJ = vmath::mat4::identity();

    CVDisplayLinkCreateWithActiveCGDisplays(&displayLink_RRJ);
    CVDisplayLinkSetOutputCallback(displayLink_RRJ, &MyDisplayLinkCallback, self);
    CGLContextObj cglContext_RRJ = (CGLContextObj)[[self openGLContext]CGLContextObj];
    CGLPixelFormatObj cglPixelFormat_RRJ = (CGLPixelFormatObj)[[self pixelFormat]CGLPixelFormatObj];
    CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext(displayLink_RRJ, cglContext_RRJ, cglPixelFormat_RRJ);
    CVDisplayLinkStart(displayLink_RRJ);
}


-(void)makeCheckImage{
	GLint c;

	for (int i = 0; i < CHECK_IMAGE_HEIGHT; i++) {
		for (int j = 0; j < CHECK_IMAGE_WIDTH; j++) {

			c = (((i & 0x8) == 0) ^ ((j & 0x8) == 0)) * 255;
			CheckImageData_RRJ[i][j][0] = (GLubyte)c;
			CheckImageData_RRJ[i][j][1] = (GLubyte)c;
			CheckImageData_RRJ[i][j][2] = (GLubyte)c;
			CheckImageData_RRJ[i][j][3] = (GLubyte)255;
		}
	}
}



-(GLuint)loadTextureFromBMP{

    GLuint texture_RRJ;

    [self makeCheckImage];

    glGenTextures(1, &texture_RRJ);
    glBindTexture(GL_TEXTURE_2D, texture_RRJ);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        glTexImage2D(GL_TEXTURE_2D,
            0,
            GL_RGBA,
            CHECK_IMAGE_WIDTH, CHECK_IMAGE_HEIGHT, 0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            (void*)CheckImageData_RRJ);

        glGenerateMipmap(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
        return(texture_RRJ);
}


-(void)reshape {
    
    CGLLockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);

    NSRect rect_RRJ = [self bounds];
    GLfloat width_RRJ = rect_RRJ.size.width;
    GLfloat height_RRJ = rect_RRJ.size.height;

    if(height_RRJ == 0)
        height_RRJ = 1;

    glViewport(0, 0, (GLsizei)width_RRJ, (GLsizei)height_RRJ);

    perspectiveProjectionMatrix_RRJ = vmath::perspective(45.0f, (GLfloat)width_RRJ / (GLfloat)height_RRJ, 0.1f, 100.0f);

    CGLUnlockContext((CGLContextObj)[[self openGLContext] CGLContextObj]);
}


-(void)drawRect:(NSRect)rect {
    [self drawView];
}


-(void) drawView {
    
    [[self openGLContext]makeCurrentContext];

    CGLLockContext((CGLContextObj) [[self openGLContext] CGLContextObj]);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    vmath::mat4 translateMatrix_RRJ;
    vmath::mat4 modelViewMatrix_RRJ;
    vmath::mat4 modelViewProjectionMatrix_RRJ;


    GLfloat CheckerBoard_Position_RRJ[3*4];

    glUseProgram(shaderProgramObject_RRJ);

    
    for(int i = 1; i <= 2; i++){


    	   if (i == 1) {

			CheckerBoard_Position_RRJ[0] = -2.0f;
			CheckerBoard_Position_RRJ[1] = -1.0f;
			CheckerBoard_Position_RRJ[2] = 0.0f;

			CheckerBoard_Position_RRJ[3] = -2.0f;
			CheckerBoard_Position_RRJ[4] = 1.0f;
			CheckerBoard_Position_RRJ[5] = 0.0f;

			CheckerBoard_Position_RRJ[6] = 0.0f;
			CheckerBoard_Position_RRJ[7] = 1.0f;
			CheckerBoard_Position_RRJ[8] = 0.0f;

			CheckerBoard_Position_RRJ[9] = 0.0f;
			CheckerBoard_Position_RRJ[10] = -1.0f;
			CheckerBoard_Position_RRJ[11] = 0.0f;


	    }
	    else if(i == 2){
            CheckerBoard_Position_RRJ[0] = 1.0f;
			CheckerBoard_Position_RRJ[1] = -1.0f;
			CheckerBoard_Position_RRJ[2] = 0.0f;

			CheckerBoard_Position_RRJ[3] = 1.0f;
			CheckerBoard_Position_RRJ[4] = 1.0f;
			CheckerBoard_Position_RRJ[5] = 0.0f;

			CheckerBoard_Position_RRJ[6] = 2.41421f;
			CheckerBoard_Position_RRJ[7] = 1.0f;
			CheckerBoard_Position_RRJ[8] = -1.41421f;

			CheckerBoard_Position_RRJ[9] = 2.41421f;
			CheckerBoard_Position_RRJ[10] = -1.0f;
			CheckerBoard_Position_RRJ[11] = -1.41421f;
	    }



	    /********** RECTANGLE **********/
	    translateMatrix_RRJ = vmath::mat4::identity();
	    modelViewMatrix_RRJ = vmath::mat4::identity();
	    modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

	    translateMatrix_RRJ = vmath::translate(0.0f, 0.0f, -5.0f);
	    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
	    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

	    glUniformMatrix4fv(mvpUniform_RRJ, 1, false, modelViewProjectionMatrix_RRJ);


	    glActiveTexture(GL_TEXTURE0);
	    glBindTexture(GL_TEXTURE_2D, texture_CheckerBoard_RRJ);
	    glUniform1i(samplerUniform_RRJ, 0);


	    glBindVertexArray(vao_Rect_RRJ);
	    	glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Position_RRJ);
	    	glBufferData(GL_ARRAY_BUFFER, sizeof(CheckerBoard_Position_RRJ), CheckerBoard_Position_RRJ, GL_DYNAMIC_DRAW);
	    	glBindBuffer(GL_ARRAY_BUFFER, 0);
	        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	    glBindVertexArray(0);


	}

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
    
    if(texture_CheckerBoard_RRJ){
        glDeleteTextures(1, &texture_CheckerBoard_RRJ);
        texture_CheckerBoard_RRJ = 0;
    }

    if(vbo_Rect_Texcoord_RRJ){
        glDeleteBuffers(1, &vbo_Rect_Texcoord_RRJ);
        vbo_Rect_Texcoord_RRJ = 0;
    }
    
    if(vbo_Rect_Position_RRJ){
        glDeleteBuffers(1, &vbo_Rect_Position_RRJ);
        vbo_Rect_Position_RRJ = 0;
    }

    if(vao_Rect_RRJ){
        glDeleteVertexArrays(1, &vao_Rect_RRJ);
        vao_Rect_RRJ = 0;
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


    CVDisplayLinkStop(displayLink_RRJ);
    CVDisplayLinkRelease(displayLink_RRJ);

    [super dealloc];
}

@end

CVReturn MyDisplayLinkCallback(CVDisplayLinkRef displayLink, const CVTimeStamp *pNow, const CVTimeStamp *pOutputTime, CVOptionFlags flagsIn, CVOptionFlags *pFlagsOut, void *pDisplayLinkContext){

    CVReturn result_RRJ = [(GLView_RRJ*)pDisplayLinkContext getFrameForTime: pOutputTime];
    return(result_RRJ);
}




