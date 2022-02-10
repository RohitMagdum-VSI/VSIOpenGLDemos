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

/*const GLfloat M_PI = 3.14159265360;*/

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

    [window_RRJ setTitle: @"Rohit_R_Jadhav-Mac-08-GraphPaper_WithShape"];
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


/********** GLView_RRJ **********/
@implementation GLView_RRJ
{
    @private
        CVDisplayLinkRef displayLink_RRJ;

        GLuint vertexShaderObject_RRJ;
        GLuint fragmentShaderObject_RRJ;
        GLuint shaderProgramObject_RRJ;

     	//For Blue_Grid
    	GLuint vao_Grid_RRJ;
    	GLuint vbo_Grid_Position_RRJ_RRJ;

    	//For Red_And_Blue_Axis
    	GLuint vao_Axis_RRJ;
    	GLuint vbo_Axis_Position_RRJ_RRJ;
    	GLuint vbo_Axis_Color_RRJ_RRJ;

        //For Yellow Triangle
        GLuint vao_Triangle_RRJ;
        GLuint vbo_Triangle_Position_RRJ_RRJ;
        GLuint vbo_Triangle_Color_RRJ_RRJ;

        //For Yellow Rect
        GLuint vao_Rect_RRJ;
        GLuint vbo_Rect_Position_RRJ_RRJ;
        GLuint vbo_Rect_Color_RRJ_RRJ;

        //For Yellow InCircle
        GLuint vao_Circle_RRJ;
        GLuint vbo_Circle_Position_RRJ_RRJ;
        GLuint vbo_Circle_Color_RRJ_RRJ;

        GLfloat Incircle_Center_RRJ[3];
        GLfloat Incircle_Radius_RRJ;


        GLuint mvpUniform_RRJ;
    

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
        "in vec4 vColor;" \
        "out vec4 outColor;" \
        "uniform mat4 u_mvp_matrix;" \
        "void main(void) {" \
            "gl_Position = u_mvp_matrix * vPosition;" \
            "outColor = vColor;" \
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
        "in vec4 outColor;" \
        "out vec4 FragColor;" \
        "void main(void) {" \
            "FragColor = outColor;" \
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
    glBindAttribLocation(shaderProgramObject_RRJ, AMC_ATTRIBUTE_COLOR, "vColor");

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


   	/********** Position and Color **********/
	GLfloat Grid_Position_RRJ[40 * 6];
	[self FillGridPosition: Grid_Position_RRJ];
	
    GLfloat Triangle_Position_RRJ[] = {
        0.0f, 0.70f, 0.0f,
        -0.70f, -0.70f, 0.0f,
        0.70f, -0.70f, 0.0f
    };

    GLfloat Triangle_Color_RRJ[] = {
        1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f
    };

    GLfloat Rect_Position_RRJ[] = {
        0.70f, 0.70f, 0.0f,
        -0.70f, 0.70f, 0.0f,
        -0.70f, -0.70f, 0.0f,
        0.70f, -0.70f, 0.0f
    };

    GLfloat Rect_Color_RRJ[] = {
        1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f
    };

    /********** To Calculate InCircle Radius and Center **********/
    [self Calculation: Triangle_Position_RRJ];


		/********** Grid **********/
		glGenVertexArrays(1, &vao_Grid_RRJ);
		glBindVertexArray(vao_Grid_RRJ);

		/********** Position **********/
		glGenBuffers(1, &vbo_Grid_Position_RRJ_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Grid_Position_RRJ_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Grid_Position_RRJ), Grid_Position_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/********** Color **********/
		glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 0.0f, 0.0f, 1.0f);

	glBindVertexArray(0);


	/********** Axis **********/
	glGenVertexArrays(1, &vao_Axis_RRJ);
	glBindVertexArray(vao_Axis_RRJ);

		/********** Position **********/
		glGenBuffers(1, &vbo_Axis_Position_RRJ_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Axis_Position_RRJ_RRJ);
		glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/********** Color **********/
		glGenBuffers(1, &vbo_Axis_Color_RRJ_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Axis_Color_RRJ_RRJ);
		glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);



    /********** Triangle **********/
    glGenVertexArrays(1, &vao_Triangle_RRJ);
    glBindVertexArray(vao_Triangle_RRJ);

            /********** Position **********/
            glGenBuffers(1, &vbo_Triangle_Position_RRJ_RRJ);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_Triangle_Position_RRJ_RRJ);
            glBufferData(GL_ARRAY_BUFFER, sizeof(Triangle_Position_RRJ), Triangle_Position_RRJ, GL_STATIC_DRAW);
            glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            /********** Color **********/
            glGenBuffers(1, &vbo_Triangle_Color_RRJ_RRJ);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_Triangle_Color_RRJ_RRJ);
            glBufferData(GL_ARRAY_BUFFER, sizeof(Triangle_Color_RRJ), Triangle_Color_RRJ, GL_STATIC_DRAW);
            glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);



    /********** Rectangle **********/
    glGenVertexArrays(1, &vao_Rect_RRJ);
    glBindVertexArray(vao_Rect_RRJ);

        /********** Position **********/
        glGenBuffers(1, &vbo_Rect_Position_RRJ_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Position_RRJ_RRJ);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Rect_Position_RRJ), Rect_Position_RRJ, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        /********** Color **********/
        glGenBuffers(1, &vbo_Rect_Color_RRJ_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Color_RRJ_RRJ);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Rect_Color_RRJ), Rect_Color_RRJ, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);


    /********** Circle **********/
    glGenVertexArrays(1, &vao_Circle_RRJ);
    glBindVertexArray(vao_Circle_RRJ);

        /********** Position **********/
        glGenBuffers(1, &vbo_Circle_Position_RRJ_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_Circle_Position_RRJ_RRJ);
        glBufferData(GL_ARRAY_BUFFER, 3000 * 3 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        /********** Color **********/
        glGenBuffers(1, &vbo_Circle_Color_RRJ_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_Circle_Color_RRJ_RRJ);
        glBufferData(GL_ARRAY_BUFFER, 3 * 3000 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);




    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glClearDepth(1.0f);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);


    perspectiveProjectionMatrix_RRJ = vmath::mat4::identity();

    CVDisplayLinkCreateWithActiveCGDisplays(&displayLink_RRJ);
    CVDisplayLinkSetOutputCallback(displayLink_RRJ, &MyDisplayLinkCallback, self);
    CGLContextObj cglContext_RRJ = (CGLContextObj)[[self openGLContext]CGLContextObj];
    CGLPixelFormatObj cglPixelFormat_RRJ = (CGLPixelFormatObj)[[self pixelFormat]CGLPixelFormatObj];
    CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext(displayLink_RRJ, cglContext_RRJ, cglPixelFormat_RRJ);
    CVDisplayLinkStart(displayLink_RRJ);
}


-(void)FillGridPosition: (GLfloat[]) arr{
	GLint j = 0;

	//Vertical Lines 20
	for (GLfloat i = 0.1f; i < 1.1f; i = i + 0.1f) {

		//1
		arr[j] = 0.0f + i;
		arr[j + 1] = 1.0f;
		arr[j + 2] = 0.0f;

		arr[j + 3] = 0.0f + i;
		arr[j + 4] = -1.0f;
		arr[j + 5] = 0.0f;

		//2
		arr[j + 6] = 0.0f - i;
		arr[j + 7] = 1.0f;
		arr[j + 8] = 0.0f;

		arr[j + 9] = 0.0f - i;
		arr[j + 10] = -1.0f;
		arr[j + 11] = 0.0f;

		j = j + 12;
		fprintf(gbFile_RRJ, "j: %d\n", j);
	}


	//Horizontal Lines 20 
	fprintf(gbFile_RRJ, "               j: %d\n", j);
	for (GLfloat i = 0.1f; i < 1.1f; i = i + 0.1f) {
		//1
		arr[j] = -1.0f;
		arr[j + 1] = 0.0f + i;
		arr[j + 2] = 0.0f;

		arr[j + 3] = 1.0f;
		arr[j + 4] = 0.0f + i;
		arr[j + 5] = 0.0f;

		//2
		arr[j + 6] = -1.0f;
		arr[j + 7] = 0.0f - i;
		arr[j + 8] = 0.0f;

		arr[j + 9] = 1.0f;
		arr[j + 10] = 0.0f - i;
		arr[j + 11] = 0.0f;

		j = j + 12;
		fprintf(gbFile_RRJ, "j: %d\n", j);
	}
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
    GLfloat Axis_Position_RRJ[6];
    GLfloat Axis_Color_RRJ[6];
    GLfloat Circle_Position_RRJ[3000 * 3];
    GLfloat Circle_Color_RRJ[3000 * 3];

    glUseProgram(shaderProgramObject_RRJ);

    /********** Grid **********/
    translateMatrix_RRJ = vmath::mat4::identity();
    modelViewMatrix_RRJ = vmath::mat4::identity();
    modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(0.0f, 0.0f, -3.0f);
    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    glUniformMatrix4fv(mvpUniform_RRJ, 1, false, modelViewProjectionMatrix_RRJ);

    glBindVertexArray(vao_Grid_RRJ);
	glDrawArrays(GL_LINES, 0, 2 * 40);
    glBindVertexArray(0);




    /********** Axis **********/
    for (int i = 1; i <= 2; i++) {

	translateMatrix_RRJ = vmath::mat4::identity();
    	modelViewMatrix_RRJ = vmath::mat4::identity();
    	modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

    	translateMatrix_RRJ = vmath::translate(0.0f, 0.0f, -3.0f);
    	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
   	modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;
	
	glUniformMatrix4fv(mvpUniform_RRJ, 1, false, modelViewProjectionMatrix_RRJ);	
	
	if (i == 1) {
		Axis_Position_RRJ[0] = 0.0f;
		Axis_Position_RRJ[1] = 1.0f;
		Axis_Position_RRJ[2] = 0.0f;

		Axis_Position_RRJ[3] = 0.0f;
		Axis_Position_RRJ[4] = -1.0f;
		Axis_Position_RRJ[5] = 0.0f;

		Axis_Color_RRJ[0] = 1.0f;
		Axis_Color_RRJ[1] = 0.0f;
		Axis_Color_RRJ[2] = 0.0f;

		Axis_Color_RRJ[3] = 1.0f;
		Axis_Color_RRJ[4] = 0.0f;
		Axis_Color_RRJ[5] = 0.0f;
	}
	else if (i == 2) {
		Axis_Position_RRJ[0] = -1.0f;
		Axis_Position_RRJ[1] = 0.0f;
		Axis_Position_RRJ[2] = 0.0f;

		Axis_Position_RRJ[3] = 1.0f;
		Axis_Position_RRJ[4] = 0.0f;
		Axis_Position_RRJ[5] = 0.0f;


		Axis_Color_RRJ[0] = 0.0f;
		Axis_Color_RRJ[1] = 1.0f;
		Axis_Color_RRJ[2] = 0.0f;

		Axis_Color_RRJ[3] = 0.0f;
		Axis_Color_RRJ[4] = 1.0f;
		Axis_Color_RRJ[5] = 0.0f;

	}

	glBindVertexArray(vao_Axis_RRJ);
		/***** Position *****/
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Axis_Position_RRJ_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Axis_Position_RRJ), Axis_Position_RRJ, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/***** Color *****/
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Axis_Color_RRJ_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Axis_Color_RRJ), Axis_Color_RRJ, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glDrawArrays(GL_LINES, 0, 6);
	glBindVertexArray(0);

	}


    /********** TRIANGLE **********/
    translateMatrix_RRJ = vmath::mat4::identity();
    modelViewMatrix_RRJ = vmath::mat4::identity();
    modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(0.0f, 0.0f, -3.0f);
    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    glUniformMatrix4fv(mvpUniform_RRJ, 1, false, modelViewProjectionMatrix_RRJ);

    glBindVertexArray(vao_Triangle_RRJ);
    glDrawArrays(GL_LINE_LOOP, 0, 3);
    glBindVertexArray(0);



    /********** RECTANGLE ***********/
    translateMatrix_RRJ = vmath::mat4::identity();
    modelViewMatrix_RRJ = vmath::mat4::identity();
    modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(0.0f, 0.0f, -3.0f);
    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    glUniformMatrix4fv(mvpUniform_RRJ, 1, false, modelViewProjectionMatrix_RRJ);

    glBindVertexArray(vao_Rect_RRJ);
    glDrawArrays(GL_LINE_LOOP, 0, 4);
    glBindVertexArray(0);


    /********** Circle **********/
    for (int i = 1; i <= 2; i++) {
       
        translateMatrix_RRJ = vmath::mat4::identity();
        modelViewMatrix_RRJ = vmath::mat4::identity();
        modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

        translateMatrix_RRJ = vmath::translate(0.0f, 0.0f, -3.0f);
        modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
        modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

        glUniformMatrix4fv(mvpUniform_RRJ, 1, false, modelViewProjectionMatrix_RRJ);


        if (i == 1) {
            [self FillCircle_Position_RRJ: Circle_Position_RRJ withColor: Circle_Color_RRJ withFlag: 1];
            //FillCircle_Position_RRJ(Circle_Position_RRJ, Circle_Color_RRJ, 1);
        }
        else if (i == 2) {
            [self FillCircle_Position_RRJ: Circle_Position_RRJ withColor: Circle_Color_RRJ withFlag: 2];
            //FillCircle_Position_RRJ(Circle_Position_RRJ, Circle_Color_RRJ, 2);
        }


        glPointSize(1.500f);
            glBindVertexArray(vao_Circle_RRJ);
                glBindBuffer(GL_ARRAY_BUFFER, vbo_Circle_Position_RRJ_RRJ);
                glBufferData(GL_ARRAY_BUFFER, sizeof(Circle_Position_RRJ), Circle_Position_RRJ, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, 0);


            glBindBuffer(GL_ARRAY_BUFFER, vbo_Circle_Color_RRJ_RRJ);
                glBufferData(GL_ARRAY_BUFFER, sizeof(Circle_Color_RRJ), Circle_Color_RRJ, GL_DYNAMIC_DRAW);
                glBindBuffer(GL_ARRAY_BUFFER, 0);
            glDrawArrays(GL_POINTS, 0, 3000);

        glBindVertexArray(0);

    }


    glUseProgram(0);

    
    CGLFlushDrawable((CGLContextObj) [[self openGLContext] CGLContextObj]);
    CGLUnlockContext((CGLContextObj) [[self openGLContext] CGLContextObj]);
}



-(void)Calculation: (GLfloat[]) arr{
    GLfloat a, b, c;
    GLfloat s;

    //Distance Formula
    a = (GLfloat)sqrt(pow((arr[6] - arr[3]), 2) + pow((arr[7] - arr[4]), 2));
    b = (GLfloat)sqrt(pow((arr[6] - arr[0]), 2) + pow((arr[7] - arr[1]), 2));
    c = (GLfloat)sqrt(pow((arr[3] - arr[0]), 2) + pow((arr[4] - arr[1]), 2));

    s = (a + b + c) / 2;

    Incircle_Radius_RRJ = (GLfloat)(sqrt(s * (s - a) * (s - b) * (s - c)) / s);

    Incircle_Center_RRJ[0] = (a * arr[0] + b * arr[3] + c * arr[6]) / (a + b + c);
    Incircle_Center_RRJ[1] = (a * arr[1] + b * arr[4] + c * arr[7]) / (a + b + c);
    Incircle_Center_RRJ[2] = 0.0f;


    fprintf(gbFile_RRJ, "Incircle_Radius_RRJ: %f\n", Incircle_Radius_RRJ);
    fprintf(gbFile_RRJ, "InCenter x: %f      y: %f      z: %f     \n", Incircle_Center_RRJ[0], Incircle_Center_RRJ[1], Incircle_Center_RRJ[2]);

}


-(void)FillCircle_Position_RRJ: (GLfloat[]) arr withColor:(GLfloat[]) arrColor withFlag:(int)iFlag {

    if (iFlag == 1) {
        //InCircle
        int i = 0;
        for (int i = 0; i < 3000; i = i + 3) {
            GLfloat x = (GLfloat)(2.0f * M_PI * i / 3000);
            arr[i] = (GLfloat)(Incircle_Radius_RRJ * cos(x)) + Incircle_Center_RRJ[0];
            arr[i + 1] = (GLfloat)(Incircle_Radius_RRJ * sin(x)) + Incircle_Center_RRJ[1];
            arr[i + 2] = 0.0f;


            arrColor[i] = 1.0f;     //R
            arrColor[i + 1] = 1.0f;     //G
            arrColor[i + 2] = 0.0f;     //B
        }
        //fprintf(gbFile_RRJ, "i: %d\n", i);

    }
    else {
        //Outer Circle
        for (int i = 0; i < 3000; i = i + 3) {
            GLfloat x = (GLfloat)(2.0f * M_PI * i / 3000);
            arr[i] = (GLfloat)(1.0f * cos(x));
            arr[i + 1] = (GLfloat)(1.0f * sin(x));
            arr[i + 2] = 0.0f;

            arrColor[i] = 1.0f;     //R
            arrColor[i + 1] = 1.0f;     //G
            arrColor[i + 2] = 0.0f;     //B

            //fprintf(gbFile_RRJ, "x: %f    y: %f    z: %f \n", arr[i], arr[i + 1], arr[i + 2]);
        }
    }

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

    if (vbo_Circle_Color_RRJ_RRJ) {
        glDeleteBuffers(1, &vbo_Circle_Color_RRJ_RRJ);
        vbo_Circle_Color_RRJ_RRJ = 0;
    }

    if (vbo_Circle_Position_RRJ_RRJ) {
        glDeleteBuffers(1, &vbo_Circle_Position_RRJ_RRJ);
        vbo_Circle_Position_RRJ_RRJ = 0;
    }

    if (vao_Circle_RRJ) {
        glDeleteVertexArrays(1, &vao_Circle_RRJ);
        vao_Circle_RRJ = 0;
    }

    if (vbo_Rect_Color_RRJ_RRJ) {
        glDeleteBuffers(1, &vbo_Rect_Color_RRJ_RRJ);
        vbo_Rect_Color_RRJ_RRJ = 0;
    }

    if (vbo_Rect_Position_RRJ_RRJ) {
        glDeleteBuffers(1, &vbo_Rect_Position_RRJ_RRJ);
        vbo_Rect_Position_RRJ_RRJ = 0;
    }

    if (vao_Rect_RRJ) {
        glDeleteVertexArrays(1, &vao_Rect_RRJ);
        vao_Rect_RRJ = 0;
    }

    if (vbo_Triangle_Color_RRJ_RRJ) {
        glDeleteBuffers(1, &vbo_Triangle_Color_RRJ_RRJ);
        vbo_Triangle_Color_RRJ_RRJ = 0;
    }

    if (vbo_Triangle_Position_RRJ_RRJ) {
        glDeleteBuffers(1, &vbo_Triangle_Position_RRJ_RRJ);
        vbo_Triangle_Position_RRJ_RRJ = 0;
    }

    if (vao_Triangle_RRJ) {
        glDeleteVertexArrays(1, &vao_Triangle_RRJ);
        vao_Triangle_RRJ = 0;
    }


    if (vbo_Axis_Color_RRJ_RRJ) {
		glDeleteBuffers(1, &vbo_Axis_Color_RRJ_RRJ);
		vbo_Axis_Color_RRJ_RRJ = 0;
	}

	if (vbo_Axis_Position_RRJ_RRJ) {
		glDeleteBuffers(1, &vbo_Axis_Position_RRJ_RRJ);
		vbo_Axis_Position_RRJ_RRJ = 0;
	}

	if (vao_Axis_RRJ) {
		glDeleteVertexArrays(1, &vao_Axis_RRJ);
		vao_Axis_RRJ = 0;
	}


	if (vbo_Grid_Position_RRJ_RRJ) {
		glDeleteBuffers(1, &vbo_Grid_Position_RRJ_RRJ);
		vbo_Grid_Position_RRJ_RRJ = 0;
	}

	if (vao_Grid_RRJ) {
		glDeleteVertexArrays(1, &vao_Grid_RRJ);
		vao_Grid_RRJ = 0;
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


