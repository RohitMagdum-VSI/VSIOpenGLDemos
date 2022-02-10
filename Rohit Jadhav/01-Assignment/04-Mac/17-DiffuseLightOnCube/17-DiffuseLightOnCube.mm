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

    [window_RRJ setTitle: @"Rohit_R_Jadhav-Mac-17-DiffuseLightOnCube"];
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



//For Light
bool bLights_RRJ = false;


/********** GLView_RRJ **********/
@implementation GLView_RRJ
{
    @private
        CVDisplayLinkRef displayLink_RRJ;

        GLuint vertexShaderObject_RRJ;
        GLuint fragmentShaderObject_RRJ;
        GLuint shaderProgramObject_RRJ;

        GLuint vao_Cube_RRJ;
        GLuint vbo_Cube_Position_RRJ;
        GLuint vbo_Cube_Normal_RRJ;

        //For Uniform
	GLuint mvUniform_RRJ;
	GLuint projectionUniform_RRJ;
	GLuint LdUniform_RRJ;
	GLuint KdUniform_RRJ;
	GLuint lightPositionUniform_RRJ;
	GLuint LKeyPressUniform_RRJ;

    
        GLfloat angle_Pyramid_RRJ;
        GLfloat angle_Cube_RRJ;

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

        NSOpenGLPixelFormat *pixelFormat_RRJ = [[[NSOpenGLPixelFormat alloc] initWithAttributes:attribs_RRJ] autorelease];

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
	"in vec3 vNormal;" \
	"uniform mat4 u_mv_matrix;" \
	"uniform mat4 u_projection_matrix;" \
	"uniform vec3 u_Ld;" \
	"uniform vec3 u_Kd;" \
	"uniform vec4 u_light_position;" \
	"uniform int u_LKeyPress;" \
	"out vec3 diffuseColor;" \
	"void main(void)" \
	"{" \
		"if(u_LKeyPress == 1)"
		"{" \
			"vec4 eye_coordinate = u_mv_matrix * vPosition;" \
			"mat3 normalMatrix = mat3(transpose(inverse(u_mv_matrix)));" \
			"vec3 tNorm = normalize(vec3(normalMatrix * vNormal));" \
			"vec3 source = normalize(vec3(u_light_position - eye_coordinate));" \
			"diffuseColor = u_Ld * u_Kd * max(dot(source, tNorm), 0.0f);" \
		"}" \
		"gl_Position = u_projection_matrix * u_mv_matrix * vPosition;" \
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
	"in vec3 diffuseColor;" \
	"out vec4 FragColor;" \
	"uniform int u_LKeyPress;" \
	"void main(void)" \
	"{" \
		"if(u_LKeyPress == 1)" \
		"{" \
			"FragColor = vec4(diffuseColor, 1.0);" \
		"}" \
		"else" \
		"{" \
			"FragColor = vec4(1.0, 1.0, 1.0, 1.0);" \
		"}" \
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
    glBindAttribLocation(shaderProgramObject_RRJ, AMC_ATTRIBUTE_NORMAL, "vNormal");

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


    mvUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_mv_matrix");
    projectionUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_projection_matrix");
    LdUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_Ld");
    KdUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_Kd");
    lightPositionUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_light_position"); 
    LKeyPressUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_LKeyPress");


    /********** Pyramid Position, vao, vbo **********/
    GLfloat cube_Position_RRJ[] = {
		//Top
		1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,
		//Bottom
		1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		//Front
		1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		//Back
		1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,
		//Right
		1.0f, 1.0f, -1.0f,
		1.0f, 1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f,
		//Left
		-1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f
	};
    
    
    GLfloat cube_Normal_RRJ[] = {
        	//Top
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,

		//Bottom
		0.0f, -1.0f, 0.0f,
		0.0f, -1.0f, 0.0f,
		0.0f, -1.0f, 0.0f,
		0.0f, -1.0f, 0.0f,

		//Front
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,

		//Back
		0.0f, 0.0f, -1.0f,
		0.0f, 0.0f, -1.0f,
		0.0f, 0.0f, -1.0f,
		0.0f, 0.0f, -1.0f,

		//Right
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,

		//Left
		-1.0f, 0.0f, 0.0f,
		-1.0f, 0.0f, 0.0f,
		-1.0f, 0.0f, 0.0f,
		-1.0f, 0.0f, 0.0f
    };

    

    /********* Vao For Cube *********/
	glGenVertexArrays(1, &vao_Cube_RRJ);
	glBindVertexArray(vao_Cube_RRJ);
		
		/********** Position *********/
		glGenBuffers(1, &vbo_Cube_Position_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Cube_Position_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(cube_Position_RRJ), cube_Position_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
    
	        /********** Normal *********/
	        glGenBuffers(1, &vbo_Cube_Normal_RRJ);
	        glBindBuffer(GL_ARRAY_BUFFER, vbo_Cube_Normal_RRJ);
	        glBufferData(GL_ARRAY_BUFFER, sizeof(cube_Normal_RRJ), cube_Normal_RRJ, GL_STATIC_DRAW);
	        glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	        glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
	        glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);


    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glClearDepth(1.0f);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    angle_Pyramid_RRJ = 0.0f;
    angle_Cube_RRJ = 360.0f;
    perspectiveProjectionMatrix_RRJ = vmath::mat4::identity();

    CVDisplayLinkCreateWithActiveCGDisplays(&displayLink_RRJ);
    CVDisplayLinkSetOutputCallback(displayLink_RRJ, &MyDisplayLinkCallback, self);
    CGLContextObj cglContext_RRJ = (CGLContextObj)[[self openGLContext]CGLContextObj];
    CGLPixelFormatObj cglPixelFormat_RRJ = (CGLPixelFormatObj)[[self pixelFormat]CGLPixelFormatObj];
    CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext(displayLink_RRJ, cglContext_RRJ, cglPixelFormat_RRJ);
    CVDisplayLinkStart(displayLink_RRJ);
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
    vmath::mat4 rotateMatrix_RRJ;
    vmath::mat4 scaleMatrix_RRJ;
    vmath::mat4 modelViewMatrix_RRJ;
    vmath::mat4 modelViewProjectionMatrix_RRJ;



    glUseProgram(shaderProgramObject_RRJ);

    

    /********** CUBE **********/
    translateMatrix_RRJ = vmath::mat4::identity();
    scaleMatrix_RRJ = vmath::mat4::identity();
    rotateMatrix_RRJ = vmath::mat4::identity();
    modelViewMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(0.0f, 0.0f, -5.0f);
    scaleMatrix_RRJ = vmath::scale(0.9f, 0.9f, 0.9f);
    rotateMatrix_RRJ = vmath::rotate(angle_Cube_RRJ, 0.0f, 0.0f) * vmath::rotate(0.0f, angle_Cube_RRJ, 0.0f) * vmath::rotate(0.0f, 0.0f, angle_Cube_RRJ);
    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ * scaleMatrix_RRJ * rotateMatrix_RRJ;

    
    glUniformMatrix4fv(mvUniform_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
    glUniformMatrix4fv(projectionUniform_RRJ, 1, GL_FALSE, perspectiveProjectionMatrix_RRJ);
	

    if (bLights_RRJ == true) {

		glUniform1i(LKeyPressUniform_RRJ, 1);
		glUniform3f(LdUniform_RRJ, 1.0f, 1.0f, 1.0f);
		glUniform3f(KdUniform_RRJ, 0.5f, 0.5f, 0.5f);
		glUniform4f(lightPositionUniform_RRJ, 0.0f, 0.0f, 0.0f, 1.0f);	// 0.0f, 0.0f, 2.0f, 1.0f;
	}
	else
		glUniform1i(LKeyPressUniform_RRJ, 0);


    glBindVertexArray(vao_Cube_RRJ);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
        glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
        glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
        glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
        glDrawArrays(GL_TRIANGLE_FAN, 20, 4);
    glBindVertexArray(0);

    glUseProgram(0);

    angle_Cube_RRJ -= 1.0f;
    if(angle_Cube_RRJ < 0.0f)
        angle_Cube_RRJ = 360.0f;
    
    
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

        case 'L':
        case 'l':
        	if(bLights_RRJ == false)
        		bLights_RRJ = true;
        	else 
        		bLights_RRJ = false;
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
    
    if(vbo_Cube_Normal_RRJ){
        glDeleteBuffers(1, &vbo_Cube_Normal_RRJ);
        vbo_Cube_Normal_RRJ = 0;
    }
    
    if(vbo_Cube_Position_RRJ){
        glDeleteBuffers(1, &vbo_Cube_Position_RRJ);
        vbo_Cube_Position_RRJ = 0;
    }

    if(vao_Cube_RRJ){
        glDeleteVertexArrays(1, &vao_Cube_RRJ);
        vao_Cube_RRJ = 0;
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




