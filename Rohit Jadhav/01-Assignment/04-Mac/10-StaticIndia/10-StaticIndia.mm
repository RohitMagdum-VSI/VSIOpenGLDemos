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

    [window_RRJ setTitle: @"Rohit_R_Jadhav-Mac-10-StaticIndia"];
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


        //For I
        GLuint vao_I_RRJ;
        GLuint vbo_I_Position_RRJ;
        GLuint vbo_I_Color_RRJ;

        //For N
        GLuint vao_N_RRJ;
        GLuint vbo_N_Position_RRJ;
        GLuint vbo_N_Color_RRJ;

        //For D
        GLuint vao_D_RRJ;
        GLuint vbo_D_Position_RRJ;
        GLuint vbo_D_Color_RRJ;

        //For A
        GLuint vao_A_RRJ;
        GLuint vbo_A_Position_RRJ;
        GLuint vbo_A_Color_RRJ;

        //For Flag
        GLuint vao_Flag_RRJ;
        GLuint vbo_Flag_Position_RRJ;
        GLuint vbo_Flag_Color_RRJ;

        GLuint mvpUniform_RRJ;
    
        vmath::mat4 translateMatrix_RRJ;
        vmath::mat4 rotateMatrix_RRJ;
        vmath::mat4 modelViewMatrix_RRJ;
        vmath::mat4 modelViewProjectionMatrix_RRJ;

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


    /********** Position And Color **********/
    GLfloat I_Position_RRJ[] = {
        //Top
        -0.3, 1.0, 0.0,
        0.3, 1.0, 0.0,

        -0.3, 1.010, 0.0,
        0.3, 1.010, 0.0,
        -0.3, 0.990, 0.0,
        0.3, 0.990, 0.0,

        -0.3, 1.020, 0.0,
        0.3, 1.020, 0.0,
        -0.3, 0.980, 0.0,
        0.3, 0.980, 0.0,




        //Mid
        0.0, 1.0, 0.0,
        0.0, -1.0, 0.0,

        0.01, 1.0, 0.0,
        0.01, -1.0, 0.0,
        -0.01, 1.0, 0.0,
        -0.01, -1.0, 0.0,

        0.02, 1.0, 0.0,
        0.02, -1.0, 0.0,
        -0.02, 1.0, 0.0,
        -0.02, -1.0, 0.0,


        //Bottom
        -0.3, -1.0, 0.0,
        0.3, -1.0, 0.0,

        -0.3, -1.010, 0.0,
        0.3, -1.010, 0.0,
        -0.3, -0.990, 0.0,
        0.3, -0.990, 0.0,

        -0.3, -1.020, 0.0,
        0.3, -1.020, 0.0,
        -0.3, -0.980, 0.0,
        0.3, -0.980, 0.0,
    };


    GLfloat I_Color_RRJ[] = {
        //Top
        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,

        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,
        
        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,


        //Mid
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        

        //Bottom
        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,

        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,

        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,
    };

    GLfloat N_Position_RRJ[] = {
        //Top
        0.0, 1.06, 0.0,
        0.0, -1.06, 0.0,

        0.01, 1.06, 0.0,
        0.01, -1.06, 0.0,
        -0.01, 1.06, 0.0,
        -0.01, -1.06, 0.0,

        0.02, 1.06, 0.0,
        0.02, -1.06, 0.0,
        -0.02, 1.06, 0.0,
        -0.02, -1.06, 0.0,


        //Mid
        0.75, 1.06, 0.0,
        0.75, -1.06, 0.0,

        0.76, 1.06, 0.0,
        0.76, -1.06, 0.0,
        0.74, 1.06, 0.0,
        0.74, -1.06, 0.0,

        0.77, 1.06, 0.0,
        0.77, -1.06, 0.0,
        0.73, 1.06, 0.0,
        0.73, -1.06, 0.0,


        //Bottom
        0.0, 1.06, 0.0,
        0.75, -1.06, 0.0,

        0.01, 1.06, 0.0,
        0.76, -1.06, 0.0,
        -0.01, 1.06, 0.0,
        0.74, -1.06, 0.0,

        0.02, 1.06, 0.0,
        0.77, -1.06, 0.0,
        -0.02, 1.06, 0.0,
        0.73, -1.06, 0.0
    };


    GLfloat N_Color_RRJ[] = {
        //Top
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        //Mid
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,


        //Bottom
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
    };

    GLfloat D_Position_RRJ[] = {
        //Left
        0.0, 1.0, 0.0,
        0.0, -1.0, 0.0,

        0.01, 1.0, 0.0,
        0.01, -1.0, 0.0,
        -0.01, 1.0, 0.0,
        -0.01, -1.0, 0.0,

        0.02, 1.0, 0.0,
        0.02, -1.0, 0.0,
        -0.02, 1.0, 0.0,
        -0.02, -1.0, 0.0,



        //Top
        -0.1, 1.0, 0.0,
        0.6, 1.0, 0.0,

        -0.1, 1.01, 0.0,
        0.6, 1.01, 0.0,
        -0.1, 0.990, 0.0,
        0.6, 0.990, 0.0,

        -0.1, 1.02, 0.0,
        0.6, 1.02, 0.0,
        -0.1, 0.980, 0.0,
        0.6, 0.980, 0.0,



        //Bottom
        -0.1, -1.0, 0.0,
        0.6, -1.0, 0.0,

        -0.1, -1.01, 0.0,
        0.6, -1.01, 0.0,
        -0.1, -0.990, 0.0,
        0.6, -0.990, 0.0,

        -0.1, -1.02, 0.0,
        0.6, -1.02, 0.0,
        -0.1, -0.980, 0.0,
        0.6, -0.980, 0.0,


        //Right
        0.6, 1.0, 0.0,
        0.6, -1.0, 0.0,

        0.61, 1.0, 0.0,
        0.61, -1.0, 0.0,
        0.59, 1.0, 0.0,
        0.59, -1.0, 0.0,

        0.62, 1.0, 0.0,
        0.62, -1.0, 0.0,
        0.58, 1.0, 0.0,
        0.58, -1.0, 0.0,
    };

    GLfloat D_Color_RRJ[] = {
        //Left
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        //Top
        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,

        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,

        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,

        //Bottom
        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,

        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,

        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,

        //Right
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
    };


    GLfloat A_Position_RRJ[] = {
        //Left
        0.0, 1.06, 0.0,
        -0.5, -1.06, 0.0,

        0.01, 1.06, 0.0,
        -0.49, -1.06, 0.0,
        0.01, 1.06, 0.0,
        -0.51, -1.06, 0.0,

        0.02, 1.06, 0.0,
        -0.48, -1.06, 0.0,
        0.02, 1.06, 0.0,
        -0.52, -1.06, 0.0,

        //Right
        0.0, 1.06, 0.0,
        0.5, -1.06, 0.0,

        0.01, 1.06, 0.0,
        0.49, -1.06, 0.0,
        0.01, 1.06, 0.0,
        0.51, -1.06, 0.0,

        0.02, 1.06, 0.0,
        0.48, -1.06, 0.0,
        0.02, 1.06, 0.0,
        0.52, -1.06, 0.0,
    };


    GLfloat A_Color_RRJ[] = {
        //Left
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,


        //Right
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
    };


    GLfloat Flag_Position_RRJ[] = {
       //Orange
        -0.210, 0.05, 0.0,
        0.245, 0.05, 0.0,

        -0.210, 0.06, 0.0,
        0.245, 0.06, 0.0,
        -0.210, 0.04, 0.0,
        0.245, 0.04, 0.0,

        -0.210, 0.07, 0.0,
        0.245, 0.07, 0.0,
        -0.210, 0.03, 0.0,
        0.245, 0.03, 0.0,

        //White
        -0.225, 0.0, 0.0,
        0.245, 0.0, 0.0,

        -0.225, 0.01, 0.0,
        0.245, 0.01, 0.0,
        -0.225, -0.01, 0.0,
        0.245, -0.01, 0.0,

        -0.225, 0.02, 0.0,
        0.245, 0.02, 0.0,
        -0.225, -0.02, 0.0,
        0.245, -0.02, 0.0,

        //Green
        -0.235, -0.05, 0.0,
        0.260, -0.05, 0.0,

        -0.235, -0.06, 0.0,
        0.260, -0.06, 0.0,
        -0.235, -0.04, 0.0,
        0.260, -0.04, 0.0,

        -0.235, -0.07, 0.0,
        0.260, -0.07, 0.0,
        -0.235, -0.03, 0.0,
        0.260, -0.03, 0.0,
    };


    GLfloat Flag_Color_RRJ[] = {
        //Orange
        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,

        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,

        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,
        1.0, 0.6, 0.2,

        //White
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,

        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,

        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,


        //Green
        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,

        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,

        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,
        0.0705, 0.533, 0.0274,
    };


    /********** I **********/
    glGenVertexArrays(1, &vao_I_RRJ);
    glBindVertexArray(vao_I_RRJ);

        /********** Position **********/
        glGenBuffers(1, &vbo_I_Position_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_I_Position_RRJ);
        glBufferData(GL_ARRAY_BUFFER, sizeof(I_Position_RRJ), I_Position_RRJ, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);


        /********** Color **********/
        glGenBuffers(1, &vbo_I_Color_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_I_Color_RRJ);
        glBufferData(GL_ARRAY_BUFFER, sizeof(I_Color_RRJ), I_Color_RRJ, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);



    /********** N **********/
    glGenVertexArrays(1, &vao_N_RRJ);
    glBindVertexArray(vao_N_RRJ);

        /********** Position **********/
        glGenBuffers(1, &vbo_N_Position_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_N_Position_RRJ);
        glBufferData(GL_ARRAY_BUFFER, sizeof(N_Position_RRJ), N_Position_RRJ, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);


        /********** Color **********/
        glGenBuffers(1, &vbo_N_Color_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_N_Color_RRJ);
        glBufferData(GL_ARRAY_BUFFER, sizeof(N_Color_RRJ), N_Color_RRJ, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);



    /********** D **********/
    glGenVertexArrays(1, &vao_D_RRJ);
    glBindVertexArray(vao_D_RRJ);

        /********** Position **********/
        glGenBuffers(1, &vbo_D_Position_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_D_Position_RRJ);
        glBufferData(GL_ARRAY_BUFFER, sizeof(D_Position_RRJ), D_Position_RRJ, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);


        /********** Color **********/
        glGenBuffers(1, &vbo_D_Color_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_D_Color_RRJ);
        glBufferData(GL_ARRAY_BUFFER, sizeof(D_Color_RRJ), D_Color_RRJ, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);



    /********** A **********/
    glGenVertexArrays(1, &vao_A_RRJ);
    glBindVertexArray(vao_A_RRJ);

        /********** Position **********/
        glGenBuffers(1, &vbo_A_Position_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_A_Position_RRJ);
        glBufferData(GL_ARRAY_BUFFER, sizeof(A_Position_RRJ), A_Position_RRJ, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);


        /********** Color **********/
        glGenBuffers(1, &vbo_A_Color_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_A_Color_RRJ);
        glBufferData(GL_ARRAY_BUFFER, sizeof(A_Color_RRJ), A_Color_RRJ, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);


    /********** Flag **********/
    glGenVertexArrays(1, &vao_Flag_RRJ);
    glBindVertexArray(vao_Flag_RRJ);

        /********** Position **********/
        glGenBuffers(1, &vbo_Flag_Position_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_Flag_Position_RRJ);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Flag_Position_RRJ), Flag_Position_RRJ, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);


        /********** Color **********/
        glGenBuffers(1, &vbo_Flag_Color_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_Flag_Color_RRJ);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Flag_Color_RRJ), Flag_Color_RRJ, GL_STATIC_DRAW);
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

    
    glUseProgram(shaderProgramObject_RRJ);

        [self My_I: -2.0f yTrans:0.0f  zTrans:-8.0f  fWidth:20.0f];
        [self My_N: -1.35f  yTrans:0.0f zTrans:-8.0f fWidth:20.0f];
        [self My_D: -0.15f  yTrans:0.0f zTrans:-8.0f fWidth:20.0f];
        [self My_I: 1.02f yTrans:0.0f zTrans:-8.0f fWidth:20.0f];
        [self My_A: 2.0f yTrans:0.0f zTrans:-8.0f fWidth:20.0f];
        [self My_Flag: 2.0f yTrans:0.0f zTrans:-8.0f fWidth:20.0f];

    glUseProgram(0);

    
    CGLFlushDrawable((CGLContextObj) [[self openGLContext] CGLContextObj]);
    CGLUnlockContext((CGLContextObj) [[self openGLContext] CGLContextObj]);
}


-(void)My_I:(GLfloat)x yTrans:(GLfloat)y zTrans:(GLfloat)z fWidth:(GLfloat)fWidth{

    translateMatrix_RRJ = vmath::mat4::identity();
    modelViewMatrix_RRJ = vmath::mat4::identity();
    modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(x, y, z);
    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    glUniformMatrix4fv(mvpUniform_RRJ, 1, GL_FALSE, modelViewProjectionMatrix_RRJ);

    glLineWidth(fWidth);
    glBindVertexArray(vao_I_RRJ);
    glDrawArrays(GL_LINES, 0, 30);
    glBindVertexArray(0);
}

-(void)My_N:(GLfloat)x yTrans:(GLfloat)y zTrans:(GLfloat)z fWidth:(GLfloat)fWidth{
    translateMatrix_RRJ = vmath::mat4::identity();
    modelViewMatrix_RRJ = vmath::mat4::identity();
    modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(x, y, z);
    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    glUniformMatrix4fv(mvpUniform_RRJ, 1, GL_FALSE, modelViewProjectionMatrix_RRJ);

    glLineWidth(fWidth);
    glBindVertexArray(vao_N_RRJ);
    glDrawArrays(GL_LINES, 0, 30);
    glBindVertexArray(0);
}


-(void)My_D: (GLfloat)x yTrans:(GLfloat)y zTrans:(GLfloat)z fWidth:(GLfloat)fWidth{
    translateMatrix_RRJ = vmath::mat4::identity();
    modelViewMatrix_RRJ = vmath::mat4::identity();
    modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(x, y, z);
    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    glUniformMatrix4fv(mvpUniform_RRJ, 1, GL_FALSE, modelViewProjectionMatrix_RRJ);

    glLineWidth(fWidth);
    glBindVertexArray(vao_D_RRJ);
    glDrawArrays(GL_LINES, 0, 40);
    glBindVertexArray(0);
}


-(void)My_A:(GLfloat)x yTrans:(GLfloat)y zTrans:(GLfloat)z fWidth:(GLfloat)fWidth{
    translateMatrix_RRJ = vmath::mat4::identity();
    modelViewMatrix_RRJ = vmath::mat4::identity();
    modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(x, y, z);
    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    glUniformMatrix4fv(mvpUniform_RRJ, 1, GL_FALSE, modelViewProjectionMatrix_RRJ);

    glLineWidth(fWidth);
    glBindVertexArray(vao_A_RRJ);
    glDrawArrays(GL_LINES, 0, 20);
    glBindVertexArray(0);
}

-(void)My_Flag:(GLfloat)x yTrans:(GLfloat)y zTrans:(GLfloat)z fWidth:(GLfloat)fWidth{
    translateMatrix_RRJ = vmath::mat4::identity();
    modelViewMatrix_RRJ = vmath::mat4::identity();
    modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(x, y, z);
    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    glUniformMatrix4fv(mvpUniform_RRJ, 1, GL_FALSE, modelViewProjectionMatrix_RRJ);

    glLineWidth(fWidth);
    glBindVertexArray(vao_Flag_RRJ);
    glDrawArrays(GL_LINES, 0, 30);
    glBindVertexArray(0);
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
    
    //Flag
    if (vbo_Flag_Color_RRJ) {
        glDeleteBuffers(1, &vbo_Flag_Color_RRJ);
        vbo_Flag_Color_RRJ = 0;
    }

    if (vbo_Flag_Position_RRJ) {
        glDeleteBuffers(1, &vbo_Flag_Position_RRJ);
        vbo_Flag_Position_RRJ = 0;
    }

    if (vao_Flag_RRJ) {
        glDeleteVertexArrays(1, &vao_Flag_RRJ);
        vao_Flag_RRJ = 0;
    }


    //A
    if (vbo_A_Color_RRJ) {
        glDeleteBuffers(1, &vbo_A_Color_RRJ);
        vbo_A_Color_RRJ = 0;
    }

    if (vbo_A_Position_RRJ) {
        glDeleteBuffers(1, &vbo_A_Position_RRJ);
        vbo_A_Position_RRJ = 0;
    }

    if (vao_A_RRJ) {
        glDeleteVertexArrays(1, &vao_A_RRJ);
        vao_A_RRJ = 0;
    }

    //D
    if (vbo_D_Color_RRJ) {
        glDeleteBuffers(1, &vbo_D_Color_RRJ);
        vbo_D_Color_RRJ = 0;
    }

    if (vbo_D_Position_RRJ) {
        glDeleteBuffers(1, &vbo_D_Position_RRJ);
        vbo_D_Position_RRJ = 0;
    }

    if (vao_D_RRJ) {
        glDeleteVertexArrays(1, &vao_D_RRJ);
        vao_D_RRJ = 0;
    }

    //N
    if (vbo_N_Color_RRJ) {
        glDeleteBuffers(1, &vbo_N_Color_RRJ);
        vbo_N_Color_RRJ = 0;
    }

    if (vbo_N_Position_RRJ) {
        glDeleteBuffers(1, &vbo_N_Position_RRJ);
        vbo_N_Position_RRJ = 0;
    }

    if (vao_N_RRJ) {
        glDeleteVertexArrays(1, &vao_N_RRJ);
        vao_N_RRJ = 0;
    }

    //I
    if (vbo_I_Color_RRJ) {
        glDeleteBuffers(1, &vbo_I_Color_RRJ);
        vbo_I_Color_RRJ = 0;
    }

    if (vbo_I_Position_RRJ) {
        glDeleteBuffers(1, &vbo_I_Position_RRJ);
        vbo_I_Position_RRJ = 0;
    }

    if (vao_I_RRJ) {
        glDeleteVertexArrays(1, &vao_I_RRJ);
        vao_I_RRJ = 0;
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




