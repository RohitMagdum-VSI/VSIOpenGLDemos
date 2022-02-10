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

    [window_RRJ setTitle: @"Rohit_R_Jadhav-Mac-11-DynamicIndia"];
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



//For Plane Movement and Translation
#define NOT_REACH 0
#define HALF_WAY 1
#define REACH 2
#define END 3

GLfloat Plane1_Count_RRJ = 1000.0f;
GLfloat Plane2_Count_RRJ = 1000.0f;
GLfloat Plane3_Count_RRJ = 1000.0f;

int bPlane1Reached_RRJ = NOT_REACH;
int bPlane2Reached_RRJ = NOT_REACH;
int bPlane3Reached_RRJ = NOT_REACH;
int iFadingFlag1_RRJ = 0;
int iFadingFlag2_RRJ = 0;
int iFadingFlag3_RRJ = 0;


//For Sequence
GLuint iSequence_RRJ = 1;

//D Fading
GLfloat D_Color_RRJ[] = {
    //Left
    1.0, 0.6, 0.2, 0.0,
    0.0705, 0.533, 0.0274,0.0,

    1.0, 0.6, 0.2,0.0,
    0.0705, 0.533, 0.0274,0.0,
    1.0, 0.6, 0.2,0.0,
    0.0705, 0.533, 0.0274,0.0,

    1.0, 0.6, 0.2,0.0,
    0.0705, 0.533, 0.0274,0.0,
    1.0, 0.6, 0.2,0.0,
    0.0705, 0.533, 0.0274,0.0,

    //Top
    1.0, 0.6, 0.2,0.0,
    1.0, 0.6, 0.2,0.0,

    1.0, 0.6, 0.2,0.0,
    1.0, 0.6, 0.2,0.0,
    1.0, 0.6, 0.2,0.0,
    1.0, 0.6, 0.2,0.0,

    1.0, 0.6, 0.2,0.0,
    1.0, 0.6, 0.2,0.0,
    1.0, 0.6, 0.2,0.0,
    1.0, 0.6, 0.2,0.0,

    //Bottom
    0.0705, 0.533, 0.0274,0.0,
    0.0705, 0.533, 0.0274,0.0,

    0.0705, 0.533, 0.0274,0.0,
    0.0705, 0.533, 0.0274,0.0,
    0.0705, 0.533, 0.0274,0.0,
    0.0705, 0.533, 0.0274,0.0,

    0.0705, 0.533, 0.0274,0.0,
    0.0705, 0.533, 0.0274,0.0,
    0.0705, 0.533, 0.0274,0.0,
    0.0705, 0.533, 0.0274,0.0,

    //Right
    1.0, 0.6, 0.2,0.0,
    0.0705, 0.533, 0.0274,0.0,

    1.0, 0.6, 0.2,0.0,
    0.0705, 0.533, 0.0274,0.0,
    1.0, 0.6, 0.2,0.0,
    0.0705, 0.533, 0.0274,0.0,

    1.0, 0.6, 0.2,0.0,
    0.0705, 0.533, 0.0274,0.0,
    1.0, 0.6, 0.2,0.0,
    0.0705, 0.533, 0.0274,0.0,
};

GLfloat fD_Fading_RRJ = 0.0f;

GLfloat Fading_Flag_Position_RRJ[] = {
    //Top
    -1.0, 0.1, 0.0,
    -0.50, 0.1, 0.0,

    -1.0, 0.11, 0.0,
    -0.50, 0.11, 0.0,
    -1.0, 0.09, 0.0,
    -0.50, 0.09, 0.0,

    -1.0, 0.12, 0.0,
    -0.50, 0.12, 0.0,
    -1.0, 0.08, 0.0,
    -0.50, 0.08, 0.0,

    //Middle
    -1.0, 0.0, 0.0,
    -0.50, 0.0, 0.0,

    -1.0, 0.01, 0.0,
    -0.50, 0.01, 0.0,
    -1.0, -0.01, 0.0,
    -0.50, -0.01, 0.0,

    -1.0, 0.02, 0.0,
    -0.50, 0.02, 0.0,
    -1.0, -0.02, 0.0,
    -0.50, -0.02, 0.0,

    //Bottom
    -1.0, -0.1, 0.0,
    -0.5, -0.1, 0.0,

    -1.0, -0.11, 0.0,
    -0.5, -0.11, 0.0,
    -1.0, -0.09, 0.0,
    -0.5, -0.09, 0.0,

    -1.0, -0.12, 0.0,
    -0.5, -0.12, 0.0,
    -1.0, -0.08, 0.0,
    -0.5, -0.08, 0.0,
};



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


        //For V A used in INDIA is Without - therfore V verticaly inverted
        GLuint vao_V_RRJ;
        GLuint vbo_V_Position_RRJ;
        GLuint vbo_V_Color_RRJ;

        //For F
        GLuint vao_F_RRJ;
        GLuint vbo_F_Position_RRJ;
        GLuint vbo_F_Color_RRJ;




        //For Flag
        GLuint vao_Flag_RRJ;
        GLuint vbo_Flag_Position_RRJ;
        GLuint vbo_Flag_Color_RRJ;

        GLuint mvpUniform_RRJ;

	   //For Plane's Triangle Part
	   GLuint vao_Plane_Triangle_RRJ;
	   GLuint vbo_Plane_Triangle_Position_RRJ;
	   GLuint vbo_Plane_Triangle_Color_RRJ;

	   //For Plane's Rectangle Part
	   GLuint vao_Plane_Rect_RRJ;
	   GLuint vbo_Plane_Rect_Position_RRJ;
	   GLuint vbo_Plane_Rect_Color_RRJ;

	   //For Plane's Polygon Part
	   GLuint vao_Plane_Polygon_RRJ;
	   GLuint vbo_Plane_Polygon_Position_RRJ;
	   GLuint vbo_Plane_Polygon_Color_RRJ;

	   //For Fading Flag
	   GLuint vao_Fading_Flag_RRJ;
	   GLuint vbo_Fading_Flag_Position_RRJ;
	   GLuint vbo_Fading_Flag_Color_RRJ;


	

    
        vmath::mat4 translateMatrix_RRJ;
        vmath::mat4 scaleMatrix_RRJ;
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

    


    GLfloat A_Position_RRJ[] = {
       0.0, 1.06, 0.0,
        -0.5, -1.06, 0.0,

        0.0, 1.06, 0.0,
        0.5, -1.06, 0.0,

        -0.250, 0.0, 0.0,
        0.25, 0.0, 0.0,
    };


    GLfloat A_Color_RRJ[] = {
       1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,

        1.0, 0.6, 0.2,
        0.0705, 0.533, 0.0274,
    };


    GLfloat V_Position_RRJ[] = {
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


    GLfloat V_Color_RRJ[] = {
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


    GLfloat F_Position_RRJ[]= {
        0.10f, 1.0f, 0.0f,
        0.10f, -1.0f, 0.0f,

        0.00f, 1.0f, 0.0f,
        0.90f, 1.0f, 0.0f,

        0.10f, 0.1f, 0.0f,
        0.80f, 0.1f, 0.0f
    };


    GLfloat F_Color_RRJ[] = {
        1.0f, 0.6f, 0.2f,
        0.0705f, 0.533f, 0.0274f,

        1.0f, 0.6f, 0.2f,
        1.0f, 0.6f, 0.2f,

        0.0705f, 0.533f, 0.0274f,
        0.0705f, 0.533f, 0.0274f
    };


    GLfloat Flag_Position_RRJ[] = {
       //Orange
        -0.207, 0.1, 0.0,
        0.227, 0.1, 0.0,

        -0.207, 0.11, 0.0,
        0.227, 0.11, 0.0,
        -0.207, 0.09, 0.0,
        0.227, 0.09, 0.0,

        -0.207, 0.12, 0.0,
        0.227, 0.12, 0.0,
        -0.207, 0.08, 0.0,
        0.227, 0.08, 0.0,

        //White
        -0.218, 0.0, 0.0,
        0.239, 0.0, 0.0,

        -0.218, 0.01, 0.0,
        0.239, 0.01, 0.0,
        -0.218, -0.01, 0.0,
        0.239, -0.01, 0.0,

        -0.218, 0.02, 0.0,
        0.239, 0.02, 0.0,
        -0.218, -0.02, 0.0,
        0.239, -0.02, 0.0,

        //Green
        -0.245, -0.1, 0.0,
        0.255, -0.1, 0.0,

        -0.245, -0.11, 0.0,
        0.255, -0.11, 0.0,
        -0.245, -0.09, 0.0,
        0.255, -0.09, 0.0,

        -0.245, -0.12, 0.0,
        0.255, -0.12, 0.0,
        -0.245, -0.08, 0.0,
        0.255, -0.08, 0.0,
    };


    GLfloat Flag_Color_RRJ[] = {
        //Orange
        0.0, 0.0, 0.0,
        1.0, 0.6, 0.2,

        0.0, 0.0, 0.0,
        1.0, 0.6, 0.2,
        0.0, 0.0, 0.0,
        1.0, 0.6, 0.2,

        0.0, 0.0, 0.0,
        1.0, 0.6, 0.2,
        0.0, 0.0, 0.0,
        1.0, 0.6, 0.2,

        //White
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0,

        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0,
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0,

        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0,
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0,


        //Green
        0.0, 0.0, 0.0,
        0.0705, 0.533, 0.0274,

        0.0, 0.0, 0.0,
        0.0705, 0.533, 0.0274,
        0.0, 0.0, 0.0,
        0.0705, 0.533, 0.0274,

        0.0, 0.0, 0.0,
        0.0705, 0.533, 0.0274,
        0.0, 0.0, 0.0,
        0.0705, 0.533, 0.0274,
    };




    GLfloat Plane_Triangle_Position_RRJ[] = {
		//Front
		5.0f, 0.0f, 0.0f,
		2.50f, 0.65f, 0.0f,
		2.50f, -0.65f, 0.0f
	};

	GLfloat Plane_Triangle_Color_RRJ[] = {
		//Front
		0.7294f, 0.8862f, 0.9333f,	//Power Blue
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f
	};


	GLfloat Plane_Rect_Position_RRJ[] = {
		//Middle
		2.50f, 0.65f, 0.0f,
		-2.50f, 0.65f, 0.0f,
		-2.50f, -0.65f, 0.0f,
		2.50f, -0.65f, 0.0f,

		//Upper_Fin
		0.75f, 0.65f, 0.0f,
		-1.20f, 2.5f, 0.0f,
		-2.50f, 2.5f, 0.0f,
		-2.0f, 0.65f, 0.0f,

		//Lower_Fin
		0.75f, -0.65f, 0.0f,
		-1.20f, -2.50f, 0.0f,
		-2.50f, -2.50f, 0.0f,
		-2.0f, -0.65f, 0.0f,

		//Back
		-2.50f, 0.65f, 0.0f,
		-3.0f, 0.75f, 0.0f,
		-3.0f, -0.75f, 0.0f,
		-2.5f, -0.65f, 0.0f
	};


	GLfloat Plane_Rect_Color_RRJ[] = {
		//Middle
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,

		//Upper_Fin
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,

		//Lower_Fin
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,

		//Back
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f
	};

	GLfloat Plane_Polygon_Position_RRJ[] = {
		//Upper Tail
		-3.0f, 0.75f, 0.0f,
		-3.90f, 1.5f, 0.0f,
		-4.5f, 1.5f, 0.0f,
		-4.0f, 0.0f, 0.0f,
		-3.0f, 0.0f, 0.0f,

		//Lower Tail
		-3.0f, -0.75f, 0.0f,
		-3.90f, -1.5f, 0.0f,
		-4.5f, -1.5f, 0.0f,
		-4.0f, 0.0f, 0.0f,
		-3.0f, 0.0f, 0.0f
	};

	GLfloat Plane_Polygon_Color_RRJ[] = {
		//Upper Tail
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,

		//Lower Tail
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f
	};


	GLfloat Fading_Flag_Color_RRJ[] = {
		//Orange
        0.0, 0.0, 0.0,
        1.0, 0.6, 0.2,

        0.0, 0.0, 0.0,
        1.0, 0.6, 0.2,
        0.0, 0.0, 0.0,
        1.0, 0.6, 0.2,

        0.0, 0.0, 0.0,
        1.0, 0.6, 0.2,
        0.0, 0.0, 0.0,
        1.0, 0.6, 0.2,



        //White
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0,

        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0,
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0,

        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0,
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0,


        //Green
        0.0, 0.0, 0.0,
        0.0705, 0.533, 0.0274,

        0.0, 0.0, 0.0,
        0.0705, 0.533, 0.0274,
        0.0, 0.0, 0.0,
        0.0705, 0.533, 0.0274,

        0.0, 0.0, 0.0,
        0.0705, 0.533, 0.0274,
        0.0, 0.0, 0.0,
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
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 4, GL_FLOAT, GL_FALSE, 0, NULL);
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



    /********** V **********/
    glGenVertexArrays(1, &vao_V_RRJ);
    glBindVertexArray(vao_V_RRJ);

        /********** Position **********/
        glGenBuffers(1, &vbo_V_Position_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_V_Position_RRJ);
        glBufferData(GL_ARRAY_BUFFER, sizeof(V_Position_RRJ), V_Position_RRJ, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);


        /********** Color **********/
        glGenBuffers(1, &vbo_V_Color_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_V_Color_RRJ);
        glBufferData(GL_ARRAY_BUFFER, sizeof(V_Color_RRJ), V_Color_RRJ, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);


    /********** F **********/
    glGenVertexArrays(1, &vao_F_RRJ);
    glBindVertexArray(vao_F_RRJ);

        /********** Position **********/
        glGenBuffers(1, &vbo_F_Position_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_F_Position_RRJ);
        glBufferData(GL_ARRAY_BUFFER, sizeof(F_Position_RRJ), F_Position_RRJ, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);


        /********** Color **********/
        glGenBuffers(1, &vbo_F_Color_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_F_Color_RRJ);
        glBufferData(GL_ARRAY_BUFFER, sizeof(F_Color_RRJ), F_Color_RRJ, GL_STATIC_DRAW);
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






   	/********** Plane's Triangle Part **********/
	glGenVertexArrays(1, &vao_Plane_Triangle_RRJ);
	glBindVertexArray(vao_Plane_Triangle_RRJ);

		/********** Position **********/
		glGenBuffers(1, &vbo_Plane_Triangle_Position_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Plane_Triangle_Position_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Plane_Triangle_Position_RRJ), Plane_Triangle_Position_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/********** Color **********/
		glGenBuffers(1, &vbo_Plane_Triangle_Color_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Plane_Triangle_Color_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Plane_Triangle_Color_RRJ), Plane_Triangle_Color_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);



	/********** Plane's Rectangle Part **********/
	glGenVertexArrays(1, &vao_Plane_Rect_RRJ);
	glBindVertexArray(vao_Plane_Rect_RRJ);

		/********** Position **********/
		glGenBuffers(1, &vbo_Plane_Rect_Position_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Plane_Rect_Position_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Plane_Rect_Position_RRJ), Plane_Rect_Position_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/********** Color **********/
		glGenBuffers(1, &vbo_Plane_Rect_Color_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Plane_Rect_Color_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Plane_Rect_Color_RRJ), Plane_Rect_Color_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);




	/********** Plane's Polygon Part **********/
	glGenVertexArrays(1, &vao_Plane_Polygon_RRJ);
	glBindVertexArray(vao_Plane_Polygon_RRJ);

		/********** Position **********/
		glGenBuffers(1, &vbo_Plane_Polygon_Position_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Plane_Polygon_Position_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Plane_Polygon_Position_RRJ), Plane_Polygon_Position_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/********** Color **********/
		glGenBuffers(1, &vbo_Plane_Polygon_Color_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Plane_Polygon_Color_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Plane_Polygon_Color_RRJ), Plane_Polygon_Color_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);




	/********** Fading Flag **********/
	glGenVertexArrays(1, &vao_Fading_Flag_RRJ);
	glBindVertexArray(vao_Fading_Flag_RRJ);

		/********** Position **********/
		glGenBuffers(1, &vbo_Fading_Flag_Position_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Fading_Flag_Position_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Fading_Flag_Position_RRJ), Fading_Flag_Position_RRJ, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** Color **********/
		glGenBuffers(1, &vbo_Fading_Flag_Color_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Fading_Flag_Color_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Fading_Flag_Color_RRJ), Fading_Flag_Color_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);




    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glClearDepth(1.0f);


    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);


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

    
    
    //For India
    static GLfloat fXTranslation = 0.0f;
    static GLfloat fYTranslation = 0.0f;

    //For Plane
    static GLfloat angle_Plane1_RRJ = (GLfloat)(M_PI);
    static GLfloat angle_Plane3_RRJ = (GLfloat)(M_PI);

    static GLfloat XTrans_Plane1_RRJ = 0.0f;
    static GLfloat YTrans_Plane1_RRJ = 0.0f;

    static GLfloat XTrans_Plane2_RRJ = 0.0f;

    static GLfloat XTrans_Plane3_RRJ = 0.0f;
    static GLfloat YTrans_Plane3_RRJ = 0.0f;

    static GLfloat ZRot_Plane1_RRJ = -60.0f;
    static GLfloat ZRot_Plane3_RRJ = 60.0f;

    
    
    glUseProgram(shaderProgramObject_RRJ);

       switch(iSequence_RRJ){
            case 1:
                [self My_Letters: 'I' xTrans:(-7.5f + fXTranslation) yTrans:0.0f  zTrans:-8.0f];
                fXTranslation = fXTranslation + 0.015f;
                if ((-7.5f + fXTranslation) >= -2.0f) {
                    fXTranslation = 0.0f;
                    iSequence_RRJ = 2;
                }
                break;

            case 2:
                [self My_Letters: 'I' xTrans:-2.0f yTrans:0.0f zTrans:-8.0f];
                [self My_Letters: 'V' xTrans:(8.50f - fXTranslation) yTrans:0.0f zTrans:-8.0f];
                fXTranslation = fXTranslation + 0.015f;;
                if ((8.5f - fXTranslation) <= 2.0f) {
                    fXTranslation = 0.0f;
                    iSequence_RRJ = 3;
                }
                break;


            case 3:
                [self My_Letters: 'I' xTrans:-2.0f yTrans:0.0f zTrans:-8.0f];
                [self My_Letters: 'V' xTrans:2.0f yTrans:0.0f zTrans:-8.0f];
                [self My_Letters: 'N' xTrans:-1.35f yTrans:(6.0f - fYTranslation) zTrans:-8.0f];
                fYTranslation = fYTranslation + 0.015f;
                if ((6.0f - fYTranslation) < 0.0f) {
                    fYTranslation = 0.0f;
                    iSequence_RRJ = 4;
                }
                break;

            case 4:
                [self My_Letters: 'I' xTrans:-2.0f yTrans:0.0f zTrans:-8.0f];
                [self My_Letters: 'V' xTrans:2.0f yTrans:0.0f zTrans:-8.0f];
                [self My_Letters: 'N' xTrans:-1.35f yTrans:0.0f zTrans:-8.0f];
                [self My_Letters: 'I' xTrans:1.02f yTrans:(-5.0f + fYTranslation) zTrans:-8.0f];
                fYTranslation = fYTranslation + 0.015f;
                if ((-5.0f + fYTranslation) > 0.0f) {
                    fYTranslation = 0.0f;
                    iSequence_RRJ = 5;
                }
                break;

            case 5:

                for(int k = 0; k < 40; k++){
                    D_Color_RRJ[(k * 4) + 3] = fD_Fading_RRJ;
                }

                [self My_Letters: 'I' xTrans:-2.0f yTrans:0.0f zTrans:-8.0f];
                [self My_Letters: 'V' xTrans:2.0f yTrans:0.0f zTrans:-8.0f];
                [self My_Letters: 'N' xTrans:-1.35f yTrans:0.0f zTrans:-8.0f];
                [self My_Letters: 'I' xTrans:1.02f yTrans:0.0f zTrans:-8.0f];
                [self My_D: -0.15f yTrans:0.0f zTrans:-8.0f];
                if (fD_Fading_RRJ > 1.0f) {
                    iSequence_RRJ = 6;
                }
                else{
                    fD_Fading_RRJ = fD_Fading_RRJ + 0.001f;
                }
                break;

            case 6:
                [self My_Letters: 'I' xTrans:-2.0f yTrans:0.0f zTrans:-8.0f];
                [self My_Letters: 'V' xTrans:2.0f yTrans:0.0f zTrans:-8.0f];
                [self My_Letters: 'N' xTrans:-1.35f yTrans:0.0f zTrans:-8.0f];
                [self My_Letters: 'I' xTrans:1.02f yTrans:0.0f zTrans:-8.0f];
                [self My_D: -0.15f yTrans:0.0f zTrans:-8.0f];


                /********** Plane 1 **********/
                if (bPlane1Reached_RRJ == NOT_REACH) {
                    XTrans_Plane1_RRJ = (GLfloat)((3.2 * cos(angle_Plane1_RRJ)) + (-2.5f));
                    YTrans_Plane1_RRJ = (GLfloat)((4.0f * sin(angle_Plane1_RRJ)) + (4.0f));
                    angle_Plane1_RRJ = angle_Plane1_RRJ + 0.005f;
                    ZRot_Plane1_RRJ = ZRot_Plane1_RRJ + 0.2f;


                    if (angle_Plane1_RRJ >= (3.0f * M_PI) / 2.0f) {
                        bPlane1Reached_RRJ = HALF_WAY;
                        YTrans_Plane1_RRJ = 0.00f;

                    }
                    else if (ZRot_Plane1_RRJ >= 0.0)
                        ZRot_Plane1_RRJ = 0.0f;

                }
                else if (bPlane1Reached_RRJ == HALF_WAY) {
                    XTrans_Plane1_RRJ = XTrans_Plane1_RRJ + 0.010f;
                    YTrans_Plane1_RRJ = 0.00f;

                    if (XTrans_Plane1_RRJ >= 3.00f) {   //2.6
                        bPlane1Reached_RRJ = REACH;
                        angle_Plane1_RRJ = (GLfloat)(3.0f * M_PI) / 2.0f;
                        ZRot_Plane1_RRJ = 0.0f;
                    }
                }
                else if (bPlane1Reached_RRJ == REACH) {

                    if (Plane1_Count_RRJ <= 0.0f) {
                        iFadingFlag1_RRJ = 2;
                        XTrans_Plane1_RRJ = (GLfloat)((3.0f * cos(angle_Plane1_RRJ)) + (3.0f));     //2.6
                        YTrans_Plane1_RRJ = (GLfloat)((4.0f * sin(angle_Plane1_RRJ)) + (4.0f));

                        if (XTrans_Plane1_RRJ >= 6.00f || YTrans_Plane1_RRJ >= 4.0f)
                            bPlane1Reached_RRJ = END;

                        angle_Plane1_RRJ = angle_Plane1_RRJ + 0.005f;
                        ZRot_Plane1_RRJ = ZRot_Plane1_RRJ + 0.2f;
                    }
                    else
                        iFadingFlag1_RRJ = 1;

                    Plane1_Count_RRJ = Plane1_Count_RRJ - 1.0f;
                }
                else if (bPlane1Reached_RRJ == END) {
                    angle_Plane1_RRJ = 0.0f;
                    ZRot_Plane1_RRJ = 0.0f;
                }

                /*********** Fading Flag ***********/
                if (bPlane1Reached_RRJ == NOT_REACH)
                    [self My_Fading_Flag: XTrans_Plane1_RRJ yTrans:YTrans_Plane1_RRJ zTrans:-8.0f zRot:ZRot_Plane1_RRJ];
                    

                [self My_Plane: XTrans_Plane1_RRJ yTrans:YTrans_Plane1_RRJ zTrans:-8.0f xScale:0.18f yScale:0.18f zScale:0.0f zRot:ZRot_Plane1_RRJ];




                /********** Plane 2 **********/
                if (bPlane2Reached_RRJ == NOT_REACH) {
                    if ((-6.0f + XTrans_Plane2_RRJ) > -2.50f) {
                        bPlane2Reached_RRJ = HALF_WAY;
                    }
                    else
                        XTrans_Plane2_RRJ = XTrans_Plane2_RRJ + 0.011f;

                }
                else if (bPlane2Reached_RRJ == HALF_WAY) {
                    XTrans_Plane2_RRJ = XTrans_Plane2_RRJ + 0.010f;
                    if ((-6.0f + XTrans_Plane2_RRJ) >= 3.0f) {  //2.6
                        bPlane2Reached_RRJ = REACH;
                    }
                }
                else if (bPlane2Reached_RRJ == REACH) {
                    if (Plane2_Count_RRJ <= 0.00f) {
                        iFadingFlag2_RRJ = 2;
                        XTrans_Plane2_RRJ = XTrans_Plane2_RRJ + 0.010f;
                    }
                    else
                        iFadingFlag2_RRJ = 1;


                    if ((-6.0f + XTrans_Plane2_RRJ) >= 8.0f)
                        bPlane2Reached_RRJ = END;


                    Plane2_Count_RRJ = Plane2_Count_RRJ - 1.0f;
                }
                else if (bPlane2Reached_RRJ == END) {
                    XTrans_Plane2_RRJ = 14.0f;
                }

                /*********** Fading_Flag **********/
                if (iFadingFlag2_RRJ < 2)
                    [self My_Fading_Flag:(-6.0f + XTrans_Plane2_RRJ) yTrans: 0.0f zTrans:-8.0f zRot:0.0f];
        
                [self My_Plane:(-6.0f + XTrans_Plane2_RRJ) yTrans:0.0f zTrans:-8.0f xScale:0.18f yScale:0.18f zScale:0.0f zRot:0.0f];
        







                /********** Plane 3 **********/
                if (bPlane3Reached_RRJ == NOT_REACH) {
                    XTrans_Plane3_RRJ = (GLfloat)((3.2 * cos(angle_Plane3_RRJ)) + (-2.5f));
                    YTrans_Plane3_RRJ = (GLfloat)((4.0f * sin(angle_Plane3_RRJ)) + (-4.0f));
                    angle_Plane3_RRJ = angle_Plane3_RRJ - 0.005f;
                    ZRot_Plane3_RRJ = ZRot_Plane3_RRJ - 0.2f;


                    if (angle_Plane3_RRJ < (M_PI) / 2.0f) {
                        bPlane3Reached_RRJ = HALF_WAY;
                        YTrans_Plane3_RRJ = 0.00f;

                    }
                    else if (ZRot_Plane3_RRJ < 0.0)
                        ZRot_Plane3_RRJ = 0.0f;

                }
                else if (bPlane3Reached_RRJ == HALF_WAY) {
                    XTrans_Plane3_RRJ = XTrans_Plane3_RRJ + 0.010f;
                    YTrans_Plane3_RRJ = 0.00f;

                    if (XTrans_Plane3_RRJ >= 3.00f) {   //2.6
                        bPlane3Reached_RRJ = REACH;
                        angle_Plane3_RRJ = (GLfloat)(M_PI) / 2.0f;
                        ZRot_Plane3_RRJ = 0.0f;
                    }
                }
                else if (bPlane3Reached_RRJ == REACH) {

                    if (Plane3_Count_RRJ <= 0.0f) {
                        iFadingFlag3_RRJ = 2;
                        XTrans_Plane3_RRJ = (GLfloat)((3.0f * cos(angle_Plane3_RRJ)) + (3.0f));     //2.6
                        YTrans_Plane3_RRJ = (GLfloat)((4.0f * sin(angle_Plane3_RRJ)) + (-4.0f));

                        if (XTrans_Plane3_RRJ >= 6.00f || YTrans_Plane3_RRJ < -4.0f)
                            bPlane3Reached_RRJ = END;

                        angle_Plane3_RRJ = angle_Plane3_RRJ - 0.005f;
                        ZRot_Plane3_RRJ = ZRot_Plane3_RRJ - 0.2f;
                    }
                    else
                        iFadingFlag3_RRJ = 1;

                    Plane3_Count_RRJ = Plane3_Count_RRJ - 1.0f;
                }
                else if (bPlane3Reached_RRJ == END) {
                    angle_Plane3_RRJ = 0.0f;
                    ZRot_Plane3_RRJ = 0.0f;
                }



                /*********** Fading Flag ***********/
                if (bPlane2Reached_RRJ == NOT_REACH)
                    [self My_Fading_Flag:XTrans_Plane3_RRJ yTrans:YTrans_Plane3_RRJ zTrans:-8.0f zRot:ZRot_Plane3_RRJ];



                [self My_Plane:XTrans_Plane3_RRJ yTrans:YTrans_Plane3_RRJ zTrans:-8.0f xScale:0.18 yScale:0.18 zScale:0.0f zRot:ZRot_Plane3_RRJ];



                if (iFadingFlag1_RRJ == 2 || iFadingFlag2_RRJ == 2 || iFadingFlag3_RRJ == 2)
                    [self My_Flag: 2.0f yTrans:0.0f zTrans:-8.0f];

                break;

        }

    glUseProgram(0);

    
    CGLFlushDrawable((CGLContextObj) [[self openGLContext] CGLContextObj]);
    CGLUnlockContext((CGLContextObj) [[self openGLContext] CGLContextObj]);
}









-(void)My_Letters:(char) c xTrans:(GLfloat)x yTrans:(GLfloat)y zTrans:(GLfloat)z {

    translateMatrix_RRJ = vmath::mat4::identity();
    modelViewMatrix_RRJ = vmath::mat4::identity();
    modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(x, y, z);
    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    glUniformMatrix4fv(mvpUniform_RRJ, 1, GL_FALSE, modelViewProjectionMatrix_RRJ);


    switch(c){

    	case 'I':
    		
	    	glBindVertexArray(vao_I_RRJ);
	    	glDrawArrays(GL_LINES, 0, 30);
	    	glBindVertexArray(0);
    		break;

    	case 'N':
    		
    		glBindVertexArray(vao_N_RRJ);
    		glDrawArrays(GL_LINES, 0, 30);
    		glBindVertexArray(0);
    		break;


    	case 'V':
    		
    		glBindVertexArray(vao_V_RRJ);
    		glDrawArrays(GL_LINES, 0, 20);
    		glBindVertexArray(0);
    		break;


        case 'A':
            glBindVertexArray(vao_A_RRJ);
            glDrawArrays(GL_LINES, 0, 6);
            glBindVertexArray(0);
            break;

        case 'F':
            glBindVertexArray(vao_Flag_RRJ);
            glDrawArrays(GL_LINES, 0, 6);
            glBindVertexArray(0);
            break;

    }

    
}



-(void)My_D: (GLfloat)x yTrans:(GLfloat)y zTrans:(GLfloat)z {
    translateMatrix_RRJ = vmath::mat4::identity();
    modelViewMatrix_RRJ = vmath::mat4::identity();
    modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(x, y, z);
    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    glUniformMatrix4fv(mvpUniform_RRJ, 1, GL_FALSE, modelViewProjectionMatrix_RRJ);

    glBindVertexArray(vao_D_RRJ);
    	glBindBuffer(GL_ARRAY_BUFFER, vbo_D_Color_RRJ);
	glBufferData(GL_ARRAY_BUFFER, sizeof(D_Color_RRJ), D_Color_RRJ, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
    	glDrawArrays(GL_LINES, 0, 40);
    glBindVertexArray(0);
}




-(void)My_Flag:(GLfloat)x yTrans:(GLfloat)y zTrans:(GLfloat)z {
    translateMatrix_RRJ = vmath::mat4::identity();
    modelViewMatrix_RRJ = vmath::mat4::identity();
    modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(x, y, z);
    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    glUniformMatrix4fv(mvpUniform_RRJ, 1, GL_FALSE, modelViewProjectionMatrix_RRJ);


    glBindVertexArray(vao_Flag_RRJ);
    glDrawArrays(GL_LINES, 0, 30);
    glBindVertexArray(0);
}



-(void)My_Fading_Flag:(GLfloat)x yTrans:(GLfloat)y zTrans:(GLfloat)z zRot:(GLfloat)ZRot_Plane{

	if (bPlane2Reached_RRJ != REACH) {
		for(int p = 0; p < 15; p++){
			Fading_Flag_Position_RRJ[(p * 6) ] -= 0.005;
		}

	}
	else if (bPlane2Reached_RRJ == REACH) {

		for(int p = 0; p < 15; p++){
			Fading_Flag_Position_RRJ[(p * 6) ] += 0.007;
		}
	}

	
        
        translateMatrix_RRJ = vmath::mat4::identity();
        rotateMatrix_RRJ = vmath::mat4::identity();
    	modelViewMatrix_RRJ = vmath::mat4::identity();
    	modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

    	translateMatrix_RRJ = vmath::translate(x, y, z);
        rotateMatrix_RRJ = vmath::rotate(0.0f, 0.0f, ZRot_Plane);
    	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ * rotateMatrix_RRJ;
    	modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    	glUniformMatrix4fv(mvpUniform_RRJ, 1, GL_FALSE, modelViewProjectionMatrix_RRJ);

    
    	glBindVertexArray(vao_Fading_Flag_RRJ);

    	   glBindBuffer(GL_ARRAY_BUFFER, vbo_Fading_Flag_Position_RRJ);
	           glBufferData(GL_ARRAY_BUFFER, sizeof(Fading_Flag_Position_RRJ), Fading_Flag_Position_RRJ, GL_DYNAMIC_DRAW);
	           glBindBuffer(GL_ARRAY_BUFFER, 0);
    	   glDrawArrays(GL_LINES, 0, 30);
    	glBindVertexArray(0);
}



-(void)My_Plane:(GLfloat)x yTrans:(GLfloat)y zTrans:(GLfloat)z xScale:(GLfloat)Sx yScale:(GLfloat)Sy zScale:(GLfloat)Sz zRot:(GLfloat)ZRot_Angle{

        translateMatrix_RRJ = vmath::mat4::identity();
        scaleMatrix_RRJ = vmath::mat4::identity();
        rotateMatrix_RRJ = vmath::mat4::identity();
        modelViewMatrix_RRJ = vmath::mat4::identity();
    	modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

    	translateMatrix_RRJ = vmath::translate(x, y, z);
        rotateMatrix_RRJ = vmath::rotate(0.0f, 0.0f, ZRot_Angle);
        scaleMatrix_RRJ = vmath::scale(Sx, Sy, Sz);
    	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ * rotateMatrix_RRJ * scaleMatrix_RRJ;
    	modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    	glUniformMatrix4fv(mvpUniform_RRJ, 1, GL_FALSE, modelViewProjectionMatrix_RRJ);


    	//Triangle
	   glBindVertexArray(vao_Plane_Triangle_RRJ);
	   glDrawArrays(GL_TRIANGLES, 0, 3);
	   glBindVertexArray(0);

	   //Rectangle
	   glBindVertexArray(vao_Plane_Rect_RRJ);

	   //For Middle
	   glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	   //For Upper and Lower Fin
	   glDrawArrays(GL_TRIANGLE_FAN, 4, 8);

	   //For Back
	   glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
	   glBindVertexArray(0);


	   //Polygon
	   glBindVertexArray(vao_Plane_Polygon_RRJ);
	   glDrawArrays(GL_TRIANGLE_FAN, 0, 10);
	   glBindVertexArray(0);



       //I
    translateMatrix_RRJ = vmath::translate(-1.5f, 0.0f, 0.0f);
    scaleMatrix_RRJ = vmath::scale(0.70f, 0.70f, 0.0f);
    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ * scaleMatrix_RRJ;
    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    glUniformMatrix4fv(mvpUniform_RRJ,
        1,
        GL_FALSE,
        modelViewProjectionMatrix_RRJ);

    glBindVertexArray(vao_I_RRJ);
    glDrawArrays(GL_LINES, 0, 30);
    glBindVertexArray(0);






    //A

    translateMatrix_RRJ = vmath::translate(1.0f, 0.0f, 0.0f);
    //scaleMatrix = scale(scaleX, 0.10f, scaleZ);

    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    glUniformMatrix4fv(mvpUniform_RRJ,
        1,
        GL_FALSE,
        modelViewProjectionMatrix_RRJ);

    glBindVertexArray(vao_A_RRJ);
    glDrawArrays(GL_LINES, 0, 6);
    glBindVertexArray(0);




    //F

    translateMatrix_RRJ = vmath::translate(0.7f, 0.0f, 0.0f);

    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    glUniformMatrix4fv(mvpUniform_RRJ,
        1,
        GL_FALSE,
        modelViewProjectionMatrix_RRJ);
    glBindVertexArray(vao_F_RRJ);
    glDrawArrays(GL_LINES, 0, 6);
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


	//Fading Flag
	if (vbo_Fading_Flag_Color_RRJ) {
		glDeleteBuffers(1, &vbo_Fading_Flag_Color_RRJ);
		vbo_Fading_Flag_Color_RRJ = 0;
	}

	if (vbo_Fading_Flag_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Fading_Flag_Position_RRJ);
		vbo_Fading_Flag_Position_RRJ = 0;
	}

	if (vao_Fading_Flag_RRJ) {
		glDeleteVertexArrays(1, &vao_Fading_Flag_RRJ);
		vao_Fading_Flag_RRJ = 0;
	}



	//Plane Polygon Part
	if (vbo_Plane_Polygon_Color_RRJ) {
		glDeleteBuffers(1, &vbo_Plane_Polygon_Color_RRJ);
		vbo_Plane_Polygon_Color_RRJ = 0;
	}

	if (vbo_Plane_Polygon_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Plane_Polygon_Position_RRJ);
		vbo_Plane_Polygon_Position_RRJ = 0;
	}

	if (vao_Plane_Polygon_RRJ) {
		glDeleteVertexArrays(1, &vao_Plane_Polygon_RRJ);
		vao_Plane_Polygon_RRJ = 0;
	}

	//Plane Rectangle Part
	if (vbo_Plane_Rect_Color_RRJ) {
		glDeleteBuffers(1, &vbo_Plane_Rect_Color_RRJ);
		vbo_Plane_Rect_Color_RRJ = 0;
	}

	if (vbo_Plane_Rect_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Plane_Rect_Position_RRJ);
		vbo_Plane_Rect_Position_RRJ = 0;
	}

	if (vao_Plane_Rect_RRJ) {
		glDeleteVertexArrays(1, &vao_Plane_Rect_RRJ);
		vao_Plane_Rect_RRJ = 0;
	}

	//Plane Triangle Part
	if (vbo_Plane_Triangle_Color_RRJ) {
		glDeleteBuffers(1, &vbo_Plane_Triangle_Color_RRJ);
		vbo_Plane_Triangle_Color_RRJ = 0;
	}

	if (vbo_Plane_Triangle_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Plane_Triangle_Position_RRJ);
		vbo_Plane_Triangle_Position_RRJ = 0;
	}

	if (vao_Plane_Triangle_RRJ) {
		glDeleteVertexArrays(1, &vao_Plane_Triangle_RRJ);
		vao_Plane_Triangle_RRJ = 0;
	}



    
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


    //F
    if (vbo_F_Color_RRJ) {
        glDeleteBuffers(1, &vbo_F_Color_RRJ);
        vbo_F_Color_RRJ = 0;
    }

    if (vbo_F_Position_RRJ) {
        glDeleteBuffers(1, &vbo_F_Position_RRJ);
        vbo_F_Position_RRJ = 0;
    }

    if (vao_F_RRJ) {
        glDeleteVertexArrays(1, &vao_F_RRJ);
        vao_F_RRJ = 0;
    }


    //V
    if (vbo_V_Color_RRJ) {
        glDeleteBuffers(1, &vbo_V_Color_RRJ);
        vbo_V_Color_RRJ = 0;
    }

    if (vbo_V_Position_RRJ) {
        glDeleteBuffers(1, &vbo_V_Position_RRJ);
        vbo_V_Position_RRJ = 0;
    }

    if (vao_V_RRJ) {
        glDeleteVertexArrays(1, &vao_V_RRJ);
        vao_V_RRJ = 0;
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




