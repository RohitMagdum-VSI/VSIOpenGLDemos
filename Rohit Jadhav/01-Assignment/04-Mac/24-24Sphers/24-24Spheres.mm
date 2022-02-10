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

    [window_RRJ setTitle: @"Rohit_R_Jadhav-Mac-24-24Sphere"];
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



const GLuint STACKS = 30;
const GLuint SLICES = 30;


//For Lights
const int X_ROT = 3;
const int Y_ROT = 4;
const int Z_ROT = 5;
float angle_X_RRJ = 0.0f;
float angle_Y_RRJ = 0.0f;
float angle_Z_RRJ = 0.0f;
int iWhichRotation_RRJ = X_ROT;
bool bLights_RRJ = false;


GLfloat lightAmbient_RRJ[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat lightDiffuse_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat lightSpecular_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat lightPosition_RRJ[] = { 0.0f, 0.0f, 0.0f, 1.0f };


//For ViewPort
GLsizei w, h;


/********** GLView_RRJ **********/
@implementation GLView_RRJ
{
    @private
        CVDisplayLinkRef displayLink_RRJ;

        GLuint vertexShaderObject_RRJ;
        GLuint fragmentShaderObject_RRJ;
        GLuint shaderProgramObject_RRJ;

        GLuint vao_Sphere_RRJ;
        GLuint vbo_Sphere_Position_RRJ;
        GLuint vbo_Sphere_Normal_RRJ;
        GLuint vbo_Sphere_Index_RRJ;


        //For Uniform
        GLuint modelMatrixUniform_RRJ;
        GLuint viewMatrixUniform_RRJ;
        GLuint projectionMatrixUniform_RRJ;
        GLuint La_Uniform_RRJ;
        GLuint Ld_Uniform_RRJ;
        GLuint Ls_Uniform_RRJ;
        GLuint lightPositionUniform_RRJ;
        GLuint Ka_Uniform_RRJ;
        GLuint Kd_Uniform_RRJ;
        GLuint Ks_Uniform_RRJ;
        GLuint materialShininess_RRJUniform_RRJ;
        GLuint LKeyPressUniform_RRJ;


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
    "in vec3 vNormal;" \
    "uniform mat4 u_model_matrix;" \
    "uniform mat4 u_view_matrix;" \
    "uniform mat4 u_projection_matrix;" \
    "uniform vec4 u_light_position;" \
    "out vec3 viewer_vector_VS;" \
    "out vec3 tNorm_VS;" \
    "out vec3 lightDirection_VS;" \
    "void main(void)" \
    "{" \
        "vec4 eye_coordinate = u_view_matrix * u_model_matrix * vPosition;" \
        "viewer_vector_VS = vec3(-eye_coordinate);" \
        "tNorm_VS = mat3(u_view_matrix * u_model_matrix) * vNormal;" \
        "lightDirection_VS = vec3(u_light_position - eye_coordinate);" \
        "gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
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
    "in vec3 viewer_vector_VS;" \
    "in vec3 tNorm_VS;" \
    "in vec3 lightDirection_VS;" \
    "out vec4 FragColor;" \
    "uniform vec3 u_La;" \
    "uniform vec3 u_Ld;" \
    "uniform vec3 u_Ls;" \
    "uniform vec3 u_Ka;" \
    "uniform vec3 u_Kd;" \
    "uniform vec3 u_Ks;" \
    "uniform float u_shininess;" \
    "uniform int u_LKeyPress;" \
    "void main(void)" \
    "{" \
        "if(u_LKeyPress == 1){" \
            "vec3 normalize_viewer_vector = normalize(viewer_vector_VS);" \
            "vec3 normalize_tNorm = normalize(tNorm_VS);" \
            "vec3 normalize_lightDirection = normalize(lightDirection_VS);" \
            "vec3 reflection_vector = reflect(-normalize_lightDirection, normalize_tNorm);" \
            "float s_dot_n = max(dot(normalize_lightDirection, normalize_tNorm), 0.0);" \
            "float r_dot_v = max(dot(reflection_vector, normalize_viewer_vector), 0.0);" \
            "vec3 ambient = u_La * u_Ka;" \
            "vec3 diffuse = u_Ld * u_Kd * s_dot_n;" \
            "vec3 specular = u_Ls * u_Ks * pow(r_dot_v, u_shininess);" \
            "vec3 Phong_ADS_Light = ambient + diffuse + specular;" \
            "FragColor = vec4(Phong_ADS_Light, 1.0);" \
        "}" \
        "else{" \
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


    modelMatrixUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_model_matrix");
    viewMatrixUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_view_matrix");
    projectionMatrixUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_projection_matrix");
    La_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_La");
    Ld_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_Ld");
    Ls_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_Ls");
    lightPositionUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_light_position");
    Ka_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_Ka");
    Kd_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_Kd");
    Ks_Uniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_Ks");
    materialShininess_RRJUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_shininess");
    LKeyPressUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_LKeyPress");



    /********** Sphere Position, vao, vbo **********/
    GLfloat sphere_Position_RRJ[STACKS * SLICES * 3];
    GLfloat sphere_Normal_RRJ[STACKS * SLICES * 3];
    GLshort sphere_Index_RRJ[(STACKS - 1) * (SLICES - 1) * 6];


    [self makeSphere: 1.0f Pos:sphere_Position_RRJ Normals:sphere_Normal_RRJ Elements:sphere_Index_RRJ];
   

    /********* Vao For Sphere *********/
    glGenVertexArrays(1, &vao_Sphere_RRJ);
    glBindVertexArray(vao_Sphere_RRJ);
        
        /********** Position *********/
        glGenBuffers(1, &vbo_Sphere_Position_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_Sphere_Position_RRJ);
        glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_Position_RRJ), sphere_Position_RRJ, GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    
            /********** Normals *********/
            glGenBuffers(1, &vbo_Sphere_Normal_RRJ);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_Sphere_Normal_RRJ);
            glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_Normal_RRJ), sphere_Normal_RRJ, GL_STATIC_DRAW);
            glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            /********** Elements **********/
            glGenBuffers(1, &vbo_Sphere_Index_RRJ);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Index_RRJ);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(sphere_Index_RRJ), sphere_Index_RRJ, GL_STATIC_DRAW);


    glBindVertexArray(0);



 

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glClearDepth(1.0f);

    glClearColor(0.250f, 0.250f, 0.250f, 1.0f);

    perspectiveProjectionMatrix_RRJ = vmath::mat4::identity();

    CVDisplayLinkCreateWithActiveCGDisplays(&displayLink_RRJ);
    CVDisplayLinkSetOutputCallback(displayLink_RRJ, &MyDisplayLinkCallback, self);
    CGLContextObj cglContext_RRJ = (CGLContextObj)[[self openGLContext]CGLContextObj];
    CGLPixelFormatObj cglPixelFormat_RRJ = (CGLPixelFormatObj)[[self pixelFormat]CGLPixelFormatObj];
    CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext(displayLink_RRJ, cglContext_RRJ, cglPixelFormat_RRJ);
    CVDisplayLinkStart(displayLink_RRJ);
}


-(void)makeSphere:(GLfloat)radius Pos:(GLfloat[])pos Normals:(GLfloat[])normals Elements:(GLshort[])elements{


    GLfloat longitute = (-M_PI);
        GLfloat lonFactor = (2.0f * M_PI) / (SLICES - 1);
    
    GLfloat latitute = (-M_PI / 2);
    GLfloat latFactor = (M_PI) / (STACKS - 1);
    

    for(int i = 0; i < STACKS; i++){

        for(int j = 0; j < SLICES; j++){

            
            GLfloat x, y, z;

            x = radius * sin(longitute) * cos(latitute);
            y = radius * sin(longitute) * sin(latitute);
            z = radius * cos(longitute);

            pos[(i * SLICES * 3) + (j * 3) + 0] = x;
            pos[(i * SLICES * 3) + (j * 3) + 1] = y;
            pos[(i * SLICES * 3) + (j * 3) + 2] = z;

            normals[(i * SLICES * 3) + (j * 3) + 0] = x;
            normals[(i * SLICES * 3) + (j * 3) + 1] = y;
            normals[(i * SLICES * 3) + (j * 3) + 2] = z;
            //fprintf(gbFile_RRJ, "%f\t%f\n",latitute, longitute);
            
            latitute = latitute + latFactor;

        }
        longitute = longitute + lonFactor;
        latitute = (-M_PI / 2);
    }


    int index = 0;
    for(int i = 0; i < STACKS - 1; i++){
        for(int j = 0; j < SLICES - 1; j++){
    
            GLshort topLeft;
            GLshort bottomLeft;
            GLshort topRight;
            GLshort bottomRight;
            
            topLeft = (i * SLICES) + j;
            bottomLeft = ((i + 1) * SLICES) + j;
                
            /*if(j == SLICES - 1){
                topRight = (i * SLICES) + 0;
                bottomRight = ((i + 1) * SLICES) + 0;
            }
            else{*/
                topRight = topLeft + 1;
                bottomRight = bottomLeft + 1;
           // }
            
                
            
                elements[index + 0] = topLeft;
                elements[index + 1] = bottomLeft;
                elements[index + 2] = topRight;

                elements[index + 3] = topRight;
                elements[index + 4] = bottomLeft;
                elements[index + 5] = bottomRight;


             index = index + 6;
        }
    }

}



-(void)reshape {
    
    CGLLockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);

    NSRect rect_RRJ = [self bounds];
    GLfloat width_RRJ = rect_RRJ.size.width;
    GLfloat height_RRJ = rect_RRJ.size.height;

    if(height_RRJ == 0)
        height_RRJ = 1;

    w = (GLsizei)width_RRJ;
    h = (GLsizei)height_RRJ;

    //glViewport(0, 0, (GLsizei)width_RRJ, (GLsizei)height_RRJ);

    perspectiveProjectionMatrix_RRJ = vmath::perspective(45.0f, (GLfloat)width_RRJ / (GLfloat)height_RRJ, 0.1f, 100.0f);

    CGLUnlockContext((CGLContextObj)[[self openGLContext] CGLContextObj]);
}


-(void)drawRect:(NSRect)rect {
    [self drawView];
}

vmath::mat4 translateMatrix_RRJ;
vmath::mat4 rotateMatrix_RRJ;
vmath::mat4 modelMatrix_RRJ;
vmath::mat4 viewMatrix_RRJ;

-(void) drawView {
    
    [[self openGLContext]makeCurrentContext];

    CGLLockContext((CGLContextObj) [[self openGLContext] CGLContextObj]);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    



    glUseProgram(shaderProgramObject_RRJ);

        [self draw24Spheres];

    glUseProgram(0);

    [self update];
    
    
    CGLFlushDrawable((CGLContextObj) [[self openGLContext] CGLContextObj]);
    CGLUnlockContext((CGLContextObj) [[self openGLContext] CGLContextObj]);
}



-(void)setViewPorts:(GLint)viewPortNo{

    

    if(viewPortNo == 1)                            /************ 1st SET ***********/
        glViewport( 0, 5 * h / 6, w / 6, h / 6);
    else if(viewPortNo == 2)
        glViewport( 0, 4 * h / 6, w / 6,  h / 6);
    else if(viewPortNo == 3)
        glViewport( 0, 3 * h / 6, w / 6, h / 6);
    else if(viewPortNo == 4)
        glViewport( 0, 2 * h / 6, w / 6, h / 6);
    else if(viewPortNo == 5)
        glViewport( 0, 1 * h / 6, w / 6, h / 6);
    else if(viewPortNo == 6)
        glViewport( 0, 0, w / 6, h / 6);
    else if(viewPortNo == 7)                        /************ 2nd SET ***********/
        glViewport( 1 * w / 4, 5 * h / 6, w / 6, h / 6);
    else if(viewPortNo == 8)
        glViewport( 1 * w / 4, 4 * h / 6, w / 6,  h / 6);
    else if(viewPortNo == 9)
        glViewport( 1 * w / 4, 3 * h / 6, w / 6, h / 6);
    else if(viewPortNo == 10)
        glViewport( 1 * w / 4, 2 * h / 6, w / 6, h / 6);
    else if(viewPortNo == 11)
        glViewport( 1 * w / 4, 1 * h / 6, w / 6, h / 6);
    else if(viewPortNo == 12)
        glViewport( 1 * w / 4, 0, w / 6, h / 6);
    else if(viewPortNo == 13)                        /************ 3rd SET ***********/
        glViewport( 2 * w / 4, 5 * h / 6, w / 6, h / 6);
    else if(viewPortNo == 14)
        glViewport( 2 * w / 4, 4 * h / 6, w / 6,  h / 6);
    else if(viewPortNo == 15)
        glViewport( 2 * w / 4, 3 * h / 6, w / 6, h / 6);
    else if(viewPortNo == 16)
        glViewport( 2 * w / 4, 2 * h / 6, w / 6, h / 6);
    else if(viewPortNo == 17)
        glViewport( 2 * w / 4, 1 * h / 6, w / 6, h / 6);
    else if(viewPortNo == 18)
        glViewport( 2 * w / 4, 0, w / 6, h / 6);
    else if(viewPortNo == 19)                        /************ 4th SET ***********/
        glViewport( 3 * w / 4, 5 * h / 6, w / 6, h / 6);
    else if(viewPortNo == 20)
        glViewport( 3 * w / 4, 4 * h / 6, w / 6,  h / 6);
    else if(viewPortNo == 21)
        glViewport( 3 * w / 4, 3 * h / 6, w / 6, h / 6);
    else if(viewPortNo == 22)
        glViewport( 3 * w / 4, 2 * h / 6, w / 6, h / 6);
    else if(viewPortNo == 23)
        glViewport( 3 * w / 4, 1 * h / 6, w / 6, h / 6);
    else if(viewPortNo == 24)
        glViewport( 3 * w / 4, 0, w / 6, h / 6);
}




-(void)draw24Spheres{

    float materialAmbient_RRJ[4];
    float materialDiffuse_RRJ[4];
    float materialSpecular_RRJ[4];
    float materialShininess_RRJ = 0.0f;

    for(int i = 1; i <= 24; i++){


        if(i == 1){
            materialAmbient_RRJ[0] = 0.0215f;
            materialAmbient_RRJ[1] = 0.1745f;
            materialAmbient_RRJ[2] = 0.215f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.07568f;
            materialDiffuse_RRJ[1] = 0.61424f;
            materialDiffuse_RRJ[2] = 0.07568f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.633f;
            materialSpecular_RRJ[1] = 0.727811f;
            materialSpecular_RRJ[2] = 0.633f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.6f * 128;

        }
        else if(i == 2){
            materialAmbient_RRJ[0] = 0.135f;
            materialAmbient_RRJ[1] = 0.2225f;
            materialAmbient_RRJ[2] = 0.1575f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.54f;
            materialDiffuse_RRJ[1] = 0.89f;
            materialDiffuse_RRJ[2] = 0.63f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.316228f;
            materialSpecular_RRJ[1] = 0.316228f;
            materialSpecular_RRJ[2] = 0.316228f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.1f * 128;
        }
        else if(i == 3){
            materialAmbient_RRJ[0] = 0.05375f;
            materialAmbient_RRJ[1] = 0.05f;
            materialAmbient_RRJ[2] = 0.06625f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.18275f;
            materialDiffuse_RRJ[1] = 0.17f;
            materialDiffuse_RRJ[2] = 0.22525f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.332741f;
            materialSpecular_RRJ[1] = 0.328634f;
            materialSpecular_RRJ[2] = 0.346435f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.3f * 128;
        }
        else if(i == 4){
            materialAmbient_RRJ[0] = 0.25f;
            materialAmbient_RRJ[1] = 0.20725f;
            materialAmbient_RRJ[2] = 0.20725f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 1.0f;
            materialDiffuse_RRJ[1] = 0.829f;
            materialDiffuse_RRJ[2] = 0.829f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.296648f;
            materialSpecular_RRJ[1] = 0.296648f;
            materialSpecular_RRJ[2] = 0.296648f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.088f * 128;
        }
        else if(i == 5){
            materialAmbient_RRJ[0] = 0.1745f;
            materialAmbient_RRJ[1] = 0.01175f;
            materialAmbient_RRJ[2] = 0.01175f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.61424f;
            materialDiffuse_RRJ[1] = 0.04136f;
            materialDiffuse_RRJ[2] = 0.04136f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.727811f;
            materialSpecular_RRJ[1] = 0.626959f;
            materialSpecular_RRJ[2] = 0.626959f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.6f * 128;
        }
        else if(i == 6){
            materialAmbient_RRJ[0] = 0.1f;
            materialAmbient_RRJ[1] = 0.18725f;
            materialAmbient_RRJ[2] = 0.1745f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.396f;
            materialDiffuse_RRJ[1] = 0.74151f;
            materialDiffuse_RRJ[2] = 0.69102f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.297254f;
            materialSpecular_RRJ[1] = 0.30829f;
            materialSpecular_RRJ[2] = 0.306678f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.1f * 128;
        }
        else if(i == 7){
            materialAmbient_RRJ[0] = 0.329412f;
            materialAmbient_RRJ[1] = 0.223529f;
            materialAmbient_RRJ[2] = 0.027451f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.780392f;
            materialDiffuse_RRJ[1] = 0.568627f;
            materialDiffuse_RRJ[2] = 0.113725f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.992157f;
            materialSpecular_RRJ[1] = 0.941176f;
            materialSpecular_RRJ[2] = 0.807843f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.21794872f * 128;
        }
        else if(i == 8){
            materialAmbient_RRJ[0] = 0.2125f;
            materialAmbient_RRJ[1] = 0.1275f;
            materialAmbient_RRJ[2] = 0.054f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.714f;
            materialDiffuse_RRJ[1] = 0.4284f;
            materialDiffuse_RRJ[2] = 0.18144f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.393548f;
            materialSpecular_RRJ[1] = 0.271906f;
            materialSpecular_RRJ[2] = 0.166721f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.2f * 128;
        }
        else if(i == 9){
            materialAmbient_RRJ[0] = 0.25f;
            materialAmbient_RRJ[1] = 0.25f;
            materialAmbient_RRJ[2] = 0.25f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.4f;
            materialDiffuse_RRJ[1] = 0.4f;
            materialDiffuse_RRJ[2] = 0.4f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.774597f;
            materialSpecular_RRJ[1] = 0.774597f;
            materialSpecular_RRJ[2] = 0.774597f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.6f * 128;
        }
        else if(i == 10){
            materialAmbient_RRJ[0] = 0.19125f;
            materialAmbient_RRJ[1] = 0.0735f;
            materialAmbient_RRJ[2] = 0.0225f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.7038f;
            materialDiffuse_RRJ[1] = 0.27048f;
            materialDiffuse_RRJ[2] = 0.0828f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.256777f;
            materialSpecular_RRJ[1] = 0.137622f;
            materialSpecular_RRJ[2] = 0.086014f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.1f * 128;
        }
        else if(i == 11){
            materialAmbient_RRJ[0] = 0.24725f;
            materialAmbient_RRJ[1] = 0.1995f;
            materialAmbient_RRJ[2] = 0.0745f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.75164f;
            materialDiffuse_RRJ[1] = 0.60648f;
            materialDiffuse_RRJ[2] = 0.22648f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.628281f;
            materialSpecular_RRJ[1] = 0.555802f;
            materialSpecular_RRJ[2] = 0.366065f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.4f * 128;
        }
        else if(i == 12){
            materialAmbient_RRJ[0] = 0.19225f;
            materialAmbient_RRJ[1] = 0.19225f;
            materialAmbient_RRJ[2] = 0.19225f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.50754f;
            materialDiffuse_RRJ[1] = 0.50754f;
            materialDiffuse_RRJ[2] = 0.50754f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.508273f;
            materialSpecular_RRJ[1] = 0.508273f;
            materialSpecular_RRJ[2] = 0.508273f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.4f * 128;
        }
        else if(i == 13){
            materialAmbient_RRJ[0] = 0.0f;
            materialAmbient_RRJ[1] = 0.0f;
            materialAmbient_RRJ[2] = 0.0f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.01f;
            materialDiffuse_RRJ[1] = 0.01f;
            materialDiffuse_RRJ[2] = 0.01f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.5f;
            materialSpecular_RRJ[1] = 0.5f;
            materialSpecular_RRJ[2] = 0.5f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.25f * 128;
        }
        else if(i == 14){
            materialAmbient_RRJ[0] = 0.0f;
            materialAmbient_RRJ[1] = 0.1f;
            materialAmbient_RRJ[2] = 0.06f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.0f;
            materialDiffuse_RRJ[1] = 0.50980392f;
            materialDiffuse_RRJ[2] = 0.52980392f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.50196078f;
            materialSpecular_RRJ[1] = 0.50196078f;
            materialSpecular_RRJ[2] = 0.50196078f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.25f * 128;
        }
        else if(i == 15){
            materialAmbient_RRJ[0] = 0.0f;
            materialAmbient_RRJ[1] = 0.0f;
            materialAmbient_RRJ[2] = 0.0f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.1f;
            materialDiffuse_RRJ[1] = 0.35f;
            materialDiffuse_RRJ[2] = 0.1f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.45f;
            materialSpecular_RRJ[1] = 0.55f;
            materialSpecular_RRJ[2] = 0.45f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.25f * 128;
        }
        else if(i == 16){
            materialAmbient_RRJ[0] = 0.0f;
            materialAmbient_RRJ[1] = 0.0f;
            materialAmbient_RRJ[2] = 0.0f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.5f;
            materialDiffuse_RRJ[1] = 0.0f;
            materialDiffuse_RRJ[2] = 0.0f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.7f;
            materialSpecular_RRJ[1] = 0.6f;
            materialSpecular_RRJ[2] = 0.6f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.25f * 128;
        }
        else if(i == 17){
            materialAmbient_RRJ[0] = 0.0f;
            materialAmbient_RRJ[1] = 0.0f;
            materialAmbient_RRJ[2] = 0.0f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.55f;
            materialDiffuse_RRJ[1] = 0.55f;
            materialDiffuse_RRJ[2] = 0.55f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.70f;
            materialSpecular_RRJ[1] = 0.70f;
            materialSpecular_RRJ[2] = 0.70f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.25f * 128;
        }
        else if(i == 18){
            materialAmbient_RRJ[0] = 0.0f;
            materialAmbient_RRJ[1] = 0.0f;
            materialAmbient_RRJ[2] = 0.0f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.5f;
            materialDiffuse_RRJ[1] = 0.5f;
            materialDiffuse_RRJ[2] = 0.0f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.60f;
            materialSpecular_RRJ[1] = 0.60f;
            materialSpecular_RRJ[2] = 0.50f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.25f * 128;
        }
        else if(i == 19){
            materialAmbient_RRJ[0] = 0.02f;
            materialAmbient_RRJ[1] = 0.02f;
            materialAmbient_RRJ[2] = 0.02f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.01f;
            materialDiffuse_RRJ[1] = 0.01f;
            materialDiffuse_RRJ[2] = 0.01f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.4f;
            materialSpecular_RRJ[1] = 0.4f;
            materialSpecular_RRJ[2] = 0.4f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.078125f * 128;
        }
        else if(i == 20){
            materialAmbient_RRJ[0] = 0.0f;
            materialAmbient_RRJ[1] = 0.05f;
            materialAmbient_RRJ[2] = 0.05f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.4f;
            materialDiffuse_RRJ[1] = 0.5f;
            materialDiffuse_RRJ[2] = 0.5f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.04f;
            materialSpecular_RRJ[1] = 0.7f;
            materialSpecular_RRJ[2] = 0.7f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.078125f * 128;
        }
        else if(i == 21){
            materialAmbient_RRJ[0] = 0.0f;
            materialAmbient_RRJ[1] = 0.05f;
            materialAmbient_RRJ[2] = 0.0f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.4f;
            materialDiffuse_RRJ[1] = 0.5f;
            materialDiffuse_RRJ[2] = 0.4f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.04f;
            materialSpecular_RRJ[1] = 0.7f;
            materialSpecular_RRJ[2] = 0.04f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.078125f * 128;
        }
        else if(i == 22){
            materialAmbient_RRJ[0] = 0.05f;
            materialAmbient_RRJ[1] = 0.0f;
            materialAmbient_RRJ[2] = 0.0f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.5f;
            materialDiffuse_RRJ[1] = 0.4f;
            materialDiffuse_RRJ[2] = 0.4f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.70f;
            materialSpecular_RRJ[1] = 0.04f;
            materialSpecular_RRJ[2] = 0.04f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.078125f * 128;
        }
        else if(i == 23){
            materialAmbient_RRJ[0] = 0.05f;
            materialAmbient_RRJ[1] = 0.05f;
            materialAmbient_RRJ[2] = 0.05f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.5f;
            materialDiffuse_RRJ[1] = 0.5f;
            materialDiffuse_RRJ[2] = 0.5f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.70f;
            materialSpecular_RRJ[1] = 0.70f;
            materialSpecular_RRJ[2] = 0.70f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.078125f * 128;
        }
        else if(i == 24){
            materialAmbient_RRJ[0] = 0.05f;
            materialAmbient_RRJ[1] = 0.05f;
            materialAmbient_RRJ[2] = 0.0f;
            materialAmbient_RRJ[3] = 1.0f;

            materialDiffuse_RRJ[0] = 0.5f;
            materialDiffuse_RRJ[1] = 0.5f;
            materialDiffuse_RRJ[2] = 0.4f;
            materialDiffuse_RRJ[3] = 1.0f;

            materialSpecular_RRJ[0] = 0.70f;
            materialSpecular_RRJ[1] = 0.70f;
            materialSpecular_RRJ[2] = 0.04f;
            materialSpecular_RRJ[3] = 1.0f;

            materialShininess_RRJ = 0.078125f * 128;
        }


        [self setViewPorts: i];



        /********** SPHERE **********/
        translateMatrix_RRJ = vmath::mat4::identity();
        rotateMatrix_RRJ = vmath::mat4::identity();
        modelMatrix_RRJ = vmath::mat4::identity();
        viewMatrix_RRJ = vmath::mat4::identity();

        translateMatrix_RRJ = vmath::translate(0.0f, 0.0f, -4.0f);
        modelMatrix_RRJ = modelMatrix_RRJ * translateMatrix_RRJ * rotateMatrix_RRJ;

        glUniformMatrix4fv(modelMatrixUniform_RRJ, 1, GL_FALSE, modelMatrix_RRJ);
        glUniformMatrix4fv(viewMatrixUniform_RRJ, 1, GL_FALSE, viewMatrix_RRJ);
        glUniformMatrix4fv(projectionMatrixUniform_RRJ, 1, GL_FALSE, perspectiveProjectionMatrix_RRJ);

        if (bLights_RRJ == true) {

            if(iWhichRotation_RRJ == X_ROT)
                [self rotateX: angle_X_RRJ];
            else if(iWhichRotation_RRJ == Y_ROT)
                [self rotateY: angle_Y_RRJ];
            else if(iWhichRotation_RRJ == Z_ROT)
                [self rotateZ: angle_Z_RRJ];
                

            glUniform1i(LKeyPressUniform_RRJ, 1);

            glUniform3fv(La_Uniform_RRJ, 1, lightAmbient_RRJ);
            glUniform3fv(Ld_Uniform_RRJ, 1, lightDiffuse_RRJ);
            glUniform3fv(Ls_Uniform_RRJ, 1, lightSpecular_RRJ);
            glUniform4fv(lightPositionUniform_RRJ, 1, lightPosition_RRJ);

            glUniform3fv(Ka_Uniform_RRJ, 1, materialAmbient_RRJ);
            glUniform3fv(Kd_Uniform_RRJ, 1, materialDiffuse_RRJ);
            glUniform3fv(Ks_Uniform_RRJ, 1, materialSpecular_RRJ);
            glUniform1f(materialShininess_RRJUniform_RRJ, materialShininess_RRJ);
        }
        else
            glUniform1i(LKeyPressUniform_RRJ, 0);



        glBindVertexArray(vao_Sphere_RRJ);
        glDrawElements(GL_TRIANGLES, (STACKS - 1) * (SLICES - 1) * 6, GL_UNSIGNED_SHORT, 0);
        glBindVertexArray(0);

    }

    
    
}


-(void) rotateX:(float)angle{
    lightPosition_RRJ[1] = (float)(5.0f * sin(angle));
    lightPosition_RRJ[2] = (float)(5.0f * cos(angle));
    lightPosition_RRJ[0] = 0.0f;
}

-(void) rotateY:(float)angle{
    lightPosition_RRJ[0] = (float)(5.0f * sin(angle));
    lightPosition_RRJ[2] = (float)(5.0f * cos(angle));
    lightPosition_RRJ[1] = 0.0f;
}

-(void) rotateZ:(float)angle{
    lightPosition_RRJ[0] = (float)(5.0f * cos(angle));
    lightPosition_RRJ[1] = (float)(5.0f * sin(angle));
    lightPosition_RRJ[2] = 0.0f;
}


-(void) update{

    if(iWhichRotation_RRJ == X_ROT)
        angle_X_RRJ = angle_X_RRJ + 0.02f;
    else if(iWhichRotation_RRJ == Y_ROT)
        angle_Y_RRJ = angle_Y_RRJ + 0.02f;
    else if(iWhichRotation_RRJ == Z_ROT)
        angle_Z_RRJ = angle_Z_RRJ + 0.02f;

    if(angle_X_RRJ > 360.0f)
        angle_X_RRJ = 0.0f;

    if(angle_Y_RRJ > 360.0f)
        angle_Y_RRJ = 0.0f;

    if(angle_Z_RRJ > 360.0f)
        angle_Z_RRJ = 0.0f;
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


        case 'X':
        case 'x':
            iWhichRotation_RRJ = X_ROT;
            break;

        case 'Y':
        case 'y':
            iWhichRotation_RRJ = Y_ROT;
            break;

        case 'Z':
        case 'z':
            iWhichRotation_RRJ = Z_ROT;
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
    

    if(vbo_Sphere_Normal_RRJ){
        glDeleteBuffers(1, &vbo_Sphere_Normal_RRJ);
        vbo_Sphere_Normal_RRJ = 0;
    }
    
    if(vbo_Sphere_Position_RRJ){
        glDeleteBuffers(1, &vbo_Sphere_Position_RRJ);
        vbo_Sphere_Position_RRJ = 0;
    }

    if(vao_Sphere_RRJ){
        glDeleteVertexArrays(1, &vao_Sphere_RRJ);
        vao_Sphere_RRJ = 0;
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





