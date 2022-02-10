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

    [window_RRJ setTitle: @"Rohit_R_Jadhav-Mac-28-Interleaved"];
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





//For Lights
bool bLights_RRJ = false;
GLfloat lightAmbient_RRJ[] = { 0.250f, 0.250f, 0.250f, 0.0f };
GLfloat lightDiffuse_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat lightSpecular_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat lightPosition_RRJ[] = { 100.0f, 100.0f, 100.0f, 1.0f };

//For Material
GLfloat materialAmbient_RRJ[] = { 0.25f, 0.25f, 0.25f, 0.0f };
GLfloat materialDiffuse_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialSpecular_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialShininess_RRJ = 128.0f;

//For Texture
GLuint texture_Marble;
GLuint samplerUniform;


/********** GLView_RRJ **********/
@implementation GLView_RRJ
{
    @private
        CVDisplayLinkRef displayLink_RRJ;

        GLuint vertexShaderObject_RRJ;
        GLuint fragmentShaderObject_RRJ;
        GLuint shaderProgramObject_RRJ;

        //For Cube
        GLuint vao_Cube_RRJ;
        GLuint vbo_Cube_Vertices_RRJ;


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
        "in vec4 vColor;" \
        "in vec2 vTex;" \
        "in vec3 vNormal;" \
        "out vec2 outTex;" \
        "out vec4 outColor;" \

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
            "outTex = vTex;" \
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

        "in vec2 outTex;" \
        "in vec4 outColor;" \

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

        "uniform sampler2D u_sampler;" \

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

                "vec4 tex = texture(u_sampler, outTex);" \

                "FragColor = tex * vec4(Phong_ADS_Light, 1.0) * outColor;" \
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
    glBindAttribLocation(shaderProgramObject_RRJ, AMC_ATTRIBUTE_COLOR, "vColor");
    glBindAttribLocation(shaderProgramObject_RRJ, AMC_ATTRIBUTE_TEXCOORD0, "vTex");
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
    samplerUniform = glGetUniformLocation(shaderProgramObject_RRJ, "u_sampler"); 


   /********** Position, Normal and Elements **********/
    GLfloat cube_Vertices[] = {
        //vPosition         //vColor        //vNormal       //vTex
        1.0, 1.0, -1.0,     1.0, 0.0, 0.0,      0.0, 1.0, 0.0,      1.0, 1.0,      
        -1.0, 1.0, -1.0,    1.0, 0.0, 0.0,      0.0, 1.0, 0.0,      0.0, 1.0,    
        -1.0, 1.0, 1.0,     1.0, 0.0, 0.0,      0.0, 1.0, 0.0,      0.0, 0.0,   
        1.0, 1.0, 1.0,      1.0, 0.0, 0.0,      0.0, 1.0, 0.0,      1.0, 0.0,
        //Bottom
        1.0, -1.0, -1.0,    0.0, 1.0, 0.0,      0.0, -1.0, 0.0,     1.0, 1.0,
        -1.0, -1.0, -1.0,   0.0, 1.0, 0.0,      0.0, -1.0, 0.0,     0.0, 1.0,   
        -1.0, -1.0, 1.0,    0.0, 1.0, 0.0,      0.0, -1.0, 0.0,     0.0, 0.0,
        1.0, -1.0, 1.0,     0.0, 1.0, 0.0,      0.0, -1.0, 0.0,     1.0, 0.0,
        //Front
        1.0, 1.0, 1.0,      0.0, 0.0, 1.0,      0.0, 0.0, 1.0,      1.0, 1.0,
        -1.0, 1.0, 1.0,     0.0, 0.0, 1.0,      0.0, 0.0, 1.0,      0.0, 1.0,
        -1.0, -1.0, 1.0,    0.0, 0.0, 1.0,      0.0, 0.0, 1.0,      0.0, 0.0,
        1.0, -1.0, 1.0,     0.0, 0.0, 1.0,      0.0, 0.0, 1.0,      1.0, 0.0,
        //Back
        1.0, 1.0, -1.0,     1.0, 1.0, 0.0,      0.0, 0.0, -1.0,     1.0, 1.0,
        -1.0, 1.0, -1.0,    1.0, 1.0, 0.0,      0.0, 0.0, -1.0,     0.0, 1.0,
        -1.0, -1.0, -1.0,   1.0, 1.0, 0.0,      0.0, 0.0, -1.0,     0.0, 0.0,
        1.0, -1.0, -1.0,    1.0, 1.0, 0.0,      0.0, 0.0, -1.0,     1.0, 0.0,
        //Right
        1.0, 1.0, -1.0,     0.0, 1.0, 1.0,      1.0, 0.0, 0.0,      1.0, 1.0,
        1.0, 1.0, 1.0,      0.0, 1.0, 1.0,      1.0, 0.0, 0.0,      0.0, 1.0,
        1.0, -1.0, 1.0,     0.0, 1.0, 1.0,      1.0, 0.0, 0.0,      0.0, 0.0,
        1.0, -1.0, -1.0,    0.0, 1.0, 1.0,      1.0, 0.0, 0.0,      1.0, 0.0,
        //Left
        -1.0, 1.0, 1.0,     1.0, 0.0, 1.0,      -1.0, 0.0, 0.0,     1.0, 1.0,
        -1.0, 1.0, -1.0,    1.0, 0.0, 1.0,      -1.0, 0.0, 0.0,     0.0, 1.0,
        -1.0, -1.0, -1.0,   1.0, 0.0, 1.0,      -1.0, 0.0, 0.0,     0.0, 0.0,
        -1.0, -1.0, 1.0,    1.0, 0.0, 1.0,      -1.0, 0.0, 0.0,     1.0, 0.0,
    };




    /********** Cube **********/
    glGenVertexArrays(1, &vao_Cube_RRJ);
    glBindVertexArray(vao_Cube_RRJ);

        /********** Position Color Tex Normal **********/
        glGenBuffers(1, &vbo_Cube_Vertices_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_Cube_Vertices_RRJ);
        glBufferData(GL_ARRAY_BUFFER, sizeof(cube_Vertices), cube_Vertices, GL_STATIC_DRAW);

        //Position
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE,
            11 * sizeof(GLfloat), (void*)(0 * sizeof(GLfloat)));
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

        //Color
        glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE,
            11 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));

        glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);

        //Normal
        glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE,
            11 * sizeof(GLfloat), (void*)(6 * sizeof(GLfloat)));
        glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
            

        //TexCoord
        glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOORD0, 2, GL_FLOAT, GL_FALSE,
            11 * sizeof(GLfloat), (void*)(9 * sizeof(GLfloat)));
        glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOORD0);

        glBindBuffer(GL_ARRAY_BUFFER, vbo_Cube_Vertices_RRJ);

    glBindVertexArray(0);


    texture_Marble =  [self loadTextureFromBMP: "marble.bmp"];
 

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



-(GLuint)loadTextureFromBMP: (const char*)fileName{

    NSBundle *mainBundle_RRJ = [NSBundle mainBundle];
    NSString *appDirName_RRJ = [mainBundle_RRJ bundlePath];
    NSString *parentDirPath_RRJ = [appDirName_RRJ stringByDeletingLastPathComponent];
    NSString *textureWithPath_RRJ = [NSString stringWithFormat: @"%@/%s", parentDirPath_RRJ, fileName];

    NSImage *bmpImage_RRJ = [[NSImage alloc] initWithContentsOfFile:textureWithPath_RRJ];
    if(!bmpImage_RRJ){
        NSLog(@"Can't Find: %@", textureWithPath_RRJ);
        return(0);
    }

    CGImageRef cgImage_RRJ = [bmpImage_RRJ CGImageForProposedRect:nil context:nil hints:nil];

    int w_RRJ = (int)CGImageGetWidth(cgImage_RRJ);
    int h_RRJ = (int)CGImageGetHeight(cgImage_RRJ);

    CFDataRef imageData_RRJ = CGDataProviderCopyData(CGImageGetDataProvider(cgImage_RRJ));

    void *pixel_RRJ = (void*)CFDataGetBytePtr(imageData_RRJ);

    GLuint texture_RRJ;

    glGenTextures(1, &texture_RRJ);
    glBindTexture(GL_TEXTURE_2D, texture_RRJ);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

        glTexImage2D(GL_TEXTURE_2D,
            0,
            GL_RGBA,
            w_RRJ, h_RRJ, 0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            pixel_RRJ);

        glGenerateMipmap(GL_TEXTURE_2D);
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
    vmath::mat4 rotateMatrix_RRJ;
    vmath::mat4 modelMatrix_RRJ;
    vmath::mat4 viewMatrix_RRJ;


    static GLfloat cube_Angle = 0.0f;

    glUseProgram(shaderProgramObject_RRJ);

    

    /********** SPHERE **********/
    translateMatrix_RRJ = vmath::mat4::identity();
    rotateMatrix_RRJ = vmath::mat4::identity();
    modelMatrix_RRJ = vmath::mat4::identity();
    viewMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(0.0f, 0.0f, -6.0f);
    rotateMatrix_RRJ = vmath::rotate(cube_Angle, 1.0f, 0.0f, 0.0f) * vmath::rotate(cube_Angle, 0.0f, 1.0f, 0.0f) * vmath::rotate(cube_Angle, 0.0f, 0.0f, 1.0f);
    modelMatrix_RRJ = modelMatrix_RRJ * translateMatrix_RRJ * rotateMatrix_RRJ;

    glUniformMatrix4fv(modelMatrixUniform_RRJ, 1, GL_FALSE, modelMatrix_RRJ);
    glUniformMatrix4fv(viewMatrixUniform_RRJ, 1, GL_FALSE, viewMatrix_RRJ);
    glUniformMatrix4fv(projectionMatrixUniform_RRJ, 1, GL_FALSE, perspectiveProjectionMatrix_RRJ);

    if (bLights_RRJ == true) {
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


    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_Marble);
    glUniform1i(samplerUniform, 0);


    glBindVertexArray(vao_Cube_RRJ);
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
            glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
            glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
            glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
            glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
            glDrawArrays(GL_TRIANGLE_FAN, 20, 4);
    glBindVertexArray(0);

    glUseProgram(0);
    
    
    cube_Angle += 1.0f;
    if(cube_Angle > 360.0f)
        cube_Angle = 0.0f;

    
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
    

   if(texture_Marble){
        glDeleteTextures(1, &texture_Marble);
        texture_Marble = 0;
    }


    if(vbo_Cube_Vertices_RRJ){
        glDeleteBuffers(1, &vbo_Cube_Vertices_RRJ);
        vbo_Cube_Vertices_RRJ = 0;
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




