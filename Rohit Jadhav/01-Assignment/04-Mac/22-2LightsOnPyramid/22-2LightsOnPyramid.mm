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

    [window_RRJ setTitle: @"Rohit_R_Jadhav-Mac-22-2LightsOnPyramid"];
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
struct LIGHTS{
	GLfloat lightAmbient_RRJ[4];
	GLfloat lightDiffuse_RRJ[4];
	GLfloat lightSpecular_RRJ[4];
	GLfloat lightPosition_RRJ[4];
};

struct LIGHTS lights_RRJ[2];

//For Material
GLfloat materialAmbient_RRJ[] = {0.0f, 0.0f, 0.0f, 0.0f};
GLfloat materialDiffuse_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
GLfloat materialSpecular_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
GLfloat materialShininess_RRJ = 128.0f;


/********** GLView_RRJ **********/
@implementation GLView_RRJ
{
    @private
        CVDisplayLinkRef displayLink_RRJ;

        GLuint vertexShaderObject_RRJ;
        GLuint fragmentShaderObject_RRJ;
        GLuint shaderProgramObject_RRJ;

        GLuint vao_Pyramid_RRJ;
        GLuint vbo_Pyramid_Position_RRJ;
        GLuint vbo_Pyramid_Normal_RRJ;
        GLfloat angle_Pyramid;


       //For Uniform Per Fragment Lighting
	GLuint modelMatrixUniform_PF_RRJ;
	GLuint viewMatrixUniform_PF_RRJ;
	GLuint projectionMatrixUniform_PF_RRJ;
	GLuint LaRed_Uniform_PF_RRJ;
	GLuint LdRed_Uniform_PF_RRJ;
	GLuint LsRed_Uniform_PF_RRJ;
	GLuint lightPositionRed_Uniform_PF_RRJ;
	GLuint LaBlue_Uniform_PF_RRJ;
	GLuint LdBlue_Uniform_PF_RRJ;
	GLuint LsBlue_Uniform_PF_RRJ;
	GLuint lightPositionBlue_Uniform_PF_RRJ;
	GLuint KaUniform_PF_RRJ;
	GLuint KdUniform_PF_RRJ;
	GLuint KsUniform_PF_RRJ;
	GLuint shininessUniform_PF_RRJ;
	GLuint LKeyPressUniform_PF_RRJ;



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
	"in vec3 vNormals;" \
	"uniform mat4 u_model_matrix;" \
	"uniform mat4 u_view_matrix;" \
	"uniform mat4 u_projection_matrix;" \
	"uniform vec4 u_light_position_Red;" \
	"uniform vec4 u_light_position_Blue;" \
	"out vec3 lightDirectionRed_VS;" \
	"out vec3 lightDirectionBlue_VS;" \
	"out vec3 Normal_VS;" \
	"out vec3 Viewer_VS;" \
	"void main(void)" \
	"{"
		"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" \
		"lightDirectionRed_VS = vec3(u_light_position_Red - eyeCoordinate);" \
		"lightDirectionBlue_VS = vec3(u_light_position_Blue - eyeCoordinate);" \
		"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" \
		"Normal_VS = vec3(normalMatrix * vNormals);" \
		"Viewer_VS = vec3(-eyeCoordinate);" \
		"gl_Position =	u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
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
        "in vec3 lightDirectionRed_VS;" \
        "in vec3 lightDirectionBlue_VS;" \
        "in vec3 Normal_VS;" \
        "in vec3 Viewer_VS;" \
        "uniform vec3 u_La_Red;" \
        "uniform vec3 u_Ld_Red;" \
        "uniform vec3 u_Ls_Red;" \

        "uniform vec3 u_La_Blue;" \
        "uniform vec3 u_Ld_Blue;" \
        "uniform vec3 u_Ls_Blue;" \

        "uniform vec3 u_Ka;" \
        "uniform vec3 u_Kd;" \
        "uniform vec3 u_Ks;" \

        "uniform float u_shininess;" \
        "uniform int u_L_keypress;" \
        "out vec4 FragColor;" \
        "void main(void)" \
        "{" \
        "vec3 phongRed_Light;" \
        "vec3 phongBlue_Light;" \
        "vec3 phongLight;" \
        "if(u_L_keypress == 1){" \
        "vec3 LightDirection_Red = normalize(lightDirectionRed_VS);" \
        "vec3 LightDirection_Blue = normalize(lightDirectionBlue_VS);" \

        "vec3 Normal = normalize(Normal_VS);" \
        "float LRed_Dot_N = max(dot(LightDirection_Red, Normal), 0.0);" \
        "float LBlue_Dot_N = max(dot(LightDirection_Blue, Normal), 0.0);" \

        "vec3 ReflectionRed = reflect(-LightDirection_Red, Normal);" \
        "vec3 ReflectionBlue = reflect(-LightDirection_Blue, Normal);" \

        "vec3 Viewer = normalize(Viewer_VS);" \

        "float RRed_Dot_V = max(dot(ReflectionRed, Viewer), 0.0);" \
        "float RBlue_Dot_V = max(dot(ReflectionBlue, Viewer), 0.0);" \

        "vec3 ambient_Red = u_La_Red * u_Ka;" \
        "vec3 diffuse_Red = u_Ld_Red * u_Kd * LRed_Dot_N;" \
        "vec3 specular_Red = u_Ls_Red * u_Ks * pow(RRed_Dot_V, u_shininess);" \
        "phongRed_Light = ambient_Red + diffuse_Red + specular_Red;" \

        "vec3 ambient_Blue = u_La_Blue * u_Ka;" \
        "vec3 diffuse_Blue = u_Ld_Blue * u_Kd * LBlue_Dot_N;" \
        "vec3 specular_Blue = u_Ls_Blue * u_Ks * pow(RBlue_Dot_V, u_shininess);" \
        "phongBlue_Light = ambient_Blue + diffuse_Blue + specular_Blue;" \

        "phongLight = phongRed_Light + phongBlue_Light;" \
        
        "}" \
        "else{" \
            "phongLight = vec3(1.0, 1.0, 1.0);" \
        "}" \
            "FragColor = vec4(phongLight, 1.0);" \
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
    glBindAttribLocation(shaderProgramObject_RRJ, AMC_ATTRIBUTE_NORMAL, "vNormals");

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


	modelMatrixUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_model_matrix");
	viewMatrixUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_view_matrix");
	projectionMatrixUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_projection_matrix");

	LaRed_Uniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_La_Red");
	LdRed_Uniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_Ld_Red");
	LsRed_Uniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_Ls_Red");
	lightPositionRed_Uniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_light_position_Red");

	LaBlue_Uniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_La_Blue");
	LdBlue_Uniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_Ld_Blue");
	LsBlue_Uniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_Ls_Blue");
	lightPositionBlue_Uniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_light_position_Blue");

	KaUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_Ka");
	KdUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_Kd");
	KsUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_Ks");
	shininessUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_shininess");
	LKeyPressUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_L_keypress");


    	/********** Positions **********/
	GLfloat Pyramid_Vertices_RRJ[] = {
		//Face
		0.0f, 1.0f, 0.0f,
		-1.0, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		//Right
		0.0f, 1.0f, 0.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f,
		//Back
		0.0f, 1.0f, 0.0f,
		1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		//Left
		0.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f
	};

	GLfloat Pyramid_Normals_RRJ[] = {
		//Face
		0.0f, 0.447214f, 0.894427f,
		0.0f, 0.447214f, 0.894427f,
		0.0f, 0.447214f, 0.894427f,


		//Right
		0.894427f, 0.447214f, 0.0f,
		0.894427f, 0.447214f, 0.0f,
		0.894427f, 0.447214f, 0.0f,


		//Back
		0.0f, 0.447214f, -0.894427f,
		0.0f, 0.447214f, -0.894427f,
		0.0f, 0.447214f, -0.894427f,

		//Left
		-0.894427f, 0.447214f, 0.0f,
		-0.894427f, 0.447214f, 0.0f,
		-0.894427f, 0.447214f, 0.0f,
	};



	/********* Vao Pyramid **********/
	glGenVertexArrays(1, &vao_Pyramid_RRJ);
	glBindVertexArray(vao_Pyramid_RRJ);

		/********** Position *********/
		glGenBuffers(1, &vbo_Pyramid_Position_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Pyramid_Position_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Pyramid_Vertices_RRJ), Pyramid_Vertices_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** Normals **********/
		glGenBuffers(1, &vbo_Pyramid_Normal_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Pyramid_Normal_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Pyramid_Normals_RRJ), Pyramid_Normals_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);



	[self FillLights_Data];


   angle_Pyramid = 0.0f;
 

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


-(void)FillLights_Data{
	//Red Light

	lights_RRJ[0].lightAmbient_RRJ[0] = 0.0f;
	lights_RRJ[0].lightAmbient_RRJ[1] = 0.0f;
	lights_RRJ[0].lightAmbient_RRJ[2] = 0.0f;
	lights_RRJ[0].lightAmbient_RRJ[3] = 0.0f;
	
	lights_RRJ[0].lightDiffuse_RRJ[0] = 1.0f;
	lights_RRJ[0].lightDiffuse_RRJ[1] = 0.0f;
	lights_RRJ[0].lightDiffuse_RRJ[2] = 0.0f;
	lights_RRJ[0].lightDiffuse_RRJ[3] = 1.0f;
	
	lights_RRJ[0].lightSpecular_RRJ[0] = 1.0f;
	lights_RRJ[0].lightSpecular_RRJ[1] = 0.0f;
	lights_RRJ[0].lightSpecular_RRJ[2] = 0.0f;
	lights_RRJ[0].lightSpecular_RRJ[3] = 1.0f;
	
	lights_RRJ[0].lightPosition_RRJ[0] = -2.0f;
	lights_RRJ[0].lightPosition_RRJ[1] = 0.0f;
	lights_RRJ[0].lightPosition_RRJ[2] = 0.0f;
	lights_RRJ[0].lightPosition_RRJ[3] = 1.0f;
	
	
	//Blue Light
	lights_RRJ[1].lightAmbient_RRJ[0] = 0.0f;
	lights_RRJ[1].lightAmbient_RRJ[1] = 0.0f;
	lights_RRJ[1].lightAmbient_RRJ[2] = 0.0f;
	lights_RRJ[1].lightAmbient_RRJ[3] = 0.0f;
	
	lights_RRJ[1].lightDiffuse_RRJ[0] = 0.0f;
	lights_RRJ[1].lightDiffuse_RRJ[1] = 0.0f;
	lights_RRJ[1].lightDiffuse_RRJ[2] = 1.0f;
	lights_RRJ[1].lightDiffuse_RRJ[3] = 1.0f;
	
	lights_RRJ[1].lightSpecular_RRJ[0] = 0.0f;
	lights_RRJ[1].lightSpecular_RRJ[1] = 0.0f;
	lights_RRJ[1].lightSpecular_RRJ[2] = 1.0f;
	lights_RRJ[1].lightSpecular_RRJ[3] = 1.0f;
	
	lights_RRJ[1].lightPosition_RRJ[0] = 2.0f;
	lights_RRJ[1].lightPosition_RRJ[1] = 0.0f;
	lights_RRJ[1].lightPosition_RRJ[2] = 0.0f;
	lights_RRJ[1].lightPosition_RRJ[3] = 1.0f;
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



    glUseProgram(shaderProgramObject_RRJ);

    

    /********** Pyramid **********/
    translateMatrix_RRJ = vmath::mat4::identity();
    rotateMatrix_RRJ = vmath::mat4::identity();
    modelMatrix_RRJ = vmath::mat4::identity();
    viewMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(0.0f, 0.0f, -6.0f);
    rotateMatrix_RRJ = vmath::rotate(angle_Pyramid, 0.0f, 1.0f, 0.0f);
    modelMatrix_RRJ = modelMatrix_RRJ * translateMatrix_RRJ * rotateMatrix_RRJ;

   	glUniformMatrix4fv(modelMatrixUniform_PF_RRJ, 1, GL_FALSE, modelMatrix_RRJ);
	glUniformMatrix4fv(viewMatrixUniform_PF_RRJ, 1, GL_FALSE, viewMatrix_RRJ);
	glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, perspectiveProjectionMatrix_RRJ);

	if (bLights_RRJ == true) {

		glUniform1i(LKeyPressUniform_PF_RRJ, 1);
		glUniform3fv(LaRed_Uniform_PF_RRJ, 1, lights_RRJ[0].lightAmbient_RRJ);
		glUniform3fv(LdRed_Uniform_PF_RRJ, 1, lights_RRJ[0].lightDiffuse_RRJ);
		glUniform3fv(LsRed_Uniform_PF_RRJ, 1, lights_RRJ[0].lightSpecular_RRJ);
		glUniform4fv(lightPositionRed_Uniform_PF_RRJ, 1, lights_RRJ[0].lightPosition_RRJ);

		glUniform3fv(LaBlue_Uniform_PF_RRJ, 1, lights_RRJ[1].lightAmbient_RRJ);
		glUniform3fv(LdBlue_Uniform_PF_RRJ, 1, lights_RRJ[1].lightDiffuse_RRJ);
		glUniform3fv(LsBlue_Uniform_PF_RRJ, 1, lights_RRJ[1].lightSpecular_RRJ);
		glUniform4fv(lightPositionBlue_Uniform_PF_RRJ, 1, lights_RRJ[1].lightPosition_RRJ);

		glUniform3fv(KaUniform_PF_RRJ, 1, materialAmbient_RRJ);
		glUniform3fv(KdUniform_PF_RRJ, 1, materialDiffuse_RRJ);
		glUniform3fv(KsUniform_PF_RRJ, 1, materialSpecular_RRJ);
		glUniform1f(shininessUniform_PF_RRJ, materialShininess_RRJ);
	}
	else
		glUniform1i(LKeyPressUniform_PF_RRJ, 0);	


    
    glBindVertexArray(vao_Pyramid_RRJ);
    	glDrawArrays(GL_TRIANGLES, 0, 4 * 3);
    glBindVertexArray(0);

    glUseProgram(0);
    
    
    angle_Pyramid += 0.50f;
    if(angle_Pyramid > 360.0f)
        angle_Pyramid = 0.0f;

    
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
    

    if(vbo_Pyramid_Normal_RRJ){
        glDeleteBuffers(1, &vbo_Pyramid_Normal_RRJ);
        vbo_Pyramid_Normal_RRJ = 0;
    }
    
    if(vbo_Pyramid_Position_RRJ){
        glDeleteBuffers(1, &vbo_Pyramid_Position_RRJ);
        vbo_Pyramid_Position_RRJ = 0;
    }

    if(vao_Pyramid_RRJ){
        glDeleteVertexArrays(1, &vao_Pyramid_RRJ);
        vao_Pyramid_RRJ = 0;
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




