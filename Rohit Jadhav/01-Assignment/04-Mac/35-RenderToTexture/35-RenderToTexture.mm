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
FILE *gbFile_Model = NULL;


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


    
 

    //For Model
    mainBundle_RRJ = [NSBundle mainBundle];
    appDirName_RRJ = [mainBundle_RRJ bundlePath];
    parentDirPath_RRJ = [appDirName_RRJ stringByDeletingLastPathComponent];
    NSString *modelFileNameWithPath_RRJ = [NSString stringWithFormat: @"%@/head.txt", parentDirPath_RRJ];
    const char *modelFileName = [modelFileNameWithPath_RRJ cStringUsingEncoding:NSASCIIStringEncoding];

    gbFile_Model = fopen(modelFileName, "r");
    if(gbFile_Model == NULL){
        printf("Model File Loading Failed!!\n");
        [self release];
        [NSApp terminate: self];
    }
    else
        fprintf(gbFile_RRJ, "Model fOpen() done!\n");


    NSRect win_rect_RRJ;
    win_rect_RRJ = NSMakeRect(0.0, 0.0, 800.0, 600.0);

    window_RRJ = [[NSWindow alloc] initWithContentRect: win_rect_RRJ styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable| NSWindowStyleMaskMiniaturizable |
        NSWindowStyleMaskResizable
        backing: NSBackingStoreBuffered
        defer: NO];

    [window_RRJ setTitle: @"Rohit_R_Jadhav-Mac-35-RenderToTexture"];
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
GLfloat lightAmbient_RRJ[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat lightDiffuse_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat lightSpecular_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat lightPosition_RRJ[] = { 100.0f, 100.0f, 100.0f, 1.0f };

//For Material
GLfloat materialAmbient_RRJ[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat materialDiffuse_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialSpecular_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialShininess_RRJ = 128.0f;


//Model Loading
struct VecFloat {

    float *pData;
    int iSize;
};


#define RRJ_SUCCESS 1
#define RRJ_ERROR 0


struct VecFloat *pVecFloat_Model_Vertices = NULL;
struct VecFloat *pVecFloat_Model_Normals = NULL;
struct VecFloat *pVecFloat_Model_Texcoord = NULL;

struct VecFloat *pVecFloat_Model_Sorted_Vertices = NULL;
struct VecFloat *pVecFloat_Model_Sorted_Normals = NULL;
struct VecFloat *pVecFloat_Model_Sorted_Texcoord = NULL;

struct VecFloat *pVecFloat_Model_Elements = NULL;


//For FrameBuffer
GLint viewPort_FBO_Width_RRJ = 1024;
GLint viewPort_FBO_Height_RRJ = 1024;


/********** GLView_RRJ **********/
@implementation GLView_RRJ
{
    @private
        CVDisplayLinkRef displayLink_RRJ;

        GLuint vertexShaderObject_RRJ;
        GLuint fragmentShaderObject_RRJ;
        GLuint shaderProgramObject_RRJ;


        //For Cube
        GLuint texture_Cube_RRJ;
        GLuint vao_Cube_RRJ;
        GLuint vbo_Cube_Position_RRJ;
        GLuint vbo_Cube_Texture_RRJ;


        //For Uniforms
        GLuint mvpUniform_RRJ;
        GLuint samplerUniform_RRJ;
        GLuint choiceUniform_RRJ;


        //For Framebuffer
        GLuint frameBufferObject_RRJ;
        GLuint renderBufferObject_Depth_RRJ;
        GLint viewPortWidth_RRJ;
        GLint viewPortHeight_RRJ;




       	//For Model
       	GLuint vao_Model_RRJ;
       	GLuint vbo_Model_Position_RRJ;
       	GLuint vbo_Model_Normal_RRJ;


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
       	GLuint materialShininessUniform_RRJ;
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

        	//For Choice
            "uniform int u_choice;" \

            //For Attribute
            "in vec4 vPosition;" \
            "in vec3 vNormal;" \
            "in vec2 vTexCoord;" \
            "out vec2 outTexCoord;" \

            //For Cube
            "uniform mat4 u_mvp_matrix;" \

            //For Model and Light
            "uniform mat4 u_model_matrix;" \
            "uniform mat4 u_view_matrix;" \
            "uniform mat4 u_projection_matrix;" \
            "uniform vec4 u_light_position;" \
            "out vec3 viewer_vector_VS;" \
            "out vec3 tNorm_VS;" \
            "out vec3 lightDirection_VS;" \


            "void main(void)" \
            "{" \
                //For Cube
                "if(u_choice == 1) { " \

                    "gl_Position = u_mvp_matrix * vPosition;" \
                    "outTexCoord = vTexCoord;" \

                "}" \

                //For Model
                "else if(u_choice == 2) {" \

                    "vec4 eye_coordinate = u_view_matrix * u_model_matrix * vPosition;" \
                    "viewer_vector_VS = vec3(-eye_coordinate);" \
                    "tNorm_VS = mat3(u_view_matrix * u_model_matrix) * vNormal;" \
                    "lightDirection_VS = vec3(u_light_position - eye_coordinate);" \
                    "gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \

                "}" \
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

        	//For Choice
            "uniform int u_choice;" \
            
            //For Cube
            "in vec2 outTexCoord;" \
            "uniform sampler2D u_sampler;" \


            //For Model
            "in vec3 viewer_vector_VS;" \
            "in vec3 tNorm_VS;" \
            "in vec3 lightDirection_VS;" \
            "uniform vec3 u_La;" \
            "uniform vec3 u_Ld;" \
            "uniform vec3 u_Ls;" \
            "uniform vec3 u_Ka;" \
            "uniform vec3 u_Kd;" \
            "uniform vec3 u_Ks;" \
            "uniform float u_shininess;" \
            "uniform int u_LKeyPress;" \
            
            //For Output
            "out vec4 FragColor;" \

            "void main(void)" \
            "{" \

                //For Cube
                "if(u_choice == 1) { " \
                        
                    "FragColor = texture(u_sampler, outTexCoord);" \

                "}" \

                //For Model
                "else if (u_choice == 2){" \
                    
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
    glBindAttribLocation(shaderProgramObject_RRJ, AMC_ATTRIBUTE_TEXCOORD0, "vTexCoord");
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

    //For Model
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
    materialShininessUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_shininess");
    LKeyPressUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_LKeyPress");

    	//For Cube
	mvpUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_mvp_matrix");
	samplerUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_sampler");
	choiceUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ , "u_choice");






	/********** Positions **********/
	GLfloat Cube_Vertices[] = {
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


	/************ TexCoord **********/
	GLfloat Cube_TexCoord[] = {
		//Top
		1.0f, 1.0,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//Back
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//Face
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//Back
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//Right
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		//Left
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};



	/********** Vao Cube **********/
	glGenVertexArrays(1, &vao_Cube_RRJ);
	glBindVertexArray(vao_Cube_RRJ);

		/******** Position **********/
		glGenBuffers(1, &vbo_Cube_Position_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Cube_Position_RRJ);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(Cube_Vertices),
			Cube_Vertices,
			GL_STATIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
			3,
			GL_FLOAT,
			GL_FALSE,
			0, NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** Texture ***********/
		glGenBuffers(1, &vbo_Cube_Texture_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Cube_Texture_RRJ);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(Cube_TexCoord),
			Cube_TexCoord,
			GL_STATIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOORD0,
			2,
			GL_FLOAT,
			GL_FALSE,
			0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOORD0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);





    /********** Position, Normal and Elements **********/
    pVecFloat_Model_Vertices = [self CreateVecFloat];
    pVecFloat_Model_Normals = [self CreateVecFloat];
    pVecFloat_Model_Texcoord = [self CreateVecFloat];

    pVecFloat_Model_Elements = [self CreateVecFloat];

    pVecFloat_Model_Sorted_Vertices = [self CreateVecFloat];
    pVecFloat_Model_Sorted_Normals = [self CreateVecFloat];
    pVecFloat_Model_Sorted_Texcoord = [self CreateVecFloat];


    [self LoadModel];

    fprintf(gbFile_RRJ, "Size: %ld\n", pVecFloat_Model_Vertices->iSize * sizeof(float));



    /********** Model Vao **********/
    glGenVertexArrays(1, &vao_Model_RRJ);
    glBindVertexArray(vao_Model_RRJ);

        /********** Position **********/
        glGenBuffers(1, &vbo_Model_Position_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_Model_Position_RRJ);
        glBufferData(GL_ARRAY_BUFFER,
           pVecFloat_Model_Sorted_Vertices->iSize * sizeof(float),
            pVecFloat_Model_Sorted_Vertices->pData,
            GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
            3,
            GL_FLOAT,
            GL_FALSE,
            0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
        glBindBuffer(GL_ARRAY_BUFFER, 0);



        /********** Normals **********/
        glGenBuffers(1, &vbo_Model_Normal_RRJ);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_Model_Normal_RRJ);
        glBufferData(GL_ARRAY_BUFFER,
            sizeof(float) * pVecFloat_Model_Sorted_Normals->iSize,
            pVecFloat_Model_Sorted_Normals->pData,
            GL_STATIC_DRAW);
        glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL,
            3,
            GL_FLOAT,
            GL_FALSE,
            0, NULL);
        glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
        glBindBuffer(GL_ARRAY_BUFFER, 0);


    glBindVertexArray(0);



    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glClearDepth(1.0f);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

  

    	/********** FRAMEBUFFER **********/
	glGenFramebuffers(1, &frameBufferObject_RRJ);
	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferObject_RRJ);

		/********** Texture **********/
		glGenTextures(1, &texture_Cube_RRJ);
		glBindTexture(GL_TEXTURE_2D, texture_Cube_RRJ);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, 
			GL_RGBA, 
			viewPort_FBO_Width_RRJ, viewPort_FBO_Width_RRJ, 0, 
			GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_Cube_RRJ, 0);


		/********** For Depth **********/
		glGenRenderbuffers(1, &renderBufferObject_Depth_RRJ);
		glBindRenderbuffer(GL_RENDERBUFFER, renderBufferObject_Depth_RRJ);	
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, viewPort_FBO_Width_RRJ, viewPort_FBO_Width_RRJ);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, renderBufferObject_Depth_RRJ);




		/********** Checking *********/
		if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
			fprintf(gbFile_RRJ, "ERROR: glCheckFramebufferStatus\n");
            [self release];
            [NSApp terminate:self];
		}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);



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


    viewPortWidth_RRJ = width_RRJ;
    viewPortHeight_RRJ = height_RRJ;

  
    CGLUnlockContext((CGLContextObj)[[self openGLContext] CGLContextObj]);
}


-(void)drawRect:(NSRect)rect {
    [self drawView];
}


-(void) drawView {
    
    [[self openGLContext]makeCurrentContext];

    CGLLockContext((CGLContextObj) [[self openGLContext] CGLContextObj]);

  
    vmath::mat4 translateMatrix_RRJ;
    vmath::mat4 rotateMatrix_RRJ;
    vmath::mat4 modelMatrix_RRJ;
    vmath::mat4 viewMatrix_RRJ;
    vmath::mat4 modelViewProjectionMatrix_RRJ;

    static GLfloat angle_Model_RRJ = 0.0f;
    static GLfloat angle_Cube_RRJ = 0.0f;

    glViewport(0, 0, (GLsizei)viewPort_FBO_Width_RRJ, (GLsizei)viewPort_FBO_Width_RRJ);
    perspectiveProjectionMatrix_RRJ = vmath::mat4::identity();
    perspectiveProjectionMatrix_RRJ = vmath::perspective(45.0f, (float)viewPort_FBO_Width_RRJ / (GLfloat)viewPort_FBO_Width_RRJ, 0.1f, 100.0f);


    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferObject_RRJ);
    glClearColor(0.0f, 0.0f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	    glUseProgram(shaderProgramObject_RRJ);

	    /********** Model **********/
	    translateMatrix_RRJ = vmath::mat4::identity();
	    rotateMatrix_RRJ = vmath::mat4::identity();
	    modelMatrix_RRJ = vmath::mat4::identity();
	    viewMatrix_RRJ = vmath::mat4::identity();


	    translateMatrix_RRJ = vmath::translate(0.0f, 0.0f, -8.0f);
	    rotateMatrix_RRJ = vmath::rotate(0.0f, angle_Model_RRJ, 0.0f);
	    modelMatrix_RRJ = modelMatrix_RRJ * translateMatrix_RRJ * rotateMatrix_RRJ;

        glUniform1i(choiceUniform_RRJ, 2);
    
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
	        glUniform1f(materialShininessUniform_RRJ, materialShininess_RRJ);
	    }
	    else
	        glUniform1i(LKeyPressUniform_RRJ, 0);


	    glBindVertexArray(vao_Model_RRJ);
	        glDrawArrays(GL_TRIANGLES, 0, pVecFloat_Model_Sorted_Vertices->iSize / 3);
	    glBindVertexArray(0);

	    glUseProgram(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    angle_Model_RRJ += 0.8f;
    if(angle_Model_RRJ > 360.0f)
        angle_Model_RRJ = 0.0f;







	//For Cube

	glViewport(0, 0, (GLsizei)viewPortWidth_RRJ, (GLsizei)viewPortHeight_RRJ);
	perspectiveProjectionMatrix_RRJ = vmath::mat4::identity();
	perspectiveProjectionMatrix_RRJ = vmath::perspective(45.0f, (float)viewPortWidth_RRJ / (float)viewPortHeight_RRJ, 0.1f, 100.0f);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(shaderProgramObject_RRJ);

	/********** Cube **********/
	translateMatrix_RRJ = vmath::mat4::identity();
	rotateMatrix_RRJ = vmath::mat4::identity();
	modelMatrix_RRJ = vmath::mat4::identity();
	modelViewProjectionMatrix_RRJ = vmath::mat4::identity();


	translateMatrix_RRJ = vmath::translate(0.0f, 0.0f, -6.0f);
	rotateMatrix_RRJ = vmath::rotate(angle_Cube_RRJ, 0.0f, 0.0f) * vmath::rotate(0.0f, angle_Cube_RRJ, 0.0f) * vmath::rotate(0.0f, 0.0f, angle_Cube_RRJ);
	modelMatrix_RRJ = modelMatrix_RRJ * translateMatrix_RRJ * rotateMatrix_RRJ;
	modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * viewMatrix_RRJ * modelMatrix_RRJ;

	glUniformMatrix4fv(mvpUniform_RRJ,
		1,
		GL_FALSE,
		modelViewProjectionMatrix_RRJ);


	//For Shader Toggle
	glUniform1i(choiceUniform_RRJ, 1);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture_Cube_RRJ);
	glUniform1i(samplerUniform_RRJ, 0);

	glBindVertexArray(vao_Cube_RRJ);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
		glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
		glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
		glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
		glDrawArrays(GL_TRIANGLE_FAN, 20, 4);
	glBindVertexArray(0);

	glUseProgram(0);

    
    
    angle_Cube_RRJ += 0.8f;
    if(angle_Cube_RRJ > 360.0f)
        angle_Cube_RRJ = 0.0;
 
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


    	if(renderBufferObject_Depth_RRJ){
		glDeleteRenderbuffers(1, &renderBufferObject_Depth_RRJ);
		renderBufferObject_Depth_RRJ = 0;
	}

	if(texture_Cube_RRJ){
		glDeleteTextures(1, &texture_Cube_RRJ);
		texture_Cube_RRJ = 0;
	}

	if(frameBufferObject_RRJ){
		glDeleteFramebuffers(1, &frameBufferObject_RRJ);
		frameBufferObject_RRJ = 0;
	}


    
    if (pVecFloat_Model_Sorted_Texcoord) {
        [self DestroyVecFloat: pVecFloat_Model_Sorted_Texcoord];
        pVecFloat_Model_Sorted_Texcoord = NULL;
    }

    if (pVecFloat_Model_Sorted_Normals) {
        [self DestroyVecFloat:pVecFloat_Model_Sorted_Normals];
        pVecFloat_Model_Sorted_Normals = NULL;
    }


    if (pVecFloat_Model_Sorted_Vertices) {
        [self DestroyVecFloat: pVecFloat_Model_Sorted_Vertices];
        pVecFloat_Model_Sorted_Vertices = NULL;
    }


    if (pVecFloat_Model_Normals) {
        [self DestroyVecFloat:pVecFloat_Model_Normals];
        pVecFloat_Model_Normals = NULL;
    }

    if (pVecFloat_Model_Texcoord) {
        [self DestroyVecFloat:pVecFloat_Model_Texcoord];
        pVecFloat_Model_Texcoord = NULL;
    }

    if (pVecFloat_Model_Vertices) {
        [self DestroyVecFloat:pVecFloat_Model_Vertices];
        pVecFloat_Model_Vertices = NULL;
    }


    if (pVecFloat_Model_Elements) {
        [self DestroyVecFloat:pVecFloat_Model_Elements];
        pVecFloat_Model_Elements = NULL;
    }



    if (vbo_Model_Normal_RRJ) {
        glDeleteBuffers(1, &vbo_Model_Normal_RRJ);
        vbo_Model_Normal_RRJ = 0;
    }

    if (vbo_Model_Position_RRJ) {
        glDeleteBuffers(1, &vbo_Model_Position_RRJ);
        vbo_Model_Position_RRJ = 0;
    }

    if (vao_Model_RRJ) {
        glDeleteVertexArrays(1, &vao_Model_RRJ);
        vao_Model_RRJ = 0;
    }



    	if (vbo_Cube_Texture_RRJ) {
		glDeleteBuffers(1, &vbo_Cube_Texture_RRJ);
		vbo_Cube_Texture_RRJ = 0;
	}

	if (vbo_Cube_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Cube_Position_RRJ);
		vbo_Cube_Position_RRJ = 0;
	}

	if (vao_Cube_RRJ) {
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




-(struct VecFloat*)CreateVecFloat{

    struct VecFloat *pTemp = NULL;

    pTemp = (struct VecFloat*)malloc(sizeof(struct VecFloat));
    if (pTemp == NULL) {
        printf("ERROR: CreateVecInt(): Malloc() Failed!\n");
        exit(0);
    }

    memset((void*)pTemp, 0, sizeof(struct VecFloat));

    return(pTemp);
}


-(int)PushBackVecFloat:(struct VecFloat*)pVec withData:(float)data{

    pVec->pData = (float*)realloc(pVec->pData, sizeof(struct VecFloat) * (pVec->iSize + 1));

    assert(pVec->pData);

    pVec->iSize = pVec->iSize + 1;
    pVec->pData[pVec->iSize - 1] = data;

    return(RRJ_SUCCESS);

}


-(void)ShowVecFloat:(struct VecFloat*)pVec {
    for (int i = 0; i < pVec->iSize; i++)
        fprintf(gbFile_RRJ, "P[%d]: %f\t", i, pVec->pData[i]);
}



-(int)DestroyVecFloat: (struct VecFloat*)pVec {


    free(pVec->pData);
    pVec->pData = NULL;
    pVec->iSize = 0;
    free(pVec);
    pVec = NULL;

    return(RRJ_SUCCESS);
}



-(void) LoadModel{

    char buffer[1024];
    char *firstToken = NULL;
    char *My_Strtok(char*, char);
    const char *space = " ";
    char *cContext = NULL;


    while (fgets(buffer, 1024, gbFile_Model) != NULL) {

        firstToken = strtok(buffer, space);

        if (strcmp(firstToken, "v") == 0) {
            //Vertices
            float x, y, z;
            x = (float)atof(strtok(NULL, space));
            y = (float)atof(strtok(NULL, space));
            z = (float)atof(strtok(NULL, space));

            //fprintf(gbFile_RRJ, "%f/%f/%f\n", x,y,z);
            [self PushBackVecFloat: pVecFloat_Model_Vertices withData:x];
            [self PushBackVecFloat: pVecFloat_Model_Vertices withData:y];
            [self PushBackVecFloat: pVecFloat_Model_Vertices withData:z];
            //fprintf(gbFile_RRJ, "%f\n", pVecFloat_Model_Vertices->pData[0]);

        }
        else if (strcmp(firstToken, "vt") == 0) {
            //Texture

            float u, v;
            u = (float)atof(strtok(NULL, space));
            v = (float)atof(strtok(NULL, space));


            [self PushBackVecFloat: pVecFloat_Model_Texcoord withData:u];
            [self PushBackVecFloat: pVecFloat_Model_Texcoord withData:v];
        }
        else if (strcmp(firstToken, "vn") == 0) {
            //Normals

            float x, y, z;
            x = (float)atof(strtok(NULL, space));
            y = (float)atof(strtok(NULL, space));
            z = (float)atof(strtok(NULL, space));

            [self PushBackVecFloat: pVecFloat_Model_Normals withData:x];
            [self PushBackVecFloat: pVecFloat_Model_Normals withData:y];
            [self PushBackVecFloat: pVecFloat_Model_Normals withData:z];

        }
        else if (strcmp(firstToken, "f") == 0) {
            //Faces


            for (int i = 0; i < 3; i++) {

                char *faces = strtok(NULL, space);
                int v, vt, vn;
                v = atoi(My_Strtok(faces, '/')) - 1;
                vt = atoi(My_Strtok(faces, '/')) - 1;
                vn = atoi(My_Strtok(faces, '/')) - 1;

                float x, y, z;

                //Sorted Vertices
                x = pVecFloat_Model_Vertices->pData[(v * 3) + 0];
                y = pVecFloat_Model_Vertices->pData[(v * 3) + 1];
                z = pVecFloat_Model_Vertices->pData[(v * 3) + 2];


                //fprintf(gbFile_RRJ, "%f/%f/%f\n", x,y,z);
                [self PushBackVecFloat: pVecFloat_Model_Sorted_Vertices withData:x];
                [self PushBackVecFloat: pVecFloat_Model_Sorted_Vertices withData:y];
                [self PushBackVecFloat: pVecFloat_Model_Sorted_Vertices withData:z];


                //Sorted Normals
                x = pVecFloat_Model_Normals->pData[(vn * 3) + 0];
                y = pVecFloat_Model_Normals->pData[(vn * 3) + 1];
                z = pVecFloat_Model_Normals->pData[(vn * 3) + 2];

                
                [self PushBackVecFloat: pVecFloat_Model_Sorted_Normals withData:x];
                [self PushBackVecFloat: pVecFloat_Model_Sorted_Normals withData:y];
                [self PushBackVecFloat: pVecFloat_Model_Sorted_Normals withData:z];


                //Sorted Texcoord;
                x = pVecFloat_Model_Texcoord->pData[(vt * 2) + 0];
                y = pVecFloat_Model_Texcoord->pData[(vt * 2) + 1];

                [self PushBackVecFloat: pVecFloat_Model_Sorted_Texcoord withData:x];
                [self PushBackVecFloat: pVecFloat_Model_Sorted_Texcoord withData:y];



                //Face Elements
                [self PushBackVecFloat: pVecFloat_Model_Elements withData:v];

            }
        }
    }
}



char gBuffer[128];

char* My_Strtok(char* str, char delimiter) {

    static int  i = 0;
    int  j = 0;
    char c;


    while ((c = str[i]) != delimiter && c != '\0') {
        gBuffer[j] = c;
        j = j + 1;
        i = i + 1;
    }

    gBuffer[j] = '\0';


    if (c == '\0') {
        i = 0;
    }
    else
        i = i + 1;


    return(gBuffer);
}



@end

CVReturn MyDisplayLinkCallback(CVDisplayLinkRef displayLink, const CVTimeStamp *pNow, const CVTimeStamp *pOutputTime, CVOptionFlags flagsIn, CVOptionFlags *pFlagsOut, void *pDisplayLinkContext){

    CVReturn result_RRJ = [(GLView_RRJ*)pDisplayLinkContext getFrameForTime: pOutputTime];
    return(result_RRJ);
}





