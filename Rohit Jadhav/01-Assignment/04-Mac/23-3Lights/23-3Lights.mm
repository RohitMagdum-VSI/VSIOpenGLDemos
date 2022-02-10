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

    [window_RRJ setTitle: @"Rohit_R_Jadhav-Mac-23-3Lights"];
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
const int PER_VERTEX = 1;
const int PER_FRAGMENT = 2;
GLuint iWhichLight_RRJ = PER_VERTEX;

struct LIGHTS{
	GLfloat lightAmbient_RRJ[4];
	GLfloat lightDiffuse_RRJ[4];
	GLfloat lightSpecular_RRJ[4];
	GLfloat lightPosition_RRJ[4];
};

struct LIGHTS lights_RRJ[3];

bool bLights_RRJ = false;
GLfloat angle_red_RRJ = 0.0f;
GLfloat angle_green_RRJ = 0.0f;
GLfloat angle_blue_RRJ = 0.0f;



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

        //Shader Object;
	GLuint vertexShaderObject_PV_RRJ;
	GLuint fragmentShaderObject_PV_RRJ;

	GLuint vertexShaderObject_PF_RRJ;
	GLuint fragmentShaderObject_PF_RRJ;

       	//For Shader
	GLuint shaderProgramObject_PV_RRJ;
	GLuint shaderProgramObject_PF_RRJ;

        GLuint vao_Sphere_RRJ;
        GLuint vbo_Sphere_Position_RRJ;
        GLuint vbo_Sphere_Normal_RRJ;
        GLuint vbo_Sphere_Index_RRJ;
        GLfloat sphere_Angle;


        //For Uniform Per Vertex Lighting
	GLuint modelMatrixUniform_PV_RRJ;
	GLuint viewMatrixUniform_PV_RRJ;
	GLuint projectionMatrixUniform_PV_RRJ;

	GLuint red_LaUniform_PV_RRJ;
	GLuint red_LdUniform_PV_RRJ;
	GLuint red_LsUniform_PV_RRJ;
	GLuint red_lightPositionUniform_PV_RRJ;

	GLuint green_LaUniform_PV_RRJ;
	GLuint green_LdUniform_PV_RRJ;
	GLuint green_LsUniform_PV_RRJ;
	GLuint green_lightPositionUniform_PV_RRJ;

	GLuint blue_LaUniform_PV_RRJ;
	GLuint blue_LdUniform_PV_RRJ;
	GLuint blue_LsUniform_PV_RRJ;
	GLuint blue_lightPositionUniform_PV_RRJ;

	GLuint KaUniform_PV_RRJ;
	GLuint KdUniform_PV_RRJ;
	GLuint KsUniform_PV_RRJ;
	GLuint shininessUniform_PV_RRJ;
	GLuint LKeyPressUniform_PV_RRJ;



	//For Uniform Per Fragment Lighting
	GLuint modelMatrixUniform_PF_RRJ;
	GLuint viewMatrixUniform_PF_RRJ;
	GLuint projectionMatrixUniform_PF_RRJ;

	GLuint red_LaUniform_PF_RRJ;
	GLuint red_LdUniform_PF_RRJ;
	GLuint red_LsUniform_PF_RRJ;
	GLuint red_lightPositionUniform_PF_RRJ;

	GLuint green_LaUniform_PF_RRJ;
	GLuint green_LdUniform_PF_RRJ;
	GLuint green_LsUniform_PF_RRJ;
	GLuint green_lightPositionUniform_PF_RRJ;

	GLuint blue_LaUniform_PF_RRJ;
	GLuint blue_LdUniform_PF_RRJ;
	GLuint blue_LsUniform_PF_RRJ;
	GLuint blue_lightPositionUniform_PF_RRJ;

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
    

   /********** Vertex Shader Per Vertex *********/
	vertexShaderObject_PV_RRJ = glCreateShader(GL_VERTEX_SHADER);
	const char *szVertexShaderCode_PV_RRJ =
		"#version 410" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vNormals;" \
		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \

		"uniform vec3 u_Red_La;" \
		"uniform vec3 u_Red_Ld;" \
		"uniform vec3 u_Red_Ls;" \
		"uniform vec4 u_Red_light_position;" \

		"uniform vec3 u_Green_La;" \
		"uniform vec3 u_Green_Ld;" \
		"uniform vec3 u_Green_Ls;" \
		"uniform vec4 u_Green_light_position;" \

		"uniform vec3 u_Blue_La;" \
		"uniform vec3 u_Blue_Ld;" \
		"uniform vec3 u_Blue_Ls;" \
		"uniform vec4 u_Blue_light_position;" \

		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_shininess;" \
		"uniform int u_L_keypress;" \
		"out vec3 phongLight;" \
		"void main(void)" \
		"{" \
			"if(u_L_keypress == 1){" \
				"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" \

				"vec3 RedSource = normalize(vec3(u_Red_light_position - eyeCoordinate));" \
				"vec3 GreenSource = normalize(vec3(u_Green_light_position - eyeCoordinate));" \
				"vec3 BlueSource = normalize(vec3(u_Blue_light_position - eyeCoordinate));" \

				"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" \
				"vec3 Normal = normalize(vec3(normalMatrix * vNormals));" \

				"float SRed_Dot_N = max(dot(RedSource, Normal), 0.0);" \
				"float SGreen_Dot_N = max(dot(GreenSource, Normal), 0.0);" \
				"float SBlue_Dot_N = max(dot(BlueSource, Normal), 0.0);" \

				"vec3 RedReflection = reflect(-RedSource, Normal);" \
				"vec3 GreenReflection = reflect(-GreenSource, Normal);" \
				"vec3 BlueReflection = reflect(-BlueSource, Normal);" \

				"vec3 Viewer = normalize(vec3(-eyeCoordinate.xyz));" \


				"float RRed_Dot_V = max(dot(RedReflection, Viewer), 0.0);" \
				"vec3 ambientRed = u_Red_La * u_Ka;" \
				"vec3 diffuseRed = u_Red_Ld * u_Kd * SRed_Dot_N;" \
				"vec3 specularRed = u_Red_Ls * u_Ks * max(pow(RRed_Dot_V, u_shininess), 0.0);" \
				"vec3 Red = ambientRed + diffuseRed + specularRed;" \


				"float RGreen_Dot_V = max(dot(GreenReflection, Viewer), 0.0);" \
				"vec3 ambientGreen = u_Green_La * u_Ka;" \
				"vec3 diffuseGreen = u_Green_Ld * u_Kd * SGreen_Dot_N;" \
				"vec3 specularGreen = u_Green_Ls * u_Ks * max(pow(RGreen_Dot_V, u_shininess), 0.0);" \
				"vec3 Green = ambientGreen + diffuseGreen + specularGreen;" \


				"float RBlue_Dot_V = max(dot(BlueReflection, Viewer), 0.0);" \
				"vec3 ambientBlue = u_Blue_La * u_Ka;" \
				"vec3 diffuseBlue = u_Blue_Ld * u_Kd * SBlue_Dot_N;" \
				"vec3 specularBlue = u_Blue_Ls * u_Ks * max(pow(RBlue_Dot_V, u_shininess), 0.0);" \
				"vec3 Blue = ambientBlue + diffuseBlue + specularBlue;" \

				"phongLight = Red + Green + Blue;" \


			"}" \
			"else{" \
				"phongLight = vec3(1.0, 1.0, 1.0);" \
			"}" \
			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
		"}";

	glShaderSource(vertexShaderObject_PV_RRJ, 1, (const GLchar**)&szVertexShaderCode_PV_RRJ, NULL);

	glCompileShader(vertexShaderObject_PV_RRJ);

	GLint iShaderCompileStatus_RRJ;
	GLint iInfoLogLength_RRJ;
	GLchar *szInfoLog_RRJ = NULL;
	glGetShaderiv(vertexShaderObject_PV_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(vertexShaderObject_PV_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(vertexShaderObject_PV_RRJ, iInfoLogLength_RRJ,
					&written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Per Vertex Lighting Vertex Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				[self release];
                [NSApp terminate: self];
			}
		}
	}


	/********** Fragment Shader Per Vertex *********/
	fragmentShaderObject_PV_RRJ = glCreateShader(GL_FRAGMENT_SHADER);
	const char* szFragmentShaderCode_PV_RRJ =
		"#version 410" \
		"\n" \
		"in vec3 phongLight;" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
			"FragColor = vec4(phongLight, 1.0);" \
		"}";


	glShaderSource(fragmentShaderObject_PV_RRJ, 1,
		(const GLchar**)&szFragmentShaderCode_PV_RRJ, NULL);

	glCompileShader(fragmentShaderObject_PV_RRJ);

	iShaderCompileStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(fragmentShaderObject_PV_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(fragmentShaderObject_PV_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject_PV_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Per Vertex Lighting Fragment Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				[self release];
                [NSApp terminate: self];
			}
		}
	}


	/********* Program Object For Per Vertex Lighting **********/
	shaderProgramObject_PV_RRJ = glCreateProgram();

	glAttachShader(shaderProgramObject_PV_RRJ, vertexShaderObject_PV_RRJ);
	glAttachShader(shaderProgramObject_PV_RRJ, fragmentShaderObject_PV_RRJ);

	glBindAttribLocation(shaderProgramObject_PV_RRJ, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(shaderProgramObject_PV_RRJ, AMC_ATTRIBUTE_NORMAL, "vNormals");

	glLinkProgram(shaderProgramObject_PV_RRJ);

	GLint iProgramLinkingStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;
	glGetProgramiv(shaderProgramObject_PV_RRJ, GL_LINK_STATUS, &iProgramLinkingStatus_RRJ);
	if (iProgramLinkingStatus_RRJ == GL_FALSE) {
		glGetProgramiv(shaderProgramObject_PV_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetProgramInfoLog(shaderProgramObject_PV_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Shader Program Object Linking Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				[self release];
                [NSApp terminate: self];
			}
		}
	}

	modelMatrixUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_model_matrix");
	viewMatrixUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_view_matrix");
	projectionMatrixUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_projection_matrix");
	
	red_LaUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Red_La");
	red_LdUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Red_Ld");
	red_LsUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Red_Ls");
	red_lightPositionUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Red_light_position");

	green_LaUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Green_La");
	green_LdUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Green_Ld");
	green_LsUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Green_Ls");
	green_lightPositionUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Green_light_position");

	blue_LaUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Blue_La");
	blue_LdUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Blue_Ld");
	blue_LsUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Blue_Ls");
	blue_lightPositionUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Blue_light_position");


	KaUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ka");
	KdUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Kd");
	KsUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ks");
	shininessUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_shininess");
	LKeyPressUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_L_keypress");



	/********** Vertex Shader Per Fragment Lighting *********/
	vertexShaderObject_PF_RRJ = glCreateShader(GL_VERTEX_SHADER);
	const char *szVertexShaderCode_PF_RRJ =
		"#version 410 " \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vNormals;" \

		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \

		"uniform vec4 u_Red_light_position;" \
		"uniform vec4 u_Green_light_position;" \
		"uniform vec4 u_Blue_light_position;" \

		"out vec3 lightDirectionRed_VS;" \
		"out vec3 lightDirectionGreen_VS;" \
		"out vec3 lightDirectionBlue_VS;" \

		"out vec3 Normal_VS;" \
		"out vec3 Viewer_VS;" \

		"void main(void)" \
		"{"
			"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" \
			"lightDirectionRed_VS = vec3(u_Red_light_position - eyeCoordinate);" \
			"lightDirectionGreen_VS = vec3(u_Green_light_position - eyeCoordinate);" \
			"lightDirectionBlue_VS = vec3(u_Blue_light_position - eyeCoordinate);" \
			
			"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" \
			"Normal_VS = vec3(normalMatrix * vNormals);" \
			"Viewer_VS = vec3(-eyeCoordinate);" \
			"gl_Position =	u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
		"}";


	glShaderSource(vertexShaderObject_PF_RRJ, 1, (const GLchar**)&szVertexShaderCode_PF_RRJ, NULL);

	glCompileShader(vertexShaderObject_PF_RRJ);


	glGetShaderiv(vertexShaderObject_PF_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(vertexShaderObject_PF_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(vertexShaderObject_PF_RRJ, iInfoLogLength_RRJ,
					&written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Per Fragment Lighting Vertex Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				[self release];
                [NSApp terminate: self];
			}
		}
	}


	/********** Fragment Shader Per Fragment *********/
	fragmentShaderObject_PF_RRJ = glCreateShader(GL_FRAGMENT_SHADER);
	const char* szFragmentShaderCode_PF_RRJ =
		"#version 410" \
		"\n" \

		"in vec3 lightDirectionRed_VS;" \
		"in vec3 lightDirectionGreen_VS;" \
		"in vec3 lightDirectionBlue_VS;" \

		"in vec3 Normal_VS;" \
		"in vec3 Viewer_VS;" \

		"uniform vec3 u_Red_La;" \
		"uniform vec3 u_Red_Ld;" \
		"uniform vec3 u_Red_Ls;" \

		"uniform vec3 u_Green_La;" \
		"uniform vec3 u_Green_Ld;" \
		"uniform vec3 u_Green_Ls;" \

		"uniform vec3 u_Blue_La;" \
		"uniform vec3 u_Blue_Ld;" \
		"uniform vec3 u_Blue_Ls;" \


		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_shininess;" \
		"uniform int u_L_keypress;" \


		"out vec4 FragColor;" \


		"void main(void)" \
		"{" \
			"vec3 phongLight;" \
			"if(u_L_keypress == 1){" \
				"vec3 RedLightDirection = normalize(lightDirectionRed_VS);" \
				"vec3 GreenLightDirection = normalize(lightDirectionGreen_VS);" \
				"vec3 BlueLightDirection = normalize(lightDirectionBlue_VS);" \
				
				
				"vec3 Normal = normalize(Normal_VS);" \

				"float LRed_Dot_N = max(dot(RedLightDirection, Normal), 0.0);" \
				"float LGreen_Dot_N = max(dot(GreenLightDirection, Normal), 0.0);" \
				"float LBlue_Dot_N = max(dot(BlueLightDirection, Normal), 0.0);" \
				
				
				"vec3 RedReflection = reflect(-RedLightDirection, Normal);" \
				"vec3 GreenReflection = reflect(-GreenLightDirection, Normal);" \
				"vec3 BlueReflection = reflect(-BlueLightDirection, Normal);" \
				
				
				"vec3 Viewer = normalize(Viewer_VS);" \


				"float RRed_Dot_V = max(dot(RedReflection, Viewer), 0.0);" \
				"float RGreen_Dot_V = max(dot(GreenReflection, Viewer), 0.0);" \
				"float RBlue_Dot_V = max(dot(BlueReflection, Viewer), 0.0);" \



				"vec3 ambientRed = u_Red_La * u_Ka;" \
				"vec3 diffuseRed = u_Red_Ld * u_Kd * LRed_Dot_N;" \
				"vec3 specularRed = u_Red_Ls * u_Ks * pow(RRed_Dot_V, u_shininess);" \
				"vec3 Red = ambientRed + diffuseRed + specularRed;" \


				"vec3 ambientGreen = u_Green_La * u_Ka;" \
				"vec3 diffuseGreen = u_Green_Ld * u_Kd * LGreen_Dot_N;" \
				"vec3 specularGreen = u_Green_Ls * u_Ks * pow(RGreen_Dot_V, u_shininess);" \
				"vec3 Green = ambientGreen + diffuseGreen + specularGreen;" \

				"vec3 ambientBlue = u_Blue_La * u_Ka;" \
				"vec3 diffuseBlue = u_Blue_Ld * u_Kd * LBlue_Dot_N;" \
				"vec3 specularBlue = u_Blue_Ls * u_Ks * pow(RBlue_Dot_V, u_shininess);" \
				"vec3 Blue = ambientBlue + diffuseBlue + specularBlue;" \

				"phongLight = Red + Green + Blue;" \

			"}" \
			"else{" \
				"phongLight = vec3(1.0, 1.0, 1.0);" \
			"}" \
			"FragColor = vec4(phongLight, 0.0);" \
		"}";


	glShaderSource(fragmentShaderObject_PF_RRJ, 1,
		(const GLchar**)&szFragmentShaderCode_PF_RRJ, NULL);

	glCompileShader(fragmentShaderObject_PF_RRJ);

	iShaderCompileStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(fragmentShaderObject_PF_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(fragmentShaderObject_PF_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject_PF_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Per Fragment Lighting Fragment Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				[self release];
                [NSApp terminate: self];
			}
		}
	}


	/********* Program Object For Per Fragment Lighting **********/
	shaderProgramObject_PF_RRJ = glCreateProgram();

	glAttachShader(shaderProgramObject_PF_RRJ, vertexShaderObject_PF_RRJ);
	glAttachShader(shaderProgramObject_PF_RRJ, fragmentShaderObject_PF_RRJ);

	glBindAttribLocation(shaderProgramObject_PF_RRJ, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(shaderProgramObject_PF_RRJ, AMC_ATTRIBUTE_NORMAL, "vNormals");

	glLinkProgram(shaderProgramObject_PF_RRJ);

	
	glGetProgramiv(shaderProgramObject_PF_RRJ, GL_LINK_STATUS, &iProgramLinkingStatus_RRJ);
	if (iProgramLinkingStatus_RRJ == GL_FALSE) {
		glGetProgramiv(shaderProgramObject_PF_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetProgramInfoLog(shaderProgramObject_PF_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Shader Program Object Linking Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				[self release];
                [NSApp terminate: self];
			}
		}
	}

	modelMatrixUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_model_matrix");
	viewMatrixUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_view_matrix");
	projectionMatrixUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_projection_matrix");
	
	red_LaUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Red_La");
	red_LdUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Red_Ld");
	red_LsUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Red_Ls");
	red_lightPositionUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Red_light_position");

	green_LaUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Green_La");
	green_LdUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Green_Ld");
	green_LsUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Green_Ls");
	green_lightPositionUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Green_light_position");

	blue_LaUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Blue_La");
	blue_LdUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Blue_Ld");
	blue_LsUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Blue_Ls");
	blue_lightPositionUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Blue_light_position");

	KaUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Ka");
	KdUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Kd");
	KsUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Ks");
	shininessUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_shininess");
	LKeyPressUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_L_keypress");



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


	[self FillLightsData];

   sphere_Angle = 0.0f;
 

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


-(void)FillLightsData{
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
	
	lights_RRJ[0].lightPosition_RRJ[0] = 0.0f;
	lights_RRJ[0].lightPosition_RRJ[1] = 0.0f;
	lights_RRJ[0].lightPosition_RRJ[2] = 0.0f;
	lights_RRJ[0].lightPosition_RRJ[3] = 1.0f;
	
	//Green Light
	lights_RRJ[1].lightAmbient_RRJ[0] = 0.0f;
	lights_RRJ[1].lightAmbient_RRJ[1] = 0.0f;
	lights_RRJ[1].lightAmbient_RRJ[2] = 0.0f;
	lights_RRJ[1].lightAmbient_RRJ[3] = 0.0f;
	
	lights_RRJ[1].lightDiffuse_RRJ[0] = 0.0f;
	lights_RRJ[1].lightDiffuse_RRJ[1] = 1.0f;
	lights_RRJ[1].lightDiffuse_RRJ[2] = 0.0f;
	lights_RRJ[1].lightDiffuse_RRJ[3] = 1.0f;
	
	lights_RRJ[1].lightSpecular_RRJ[0] = 0.0f;
	lights_RRJ[1].lightSpecular_RRJ[1] = 1.0f;
	lights_RRJ[1].lightSpecular_RRJ[2] = 0.0f;
	lights_RRJ[1].lightSpecular_RRJ[3] = 1.0f;
	
	lights_RRJ[1].lightPosition_RRJ[0] = 0.0f;
	lights_RRJ[1].lightPosition_RRJ[1] = 0.0f;
	lights_RRJ[1].lightPosition_RRJ[2] = 0.0f;
	lights_RRJ[1].lightPosition_RRJ[3] = 1.0f;
	
	
	//Blue Light
	lights_RRJ[2].lightAmbient_RRJ[0] = 0.0f;
	lights_RRJ[2].lightAmbient_RRJ[1] = 0.0f;
	lights_RRJ[2].lightAmbient_RRJ[2] = 0.0f;
	lights_RRJ[2].lightAmbient_RRJ[3] = 0.0f;
	
	lights_RRJ[2].lightDiffuse_RRJ[0] = 0.0f;
	lights_RRJ[2].lightDiffuse_RRJ[1] = 0.0f;
	lights_RRJ[2].lightDiffuse_RRJ[2] = 1.0f;
	lights_RRJ[2].lightDiffuse_RRJ[3] = 1.0f;
	
	lights_RRJ[2].lightSpecular_RRJ[0] = 0.0f;
	lights_RRJ[2].lightSpecular_RRJ[1] = 0.0f;
	lights_RRJ[2].lightSpecular_RRJ[2] = 1.0f;
	lights_RRJ[2].lightSpecular_RRJ[3] = 1.0f;
	
	lights_RRJ[2].lightPosition_RRJ[0] = 0.0f;
	lights_RRJ[2].lightPosition_RRJ[1] = 0.0f;
	lights_RRJ[2].lightPosition_RRJ[2] = 0.0f;
	lights_RRJ[2].lightPosition_RRJ[3] = 1.0f;
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
                
           
            topRight = topLeft + 1;
            bottomRight = bottomLeft + 1;
         
            
                
            
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

    glViewport(0, 0, (GLsizei)width_RRJ, (GLsizei)height_RRJ);

    perspectiveProjectionMatrix_RRJ = vmath::perspective(45.0f, (GLfloat)width_RRJ / (GLfloat)height_RRJ, 0.1f, 100.0f);

    CGLUnlockContext((CGLContextObj)[[self openGLContext] CGLContextObj]);
}


-(void)drawRect:(NSRect)rect {
    [self drawView];
}




-(void)update{
	angle_red_RRJ = angle_red_RRJ + .020f;
	angle_green_RRJ = angle_green_RRJ + .02f;
	angle_blue_RRJ = angle_blue_RRJ + .02f;

	if (angle_red_RRJ > 360.0f)
		angle_red_RRJ = 0.0f;

	if (angle_green_RRJ > 360.0f)
		angle_green_RRJ = 0.0f;

	if (angle_blue_RRJ > 360.0f)
		angle_blue_RRJ = 0.0f;
}




-(void) drawView {
    
    [[self openGLContext]makeCurrentContext];

    CGLLockContext((CGLContextObj) [[self openGLContext] CGLContextObj]);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    vmath::mat4 translateMatrix_RRJ;
    vmath::mat4 rotateMatrix_RRJ;
    vmath::mat4 modelMatrix_RRJ;
    vmath::mat4 viewMatrix_RRJ;


    

    /********** SPHERE **********/
    translateMatrix_RRJ = vmath::mat4::identity();
    rotateMatrix_RRJ = vmath::mat4::identity();
    modelMatrix_RRJ = vmath::mat4::identity();
    viewMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(0.0f, 0.0f, -4.0f);
    modelMatrix_RRJ = modelMatrix_RRJ * translateMatrix_RRJ;

   

	if (iWhichLight_RRJ == PER_VERTEX){
		glUseProgram(shaderProgramObject_PV_RRJ);
		
		glUniformMatrix4fv(modelMatrixUniform_PV_RRJ, 1, GL_FALSE, modelMatrix_RRJ);
		glUniformMatrix4fv(viewMatrixUniform_PV_RRJ, 1, GL_FALSE, viewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, perspectiveProjectionMatrix_RRJ);
			
			
		if (bLights_RRJ == true) {


			[self rotateRedLight: angle_red_RRJ];
			[self rotateGreenLight: angle_green_RRJ];
			[self rotateBlueLight: angle_blue_RRJ];
			
			
			glUniform1i(LKeyPressUniform_PV_RRJ, 1);

			glUniform3fv(red_LaUniform_PV_RRJ, 1, lights_RRJ[0].lightAmbient_RRJ);
			glUniform3fv(red_LdUniform_PV_RRJ, 1, lights_RRJ[0].lightDiffuse_RRJ);
			glUniform3fv(red_LsUniform_PV_RRJ, 1, lights_RRJ[0].lightSpecular_RRJ);
			glUniform4fv(red_lightPositionUniform_PV_RRJ, 1, lights_RRJ[0].lightPosition_RRJ);

			glUniform3fv(green_LaUniform_PV_RRJ, 1, lights_RRJ[1].lightAmbient_RRJ);
			glUniform3fv(green_LdUniform_PV_RRJ, 1, lights_RRJ[1].lightDiffuse_RRJ);
			glUniform3fv(green_LsUniform_PV_RRJ, 1, lights_RRJ[1].lightSpecular_RRJ);
			glUniform4fv(green_lightPositionUniform_PV_RRJ, 1, lights_RRJ[1].lightPosition_RRJ);

			glUniform3fv(blue_LaUniform_PV_RRJ, 1, lights_RRJ[2].lightAmbient_RRJ);
			glUniform3fv(blue_LdUniform_PV_RRJ, 1, lights_RRJ[2].lightDiffuse_RRJ);
			glUniform3fv(blue_LsUniform_PV_RRJ, 1, lights_RRJ[2].lightSpecular_RRJ);
			glUniform4fv(blue_lightPositionUniform_PV_RRJ, 1, lights_RRJ[2].lightPosition_RRJ);


			glUniform3fv(KaUniform_PV_RRJ, 1, materialAmbient_RRJ);
			glUniform3fv(KdUniform_PV_RRJ, 1, materialDiffuse_RRJ);
			glUniform3fv(KsUniform_PV_RRJ, 1, materialSpecular_RRJ);
			glUniform1f(shininessUniform_PV_RRJ, materialShininess_RRJ);
		}
		else
			glUniform1i(LKeyPressUniform_PV_RRJ, 0);		

		glBindVertexArray(vao_Sphere_RRJ);
        	glDrawElements(GL_TRIANGLES, (STACKS - 1) * (SLICES - 1) * 6, GL_UNSIGNED_SHORT, 0);
    		glBindVertexArray(0);

    		glUseProgram(0);		
			
	}
	else{
		glUseProgram(shaderProgramObject_PF_RRJ);
		
		glUniformMatrix4fv(modelMatrixUniform_PF_RRJ, 1, GL_FALSE, modelMatrix_RRJ);
		glUniformMatrix4fv(viewMatrixUniform_PF_RRJ, 1, GL_FALSE, viewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, perspectiveProjectionMatrix_RRJ);
			
			
		if (bLights_RRJ == true) {
		
			[self rotateRedLight: angle_red_RRJ];
			[self rotateGreenLight: angle_green_RRJ];
			[self rotateBlueLight: angle_blue_RRJ];

			glUniform1i(LKeyPressUniform_PF_RRJ, 1);
			glUniform3fv(red_LaUniform_PF_RRJ, 1, lights_RRJ[0].lightAmbient_RRJ);
			glUniform3fv(red_LdUniform_PF_RRJ, 1, lights_RRJ[0].lightDiffuse_RRJ);
			glUniform3fv(red_LsUniform_PF_RRJ, 1, lights_RRJ[0].lightSpecular_RRJ);
			glUniform4fv(red_lightPositionUniform_PF_RRJ, 1, lights_RRJ[0].lightPosition_RRJ);

			glUniform3fv(green_LaUniform_PF_RRJ, 1, lights_RRJ[1].lightAmbient_RRJ);
			glUniform3fv(green_LdUniform_PF_RRJ, 1, lights_RRJ[1].lightDiffuse_RRJ);
			glUniform3fv(green_LsUniform_PF_RRJ, 1, lights_RRJ[1].lightSpecular_RRJ);
			glUniform4fv(green_lightPositionUniform_PF_RRJ, 1, lights_RRJ[1].lightPosition_RRJ);

			glUniform3fv(blue_LaUniform_PF_RRJ, 1, lights_RRJ[2].lightAmbient_RRJ);
			glUniform3fv(blue_LdUniform_PF_RRJ, 1, lights_RRJ[2].lightDiffuse_RRJ);
			glUniform3fv(blue_LsUniform_PF_RRJ, 1, lights_RRJ[2].lightSpecular_RRJ);
			glUniform4fv(blue_lightPositionUniform_PF_RRJ, 1, lights_RRJ[2].lightPosition_RRJ);
			
		
			glUniform3fv(KaUniform_PF_RRJ, 1, materialAmbient_RRJ);
			glUniform3fv(KdUniform_PF_RRJ, 1, materialDiffuse_RRJ);
			glUniform3fv(KsUniform_PF_RRJ, 1, materialSpecular_RRJ);
			glUniform1f(shininessUniform_PF_RRJ, materialShininess_RRJ);
		}
		else
			glUniform1i(LKeyPressUniform_PF_RRJ, 0);	

		glBindVertexArray(vao_Sphere_RRJ);
        	glDrawElements(GL_TRIANGLES, (STACKS - 1) * (SLICES - 1) * 6, GL_UNSIGNED_SHORT, 0);
    		glBindVertexArray(0);

    		glUseProgram(0);
	}
    

	[self update];
    
    CGLFlushDrawable((CGLContextObj) [[self openGLContext] CGLContextObj]);
    CGLUnlockContext((CGLContextObj) [[self openGLContext] CGLContextObj]);
}

-(void) rotateRedLight: (GLfloat)angle{
	
	lights_RRJ[0].lightPosition_RRJ[1] = (float)(5.0f * sin(angle));
	lights_RRJ[0].lightPosition_RRJ[2] = (float)(5.0f * cos(angle));
	lights_RRJ[0].lightPosition_RRJ[3] = 0.0f;

}

-(void) rotateGreenLight:(GLfloat)angle {

	lights_RRJ[1].lightPosition_RRJ[0] = (float)(5.0f * sin(angle));
	lights_RRJ[1].lightPosition_RRJ[1] = 0.0f; 
	lights_RRJ[1].lightPosition_RRJ[2] = (float)(5.0f * cos(angle)); 

}

-(void) rotateBlueLight:(GLfloat)angle {

	lights_RRJ[2].lightPosition_RRJ[0] = (float)(5.0f * cos(angle)); 
	lights_RRJ[2].lightPosition_RRJ[1] = (float)(5.0f * sin(angle));
	lights_RRJ[2].lightPosition_RRJ[2] = 0.0f;
}


-(BOOL) acceptsFirstResponder {
    [[self window]makeFirstResponder: self];
    return(YES);
}

-(void) keyDown: (NSEvent*) event {

    int key = (int)[[event characters]characterAtIndex: 0];
    switch(key){
        case 27:
            [[self window]toggleFullScreen: self];
            break;

        case 'F':
        case 'f':
        	iWhichLight_RRJ = PER_FRAGMENT;
        	break;

        case 'V':
        case 'v':
        	iWhichLight_RRJ = PER_VERTEX;
        	break;
           
       case 'Q':
       case 'q':
       		[self release];
		[NSApp terminate: self];
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
    

    if(vbo_Sphere_Index_RRJ){
    	glDeleteBuffers(1, &vbo_Sphere_Index_RRJ);
    	vbo_Sphere_Index_RRJ = 0;
    }

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
    
    
    
    	GLsizei ShaderCount_RRJ;
	GLsizei ShaderNumber_RRJ;

	if (shaderProgramObject_PV_RRJ) {
		glUseProgram(shaderProgramObject_PV_RRJ);

		glGetProgramiv(shaderProgramObject_PV_RRJ, GL_ATTACHED_SHADERS, &ShaderCount_RRJ);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount_RRJ);
		if (pShader) {
			glGetAttachedShaders(shaderProgramObject_PV_RRJ, ShaderCount_RRJ, &ShaderCount_RRJ, pShader);
			for (ShaderNumber_RRJ = 0; ShaderNumber_RRJ < ShaderCount_RRJ; ShaderNumber_RRJ++) {
				glDetachShader(shaderProgramObject_PV_RRJ, pShader[ShaderNumber_RRJ]);
				glDeleteShader(pShader[ShaderNumber_RRJ]);
				pShader[ShaderNumber_RRJ] = 0;
			}
			free(pShader);
			pShader = NULL;
		}
		glDeleteProgram(shaderProgramObject_PV_RRJ);
		shaderProgramObject_PV_RRJ = 0;
		glUseProgram(0);
	}


	ShaderCount_RRJ = 0;
	ShaderNumber_RRJ = 0;
	if (shaderProgramObject_PF_RRJ) {
		glUseProgram(shaderProgramObject_PF_RRJ);

		glGetProgramiv(shaderProgramObject_PF_RRJ, GL_ATTACHED_SHADERS, &ShaderCount_RRJ);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount_RRJ);
		if (pShader) {
			glGetAttachedShaders(shaderProgramObject_PF_RRJ, ShaderCount_RRJ, &ShaderCount_RRJ, pShader);
			for (ShaderNumber_RRJ = 0; ShaderNumber_RRJ < ShaderCount_RRJ; ShaderNumber_RRJ++) {
				glDetachShader(shaderProgramObject_PF_RRJ, pShader[ShaderNumber_RRJ]);
				glDeleteShader(pShader[ShaderNumber_RRJ]);
				pShader[ShaderNumber_RRJ] = 0;
			}
			free(pShader);
			pShader = NULL;
		}
		glDeleteProgram(shaderProgramObject_PF_RRJ);
		shaderProgramObject_PF_RRJ = 0;
		glUseProgram(0);
	}


    CVDisplayLinkStop(displayLink_RRJ);
    CVDisplayLinkRelease(displayLink_RRJ);

    [super dealloc];
}

@end

CVReturn MyDisplayLinkCallback(CVDisplayLinkRef displayLink, const CVTimeStamp *pNow, const CVTimeStamp *pOutputTime, CVOptionFlags flagsIn, CVOptionFlags *pFlagsOut, void *pDisplayLinkContext){

    CVReturn result_RRJ = [(GLView_RRJ*)pDisplayLinkContext getFrameForTime: pOutputTime];
    return(result_RRJ);
}




