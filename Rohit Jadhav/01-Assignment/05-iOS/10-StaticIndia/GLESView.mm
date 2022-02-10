//
//  GLESView.m
//  10-StaticIndia
//
//  Created by user160249 on 3/24/20.
//

#import<OpenGLES/ES3/gl.h>
#import<OpenGLES/ES3/glext.h>

#import"vmath.h"

#import"GLESView.h"



enum {
    AMC_ATTRIBUTE_POSITION = 0,
    AMC_ATTRIBUTE_COLOR,
    AMC_ATTRIBUTE_NORMAL,
    AMC_ATTRIBUTE_TEXCOORD0,
};


@implementation GLESView{
    
    EAGLContext *eaglContext_RRJ;

    GLuint defaultFramebuffer_RRJ;
    GLuint colorRenderbuffer_RRJ;
    GLuint depthRenderbuffer_RRJ;


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

    id displayLink_RRJ;
    NSInteger animationFrameInterval_RRJ;
    BOOL isAnimating_RRJ;
}

-(id)initWithFrame:(CGRect)frame{

    self = [super initWithFrame: frame];

    if(self){

        CAEAGLLayer *caeaglLayer_RRJ = (CAEAGLLayer*)super.layer;

        caeaglLayer_RRJ.opaque = YES;
        caeaglLayer_RRJ.drawableProperties = [NSDictionary dictionaryWithObjectsAndKeys:
            [NSNumber numberWithBool: FALSE],  kEAGLDrawablePropertyRetainedBacking,
            kEAGLColorFormatRGBA8, kEAGLDrawablePropertyColorFormat, nil
            ];

        eaglContext_RRJ = [[EAGLContext alloc]initWithAPI: kEAGLRenderingAPIOpenGLES3];
        if(eaglContext_RRJ == nil){
            [self release];
            return(nil);
        }

        [EAGLContext setCurrentContext:eaglContext_RRJ];

        glGenFramebuffers(1, &defaultFramebuffer_RRJ);
        glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebuffer_RRJ);
        
        glGenRenderbuffers(1, &colorRenderbuffer_RRJ);
        glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer_RRJ);

        [eaglContext_RRJ renderbufferStorage:GL_RENDERBUFFER fromDrawable:caeaglLayer_RRJ];

        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, colorRenderbuffer_RRJ);

        GLint backingWidth_RRJ;
        GLint backingHeight_RRJ;
        glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &backingWidth_RRJ);
        glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &backingHeight_RRJ);

        glGenRenderbuffers(1, &depthRenderbuffer_RRJ);
        glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer_RRJ);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, backingWidth_RRJ, backingHeight_RRJ);

        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer_RRJ);

        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
            printf("ERROR: Failed to Create Complete Framebuffer Object %x\n", glCheckFramebufferStatus(GL_FRAMEBUFFER));
            glDeleteFramebuffers(1, &defaultFramebuffer_RRJ);
            glDeleteRenderbuffers(1, &depthRenderbuffer_RRJ);
            glDeleteRenderbuffers(1, &colorRenderbuffer_RRJ);

            return(nil);
        }


        printf("SUCCESS: Renderer: %s | GL_VERSION: %s | GLSL Version: %s\n",
            glGetString(GL_RENDERER), glGetString(GL_VERSION), glGetString(GL_SHADING_LANGUAGE_VERSION));

        isAnimating_RRJ = NO;
        animationFrameInterval_RRJ = 60;



        /********** Vertex Shader **********/
        vertexShaderObject_RRJ = glCreateShader(GL_VERTEX_SHADER);
        const GLchar *vertexShaderSourceCode_RRJ =
            "#version 300 es" \
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
                    printf("Vertex Shader Compilation Error: %s\n", szInfoLog_RRJ);
                    free(szInfoLog_RRJ);
                    szInfoLog_RRJ = NULL;
                    [self release];
                }
            }
        }



        /********** Fragment Shader **********/
        fragmentShaderObject_RRJ = glCreateShader(GL_FRAGMENT_SHADER);
        const GLchar *fragmentShaderSourceCode_RRJ =
            "#version 300 es" \
            "\n" \
            "precision highp float;" \
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
                    printf("Fragment Shader Compilation Error: %s\n", szInfoLog_RRJ);
                    free(szInfoLog_RRJ);
                    szInfoLog_RRJ = NULL;
                    [self release];
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

        GLint iProgramLinkStatus;
        iInfoLogLength_RRJ = 0;
        szInfoLog_RRJ = NULL;

        glGetProgramiv(shaderProgramObject_RRJ, GL_LINK_STATUS, &iProgramLinkStatus);
        if(iProgramLinkStatus == GL_FALSE){
            glGetProgramiv(shaderProgramObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
            if(iInfoLogLength_RRJ > 0){
                szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
                if(szInfoLog_RRJ){
                    GLsizei written;
                    glGetProgramInfoLog(shaderProgramObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
                    printf("Shader Program Linking Error: %s\n", szInfoLog_RRJ);
                    szInfoLog_RRJ = NULL;
                    [self release];
                    
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
        glClearDepthf(1.0f);

        glEnable(GL_CULL_FACE);


        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        perspectiveProjectionMatrix_RRJ = vmath::mat4::identity();

        //Gesture
        UITapGestureRecognizer *singleTapGestureRecognizer_RRJ = [[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(onSingleTap:)];
        [singleTapGestureRecognizer_RRJ setNumberOfTapsRequired:1];
        [singleTapGestureRecognizer_RRJ setNumberOfTouchesRequired:1];
        [singleTapGestureRecognizer_RRJ setDelegate:self];
        [self addGestureRecognizer: singleTapGestureRecognizer_RRJ];


        UITapGestureRecognizer *doubleTapGestureRecognizer_RRJ = [[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(onDoubleTap:)];
        [doubleTapGestureRecognizer_RRJ setNumberOfTapsRequired:2];
        [doubleTapGestureRecognizer_RRJ setNumberOfTouchesRequired:1];
        [doubleTapGestureRecognizer_RRJ setDelegate:self];
        [self addGestureRecognizer: doubleTapGestureRecognizer_RRJ];


        [singleTapGestureRecognizer_RRJ requireGestureRecognizerToFail: doubleTapGestureRecognizer_RRJ];


        UISwipeGestureRecognizer *swipeGestureRecognizer_RRJ = [[UISwipeGestureRecognizer alloc]initWithTarget:self action:@selector(onSwipe:)];
        [self addGestureRecognizer: swipeGestureRecognizer_RRJ];

        UILongPressGestureRecognizer *longPressGestureRecognizer_RRJ = [[UILongPressGestureRecognizer alloc]initWithTarget:self action:@selector(onLongPress:)];
        [self addGestureRecognizer: longPressGestureRecognizer_RRJ];
    }

    return(self);
}

+(Class)layerClass{
    return([CAEAGLLayer class]);
}

-(void)drawView:(id)sender{

    [EAGLContext setCurrentContext:eaglContext_RRJ];

    glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebuffer_RRJ);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    
    glUseProgram(shaderProgramObject_RRJ);

        [self My_I: -2.0f yTrans:0.0f  zTrans:-8.0f  fWidth:20.0f];
        [self My_N: -1.35f  yTrans:0.0f zTrans:-8.0f fWidth:20.0f];
        [self My_D: -0.15f  yTrans:0.0f zTrans:-8.0f fWidth:20.0f];
        [self My_I: 1.02f yTrans:0.0f zTrans:-8.0f fWidth:20.0f];
        [self My_A: 2.0f yTrans:0.0f zTrans:-8.0f fWidth:20.0f];
        [self My_Flag: 2.0f yTrans:0.0f zTrans:-8.0f fWidth:20.0f];

    glUseProgram(0);


    glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer_RRJ);
    [eaglContext_RRJ presentRenderbuffer: GL_RENDERBUFFER];

}


-(void)My_I:(GLfloat)x yTrans:(GLfloat)y zTrans:(GLfloat)z fWidth:(GLfloat)fWidth{

    translateMatrix_RRJ = vmath::mat4::identity();
    modelViewMatrix_RRJ = vmath::mat4::identity();
    modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(x, y, z);
    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    glUniformMatrix4fv(mvpUniform_RRJ, 1, GL_FALSE, modelViewProjectionMatrix_RRJ);

    //glLineWidth(fWidth);
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

    //glLineWidth(fWidth);
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

    //glLineWidth(fWidth);
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

    //glLineWidth(fWidth);
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

    //glLineWidth(fWidth);
    glBindVertexArray(vao_Flag_RRJ);
    glDrawArrays(GL_LINES, 0, 30);
    glBindVertexArray(0);
}




-(void)layoutSubviews{
    
    GLint width_RRJ;
    GLint height_RRJ;

    glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer_RRJ);
    [eaglContext_RRJ renderbufferStorage: GL_RENDERBUFFER fromDrawable:(CAEAGLLayer*)self.layer];

    glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &width_RRJ);
    glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &height_RRJ);

    glGenRenderbuffers(1, &depthRenderbuffer_RRJ);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer_RRJ);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, width_RRJ, height_RRJ);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer_RRJ);

    glViewport(0, 0, width_RRJ, height_RRJ);

    perspectiveProjectionMatrix_RRJ = vmath::perspective(45.0f, (GLfloat)width_RRJ / (GLfloat)height_RRJ, 0.1f, 100.0f);

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
        printf("ERROR: Failed to Create Complete Framebuffer Object: %x\n", glCheckFramebufferStatus(GL_FRAMEBUFFER));
    }

    [self drawView:nil];
}

-(void)startAnimation{

    if(!isAnimating_RRJ){
        displayLink_RRJ  = [NSClassFromString(@"CADisplayLink")
                displayLinkWithTarget:self selector:@selector(drawView:)];

        [displayLink_RRJ setPreferredFramesPerSecond:animationFrameInterval_RRJ];
        [displayLink_RRJ addToRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];

        isAnimating_RRJ = YES;
    }
}

-(void)stopAnimation{
    if(isAnimating_RRJ){
        [displayLink_RRJ invalidate];
        displayLink_RRJ = nil;

        isAnimating_RRJ = NO;
    }
}

-(BOOL)acceptsFirstResponder{
    return(YES);
}

-(void)touchesBegan:(NSSet*)touches withEvent:(UIEvent*)event{

}

-(void)onSingleTap:(UITapGestureRecognizer*)gr{

}

-(void)onDoubleTap:(UITapGestureRecognizer*)gr{

}

-(void)onLongPress:(UILongPressGestureRecognizer*)gr{

}

-(void)onSwipe:(UISwipeGestureRecognizer*)gr{

    [self release];
    exit(0);
}

-(void)dealloc{

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



    GLsizei shaderCount_RRJ = 0;
    GLuint shaderNo_RRJ = 0;

    glUseProgram(shaderProgramObject_RRJ);

    glGetProgramiv(shaderProgramObject_RRJ, GL_ATTACHED_SHADERS, &shaderCount_RRJ);
    printf("Shader Count: %d\n",  shaderCount_RRJ);
    GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * shaderCount_RRJ);
    if(pShader){

        glGetAttachedShaders(shaderProgramObject_RRJ, shaderCount_RRJ, &shaderCount_RRJ, pShader);
        for(shaderNo_RRJ = 0; shaderNo_RRJ < shaderCount_RRJ; shaderNo_RRJ++){
            glDetachShader(shaderProgramObject_RRJ, pShader[shaderNo_RRJ]);
            glDeleteShader(pShader[shaderNo_RRJ]);
            pShader[shaderNo_RRJ] = 0;
        }
        free(pShader);
        pShader = NULL;
    }

        glUseProgram(0);
        glDeleteProgram(shaderProgramObject_RRJ);
        shaderProgramObject_RRJ = 0;



    if(depthRenderbuffer_RRJ){
        glDeleteRenderbuffers(1, &depthRenderbuffer_RRJ);
        depthRenderbuffer_RRJ = 0;
    }

    if(colorRenderbuffer_RRJ){
        glDeleteRenderbuffers(1, &colorRenderbuffer_RRJ);
        colorRenderbuffer_RRJ = 0;
    }

    if(defaultFramebuffer_RRJ){
        glDeleteFramebuffers(1, &defaultFramebuffer_RRJ);
        defaultFramebuffer_RRJ = 0;
    }


    if([EAGLContext currentContext] == eaglContext_RRJ){
        [EAGLContext setCurrentContext:nil];
    }

    [eaglContext_RRJ release];
    eaglContext_RRJ = nil;

    [super dealloc];

}

@end
