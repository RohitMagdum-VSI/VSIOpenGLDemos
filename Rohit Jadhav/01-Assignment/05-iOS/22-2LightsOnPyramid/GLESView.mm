//
//  GLESView.m
//  22-2LightsOnPyramid
//
//  Created by user160249 on 3/28/20.
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


@implementation GLESView{
    
    EAGLContext *eaglContext_RRJ;

    GLuint defaultFramebuffer_RRJ;
    GLuint colorRenderbuffer_RRJ;
    GLuint depthRenderbuffer_RRJ;


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
                "gl_Position =  u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
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
        glBindAttribLocation(shaderProgramObject_RRJ, AMC_ATTRIBUTE_NORMAL, "vNormals");

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
        glClearDepthf(1.0f);

        glDisable(GL_CULL_FACE);


        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        angle_Pyramid = 0.0f;
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








+(Class)layerClass{
    return([CAEAGLLayer class]);
}

-(void)drawView:(id)sender{

    [EAGLContext setCurrentContext:eaglContext_RRJ];

    glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebuffer_RRJ);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

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


    glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer_RRJ);
    [eaglContext_RRJ presentRenderbuffer: GL_RENDERBUFFER];

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

    if(bLights_RRJ == false)
        bLights_RRJ = true;
    else
        bLights_RRJ = false;
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

