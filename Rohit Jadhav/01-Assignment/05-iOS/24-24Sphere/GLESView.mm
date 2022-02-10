//
//  GLESView.m
//  24-24Sphere
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

@implementation GLESView{
    
    EAGLContext *eaglContext_RRJ;

    GLuint defaultFramebuffer_RRJ;
    GLuint colorRenderbuffer_RRJ;
    GLuint depthRenderbuffer_RRJ;


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
        glBindAttribLocation(shaderProgramObject_RRJ, AMC_ATTRIBUTE_NORMAL, "vNormal");

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
        glClearDepthf(1.0f);

        glDisable(GL_CULL_FACE);




        glClearColor(0.250f, 0.250f, 0.250f, 1.0f);

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




+(Class)layerClass{
    return([CAEAGLLayer class]);
}



vmath::mat4 translateMatrix_RRJ;
vmath::mat4 rotateMatrix_RRJ;
vmath::mat4 modelMatrix_RRJ;
vmath::mat4 viewMatrix_RRJ;


-(void)drawView:(id)sender{

    [EAGLContext setCurrentContext:eaglContext_RRJ];

    glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebuffer_RRJ);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    glUseProgram(shaderProgramObject_RRJ);

        [self draw24Spheres];

    glUseProgram(0);

    [self update];


    glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer_RRJ);
    [eaglContext_RRJ presentRenderbuffer: GL_RENDERBUFFER];

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

    w  = width_RRJ;
    h = height_RRJ;

    //glViewport(0, 0, width_RRJ, height_RRJ);

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

    if(bLights_RRJ  == false)
        bLights_RRJ = true;
    else
        bLights_RRJ = false;

}

-(void)onDoubleTap:(UITapGestureRecognizer*)gr{

    if(iWhichRotation_RRJ == X_ROT)
        iWhichRotation_RRJ = Y_ROT;
    else if(iWhichRotation_RRJ == Y_ROT)
        iWhichRotation_RRJ = Z_ROT;
    else
        iWhichRotation_RRJ = X_ROT;

}

-(void)onLongPress:(UILongPressGestureRecognizer*)gr{

}

-(void)onSwipe:(UISwipeGestureRecognizer*)gr{

    [self release];
    exit(0);
}

-(void)dealloc{

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
