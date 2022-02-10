//
//  GLESView.m
//  28-Interleaved
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
GLfloat lightAmbient_RRJ[] = { 0.250f, 0.250f, 0.250f, 0.0f };
GLfloat lightDiffuse_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat lightSpecular_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat lightPosition_RRJ[] = { 100.0f, 100.0f, 100.0f, 1.0f };

//For Material
GLfloat materialAmbient_RRJ[] = { 0.25f, 0.25f, 0.25f, 0.0f };
GLfloat materialDiffuse_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialSpecular_RRJ[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialShininess_RRJ = 128.0f;



@implementation GLESView{
    
    EAGLContext *eaglContext_RRJ;

    GLuint defaultFramebuffer_RRJ;
    GLuint colorRenderbuffer_RRJ;
    GLuint depthRenderbuffer_RRJ;


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
    
    //For Texture
    GLuint texture_Marble;
    GLuint samplerUniform;

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
            "precision highp sampler2D;" \
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
        glBindAttribLocation(shaderProgramObject_RRJ, AMC_ATTRIBUTE_TEXCOORD0, "vTex");
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


        texture_Marble = [self loadTextureFromBMP: @"marble" :@"bmp"];


        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glClearDepthf(1.0f);

        glDisable(GL_CULL_FACE);

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

-(GLuint)loadTextureFromBMP:(NSString*)fileName :(NSString*)extention{

    NSString *textureFilePath_RRJ = [[NSBundle mainBundle] pathForResource: fileName ofType:extention];

    UIImage *bmpImage_RRJ = [[UIImage alloc]initWithContentsOfFile:textureFilePath_RRJ];
    if(!bmpImage_RRJ){
        NSLog(@"ERROR: *bmpImage_RRJ cant find: %@\n", textureFilePath_RRJ);
        return(0);
    }


    CGImageRef cgImage_RRJ = bmpImage_RRJ.CGImage;
    int w_RRJ = (int)CGImageGetWidth(cgImage_RRJ);
    int h_RRJ = (int)CGImageGetHeight(cgImage_RRJ);

    CFDataRef imageData_RRJ = CGDataProviderCopyData(CGImageGetDataProvider(cgImage_RRJ));

    void *pixel_RRJ = (void*)CFDataGetBytePtr(imageData_RRJ);

    GLuint bmpTexture_RRJ;

    glGenTextures(1, &bmpTexture_RRJ);
    glBindTexture(GL_TEXTURE_2D, bmpTexture_RRJ);
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

    CFRelease(imageData_RRJ);
    return(bmpTexture_RRJ);

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

