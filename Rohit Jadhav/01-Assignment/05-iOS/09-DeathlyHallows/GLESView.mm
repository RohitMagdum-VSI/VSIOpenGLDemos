//
//  GLESView.m
//  09-DeathlyHallows
//
//  Created by user160249 on 3/23/20.
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

    //For  Triangle_RRJ
    GLuint vao_Triangle_RRJ;
    GLuint vbo_Triangle_Position_RRJ;
    GLuint vbo_Triangle_Color_RRJ;

    //For  InCircle
    GLuint vao_Circle_RRJ;
    GLuint vbo_Circle_Position_RRJ;
    GLuint vbo_Circle_Color_RRJ;

    GLfloat Incircle_Center_RRJ[3];
    GLfloat Incircle_Radius_RRJ;

    //For Wand
    GLuint vao_Wand_RRJ;
    GLuint vbo_Wand_Color_RRJ;
    GLuint vbo_Wand_Position_RRJ;

    GLuint mvpUniform_RRJ;
    GLuint pointSizeUniform_RRJ;

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
            "uniform float u_pointSize;" \
            "uniform mat4 u_mvp_matrix;" \
            "void main(void) {" \
                "gl_PointSize = u_pointSize;" \
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
        
        pointSizeUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ,"u_pointSize");


        /********** Position  **********/
        GLfloat Triangle_Position_RRJ[] = {
            0.0f, 0.70f, 0.0f,
            -0.70f, -0.70f, 0.0f,
            0.70f, -0.70f, 0.0f
        };

        GLfloat Triangle_Color_RRJ[] = {
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f
        };

        GLfloat Circle_Position_RRJ[3 * 3000];
        //GLfloat Circle_Color_RRJ[3 * 3000];

        GLfloat X = (GLfloat)(Triangle_Position_RRJ[6] + Triangle_Position_RRJ[3]) / 2.0f;

        GLfloat Wand_Position_RRJ[] = {
            Triangle_Position_RRJ[0], Triangle_Position_RRJ[1], Triangle_Position_RRJ[2],
            X, Triangle_Position_RRJ[7], 0.0f
        };

        GLfloat Wand_Color_RRJ[] = {
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f
        };


        /********** To Calculate InCircle Radius and Center **********/
        [self Calculation: Triangle_Position_RRJ];


        /********** Fill Circle_Position **********/
        [self FillCircle_Position: Circle_Position_RRJ];




        /********** Triangle_RRJ **********/
        glGenVertexArrays(1, &vao_Triangle_RRJ);
        glBindVertexArray(vao_Triangle_RRJ);

            /********** Position **********/
            glGenBuffers(1, &vbo_Triangle_Position_RRJ);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_Triangle_Position_RRJ);
            glBufferData(GL_ARRAY_BUFFER, sizeof(Triangle_Position_RRJ), Triangle_Position_RRJ, GL_STATIC_DRAW);
            glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            /********** Color **********/
            glGenBuffers(1, &vbo_Triangle_Color_RRJ);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_Triangle_Color_RRJ);
            glBufferData(GL_ARRAY_BUFFER, sizeof(Triangle_Color_RRJ), Triangle_Color_RRJ, GL_STATIC_DRAW);
            glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindVertexArray(0);



        /********** Circle **********/
        glGenVertexArrays(1, &vao_Circle_RRJ);
        glBindVertexArray(vao_Circle_RRJ);

            /********** Position **********/
            glGenBuffers(1, &vbo_Circle_Position_RRJ);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_Circle_Position_RRJ);
            glBufferData(GL_ARRAY_BUFFER, sizeof(Circle_Position_RRJ), Circle_Position_RRJ, GL_STATIC_DRAW);
            glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindVertexArray(0);



        /********** Wand **********/
        glGenVertexArrays(1, &vao_Wand_RRJ);
        glBindVertexArray(vao_Wand_RRJ);

            /********** Position **********/
            glGenBuffers(1, &vbo_Wand_Position_RRJ);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_Wand_Position_RRJ);
            glBufferData(GL_ARRAY_BUFFER, sizeof(Wand_Position_RRJ), Wand_Position_RRJ, GL_STATIC_DRAW);
            glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            /********** Color **********/
            glGenBuffers(1, &vbo_Wand_Color_RRJ);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_Wand_Color_RRJ);
            glBufferData(GL_ARRAY_BUFFER, sizeof(Wand_Color_RRJ), Wand_Color_RRJ, GL_STATIC_DRAW);
            glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindVertexArray(0);


        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glClearDepthf(1.0f);

        glEnable(GL_CULL_FACE);
        
        //glEnable(GL_PROGRAM_POINT_SIZE);


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

    vmath::mat4 translateMatrix_RRJ;
    vmath::mat4 rotateMatrix_RRJ;
    vmath::mat4 modelViewMatrix_RRJ;
    vmath::mat4 modelViewProjectionMatrix_RRJ;

    static GLfloat Tri_X_RRJ = 0.001f;
    static GLfloat Tri_Y_RRJ = 0.001f;
    static GLfloat Cir_X_RRJ = 0.001f;
    static GLfloat Cir_Y_RRJ = 0.001f;
    static GLfloat Wand_Y_RRJ = 0.001f;
    static GLfloat angle_RRJ = 0.0f;


    glUseProgram(shaderProgramObject_RRJ);

    /********** TRIANGLE **********/
    translateMatrix_RRJ = vmath::mat4::identity();
    rotateMatrix_RRJ = vmath::mat4::identity();
    modelViewMatrix_RRJ = vmath::mat4::identity();
    modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(3.6f - Tri_X_RRJ, -1.8f + Tri_Y_RRJ, -6.0f);

    if (Tri_X_RRJ < 3.6f && Cir_X_RRJ < 3.6f)
        rotateMatrix_RRJ = vmath::rotate(angle_RRJ, 0.0f, 1.0f, 0.0f);

    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ * rotateMatrix_RRJ;
    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    glUniformMatrix4fv(mvpUniform_RRJ, 1, false, modelViewProjectionMatrix_RRJ);


    glLineWidth(1.5f);
    glBindVertexArray(vao_Triangle_RRJ);
        glDrawArrays(GL_LINE_LOOP, 0, 3);
    glBindVertexArray(0);




    /********** Circle **********/
    translateMatrix_RRJ = vmath::mat4::identity();
    rotateMatrix_RRJ = vmath::mat4::identity();
    modelViewMatrix_RRJ = vmath::mat4::identity();
    modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(-3.6f + Cir_X_RRJ, -1.8f + Cir_Y_RRJ, -6.0f);
    rotateMatrix_RRJ = vmath::rotate(angle_RRJ, 0.0f, 1.0f, 0.0f);
    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ * rotateMatrix_RRJ;
    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    glUniformMatrix4fv(mvpUniform_RRJ, 1, false,
                    modelViewProjectionMatrix_RRJ);

    glUniform1f(pointSizeUniform_RRJ, 2.f);
    glBindVertexArray(vao_Circle_RRJ);
        glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);
        glDrawArrays(GL_POINTS, 0, 1000);
    glBindVertexArray(0);


    /********** Wand **********/
    translateMatrix_RRJ = vmath::mat4::identity();
    rotateMatrix_RRJ = vmath::mat4::identity();
    modelViewMatrix_RRJ = vmath::mat4::identity();
    modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(0.0f, 1.80f - Wand_Y_RRJ, -6.0f);
    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    glUniformMatrix4fv(mvpUniform_RRJ, 1, false, modelViewProjectionMatrix_RRJ);

    glLineWidth(1.5f);
    glBindVertexArray(vao_Wand_RRJ);
    glDrawArrays(GL_LINES, 0, 2);
    glBindVertexArray(0);

    glUseProgram(0);

    Tri_X_RRJ = Tri_X_RRJ + 0.008f;
    Tri_Y_RRJ = Tri_Y_RRJ + 0.004f;

    if (Tri_X_RRJ > 3.6f && Tri_Y_RRJ > 1.8f) {
        Tri_X_RRJ = 3.6f;
        Tri_Y_RRJ = 1.8f;
    }

    Cir_X_RRJ = Cir_X_RRJ + 0.008f;
    Cir_Y_RRJ = Cir_Y_RRJ + 0.004f;

    if (Cir_X_RRJ > 3.6f && Cir_Y_RRJ > 1.8f) {
        Cir_X_RRJ = 3.6f;
        Cir_Y_RRJ = 1.8f;
    }

    Wand_Y_RRJ = Wand_Y_RRJ + 0.004f;
    if (Wand_Y_RRJ > 1.8f)
        Wand_Y_RRJ = 1.8f;

    angle_RRJ = angle_RRJ + 2.0f;


    glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer_RRJ);
    [eaglContext_RRJ presentRenderbuffer: GL_RENDERBUFFER];

}


-(void)Calculation:(GLfloat[]) arr{
    GLfloat a, b, c;
    GLfloat s;

    //Distance Formula
    a = (GLfloat)sqrt(pow((arr[6] - arr[3]), 2) + pow((arr[7] - arr[4]), 2));
    b = (GLfloat)sqrt(pow((arr[6] - arr[0]), 2) + pow((arr[7] - arr[1]), 2));
    c = (GLfloat)sqrt(pow((arr[3] - arr[0]), 2) + pow((arr[4] - arr[1]), 2));

    s = (a + b + c) / 2;

    Incircle_Radius_RRJ = (GLfloat)(sqrt(s * (s - a) * (s - b) * (s - c)) / s);

    Incircle_Center_RRJ[0] = (a * arr[0] + b * arr[3] + c * arr[6]) / (a + b + c);
    Incircle_Center_RRJ[1] = (a * arr[1] + b * arr[4] + c * arr[7]) / (a + b + c);
    Incircle_Center_RRJ[2] = 0.0f;


    printf("Incircle_Radius_RRJ: %f\n", Incircle_Radius_RRJ);
    printf("InCenter x: %f      y: %f      z: %f     \n", Incircle_Center_RRJ[0], Incircle_Center_RRJ[1], Incircle_Center_RRJ[2]);
}


-(void)FillCircle_Position: (GLfloat[])arr{
    //InCircle
    for (int i = 0; i < 3000; i = i + 3) {
        GLfloat x = (GLfloat)(2.0f * M_PI * i / 3000);
        arr[i] = (GLfloat)(Incircle_Radius_RRJ * cos(x)) + Incircle_Center_RRJ[0];
        arr[i + 1] = (GLfloat)(Incircle_Radius_RRJ * sin(x)) + Incircle_Center_RRJ[1];
        arr[i + 2] = 0.0f;


        /*arrColor[i] = 1.0f;     //R
        arrColor[i + 1] = 1.0f;     //G
        arrColor[i + 2] = 1.0f;     //B*/
    }
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

    if (vbo_Wand_Color_RRJ) {
        glDeleteBuffers(1, &vbo_Wand_Color_RRJ);
        vbo_Wand_Color_RRJ = 0;
    }

    if (vbo_Wand_Position_RRJ) {
        glDeleteBuffers(1, &vbo_Wand_Position_RRJ);
        vbo_Wand_Position_RRJ = 0;
    }

    if (vao_Wand_RRJ) {
        glDeleteVertexArrays(1, &vao_Wand_RRJ);
        vao_Wand_RRJ = 0;
    }


    /*if (vbo_Circle_Color_RRJ) {
        glDeleteBuffers(1, &vbo_Circle_Color_RRJ);
        vbo_Circle_Color_RRJ = 0;
    }*/

    if (vbo_Circle_Position_RRJ) {
        glDeleteBuffers(1, &vbo_Circle_Position_RRJ);
        vbo_Circle_Position_RRJ = 0;
    }

    if (vao_Circle_RRJ) {
        glDeleteVertexArrays(1, &vao_Circle_RRJ);
        vao_Circle_RRJ = 0;
    }


    if (vbo_Triangle_Color_RRJ) {
        glDeleteBuffers(1, &vbo_Triangle_Color_RRJ);
        vbo_Triangle_Color_RRJ = 0;
    }

    if (vbo_Triangle_Position_RRJ) {
        glDeleteBuffers(1, &vbo_Triangle_Position_RRJ);
        vbo_Triangle_Position_RRJ = 0;
    }

    if (vao_Triangle_RRJ) {
        glDeleteVertexArrays(1, &vao_Triangle_RRJ);
        vao_Triangle_RRJ = 0;
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
