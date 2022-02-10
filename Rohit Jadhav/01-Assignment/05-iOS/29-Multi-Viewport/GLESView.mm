//
//  GLESView.m
//  29-Multi-Viewport
//
//  Created by user160249 on 4/14/20.
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

    GLuint vao_Triangle_RRJ;
    GLuint vbo_Triangle_Position_RRJ;
    GLuint vbo_Triangle_Color_RRJ;

    GLuint mvpUniform_RRJ;

    GLuint iViewPort_RRJ;
    GLsizei viewPortWidth_RRJ;
    GLsizei viewPortHeight_RRJ;


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


        /********** Triangle Position, vao, vbo **********/
        GLfloat tri_Position[] = {
           0.0f, 1.0f, 0.0f,
            -1.0, -1.0f, 0.0f,
            1.0f, -1.0f, 0.0f,
        };
        
        GLfloat tri_Color[] = {
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f,
        };

        
        
        glGenVertexArrays(1, &vao_Triangle_RRJ);
        glBindVertexArray(vao_Triangle_RRJ);

            /********** Position **********/
            glGenBuffers(1, &vbo_Triangle_Position_RRJ);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_Triangle_Position_RRJ);
            glBufferData(GL_ARRAY_BUFFER, sizeof(tri_Position), tri_Position, GL_STATIC_DRAW);
            glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            /*********** Color **********/
            glGenBuffers(1, &vbo_Triangle_Color_RRJ);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_Triangle_Color_RRJ);
            glBufferData(GL_ARRAY_BUFFER, sizeof(tri_Color), tri_Color, GL_STATIC_DRAW);
            glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindVertexArray(0);


        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glClearDepthf(1.0f);

        glEnable(GL_CULL_FACE);


        iViewPort_RRJ = 0;

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);


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

    vmath::mat4 translateMatrix_RRJ = vmath::mat4::identity();
    vmath::mat4 modelViewMatrix_RRJ = vmath::mat4::identity();
    vmath::mat4 modelViewProjectionMatrix_RRJ = vmath::mat4::identity();

    translateMatrix_RRJ = vmath::translate(0.0f, 0.0f, -3.0f);
    modelViewMatrix_RRJ = modelViewMatrix_RRJ * translateMatrix_RRJ;
    modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

    glUniformMatrix4fv(mvpUniform_RRJ, 1, false, modelViewProjectionMatrix_RRJ);


    if (iViewPort_RRJ == 0)
        glViewport(0, 0, (GLsizei)viewPortWidth_RRJ, (GLsizei)viewPortHeight_RRJ);
    else if (iViewPort_RRJ == 1)
        glViewport(0, 0, ((GLsizei)viewPortWidth_RRJ) / (GLsizei)2, ((GLsizei)viewPortHeight_RRJ) / (GLsizei)2);
    else if (iViewPort_RRJ == 2)
        glViewport((GLsizei)viewPortWidth_RRJ / 2, 0, (GLsizei)viewPortWidth_RRJ / 2, (GLsizei)viewPortHeight_RRJ / 2);
    else if (iViewPort_RRJ == 3)
        glViewport((GLsizei)viewPortWidth_RRJ / 2, (GLsizei)viewPortHeight_RRJ / 2, (GLsizei)viewPortWidth_RRJ / 2, (GLsizei)viewPortHeight_RRJ / 2);
    else if (iViewPort_RRJ == 4)
        glViewport(0, (GLsizei)viewPortHeight_RRJ / 2, (GLsizei)viewPortWidth_RRJ / 2, (GLsizei)viewPortHeight_RRJ / 2);
    else if (iViewPort_RRJ == 5)
        glViewport(0, 0, (GLsizei)viewPortWidth_RRJ / 2, (GLsizei)viewPortHeight_RRJ);
    else if (iViewPort_RRJ == 6)
        glViewport((GLsizei)viewPortWidth_RRJ / 2, 0, (GLsizei)viewPortWidth_RRJ / 2, (GLsizei)viewPortHeight_RRJ);
    else if (iViewPort_RRJ == 7)
        glViewport(0, (GLsizei)viewPortHeight_RRJ / 2, (GLsizei)viewPortWidth_RRJ, (GLsizei)viewPortHeight_RRJ / 2);
    else if (iViewPort_RRJ == 8)
        glViewport(0, 0, (GLsizei)viewPortWidth_RRJ, (GLsizei)viewPortHeight_RRJ / 2);
    else if (iViewPort_RRJ == 9)
        glViewport((GLsizei)viewPortWidth_RRJ / 4, (GLsizei)viewPortHeight_RRJ / 4, (GLsizei)viewPortWidth_RRJ / 2, (GLsizei)viewPortHeight_RRJ / 2);




    glBindVertexArray(vao_Triangle_RRJ);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    glUseProgram(0);


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


    viewPortWidth_RRJ = width_RRJ;
    viewPortHeight_RRJ = height_RRJ;

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

    iViewPort_RRJ++;
    if(iViewPort_RRJ > 9)
        iViewPort_RRJ = 0;

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

    if(vbo_Triangle_Color_RRJ){
        glDeleteBuffers(1, &vbo_Triangle_Color_RRJ);
        vbo_Triangle_Color_RRJ = 0;
    }

    if(vbo_Triangle_Position_RRJ){
        glDeleteBuffers(1, &vbo_Triangle_Position_RRJ);
        vbo_Triangle_Position_RRJ = 0;
    }

    if(vao_Triangle_RRJ){
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
