//
//  GLESView.m
//  26-SolarSystemWithMoon
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

//For Stack
struct STACK {
    vmath::mat4 ModelViewMatrix;
    struct STACK *next;
    struct STACK *prev;
};

typedef struct STACK ModelViewStack;
ModelViewStack *TopNode_RRJ = NULL;
int MaxTop_RRJ = 32;
int iTop_RRJ = -1;


//For Planet
int year_RRJ;
int day_RRJ;
int moon_RRJ;

//For Movement
#define YEAR_RRJ 1
#define DAY_RRJ 2
#define MOON_RRJ 3
int iWhichMovement_RRJ = YEAR_RRJ;


//For Rotation
#define CLKWISE 4
#define ANTICLK 5
int iWhichRotation_RRJ = CLKWISE;


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
GLfloat sphere_Angle;

//For Uniform
GLuint mvUniform_RRJ;
GLuint projectionUniform_RRJ;

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
    "in vec3 vColor;" \
    "out vec3 outColor;" \
    "uniform mat4 u_mv_matrix;" \
    "uniform mat4 u_projection_matrix;" \
    "void main(void) {" \
        "outColor = vColor;" \
        "gl_Position = u_projection_matrix * u_mv_matrix * vPosition;" \
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
           "in vec3 outColor;" \
       "out vec4 FragColor;" \
        "void main(void) {" \
            "FragColor = vec4(outColor, 1.0);" \
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


        mvUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_mv_matrix");
    projectionUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_projection_matrix");


    /********** Sphere Position, vao, vbo **********/
    GLfloat sphere_Position_RRJ[STACKS * SLICES * 3];
    GLfloat sphere_Normal_RRJ[STACKS * SLICES * 3];
    GLshort sphere_Index_RRJ[(STACKS) * (SLICES) * 6];


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
            //printf("%f\t%f\n",latitute, longitute);
            
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

-(void)drawView:(id)sender{

    [EAGLContext setCurrentContext:eaglContext_RRJ];

    glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebuffer_RRJ);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    glUseProgram(shaderProgramObject_RRJ);

        vmath::mat4 modelMatrix_RRJ;
        vmath::mat4 viewMatrix_RRJ;
        vmath::mat4 modelViewMatrix_RRJ;

        /********** Sphere **********/
        modelMatrix_RRJ = vmath::mat4::identity();
        viewMatrix_RRJ = vmath::mat4::identity();
        modelViewMatrix_RRJ = vmath::mat4::identity();


        viewMatrix_RRJ =  vmath::lookat(vmath::vec3(0.0f, 0.0f, 6.0f),
                            vmath::vec3(0.0f, 0.0f, 0.0f),
                            vmath::vec3(0.0f, 1.0f, 0.0));

        modelViewMatrix_RRJ = viewMatrix_RRJ * modelMatrix_RRJ;


        //Sun
        [self my_glPushMatrix: modelViewMatrix_RRJ];
        

        //modelViewMatrix_RRJ = modelViewMatrix_RRJ * vmath::rotate(90.0f, 1.0f, 0.0f, 0.0f);
        glUniformMatrix4fv(mvUniform_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
        glUniformMatrix4fv(projectionUniform_RRJ, 1, GL_FALSE, perspectiveProjectionMatrix_RRJ);

        glBindVertexArray(vao_Sphere_RRJ);
        glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 0.0f);
            glDrawElements(GL_TRIANGLES, (STACKS - 1) * (SLICES - 1) * 6, GL_UNSIGNED_SHORT, 0);
            glBindVertexArray(0);

        
        //Earth
        modelViewMatrix_RRJ = [self my_glPopMatrix];
        
        
        modelViewMatrix_RRJ = modelViewMatrix_RRJ * vmath::rotate((GLfloat)year_RRJ, 0.0f, 1.0f, 0.0f);
        modelViewMatrix_RRJ = modelViewMatrix_RRJ * vmath::translate(2.50f, 0.0f, 0.0f);

        [self my_glPushMatrix: modelViewMatrix_RRJ];

        modelViewMatrix_RRJ = modelViewMatrix_RRJ * vmath::scale(0.4f, 0.4f, 0.4f);
        modelViewMatrix_RRJ = modelViewMatrix_RRJ * vmath::rotate(90.0f, 1.0f, 0.0f, 0.0f);
        modelViewMatrix_RRJ = modelViewMatrix_RRJ * vmath::rotate((GLfloat)day_RRJ, 0.0f, 0.0f, 1.0f);
        
        glUniformMatrix4fv(mvUniform_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
        glUniformMatrix4fv(projectionUniform_RRJ, 1, GL_FALSE, perspectiveProjectionMatrix_RRJ);

        glBindVertexArray(vao_Sphere_RRJ);
        glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 0.0f, 0.0f, 0.50f);
            glDrawElements(GL_LINE_STRIP, (STACKS - 1) * (SLICES - 1) * 6, GL_UNSIGNED_SHORT, 0);
            glBindVertexArray(0);


            //Moon
            modelViewMatrix_RRJ = [self my_glPopMatrix];

            modelViewMatrix_RRJ = modelViewMatrix_RRJ * vmath::rotate((GLfloat)moon_RRJ, 0.0f, 1.0f, 0.0f);
        modelViewMatrix_RRJ = modelViewMatrix_RRJ * vmath::translate(1.0f, 0.0f, 0.0f);

        [self my_glPushMatrix: modelViewMatrix_RRJ];

        modelViewMatrix_RRJ = modelViewMatrix_RRJ * vmath::scale(0.1f, 0.1f, 0.1f);


        glUniformMatrix4fv(mvUniform_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
        glUniformMatrix4fv(projectionUniform_RRJ, 1, GL_FALSE, perspectiveProjectionMatrix_RRJ);

        glBindVertexArray(vao_Sphere_RRJ);
        glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 0.50f, 0.50f, 0.50f);
            glDrawElements(GL_TRIANGLES, (STACKS - 1) * (SLICES - 1) * 6, GL_UNSIGNED_SHORT, 0);
        glBindVertexArray(0);

        [self my_glPopMatrix];


    glUseProgram(0);


    glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer_RRJ);
    [eaglContext_RRJ presentRenderbuffer: GL_RENDERBUFFER];

}




-(void) my_glPushMatrix:(vmath::mat4)matrix{

    void uninitialize(void);

    ModelViewStack *temp_RRJ = (ModelViewStack*)malloc(sizeof(ModelViewStack));
    if (temp_RRJ == NULL) {
        printf("ERROR: Malloc Failed!!\n");
        [self release];
        
    }
    else {

        temp_RRJ->ModelViewMatrix = matrix;
        temp_RRJ->next = NULL;

        if (TopNode_RRJ == NULL) {
            TopNode_RRJ = temp_RRJ;
            TopNode_RRJ->prev = NULL;
            printf("Node Added!!\n");
        }
        else {
            TopNode_RRJ->next = temp_RRJ;
            temp_RRJ->prev = TopNode_RRJ;
            TopNode_RRJ = temp_RRJ;
            printf("Node Added!!\n");
        }
    }

    if (iTop_RRJ > MaxTop_RRJ) {
        printf("ERROR: Stack Overflow!!\n");
        [self release];
    }
    

    
}

-(vmath::mat4) my_glPopMatrix{
    

    ModelViewStack *temp_RRJ = TopNode_RRJ;
    vmath::mat4 matrix_RRJ;
    if (temp_RRJ->prev != NULL) {
        TopNode_RRJ = temp_RRJ->prev;
        temp_RRJ->next = NULL;
        temp_RRJ->prev = NULL;
        matrix_RRJ = temp_RRJ->ModelViewMatrix;
        printf("Node Delete!!\n");
        free(temp_RRJ);
    }
    else {
        temp_RRJ->next = NULL;
        temp_RRJ->prev = NULL;
        matrix_RRJ = temp_RRJ->ModelViewMatrix;
        printf("Node Delete!!\n");
        free(temp_RRJ);
        TopNode_RRJ = NULL;
    }
    return(matrix_RRJ);
    
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


    if(iWhichMovement_RRJ == YEAR_RRJ){
        if(iWhichRotation_RRJ == CLKWISE)
            year_RRJ = (year_RRJ + 20) % 360;
        else
            year_RRJ = (year_RRJ - 20) % 360;
    }
    else if (iWhichMovement_RRJ == DAY_RRJ){
        if(iWhichRotation_RRJ == CLKWISE)
            day_RRJ = (day_RRJ + 20) % 360;
        else
            day_RRJ = (day_RRJ - 20) % 360;
    }
    else if(iWhichMovement_RRJ == MOON_RRJ){
        if(iWhichRotation_RRJ == CLKWISE)
            moon_RRJ = (moon_RRJ + 20) % 360;
        else
            moon_RRJ = (moon_RRJ - 20) % 360;
    }


}

-(void)onDoubleTap:(UITapGestureRecognizer*)gr{

    if(iWhichMovement_RRJ == YEAR_RRJ)
        iWhichMovement_RRJ = DAY_RRJ;
    else if(iWhichMovement_RRJ == DAY_RRJ)
        iWhichMovement_RRJ = MOON_RRJ;
    else
        iWhichMovement_RRJ = YEAR_RRJ;


}

-(void)onLongPress:(UILongPressGestureRecognizer*)gr{

    if(iWhichRotation_RRJ == CLKWISE)
        iWhichRotation_RRJ = ANTICLK;
    else
        iWhichRotation_RRJ = CLKWISE;

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
