//
//  GLESView.m
//  02-Blue_Window
//
//  Created by Samarth Mabrukar on 28/06/18.
//  Copyright © 2018 Hrishikiesh Dombe. All rights reserved.
//

#import <OpenGLES/ES3/gl.h>
#import <OpenGLES/ES3/glext.h>
#import "vmath.h"
#import "GLESView.h"

enum
{
    HAD_ATTRIBUTE_POSITION = 0,
    HAD_ATTRIBUTE_COLOR,
    HAD_ATTRIBUTE_NORMAL,
    HAD_ATTRIBUTE_TEXTURE0,
};

@implementation GLESView
{
    EAGLContext *eaglContext;
    
    GLuint defaultFramebuffer;
    GLuint colorRenderbuffer;
    GLuint depthRenderbuffer;
    
    id displayLink;
    NSInteger animationFrameInterval;
    BOOL isAnimating;
    
    GLuint gVertexShaderObject;
    GLuint gFragmentShaderObject;
    GLuint gShaderProgramObject;
    
    GLuint gVao_Cube;
    GLuint gVbo_Position,gVbo_Normal;
    GLuint gModelViewMatrixUniform, gProjectionMatrixUniform;
    GLuint gLdUniform, gKdUniform, gLightPositionUniform;
    GLuint gLKeyPressedUniform;
    
    vmath::mat4 gPerspectiveProjectionMatrix;
    
    GLfloat gAngle_Cube;
    
    bool gbAnimate;
    bool gbLight;
    bool gbIsSingleTap;
    bool gbIsDoubleTap;
}
/*
// Only override drawRect: if you perform custom drawing.
// An empty implementation adversely affects performance during animation.
- (void)drawRect:(CGRect)rect {
    // Drawing code
}
*/
-(id)initWithFrame:(CGRect)frame;
{
    self=[super initWithFrame:frame];
    
    if(self)
    {
        CAEAGLLayer *eaglLayer=(CAEAGLLayer *)super.layer;
        
        eaglLayer.opaque=YES;
        eaglLayer.drawableProperties=[NSDictionary dictionaryWithObjectsAndKeys:[NSNumber numberWithBool:FALSE], kEAGLDrawablePropertyRetainedBacking,kEAGLColorFormatRGBA8,kEAGLDrawablePropertyColorFormat, nil];
        
        eaglContext=[[EAGLContext alloc]initWithAPI:kEAGLRenderingAPIOpenGLES3];
        if(eaglContext==nil)
        {
            [self release];
            return(nil);
        }
        
        [EAGLContext setCurrentContext:eaglContext];
        
        glGenFramebuffers(1,&defaultFramebuffer);
        glGenRenderbuffers(1, &colorRenderbuffer);
        glBindFramebuffer(GL_FRAMEBUFFER,defaultFramebuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer);
        
        [eaglContext renderbufferStorage:GL_RENDERBUFFER fromDrawable:eaglLayer];
        
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, colorRenderbuffer);
        
        GLint backingWidth;
        GLint backingHeight;
        
        glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &backingWidth);
        glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &backingHeight);
        
        glGenRenderbuffers(1, &depthRenderbuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, backingWidth, backingHeight);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer);
        
        if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)
        {
            printf("Failed To Create Complete FrameBuffer Object %x\n",glCheckFramebufferStatus(GL_FRAMEBUFFER));
            glDeleteFramebuffers(1, &defaultFramebuffer);
            glDeleteRenderbuffers(1, &colorRenderbuffer);
            glDeleteRenderbuffers(1, &depthRenderbuffer);
            
            return(nil);
        }
        
        printf("Renderer : %s | Version : %s | GLSL Version : %s\n",glGetString(GL_RENDERER),glGetString(GL_VERSION),glGetString(GL_SHADING_LANGUAGE_VERSION));
        
        isAnimating=NO;
        animationFrameInterval=60;
        
        gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
        
        const GLchar *vertexShaderSourceCode =
        "#version 300 es" \
        "\n" \
        "in vec4 vPosition;" \
        "in vec3 vNormal;" \
        "uniform mat4 u_model_view_matrix;" \
        "uniform mat4 u_projection_matrix;" \
        "uniform int u_LKeyPressed;" \
        "uniform vec3 u_Ld;" \
        "uniform vec3 u_Kd;" \
        "uniform vec4 u_light_position;" \
        "out vec3 diffuse_light;" \
        "void main(void)" \
        "{" \
        "if (u_LKeyPressed == 1)" \
        "{" \
        "vec4 eyeCoordinates = u_model_view_matrix * vPosition;" \
        "vec3 tnorm = normalize(mat3(u_model_view_matrix) * vNormal);" \
        "vec3 s = normalize(vec3(u_light_position - eyeCoordinates));" \
        "diffuse_light = u_Ld * u_Kd * max(dot(s,tnorm),0.0);" \
        "}" \
        "gl_Position = u_projection_matrix * u_model_view_matrix * vPosition;" \
        "}";
        
        glShaderSource(gVertexShaderObject, 1, (const GLchar **)&vertexShaderSourceCode, NULL);
        
        glCompileShader(gVertexShaderObject);
        GLint iInfoLogLength = 0;
        GLint iShaderCompiledStatus = 0;
        char *szInfoLog = NULL;
        
        glGetShaderiv(gVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
        if (iShaderCompiledStatus == GL_FALSE)
        {
            glGetShaderiv(gVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
            if (iInfoLogLength > 0)
            {
                szInfoLog = (char *)malloc(iInfoLogLength);
                if (szInfoLog != NULL)
                {
                    GLsizei written;
                    glGetShaderInfoLog(gVertexShaderObject, iInfoLogLength, &written, szInfoLog);
                    printf("Vertex Shader Compilation Log : %s\n", szInfoLog);
                    free(szInfoLog);
                    [self dealloc];
                    exit(0);
                }
            }
        }
        
        //Fragment Shader
        gFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);
        
        const GLchar *fragmentShaderSourceCode =
        "#version 300 es"\
        "\n"\
        "precision highp float;" \
        "in vec3 diffuse_light;" \
        "out vec4 FragColor;" \
        "uniform highp int u_LKeyPressed;" \
        "void main(void)"\
        "{"\
        "vec4 color;" \
        "if(u_LKeyPressed == 1)" \
        "{" \
        "color = vec4(diffuse_light,1.0);" \
        "}" \
        "else" \
        "{" \
        "color = vec4(1.0,1.0,1.0,1.0);" \
        "}" \
        "FragColor = color;"\
        "}";
        
        glShaderSource(gFragmentShaderObject, 1, (const GLchar **)&fragmentShaderSourceCode, NULL);
        
        glCompileShader(gFragmentShaderObject);
        
        glGetShaderiv(gFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
        if (iShaderCompiledStatus == GL_FALSE)
        {
            glGetShaderiv(gFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
            if (iInfoLogLength > 0)
            {
                szInfoLog = (char*)malloc(iInfoLogLength);
                if (szInfoLog != NULL)
                {
                    GLsizei written;
                    glGetShaderInfoLog(gFragmentShaderObject, iInfoLogLength, &written, szInfoLog);
                    printf( "Fragment Shader Compilation Log : %s\n", szInfoLog);
                    free(szInfoLog);
                    [self dealloc];
                    exit(0);
                }
            }
        }
        
        //Shader Program
        gShaderProgramObject = glCreateProgram();
        
        glAttachShader(gShaderProgramObject, gVertexShaderObject);
        
        glAttachShader(gShaderProgramObject, gFragmentShaderObject);
        
        glBindAttribLocation(gShaderProgramObject, HAD_ATTRIBUTE_POSITION, "vPosition");
        
        glBindAttribLocation(gShaderProgramObject, HAD_ATTRIBUTE_NORMAL, "vNormal");
        
        glLinkProgram(gShaderProgramObject);
        
        GLint iShaderProgramLinkStatus = 0;
        
        glGetProgramiv(gShaderProgramObject, GL_LINK_STATUS, &iShaderProgramLinkStatus);
        if (iShaderProgramLinkStatus == GL_FALSE)
        {
            glGetProgramiv(gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
            if (iInfoLogLength > 0)
            {
                szInfoLog = (char *)malloc(iInfoLogLength);
                if (szInfoLog != NULL)
                {
                    GLsizei written;
                    glGetProgramInfoLog(gShaderProgramObject, iInfoLogLength, &written, szInfoLog);
                    printf("Shader Program Link Log : %s\n", szInfoLog);
                    free(szInfoLog);
                    [self dealloc];
                    exit(0);
                }
            }
        }
        
        gModelViewMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_model_view_matrix");
        gProjectionMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_projection_matrix");
        gLKeyPressedUniform = glGetUniformLocation(gShaderProgramObject, "u_LKeyPressed");
        gLdUniform = glGetUniformLocation(gShaderProgramObject, "u_Ld");
        gKdUniform = glGetUniformLocation(gShaderProgramObject, "u_Kd");
        gLightPositionUniform = glGetUniformLocation(gShaderProgramObject, "u_light_position");
        
        const GLfloat cubeVertices[] =
        {
            1.0f,1.0f,1.0f,
            -1.0f,1.0f,1.0f,
            -1.0f,-1.0f,1.0f,
            1.0f,-1.0f,1.0f,
            
            1.0f,1.0f,-1.0f,
            1.0f,1.0f,1.0f,
            1.0f,-1.0f,1.0f,
            1.0f,-1.0f,-1.0f,
            
            -1.0f,1.0f,-1.0f,
            1.0f,1.0f,-1.0f,
            1.0f,-1.0f,-1.0f,
            -1.0f,-1.0f,-1.0f,
            
            -1.0f,1.0f,1.0f,
            -1.0f,1.0f,-1.0f,
            -1.0f,-1.0f,-1.0f,
            -1.0f,-1.0f,1.0f,
            
            1.0f,1.0f,-1.0f,
            -1.0f,1.0f,-1.0f,
            -1.0f,1.0f,1.0f,
            1.0f,1.0f,1.0f,
            
            1.0f,-1.0f,1.0f,
            -1.0f,-1.0f,1.0f,
            -1.0f,-1.0f,-1.0f,
            1.0f,-1.0f,-1.0f
        };
        
        const GLfloat cubeNormal[] =
        {
            0.0f,0.0f,1.0f,
            0.0f,0.0f,1.0f,
            0.0f,0.0f,1.0f,
            0.0f,0.0f,1.0f,
            
            1.0f,0.0f,0.0f,
            1.0f,0.0f,0.0f,
            1.0f,0.0f,0.0f,
            1.0f,0.0f,0.0f,
            
            0.0f,0.0f,-1.0f,
            0.0f,0.0f,-1.0f,
            0.0f,0.0f,-1.0f,
            0.0f,0.0f,-1.0f,
            
            -1.0f,0.0f,0.0f,
            -1.0f,0.0f,0.0f,
            -1.0f,0.0f,0.0f,
            -1.0f,0.0f,0.0f,
            
            0.0f,1.0f,0.0f,
            0.0f,1.0f,0.0f,
            0.0f,1.0f,0.0f,
            0.0f,1.0f,0.0f,
            
            0.0f,-1.0f,0.0f,
            0.0f,-1.0f,0.0f,
            0.0f,-1.0f,0.0f,
            0.0f,-1.0f,0.0f
        };
        
        /*****************VAO For Cube*****************/
        glGenVertexArrays(1, &gVao_Cube);
        glBindVertexArray(gVao_Cube);
        
        /*****************Cube Position****************/
        glGenBuffers(1, &gVbo_Position);
        glBindBuffer(GL_ARRAY_BUFFER, gVbo_Position);
        glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
        
        glVertexAttribPointer(HAD_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        
        glEnableVertexAttribArray(HAD_ATTRIBUTE_POSITION);
        
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        /*****************Cube Color****************/
        glGenBuffers(1, &gVbo_Normal);
        glBindBuffer(GL_ARRAY_BUFFER, gVbo_Normal);
        glBufferData(GL_ARRAY_BUFFER, sizeof(cubeNormal), cubeNormal, GL_STATIC_DRAW);
        
        glVertexAttribPointer(HAD_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        
        glEnableVertexAttribArray(HAD_ATTRIBUTE_NORMAL);
        
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        glBindVertexArray(0);
        
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        //glHint(GL_PERSPECTIVE_CORRECTION_HINT , GL_NICEST);
        //glEnable(GL_CULL_FACE);
        
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        
        gPerspectiveProjectionMatrix = vmath::mat4::identity();
        
        gbAnimate = NO;
        gbLight = NO;
        gbIsSingleTap = NO;
        gbIsDoubleTap = NO;
        
        //Tap Gesture Code
        UITapGestureRecognizer *singleTapGestureRecognizer=[[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(onSingleTap:)];
        [singleTapGestureRecognizer setNumberOfTapsRequired:1];
        [singleTapGestureRecognizer setNumberOfTouchesRequired:1]; // touch of 1 finger
        [singleTapGestureRecognizer setDelegate:self];
        [self addGestureRecognizer:singleTapGestureRecognizer];
        
        UITapGestureRecognizer *doubleTapGestureRecognizer=[[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(onDoubleTap:)];
        [doubleTapGestureRecognizer setNumberOfTapsRequired:2];
        [doubleTapGestureRecognizer setNumberOfTouchesRequired:1]; // touch of 1 finger
        [doubleTapGestureRecognizer setDelegate:self];
        [self addGestureRecognizer:doubleTapGestureRecognizer];
        
        //This will allow to differentiate between single tap and double tap
        [singleTapGestureRecognizer requireGestureRecognizerToFail:doubleTapGestureRecognizer];
        
        //Swipe
        UISwipeGestureRecognizer *swipeGestureRecognizer = [[UISwipeGestureRecognizer alloc]initWithTarget:self action:@selector(onSwipe:)];
        [self addGestureRecognizer:swipeGestureRecognizer];
        
        //Long Press
        UILongPressGestureRecognizer *longPressGestureRecognizer = [[UILongPressGestureRecognizer alloc]initWithTarget:self action:@selector(onLongPress:)];
        [self addGestureRecognizer:longPressGestureRecognizer];
    }
    return(self);
}

+(Class)layerClass
{
    return([CAEAGLLayer class]);
}

-(void)drawView:(id)sender
{
    [EAGLContext setCurrentContext:eaglContext];
    
    glBindFramebuffer(GL_FRAMEBUFFER,defaultFramebuffer);
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glUseProgram(gShaderProgramObject);
    
    if (gbLight == YES)
    {
        glUniform1i(gLKeyPressedUniform, 1);
        
        glUniform3f(gLdUniform, 1.0f, 1.0f, 1.0f);
        glUniform3f(gKdUniform, 0.5f, 0.5f, 0.5f);
        
        float lightPosition[] = { 0.0f,0.0f,2.0f,1.0f };
        glUniform4fv(gLightPositionUniform, 1, (GLfloat *)lightPosition);
    }
    else
    {
        glUniform1i(gLKeyPressedUniform, 0);
    }
    
    vmath::mat4 modelMatrix = vmath::mat4::identity();
    vmath::mat4 modelViewMatrix = vmath::mat4::identity();
    vmath::mat4 scaleMatrix = vmath::mat4::identity();
    vmath::mat4 rotationMatrix = vmath::mat4::identity();
    
    modelMatrix = vmath::translate(0.0f, 0.0f, -6.0f);
    
    scaleMatrix = vmath::scale(0.75f, 0.75f, 0.75f);
    modelViewMatrix = modelMatrix * scaleMatrix;
    
    rotationMatrix = vmath::rotate(gAngle_Cube, gAngle_Cube, gAngle_Cube);
    modelViewMatrix = modelMatrix*rotationMatrix;
    
    glUniformMatrix4fv(gModelViewMatrixUniform, 1, GL_FALSE, modelViewMatrix);
    
    glUniformMatrix4fv(gProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);
    
    glBindVertexArray(gVao_Cube);
    
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 20, 4);
    
    glBindVertexArray(0);
    
    glUseProgram(0);

    
    glBindRenderbuffer(GL_RENDERBUFFER,colorRenderbuffer);
    [eaglContext presentRenderbuffer:GL_RENDERBUFFER];
    
    if(gbAnimate == YES)
    {
        gAngle_Cube = gAngle_Cube + 0.8f;
        if(gAngle_Cube >= 360.0f)
            gAngle_Cube = gAngle_Cube - 360.0f;
    }
}

-(void)layoutSubviews
{
    GLint width,height;
    
    glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer);
    [eaglContext renderbufferStorage:GL_RENDERBUFFER fromDrawable:(CAEAGLLayer*)self.layer];
    glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &width);
    glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &height);

    glGenRenderbuffers(1, &depthRenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer);
    
    glViewport(0,0,(GLsizei)width,(GLsizei)height);
    
    GLfloat fwidth = (GLfloat)width;
    GLfloat fheight = (GLfloat)height;
    
    gPerspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)fwidth / (GLfloat)fheight, 0.1f, 100.0f);
    
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)
    {
        printf("Failed To Create Complete Framebuffer Object %x", glCheckFramebufferStatus(GL_FRAMEBUFFER));
    }
    
    [self drawView:nil];
}

-(void)startAnimation
{
    if(!isAnimating)
    {
        displayLink=[NSClassFromString(@"CADisplayLink")displayLinkWithTarget:self selector:@selector(drawView:)];
        [displayLink setPreferredFramesPerSecond:animationFrameInterval];
        [displayLink addToRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
        
        isAnimating=YES;
    }
}

-(void)stopAnimation
{
    if(isAnimating)
    {
        [displayLink invalidate];
        displayLink=nil;
        
        isAnimating=NO;
    }
}

-(BOOL)acceptsFirstResponder
{
    return(YES);
}

-(void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event
{
    
}

-(void)onSingleTap:(UITapGestureRecognizer *)gr
{
    if(gbIsSingleTap == NO)
    {
        gbIsSingleTap = YES;
        gbLight = YES;
    }
    else
    {
        gbIsSingleTap = NO;
        gbLight = NO;
    }
}

-(void)onDoubleTap:(UITapGestureRecognizer *)gr
{
    if(gbIsDoubleTap == NO)
    {
        gbIsDoubleTap = YES;
        gbAnimate = YES;
    }
    else
    {
        gbIsDoubleTap = NO;
        gbAnimate = NO;
    }
}

-(void)onSwipe:(UISwipeGestureRecognizer *)gr
{
    [self release];
    exit(0);
}

-(void)onLongPress:(UILongPressGestureRecognizer *)gr
{
    
}

- (void)dealloc
{
    if (gVao_Cube)
    {
        glDeleteVertexArrays(1, &gVao_Cube);
        gVao_Cube = 0;
    }
    
    if (gVbo_Position)
    {
        glDeleteBuffers(1, &gVbo_Position);
        gVbo_Position = 0;
    }
    
    if (gVbo_Normal)
    {
        glDeleteBuffers(1, &gVbo_Normal);
        gVbo_Normal = 0;
    }
    
    //Detach Shader
    glDetachShader(gShaderProgramObject, gVertexShaderObject);
    glDetachShader(gShaderProgramObject, gFragmentShaderObject);
    
    //Delete Shader
    glDeleteShader(gVertexShaderObject);
    gVertexShaderObject = 0;
    
    glDeleteShader(gFragmentShaderObject);
    gFragmentShaderObject = 0;
    
    //Delete Program
    glDeleteProgram(gShaderProgramObject);
    gShaderProgramObject = 0;
    
    //Stray call to glUseProgram(0)
    glUseProgram(0);

    if(depthRenderbuffer)
    {
        glDeleteRenderbuffers(1, &depthRenderbuffer);
        depthRenderbuffer=0;
    }
    
    if(colorRenderbuffer)
    {
        glDeleteRenderbuffers(1, &colorRenderbuffer);
        colorRenderbuffer=0;
    }
    
    if(defaultFramebuffer)
    {
        glDeleteFramebuffers(1, &defaultFramebuffer);
        defaultFramebuffer=0;
    }
    
    if([EAGLContext currentContext]==eaglContext)
    {
        [EAGLContext setCurrentContext:nil];
    }
    [eaglContext release];
    eaglContext=nil;
    
    [super dealloc];
}

@end
