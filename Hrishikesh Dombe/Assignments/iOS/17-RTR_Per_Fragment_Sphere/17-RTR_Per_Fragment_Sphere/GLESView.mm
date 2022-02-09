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
#import "Sphere.h"

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
    
    vmath::mat4 gPerspectiveProjectionMatrix;
    
    GLfloat gAngle_Sphere;
    
    GLuint gVertexShaderObject;
    GLuint gFragmentShaderObject;
    GLuint gShaderProgramObject;
    
    GLuint gVao_Sphere;
    GLuint gVbo_Position, gVbo_Normal, gVbo_Elements;
    
    GLuint gModelMatrixUniform, gViewMatrixUniform, gProjectionMatrixUniform;
    GLuint gLKeyPressedUniform;
    
    GLuint gLaUniform, gLdUniform, gLsUniform;
    GLuint gLightPositionUniform;
    
    GLuint gKaUniform, gKdUniform, gKsUniform;
    GLuint gMaterialShininessUniform;
    
    GLfloat lightAmbient[4];
    GLfloat lightDiffuse[4];
    GLfloat lightSpecular[4];
    GLfloat lightPosition[4];
    
    GLfloat materialAmbient[4];
    GLfloat materialDiffuse[4];
    GLfloat materialSpecular[4];
    GLfloat materialShininess;
    
    bool gbLight;
    bool gbIsSingleTap;
    
    GLfloat sphere_vertices[1146];
    GLfloat sphere_normals[1146];
    GLfloat sphere_textures[764];
    unsigned short sphere_elements[2280];
    int gNumVertices, gNumElements;
    
    Sphere *mySphere;
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
        
        lightAmbient[0]=0.0f;
        lightAmbient[1]=0.0f;
        lightAmbient[2]=0.0f;
        lightAmbient[3]=1.0f;
        
        lightDiffuse[0]=1.0f;
        lightDiffuse[1]=1.0f;
        lightDiffuse[2]=1.0f;
        lightDiffuse[3]=1.0f;
        
        
        lightSpecular[0]=1.0f;
        lightSpecular[1]=1.0f;
        lightSpecular[2]=1.0f;
        lightSpecular[3]=1.0f;
        
        lightPosition[0]=100.0f;
        lightPosition[1]=100.0f;
        lightPosition[2]=100.0f;
        lightPosition[3]=1.0f;
        
        materialShininess = 50.0f;
        
        materialAmbient[0]=0.0f;
        materialAmbient[1]=0.0f;
        materialAmbient[2]=0.0f;
        materialAmbient[3]=1.0f;
        
        materialDiffuse[0]=1.0f;
        materialDiffuse[1]=1.0f;
        materialDiffuse[2]=1.0f;
        materialDiffuse[3]=1.0f;
        
        
        materialSpecular[0]=1.0f;
        materialSpecular[1]=1.0f;
        materialSpecular[2]=1.0f;
        materialSpecular[3]=1.0f;
        
        mySphere = [[Sphere alloc]init];
        
        [mySphere getSphereVertexDataWithPosition:sphere_vertices withNormals:sphere_normals withTexCoords:sphere_textures andElements:sphere_elements];
        gNumVertices = [mySphere getNumberOfSphereVertices];
        gNumElements = [mySphere getNumberOfSphereElements];
        
        gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
        
        const GLchar *vertexShaderSourceCode =
        "#version 300 es" \
        "\n" \
        "in vec4 vPosition;" \
        "in vec3 vNormal;" \
        "uniform mat4 u_model_matrix;" \
        "uniform mat4 u_view_matrix;" \
        "uniform mat4 u_projection_matrix;" \
        "uniform int u_lighting_enabled;" \
        "uniform vec4 u_light_position;" \
        "out vec3 transformed_normals;" \
        "out vec3 light_direction;" \
        "out vec3 viewer_vector;" \
        "void main(void)" \
        "{" \
        "if(u_lighting_enabled==1)" \
        "{" \
        "vec4 eye_coordinates = u_view_matrix*u_model_matrix*vPosition;" \
        "transformed_normals = mat3(u_view_matrix*u_model_matrix)*vNormal;" \
        "light_direction = vec3(u_light_position)-eye_coordinates.xyz;" \
        "viewer_vector = -eye_coordinates.xyz;" \
        "}" \
        "gl_Position = u_projection_matrix*u_view_matrix*u_model_matrix*vPosition;" \
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
        "in vec3 transformed_normals;" \
        "in vec3 light_direction;" \
        "in vec3 viewer_vector;" \
        "out vec4 FragColor;" \
        "uniform vec3 u_La;" \
        "uniform vec3 u_Ld;" \
        "uniform vec3 u_Ls;" \
        "uniform vec3 u_Ka;" \
        "uniform vec3 u_Kd;" \
        "uniform vec3 u_Ks;" \
        "uniform float u_material_shininess;" \
        "uniform highp int u_lighting_enabled;" \
        "void main(void)" \
        "{" \
        "vec3 phong_ads_color;" \
        "if(u_lighting_enabled == 1)" \
        "{" \
        "vec3 normalized_transformed_normals = normalize(transformed_normals);" \
        "vec3 normalized_light_direction = normalize(light_direction);" \
        "vec3 normalized_viewer_vector = normalize(viewer_vector);" \
        "vec3 ambient = u_La * u_Ka;" \
        "float tn_dot_ld = max(dot(normalized_transformed_normals,normalized_light_direction),0.0);" \
        "vec3 diffuse = u_Ld * u_Kd * tn_dot_ld;" \
        "vec3 reflection_vector = reflect(-normalized_light_direction,normalized_transformed_normals);" \
        "vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector,normalized_viewer_vector),0.0),u_material_shininess);" \
        "phong_ads_color = ambient + diffuse + specular;" \
        "}" \
        "else" \
        "{" \
        "phong_ads_color = vec3(1.0f,1.0f,1.0f);" \
        "}" \
        "FragColor = vec4(phong_ads_color,1.0);" \
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
        
        gModelMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_model_matrix");
        gViewMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_view_matrix");
        gProjectionMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_projection_matrix");
        
        gLKeyPressedUniform = glGetUniformLocation(gShaderProgramObject, "u_lighting_enabled");
        
        gLaUniform = glGetUniformLocation(gShaderProgramObject, "u_La");
        gLdUniform = glGetUniformLocation(gShaderProgramObject, "u_Ld");
        gLsUniform = glGetUniformLocation(gShaderProgramObject, "u_Ls");
        
        gLightPositionUniform = glGetUniformLocation(gShaderProgramObject, "u_light_position");
        
        gKaUniform = glGetUniformLocation(gShaderProgramObject, "u_Ka");
        gKdUniform = glGetUniformLocation(gShaderProgramObject, "u_Kd");
        gKsUniform = glGetUniformLocation(gShaderProgramObject, "u_Ks");
        
        gMaterialShininessUniform = glGetUniformLocation(gShaderProgramObject, "u_material_shininess");
        
        /*****************VAO For Cube*****************/
        glGenVertexArrays(1, &gVao_Sphere);
        glBindVertexArray(gVao_Sphere);
        
        /*****************Cube Position****************/
        glGenBuffers(1, &gVbo_Position);
        glBindBuffer(GL_ARRAY_BUFFER, gVbo_Position);
        glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_vertices), sphere_vertices, GL_STATIC_DRAW);
        
        glVertexAttribPointer(HAD_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        
        glEnableVertexAttribArray(HAD_ATTRIBUTE_POSITION);
        
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        /*****************Cube Color****************/
        glGenBuffers(1, &gVbo_Normal);
        glBindBuffer(GL_ARRAY_BUFFER, gVbo_Normal);
        glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_normals), sphere_normals, GL_STATIC_DRAW);
        
        glVertexAttribPointer(HAD_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        
        glEnableVertexAttribArray(HAD_ATTRIBUTE_NORMAL);
        
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        glGenBuffers(1, &gVbo_Elements);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(sphere_elements), sphere_elements, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        
        glBindVertexArray(0);
        
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        //glHint(GL_PERSPECTIVE_CORRECTION_HINT , GL_NICEST);
        //glEnable(GL_CULL_FACE);
        
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        
        gPerspectiveProjectionMatrix = vmath::mat4::identity();
        
        gbLight = NO;
        gbIsSingleTap = NO;
        
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
    
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    
    //Use Shader Program Object
    glUseProgram(gShaderProgramObject);
    
    if (gbLight == true)
    {
        glUniform1i(gLKeyPressedUniform, 1);
        
        glUniform3fv(gLaUniform, 1, lightAmbient);
        glUniform3fv(gLdUniform, 1, lightDiffuse);
        glUniform3fv(gLsUniform, 1, lightSpecular);
        glUniform4fv(gLightPositionUniform, 1, lightPosition);
        
        glUniform3fv(gKaUniform, 1, materialAmbient);
        glUniform3fv(gKdUniform, 1, materialDiffuse);
        glUniform3fv(gKsUniform, 1, materialSpecular);
        glUniform1f(gMaterialShininessUniform, materialShininess);
    }
    else
    {
        glUniform1i(gLKeyPressedUniform, 0);
    }
    
    vmath::mat4 modelMatrix = vmath::mat4::identity();
    vmath::mat4 viewMatrix = vmath::mat4::identity();
    
    modelMatrix = vmath::translate(0.0f, 0.0f, -2.0f);
    
    glUniformMatrix4fv(gModelMatrixUniform, 1, GL_FALSE, modelMatrix);
    
    glUniformMatrix4fv(gViewMatrixUniform, 1, GL_FALSE, viewMatrix);
    
    glUniformMatrix4fv(gProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);
    
    glBindVertexArray(gVao_Sphere);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
    glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
    
    glBindVertexArray(0);
    
    glUseProgram(0);
    
    glBindRenderbuffer(GL_RENDERBUFFER,colorRenderbuffer);
    [eaglContext presentRenderbuffer:GL_RENDERBUFFER];
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
    if (gVao_Sphere)
    {
        glDeleteVertexArrays(1, &gVao_Sphere);
        gVao_Sphere = 0;
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
    
    if (gVbo_Elements)
    {
        glDeleteBuffers(1, &gVbo_Elements);
        gVbo_Elements = 0;
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
