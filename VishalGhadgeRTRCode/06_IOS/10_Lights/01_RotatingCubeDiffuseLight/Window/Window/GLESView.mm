//
//  GLESView.m
//  Rotating Cube Diffuse Light
//
//  Created by Vishal on 7/10/18.
//

#import <OpenGLES/ES3/gl.h>
#import <OpenGLES/ES3/glext.h>
#import "vmath.h"
#import "GLESView.h"

enum
{
	RTR_ATTRIBUTE_POSITION = 0,
	RTR_ATTRIBUTE_COLOR,
	RTR_ATTRIBUTE_NORMAL,
	RTR_ATTRIBUTE_TEXTURE0
};



GLfloat g_glfAngle = 0.0f;
bool g_bAnimate = false;
bool g_bLight = false;

@implementation GLESView
{
	EAGLContext *eaglContext;
	
	GLuint defaultFrameBuffer;
	GLuint colorRenderBuffer;
	GLuint depthRenderBuffer;
	
	id displayLink;
	NSInteger animationFrameInterval;
	BOOL isAnimation;

	GLuint g_gluiShaderObjectVertex;
	GLuint g_gluiShaderObjectFragment;
	GLuint g_gluiShaderObjectProgram;

	GLuint g_gluiVAOCube;
	GLuint g_gluiVBOPosition;
	GLuint g_gluiVBONormal;

	//	Uniforms.
	GLuint g_gluiModelViewMat4Uniform;
	GLuint g_gluiProjectionMat4Uniform;
	GLuint g_gluiLightPositionVec4Uniform;
	GLuint g_gluiLdVec3Uniform;
	GLuint g_gluiKdVec3Uniform;
	GLuint g_gluiKeyPressedUniform;


	vmath::mat4 g_matPerspectiveProjection;
}


- (id)initWithFrame:(CGRect)frame	//	Flow 2
{
	self=[super initWithFrame:frame];
	if (self)
	{
		//	Initialization
		CAEAGLLayer *eaglLayer = (CAEAGLLayer *)super.layer;
		
		eaglLayer.opaque = YES;
		eaglLayer.drawableProperties = [NSDictionary dictionaryWithObjectsAndKeys:[NSNumber numberWithBool:FALSE],kEAGLDrawablePropertyRetainedBacking, kEAGLColorFormatRGBA8, kEAGLDrawablePropertyColorFormat, nil];
		eaglContext = [[EAGLContext alloc]initWithAPI:kEAGLRenderingAPIOpenGLES3];
		if (nil == eaglContext)
		{
			[self release];
			return nil;
		}
		
		[EAGLContext setCurrentContext:eaglContext];	//	Class method
		
		glGenFramebuffers(1, &defaultFrameBuffer);
		glGenRenderbuffers(1, &colorRenderBuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, defaultFrameBuffer);
		glBindRenderbuffer(GL_RENDERBUFFER, colorRenderBuffer);
		
		[eaglContext renderbufferStorage:GL_RENDERBUFFER fromDrawable:eaglLayer];
		
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, colorRenderBuffer);
		
		GLint backingWidth;
		GLint backingHeight;
		glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &backingWidth);
		glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &backingHeight);
		
		glGenRenderbuffers(1, &depthRenderBuffer);
		glBindRenderbuffer(GL_RENDERBUFFER, depthRenderBuffer);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, backingWidth, backingHeight);	//	For IOS 16
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderBuffer);
		
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		{
			printf("Failed to create complete frame buffer object %x\n", glCheckFramebufferStatus(GL_FRAMEBUFFER));
			glDeleteFramebuffers(1, &defaultFrameBuffer);
			glDeleteRenderbuffers(1, &colorRenderBuffer);
			glDeleteRenderbuffers(1,&depthRenderBuffer);
			
			return nil;
		}
		
		printf("Renderer : %s | GL version: %s | GLSL version : %s \n ", glGetString(GL_RENDERER), glGetString(GL_VERSION),glGetString(GL_SHADING_LANGUAGE_VERSION));
		
		//	Hard coded initialization
		isAnimation = NO;
		animationFrameInterval = 60;	//	Default since ios 8.2
	
		////////////////////////////////////////////////////////////////////
		//+	Shader code

		////////////////////////////////////////////////////////////////////
		//+	Vertex shader.

		//	Create shader.
		g_gluiShaderObjectVertex = glCreateShader(GL_VERTEX_SHADER);

		//	Provide source code.
		const GLchar *szVertexShaderSourceCode =
			"#version 300 es"							\
			"\n"										\
			"in vec4 vPosition;"						\
			"in vec3 vNormal;"							\
			"uniform mat4 u_model_view_matrix;"	\
			"uniform mat4 u_projection_matrix;"	\
			"uniform mediump int u_L_key_pressed;"			\
			"uniform vec3 u_LD;	"				\
			"uniform vec3 u_KD;"					\
			"uniform vec4 u_light_position;"		\
			"out vec3 out_diffuse_light;"			\
			"void main(void)"							\
			"{"											\
				"if (1 == u_L_key_pressed)"										\
				"{"											\
					"vec4 eyeCoordinates = u_model_view_matrix * vPosition;"											\
					"vec3 tnorm = normalize(mat3(u_model_view_matrix) * vNormal);"											\
					"vec3 srcVec = normalize(vec3(u_light_position - eyeCoordinates));"											\
					"out_diffuse_light = u_LD * u_KD * max(dot(srcVec, tnorm), 0.0);"											\
				"}"											\
			"gl_Position = u_projection_matrix * u_model_view_matrix * vPosition;"	\
			"}";

		glShaderSource(g_gluiShaderObjectVertex, 1, &szVertexShaderSourceCode, NULL);

		//	Compile shader.
		glCompileShader(g_gluiShaderObjectVertex);

		GLint gliCompileStatus;
		GLint gliInfoLogLength;
		char *pszInfoLog = NULL;
		GLsizei glsiWritten;
		glGetShaderiv(g_gluiShaderObjectVertex, GL_COMPILE_STATUS, &gliCompileStatus);
		if (GL_FALSE == gliCompileStatus)
		{
			glGetShaderiv(g_gluiShaderObjectVertex, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
			if (gliInfoLogLength <= 0)
			{
				printf("GL_INFO_LOG_LENGTH is less than 0.");
				[self release];
			}

			pszInfoLog = (char*)malloc(gliInfoLogLength);
			if (NULL == pszInfoLog)
			{
				printf("malloc failed.");
				[self release];
			}

			glGetShaderInfoLog(g_gluiShaderObjectVertex, gliInfoLogLength, &glsiWritten, pszInfoLog);

			printf("Vertex shader compilation log : %s \n", pszInfoLog);
			free(pszInfoLog);
			[self release];
		}
		//-	Vertex shader.
		////////////////////////////////////////////////////////////////////

		////////////////////////////////////////////////////////////////////
		//+	Fragment shader.

		//	Create shader.
		g_gluiShaderObjectFragment = glCreateShader(GL_FRAGMENT_SHADER);

		//	Provide source code.
		const GLchar *szFragmentShaderSourceCode =
			"#version 300 es"							\
			"\n"										\
			"precision highp float;"					\
			"in vec3 out_diffuse_light;"				\
			"uniform int u_L_key_pressed;"						\
			"out vec4 vFragColor;"						\
			"void main(void)"							\
			"{"											\
				"vec4 color;"											\
				"if (1 == u_L_key_pressed)"											\
				"{"											\
					"color = vec4(out_diffuse_light, 1.0);"											\
				"}"											\
				"else"											\
				"{"											\
					"color = vec4(1.0, 1.0, 1.0, 1.0);"											\
				"}"											\
				"vFragColor = color;"					\
			"}";

		glShaderSource(g_gluiShaderObjectFragment, 1, &szFragmentShaderSourceCode, NULL);

		//	Compile shader.
		glCompileShader(g_gluiShaderObjectFragment);

		gliCompileStatus = 0;
		gliInfoLogLength = 0;
		pszInfoLog = NULL;
		glsiWritten = 0;
		glGetShaderiv(g_gluiShaderObjectFragment, GL_COMPILE_STATUS, &gliCompileStatus);
		if (GL_FALSE == gliCompileStatus)
		{
			glGetShaderiv(g_gluiShaderObjectFragment, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
			if (gliInfoLogLength <= 0)
			{
				printf("Fragment : GL_INFO_LOG_LENGTH is less than 0.");
				[self release];
			}

			pszInfoLog = (char*)malloc(gliInfoLogLength);
			if (NULL == pszInfoLog)
			{
				printf("Fragment : malloc failed.");
				[self release];
			}

			glGetShaderInfoLog(g_gluiShaderObjectFragment, gliInfoLogLength, &glsiWritten, pszInfoLog);

			printf("Fragment shader compilation log : %s \n", pszInfoLog);
			free(pszInfoLog);
			[self release];
		}
		//-	Fragment shader.
		////////////////////////////////////////////////////////////////////

		////////////////////////////////////////////////////////////////////
		//+	Shader program.

		//	Create.
		g_gluiShaderObjectProgram = glCreateProgram();

		//	Attach vertex shader to shader program.
		glAttachShader(g_gluiShaderObjectProgram, g_gluiShaderObjectVertex);

		//	Attach Fragment shader to shader program.
		glAttachShader(g_gluiShaderObjectProgram, g_gluiShaderObjectFragment);

		//	IMP:Pre-link binding of shader program object with vertex shader position attribute.
		glBindAttribLocation(g_gluiShaderObjectProgram, RTR_ATTRIBUTE_POSITION, "vPosition");

		glBindAttribLocation(g_gluiShaderObjectProgram, RTR_ATTRIBUTE_NORMAL, "vNormal");

		//	Link shader.
		glLinkProgram(g_gluiShaderObjectProgram);

		GLint gliLinkStatus;
		gliInfoLogLength = 0;
		pszInfoLog = NULL;
		glsiWritten = 0;
		glGetShaderiv(g_gluiShaderObjectProgram, GL_LINK_STATUS, &gliLinkStatus);
		if (GL_FALSE == gliLinkStatus)
		{
			glGetProgramiv(g_gluiShaderObjectProgram, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
			if (gliInfoLogLength <= 0)
			{
				printf("Link : GL_INFO_LOG_LENGTH is less than 0.");
				[self release];
			}

			pszInfoLog = (char*)malloc(gliInfoLogLength);
			if (NULL == pszInfoLog)
			{
				printf("Link : malloc failed.");
				[self release];
			}

			glGetProgramInfoLog(g_gluiShaderObjectProgram, gliInfoLogLength, &glsiWritten, pszInfoLog);

			printf("Shader Link log : %s \n", pszInfoLog);
			free(pszInfoLog);
			[self release];
		}
		//-	Shader program.
		////////////////////////////////////////////////////////////////////

		//-	Shader code
		////////////////////////////////////////////////////////////////////

		//
		//	The actual locations assigned to uniform variables are not known until the program object is linked successfully.
		//	After a program object has been linked successfully, the index values for uniform variables remain fixed until the next link command occurs.
		//
		g_gluiModelViewMat4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_model_view_matrix");

		g_gluiProjectionMat4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_projection_matrix");

		g_gluiKeyPressedUniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_L_key_pressed");

		g_gluiLdVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_LD");

		g_gluiKdVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_KD");

		g_gluiLightPositionVec4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_light_position");

		////////////////////////////////////////////////////////////////////
		//+	Vertices,color, shader attribute, vbo,vao initialization.

		const GLfloat glfarrCubeVertices[] =
		{
			//	Top face
			1.0f, 1.0f, -1.0f,	//	top right 
			-1.0f, 1.0f, -1.0f,	//	top left 
			-1.0f, 1.0f, 1.0f,	//	bottom left
			1.0f, 1.0f, 1.0f,	//	bottom right 

			//	Bottom face
			1.0f, -1.0f, 1.0f,//	top right 
			-1.0f, -1.0f, 1.0f,	//	top left 
			-1.0f, -1.0f, -1.0f,	//	bottom left 
			1.0f, -1.0f, -1.0f,	//	bottom right

			//	Front face
			1.0f, 1.0f, 1.0f,	//	top right 
			-1.0f, 1.0f, 1.0f,	//	top left 
			-1.0f, -1.0f, 1.0f,	//	bottom left 
			1.0f, -1.0f, 1.0f,	//	bottom Right 

			//	Back face
			1.0f, -1.0f, -1.0f,	//	top Right 
			-1.0f, -1.0f, -1.0f,//	top left 
			-1.0f, 1.0f, -1.0f,	//	bottom left 
			1.0f, 1.0f, -1.0f,	//	bottom right 

			//	Left face
			-1.0f, 1.0f, 1.0f,	//	top right 
			-1.0f, 1.0f, -1.0f,	//	top left 
			-1.0f, -1.0f, -1.0f,//	bottom left 
			-1.0f, -1.0f, 1.0f,	//	bottom right

			//	Right face
			1.0f, 1.0f, -1.0f,	//	top right 
			1.0f, 1.0f, 1.0f,	//	top left 
			1.0f, -1.0f, 1.0f,	//	bottom left
			1.0f, -1.0f, -1.0f,	//	bottom Right 
		};

		const GLfloat glfarrCubeNormals[] =
		{
			//	Top face
			0.0f, 1.0f, 0.0f,	//	top right 
			0.0f, 1.0f, 0.0f,	//	top left 
			0.0f, 1.0f, 0.0f,	//	bottom left
			0.0f, 1.0f, 0.0f,	//	bottom right 

			//	Bottom face
			0.0f, -1.0f, 0.0f,//	top right 
			0.0f, -1.0f, 0.0f,	//	top left 
			0.0f, -1.0f, 0.0f,	//	bottom left 
			0.0f, -1.0f, 0.0f,	//	bottom right

			//	Front face
			0.0f, 0.0f, 1.0f,	//	top right 
			0.0f, 0.0f, 1.0f,	//	top left 
			0.0f, 0.0f, 1.0f,	//	bottom left 
			0.0f, 0.0f, 1.0f,	//	bottom Right 

			//	Back face
			0.0f, 0.0f, -1.0f,	//	top Right 
			0.0f, 0.0f, -1.0f,//	top left 
			0.0f, 0.0f, -1.0f,	//	bottom left 
			0.0f, 0.0f, -1.0f,	//	bottom right 

			//	Left face
			-1.0f, 0.0f, 0.0f,	//	top right 
			-1.0f, 0.0f, 0.0f,	//	top left 
			-1.0f, 0.0f, 0.0f,//	bottom left 
			-1.0f, 0.0f, 0.0f,	//	bottom right

			//	Right face
			1.0f, 0.0f, 0.0f,	//	top right 
			1.0f, 0.0f, 0.0f,	//	top left 
			1.0f, 0.0f, 0.0f,	//	bottom left
			1.0f, 0.0f, 0.0f,	//	bottom Right 
		};

		glGenVertexArrays(1, &g_gluiVAOCube);	//	It is like recorder.
		glBindVertexArray(g_gluiVAOCube);

		////////////////////////////////////////////////////////////////////
		//+ Vertex position
		glGenBuffers(1, &g_gluiVBOPosition);
		glBindBuffer(GL_ARRAY_BUFFER, g_gluiVBOPosition);

		glBufferData(GL_ARRAY_BUFFER, sizeof(glfarrCubeVertices), glfarrCubeVertices, GL_STATIC_DRAW);

		glVertexAttribPointer(RTR_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);

		glEnableVertexAttribArray(RTR_ATTRIBUTE_POSITION);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		//- Vertex position
		////////////////////////////////////////////////////////////////////

		////////////////////////////////////////////////////////////////////
		//+ Vertex Color
		glGenBuffers(1, &g_gluiVBONormal);
		glBindBuffer(GL_ARRAY_BUFFER, g_gluiVBONormal);

		glBufferData(GL_ARRAY_BUFFER, sizeof(glfarrCubeNormals), glfarrCubeNormals, GL_STATIC_DRAW);

		glVertexAttribPointer(RTR_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);

		glEnableVertexAttribArray(RTR_ATTRIBUTE_NORMAL);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		//- Vertex Color
		////////////////////////////////////////////////////////////////////

		glBindVertexArray(0);

		//-	Vertices,color, shader attribute, vbo,vao initialization.
		////////////////////////////////////////////////////////////////////

		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

		//+	Change 2 For 3D
		glClearDepthf(1.0f);

		glEnable(GL_DEPTH_TEST);

		glDepthFunc(GL_LEQUAL);

		//
		//	Optional.
		//
		//glShadeModel(GL_SMOOTH);
		//glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

		//
		//	We will always cull back faces for better performance.
		//	We will this in case of 3-D rotation/graphics.
		//
		//glEnable(GL_CULL_FACE);

		//-	Change 2 For 3D

		g_matPerspectiveProjection = vmath::mat4::identity();
		
		//	Gesture Recognition
		
		//	Tap gesture code.
		UITapGestureRecognizer *singleTapGestureRecognizer = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(onSingleTap:)];
		[singleTapGestureRecognizer setNumberOfTapsRequired:1];
		[singleTapGestureRecognizer setNumberOfTouchesRequired:1];
		[singleTapGestureRecognizer setDelegate:self];
		[self addGestureRecognizer:singleTapGestureRecognizer];
		
		UITapGestureRecognizer *doubleTapGestureRecognizer = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(onDoubleTap:)];
		[doubleTapGestureRecognizer setNumberOfTapsRequired:2];
		[doubleTapGestureRecognizer setNumberOfTouchesRequired:1];	//	Touch of 1 finger.
		[doubleTapGestureRecognizer setDelegate:self];
		[self addGestureRecognizer:doubleTapGestureRecognizer];
		
		//	This will allow to diffrentiate between single tap and double tap.
		[singleTapGestureRecognizer requireGestureRecognizerToFail:doubleTapGestureRecognizer];
		
		//	Swipe gesture
		UISwipeGestureRecognizer *swipeGestureRecognizer = [[UISwipeGestureRecognizer alloc] initWithTarget:self action:@selector(onSwipe:)];
		[self addGestureRecognizer:swipeGestureRecognizer];
		
		//	long-press gesture
		UILongPressGestureRecognizer *longPressGestureRecognizer = [[UILongPressGestureRecognizer alloc] initWithTarget:self action:@selector(onLongPress:)];
		[self addGestureRecognizer:longPressGestureRecognizer];
	}

    return self;
}

/*
// Only override drawRect: if you perform custom drawing.
// An empty implementation adversely affects performance during animation.
*/
/*
- (void)drawRect:(CGRect)rect
{
	//	Drawing code
}
*/

+(Class)layerClass	//	From CALayerDelegate
{
	//	code
	return ([CAEAGLLayer class]);
}

-(void)drawView:(id)sender
{
	[EAGLContext setCurrentContext:eaglContext];
	
	glBindFramebuffer(GL_FRAMEBUFFER, defaultFrameBuffer);
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);	//	Change 3 for 3D

	//	Start using opengl program.
	glUseProgram(g_gluiShaderObjectProgram);

	if (true == g_bLight)
	{
		glUniform1i(g_gluiKeyPressedUniform, 1);

		glUniform3f(g_gluiLdVec3Uniform, 1.0f, 1.0f, 1.0f);	//	Diffuse
		glUniform3f(g_gluiKdVec3Uniform, 0.5f, 0.5f, 0.5f);	//	grey effect

		float farrLightPosition[] = {0.0f, 0.0f, 2.0f, 1.0f};
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, farrLightPosition);
	}
	else
	{
		glUniform1i(g_gluiKeyPressedUniform, 0);
	}

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	vmath::mat4 matModel = vmath::mat4::identity();
	vmath::mat4 matModelView = vmath::mat4::identity();
	vmath::mat4 matRotation = vmath::mat4::identity();	//	Good practice to initialize to identity matrix though it will change in next call.

	matModel = vmath::translate(0.0f, 0.0f, -6.0f);

	matRotation = vmath::rotate(g_glfAngle, g_glfAngle, g_glfAngle);

	matModelView = matModel * matRotation;

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelViewMat4Uniform, 1, GL_FALSE, matModelView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOCube);

	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 20, 4);

	//	Unbind 'VAO'
	glBindVertexArray(0);

	//	Stop using opengl program.
	glUseProgram(0);
	
	[self updateGL];
	
	glBindRenderbuffer(GL_RENDERBUFFER, colorRenderBuffer);
	[eaglContext presentRenderbuffer:GL_RENDERBUFFER];
}


-(void)updateGL
{
	g_glfAngle = g_glfAngle + 0.1f;

	if (g_glfAngle >= 360)
	{
		g_glfAngle = 0.0f;
	}
}

-(void)layoutSubviews	//	Resize
{
	//	code
	GLint width;
	GLint height;
	
	glBindRenderbuffer(GL_RENDERBUFFER, colorRenderBuffer);
	[eaglContext renderbufferStorage:GL_RENDERBUFFER fromDrawable:(CAEAGLLayer*)self.layer];
	glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &width);
	glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &height);
	
	glGenRenderbuffers(1, &depthRenderBuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, depthRenderBuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, width, height);	//	For IOS 16
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderBuffer);
		
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		printf("Failed to create complete frame buffer object %x\n", glCheckFramebufferStatus(GL_FRAMEBUFFER));
	}	
	
	if (height == 0)
	{
		height = 1;
	}
	
	GLfloat fwidth = (GLfloat)width;
	GLfloat fheight = (GLfloat)height;
	
	glViewport(0,0,(GLsizei)width,(GLsizei)height);
	
	//	perspective(float fovy, float aspect, float n, float f)
	g_matPerspectiveProjection = vmath::perspective(45, fwidth / fheight, 0.1f, 100.0f);
	
	[self drawView:nil];
}

-(void)startAnimation
{
	if (!isAnimation)
	{
		displayLink = [NSClassFromString(@"CADisplayLink") displayLinkWithTarget:self selector:@selector(drawView:)];
		[displayLink setPreferredFramesPerSecond:animationFrameInterval];
		[displayLink addToRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
		isAnimation = YES;
	}
}

-(void)stopAnimation
{
	if (isAnimation)
	{
		[displayLink invalidate];
		displayLink = nil;
		
		isAnimation = NO;
	}
}

//	To become first responder
- (BOOL)acceptsFirstResponder
{
	return YES;
}

- (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event
{
	
}

- (void)onSingleTap:(UITapGestureRecognizer *)gr
{
	
}

- (void)onDoubleTap:(UITapGestureRecognizer *)gr
{
    if (false == g_bLight)
    {
        g_bLight = true;
    }
    else
    {
        g_bLight = false;
    }
	
}

- (void)onLongPress:(UILongPressGestureRecognizer *)gr
{
	
}

- (void)onSwipe:(UISwipeGestureRecognizer *)gr
{
	[self release];
	exit(0);
}

- (void)dealloc
{
	// destroy vao
	if (g_gluiVBONormal)
	{
		glDeleteBuffers(1, &g_gluiVBONormal);
		g_gluiVBONormal = 0;
	}

	if (g_gluiVBOPosition)
	{
		glDeleteBuffers(1, &g_gluiVBOPosition);
		g_gluiVBOPosition = 0;
	}

	if (g_gluiVAOCube)
	{
		glDeleteVertexArrays(1, &g_gluiVAOCube);
		g_gluiVAOCube = 0;
	}

	if (g_gluiShaderObjectVertex)
	{
		glDetachShader(g_gluiShaderObjectProgram, g_gluiShaderObjectVertex);
		glDeleteShader(g_gluiShaderObjectVertex);
		g_gluiShaderObjectVertex = 0;
	}

	if (g_gluiShaderObjectFragment)
	{
		glDetachShader(g_gluiShaderObjectProgram, g_gluiShaderObjectFragment);
		glDeleteShader(g_gluiShaderObjectFragment);
		g_gluiShaderObjectFragment = 0;
	}

	//
	//	Unlink shader program
	//	This will be useful when detach multiple shaders in loop.
	//	1.glUseProgram(Shader_Program_Object)
	//	2.Get Attach shader list
	//	3.Detach i loop.
	//	4.glUseProgram(0)
	//
	glUseProgram(0);

	if (g_gluiShaderObjectProgram)
	{
		glDeleteProgram(g_gluiShaderObjectProgram);
		g_gluiShaderObjectProgram = 0;
	}
	
	if (depthRenderBuffer)
	{
		glDeleteRenderbuffers(1, &depthRenderBuffer);
		depthRenderBuffer = 0;
	}
	
	if (colorRenderBuffer)
	{
		glDeleteRenderbuffers(1, &colorRenderBuffer);
		colorRenderBuffer = 0;
	}
	
	if (defaultFrameBuffer)
	{
		glDeleteFramebuffers(1, &defaultFrameBuffer);
		defaultFrameBuffer = 0;
	}
	
	if ([EAGLContext currentContext] == eaglContext)
	{
		[EAGLContext setCurrentContext:nil];
	}
	[eaglContext release];
	eaglContext = nil;
	
	[super dealloc];
}

@end
