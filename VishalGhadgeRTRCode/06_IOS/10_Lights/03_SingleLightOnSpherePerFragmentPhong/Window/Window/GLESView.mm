//
//  GLESView.m
//  Per fragment lighting.
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

GLfloat g_glfarrLightAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_glfarrLightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfarrLightSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfarrLightPosition[] = { 100.0f, 100.0f, 100.0f, 1.0f };

GLfloat g_glfarrMaterialAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_glfarrMaterialDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfarrMaterialSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfMaterialShininess = 50.0f;


@implementation GLESView
{
	EAGLContext *eaglContext;
	
	GLuint defaultFrameBuffer;
	GLuint colorRenderBuffer;
	GLuint depthRenderBuffer;
	
	id displayLink;
	NSInteger animationFrameInterval;
	BOOL isAnimation;

	GLuint vao;
	GLuint vbo_position;
	GLuint vbo_normal;
	GLuint vbo_texture;
	GLuint vbo_index;

	unsigned short *elements;
	float *verts;
	float *norms;
	float *texCoords;

	unsigned int numElements;
	unsigned int maxElements;
	unsigned int numVertices;

	GLuint g_gluiShaderObjectVertex;
	GLuint g_gluiShaderObjectFragment;
	GLuint g_gluiShaderObjectProgram;

	GLuint g_gluiVAOSphere;
	GLuint g_gluiVBOPosition;
	GLuint g_gluiVBONormal;
	GLuint g_gluiVBOElement;

	/////////////////////////////////////////////////////////////////
	//+Uniforms.
	GLuint g_gluiModelMat4Uniform;
	GLuint g_gluiViewMat4Uniform;
	GLuint g_gluiProjectionMat4Uniform;

	GLuint g_gluiKeyPressedUniform;

	GLuint g_gluiLaVec3Uniform;	//	light ambient
	GLuint g_gluiLdVec3Uniform;	//	light diffuse
	GLuint g_gluiLsVec3Uniform;	//	light specular
	GLuint g_gluiLightPositionVec4Uniform;

	GLuint g_gluiKaVec3Uniform;//	Material ambient
	GLuint g_gluiKdVec3Uniform;//	Material diffuse
	GLuint g_gluiKsVec3Uniform;//	Material specular
	GLuint g_gluiMaterialShininessUniform;
	//-Uniforms.
	/////////////////////////////////////////////////////////////////

	vmath::mat4 g_matPerspectiveProjection;

	bool g_bAnimate;
	bool g_bLight;
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
        g_bLight = false;
	
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
			"uniform mat4 u_model_matrix;"	\
			"uniform mat4 u_view_matrix;"	\
			"uniform mat4 u_projection_matrix;"	\
			"uniform vec4 u_light_position;"		\
			"uniform mediump int u_L_key_pressed;"			\
			"out vec3 transformed_normals;"			\
			"out vec3 light_direction;"			\
			"out vec3 viewer_vector;"			\
			"void main(void)"							\
			"{"											\
				"if (1 == u_L_key_pressed)"										\
				"{"											\
					"vec4 eyeCoordinates = u_view_matrix * u_model_matrix * vPosition;"											\
					"transformed_normals = mat3(u_view_matrix * u_model_matrix) * vNormal;"											\
					"light_direction = vec3(u_light_position) - eyeCoordinates.xyz;"											\
					"viewer_vector = -eyeCoordinates.xyz;"											\
				"}"											\
				"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;"	\
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
			"in vec3 transformed_normals;"			\
			"in vec3 light_direction;"			\
			"in vec3 viewer_vector;"			\
			"uniform vec3 u_La;	"				\
			"uniform vec3 u_Ld;	"				\
			"uniform vec3 u_Ls;	"				\
			"uniform vec3 u_Ka;"					\
			"uniform vec3 u_Kd;"					\
			"uniform vec3 u_Ks;"					\
			"uniform float u_material_shininess;"		\
			"uniform int u_L_key_pressed;"			\
			"out vec4 vFragColor;"						\
			"void main(void)"							\
			"{"											\
			"vec3 phong_ads_color;"					\
				"if (1 == u_L_key_pressed)"										\
				"{"											\
					"vec3 normalized_transformed_normals = normalize(transformed_normals);"											\
					"vec3 normalized_light_direction = normalize(light_direction);"											\
					"vec3 normalized_viewer_vector = normalize(viewer_vector);"											\
					"vec3 ambient = u_La * u_Ka;"											\
					"float tn_dot_ld = max(dot(normalized_transformed_normals, normalized_light_direction), 0.0);"											\
					"vec3 diffuse = u_Ld * u_Kd * tn_dot_ld;"											\
					"vec3 reflection_vector = reflect(-normalized_light_direction, normalized_transformed_normals);"											\
					"vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector, normalized_viewer_vector), 0.0), u_material_shininess);"											\
					"phong_ads_color = ambient + diffuse + specular;"											\
				"}"											\
				"else"											\
				"{"											\
				"	phong_ads_color = vec3(1.0,1.0,1.0);"											\
				"}"											\
			"vFragColor = vec4(phong_ads_color, 1.0);"					\
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
		g_gluiModelMat4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_model_matrix");

		g_gluiViewMat4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_view_matrix");

		g_gluiProjectionMat4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_projection_matrix");

		g_gluiKeyPressedUniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_L_key_pressed");

		g_gluiLaVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_La");

		g_gluiLdVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Ld");

		g_gluiLsVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Ls");

		g_gluiLightPositionVec4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_light_position");

		g_gluiKaVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Ka");

		g_gluiKdVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Kd");

		g_gluiKsVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Ks");

		g_gluiMaterialShininessUniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_material_shininess");

		// *** vertices, colors, shader attribs, vbo, vao initializations ***
		[self makeSphere:2.0f :30 :30];

		glClearColor(0.0f, 0.0f, 1.0f, 0.0f);

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
		glEnable(GL_CULL_FACE);

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
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	//	Start using opengl program.
	glUseProgram(g_gluiShaderObjectProgram);

	if (true == g_bLight)
	{
		glUniform1i(g_gluiKeyPressedUniform, 1);

		glUniform3fv(g_gluiLaVec3Uniform, 1, g_glfarrLightAmbient);	//	Ambient
		glUniform3fv(g_gluiLdVec3Uniform, 1, g_glfarrLightDiffuse);	//	Diffuse
		glUniform3fv(g_gluiLsVec3Uniform, 1, g_glfarrLightSpecular);	//	Specular
		glUniform4fv(g_gluiLightPositionVec4Uniform, 1, g_glfarrLightPosition);

		glUniform3fv(g_gluiKaVec3Uniform, 1, g_glfarrMaterialAmbient);
		glUniform3fv(g_gluiKdVec3Uniform, 1, g_glfarrMaterialDiffuse);
		glUniform3fv(g_gluiKsVec3Uniform, 1, g_glfarrMaterialSpecular);
		glUniform1f(g_gluiMaterialShininessUniform, g_glfMaterialShininess);
	}
	else
	{
		glUniform1i(g_gluiKeyPressedUniform, 0);
	}

	//	OpenGl drawing.
	//	Set modelview and modelviewprojection matrix to identity.
	vmath::mat4 matModel = vmath::mat4::identity();
	vmath::mat4 matView = vmath::mat4::identity();

	matModel = vmath::translate(0.0f, 0.0f, -8.0f);

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	[self drawSphere];

	//	Stop using opengl program.
	glUseProgram(0);
	
	glBindRenderbuffer(GL_RENDERBUFFER, colorRenderBuffer);
	[eaglContext presentRenderbuffer:GL_RENDERBUFFER];
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
	if (vao)
	{
		glDeleteVertexArrays(1, &vao);
		vao = 0;
	}

	// destroy vbo_position
	if (vbo_position)
	{
		glDeleteBuffers(1, &vbo_position);
		vbo_position = 0;
	}
	
	// destroy vbo_normal
	if (vbo_normal)
	{
		glDeleteBuffers(1, &vbo_normal);
		vbo_normal = 0;
	}
	
	// destroy vbo_texture
	if (vbo_texture)
	{
		glDeleteBuffers(1, &vbo_texture);
		vbo_texture = 0;
	}
	
	// destroy vbo_index
	if (vbo_index)
	{
		glDeleteBuffers(1, &vbo_index);
		vbo_index = 0;
	}
	
	[self cleanupMeshData];

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


//	Sphere
-(void)allocate:(int)numIndices
{
	// code
	// first cleanup, if not initially empty
	// [self cleanupMeshData];
	
	maxElements = numIndices;
	numElements = 0;
	numVertices = 0;
	
	int iNumIndices = numIndices/3;
	
	elements = (unsigned short *)malloc(iNumIndices * 3 * sizeof(unsigned short)); // 3 is x,y,z and 2 is sizeof short
	verts = (float *)malloc(iNumIndices * 3 * sizeof(float)); // 3 is x,y,z and 4 is sizeof float
	norms = (float *)malloc(iNumIndices * 3 * sizeof(float)); // 3 is x,y,z and 4 is sizeof float
	texCoords = (float *)malloc(iNumIndices * 2 * sizeof(float)); // 2 is s,t and 4 is sizeof float
}

// Add 3 vertices, 3 normal and 2 texcoords i.e. one triangle to the geometry.
// This searches the current list for identical vertices (exactly or nearly) and
// if one is found, it is added to the index array.
// if not, it is added to both the index array and the vertex array.
-(void)addTriangle:(float **)single_vertex :(float **)single_normal :(float **)single_texture
{
        const float diff = 0.00001f;
        int i, j;

        // code
        // normals should be of unit length
        [self normalizeVector:single_normal[0]];
        [self normalizeVector:single_normal[1]];
        [self normalizeVector:single_normal[2]];
        
        for (i = 0; i < 3; i++)
        {
            for (j = 0; j < numVertices; j++) //for the first ever iteration of 'j', numVertices will be 0 because of it's initialization in the parameterized constructor
            {
                if ([self isFoundIdentical:verts[j * 3] :single_vertex[i][0] :diff] &&
                    [self isFoundIdentical:verts[(j * 3) + 1] :single_vertex[i][1] :diff] &&
                    [self isFoundIdentical:verts[(j * 3) + 2] :single_vertex[i][2] :diff] &&
                    
                    [self isFoundIdentical:norms[j * 3] :single_normal[i][0] :diff] &&
                    [self isFoundIdentical:norms[(j * 3) + 1] :single_normal[i][1] :diff] &&
                    [self isFoundIdentical:norms[(j * 3) + 2] :single_normal[i][2] :diff] &&
                    
                    [self isFoundIdentical:texCoords[j * 2] :single_texture[i][0] :diff] &&
                    [self isFoundIdentical:texCoords[(j * 2) + 1] :single_texture[i][1] :diff])
                {
                    elements[numElements] = (short)j;
                    numElements++;
                    break;
                }
            }
            
            //If the single vertex, normal and texture do not match with the given, then add the corressponding triangle to the end of the list
            if (j == numVertices && numVertices < maxElements && numElements < maxElements)
            {
                verts[numVertices * 3] = single_vertex[i][0];
                verts[(numVertices * 3) + 1] = single_vertex[i][1];
                verts[(numVertices * 3) + 2] = single_vertex[i][2];

                norms[numVertices * 3] = single_normal[i][0];
                norms[(numVertices * 3) + 1] = single_normal[i][1];
                norms[(numVertices * 3) + 2] = single_normal[i][2];
                
                texCoords[numVertices * 2] = single_texture[i][0];
                texCoords[(numVertices * 2) + 1] = single_texture[i][1];
                
                elements[numElements] = (short)numVertices; //adding the index to the end of the list of elements/indices
                numElements++; //incrementing the 'end' of the list
                numVertices++; //incrementing coun of vertices
            }
        }
}

-(void)prepareToDraw
{
	// vao
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

        // vbo for position
	glGenBuffers(1, &vbo_position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
	glBufferData(GL_ARRAY_BUFFER, (maxElements * 3 * sizeof(float) / 3), verts, GL_STATIC_DRAW);

	glVertexAttribPointer(RTR_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(RTR_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind with vbo_position
        
        // vbo for normals
	glGenBuffers(1, &vbo_normal);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
	glBufferData(GL_ARRAY_BUFFER, (maxElements * 3 * sizeof(float) / 3), norms, GL_STATIC_DRAW);

	glVertexAttribPointer(RTR_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(RTR_ATTRIBUTE_NORMAL);

	glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind with vbo_normal
        
        // vbo for texture
	glGenBuffers(1, &vbo_texture);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_texture);
	glBufferData(GL_ARRAY_BUFFER, (maxElements * 2 * sizeof(float) / 3), texCoords, GL_STATIC_DRAW);

	glVertexAttribPointer(RTR_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(RTR_ATTRIBUTE_TEXTURE0);

	glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind with vbo_texture
        
        // vbo for index
	glGenBuffers(1, &vbo_index);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_index);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, (maxElements * 3 * sizeof(unsigned short) / 3), elements, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); // Unbind with vbo_index
        
	glBindVertexArray(0); // Unbind with vao
        
        // after sending data to GPU, now we can free our arrays
        // [self cleanupMeshData];
}

-(void)drawSphere
{
        // code
        // bind vao
	glBindVertexArray(vao);

        // draw
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_index);
	glDrawElements(GL_TRIANGLES, numElements, GL_UNSIGNED_SHORT, 0);

        // unbind vao
	glBindVertexArray(0); // Unbind with vao
}

-(int)getIndexCount
{
        // code
        return(numElements);
}

-(int)getVertexCount
{
        // code
        return(numVertices);
}

-(void)normalizeVector:(float *)v
{
        // code
        
        // square the vector length
        float squaredVectorLength = (v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2]);
        
        // get square root of above 'squared vector length'
        float squareRootOfSquaredVectorLength = (float)sqrt(squaredVectorLength);
        
        // scale the vector with 1/squareRootOfSquaredVectorLength
        v[0] = v[0] * 1.0f/squareRootOfSquaredVectorLength;
        v[1] = v[1] * 1.0f/squareRootOfSquaredVectorLength;
        v[2] = v[2] * 1.0f/squareRootOfSquaredVectorLength;
}

-(bool)isFoundIdentical:(float)val1 :(float)val2 :(float)diff
{
        // code
        if(fabs(val1 - val2) < diff)
            return(true);
        else
            return(false);
}

-(void)cleanupMeshData
{
        // code
        if(elements != NULL)
        {
	    free(elements);
            elements = NULL;
        }
        
        if(verts != NULL)
        {
	    free(verts);
            verts = NULL;
        }
        
        if(norms != NULL)
        {
	    free(norms);
            norms = NULL;
        }
        
        if(texCoords != NULL)
        {
	    free(texCoords);
            texCoords = NULL;
        }
}

-(void)releaseMemory:(float **)vertex :(float **)normal :(float **)texture
{
        for(int a = 0; a < 4; a++)
	{
		free(vertex[a]);
		free(normal[a]);
		free(texture[a]);
	}
	free(vertex);
	free(normal);
	free(texture);
}

-(void)makeSphere:(float)fRadius :(int)iSlices :(int)iStacks
{
    const float VDG_PI = 3.14159265358979323846;

    // code
    float drho = (float)VDG_PI / (float)iStacks;
    float dtheta = 2.0 * (float)VDG_PI / (float)iSlices;
    float ds = 1.0 / (float)(iSlices);
    float dt = 1.0 / (float)(iStacks);
    float t = 1.0;
    float s = 0.0;
    int i = 0;
    int j = 0;
    
    [self allocate:iSlices * iStacks * 6];
    
    for (i = 0; i < iStacks; i++)
    {
        float rho = (float)(i * drho);
        float srho = (float)(sin(rho));
        float crho = (float)(cos(rho));
        float srhodrho = (float)(sin(rho + drho));
        float crhodrho = (float)(cos(rho + drho));
        
        // Many sources of OpenGL sphere drawing code uses a triangle fan
        // for the caps of the sphere. This however introduces texturing
        // artifacts at the poles on some OpenGL implementations
        s = 0.0;
        
        // initialization of three 2-D arrays, two are 4 x 3 and one is 4 x 2
        float **vertex = (float **)malloc(sizeof(float *) * 4); // 4 rows
        for(int a = 0; a < 4; a++)
            vertex[a]= (float *)malloc(sizeof(float) * 3); // 3 columns
        float **normal = (float **)malloc(sizeof(float *) * 4); // 4 rows
        for(int a = 0;a < 4;a++)
            normal[a]= (float *)malloc(sizeof(float) * 3); // 3 columns
        float **texture = (float **)malloc(sizeof(float *) * 4); // 4 rows
        for(int a = 0;a < 4;a++)
            texture[a]= (float *)malloc(sizeof(float) * 2); // 2 columns

        for ( j = 0; j < iSlices; j++)
        {
            float theta = (j == iSlices) ? 0.0 : j * dtheta;
            float stheta = (float)(-sin(theta));
            float ctheta = (float)(cos(theta));
            
            float x = stheta * srho;
            float y = ctheta * srho;
            float z = crho;
           
            texture[0][0] = s;
            texture[0][1] = t;
            normal[0][0] = x;
            normal[0][1] = y;
            normal[0][2] = z;
            vertex[0][0] = x * fRadius;
            vertex[0][1] = y * fRadius;
            vertex[0][2] = z * fRadius;
            
            x = stheta * srhodrho;
            y = ctheta * srhodrho;
            z = crhodrho;
            
            texture[1][0] = s;
            texture[1][1] = t - dt;
            normal[1][0] = x;
            normal[1][1] = y;
            normal[1][2] = z;
            vertex[1][0] = x * fRadius;
            vertex[1][1] = y * fRadius;
            vertex[1][2] = z * fRadius;
            
            theta = ((j+1) == iSlices) ? 0.0 : (j+1) * dtheta;
            stheta = (float)(-sin(theta));
            ctheta = (float)(cos(theta));
            
            x = stheta * srho;
            y = ctheta * srho;
            z = crho;
            
            s += ds;
            texture[2][0] = s;
            texture[2][1] = t;
            normal[2][0] = x;
            normal[2][1] = y;
            normal[2][2] = z;
            vertex[2][0] = x * fRadius;
            vertex[2][1] = y * fRadius;
            vertex[2][2] = z * fRadius;
            
            x = stheta * srhodrho;
            y = ctheta * srhodrho;
            z = crhodrho;
            
            texture[3][0] = s;
            texture[3][1] = t - dt;
            normal[3][0] = x;
            normal[3][1] = y;
            normal[3][2] = z;
            vertex[3][0] = x * fRadius;
            vertex[3][1] = y * fRadius;
            vertex[3][2] = z * fRadius;
		
            [self addTriangle:vertex :normal :texture];
            
            // Rearrange for next triangle
            vertex[0][0]=vertex[1][0];
            vertex[0][1]=vertex[1][1];
            vertex[0][2]=vertex[1][2];
            normal[0][0]=normal[1][0];
            normal[0][1]=normal[1][1];
            normal[0][2]=normal[1][2];
            texture[0][0]=texture[1][0];
            texture[0][1]=texture[1][1];
            
            vertex[1][0]=vertex[3][0];
            vertex[1][1]=vertex[3][1];
            vertex[1][2]=vertex[3][2];
            normal[1][0]=normal[3][0];
            normal[1][1]=normal[3][1];
            normal[1][2]=normal[3][2];
            texture[1][0]=texture[3][0];
            texture[1][1]=texture[3][1];
            
            [self addTriangle:vertex :normal :texture];
        }
        t -= dt;
	[self releaseMemory:vertex :normal :texture];
    }

    [self prepareToDraw];
}

@end
