//
//  GLESView.m
//  2Lights On Rotating Pyramid
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

//
//	Light R == Red Light = Right Side light.
//
GLfloat g_glfarrLightRAmbient[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Decides general light
GLfloat g_glfarrLightRDiffuse[] = { 1.0f, 0.0f, 0.0f, 0.0f };	//	Decides color of light
GLfloat g_glfarrLightRSpecular[] = { 1.0f, 0.0f, 0.0f, 0.0f };	//	Decides height of light
GLfloat g_glfarrLightRPosition[] = { 2.0f, 1.0f, 1.0f, 0.0f };	//	Decides position of light

//
//	Light L == Blue Light = Left Side light.
//
GLfloat g_glfarrLightLAmbient[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Decides general light
GLfloat g_glfarrLightLDiffuse[] = { 0.0f, 0.0f, 1.0f, 0.0f };	//	Decides color of light
GLfloat g_glfarrLightLSpecular[] = { 0.0f, 0.0f, 1.0f, 0.0f };	//	Decides height of light
GLfloat g_glfarrLightLPosition[] = { -2.0f, 1.0f, 1.0f, 0.0f };	//	Decides position of light

GLfloat g_glfarrMaterialAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_glfarrMaterialDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfarrMaterialSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_glfMaterialShininess = 50.0f;

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

	GLuint g_gluiVAOPyramid;
	GLuint g_gluiVBOPosition;
	GLuint g_gluiVBONormal;

	/////////////////////////////////////////////////////////////////
	//+Uniforms.
	GLuint g_gluiModelMat4Uniform;
	GLuint g_gluiViewMat4Uniform;
	GLuint g_gluiProjectionMat4Uniform;

	GLuint g_gluiKeyPressedUniform;

	GLuint g_gluiLaRVec3Uniform;	//	light ambient
	GLuint g_gluiLdRVec3Uniform;	//	light diffuse
	GLuint g_gluiLsRVec3Uniform;	//	light specular
	GLuint g_gluiLightPositionRVec4Uniform;

	GLuint g_gluiLaLVec3Uniform;	//	light ambient
	GLuint g_gluiLdLVec3Uniform;	//	light diffuse
	GLuint g_gluiLsLVec3Uniform;	//	light specular
	GLuint g_gluiLightPositionLVec4Uniform;

	GLuint g_gluiKaVec3Uniform;//	Material ambient
	GLuint g_gluiKdVec3Uniform;//	Material diffuse
	GLuint g_gluiKsVec3Uniform;//	Material specular
	GLuint g_gluiMaterialShininessUniform;
	//-Uniforms.
	/////////////////////////////////////////////////////////////////

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
			"uniform mat4 u_model_matrix;"	\
			"uniform mat4 u_view_matrix;"	\
			"uniform mat4 u_projection_matrix;"	\
			"uniform mediump int u_L_key_pressed;"			\
			"uniform vec3 u_LaR;	"				\
			"uniform vec3 u_LdR;	"				\
			"uniform vec3 u_LsR;	"				\
			"uniform vec4 u_light_positionR;"		\
			"uniform vec3 u_LaL;	"				\
			"uniform vec3 u_LdL;	"				\
			"uniform vec3 u_LsL;	"				\
			"uniform vec4 u_light_positionL;"		\
			"uniform vec3 u_Ka;"					\
			"uniform vec3 u_Kd;"					\
			"uniform vec3 u_Ks;"					\
			"uniform float u_material_shininess;"		\
			"out vec3 out_phong_ads_color;"			\
			"void main(void)"							\
			"{"											\
			"vec4 matNotUsed = u_model_matrix *  vPosition;"											\
			"mat4 matModelView = u_view_matrix;"											\
			"if (1 == u_L_key_pressed)"										\
			"{"											\
			/*"vec4 eyeCoordinates = u_view_matrix * u_model_matrix * vPosition;"											\*/
			"vec4 eyeCoordinates = matModelView *  vPosition;"											\
			"vec3 tnorm = normalize(mat3(matModelView) * vNormal);"											\
			"vec3 light_directionL = normalize(vec3(u_light_positionL - eyeCoordinates));"											\
			"float tn_dot_ldL = max(dot(tnorm, light_directionL), 0.0);"											\
			"vec3 ambientL = u_LaL * u_Ka;"											\
			"vec3 diffuseL = u_LdL * u_Kd * tn_dot_ldL;"											\
			"vec3 reflection_vectorL = reflect(-light_directionL, tnorm);"											\
			"vec3 viewer_vectorL = normalize(-eyeCoordinates.xyz);"											\
			"vec3 specularL = u_LsL * u_Ks * pow(max(dot(reflection_vectorL, viewer_vectorL), 0.0), u_material_shininess);"											\
			"out_phong_ads_color = ambientL + diffuseL + specularL;"											\
			"vec3 light_directionR = normalize(vec3(u_light_positionR - eyeCoordinates));"											\
			"float tn_dot_ldR = max(dot(tnorm, light_directionR), 0.0);"											\
			"vec3 ambientR = u_LaR * u_Ka;"											\
			"vec3 diffuseR = u_LdR * u_Kd * tn_dot_ldR;"											\
			"vec3 reflection_vectorR = reflect(-light_directionR, tnorm);"											\
			"vec3 viewer_vectorR = normalize(-eyeCoordinates.xyz);"											\
			"vec3 specularR = u_LsR * u_Ks * pow(max(dot(reflection_vectorR, viewer_vectorR), 0.0), u_material_shininess);"											\
			/*"out_phong_ads_color = ambientR + diffuseR + specularR;"											\*/
			"out_phong_ads_color = ambientL + ambientR + diffuseL+ diffuseR + specularL+ specularR;"											\
			"}"											\
			"else"											\
			"{"											\
			"out_phong_ads_color = vec3(1.0,1.0,1.0);"											\
			"}"											\
			"gl_Position = u_projection_matrix * matModelView * vPosition;"	\
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
			"in vec3 out_phong_ads_color;"				\
			"out vec4 vFragColor;"						\
			"void main(void)"							\
			"{"											\
			"vFragColor = vec4(out_phong_ads_color, 1.0);"					\
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

		g_gluiLaRVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_LaR");

		g_gluiLdRVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_LdR");

		g_gluiLsRVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_LsR");

		g_gluiLightPositionRVec4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_light_positionR");

		g_gluiLaLVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_LaL");
		
		g_gluiLdLVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_LdL");
		
		g_gluiLsLVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_LsL");

		g_gluiLightPositionLVec4Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_light_positionL");

		g_gluiKaVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Ka");

		g_gluiKdVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Kd");
		
		g_gluiKsVec3Uniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_Ks");
		
		g_gluiMaterialShininessUniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_material_shininess");

		////////////////////////////////////////////////////////////////////
		//+	Vertices,color, shader attribute, vbo,vao initialization.

		const GLfloat glfarrPyramidVertices[] =
		{
			//	Front face
			0.0f, 1.0f, 0.0f,	//	apex
			-1.0f, -1.0f, 1.0f,	//	left_bottom
			1.0f, -1.0f, 1.0f,	//	right_bottom
			//	Right face
			0.0f, 1.0f, 0.0f,	//	apex
			1.0f, -1.0f, 1.0f,	//	left_bottom
			1.0f, -1.0f, -1.0f,	//	right_bottom
			//	Back face
			0.0f, 1.0f, 0.0f,	//	apex
			1.0f, -1.0f, -1.0f,	//	left_bottom
			-1.0f, -1.0f, -1.0f,	//	right_bottom
			//	Left face
			0.0f, 1.0f, 0.0f,	//	apex
			-1.0f, -1.0f, -1.0f,	//	left_bottom
			-1.0f, -1.0f, 1.0f,	//	right_bottom

		};

		const GLfloat glfarrPyramidNormals[] =
		{
			//	Front face
			0.0f, 0.447214f, 0.894427f,	//	apex
			0.0f, 0.447214f, 0.894427f,	//	left_bottom
			0.0f, 0.447214f, 0.894427f,	//	right_bottom

			//	Right face
			0.894427f, 0.447214f, 0.0f,	//	apex
			0.894427f, 0.447214f, 0.0f,	//	left_bottom
			0.894427f, 0.447214f, 0.0f,	//	right_bottom

			//	Back face
			0.0f, 0.447214f, -0.894427f,	//	apex
			0.0f, 0.447214f, -0.894427f,	//	left_bottom
			0.0f, 0.447214f, -0.894427f,	//	right_bottom

			//	Left face
			-0.894427f, 0.447214f, 0.0f,	//	apex
			-0.894427f, 0.447214f, 0.0f,	//	left_bottom
			-0.894427f, 0.447214f, 0.0f,	//	right_bottom
		};

		glGenVertexArrays(1, &g_gluiVAOPyramid);	//	It is like recorder.
		glBindVertexArray(g_gluiVAOPyramid);

		////////////////////////////////////////////////////////////////////
		//+ Vertex position
		glGenBuffers(1, &g_gluiVBOPosition);
		glBindBuffer(GL_ARRAY_BUFFER, g_gluiVBOPosition);

		glBufferData(GL_ARRAY_BUFFER, sizeof(glfarrPyramidVertices), glfarrPyramidVertices, GL_STATIC_DRAW);

		glVertexAttribPointer(RTR_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);

		glEnableVertexAttribArray(RTR_ATTRIBUTE_POSITION);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		//- Vertex position
		////////////////////////////////////////////////////////////////////

		////////////////////////////////////////////////////////////////////
		//+ Vertex Normal
		glGenBuffers(1, &g_gluiVBONormal);
		glBindBuffer(GL_ARRAY_BUFFER, g_gluiVBONormal);

		glBufferData(GL_ARRAY_BUFFER, sizeof(glfarrPyramidNormals), glfarrPyramidNormals, GL_STATIC_DRAW);

		glVertexAttribPointer(RTR_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);

		glEnableVertexAttribArray(RTR_ATTRIBUTE_NORMAL);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		//- Vertex Normal
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
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	//	Start using opengl program.
	glUseProgram(g_gluiShaderObjectProgram);

	if (true == g_bLight)
	{
		glUniform1i(g_gluiKeyPressedUniform, 1);

		glUniform3fv(g_gluiLaRVec3Uniform, 1, g_glfarrLightRAmbient);	//	Ambient
		glUniform3fv(g_gluiLdRVec3Uniform, 1, g_glfarrLightRDiffuse);	//	Diffuse
		glUniform3fv(g_gluiLsRVec3Uniform, 1, g_glfarrLightRSpecular);	//	Specular
		glUniform4fv(g_gluiLightPositionRVec4Uniform, 1, g_glfarrLightRPosition);

		glUniform3fv(g_gluiLaLVec3Uniform, 1, g_glfarrLightLAmbient);	//	Ambient
		glUniform3fv(g_gluiLdLVec3Uniform, 1, g_glfarrLightLDiffuse);	//	Diffuse
		glUniform3fv(g_gluiLsLVec3Uniform, 1, g_glfarrLightLSpecular);	//	Specular
		glUniform4fv(g_gluiLightPositionLVec4Uniform, 1, g_glfarrLightLPosition);

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
	vmath::mat4 matRotation = vmath::mat4::identity();	//	Good practice to initialize to identity matrix though it will change in next call.

	matModel = vmath::translate(0.0f, 0.0f, -5.0f);
	matRotation = vmath::rotate(g_glfAngle, 0.0f, 1.0f, 0.0f);
	matView = matModel * matRotation;

	//
	//	Pass above modelview matrix to the vertex shader in 'u_model_view_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gluiModelMat4Uniform, 1, GL_FALSE, matModel);
	glUniformMatrix4fv(g_gluiViewMat4Uniform, 1, GL_FALSE, matView);
	glUniformMatrix4fv(g_gluiProjectionMat4Uniform, 1, GL_FALSE, g_matPerspectiveProjection);

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOPyramid);
	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glDrawArrays(GL_TRIANGLES,0, 12);
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

	if (g_gluiVAOPyramid)
	{
		glDeleteVertexArrays(1, &g_gluiVAOPyramid);
		g_gluiVAOPyramid = 0;
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
