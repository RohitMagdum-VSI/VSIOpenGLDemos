#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>
#import <QuartzCore/CVDisplayLink.h>
#import <OpenGL/gl3.h>
#import <OpenGL/gl3ext.h>
#import "vmath.h"
#import "Sphere.h"

enum
{
	HAD_ATTRIBUTE_POSITION = 0,
	HAD_ATTRIBUTE_COLOR,
	HAD_ATTRIBUTE_NORMAL,
	HAD_ATTRIBUTE_TEXTURE0,
};

// 'C' style global function declarations
CVReturn MyDisplayLinkCallback(CVDisplayLinkRef,const CVTimeStamp *,const CVTimeStamp *,CVOptionFlags,CVOptionFlags *,void *);

//Global Declarations
FILE *gpFile = NULL;

//Interface Declarations
@interface AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@end

@interface GLView : NSOpenGLView
@end

//Entry-point function
int main(int argc, const char * argv[])
{
	NSAutoreleasePool *pPool=[[NSAutoreleasePool alloc]init];

	NSApp=[NSApplication sharedApplication];

	[NSApp setDelegate:[[AppDelegate alloc]init]];

	[NSApp run];

	[pPool release];

	return(0);
}

//Interface Implementations
@implementation AppDelegate
{
@private
	NSWindow *window;
	GLView *glView;
}

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification
{
	//Creating the required path and creating Log File
	NSBundle *mainBundle=[NSBundle mainBundle];
	NSString *appDirName=[mainBundle bundlePath];
	NSString *parentDirPath=[appDirName stringByDeletingLastPathComponent];
	NSString *logFileNameWithPath=[NSString stringWithFormat:@"%@/Log.txt",parentDirPath];
	const char *pszLogFileNameWithPath=[logFileNameWithPath cStringUsingEncoding:NSASCIIStringEncoding];
	gpFile=fopen(pszLogFileNameWithPath,"w");
	if(gpFile==NULL)
	{
		printf("Cannot Create Log File.\nExitting ...\n");
		[self release];
		[NSApp terminate:self];
	}
	fprintf(gpFile,"Program Is Started Successfully\n");

	//Window
	NSRect win_rect;
	win_rect=NSMakeRect(0.0,0.0,800.0,600.0);

	//Create Simple Window
	window =[[NSWindow alloc]initWithContentRect:win_rect 
							 styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable 
							 backing:NSBackingStoreBuffered 
							 defer:NO];

	[window setTitle:@"MacOS : 3 Rotating Lights"];
	[window center];

	glView=[[GLView alloc]initWithFrame:win_rect];

	[window setContentView:glView];
	[window setDelegate:self];
	[window makeKeyAndOrderFront:self];
}

- (void)applicationWillTerminate:(NSNotification *)notification
{
	fprintf(gpFile,"Program Is Terminated Successfully\n");

	if(gpFile)
	{
		fclose(gpFile);
		gpFile=NULL;	
	}
}

- (void)windowWillClose:(NSNotification *)notification
{
	[NSApp terminate:self];
}

- (void)dealloc
{
	[glView release];

	[window release];

	[super dealloc];
}
@end

@implementation GLView
{
@private
	CVDisplayLinkRef displayLink;

	vmath::mat4 gPerspectiveProjectionMatrix;

	GLfloat gAngle_Sphere;

	GLuint gVertexShaderObject;
	GLuint gFragmentShaderObject;
	GLuint gShaderProgramObject;

	GLuint gVao_Sphere;
	GLuint gVbo_Position, gVbo_Normal, gVbo_Elements;

	GLuint gModelMatrixUniform, gViewMatrixUniform, gProjectionMatrixUniform;
	GLuint gLKeyPressedUniform,gShaderToggleUniform;

	GLuint gLaUniform_red, gLdUniform_red, gLsUniform_red;
	GLuint glightPosition_Uniform_red;

	GLuint gLaUniform_blue, gLdUniform_blue, gLsUniform_blue;
	GLuint glightPosition_Uniform_blue;

	GLuint gLaUniform_green, gLdUniform_green, gLsUniform_green;
	GLuint glightPosition_Uniform_green;

	GLuint gKaUniform, gKdUniform, gKsUniform;
	GLuint gMaterialShininessUniform;

	GLfloat lightAmbient_Red[4];
	GLfloat lightDiffuse_Red[4];
	GLfloat lightSpecular_Red[4]; 
	GLfloat lightPosition_Red[4];
	GLfloat lightAmbient_Blue[4]; 
	GLfloat lightDiffuse_Blue[4]; 
	GLfloat lightSpecular_Blue[4];
	GLfloat lightPosition_Blue[4];
	GLfloat lightAmbient_Green[4]; 
	GLfloat lightDiffuse_Green[4]; 
	GLfloat lightSpecular_Green[4];
	GLfloat lightPosition_Green[4];

	GLfloat materialAmbient[4];
	GLfloat materialDiffuse[4];
	GLfloat materialSpecular[4];
	GLfloat materialShininess;


	BOOL gbLight;
	BOOL gbIsLKeyPressed;
	BOOL gbShaderToggleFlag;

	BOOL gbAnimate;
	BOOL gbIsAKeyPressed;

	GLfloat sphere_vertices[1146];
	GLfloat sphere_normals[1146];
	GLfloat sphere_textures[764];
	unsigned short sphere_elements[2280];
	int gNumVertices, gNumElements;

	Sphere *mySphere;
}

-(id)initWithFrame:(NSRect)frame;
{
	self=[super initWithFrame:frame];

	if(self)
	{
		[[self window]setContentView:self];

		NSOpenGLPixelFormatAttribute attribs[]=
		{
			NSOpenGLPFAOpenGLProfile,NSOpenGLProfileVersion4_1Core,
			NSOpenGLPFAScreenMask,CGDisplayIDToOpenGLDisplayMask(kCGDirectMainDisplay),
			NSOpenGLPFANoRecovery,
			NSOpenGLPFAAccelerated,
			NSOpenGLPFAColorSize,24,
			NSOpenGLPFADepthSize,24,
			NSOpenGLPFAAlphaSize,8,
			NSOpenGLPFADoubleBuffer,
			0
		};

		NSOpenGLPixelFormat *pixelFormat=[[[NSOpenGLPixelFormat alloc]initWithAttributes:attribs] autorelease];

		if(pixelFormat==nil)
		{
			fprintf(gpFile,"No Valid OpenGL Pixel Format Is Available. Exitting...\n");
			[self release];
			[NSApp terminate:self];
		}

		NSOpenGLContext *glContext=[[[NSOpenGLContext alloc]initWithFormat:pixelFormat shareContext:nil]autorelease];

		[self setPixelFormat:pixelFormat];

		[self setOpenGLContext:glContext];
	}

	gbLight = NO;
	lightAmbient_Red[0]=0.0f;
	lightAmbient_Red[1]=0.0f;
	lightAmbient_Red[2]=0.0f;
	lightAmbient_Red[3]=1.0f;

	lightDiffuse_Red[0]=1.0f;
	lightDiffuse_Red[1]=0.0f;
	lightDiffuse_Red[2]=0.0f;
	lightDiffuse_Red[3]=1.0f;

	lightSpecular_Red[0]=1.0f;
	lightSpecular_Red[1]=0.0f;
	lightSpecular_Red[2]=0.0f;
	lightSpecular_Red[3]=1.0f;

	lightPosition_Red[0]=0.0f;
	lightPosition_Red[1]=10.0f;
	lightPosition_Red[2]=10.0f;
	lightPosition_Red[3]=0.0f;

	lightAmbient_Blue[0]=0.0f;
	lightAmbient_Blue[1]=0.0f;
	lightAmbient_Blue[2]=0.0f;
	lightAmbient_Blue[3]=1.0f;

	lightDiffuse_Blue[0]=0.0f;
	lightDiffuse_Blue[1]=0.0f;
	lightDiffuse_Blue[2]=1.0f;
	lightDiffuse_Blue[3]=1.0f;

	lightSpecular_Blue[0]=0.0f;
	lightSpecular_Blue[1]=0.0f;
	lightSpecular_Blue[2]=1.0f;
	lightSpecular_Blue[3]=1.0f;

	lightPosition_Blue[0]=-10.0f;
	lightPosition_Blue[1]=10.0f;
	lightPosition_Blue[2]=0.0f;
	lightPosition_Blue[3]=0.0f;

	lightAmbient_Green[0]=0.0f;
	lightAmbient_Green[1]=0.0f;
	lightAmbient_Green[2]=0.0f;
	lightAmbient_Green[3]=1.0f;

	lightDiffuse_Green[0]=0.0f;
	lightDiffuse_Green[1]=1.0f;
	lightDiffuse_Green[2]=0.0f;
	lightDiffuse_Green[3]=1.0f;

	lightSpecular_Green[0]=0.0f;
	lightSpecular_Green[1]=1.0f;
	lightSpecular_Green[2]=0.0f;
	lightSpecular_Green[3]=1.0f;

	lightPosition_Green[0]=10.0f;
	lightPosition_Green[1]=0.0f;
	lightPosition_Green[2]=10.0f;
	lightPosition_Green[3]=0.0f;

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

	gbIsLKeyPressed=NO;
	gbShaderToggleFlag = NO;

	gbAnimate = NO;
	gbIsAKeyPressed = NO;

	mySphere = [[Sphere alloc]init];

	return(self);
}

-(CVReturn)getFrameForTime:(const CVTimeStamp *)pOutputTime
{
	NSAutoreleasePool *pool=[[NSAutoreleasePool alloc]init];

	[self drawView];

	[pool release];
	return(kCVReturnSuccess);
}

-(void)prepareOpenGL
{
	fprintf(gpFile,"OpenGL Version : %s\n",glGetString(GL_VERSION));
	fprintf(gpFile,"GLSL Version   : %s\n",glGetString(GL_SHADING_LANGUAGE_VERSION));

	[[self openGLContext]makeCurrentContext];

	GLint swapInt=1;
	[[self openGLContext]setValues:&swapInt forParameter:NSOpenGLCPSwapInterval];

	/*Sphere *sphere;

	sphere.getSphereVertexData(sphere_vertices,sphere_normals,sphere_textures,sphere_elements);
	gNumVertices=sphere.getNumberOfSphereVertices();
	gNumElements=sphere.getNumberOfSphereElements();*/

	[mySphere getSphereVertexDataWithPosition:sphere_vertices withNormals:sphere_normals withTexCoords:sphere_textures andElements:sphere_elements];
    gNumVertices = [mySphere getNumberOfSphereVertices];
    gNumElements = [mySphere getNumberOfSphereElements];


	gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *vertexShaderSourceCode =
		"#version 410" \
		"\n" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vNormal;" \
		"uniform mat4 u_model_matrix;" \
		"uniform mat4 u_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"uniform int u_lighting_enabled;" \
		"uniform vec4 u_light_position_red;" \
		"uniform vec4 u_light_position_blue;" \
		"uniform vec4 u_light_position_green;" \
		"uniform int u_toggle_shader;" \
		"uniform vec3 u_La_red;" \
		"uniform vec3 u_Ld_red;" \
		"uniform vec3 u_Ls_red;" \
		"uniform vec3 u_La_blue;" \
		"uniform vec3 u_Ld_blue;" \
		"uniform vec3 u_Ls_blue;" \
		"uniform vec3 u_La_green;" \
		"uniform vec3 u_Ld_green;" \
		"uniform vec3 u_Ls_green;" \
		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_material_shininess;" \
		"out vec3 transformed_normals;" \
		"out vec3 light_direction_red;" \
		"out vec3 light_direction_blue;" \
		"out vec3 light_direction_green;" \
		"out vec3 viewer_vector;" \
		"out vec3 phong_ads_color_vertex;" \
		"void main(void)" \
		"{" \
		"if(u_lighting_enabled==1)" \
		"{" \
		"vec4 eye_coordinates = u_view_matrix*u_model_matrix*vPosition;" \
		"transformed_normals = mat3(u_view_matrix*u_model_matrix)*vNormal;" \
		"light_direction_red = vec3(u_light_position_red)-eye_coordinates.xyz;" \
		"light_direction_blue = vec3(u_light_position_blue)-eye_coordinates.xyz;" \
		"light_direction_green = vec3(u_light_position_green)-eye_coordinates.xyz;" \
		"viewer_vector = -eye_coordinates.xyz;" \
		"if(u_toggle_shader == 1)" \
		"{" \
		"vec3 normalized_transformed_normals = normalize(transformed_normals);" \
		"vec3 normalized_light_direction_red = normalize(light_direction_red);" \
		"vec3 normalized_light_direction_blue = normalize(light_direction_blue);" \
		"vec3 normalized_light_direction_green = normalize(light_direction_green);" \
		"vec3 normalized_viewer_vector = normalize(viewer_vector);" \
		"vec3 ambient = u_La_red * u_Ka + u_La_blue * u_Ka + u_La_green * u_Ka;" \
		"float tn_dot_ld_red = max(dot(normalized_transformed_normals,normalized_light_direction_red),0.0);" \
		"float tn_dot_ld_blue = max(dot(normalized_transformed_normals,normalized_light_direction_blue),0.0);" \
		"float tn_dot_ld_green = max(dot(normalized_transformed_normals,normalized_light_direction_green),0.0);" \
		"vec3 diffuse = u_Ld_red * u_Kd * tn_dot_ld_red + u_Ld_blue * u_Kd * tn_dot_ld_blue + u_Ld_green * u_Kd * tn_dot_ld_green;" \
		"vec3 reflection_vector_red = reflect(-normalized_light_direction_red,normalized_transformed_normals);" \
		"vec3 reflection_vector_blue = reflect(-normalized_light_direction_blue,normalized_transformed_normals);" \
		"vec3 reflection_vector_green = reflect(-normalized_light_direction_green,normalized_transformed_normals);" \
		"vec3 specular = u_Ls_red * u_Ks * pow(max(dot(reflection_vector_red,normalized_viewer_vector),0.0),u_material_shininess) + u_Ls_blue * u_Ks * pow(max(dot(reflection_vector_blue,normalized_viewer_vector),0.0),u_material_shininess) + u_Ls_green * u_Ks * pow(max(dot(reflection_vector_green,normalized_viewer_vector),0.0),u_material_shininess);" \
		"phong_ads_color_vertex = ambient + diffuse + specular;" \
		"}" \
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
				fprintf(gpFile, "Vertex Shader Compilation Log : %s\n", szInfoLog);
				free(szInfoLog);
				//uninitialize(1);
				[self dealloc];
				exit(0);
			}
		}
	}

	//Fragment Shader
	gFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *fragmentShaderSourceCode =
		"#version 410" \
		"\n" \
		"in vec3 transformed_normals;" \
		"in vec3 light_direction_red;" \
		"in vec3 light_direction_blue;" \
		"in vec3 light_direction_green;" \
		"in vec3 viewer_vector;" \
		"in vec3 phong_ads_color_vertex;" \
		"out vec4 FragColor;" \
		"uniform vec3 u_La_red;" \
		"uniform vec3 u_Ld_red;" \
		"uniform vec3 u_Ls_red;" \
		"uniform vec3 u_La_blue;" \
		"uniform vec3 u_Ld_blue;" \
		"uniform vec3 u_Ls_blue;" \
		"uniform vec3 u_La_green;" \
		"uniform vec3 u_Ld_green;" \
		"uniform vec3 u_Ls_green;" \
		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_material_shininess;" \
		"uniform int u_lighting_enabled;" \
		"uniform int u_toggle_shader;" \
		"void main(void)" \
		"{" \
		"vec3 phong_ads_color;" \
		"if(u_lighting_enabled == 1)" \
		"{" \
		"if(u_toggle_shader == 0)" \
		"{" \
		"vec3 normalized_transformed_normals = normalize(transformed_normals);" \
		"vec3 normalized_light_direction_red = normalize(light_direction_red);" \
		"vec3 normalized_light_direction_blue = normalize(light_direction_blue);" \
		"vec3 normalized_light_direction_green = normalize(light_direction_green);" \
		"vec3 normalized_viewer_vector = normalize(viewer_vector);" \
		"float tn_dot_ld_red = max(dot(normalized_transformed_normals,normalized_light_direction_red),0.0);" \
		"float tn_dot_ld_blue = max(dot(normalized_transformed_normals,normalized_light_direction_blue),0.0);" \
		"float tn_dot_ld_green = max(dot(normalized_transformed_normals,normalized_light_direction_green),0.0);" \
		"vec3 reflection_vector_red = reflect(-normalized_light_direction_red,normalized_transformed_normals);" \
		"vec3 reflection_vector_blue = reflect(-normalized_light_direction_blue,normalized_transformed_normals);" \
		"vec3 reflection_vector_green = reflect(-normalized_light_direction_green,normalized_transformed_normals);" \
		"vec3 ambient = u_La_red * u_Ka + u_La_blue * u_Ka + u_La_green * u_Ka;" \
		"vec3 diffuse = u_Ld_red * u_Kd * tn_dot_ld_red + u_Ld_blue * u_Kd * tn_dot_ld_blue + u_Ld_green * u_Kd * tn_dot_ld_green;" \
		"vec3 specular = u_Ls_red * u_Ks * pow(max(dot(reflection_vector_red,normalized_viewer_vector),0.0),u_material_shininess) + u_Ls_blue * u_Ks * pow(max(dot(reflection_vector_blue,normalized_viewer_vector),0.0),u_material_shininess) + u_Ls_green * u_Ks * pow(max(dot(reflection_vector_green,normalized_viewer_vector),0.0),u_material_shininess);" \
		"phong_ads_color = ambient + diffuse + specular;" \
		"}" \
		"else" \
		"{" \
		"phong_ads_color = phong_ads_color_vertex;" \
		"}" \
		"}" \
		"else" \
		"{" \
		"phong_ads_color = vec3(1.0,1.0,1.0);" \
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
				fprintf(gpFile, "Fragment Shader Compilation Log : %s\n", szInfoLog);
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
				fprintf(gpFile, "Shader Program Link Log : %s\n", szInfoLog);
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

	gLaUniform_red = glGetUniformLocation(gShaderProgramObject, "u_La_red");
	gLdUniform_red = glGetUniformLocation(gShaderProgramObject, "u_Ld_red");
	gLsUniform_red = glGetUniformLocation(gShaderProgramObject, "u_Ls_red");

	glightPosition_Uniform_red = glGetUniformLocation(gShaderProgramObject, "u_light_position_red");

	gLaUniform_blue = glGetUniformLocation(gShaderProgramObject, "u_La_blue");
	gLdUniform_blue = glGetUniformLocation(gShaderProgramObject, "u_Ld_blue");
	gLsUniform_blue = glGetUniformLocation(gShaderProgramObject, "u_Ls_blue");

	glightPosition_Uniform_blue = glGetUniformLocation(gShaderProgramObject, "u_light_position_blue");

	gLaUniform_green = glGetUniformLocation(gShaderProgramObject, "u_La_green");
	gLdUniform_green = glGetUniformLocation(gShaderProgramObject, "u_Ld_green");
	gLsUniform_green = glGetUniformLocation(gShaderProgramObject, "u_Ls_green");

	glightPosition_Uniform_green = glGetUniformLocation(gShaderProgramObject, "u_light_position_green");

	gKaUniform = glGetUniformLocation(gShaderProgramObject, "u_Ka");
	gKdUniform = glGetUniformLocation(gShaderProgramObject, "u_Kd");
	gKsUniform = glGetUniformLocation(gShaderProgramObject, "u_Ks");

	gMaterialShininessUniform = glGetUniformLocation(gShaderProgramObject, "u_material_shininess");

	gShaderToggleUniform = glGetUniformLocation(gShaderProgramObject, "u_toggle_shader");

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

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix = vmath::mat4::identity();

	

	CVDisplayLinkCreateWithActiveCGDisplays(&displayLink);
	CVDisplayLinkSetOutputCallback(displayLink,&MyDisplayLinkCallback,self);
	CGLContextObj cglContext = (CGLContextObj)[[self openGLContext]CGLContextObj];
	CGLPixelFormatObj cglPixelFormat = (CGLPixelFormatObj)[[self pixelFormat]CGLPixelFormatObj];
	CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext(displayLink,cglContext,cglPixelFormat);
	CVDisplayLinkStart(displayLink);
}

-(void)reshape
{
	CGLLockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);

	NSRect rect = [self bounds];

	GLfloat width=rect.size.width;
	GLfloat height=rect.size.height;

	if(height == 0)
		height=1;

	glViewport(0,0,(GLsizei)width,(GLsizei)height);

	gPerspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);

	CGLUnlockContext((CGLContextObj)[[self openGLContext] CGLContextObj]);	
}

- (void)drawRect:(NSRect)dirtyRect
{
	[self drawView];
}

- (void)drawView
{
	[[self openGLContext]makeCurrentContext];

	CGLLockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);

	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

	GLfloat X_Coord_Of_Light;
	GLfloat Y_Coord_Of_Light;
	
	//Use Shader Program Object
	glUseProgram(gShaderProgramObject);

	if (gbLight == YES)
	{
		glUniform1i(gLKeyPressedUniform, 1);

		X_Coord_Of_Light = sin(gAngle_Sphere);
		Y_Coord_Of_Light = cos(gAngle_Sphere);

		lightPosition_Blue[0] = X_Coord_Of_Light * 2.0f;
		lightPosition_Blue[1] = Y_Coord_Of_Light * 2.0f;
		lightPosition_Blue[2] = -2.0f;

		lightPosition_Red[1] = X_Coord_Of_Light * 2.0f;
		lightPosition_Red[2] = Y_Coord_Of_Light * 2.0f;

		lightPosition_Green[0] = X_Coord_Of_Light * 2.0f;
		lightPosition_Green[2] = Y_Coord_Of_Light * 2.0f;

		if (gbShaderToggleFlag == YES)
			glUniform1i(gShaderToggleUniform, 1);
		else
			glUniform1i(gShaderToggleUniform, 0);

		glUniform3fv(gLaUniform_red, 1, lightAmbient_Red);
		glUniform3fv(gLdUniform_red, 1, lightDiffuse_Red);
		glUniform3fv(gLsUniform_red, 1, lightSpecular_Red);
		glUniform4fv(glightPosition_Uniform_red, 1, lightPosition_Red);

		glUniform3fv(gLaUniform_blue, 1, lightAmbient_Blue);
		glUniform3fv(gLdUniform_blue, 1, lightDiffuse_Blue);
		glUniform3fv(gLsUniform_blue, 1, lightSpecular_Blue);
		glUniform4fv(glightPosition_Uniform_blue, 1, lightPosition_Blue);

		glUniform3fv(gLaUniform_green, 1, lightAmbient_Green);
		glUniform3fv(gLdUniform_green, 1, lightDiffuse_Green);
		glUniform3fv(gLsUniform_green, 1, lightSpecular_Green);
		glUniform4fv(glightPosition_Uniform_green, 1, lightPosition_Green);

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
	vmath::mat4 rotationMatrix=vmath::mat4::identity();

	modelMatrix = vmath::translate(0.0f, 0.0f, -2.0f);

	//rotationMatrix = vmath::rotate(gAngle_Sphere,0.0f,1.0f,0.0f);

	//modelMatrix = modelMatrix * rotationMatrix;

	glUniformMatrix4fv(gModelMatrixUniform, 1, GL_FALSE, modelMatrix);

	glUniformMatrix4fv(gViewMatrixUniform, 1, GL_FALSE, viewMatrix);

	glUniformMatrix4fv(gProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);

	glBindVertexArray(0);

	glUseProgram(0);


	CGLFlushDrawable((CGLContextObj)[[self openGLContext]CGLContextObj]);
	CGLUnlockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);

	if(gbAnimate==YES)
	{
		gAngle_Sphere = gAngle_Sphere + 0.01f;
		if(gAngle_Sphere >= 360.0f)
			gAngle_Sphere = gAngle_Sphere - 360.0f;		
	}
}

-(BOOL)acceptsFirstResponder
{
	[[self window]makeFirstResponder:self];

	return(YES);
}

-(void)keyDown:(NSEvent *)theEvent
{
	int key=(int)[[theEvent characters]characterAtIndex:0];
	switch(key)
	{
		case 27:
			[self release];
			[NSApp terminate:self];
			break;

		case 'A':
		case 'a':
			if (gbIsAKeyPressed == NO)
			{
				gbAnimate = YES;
				gbIsAKeyPressed = YES;
			}
			else
			{
				gbAnimate = NO;
				gbIsAKeyPressed = NO;
			}
			break;

		case 'F':
		case 'f':
			[[self window]toggleFullScreen:self];
			break;
		case 'L':
		case 'l':
			if (gbIsLKeyPressed == NO)
			{
				gbLight = YES;
				gbIsLKeyPressed = YES;
			}
			else
			{
				gbLight = NO;
				gbIsLKeyPressed = NO;
			}
			break;

		case 'T':
		case 't':
			if(gbShaderToggleFlag==NO)
				gbShaderToggleFlag=YES;
			else
				gbShaderToggleFlag=NO;
			break;	
		
		default:
			break;
	}
}


-(void)mouseDown:(NSEvent *)theEvent
{

}

-(void)mouseDragged:(NSEvent *)theEvent
{

}

-(void)rightMouseDown:(NSEvent *)theEvent
{

}

- (void) dealloc
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

	CVDisplayLinkStop(displayLink);
	CVDisplayLinkRelease(displayLink);

	[super dealloc];
}
@end

CVReturn MyDisplayLinkCallback(CVDisplayLinkRef displayLink,const CVTimeStamp *pNow,const CVTimeStamp *pOutputTime,CVOptionFlags flagsIn,
								CVOptionFlags *pFlagsOut,void *pDisplayLinkContext)
{
	CVReturn result=[(GLView *)pDisplayLinkContext getFrameForTime:pOutputTime];
	return(result);
}
