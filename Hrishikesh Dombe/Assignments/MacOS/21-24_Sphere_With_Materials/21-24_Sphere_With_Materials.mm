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

	[window setTitle:@"MacOS : 24 Sphere With Materials"];
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

	GLfloat material_ambient_1[4];
	GLfloat material_diffuse_1[4];
	GLfloat material_specular_1[4]; 
	GLfloat material_shininess_1;

	GLfloat material_ambient_2[4];
	GLfloat material_diffuse_2[4];
	GLfloat material_specular_2[4]; 
	GLfloat material_shininess_2;

	GLfloat material_ambient_3[4];
	GLfloat material_diffuse_3[4];
	GLfloat material_specular_3[4];
	GLfloat material_shininess_3;

	GLfloat material_ambient_4[4];
	GLfloat material_diffuse_4[4];
	GLfloat material_specular_4[4]; 
	GLfloat material_shininess_4;

	GLfloat material_ambient_5[4];
	GLfloat material_diffuse_5[4];
	GLfloat material_specular_5[4]; 
	GLfloat material_shininess_5;

	GLfloat material_ambient_6[4];
	GLfloat material_diffuse_6[4];
	GLfloat material_specular_6[4];
	GLfloat material_shininess_6;

	GLfloat material_ambient_7[4];
	GLfloat material_diffuse_7[4];
	GLfloat material_specular_7[4]; 
	GLfloat material_shininess_7;

	GLfloat material_ambient_8[4];
	GLfloat material_diffuse_8[4];
	GLfloat material_specular_8[4];
	GLfloat material_shininess_8;

	GLfloat material_ambient_9[4];
	GLfloat material_diffuse_9[4];
	GLfloat material_specular_9[4]; 
	GLfloat material_shininess_9;

	GLfloat material_ambient_10[4]; 
	GLfloat material_diffuse_10[4];
	GLfloat material_specular_10[4];
	GLfloat material_shininess_10; 

	GLfloat material_ambient_11[4]; 
	GLfloat material_diffuse_11[4]; 
	GLfloat material_specular_11[4];
	GLfloat material_shininess_11; 

	GLfloat material_ambient_12[4]; 
	GLfloat material_diffuse_12[4]; 
	GLfloat material_specular_12[4];
	GLfloat material_shininess_12; 

	GLfloat material_ambient_13[4]; 
	GLfloat material_diffuse_13[4]; 
	GLfloat material_specular_13[4];
	GLfloat material_shininess_13; 

	GLfloat material_ambient_14[4]; 
	GLfloat material_diffuse_14[4]; 
	GLfloat material_specular_14[4];
	GLfloat material_shininess_14; 

	GLfloat material_ambient_15[4]; 
	GLfloat material_diffuse_15[4]; 
	GLfloat material_specular_15[4];
	GLfloat material_shininess_15; 

	GLfloat material_ambient_16[4]; 
	GLfloat material_diffuse_16[4]; 
	GLfloat material_specular_16[4];
	GLfloat material_shininess_16; 

	GLfloat material_ambient_17[4]; 
	GLfloat material_diffuse_17[4]; 
	GLfloat material_specular_17[4];
	GLfloat material_shininess_17; 

	GLfloat material_ambient_18[4]; 
	GLfloat material_diffuse_18[4]; 
	GLfloat material_specular_18[4];
	GLfloat material_shininess_18; 

	GLfloat material_ambient_19[4]; 
	GLfloat material_diffuse_19[4]; 
	GLfloat material_specular_19[4];
	GLfloat material_shininess_19; 

	GLfloat material_ambient_20[4]; 
	GLfloat material_diffuse_20[4]; 
	GLfloat material_specular_20[4];
	GLfloat material_shininess_20; 

	GLfloat material_ambient_21[4]; 
	GLfloat material_diffuse_21[4]; 
	GLfloat material_specular_21[4];
	GLfloat material_shininess_21; 

	GLfloat material_ambient_22[4]; 
	GLfloat material_diffuse_22[4]; 
	GLfloat material_specular_22[4];
	GLfloat material_shininess_22; 

	GLfloat material_ambient_23[4]; 
	GLfloat material_diffuse_23[4]; 
	GLfloat material_specular_23[4];
	GLfloat material_shininess_23;

	GLfloat material_ambient_24[4]; 
	GLfloat material_diffuse_24[4]; 
	GLfloat material_specular_24[4];
	GLfloat material_shininess_24; 


	BOOL gbLight;
	BOOL gbAnimate;
	BOOL gbIsLKeyPressed;
	BOOL gbIsAKeyPressed;
	BOOL gbIsXKeyPressed;
	BOOL gbIsYKeyPressed;
	BOOL gbIsZKeyPressed;

	int giHeight, giWidth;

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
	gbAnimate = NO;
	gbIsLKeyPressed = NO;
	gbIsXKeyPressed = NO;
	gbIsYKeyPressed = NO;
	gbIsZKeyPressed = NO;

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

	material_ambient_1[0]=0.0215f;
	material_ambient_1[1]=0.1745f;
	material_ambient_1[2]=0.0215f;
	material_ambient_1[3]=1.0f;

	material_diffuse_1[0]=0.07568f;
	material_diffuse_1[1]=0.61424f;
	material_diffuse_1[2]=0.07568f;
	material_diffuse_1[3]=1.0f;

	material_specular_1[0]=0.633f;
	material_specular_1[1]=0.727811f; 
	material_specular_1[2]=0.633f; 
	material_specular_1[3]=1.0f;  
	
	material_shininess_1=0.6f * 128.0f;

	material_ambient_2[0]=0.135f;
	material_ambient_2[1]=0.2225f;
	material_ambient_2[2]=0.1575f;
	material_ambient_2[3]=1.0f;

	material_diffuse_2[0]=0.54f;
	material_diffuse_2[1]=0.89f;
	material_diffuse_2[2]=0.63f;
	material_diffuse_2[3]=1.0f;

	material_specular_2[0]=0.316228f;
	material_specular_2[1]=0.316228f; 
	material_specular_2[2]=0.316228f; 
	material_specular_2[3]=1.0f;  
	
	material_shininess_2=0.1f * 128.0f;

	material_ambient_3[0]=0.05375f;
	material_ambient_3[1]=0.05f;
	material_ambient_3[2]=0.06625f;
	material_ambient_3[3]=1.0f;

	material_diffuse_3[0]=0.18275f;
	material_diffuse_3[1]=0.17f;
	material_diffuse_3[2]=0.22525f;
	material_diffuse_3[3]=1.0f;

	material_specular_3[0]=0.332741f;
	material_specular_3[1]=0.328634f; 
	material_specular_3[2]=0.346435f; 
	material_specular_3[3]=1.0f;  
	
	material_shininess_3=0.3f * 128.0f;

	material_ambient_4[0]=0.25f;
	material_ambient_4[1]=0.20725f;
	material_ambient_4[2]=0.20725f;
	material_ambient_4[3]=1.0f;

	material_diffuse_4[0]=1.0f;
	material_diffuse_4[1]=0.829f;
	material_diffuse_4[2]=0.829f;
	material_diffuse_4[3]=1.0f;

	material_specular_4[0]=0.296648f;
	material_specular_4[1]=0.296648f; 
	material_specular_4[2]=0.296648f; 
	material_specular_4[3]=1.0f;  
	
	material_shininess_4=0.088f * 128.0f;

	material_ambient_5[0]=0.1745f;
	material_ambient_5[1]=0.01175f;
	material_ambient_5[2]=0.01175f;
	material_ambient_5[3]=1.0f;

	material_diffuse_5[0]=0.61424f;
	material_diffuse_5[1]=0.04136f;
	material_diffuse_5[2]=0.04136f;
	material_diffuse_5[3]=1.0f;

	material_specular_5[0]=0.727811f;
	material_specular_5[1]=0.626959f; 
	material_specular_5[2]=0.626959f; 
	material_specular_5[3]=1.0f;  
	
	material_shininess_5=0.6f * 128.0f;

	material_ambient_6[0]=0.1f;
	material_ambient_6[1]=0.18725f;
	material_ambient_6[2]=0.1745f;
	material_ambient_6[3]=1.0f;

	material_diffuse_6[0]=0.396f;
	material_diffuse_6[1]=0.74151f;
	material_diffuse_6[2]=0.69102f;
	material_diffuse_6[3]=1.0f;

	material_specular_6[0]=0.297254f;
	material_specular_6[1]=0.30829f; 
	material_specular_6[2]=0.306678f; 
	material_specular_6[3]=1.0f;  
	
	material_shininess_6=0.1f * 128.0f;

	material_ambient_7[0]=0.329412f;
	material_ambient_7[1]=0.223529f;
	material_ambient_7[2]=0.027451f;
	material_ambient_7[3]=1.0f;

	material_diffuse_7[0]=0.780392f;
	material_diffuse_7[1]=0.568627f;
	material_diffuse_7[2]=0.113725f;
	material_diffuse_7[3]=1.0f;

	material_specular_7[0]=0.992157f;
	material_specular_7[1]=0.941176f; 
	material_specular_7[2]=0.807843f; 
	material_specular_7[3]=1.0f;  
	
	material_shininess_7=0.21794872f * 128.0f;

	material_ambient_8[0]=0.2125f;
	material_ambient_8[1]=0.1275f;
	material_ambient_8[2]=0.054f;
	material_ambient_8[3]=1.0f;

	material_diffuse_8[0]=0.714f;
	material_diffuse_8[1]=0.4284f;
	material_diffuse_8[2]=0.18144f;
	material_diffuse_8[3]=1.0f;

	material_specular_8[0]=0.393548f;
	material_specular_8[1]=0.271906f; 
	material_specular_8[2]=0.166721f; 
	material_specular_8[3]=1.0f;  
	
	material_shininess_8=0.2f * 128.0f;

	material_ambient_9[0]=0.25f;
	material_ambient_9[1]=0.25f;
	material_ambient_9[2]=0.25f;
	material_ambient_9[3]=1.0f;

	material_diffuse_9[0]=0.4f;
	material_diffuse_9[1]=0.4f;
	material_diffuse_9[2]=0.4f;
	material_diffuse_9[3]=1.0f;

	material_specular_9[0]=0.774597f;
	material_specular_9[1]=0.774597f; 
	material_specular_9[2]=0.774597f; 
	material_specular_9[3]=1.0f;  
	
	material_shininess_9=0.6f * 128.0f;

	material_ambient_10[0]=0.19125f;
	material_ambient_10[1]=0.0735f;
	material_ambient_10[2]=0.0225f;
	material_ambient_10[3]=1.0f;

	material_diffuse_10[0]=0.7038f;
	material_diffuse_10[1]=0.27048f;
	material_diffuse_10[2]=0.0828f;
	material_diffuse_10[3]=1.0f;

	material_specular_10[0]=0.256777f;
	material_specular_10[1]=0.137622f; 
	material_specular_10[2]=0.086014f; 
	material_specular_10[3]=1.0f;  
	
	material_shininess_10=0.1f * 128.0f;

	material_ambient_11[0]=0.24725f;
	material_ambient_11[1]=0.1995f;
	material_ambient_11[2]=0.0745f;
	material_ambient_11[3]=1.0f;

	material_diffuse_11[0]=0.75164f;
	material_diffuse_11[1]=0.60648f;
	material_diffuse_11[2]=0.22648f;
	material_diffuse_11[3]=1.0f;

	material_specular_11[0]=0.628281f;
	material_specular_11[1]=0.555802f; 
	material_specular_11[2]=0.366065f; 
	material_specular_11[3]=1.0f;  
	
	material_shininess_11=0.4f * 128.0f;

	material_ambient_12[0]=0.19225f;
	material_ambient_12[1]=0.19225f;
	material_ambient_12[2]=0.19225f;
	material_ambient_12[3]=1.0f;

	material_diffuse_12[0]=0.50754;
	material_diffuse_12[1]=0.50754;
	material_diffuse_12[2]=0.50754;
	material_diffuse_12[3]=1.0f;

	material_specular_12[0]=0.508273;
	material_specular_12[1]=0.508273; 
	material_specular_12[2]=0.508273; 
	material_specular_12[3]=1.0f;  
	
	material_shininess_12=0.4f * 128.0f;

	material_ambient_13[0]=0.0f;
	material_ambient_13[1]=0.0f;
	material_ambient_13[2]=0.0f;
	material_ambient_13[3]=1.0f;

	material_diffuse_13[0]=0.01f;
	material_diffuse_13[1]=0.01f;
	material_diffuse_13[2]=0.01f;
	material_diffuse_13[3]=1.0f;

	material_specular_13[0]=0.5f;
	material_specular_13[1]=0.5f; 
	material_specular_13[2]=0.5f; 
	material_specular_13[3]=1.0f;  
	
	material_shininess_13=0.25f * 128.0f;

	material_ambient_14[0]=0.0f;
	material_ambient_14[1]=0.1f;
	material_ambient_14[2]=0.06f;
	material_ambient_14[3]=1.0f;

	material_diffuse_14[0]=0.0f;
	material_diffuse_14[1]=0.50980392f;
	material_diffuse_14[2]=0.50980392f;
	material_diffuse_14[3]=1.0f;

	material_specular_14[0]=0.50196078f;
	material_specular_14[1]=0.50196078f; 
	material_specular_14[2]=0.50196078f; 
	material_specular_14[3]=1.0f;  
	
	material_shininess_14=0.25f * 128.0f;

	material_ambient_15[0]=0.0f;
	material_ambient_15[1]=0.0f;
	material_ambient_15[2]=0.0f;
	material_ambient_15[3]=1.0f;

	material_diffuse_15[0]=0.1f;
	material_diffuse_15[1]=0.35f;
	material_diffuse_15[2]=0.1f;
	material_diffuse_15[3]=1.0f;

	material_specular_15[0]=0.45f;
	material_specular_15[1]=0.55f; 
	material_specular_15[2]=0.45f; 
	material_specular_15[3]=1.0f;  
	
	material_shininess_15=0.25f * 128.0f;

	material_ambient_16[0]=0.0f;
	material_ambient_16[1]=0.0f;
	material_ambient_16[2]=0.0f;
	material_ambient_16[3]=1.0f;

	material_diffuse_16[0]=0.5f;
	material_diffuse_16[1]=0.0f;
	material_diffuse_16[2]=0.0f;
	material_diffuse_16[3]=1.0f;

	material_specular_16[0]=0.7f;
	material_specular_16[1]=0.6f; 
	material_specular_16[2]=0.6f; 
	material_specular_16[3]=1.0f;  
	
	material_shininess_16=0.25f * 128.0f;

	material_ambient_17[0]=0.0f;
	material_ambient_17[1]=0.0f;
	material_ambient_17[2]=0.0f;
	material_ambient_17[3]=1.0f;

	material_diffuse_17[0]=0.55f;
	material_diffuse_17[1]=0.55f;
	material_diffuse_17[2]=0.55f;
	material_diffuse_17[3]=1.0f;

	material_specular_17[0]=0.70f;
	material_specular_17[1]=0.70f; 
	material_specular_17[2]=0.70f; 
	material_specular_17[3]=1.0f;  
	
	material_shininess_17=0.25f * 128.0f;

	material_ambient_18[0]=0.0f;
	material_ambient_18[1]=0.0f;
	material_ambient_18[2]=0.0f;
	material_ambient_18[3]=1.0f;

	material_diffuse_18[0]=0.5f;
	material_diffuse_18[1]=0.5f;
	material_diffuse_18[2]=0.0f;
	material_diffuse_18[3]=1.0f;

	material_specular_18[0]=0.6f;
	material_specular_18[1]=0.6f; 
	material_specular_18[2]=0.5f; 
	material_specular_18[3]=1.0f;  
	
	material_shininess_18=0.25f * 128.0f;

	material_ambient_19[0]=0.02f;
	material_ambient_19[1]=0.02f;
	material_ambient_19[2]=0.02f;
	material_ambient_19[3]=1.0f;

	material_diffuse_19[0]=0.1f;
	material_diffuse_19[1]=0.1f;
	material_diffuse_19[2]=0.1f;
	material_diffuse_19[3]=1.0f;

	material_specular_19[0]=0.4f;
	material_specular_19[1]=0.4f; 
	material_specular_19[2]=0.4f; 
	material_specular_19[3]=1.0f;  
	
	material_shininess_19=0.078125f * 128.0f;

	material_ambient_20[0]=0.0f;
	material_ambient_20[1]=0.05f;
	material_ambient_20[2]=0.05f;
	material_ambient_20[3]=1.0f;

	material_diffuse_20[0]=0.4f;
	material_diffuse_20[1]=0.5f;
	material_diffuse_20[2]=0.5f;
	material_diffuse_20[3]=1.0f;

	material_specular_20[0]=0.04f;
	material_specular_20[1]=0.7f; 
	material_specular_20[2]=0.7f; 
	material_specular_20[3]=1.0f;  
	
	material_shininess_20=0.078125f * 128.0f;

	material_ambient_21[0]=0.0f;
	material_ambient_21[1]=0.05f;
	material_ambient_21[2]=0.0f;
	material_ambient_21[3]=1.0f;

	material_diffuse_21[0]=0.4f;
	material_diffuse_21[1]=0.5f;
	material_diffuse_21[2]=0.4f;
	material_diffuse_21[3]=1.0f;

	material_specular_21[0]=0.04f;
	material_specular_21[1]=0.7f; 
	material_specular_21[2]=0.04f; 
	material_specular_21[3]=1.0f;  
	
	material_shininess_21=0.078125f * 128.0f;

	material_ambient_22[0]=0.05f;
	material_ambient_22[1]=0.0f;
	material_ambient_22[2]=0.0f;
	material_ambient_22[3]=1.0f;

	material_diffuse_22[0]=0.5f;
	material_diffuse_22[1]=0.4f;
	material_diffuse_22[2]=0.4f;
	material_diffuse_22[3]=1.0f;

	material_specular_22[0]=0.7f;
	material_specular_22[1]=0.04f; 
	material_specular_22[2]=0.04f; 
	material_specular_22[3]=1.0f;  
	
	material_shininess_22=0.078125f * 128.0f;

	material_ambient_23[0]=0.05f;
	material_ambient_23[1]=0.05f;
	material_ambient_23[2]=0.05f;
	material_ambient_23[3]=1.0f;

	material_diffuse_23[0]=0.5f;
	material_diffuse_23[1]=0.5f;
	material_diffuse_23[2]=0.5f;
	material_diffuse_23[3]=1.0f;

	material_specular_23[0]=0.7f;
	material_specular_23[1]=0.7f; 
	material_specular_23[2]=0.7f; 
	material_specular_23[3]=1.0f;  
	
	material_shininess_23=0.078125f * 128.0f;

	material_ambient_24[0]=0.05f;
	material_ambient_24[1]=0.05f;
	material_ambient_24[2]=0.0f;
	material_ambient_24[3]=1.0f;

	material_diffuse_24[0]=0.5f;
	material_diffuse_24[1]=0.5f;
	material_diffuse_24[2]=0.4f;
	material_diffuse_24[3]=1.0f;

	material_specular_24[0]=0.7f;
	material_specular_24[1]=0.7f; 
	material_specular_24[2]=0.04f; 
	material_specular_24[3]=1.0f;  
	
	material_shininess_24=0.078125f * 128.0f;


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
		"uniform int u_lighting_enabled;" \
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

	giWidth=rect.size.width;
	giHeight=rect.size.height;

	if(giHeight == 0)
		giHeight=1;

	glViewport(0,0,(GLsizei)giWidth,(GLsizei)giHeight);

	gPerspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)giWidth / (GLfloat)giHeight, 0.1f, 100.0f);

	CGLUnlockContext((CGLContextObj)[[self openGLContext] CGLContextObj]);	
}

- (void)drawRect:(NSRect)dirtyRect
{
	[self drawView];
}

- (void)drawView
{
	GLfloat fradius = 10.0f;
	[[self openGLContext]makeCurrentContext];

	CGLLockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);

	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	
	//Use Shader Program Object
	glUseProgram(gShaderProgramObject);

	if (gbLight == YES)
	{
		if (gbIsXKeyPressed == YES)
		{
			lightPosition[0] = 0.0f;
			lightPosition[1] = (GLfloat)sin(gAngle_Sphere)*fradius;
			lightPosition[2] = (GLfloat)(cos(gAngle_Sphere)*fradius - 2.0f);
		}
		else if (gbIsYKeyPressed == YES)
		{
			lightPosition[0] = (GLfloat)sin(gAngle_Sphere)*fradius;
			lightPosition[2] = (GLfloat)(cos(gAngle_Sphere)*fradius - 2.0f);
			lightPosition[1] = 0.0f;
		}
		else if (gbIsZKeyPressed == YES)
		{
			lightPosition[0] = (GLfloat)sin(gAngle_Sphere)*fradius;
			lightPosition[1] = (GLfloat)cos(gAngle_Sphere)*fradius;
			lightPosition[2] = -2.0f;
		}
		else if (gbIsXKeyPressed == NO && gbIsYKeyPressed == NO && gbIsZKeyPressed == NO)
		{
			lightPosition[0] = 0.0f;
			lightPosition[1] = 0.0f;
			lightPosition[2] = 0.0f;
		}

		glUniform1i(gLKeyPressedUniform, 1);

		glUniform3fv(gLaUniform, 1, lightAmbient);
		glUniform3fv(gLdUniform, 1, lightDiffuse);
		glUniform3fv(gLsUniform, 1, lightSpecular);
		glUniform4fv(gLightPositionUniform, 1, lightPosition);
	}
	else
	{
		glUniform1i(gLKeyPressedUniform, 0);
	}

	vmath::mat4 modelMatrix = vmath::mat4::identity();
	vmath::mat4 viewMatrix = vmath::mat4::identity();

	modelMatrix = vmath::translate(0.0f, 0.0f, -4.0f);

	glUniformMatrix4fv(gModelMatrixUniform, 1, GL_FALSE, modelMatrix);

	glUniformMatrix4fv(gViewMatrixUniform, 1, GL_FALSE, viewMatrix);

	glUniformMatrix4fv(gProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	glViewport(0, giHeight * 5 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_1];
	glViewport(0, giHeight * 4 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_2];
	glViewport(0, giHeight * 3 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_3];
	glViewport(0, giHeight * 2 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_4];
	glViewport(0, giHeight * 1 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_5];
	glViewport(0, 0 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_6];
	glViewport(giWidth / 4, giHeight * 5 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_7];
	glViewport(giWidth / 4, giHeight * 4 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_8];
	glViewport(giWidth / 4, giHeight * 3 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_9];
	glViewport(giWidth / 4, giHeight * 2 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_10];
	glViewport(giWidth / 4, giHeight * 1 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_11];
	glViewport(giWidth / 4, 0 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_12];
	glViewport(giWidth / 2, giHeight * 5 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_13];
	glViewport(giWidth / 2, giHeight * 4 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_14];
	glViewport(giWidth / 2, giHeight * 3 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_15];
	glViewport(giWidth / 2, giHeight * 2 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_16];
	glViewport(giWidth / 2, giHeight * 1 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_17];
	glViewport(giWidth / 2, 0 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_18];
	glViewport((giWidth /2) + (giWidth /4), giHeight * 5 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_19];
	glViewport((giWidth / 2) + (giWidth / 4), giHeight * 4 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_20];
	glViewport((giWidth / 2) + (giWidth / 4), giHeight * 3 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_21];
	glViewport((giWidth / 2) + (giWidth / 4), giHeight * 2 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_22];
	glViewport((giWidth / 2) + (giWidth / 4), giHeight * 1 / 6 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_23];
	glViewport((giWidth / 2) + (giWidth / 4), 0 - 20, (GLsizei)giWidth / 4, (GLsizei)giHeight / 4);
	[self Draw_Sphere_24];

	glBindVertexArray(0);

	glUseProgram(0);


	CGLFlushDrawable((CGLContextObj)[[self openGLContext]CGLContextObj]);
	CGLUnlockContext((CGLContextObj)[[self openGLContext]CGLContextObj]);

	if(gbIsAKeyPressed == YES)
	{
		gAngle_Sphere = gAngle_Sphere + 0.04f;
		if(gAngle_Sphere >= 360.0f)
			gAngle_Sphere = gAngle_Sphere - 360.0f;	
	}	
}

-(void) Draw_Sphere_1
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_1);
		glUniform3fv(gKdUniform, 1, material_diffuse_1);
		glUniform3fv(gKsUniform, 1, material_specular_1);
		glUniform1f(gMaterialShininessUniform, material_shininess_1);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_2
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_2);
		glUniform3fv(gKdUniform, 1, material_diffuse_2);
		glUniform3fv(gKsUniform, 1, material_specular_2);
		glUniform1f(gMaterialShininessUniform, material_shininess_2);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_3
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_3);
		glUniform3fv(gKdUniform, 1, material_diffuse_3);
		glUniform3fv(gKsUniform, 1, material_specular_3);
		glUniform1f(gMaterialShininessUniform, material_shininess_3);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_4
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_4);
		glUniform3fv(gKdUniform, 1, material_diffuse_4);
		glUniform3fv(gKsUniform, 1, material_specular_4);
		glUniform1f(gMaterialShininessUniform, material_shininess_4);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_5
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_5);
		glUniform3fv(gKdUniform, 1, material_diffuse_5);
		glUniform3fv(gKsUniform, 1, material_specular_5);
		glUniform1f(gMaterialShininessUniform, material_shininess_5);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_6
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_6);
		glUniform3fv(gKdUniform, 1, material_diffuse_6);
		glUniform3fv(gKsUniform, 1, material_specular_6);
		glUniform1f(gMaterialShininessUniform, material_shininess_6);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_7
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_7);
		glUniform3fv(gKdUniform, 1, material_diffuse_7);
		glUniform3fv(gKsUniform, 1, material_specular_7);
		glUniform1f(gMaterialShininessUniform, material_shininess_7);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_8
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_8);
		glUniform3fv(gKdUniform, 1, material_diffuse_8);
		glUniform3fv(gKsUniform, 1, material_specular_8);
		glUniform1f(gMaterialShininessUniform, material_shininess_8);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_9
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_9);
		glUniform3fv(gKdUniform, 1, material_diffuse_9);
		glUniform3fv(gKsUniform, 1, material_specular_9);
		glUniform1f(gMaterialShininessUniform, material_shininess_9);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_10
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_10);
		glUniform3fv(gKdUniform, 1, material_diffuse_10);
		glUniform3fv(gKsUniform, 1, material_specular_10);
		glUniform1f(gMaterialShininessUniform, material_shininess_10);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_11
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_11);
		glUniform3fv(gKdUniform, 1, material_diffuse_11);
		glUniform3fv(gKsUniform, 1, material_specular_11);
		glUniform1f(gMaterialShininessUniform, material_shininess_11);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}


-(void) Draw_Sphere_12
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_12);
		glUniform3fv(gKdUniform, 1, material_diffuse_12);
		glUniform3fv(gKsUniform, 1, material_specular_12);
		glUniform1f(gMaterialShininessUniform, material_shininess_12);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_13
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_13);
		glUniform3fv(gKdUniform, 1, material_diffuse_13);
		glUniform3fv(gKsUniform, 1, material_specular_13);
		glUniform1f(gMaterialShininessUniform, material_shininess_13);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_14
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_14);
		glUniform3fv(gKdUniform, 1, material_diffuse_14);
		glUniform3fv(gKsUniform, 1, material_specular_14);
		glUniform1f(gMaterialShininessUniform, material_shininess_14);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}


-(void) Draw_Sphere_15
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_15);
		glUniform3fv(gKdUniform, 1, material_diffuse_15);
		glUniform3fv(gKsUniform, 1, material_specular_15);
		glUniform1f(gMaterialShininessUniform, material_shininess_15);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_16
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_16);
		glUniform3fv(gKdUniform, 1, material_diffuse_16);
		glUniform3fv(gKsUniform, 1, material_specular_16);
		glUniform1f(gMaterialShininessUniform, material_shininess_16);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_17
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_17);
		glUniform3fv(gKdUniform, 1, material_diffuse_17);
		glUniform3fv(gKsUniform, 1, material_specular_17);
		glUniform1f(gMaterialShininessUniform, material_shininess_17);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_18
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_18);
		glUniform3fv(gKdUniform, 1, material_diffuse_18);
		glUniform3fv(gKsUniform, 1, material_specular_18);
		glUniform1f(gMaterialShininessUniform, material_shininess_18);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_19
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_19);
		glUniform3fv(gKdUniform, 1, material_diffuse_19);
		glUniform3fv(gKsUniform, 1, material_specular_19);
		glUniform1f(gMaterialShininessUniform, material_shininess_19);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_20
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_20);
		glUniform3fv(gKdUniform, 1, material_diffuse_20);
		glUniform3fv(gKsUniform, 1, material_specular_20);
		glUniform1f(gMaterialShininessUniform, material_shininess_20);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_21
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_21);
		glUniform3fv(gKdUniform, 1, material_diffuse_21);
		glUniform3fv(gKsUniform, 1, material_specular_21);
		glUniform1f(gMaterialShininessUniform, material_shininess_21);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_22
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_22);
		glUniform3fv(gKdUniform, 1, material_diffuse_22);
		glUniform3fv(gKsUniform, 1, material_specular_22);
		glUniform1f(gMaterialShininessUniform, material_shininess_22);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_23
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_23);
		glUniform3fv(gKdUniform, 1, material_diffuse_23);
		glUniform3fv(gKsUniform, 1, material_specular_23);
		glUniform1f(gMaterialShininessUniform, material_shininess_23);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
}

-(void) Draw_Sphere_24
{
	if (gbLight == YES)
	{
		glUniform3fv(gKaUniform, 1, material_ambient_24);
		glUniform3fv(gKdUniform, 1, material_diffuse_24);
		glUniform3fv(gKsUniform, 1, material_specular_24);
		glUniform1f(gMaterialShininessUniform, material_shininess_24);
	}

	glBindVertexArray(gVao_Sphere);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
	glDrawElements(GL_TRIANGLES, gNumElements, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVbo_Elements);
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

		case 'X':
		case 'x':
			if (gbIsXKeyPressed == NO)
			{
				gbIsXKeyPressed = YES;
				gbIsYKeyPressed = NO;
				gbIsZKeyPressed = NO;
			}
			else
				gbIsXKeyPressed = NO;
			break;

		case 'Y':
		case 'y':
			if (gbIsYKeyPressed == NO)
			{
				gbIsXKeyPressed = NO;
				gbIsYKeyPressed = YES;
				gbIsZKeyPressed = NO;
			}
			else
				gbIsYKeyPressed = NO;
			break;

		case 'Z':
		case 'z':
			if (gbIsZKeyPressed == NO)
			{
				gbIsXKeyPressed = NO;
				gbIsYKeyPressed = NO;
				gbIsZKeyPressed = YES;
			}
			else
				gbIsZKeyPressed = NO;
			break;
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
