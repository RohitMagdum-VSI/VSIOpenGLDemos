#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<memory.h>

#include<X11/Xlib.h>
#include<X11/Xutil.h>
#include<X11/XKBlib.h>
#include<X11/keysym.h>

#include<GL/glew.h>
#include<GL/gl.h>
#include<GL/glx.h>


#include<AL/alc.h>
#include<AL/alut.h>

#include"vmath.h"
#include"01-Shape.h"



using namespace vmath;

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

enum {
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_TEXCOORD0,
	AMC_ATTRIBUTE_NORMAL
};


//For Window
Display *gpDisplay_RRJ = NULL;
Colormap gColormap_RRJ;
Window gWindow_RRJ;
XVisualInfo *gpXVisualInfo_RRJ = NULL;


//For Keys
char keys[26];


//For FullScreen
bool bIsFullScreen_RRJ = false;

//For Error
FILE *gbFile_RRJ = NULL;

//For OpenGL
GLXContext gGLXContext_RRJ;
GLXFBConfig gGLXFBConfig_RRJ;
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
glXCreateContextAttribsARBProc glXCreateContextAttribsARB_RRJ = NULL;

//For Shader
GLuint shaderProgramObject_PV_RRJ;
GLuint shaderProgramObject_PF_RRJ;

//For Projection
mat4 gPerspectiveProjectionMatrix_RRJ;



#define STACK 20
#define SLICES 20


//For Sphere
GLuint vao_Sphere_RRJ;
GLuint vbo_Sphere_Position_RRJ;
GLuint vbo_Sphere_Normal_RRJ;
GLuint vbo_Sphere_Element_RRJ;

GLfloat sphere_Pos_RRJ[(STACK + 1) * (SLICES) * 3]; 
GLfloat sphere_Nor_RRJ[(STACK + 1) * (SLICES) * 3];
GLfloat sphere_Texcoord_RRJ[(STACK + 1) * (SLICES) * 2];
unsigned short sphere_Element_RRJ[(STACK + 1) * (SLICES) * 6];
GLuint numOfSphereElements_RRJ;


//For SemiSphere
GLuint vao_SemiSphere_RRJ;
GLuint vbo_SemiSphere_Pos_RRJ;
GLuint vbo_SemiSphere_Tex_RRJ;
GLuint vbo_SemiSphere_Nor_RRJ;
GLuint vbo_SemiSphere_Elem_RRJ;

GLfloat semiSphere_Pos_RRJ[(STACK + 1) * (SLICES) * 3]; 
GLfloat semiSphere_Nor_RRJ[(STACK + 1) * (SLICES) * 3];
GLfloat semiSphere_Texcoord_RRJ[(STACK + 1) * (SLICES) * 2];
unsigned short semiSphere_Element_RRJ[(STACK + 1) * (SLICES) * 6];
GLuint numOfSemisphereElements_RRJ;


//For Cylinder
GLuint vao_Cylinder_RRJ;
GLuint vbo_Cylinder_Pos_RRJ;
GLuint vbo_Cylinder_Nor_RRJ;
GLuint vbo_Cylinder_Elem_RRJ;

GLfloat cylinder_Pos_RRJ[4 * SLICES * 3];
GLfloat cylinder_Nor_RRJ[4 * SLICES * 3];
unsigned short cylinder_Element_RRJ[3 * SLICES * 6];
int numOfCylinderElements_RRJ;


//For Frustum
GLuint vao_Frustum_RRJ;
GLuint vbo_Frustum_Pos_RRJ;
GLuint vbo_Frustum_Nor_RRJ;
GLuint vbo_Frustum_Elem_RRJ;

GLfloat frustum_Pos_RRJ[4 * SLICES * 3];
GLfloat frustum_Nor_RRJ[4 * SLICES * 3];
unsigned short frustum_Element_RRJ[3 * SLICES * 6];
int numOfFrustumElements_RRJ;




//For Uniform Per Vertex Lighting
GLuint modelViewMatrixUniform_PV_RRJ;
GLuint projectionMatrixUniform_PV_RRJ;
GLuint LaUniform_PV_RRJ;
GLuint LdUniform_PV_RRJ;
GLuint LsUniform_PV_RRJ;
GLuint lightPositionUniform_PV_RRJ;
GLuint KaUniform_PV_RRJ;
GLuint KdUniform_PV_RRJ;
GLuint KsUniform_PV_RRJ;
GLuint shininessUniform_PV_RRJ;
GLuint LKeyPressUniform_PV_RRJ;
GLuint blendingUniform_PV_RRJ;


//For Uniform Per Fragment Lighting
GLuint modelViewMatrixUniform_PF_RRJ;
GLuint projectionMatrixUniform_PF_RRJ;
GLuint LaUniform_PF_RRJ;
GLuint LdUniform_PF_RRJ;
GLuint LsUniform_PF_RRJ;
GLuint lightPositionUniform_PF_RRJ;
GLuint KaUniform_PF_RRJ;
GLuint KdUniform_PF_RRJ;
GLuint KsUniform_PF_RRJ;
GLuint shininessUniform_PF_RRJ;
GLuint LKeyPressUniform_PF_RRJ;
GLuint blendingUniform_PF_RRJ;


//For Light
#define PER_VERTEX 1
#define PER_FRAGMENT 2

int iWhichLight_RRJ = PER_VERTEX;
bool bLight_RRJ = false;
GLfloat lightAmbient_RRJ[] = {0.0f, 0.0f, 0.0f, 0.0f};
GLfloat lightDiffuse_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
GLfloat lightSpecular_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
GLfloat lightPosition_RRJ[] = {0.0f, 0.0f, 2.0f, 1.0f};

//For Material
GLfloat materialAmbient_RRJ[] = {0.0f, 0.0f, 0.0f, 0.0f};
GLfloat materialDiffuse_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
GLfloat materialSpecular_RRJ[] = {1.0f, 1.0f, 1.0f, 1.0f};
GLfloat materialShininess_RRJ = 50.0f;





//For Movement
enum{
	RRJ_ASSEMBLY = 1,
	RRJ_WALKING,
	RRJ_RUNNING,
	RRJ_SWIMMING,
	RRJ_SQUARTING,
	RRJ_SKIPING,
	RRJ_NONE
};

GLuint iWhichMovement = RRJ_NONE;

int iMoveFlag = 0;


//For Body
GLfloat chest_RRJ = 0.0f;
GLfloat hip_RRJ = 0.0;


//ARMS
GLfloat shoulderLeft_RRJ = 0.0f;
GLfloat shoulderRight_RRJ = 0.0f;
GLfloat elbowLeft_RRJ = 0.0f;
GLfloat elbowRight_RRJ = 0.0f;

//LEGS
GLfloat thighLeft_RRJ = 0.0f;
GLfloat thighRight_RRJ = 0.0f;
GLfloat shinBoneLeft_RRJ = 0.0f;
GLfloat shinBoneRight_RRJ = 0.0f;


//For Body
GLfloat body_XTranslation = 0.0f;
GLfloat body_YTranslation = 0.0f;
GLfloat body_ZTranslation = 0.0f;
GLfloat body_YRot_Angle = 180.0f;

const GLfloat MAX_SHOULDER_MOVEMENT_WALKING = 40.0f;
const GLfloat MAX_SHOULDER_MOVEMENT_RUNNING = 65.0f;
const GLfloat MAX_THIGH_MOVEMENT_SWIMMING = 30.0f;
const GLfloat MAX_ELBOW_MOVEMENT_SWIMMING = -40.0f;
const GLfloat MAX_THIGH_SHINBONE_MOVEMENT_SQUARTING = 90.0f;




//For ModelViewMatrix Stack
struct Stack{
	mat4 ModelViewMatrix;
	struct Stack *next;
	struct Stack *prev;
};

typedef struct Stack MVStack;
MVStack *TopNode = NULL;
const GLint maxStackSize = 32;


void my_glPushMatrix(mat4);
mat4 my_glPopMatrix(void);




//For Animation
bool isDemoMode = false;
bool isGameMode = false;
bool isAnimating = false;
GLint iScene = 1;
GLfloat gTimePerFrame = 5.0f;
GLfloat fLeftArm = -30.0f;
GLfloat fRightArm = 30.0f;
GLfloat fLeftLeg = -50.0f;
GLfloat fRightLeg = -50.0f;
GLfloat fBlendingFactor = 0.0f;





//For Sound
ALCdevice *pALCdevice = NULL;
ALCcontext *pALCcontext = NULL;
ALenum error;

ALuint gBuffer_Start;
ALuint gBuffer_Walk_Running;
ALuint gBuffer_Squarting_Skipping;
ALuint gBuffer_Swimming;
ALuint gBuffer_End;
ALuint gBuffer_GameMode;


ALuint gSource_Start;
ALuint gSource_Walk_Running;
ALuint gSource_Squarting_Skipping;
ALuint gSource_Swimming;
ALuint gSource_End;
ALuint gSource_GameMode;





int main(void){
	
	void CreateWindow(void);
	void initialize(void);
	void ToggleFullScreen(void);
	void resize(int, int);
	void display(void);
	void uninitialize(void);
	void update(void);
	float degToRad(float);

	void initializeSound(void);

	void BodyWalking(float);
	void BodyRunning(float);

	void StopAllSoundSources(void);


	int winWidth_RRJ = WIN_WIDTH;
	int winHeight_RRJ = WIN_HEIGHT;


	gbFile_RRJ = fopen("Log.txt", "w");
	if(gbFile_RRJ == NULL){
		printf("Log  Creation Failed!!\n");
		uninitialize();
		exit(1);
	}
	else
		fprintf(gbFile_RRJ, "Log Created!!\n");


	CreateWindow();
	initialize();
	initializeSound();
	ToggleFullScreen();

	//For Event Loop
	XEvent event_RRJ;
	KeySym keysym_RRJ;
	bool bDone_RRJ = false;

	while(bDone_RRJ == false){
		while(XPending(gpDisplay_RRJ)){
	
			XNextEvent(gpDisplay_RRJ, &event_RRJ);
			switch(event_RRJ.type){
				case MapNotify:
					break;
				case Expose:
					break;
				case MotionNotify:
					break;
				case DestroyNotify:
					break;

				case ConfigureNotify:
					winWidth_RRJ = event_RRJ.xconfigure.width;
					winHeight_RRJ = event_RRJ.xconfigure.height;
					resize(winWidth_RRJ, winHeight_RRJ);
					break;

				case KeyPress:
					keysym_RRJ = XkbKeycodeToKeysym(gpDisplay_RRJ, event_RRJ.xkey.keycode, 0, 0);
					switch(keysym_RRJ){
						case XK_Q:
						case XK_q:
							bDone_RRJ = true;
							break;
						

						case XK_Escape:
							if(bIsFullScreen_RRJ == false){
								ToggleFullScreen();
								bIsFullScreen_RRJ = true;
							}
							else{
								ToggleFullScreen();
								bIsFullScreen_RRJ = false;
							}
							break;



						case XK_A:
						case XK_a:
							if(isAnimating == false)
								isAnimating = true;
							else
								isAnimating = false;
							break;



						case XK_L:
						case XK_l:
							if(bLight_RRJ == false)
								bLight_RRJ = true;
							else
								bLight_RRJ = false;
							break;



						case XK_V:
						case XK_v:
							iWhichLight_RRJ = PER_VERTEX;
							break;


						case XK_F:
						case XK_f:
							iWhichLight_RRJ = PER_FRAGMENT;
							break;



						case XK_T:
						case XK_t:
							body_YRot_Angle = body_YRot_Angle + 1.0f;
							break;

						case XK_D:
						case XK_d:
							isDemoMode = true;
							isGameMode = false;


							StopAllSoundSources();
							alSourcePlay(gSource_Start);
							
							fLeftArm = -30.0f;
							fRightArm = 30.0f;
							fLeftLeg = -50.0f;
							fRightLeg = -50.0f;

							fBlendingFactor = 0.0f;

							chest_RRJ = 0.0f;
							hip_RRJ = 0.0f;
							body_XTranslation = 0.0;
							body_YTranslation = 0.0f;
							body_ZTranslation = 0.0f;
							body_YRot_Angle = 180.0f;
							shoulderLeft_RRJ = 0.0f;
							shoulderRight_RRJ = 0.0f;
							elbowLeft_RRJ = 0.0f;
							elbowRight_RRJ = 0.0f;
							thighLeft_RRJ = 0.0f;
							thighRight_RRJ = 0.0f;
							shinBoneLeft_RRJ = 0.0f;
							shinBoneRight_RRJ = 0.0f;							

							break;


						case XK_G:
						case XK_g:
							isDemoMode = false;
							isGameMode = true;

							StopAllSoundSources();
							alSourcePlay(gSource_GameMode);

							iWhichMovement = RRJ_NONE;

							fLeftArm = 0.0f;
							fRightArm = 0.0f;
							fLeftLeg = 0.0f;
							fRightLeg = 0.0f;

							fBlendingFactor = 1.0f;

							chest_RRJ = 0.0f;
							hip_RRJ = 0.0f;
							body_XTranslation = 0.0;
							body_YTranslation = 0.0f;
							body_ZTranslation = 0.0f;
							body_YRot_Angle = 180.0f;
							shoulderLeft_RRJ = 0.0f;
							shoulderRight_RRJ = 0.0f;
							elbowLeft_RRJ = 0.0f;
							elbowRight_RRJ = 0.0f;
							thighLeft_RRJ = 0.0f;
							thighRight_RRJ = 0.0f;
							shinBoneLeft_RRJ = 0.0f;
							shinBoneRight_RRJ = 0.0f;


							break;



						case XK_Up:
							if(iWhichMovement == RRJ_WALKING)
								BodyWalking(body_YRot_Angle);
							else if(iWhichMovement == RRJ_RUNNING)
								BodyRunning(body_YRot_Angle);
							break;


						case XK_Left:
							body_YRot_Angle = body_YRot_Angle + 90.0f;
							break;

						case XK_Right: 
							body_YRot_Angle = body_YRot_Angle - 90.0f;
							break;

						case XK_Down:
							body_YRot_Angle = body_YRot_Angle + 180.0f;
							break;


						default:
							break;
					}
					XLookupString(&event_RRJ.xkey, keys, sizeof(keys), NULL, NULL);
					switch(keys[0]){

						case 'R':
						case 'r':
							iWhichMovement = RRJ_RUNNING;
							chest_RRJ = -5.0f;
							hip_RRJ = -5.0f;


							body_XTranslation = 0.0f;
							body_ZTranslation = 0.0f;
							body_YTranslation = 0.0f;
							body_YRot_Angle = 150.0f;

							shoulderLeft_RRJ = 0.0f;
							shoulderRight_RRJ = 0.0f;
							elbowLeft_RRJ = 0.0f;
							elbowRight_RRJ = 0.0f;
							thighRight_RRJ = 0.0f;
							thighLeft_RRJ = 0.0f;
							shinBoneLeft_RRJ = 0.0f;
							shinBoneRight_RRJ = 0.0f;
							iMoveFlag = 0;
							break;

						case 'W':
						case 'w':
							iWhichMovement = RRJ_WALKING;
							chest_RRJ = 0.0f;
							hip_RRJ = 0.0f;

							body_XTranslation = 0.0f;
							body_YTranslation = 0.0f;
							body_ZTranslation = 0.0f;
							body_YRot_Angle = 170.0f;

							shoulderLeft_RRJ = 0.0f;
							shoulderRight_RRJ = 0.0f;
							elbowLeft_RRJ = 0.0f;
							elbowRight_RRJ = 0.0f;
							thighRight_RRJ = 0.0f;
							thighLeft_RRJ = 0.0f;
							shinBoneLeft_RRJ = 0.0f;
							shinBoneRight_RRJ = 0.0f;
							iMoveFlag = 0;

							break;


						case 'M':
						case 'm':
							iWhichMovement = RRJ_SWIMMING;
							chest_RRJ = -90.0f;
							hip_RRJ = -90.0f;

							body_YRot_Angle = 90.0f;

							body_XTranslation = 0.0f;
							body_ZTranslation = 0.0f;
							body_YTranslation = 0.0f;

							shoulderLeft_RRJ = 180.0f;
							shoulderRight_RRJ = 0.0f;
							elbowLeft_RRJ = -40.0f;
							elbowRight_RRJ = -40.0f;
							thighRight_RRJ = 0.0f;
							thighLeft_RRJ = 0.0f;
							shinBoneLeft_RRJ = -30.0f;
							shinBoneRight_RRJ = -30.0f;
							iMoveFlag = 0;

							body_YTranslation = 0.0f;
							break;


						case 'S':
						case 's':
							iWhichMovement = RRJ_SQUARTING;
							
							chest_RRJ = 0.0f;
							hip_RRJ = 0.0f;

							body_XTranslation = 0.0f;
							body_ZTranslation = 0.0f;
							body_YTranslation = 0.0f;

							body_YTranslation = 0.0f;
							body_YRot_Angle = 0.0f;

							shoulderLeft_RRJ = 90.0f;
							shoulderRight_RRJ = 90.0f;
							elbowLeft_RRJ = 0.0f;
							elbowRight_RRJ = 0.0f;

							thighRight_RRJ = 0.0f;
							thighLeft_RRJ = 0.0f;
							shinBoneLeft_RRJ = 0.0f;
							shinBoneRight_RRJ = 0.0f;
							iMoveFlag = 0;
							break;


						case 'K':
						case 'k':
							iWhichMovement = RRJ_SKIPING;

							body_XTranslation = 0.0f;
							body_ZTranslation = 0.0f;
							body_YTranslation = 0.0f;

							body_YRot_Angle = 90.0f;

							chest_RRJ = 0.0f;
							hip_RRJ = 0.0f;

							iMoveFlag = 0;

							shoulderLeft_RRJ = 0.0f;
							shoulderRight_RRJ = 0.0f;
							elbowLeft_RRJ = 0.0f;
							elbowRight_RRJ = 0.0f;

							thighRight_RRJ = 0.0f;
							thighLeft_RRJ = 0.0f;
							shinBoneLeft_RRJ = 0.0f;
							shinBoneRight_RRJ = 0.0f;
							iMoveFlag = 0;
							break;
					}


					break;

				case ButtonPress:
					switch(event_RRJ.xbutton.button){
						case 1:
							break;
						case 2:
							break;
						case 3:
							break;
						default:
							break;
					}
					break;

				case 33:
					bDone_RRJ = true;
					break;

				default:
					break;
			}

		}
		display();
		if(isAnimating == true)
			update();
	}

	uninitialize();
	return(0);
}



void StopAllSoundSources(void){

	alSourceStop(gSource_GameMode);
	alSourceStop(gSource_Start);
	alSourceStop(gSource_Walk_Running);
	alSourceStop(gSource_Squarting_Skipping);
	alSourceStop(gSource_Swimming);
	alSourceStop(gSource_End);

}




void initializeSound(void){

	void uninitialize(void);

		
	/********** DEVICE **********/	
	pALCdevice = alcOpenDevice(NULL);
	if(pALCdevice == NULL){
		fprintf(gbFile_RRJ, "ERROR: alcOpenDevice()\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alcOpenDevice()\n");



	/********** CONTEXT **********/
	pALCcontext = alcCreateContext(pALCdevice, NULL);
	if(pALCcontext == NULL){
		fprintf(gbFile_RRJ, "ERROR: alcCreateContext()\n");
		uninitialize();
		exit(0);
	}
	else{
		fprintf(gbFile_RRJ, "SUCCESS: alcCreateContext()\n");
		alcMakeContextCurrent(pALCcontext);
		if((error = alGetError()) != AL_NO_ERROR){
			fprintf(gbFile_RRJ, "ERROR: alcMakeContextCurrent()\n");
			uninitialize();
			exit(0);
		}
		else
			fprintf(gbFile_RRJ, "SUCCESS: alcMakeContextCurrent()\n");
	}








	/*********** LOADING STARTING WAV ***********/
	ALbyte *fileName = (ALbyte*)"Start.wav";
	ALenum format;
	void *data = NULL;
	ALsizei size;
	ALsizei freq;
	ALboolean isLoop;

	alutLoadWAVFile(fileName, &format, &data, &size, &freq, &isLoop);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alutLoadWAVFile() Walking and Running\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alutLoadWAVFile() Walking and Running\n");




	/********** BUFFER **********/
	alGenBuffers(1, &gBuffer_Start);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alGenBuffers() Start\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alGenBuffers() Start\n");


	alBufferData(gBuffer_Start, format, data, size, freq);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alBufferData() Start\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alBufferData() Start\n");



	/********** UNLOADING WAV **********/
	alutUnloadWAV(format, data, size, freq);







	/*********** LOADING WALKING AND RUNNING WAV ***********/
	fileName = (ALbyte*)"W&R1.wav";
	format = 0;
	data = NULL;
	size = 0;
	freq = 0;
	isLoop = 0;

	alutLoadWAVFile(fileName, &format, &data, &size, &freq, &isLoop);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alutLoadWAVFile() Walking and Running\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alutLoadWAVFile() Walking and Running\n");




	/********** BUFFER **********/
	alGenBuffers(1, &gBuffer_Walk_Running);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alGenBuffers() Walking and Running\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alGenBuffers() Walking and Running\n");


	alBufferData(gBuffer_Walk_Running, format, data, size, freq);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alBufferData() Walking and Running\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alBufferData() Walking and Running\n");



	/********** UNLOADING WAV **********/
	alutUnloadWAV(format, data, size, freq);










	/*********** LOADING SQUARTING AND SKIPPING WAV ***********/
	fileName = (ALbyte*)"S&S.wav";
	format = 0;
	data = NULL;
	size = 0;
	freq = 0;
	isLoop = 0;

	alutLoadWAVFile(fileName, &format, &data, &size, &freq, &isLoop);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alutLoadWAVFile() Squarting and Skipping\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alutLoadWAVFile() Squarting and Skipping\n");




	/********** BUFFER **********/
	alGenBuffers(1, &gBuffer_Squarting_Skipping);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alGenBuffers() Squarting and Skipping\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alGenBuffers() Squarting and Skipping\n");


	alBufferData(gBuffer_Squarting_Skipping, format, data, size, freq);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alBufferData() Squarting and Skipping\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alBufferData() Squarting and Skipping\n");



	/********** UNLOADING WAV **********/
	alutUnloadWAV(format, data, size, freq);









	/*********** LOADING SWIMMING WAV ***********/
	fileName = (ALbyte*)"S1.wav";
	format = 0;
	data = NULL;
	size = 0;
	freq = 0;
	isLoop = 0;

	alutLoadWAVFile(fileName, &format, &data, &size, &freq, &isLoop);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alutLoadWAVFile() Swimming\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alutLoadWAVFile() Swimming\n");




	/********** BUFFER **********/
	alGenBuffers(1, &gBuffer_Swimming);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alGenBuffers() Swimming\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alGenBuffers() Swimming\n");


	alBufferData(gBuffer_Swimming, format, data, size, freq);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alBufferData() Swimming\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alBufferData() Swimming\n");



	/********** UNLOADING WAV **********/
	alutUnloadWAV(format, data, size, freq);








	/*********** LOADING END WAV ***********/
	fileName = (ALbyte*)"E1.wav";
	format = 0;
	data = NULL;
	size = 0;
	freq = 0;
	isLoop = 0;

	alutLoadWAVFile(fileName, &format, &data, &size, &freq, &isLoop);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alutLoadWAVFile() End\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alutLoadWAVFile() End\n");




	/********** BUFFER **********/
	alGenBuffers(1, &gBuffer_End);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alGenBuffers() End\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alGenBuffers() End\n");


	alBufferData(gBuffer_End, format, data, size, freq);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alBufferData() End\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alBufferData() End\n");



	/********** UNLOADING WAV **********/
	alutUnloadWAV(format, data, size, freq);








	/*********** LOADING GAME MODE WAV ***********/
	fileName = (ALbyte*)"GM.wav";
	format = 0;
	data = NULL;
	size = 0;
	freq = 0;
	isLoop = 0;

	alutLoadWAVFile(fileName, &format, &data, &size, &freq, &isLoop);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alutLoadWAVFile() Game Mode\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alutLoadWAVFile() Game Mode\n");




	/********** BUFFER **********/
	alGenBuffers(1, &gBuffer_GameMode);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alGenBuffers() Game Mode\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alGenBuffers() Game Mode\n");


	alBufferData(gBuffer_GameMode, format, data, size, freq);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alBufferData() Game Mode\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alBufferData() Game Mode\n");



	/********** UNLOADING WAV **********/
	alutUnloadWAV(format, data, size, freq);








	/********** START SOURCE **********/
	alGenSources(1, &gSource_Start);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alGenSources() Start\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alGenSources() Start\n");






	/********** SOURCE WALKING AND RUNNING **********/
	alGenSources(1, &gSource_Walk_Running);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alGenSources() Walk_Running\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alGenSources() Walk_Running\n");




	/********** SKIPPING AND SQUARTING SOURCE **********/
	alGenSources(1, &gSource_Squarting_Skipping);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alGenSources() Squarting_Skipping\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alGenSources() Squarting_Skipping\n");




	/********** SWIMMING SOURCE **********/
	alGenSources(1, &gSource_Swimming);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alGenSources() Swimming\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alGenSources() Swimming\n");




	/********** END SOURCE **********/
	alGenSources(1, &gSource_End);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alGenSources() End\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alGenSources() End\n");




	/********** GAME MODE SOURCE **********/
	alGenSources(1, &gSource_GameMode);
	if((error = alGetError()) != AL_NO_ERROR){
		fprintf(gbFile_RRJ, "ERROR: alGenSources() Game Mode\n");
		uninitialize();
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: alGenSources() Game Mode\n");



	alSourcei(gSource_Start, AL_BUFFER, gBuffer_Start);
	alSourcei(gSource_Walk_Running, AL_BUFFER, gBuffer_Walk_Running);
	alSourcei(gSource_Squarting_Skipping, AL_BUFFER, gBuffer_Squarting_Skipping);
	alSourcei(gSource_Swimming, AL_BUFFER, gBuffer_Swimming);
	alSourcei(gSource_End, AL_BUFFER, gBuffer_End);
	alSourcei(gSource_GameMode, AL_BUFFER, gBuffer_GameMode);


}






float degToRad(float angle){
	return(angle * (3.1415926535f / 180.0f));
}



void ToggleFullScreen(void){
	
	Atom wm_state_RRJ;
	Atom fullscreen_RRJ;
	XEvent xev_RRJ = {0};

	wm_state_RRJ = XInternAtom(gpDisplay_RRJ, "_NET_WM_STATE", False);
	memset(&xev_RRJ, 0, sizeof(XEvent));

	xev_RRJ.type = ClientMessage;
	xev_RRJ.xclient.window = gWindow_RRJ;
	xev_RRJ.xclient.message_type = wm_state_RRJ;
	xev_RRJ.xclient.format = 32;
	xev_RRJ.xclient.data.l[0] = bIsFullScreen_RRJ ? 0 : 1;
	
	fullscreen_RRJ = XInternAtom(gpDisplay_RRJ, "_NET_WM_STATE_FULLSCREEN", False);
	xev_RRJ.xclient.data.l[1] = fullscreen_RRJ;

	XSendEvent(gpDisplay_RRJ,
		RootWindow(gpDisplay_RRJ, gpXVisualInfo_RRJ->screen),
		False,
		StructureNotifyMask,
		&xev_RRJ);
}

void CreateWindow(void){

	void uninitialize(void);

	XSetWindowAttributes winAttrib_RRJ;
	int defaultScreen_RRJ;
	int styleMask_RRJ;

	GLXFBConfig *pGLXFBConfig_RRJ = NULL;
	GLXFBConfig bestFBConfig_RRJ;
	int iNumberOfFBConfig_RRJ = 0;
	XVisualInfo *pTempXVisualInfo_RRJ = NULL;


	static int frameBufferAttributes_RRJ[] = {
		GLX_X_RENDERABLE, True,
		GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
		GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
		GLX_RENDER_TYPE, GLX_RGBA_BIT,
		GLX_RED_SIZE, 8,
		GLX_GREEN_SIZE, 8,
		GLX_BLUE_SIZE, 8,
		GLX_ALPHA_SIZE, 8,
		GLX_DEPTH_SIZE, 24,
		GLX_DOUBLEBUFFER, True,
		None
	};

	gpDisplay_RRJ = XOpenDisplay(NULL);
	if(gpDisplay_RRJ == NULL){
		fprintf(gbFile_RRJ, "XOpenDisplay() Failed!!\n");
		uninitialize();
		exit(1);
	}

	defaultScreen_RRJ = XDefaultScreen(gpDisplay_RRJ);

	pGLXFBConfig_RRJ = glXChooseFBConfig(gpDisplay_RRJ, defaultScreen_RRJ, frameBufferAttributes_RRJ, &iNumberOfFBConfig_RRJ);
	if(pGLXFBConfig_RRJ == NULL){
		fprintf(gbFile_RRJ, "glXChooseFBConfig() Failed!!\n");
		uninitialize();
		exit(1);
	}

	
	int bestFrameBufferConfig_RRJ = -1;
	int bestNumberOfSamples_RRJ = -1;
	int worstFrameBufferConfig_RRJ = -1;
	int worstNumberOfSamples_RRJ = -1;


	fprintf(gbFile_RRJ, "FBConfig: %d\n", iNumberOfFBConfig_RRJ);

	for(int i = 0; i < iNumberOfFBConfig_RRJ; i++){
		pTempXVisualInfo_RRJ = glXGetVisualFromFBConfig(gpDisplay_RRJ, pGLXFBConfig_RRJ[i]);
		if(pTempXVisualInfo_RRJ){
			int samples, sampleBuffers;

			glXGetFBConfigAttrib(gpDisplay_RRJ, pGLXFBConfig_RRJ[i], GLX_SAMPLES, &samples);
			glXGetFBConfigAttrib(gpDisplay_RRJ, pGLXFBConfig_RRJ[i], GLX_SAMPLE_BUFFERS, &sampleBuffers);

			if(bestFrameBufferConfig_RRJ < 0 || sampleBuffers && samples > bestNumberOfSamples_RRJ){
				bestFrameBufferConfig_RRJ = i;
				bestNumberOfSamples_RRJ = samples;
			}

			if(worstFrameBufferConfig_RRJ < 0 || sampleBuffers && samples < worstNumberOfSamples_RRJ){
				worstFrameBufferConfig_RRJ = i;
				worstNumberOfSamples_RRJ = samples;
			}
		}
		XFree(pTempXVisualInfo_RRJ);
		pTempXVisualInfo_RRJ = NULL;
	}

	bestFBConfig_RRJ = pGLXFBConfig_RRJ[bestFrameBufferConfig_RRJ];
	gGLXFBConfig_RRJ = bestFBConfig_RRJ;
	XFree(pGLXFBConfig_RRJ);
	pGLXFBConfig_RRJ = NULL;

	gpXVisualInfo_RRJ = glXGetVisualFromFBConfig(gpDisplay_RRJ, bestFBConfig_RRJ);
	if(gpXVisualInfo_RRJ == NULL){
		fprintf(gbFile_RRJ, "glXGetVisualFromFBConfig() Failed!!\n");
		uninitialize();
		exit(1);
	}

	winAttrib_RRJ.border_pixel = 0;
	winAttrib_RRJ.border_pixmap = 0;
	winAttrib_RRJ.background_pixel = BlackPixel(gpDisplay_RRJ, defaultScreen_RRJ);
	winAttrib_RRJ.background_pixmap = 0;
	winAttrib_RRJ.colormap = XCreateColormap(gpDisplay_RRJ,
				RootWindow(gpDisplay_RRJ, gpXVisualInfo_RRJ->screen),
				gpXVisualInfo_RRJ->visual,
				AllocNone);
	gColormap_RRJ = winAttrib_RRJ.colormap;
	winAttrib_RRJ.event_mask = ExposureMask | VisibilityChangeMask | PointerMotionMask |
				KeyPressMask | ButtonPressMask | StructureNotifyMask;

	styleMask_RRJ = CWBorderPixel | CWBackPixel | CWEventMask | CWColormap;

	gWindow_RRJ = XCreateWindow(gpDisplay_RRJ,
			RootWindow(gpDisplay_RRJ, gpXVisualInfo_RRJ->screen),
			0, 0,
			WIN_WIDTH, WIN_HEIGHT,
			0,
			gpXVisualInfo_RRJ->depth,
			InputOutput,
			gpXVisualInfo_RRJ->visual,
			styleMask_RRJ,
			&winAttrib_RRJ);

	if(!gWindow_RRJ){
		fprintf(gbFile_RRJ, "XCreateWindow() Failed!!\n");
		uninitialize();
		exit(1);
	}
	
	XStoreName(gpDisplay_RRJ, gWindow_RRJ, "09-Final");

	Atom windowManagerDelete = XInternAtom(gpDisplay_RRJ, "WM_DELETE_WINDOW", True);
	XSetWMProtocols(gpDisplay_RRJ, gWindow_RRJ, &windowManagerDelete, 1);

	XMapWindow(gpDisplay_RRJ, gWindow_RRJ);
}

void initialize(void){
	
	void uninitialize(void);
	void resize(int, int);
	


	//Shader Object;
	GLuint vertexShaderObject_PV_RRJ;
	GLuint fragmentShaderObject_PV_RRJ;

	GLuint vertexShaderObject_PF_RRJ;
	GLuint fragmentShaderObject_PF_RRJ;


	
	glXCreateContextAttribsARB_RRJ = (glXCreateContextAttribsARBProc)glXGetProcAddressARB((GLubyte*)"glXCreateContextAttribsARB");
	if(glXCreateContextAttribsARB_RRJ == NULL){
		fprintf(gbFile_RRJ, "glXGetProcAddressARB() Failed!!\n");
		uninitialize();
		exit(1);
	}

	const int attributes_RRJ[] = {
		GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
		GLX_CONTEXT_MINOR_VERSION_ARB, 5,
		GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
		None
	};

	gGLXContext_RRJ = glXCreateContextAttribsARB_RRJ(gpDisplay_RRJ, gGLXFBConfig_RRJ, NULL, True, attributes_RRJ);
	if(gGLXContext_RRJ == NULL){
		fprintf(gbFile_RRJ, "glXCreateContextAttribsARB_RRJ() Failed!!\n");
		fprintf(gbFile_RRJ, "Getting Context give by System!!\n");

		const int attribs_RRJ[] = {
			GLX_CONTEXT_MAJOR_VERSION_ARB, 1,
			GLX_CONTEXT_MINOR_VERSION_ARB, 0,
			None
		};

		gGLXContext_RRJ = glXCreateContextAttribsARB_RRJ(gpDisplay_RRJ, gGLXFBConfig_RRJ, NULL, True, attribs_RRJ);
	}

	if(!glXIsDirect(gpDisplay_RRJ, gGLXContext_RRJ))
		fprintf(gbFile_RRJ, "S/W Context!!\n");
	else
		fprintf(gbFile_RRJ, "H/W Context!!\n");

	glXMakeCurrent(gpDisplay_RRJ, gWindow_RRJ, gGLXContext_RRJ);



	GLenum result_RRJ;
	result_RRJ = glewInit();
	if(result_RRJ != GLEW_OK){
		fprintf(gbFile_RRJ, "glewInit() Failed!!\n");
		uninitialize();
		exit(1);
	}


	

	/********** Vertex Shader Per Vertex *********/
	vertexShaderObject_PV_RRJ = glCreateShader(GL_VERTEX_SHADER);
	const char *szVertexShaderCode_PV_RRJ =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vNormals;" \
		"uniform mat4 u_modelview_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"uniform vec3 u_La;" \
		"uniform vec3 u_Ld;" \
		"uniform vec3 u_Ls;" \
		"uniform vec4 u_light_position;" \
		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3 u_Ks;" \
		"uniform float u_shininess;" \
		"uniform int u_L_keypress;" \
		"out vec3 phongLight;" \
		"void main(void)" \
		"{" \
			"if(u_L_keypress == 1){" \
				"vec4 eyeCoordinate = u_modelview_matrix * vPosition;" \
				"vec3 Source = normalize(vec3(u_light_position - eyeCoordinate));" \
				"mat3 normalMatrix = mat3(u_modelview_matrix);" \
				"vec3 Normal = normalize(normalMatrix * vNormals);" \
				"float S_Dot_N = max(dot(Source, Normal), 0.0);" \
				"vec3 Reflection = reflect(-Source, Normal);" \
				"vec3 Viewer = normalize(vec3(-eyeCoordinate.xyz));" \
				"float R_Dot_V = max(dot(Reflection, Viewer), 0.0);" \
				"vec3 ambient = u_La * u_Ka;" \
				"vec3 diffuse = u_Ld * u_Kd * S_Dot_N;" \
				"vec3 specular = u_Ls * u_Ks * pow(R_Dot_V, u_shininess);" \
				"phongLight = ambient + diffuse + specular;" \
			"}" \
			"else{" \
				"phongLight = vec3(1.0, 1.0, 1.0);" \
			"}" \
			"gl_Position = u_projection_matrix * u_modelview_matrix * vPosition;"
		"}";

	glShaderSource(vertexShaderObject_PV_RRJ, 1, (const GLchar**)&szVertexShaderCode_PV_RRJ, NULL);

	glCompileShader(vertexShaderObject_PV_RRJ);

	GLint iShaderCompileStatus_RRJ;
	GLint iInfoLogLength_RRJ;
	GLchar *szInfoLog_RRJ = NULL;
	glGetShaderiv(vertexShaderObject_PV_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(vertexShaderObject_PV_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(vertexShaderObject_PV_RRJ, iInfoLogLength_RRJ,
					&written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Per Vertex Lighting Vertex Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(0);
			}
		}
	}


	/********** Fragment Shader Per Vertex *********/
	fragmentShaderObject_PV_RRJ = glCreateShader(GL_FRAGMENT_SHADER);
	const char* szFragmentShaderCode_PV_RRJ =
		"#version 450 core" \
		"\n" \
		"in vec3 phongLight;" \
		"uniform float u_blend;" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
			"FragColor = vec4(phongLight, u_blend);" \
		"}";


	glShaderSource(fragmentShaderObject_PV_RRJ, 1,
		(const GLchar**)&szFragmentShaderCode_PV_RRJ, NULL);

	glCompileShader(fragmentShaderObject_PV_RRJ);

	iShaderCompileStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(fragmentShaderObject_PV_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(fragmentShaderObject_PV_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject_PV_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Per Vertex Lighting Fragment Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(0);
			}
		}
	}


	/********* Program Object For Per Vertex Lighting **********/
	shaderProgramObject_PV_RRJ = glCreateProgram();

	glAttachShader(shaderProgramObject_PV_RRJ, vertexShaderObject_PV_RRJ);
	glAttachShader(shaderProgramObject_PV_RRJ, fragmentShaderObject_PV_RRJ);

	glBindAttribLocation(shaderProgramObject_PV_RRJ, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(shaderProgramObject_PV_RRJ, AMC_ATTRIBUTE_NORMAL, "vNormals");

	glLinkProgram(shaderProgramObject_PV_RRJ);

	GLint iProgramLinkingStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;
	glGetProgramiv(shaderProgramObject_PV_RRJ, GL_LINK_STATUS, &iProgramLinkingStatus_RRJ);
	if (iProgramLinkingStatus_RRJ == GL_FALSE) {
		glGetProgramiv(shaderProgramObject_PV_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetProgramInfoLog(shaderProgramObject_PV_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Shader Program Object Linking Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(0);
			}
		}
	}

	modelViewMatrixUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_modelview_matrix");
	projectionMatrixUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_projection_matrix");
	LaUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_La");
	LdUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ld");
	LsUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ls");
	lightPositionUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_light_position");
	KaUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ka");
	KdUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Kd");
	KsUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_Ks");
	shininessUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_shininess");
	LKeyPressUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_L_keypress");

	blendingUniform_PV_RRJ = glGetUniformLocation(shaderProgramObject_PV_RRJ, "u_blend");







	/********** Vertex Shader Per Fragment Lighting *********/
	vertexShaderObject_PF_RRJ = glCreateShader(GL_VERTEX_SHADER);
	const char *szVertexShaderCode_PF_RRJ =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vNormals;" \
		"uniform mat4 u_modelview_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"uniform vec4 u_light_position;" \
		"out vec3 lightDirection_VS;" \
		"out vec3 Normal_VS;" \
		"out vec3 Viewer_VS;" \
		"void main(void)" \
		"{"
			"vec4 eyeCoordinate = u_modelview_matrix * vPosition;" \
			"lightDirection_VS = vec3(u_light_position - eyeCoordinate);" \
			"mat3 normalMatrix = mat3(u_modelview_matrix);" \
			"Normal_VS = vec3(normalMatrix * vNormals);" \
			"Viewer_VS = vec3(-eyeCoordinate);" \
			"gl_Position =	u_projection_matrix * u_modelview_matrix * vPosition;" \
		"}";


	glShaderSource(vertexShaderObject_PF_RRJ, 1, (const GLchar**)&szVertexShaderCode_PF_RRJ, NULL);

	glCompileShader(vertexShaderObject_PF_RRJ);


	glGetShaderiv(vertexShaderObject_PF_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(vertexShaderObject_PF_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(vertexShaderObject_PF_RRJ, iInfoLogLength_RRJ,
					&written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Per Fragment Lighting Vertex Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(0);
			}
		}
	}


	/********** Fragment Shader Per Fragment *********/
	fragmentShaderObject_PF_RRJ = glCreateShader(GL_FRAGMENT_SHADER);
	const char* szFragmentShaderCode_PF_RRJ =
		"#version 450 core" \
		"\n" \
		"uniform float u_blend;" \

		"in vec3 lightDirection_VS;" \
		"in vec3 Normal_VS;" \
		"in vec3 Viewer_VS;" \
		"uniform vec3 u_La;" \
		"uniform vec3 u_Ld;" \
		"uniform vec3 u_Ls;" \
		"uniform vec3 u_Ka;" \
		"uniform vec3 u_Kd;" \
		"uniform vec3	 u_Ks;" \
		"uniform float u_shininess;" \
		"uniform int u_L_keypress;" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
			"vec3 phongLight;" \
			"if(u_L_keypress == 1){" \
				"vec3 LightDirection = normalize(lightDirection_VS);" \
				"vec3 Normal = normalize(Normal_VS);" \
				"float L_Dot_N = max(dot(LightDirection, Normal), 0.0);" \
				"vec3 Reflection = reflect(-LightDirection, Normal);" \
				"vec3 Viewer = normalize(Viewer_VS);" \
				"float R_Dot_V = max(dot(Reflection, Viewer), 0.0);" \
				"vec3 ambient = u_La * u_Ka;" \
				"vec3 diffuse = u_Ld * u_Kd * L_Dot_N;" \
				"vec3 specular = u_Ls * u_Ks * pow(R_Dot_V, u_shininess);" \
				"phongLight = ambient + diffuse + specular;" \
			"}" \
			"else{" \
				"phongLight = vec3(1.0, 1.0, 1.0);" \
			"}" \
			"FragColor = vec4(phongLight, u_blend);" \
		"}";


	glShaderSource(fragmentShaderObject_PF_RRJ, 1,
		(const GLchar**)&szFragmentShaderCode_PF_RRJ, NULL);

	glCompileShader(fragmentShaderObject_PF_RRJ);

	iShaderCompileStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(fragmentShaderObject_PF_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(fragmentShaderObject_PF_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject_PF_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Per Fragment Lighting Fragment Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(0);
			}
		}
	}


	/********* Program Object For Per Fragment Lighting **********/
	shaderProgramObject_PF_RRJ = glCreateProgram();

	glAttachShader(shaderProgramObject_PF_RRJ, vertexShaderObject_PF_RRJ);
	glAttachShader(shaderProgramObject_PF_RRJ, fragmentShaderObject_PF_RRJ);

	glBindAttribLocation(shaderProgramObject_PF_RRJ, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(shaderProgramObject_PF_RRJ, AMC_ATTRIBUTE_NORMAL, "vNormals");

	glLinkProgram(shaderProgramObject_PF_RRJ);

	
	glGetProgramiv(shaderProgramObject_PF_RRJ, GL_LINK_STATUS, &iProgramLinkingStatus_RRJ);
	if (iProgramLinkingStatus_RRJ == GL_FALSE) {
		glGetProgramiv(shaderProgramObject_PF_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetProgramInfoLog(shaderProgramObject_PF_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Shader Program Object Linking Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				exit(0);
			}
		}
	}

	modelViewMatrixUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_modelview_matrix");
	projectionMatrixUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_projection_matrix");
	LaUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_La");
	LdUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Ld");
	LsUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Ls");
	lightPositionUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_light_position");
	KaUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Ka");
	KdUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Kd");
	KsUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_Ks");
	shininessUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_shininess");
	LKeyPressUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_L_keypress");
	blendingUniform_PF_RRJ = glGetUniformLocation(shaderProgramObject_PF_RRJ, "u_blend");




	/********** Position, Normal and Elements **********/
	numOfSphereElements_RRJ = CreateSphere_RRJ(1.0f, STACK, SLICES, 
		sphere_Pos_RRJ, sphere_Texcoord_RRJ, sphere_Nor_RRJ, sphere_Element_RRJ);

	/********** Sphere Vao **********/
	glGenVertexArrays(1, &vao_Sphere_RRJ);
	glBindVertexArray(vao_Sphere_RRJ);

		/********** Position **********/
		glGenBuffers(1, &vbo_Sphere_Position_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Sphere_Position_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_Pos_RRJ), sphere_Pos_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** Normals **********/
		glGenBuffers(1, &vbo_Sphere_Normal_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Sphere_Normal_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_Nor_RRJ), sphere_Nor_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** Element Vbo **********/
		glGenBuffers(1, &vbo_Sphere_Element_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(sphere_Element_RRJ), sphere_Element_RRJ, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


	glBindVertexArray(0);




	/********** SEMI SPHERE **********/
	numOfSemisphereElements_RRJ = CreateSemiSphere_RRJ(1.0f, STACK, SLICES, 
		semiSphere_Pos_RRJ, semiSphere_Texcoord_RRJ, semiSphere_Nor_RRJ, semiSphere_Element_RRJ);
	glGenVertexArrays(1, &vao_SemiSphere_RRJ);
	glBindVertexArray(vao_SemiSphere_RRJ);

		/********** POSITION **********/
		glGenBuffers(1, &vbo_SemiSphere_Pos_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_SemiSphere_Pos_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(semiSphere_Pos_RRJ), semiSphere_Pos_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		
		/********** NORMALS **********/
		glGenBuffers(1, &vbo_SemiSphere_Nor_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_SemiSphere_Nor_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(semiSphere_Nor_RRJ), semiSphere_Nor_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** ELEMENTS **********/
		glGenBuffers(1, &vbo_SemiSphere_Elem_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_SemiSphere_Elem_RRJ);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(semiSphere_Element_RRJ), semiSphere_Element_RRJ, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);	

	glBindVertexArray(0);




	/********** CYLINDER **********/
	numOfCylinderElements_RRJ = CreateCylinder_RRJ(1.0f, 4.0f, SLICES, 
		cylinder_Pos_RRJ, cylinder_Nor_RRJ, cylinder_Element_RRJ);

	glGenVertexArrays(1, &vao_Cylinder_RRJ);
	glBindVertexArray(vao_Cylinder_RRJ);

		/********** POSITION **********/
		glGenBuffers(1, &vbo_Cylinder_Pos_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Cylinder_Pos_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(cylinder_Pos_RRJ), cylinder_Pos_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** NORMAL **********/
		glGenBuffers(1, &vbo_Cylinder_Nor_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Cylinder_Nor_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(cylinder_Nor_RRJ), cylinder_Nor_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** ELEMENTS **********/
		glGenBuffers(1, &vbo_Cylinder_Elem_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Cylinder_Elem_RRJ);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cylinder_Element_RRJ), cylinder_Element_RRJ, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);




	/********** FRUSTUM **********/
	numOfFrustumElements_RRJ = CreateFrustum_RRJ(1.0f, 0.8f, 4.0f, SLICES,
		frustum_Pos_RRJ, frustum_Nor_RRJ, frustum_Element_RRJ);

	glGenVertexArrays(1, &vao_Frustum_RRJ);
	glBindVertexArray(vao_Frustum_RRJ);

		/********** POSITION **********/
		glGenBuffers(1, &vbo_Frustum_Pos_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Frustum_Pos_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(frustum_Pos_RRJ), frustum_Pos_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** NORMAL **********/
		glGenBuffers(1, &vbo_Frustum_Nor_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Frustum_Nor_RRJ);
		glBufferData(GL_ARRAY_BUFFER, sizeof(frustum_Nor_RRJ), frustum_Nor_RRJ, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** ELEMENTS **********/
		glGenBuffers(1, &vbo_Frustum_Elem_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Frustum_Elem_RRJ);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(frustum_Element_RRJ), frustum_Element_RRJ, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


	glBindVertexArray(0);




	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	gPerspectiveProjectionMatrix_RRJ = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);

}


void uninitialize(void) {

	GLXContext currentContext_RRJ = glXGetCurrentContext();



	/********** SOUND **********/
	if(gSource_End){
		alDeleteSources(1, &gSource_End);
		gSource_End = 0;
	}

	if(gSource_Swimming){
		alDeleteSources(1, &gSource_Swimming);
		gSource_Swimming = 0;
	}

	if(gSource_Squarting_Skipping){
		alDeleteSources(1, &gSource_Squarting_Skipping);
		gSource_Squarting_Skipping = 0;
	}


	if(gSource_Walk_Running){
		alDeleteSources(1, &gSource_Walk_Running);
		gSource_Walk_Running = 0;
	}


	if(gSource_Start){
		alDeleteSources(1, &gSource_Start);
		gSource_Start = 0;
	}



	if(gBuffer_End){
		alDeleteBuffers(1, &gBuffer_End);
		gBuffer_End = 0;
	}

	if(gBuffer_Swimming){
		alDeleteBuffers(1, &gBuffer_Swimming);
		gBuffer_Swimming = 0;
	}

	if(gBuffer_Squarting_Skipping){
		alDeleteBuffers(1, &gBuffer_Squarting_Skipping);
		gBuffer_Squarting_Skipping = 0;
	}


	if(gBuffer_Walk_Running){
		alDeleteBuffers(1, &gBuffer_Walk_Running);
		gBuffer_Walk_Running = 0;
	}


	if(gBuffer_Start){
		alDeleteBuffers(1, &gBuffer_Start);
		gBuffer_Start = 0;
	}


	pALCcontext = alcGetCurrentContext();
	pALCdevice = alcGetContextsDevice(pALCcontext);

	if(pALCcontext){
		alcMakeContextCurrent(NULL);
		alcDestroyContext(pALCcontext);
		pALCcontext = NULL;
	}


	if(pALCdevice){
		alcCloseDevice(pALCdevice);
		pALCdevice = NULL;
	}





	//Frustum
	if(vbo_Frustum_Elem_RRJ){
		glDeleteBuffers(1, &vbo_Frustum_Elem_RRJ);
		vbo_Frustum_Elem_RRJ = 0;
	}

	if(vbo_Frustum_Nor_RRJ){
		glDeleteBuffers(1, &vbo_Frustum_Nor_RRJ);
		vbo_Frustum_Nor_RRJ = 0;
	}


	if(vbo_Frustum_Pos_RRJ){
		glDeleteBuffers(1, &vbo_Frustum_Pos_RRJ);
		vbo_Frustum_Pos_RRJ = 0;
	}

	if(vao_Frustum_RRJ){
		glDeleteVertexArrays(1, &vao_Frustum_RRJ);
		vao_Frustum_RRJ = 0;
	}



	//Cylinder
	if(vbo_Cylinder_Elem_RRJ){
		glDeleteBuffers(1, &vbo_Cylinder_Elem_RRJ);
		vbo_Cylinder_Elem_RRJ = 0;	
	}

	if(vbo_Cylinder_Nor_RRJ){
		glDeleteBuffers(1, &vbo_Cylinder_Nor_RRJ);
		vbo_Cylinder_Nor_RRJ = 0;
	}


	if(vbo_Cylinder_Pos_RRJ){
		glDeleteBuffers(1, &vbo_Cylinder_Pos_RRJ);
		vbo_Cylinder_Pos_RRJ = 0;
	}

	if(vao_Cylinder_RRJ){
		glDeleteVertexArrays(1, &vao_Cylinder_RRJ);
		vao_Cylinder_RRJ = 0;
	}


	//SemiSphere
	if (vbo_SemiSphere_Elem_RRJ) {
		glDeleteBuffers(1, &vbo_SemiSphere_Elem_RRJ);
		vbo_SemiSphere_Elem_RRJ = 0;
	}

	if (vbo_SemiSphere_Nor_RRJ) {
		glDeleteBuffers(1, &vbo_SemiSphere_Nor_RRJ);
		vbo_SemiSphere_Nor_RRJ = 0;
	}

	if (vbo_SemiSphere_Pos_RRJ) {
		glDeleteBuffers(1, &vbo_SemiSphere_Pos_RRJ);
		vbo_SemiSphere_Pos_RRJ = 0;
	}

	if (vao_SemiSphere_RRJ) {
		glDeleteVertexArrays(1, &vao_SemiSphere_RRJ);
		vao_SemiSphere_RRJ = 0;
	}



	//Sphere
	if (vbo_Sphere_Element_RRJ) {
		glDeleteBuffers(1, &vbo_Sphere_Element_RRJ);
		vbo_Sphere_Element_RRJ = 0;
	}

	if (vbo_Sphere_Normal_RRJ) {
		glDeleteBuffers(1, &vbo_Sphere_Normal_RRJ);
		vbo_Sphere_Normal_RRJ = 0;
	}

	if (vbo_Sphere_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Sphere_Position_RRJ);
		vbo_Sphere_Position_RRJ = 0;
	}

	if (vao_Sphere_RRJ) {
		glDeleteVertexArrays(1, &vao_Sphere_RRJ);
		vao_Sphere_RRJ = 0;
	}


	GLsizei ShaderCount_RRJ;
	GLsizei ShaderNumber_RRJ;

	if (shaderProgramObject_PV_RRJ) {
		glUseProgram(shaderProgramObject_PV_RRJ);

		glGetProgramiv(shaderProgramObject_PV_RRJ, GL_ATTACHED_SHADERS, &ShaderCount_RRJ);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount_RRJ);
		if (pShader) {
			glGetAttachedShaders(shaderProgramObject_PV_RRJ, ShaderCount_RRJ, &ShaderCount_RRJ, pShader);
			for (ShaderNumber_RRJ = 0; ShaderNumber_RRJ < ShaderCount_RRJ; ShaderNumber_RRJ++) {
				glDetachShader(shaderProgramObject_PV_RRJ, pShader[ShaderNumber_RRJ]);
				glDeleteShader(pShader[ShaderNumber_RRJ]);
				pShader[ShaderNumber_RRJ] = 0;
			}
			free(pShader);
			pShader = NULL;
		}
		glDeleteProgram(shaderProgramObject_PV_RRJ);
		shaderProgramObject_PV_RRJ = 0;
		glUseProgram(0);
	}


	ShaderCount_RRJ = 0;
	ShaderNumber_RRJ = 0;
	if (shaderProgramObject_PF_RRJ) {
		glUseProgram(shaderProgramObject_PF_RRJ);

		glGetProgramiv(shaderProgramObject_PF_RRJ, GL_ATTACHED_SHADERS, &ShaderCount_RRJ);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount_RRJ);
		if (pShader) {
			glGetAttachedShaders(shaderProgramObject_PF_RRJ, ShaderCount_RRJ, &ShaderCount_RRJ, pShader);
			for (ShaderNumber_RRJ = 0; ShaderNumber_RRJ < ShaderCount_RRJ; ShaderNumber_RRJ++) {
				glDetachShader(shaderProgramObject_PF_RRJ, pShader[ShaderNumber_RRJ]);
				glDeleteShader(pShader[ShaderNumber_RRJ]);
				pShader[ShaderNumber_RRJ] = 0;
			}
			free(pShader);
			pShader = NULL;
		}
		glDeleteProgram(shaderProgramObject_PF_RRJ);
		shaderProgramObject_PF_RRJ = 0;
		glUseProgram(0);
	}
	

	if(currentContext_RRJ != NULL && currentContext_RRJ == gGLXContext_RRJ)
		glXMakeCurrent(gpDisplay_RRJ, 0, 0);
		
	if(glXCreateContextAttribsARB_RRJ)
		glXCreateContextAttribsARB_RRJ = NULL;
	
	
	if(gGLXContext_RRJ)
		glXDestroyContext(gpDisplay_RRJ, gGLXContext_RRJ);
		
	if(gGLXFBConfig_RRJ)
		gGLXFBConfig_RRJ = 0;
		
	if(gWindow_RRJ)
		XDestroyWindow(gpDisplay_RRJ, gWindow_RRJ);
		
	if(gColormap_RRJ)
		XFreeColormap(gpDisplay_RRJ, gColormap_RRJ);
		
	
	if(gpXVisualInfo_RRJ){
		XFree(gpXVisualInfo_RRJ);
		gpXVisualInfo_RRJ = NULL;
	}
	
	if(gpDisplay_RRJ){
		XCloseDisplay(gpDisplay_RRJ);
		gpDisplay_RRJ = NULL;
	}
	
	if(gbFile_RRJ){
		fprintf(gbFile_RRJ, "Log Close!!\n");
		fclose(gbFile_RRJ);
		gbFile_RRJ = NULL;
	}
	
}


void resize(int width, int height) {
	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	gPerspectiveProjectionMatrix_RRJ = mat4::identity();
	gPerspectiveProjectionMatrix_RRJ = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}



mat4 modelViewMatrix_RRJ;

void display(void) {

	void BodyAssembly(void);
	void BodyWalking(float);
	void BodyRunning(float);
	void BodySquarting(void);
	void BodySkipping(void);
	void BodySwimming(void);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	
	/********** Sphere **********/
	modelViewMatrix_RRJ = mat4::identity();


	if (iWhichLight_RRJ == PER_VERTEX){
		glUseProgram(shaderProgramObject_PV_RRJ);
		
				
		if (bLight_RRJ == true) {

			glUniform1i(LKeyPressUniform_PV_RRJ, 1);
			glUniform3fv(LaUniform_PV_RRJ, 1, lightAmbient_RRJ);
			glUniform3fv(LdUniform_PV_RRJ, 1, lightDiffuse_RRJ);
			glUniform3fv(LsUniform_PV_RRJ, 1, lightSpecular_RRJ);
			glUniform4fv(lightPositionUniform_PV_RRJ, 1, lightPosition_RRJ);
			glUniform3fv(KaUniform_PV_RRJ, 1, materialAmbient_RRJ);
			glUniform3fv(KdUniform_PV_RRJ, 1, materialDiffuse_RRJ);
			glUniform3fv(KsUniform_PV_RRJ, 1, materialSpecular_RRJ);
			glUniform1f(shininessUniform_PV_RRJ, materialShininess_RRJ);
		}
		else
			glUniform1i(LKeyPressUniform_PV_RRJ, 0);


		glUniform1f(blendingUniform_PV_RRJ, fBlendingFactor);			
			
	}else{
		glUseProgram(shaderProgramObject_PF_RRJ);
	
		if (bLight_RRJ == true) {

			glUniform1i(LKeyPressUniform_PF_RRJ, 1);
			glUniform3fv(LaUniform_PF_RRJ, 1, lightAmbient_RRJ);
			glUniform3fv(LdUniform_PF_RRJ, 1, lightDiffuse_RRJ);
			glUniform3fv(LsUniform_PF_RRJ, 1, lightSpecular_RRJ);
			glUniform4fv(lightPositionUniform_PF_RRJ, 1, lightPosition_RRJ);
			glUniform3fv(KaUniform_PF_RRJ, 1, materialAmbient_RRJ);
			glUniform3fv(KdUniform_PF_RRJ, 1, materialDiffuse_RRJ);
			glUniform3fv(KsUniform_PF_RRJ, 1, materialSpecular_RRJ);
			glUniform1f(shininessUniform_PF_RRJ, materialShininess_RRJ);
		}
		else
			glUniform1i(LKeyPressUniform_PF_RRJ, 0);


		glUniform1f(blendingUniform_PF_RRJ, fBlendingFactor);	
	}






	if(isDemoMode == true){

		switch(iScene){

			case 1:
				iWhichMovement = RRJ_ASSEMBLY;
				BodyAssembly();
				gTimePerFrame = gTimePerFrame - 0.01f;
				if(gTimePerFrame < 0.0f){
					gTimePerFrame = 2.0f;
					iScene = 2;
				}
				break;


			case 2:
				iWhichMovement = RRJ_ASSEMBLY;
				BodyAssembly();
				gTimePerFrame = gTimePerFrame - 0.01f;
				body_YRot_Angle = body_YRot_Angle - 0.1f;
				if(gTimePerFrame < 0.0f){
					elbowLeft_RRJ = 0.0f;
					gTimePerFrame = 22.0f;
					iWhichMovement = RRJ_WALKING;
					

					alSourcePlay(gSource_Walk_Running);
					
					

					iScene = 3;
				}
				break;


			case 3:
				iWhichMovement = RRJ_WALKING;
				BodyWalking(body_YRot_Angle);
				gTimePerFrame = gTimePerFrame - 0.01f;

				if(gTimePerFrame < 0.0f){
					gTimePerFrame = 21.00f;
					iScene = 4;
					iWhichMovement = RRJ_RUNNING;

					chest_RRJ = -5.0f;
					hip_RRJ = -5.0f;
					body_XTranslation = -body_XTranslation;
					body_YRot_Angle = 60.0f;


					shoulderLeft_RRJ = 0.0f;
					shoulderRight_RRJ = 0.0f;
					elbowLeft_RRJ = 0.0f;
					elbowRight_RRJ = 0.0f;
					thighLeft_RRJ = 0.0f;
					thighRight_RRJ = 0.0f;
					shinBoneLeft_RRJ = 0.0f;
					shinBoneRight_RRJ = 0.0f;
					iMoveFlag = 0;
				}
				break;


			case 4:
				iWhichMovement = RRJ_RUNNING;
				BodyRunning(body_YRot_Angle);
				gTimePerFrame = gTimePerFrame - 0.01f;
				if(gTimePerFrame <= 11.50f){
					body_YRot_Angle = -90.0f;
				}
				if(gTimePerFrame < 0.0f){
					
					gTimePerFrame = 12.0f;
					iScene = 5;
					iWhichMovement = RRJ_WALKING;

					alSourcePlay(gSource_Squarting_Skipping);
					
					body_YRot_Angle = 90.0f;
					chest_RRJ = 0.0f;
					hip_RRJ = 0.0f;
					shoulderLeft_RRJ = 0.0f;
					shoulderRight_RRJ = 0.0f;
					elbowLeft_RRJ = 0.0f;
					elbowRight_RRJ = 0.0f;
					thighLeft_RRJ = 0.0f;
					thighRight_RRJ = 0.0f;
					shinBoneLeft_RRJ = 0.0f;
					shinBoneRight_RRJ = 0.0f;
					iMoveFlag = 0;

				}
				break;


			case 5:
				iWhichMovement = RRJ_WALKING;
				BodyWalking(body_YRot_Angle);
				gTimePerFrame = gTimePerFrame - 0.01f;
				if(gTimePerFrame < 0.0f){
					
					gTimePerFrame = 8.8f;
					iWhichMovement = RRJ_SQUARTING;
					iScene = 6;


					chest_RRJ = 0.0f;
					hip_RRJ = 0.0f;

					body_YTranslation = 0.0f;

					shoulderLeft_RRJ = 90.0f;
					shoulderRight_RRJ = 90.0f;
					elbowLeft_RRJ = 0.0f;
					elbowRight_RRJ = 0.0f;

					thighRight_RRJ = 0.0f;
					thighLeft_RRJ = 0.0f;
					shinBoneLeft_RRJ = 0.0f;
					shinBoneRight_RRJ = 0.0f;
					iMoveFlag = 0;
					
				}
				break;


			case 6:
				iWhichMovement = RRJ_SQUARTING;
				BodySquarting();
				gTimePerFrame = gTimePerFrame - 0.01f;
				if(gTimePerFrame < 0.0f){
					
					gTimePerFrame = 12.0f;
					iScene = 7;

					iWhichMovement = RRJ_WALKING;

					chest_RRJ = 0.0f;
					hip_RRJ = 0.0f;


					shoulderLeft_RRJ = 0.0f;
					shoulderRight_RRJ = 0.0f;
					elbowLeft_RRJ = 0.0f;
					elbowRight_RRJ = 0.0f;

					thighRight_RRJ = 0.0f;
					thighLeft_RRJ = 0.0f;
					shinBoneLeft_RRJ = 0.0f;
					shinBoneRight_RRJ = 0.0f;
					iMoveFlag = 0;
				}
				break;


			case 7:
				iWhichMovement = RRJ_WALKING;
				BodyWalking(body_YRot_Angle);
				gTimePerFrame = gTimePerFrame - 0.01f;
				if(gTimePerFrame < 0.0f){

					gTimePerFrame = 11.6f;
					iScene = 8;

					iWhichMovement = RRJ_SKIPING;

					chest_RRJ = 0.0f;
					hip_RRJ = 0.0f;


					shoulderLeft_RRJ = 0.0f;
					shoulderRight_RRJ = 0.0f;
					elbowLeft_RRJ = 0.0f;
					elbowRight_RRJ = 0.0f;

					thighRight_RRJ = 0.0f;
					thighLeft_RRJ = 0.0f;
					shinBoneLeft_RRJ = 0.0f;
					shinBoneRight_RRJ = 0.0f;
					iMoveFlag = 0;
				}
				break;


			case 8:
				iWhichMovement = RRJ_SKIPING;
				BodySkipping();
				gTimePerFrame = gTimePerFrame - 0.01f;
				if(gTimePerFrame < 0.0f){
					gTimePerFrame = 10.0f;
					iScene = 9;

					iWhichMovement = RRJ_WALKING;

					chest_RRJ = 0.0f;
					hip_RRJ = 0.0f;


					shoulderLeft_RRJ = 0.0f;
					shoulderRight_RRJ = 0.0f;
					elbowLeft_RRJ = 0.0f;
					elbowRight_RRJ = 0.0f;

					thighRight_RRJ = 0.0f;
					thighLeft_RRJ = 0.0f;
					shinBoneLeft_RRJ = 0.0f;
					shinBoneRight_RRJ = 0.0f;
					iMoveFlag = 0;	
				}
				break;



			case 9:
				iWhichMovement = RRJ_WALKING;
				BodyWalking(body_YRot_Angle);
				gTimePerFrame = gTimePerFrame - 0.01f;
				if(gTimePerFrame < 0.0f){

					gTimePerFrame = 16.3f;
					iScene = 10;

					chest_RRJ = 0.0f;
					hip_RRJ = 0.0f;

					iWhichMovement = RRJ_SWIMMING;

					alSourcePlay(gSource_Swimming);

					
					chest_RRJ = -90.0f;
					hip_RRJ = -90.0f;

					body_YRot_Angle = -90.0f;

					shoulderLeft_RRJ = 180.0f;
					shoulderRight_RRJ = 0.0f;
					elbowLeft_RRJ = -40.0f;
					elbowRight_RRJ = -40.0f;
					thighRight_RRJ = 0.0f;
					thighLeft_RRJ = 0.0f;
					shinBoneLeft_RRJ = -20.0f;
					shinBoneRight_RRJ = -20.0f;
					iMoveFlag = 0;
				}
				break;


			case 10:
				iWhichMovement = RRJ_SWIMMING;
				BodySwimming();
				gTimePerFrame = gTimePerFrame - 0.01f;
				if(gTimePerFrame < 0.0f){
					gTimePerFrame = 18.0f;
					iScene = 11;

					iWhichMovement = RRJ_WALKING;

					alSourcePlay(gSource_End);

					body_YRot_Angle = 90.0f;

					chest_RRJ = 0.0f;
					hip_RRJ = 0.0f;

					shoulderLeft_RRJ = 0.0f;
					shoulderRight_RRJ = 0.0f;
					elbowLeft_RRJ = 0.0f;
					elbowRight_RRJ = 0.0f;

					thighRight_RRJ = 0.0f;
					thighLeft_RRJ = 0.0f;
					shinBoneLeft_RRJ = 0.0f;
					shinBoneRight_RRJ = 0.0f;
					iMoveFlag = 0;
				}
				break;


			case 11:
				iWhichMovement = RRJ_WALKING;
				BodyWalking(body_YRot_Angle);
				gTimePerFrame = gTimePerFrame - 0.01f;

				if(gTimePerFrame < 5.0f){
					body_YRot_Angle = body_YRot_Angle - 0.18f;
				}
				if(gTimePerFrame < 0.0f){
					iScene = 12;
					gTimePerFrame = 10.0f;

				}
				break;



			case 12:
				iWhichMovement = RRJ_WALKING;
				BodyWalking(body_YRot_Angle);
				gTimePerFrame = gTimePerFrame - 0.01f;
				fBlendingFactor = fBlendingFactor - 0.001f;
				if(gTimePerFrame < 0.0f){
					iWhichMovement = RRJ_NONE;
					iScene = 13;
					gTimePerFrame = 0.0f;
				}
		}
	}









	/********** MAIN BODY CENTER **********/
	//Sphere
	modelViewMatrix_RRJ = lookat(vec3(0.0f, 0.0f, 25.0f), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f));
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(body_XTranslation, body_YTranslation, body_ZTranslation);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(0.0f, body_YRot_Angle, 0.0f);


	my_glPushMatrix(modelViewMatrix_RRJ);

	my_glPushMatrix(modelViewMatrix_RRJ);

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(chest_RRJ, 0.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * scale(0.60f, 0.10f, 0.450f);
	

	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}

	glBindVertexArray(vao_Cylinder_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Cylinder_Elem_RRJ);
		glDrawElements(GL_TRIANGLES, numOfCylinderElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);











	/********** UPPER BODY **********/


	//Upper Frustum
	modelViewMatrix_RRJ = my_glPopMatrix();

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(chest_RRJ, 0.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * scale(0.8f, 0.2f, 0.6f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, 1.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, 2.0f, 0.0f);

	my_glPushMatrix(modelViewMatrix_RRJ);


	

	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}

	glBindVertexArray(vao_Frustum_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Frustum_Elem_RRJ);
		glDrawElements(GL_TRIANGLES, numOfFrustumElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);




	//Upper Sphere(Chest)
	modelViewMatrix_RRJ = my_glPopMatrix();


	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, 1.80f, 0.0f);

	my_glPushMatrix(modelViewMatrix_RRJ);

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(-90.0f, 0.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * scale(1.0f, 1.0f, 2.0f);


	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}

	glBindVertexArray(vao_SemiSphere_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_SemiSphere_Elem_RRJ);
		glDrawElements(GL_TRIANGLES, numOfSemisphereElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);





	//Cylinder(NECK)
	modelViewMatrix_RRJ = my_glPopMatrix();
	my_glPushMatrix(modelViewMatrix_RRJ);

	

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, 0.50f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, 2.0f, 0.0f);


	my_glPushMatrix(modelViewMatrix_RRJ);

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * scale(0.2f, 0.50f, 0.20f);

	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}

	glBindVertexArray(vao_Cylinder_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Cylinder_Elem_RRJ);
		glDrawElements(GL_TRIANGLES, numOfCylinderElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);




	//Upper Sphere(HEAD)
	modelViewMatrix_RRJ = my_glPopMatrix();
	//my_glPushMatrix(modelViewMatrix_RRJ);


	modelViewMatrix_RRJ = modelViewMatrix_RRJ * scale(0.5f, 2.0f, 0.70f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, 1.0f, 0.0f);

	
	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}

	glBindVertexArray(vao_Sphere_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ);
		glDrawElements(GL_TRIANGLES, numOfSphereElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);






	/********** LEFT ARM **********/
	modelViewMatrix_RRJ = my_glPopMatrix();
	my_glPushMatrix(modelViewMatrix_RRJ);

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(fLeftArm, 0.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * scale(0.5f, 1.0f, 0.33333f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * scale(0.5f, 1.0f, 1.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(-4.50f, 0.0f, 0.0f);
	
	my_glPushMatrix(modelViewMatrix_RRJ);

	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}

	glBindVertexArray(vao_Sphere_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ);
		glDrawElements(GL_TRIANGLES, numOfSphereElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);




	//Cylinder
	modelViewMatrix_RRJ = my_glPopMatrix();

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(0.0f, 0.0f, -10.0f);

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(shoulderLeft_RRJ, 0.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -3.0f, 0.0f);

	

	my_glPushMatrix(modelViewMatrix_RRJ);


	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}


	glBindVertexArray(vao_Cylinder_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Cylinder_Elem_RRJ);
		glDrawElements(GL_TRIANGLES, numOfCylinderElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);




	//Sphere
	modelViewMatrix_RRJ = my_glPopMatrix();

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -2.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -1.0f, 0.0f);
	if(iWhichMovement == RRJ_SKIPING)
		modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(elbowLeft_RRJ, 0.0f, 0.0f);

	my_glPushMatrix(modelViewMatrix_RRJ);

	

	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}



	glBindVertexArray(vao_Sphere_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ);
		glDrawElements(GL_TRIANGLES, numOfSphereElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);



	//Cylinder
	modelViewMatrix_RRJ = my_glPopMatrix();


	
	if(iWhichMovement == RRJ_SKIPING){
		modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(0.0f, 0.0f, -25.0f);
	}
	else{
		modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(0.0f, 0.0f, 10.0f);
		modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(elbowLeft_RRJ, 0.0f, 0.0f);
	}

	
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -1.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -2.0f, 0.0f);

	my_glPushMatrix(modelViewMatrix_RRJ);


	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}



	glBindVertexArray(vao_Cylinder_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Cylinder_Elem_RRJ);
		glDrawElements(GL_TRIANGLES, numOfCylinderElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);




	//Sphere
	modelViewMatrix_RRJ = my_glPopMatrix();

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -2.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -1.0f, 0.0f);


	
	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}



	glBindVertexArray(vao_Sphere_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ);
		glDrawElements(GL_TRIANGLES, numOfSphereElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);










	/********** RIGHT ARM **********/
	modelViewMatrix_RRJ = my_glPopMatrix();
	//my_glPushMatrix(modelViewMatrix_RRJ);

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(fRightArm, 0.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * scale(0.5f, 1.0f, 0.33333f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * scale(0.5f, 1.0f, 1.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(4.50f, 0.0f, 0.0f);
	
	my_glPushMatrix(modelViewMatrix_RRJ);

	
	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}


	glBindVertexArray(vao_Sphere_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ);
		glDrawElements(GL_TRIANGLES, numOfSphereElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);




	//Cylinder
	modelViewMatrix_RRJ = my_glPopMatrix();

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(0.0f, 0.0f, 10.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(shoulderRight_RRJ, 0.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -3.0f, 0.0f);

	

	my_glPushMatrix(modelViewMatrix_RRJ);

	
	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}


	glBindVertexArray(vao_Cylinder_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Cylinder_Elem_RRJ);
		glDrawElements(GL_TRIANGLES, numOfCylinderElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);




	//Sphere
	modelViewMatrix_RRJ = my_glPopMatrix();

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -2.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -1.0f, 0.0f);
	if(iWhichMovement == RRJ_SKIPING)
		modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(elbowRight_RRJ, 0.0f, 0.0f);

	my_glPushMatrix(modelViewMatrix_RRJ);

	
	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}


	glBindVertexArray(vao_Sphere_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ);
		glDrawElements(GL_TRIANGLES, numOfSphereElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);



	//Cylinder
	modelViewMatrix_RRJ = my_glPopMatrix();



	if(iWhichMovement == RRJ_SKIPING){
		modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(0.0f, 0.0f, 25.0f);
	}
	else{
		modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(0.0f, 0.0f, -10.0f);
		modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(elbowRight_RRJ, 0.0f, 0.0f);
	}

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -1.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -2.0f, 0.0f);

	my_glPushMatrix(modelViewMatrix_RRJ);


	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}



	glBindVertexArray(vao_Cylinder_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Cylinder_Elem_RRJ);
		glDrawElements(GL_TRIANGLES, numOfCylinderElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);




	//Sphere
	modelViewMatrix_RRJ = my_glPopMatrix();

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -2.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -1.0f, 0.0f);


	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}


	glBindVertexArray(vao_Sphere_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ);
		glDrawElements(GL_TRIANGLES, numOfSphereElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

















	/********** LOWER BODY **********/

	//Frustum
	modelViewMatrix_RRJ = my_glPopMatrix();

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(hip_RRJ, 0.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * scale(0.8f, 0.2f, 0.6f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -1.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -2.0f, 0.0f);

	my_glPushMatrix(modelViewMatrix_RRJ);

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(180.0f, 0.0f, 0.0f);
	

	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}


	glBindVertexArray(vao_Frustum_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Frustum_Elem_RRJ);
		glDrawElements(GL_TRIANGLES, numOfFrustumElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);













	/*********** LEFT LEG **********/
	modelViewMatrix_RRJ = my_glPopMatrix();
	my_glPushMatrix(modelViewMatrix_RRJ);

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, fLeftLeg, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * scale(0.3125f, 1.05f, 0.416667f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -2.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(-1.50f, -1.0f, 0.0f);
	


	my_glPushMatrix(modelViewMatrix_RRJ);


	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}



	glBindVertexArray(vao_Sphere_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ);
		glDrawElements(GL_TRIANGLES, numOfSphereElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);



	//Frustum
	modelViewMatrix_RRJ = my_glPopMatrix();

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(thighLeft_RRJ, 0.0f, 0.0F);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -3.0f, 0.0f);
	

	my_glPushMatrix(modelViewMatrix_RRJ);


	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}


	glBindVertexArray(vao_Frustum_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Frustum_Elem_RRJ);
		glDrawElements(GL_TRIANGLES, numOfFrustumElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);



	//Sphere
	modelViewMatrix_RRJ = my_glPopMatrix();

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -2.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -1.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * scale(0.9f, 0.9f, 0.9f);


	my_glPushMatrix(modelViewMatrix_RRJ);

	
	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}


	glBindVertexArray(vao_Sphere_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ);
		glDrawElements(GL_TRIANGLES, numOfSphereElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);



	//Cylinder
	modelViewMatrix_RRJ = my_glPopMatrix();

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(shinBoneLeft_RRJ, 0.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -1.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -2.0f, 0.0f);

	my_glPushMatrix(modelViewMatrix_RRJ);

	
	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}


	glBindVertexArray(vao_Cylinder_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Cylinder_Elem_RRJ);
		glDrawElements(GL_TRIANGLES, numOfCylinderElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);



	//SemiSphere	
	modelViewMatrix_RRJ = my_glPopMatrix();

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -2.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -1.0f, -0.30f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(-90.0f, 0.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * scale(1.0f, 1.50f, 2.0f);



	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	

	glBindVertexArray(vao_SemiSphere_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_SemiSphere_Elem_RRJ);
		glDrawElements(GL_TRIANGLES, numOfSemisphereElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);











	/*********** RIGHT LEG **********/
	modelViewMatrix_RRJ = my_glPopMatrix();
	my_glPushMatrix(modelViewMatrix_RRJ);


	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, fRightLeg, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * scale(0.3125f, 1.05f, 0.416667f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -2.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(1.50f, -1.0f, 0.0f);
	


	my_glPushMatrix(modelViewMatrix_RRJ);

	

	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}


	glBindVertexArray(vao_Sphere_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ);
		glDrawElements(GL_TRIANGLES, numOfSphereElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	


	//Frustum
	modelViewMatrix_RRJ = my_glPopMatrix();

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(thighRight_RRJ, 0.0f, 0.0F);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -3.0f, 0.0f);
	

	my_glPushMatrix(modelViewMatrix_RRJ);

	
	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}


	glBindVertexArray(vao_Frustum_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Frustum_Elem_RRJ);
		glDrawElements(GL_TRIANGLES, numOfFrustumElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);



	//Sphere
	modelViewMatrix_RRJ = my_glPopMatrix();

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -2.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -1.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * scale(0.9f, 0.9f, 0.9f);


	my_glPushMatrix(modelViewMatrix_RRJ);

	
	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}


	glBindVertexArray(vao_Sphere_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ);
		glDrawElements(GL_TRIANGLES, numOfSphereElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);



	//Cylinder
	modelViewMatrix_RRJ = my_glPopMatrix();

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(shinBoneRight_RRJ, 0.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -1.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -2.0f, 0.0f);

	my_glPushMatrix(modelViewMatrix_RRJ);

	
	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}


	glBindVertexArray(vao_Cylinder_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Cylinder_Elem_RRJ);
		glDrawElements(GL_TRIANGLES, numOfCylinderElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);



	//SemiSphere	
	modelViewMatrix_RRJ = my_glPopMatrix();

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -2.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.0f, -1.0f, -0.30f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate(-90.0f, 0.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * scale(1.0f, 1.50f, 2.0f);


	if(iWhichLight_RRJ == PER_VERTEX){
		glUniformMatrix4fv(modelViewMatrixUniform_PV_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PV_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}
	else{
		glUniformMatrix4fv(modelViewMatrixUniform_PF_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
		glUniformMatrix4fv(projectionMatrixUniform_PF_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);
	}


	glBindVertexArray(vao_SemiSphere_RRJ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_SemiSphere_Elem_RRJ);
		glDrawElements(GL_TRIANGLES, numOfSemisphereElements_RRJ, GL_UNSIGNED_SHORT, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);



	my_glPopMatrix();



	glUseProgram(0);
	

	glXSwapBuffers(gpDisplay_RRJ, gWindow_RRJ);
}



void my_glPushMatrix(mat4 matrix){

	MVStack *temp = NULL;

	if(TopNode == NULL){
		TopNode = (MVStack*)malloc(sizeof(MVStack));
		TopNode->next = NULL;
		TopNode->prev = NULL;
		TopNode->ModelViewMatrix = matrix;

	}
	else{

		if(TopNode->next == NULL){
			temp = (MVStack*)malloc(sizeof(MVStack));
			temp->next = NULL;
			temp->ModelViewMatrix = matrix;

			TopNode->next = temp;
			temp->prev = TopNode;

			TopNode = temp;	
		}


	}

}



mat4 my_glPopMatrix(void){

	mat4 matrix;
	MVStack *temp = TopNode;

	if(TopNode->prev != NULL){

		TopNode = temp->prev;
		temp->next = NULL;
		temp->prev = NULL;
		matrix = temp->ModelViewMatrix;

		TopNode->next = NULL;

		free(temp);
		temp = NULL;
	}
	else if(TopNode->prev == NULL){

		TopNode->next = NULL;
		TopNode->prev = NULL;
		matrix = TopNode->ModelViewMatrix;

		free(TopNode);
		TopNode = NULL;
	}

	return(matrix);

}


void BodyAssembly(void){


	fLeftArm = fLeftArm + 0.09f;
	fRightArm = fRightArm - 0.09f;

	fRightLeg = fRightLeg + 0.15f;
	fLeftLeg = fLeftLeg + 0.15f;

	fBlendingFactor = fBlendingFactor + 0.002f;

	if(fLeftArm > 0.0f)
		fLeftArm = 0.0f;
	if(fRightArm < 0.0f)
		fRightArm = 0.0;

	if(fLeftLeg > 0.0f)
		fLeftLeg = 0.0f;

	if(fRightLeg > 0.0f)
		fRightLeg = 0.0f;

	if(fBlendingFactor > 1.0f)
		fBlendingFactor = 1.0f;
}





void update(void){

	if(iWhichMovement == RRJ_WALKING){


		/********** ARM MOVEMENT **********/
		if(iMoveFlag == 0){

			//Left Arm Forward

			//Shoulder
			shoulderLeft_RRJ = shoulderLeft_RRJ + 1.0f;
			shoulderRight_RRJ = shoulderRight_RRJ - 1.0f;
			if(shoulderLeft_RRJ >= MAX_SHOULDER_MOVEMENT_WALKING){
				iMoveFlag = 1;
				shoulderLeft_RRJ = MAX_SHOULDER_MOVEMENT_WALKING;
				shoulderRight_RRJ = -MAX_SHOULDER_MOVEMENT_WALKING;
			}


			//Left Elbow
			if(shoulderLeft_RRJ >= -MAX_SHOULDER_MOVEMENT_WALKING)
				elbowLeft_RRJ = elbowLeft_RRJ + 0.5f;

			//Right Elbow
			if(shoulderRight_RRJ >= -MAX_SHOULDER_MOVEMENT_WALKING && elbowRight_RRJ > 0.0f)
				elbowRight_RRJ = elbowRight_RRJ - 0.5f;






			//Right Thigh Forward
			thighRight_RRJ = thighRight_RRJ + 0.80f;
			thighLeft_RRJ = thighLeft_RRJ - 0.80f;



			if(thighLeft_RRJ <= 0.0f)
				shinBoneLeft_RRJ = shinBoneLeft_RRJ - 0.5f;


			if(thighRight_RRJ <= 0.0f)
				shinBoneRight_RRJ = shinBoneRight_RRJ - 0.5f;


			if(thighRight_RRJ >= 0.0f && shinBoneRight_RRJ < 0.0f)
				shinBoneRight_RRJ = shinBoneRight_RRJ + 1.0f;

			
		}
		else if(iMoveFlag == 1){

			//Right Arm Forward
			shoulderLeft_RRJ = shoulderLeft_RRJ - 1.0f;
			shoulderRight_RRJ = shoulderRight_RRJ + 1.0f;
			if(shoulderRight_RRJ >= MAX_SHOULDER_MOVEMENT_WALKING){
				iMoveFlag = 0;
				shoulderRight_RRJ = MAX_SHOULDER_MOVEMENT_WALKING;
				shoulderLeft_RRJ = -MAX_SHOULDER_MOVEMENT_WALKING;
			}


			//Right Elbow
			if(shoulderRight_RRJ >= -MAX_SHOULDER_MOVEMENT_WALKING)
				elbowRight_RRJ = elbowRight_RRJ + 0.5f;

			if(shoulderLeft_RRJ >= -MAX_SHOULDER_MOVEMENT_WALKING && elbowLeft_RRJ > 0.0f)
				elbowLeft_RRJ = elbowLeft_RRJ - 0.5f;






			//Left Thigh Forward
			thighLeft_RRJ = thighLeft_RRJ + 0.80f;
			thighRight_RRJ = thighRight_RRJ - 0.80f;



			if(thighRight_RRJ <= 0.0f)
				shinBoneRight_RRJ = shinBoneRight_RRJ - 0.5f;

			if(thighLeft_RRJ <= 0.0f)
				shinBoneLeft_RRJ = shinBoneLeft_RRJ - 0.5f;


			if(thighLeft_RRJ >= 0.0f && shinBoneLeft_RRJ < 0.0f)
				shinBoneLeft_RRJ = shinBoneLeft_RRJ + 1.0f;
		}



		//body_XTranslation = 0.01f * cos(degToRad(90.0f + body_YRot_Angle)) + body_XTranslation;
		//body_ZTranslation = -(0.01f * sin(degToRad(90.0f + body_YRot_Angle)) - body_ZTranslation);


	}
	else if(iWhichMovement == RRJ_RUNNING){


		/********** ARM MOVEMENT **********/
		if(iMoveFlag == 0){

			//Left Arm Forward

			//Shoulder
			shoulderLeft_RRJ = shoulderLeft_RRJ + 2.5f;
			shoulderRight_RRJ = shoulderRight_RRJ - 2.5f;
			if(shoulderLeft_RRJ >= MAX_SHOULDER_MOVEMENT_RUNNING){
				iMoveFlag = 1;
				//printf("In 0: %f\n", shoulderLeft_RRJ);
				shoulderLeft_RRJ = MAX_SHOULDER_MOVEMENT_RUNNING;
				shoulderRight_RRJ = -MAX_SHOULDER_MOVEMENT_RUNNING;
			}


			//Left Elbow
			if(shoulderLeft_RRJ >= -MAX_SHOULDER_MOVEMENT_RUNNING)
				elbowLeft_RRJ = elbowLeft_RRJ + 1.5f;

			//Right Elbow
			if(shoulderRight_RRJ >= -MAX_SHOULDER_MOVEMENT_RUNNING && elbowRight_RRJ > 0.0f)
				elbowRight_RRJ = elbowRight_RRJ - 1.5f;



			//Right Thigh Forward
			thighRight_RRJ = thighRight_RRJ + 2.5f;
			thighLeft_RRJ = thighLeft_RRJ - 2.5f;



			if(thighRight_RRJ <= 0.0f)
				shinBoneRight_RRJ = shinBoneRight_RRJ - 5.0;

			if(thighRight_RRJ >= 0.0f && shinBoneRight_RRJ < 0.0f)
				shinBoneRight_RRJ = shinBoneRight_RRJ + 4.0f;

			if(thighLeft_RRJ <= 0.0f && shinBoneLeft_RRJ < 0.0f)
				shinBoneLeft_RRJ = shinBoneLeft_RRJ + 1.0f;


			
		}
		else if(iMoveFlag == 1){

			//Right Arm Forward
			shoulderLeft_RRJ = shoulderLeft_RRJ - 2.5f;
			shoulderRight_RRJ = shoulderRight_RRJ + 2.5f;
			if(shoulderRight_RRJ >= MAX_SHOULDER_MOVEMENT_RUNNING){
				iMoveFlag = 0;
				//printf("In 1: %f\n", shoulderRight_RRJ);
				shoulderLeft_RRJ = -MAX_SHOULDER_MOVEMENT_RUNNING;
				shoulderRight_RRJ = MAX_SHOULDER_MOVEMENT_RUNNING;
			}


			//Right Elbow
			if(shoulderRight_RRJ >= -MAX_SHOULDER_MOVEMENT_RUNNING)
				elbowRight_RRJ = elbowRight_RRJ + 1.5f;

			if(shoulderLeft_RRJ >= -MAX_SHOULDER_MOVEMENT_RUNNING && elbowLeft_RRJ > 0.0f)
				elbowLeft_RRJ = elbowLeft_RRJ - 1.5f;






			//Left Thigh Forward
			thighLeft_RRJ = thighLeft_RRJ + 2.5f;
			thighRight_RRJ = thighRight_RRJ - 2.5f;


			if(thighLeft_RRJ <= 0.0f)
				shinBoneLeft_RRJ = shinBoneLeft_RRJ - 5.0f;


			if(thighLeft_RRJ >= 0.0f && shinBoneLeft_RRJ < 0.0f)
				shinBoneLeft_RRJ = shinBoneLeft_RRJ + 4.0f;


			if(thighRight_RRJ <= 0.0f && shinBoneRight_RRJ < 0.0f)
				shinBoneRight_RRJ = shinBoneRight_RRJ + 1.0f;

		}

		//printf("TL: %f/ TR: %f/ SBL: %f/ SBR: %f\n", thighLeft_RRJ, thighRight_RRJ, shinBoneLeft_RRJ, shinBoneRight_RRJ);
		//printf("SL:%f/ SR:%f/ EL:%f/ ER:%f\n", shoulderLeft_RRJ, shoulderRight_RRJ, elbowLeft_RRJ, elbowRight_RRJ);
	}
	else if(iWhichMovement == RRJ_SWIMMING){



		/********** ARMS **********/
		shoulderLeft_RRJ = shoulderLeft_RRJ - 1.5f;
		shoulderRight_RRJ = shoulderRight_RRJ - 1.5f;
		
		if(shoulderLeft_RRJ < -360.0f)
			shoulderLeft_RRJ = 0.0f;

		if(shoulderRight_RRJ < -360.0f)
			shoulderRight_RRJ = 0.0f;


		if(iMoveFlag == 0){
			/********** LEGS **********/
			thighLeft_RRJ = thighLeft_RRJ + 1.0f;
			thighRight_RRJ = thighRight_RRJ - 1.0f;
			if(thighLeft_RRJ > MAX_THIGH_MOVEMENT_SWIMMING){
				thighLeft_RRJ = MAX_THIGH_MOVEMENT_SWIMMING;
				thighRight_RRJ = -MAX_THIGH_MOVEMENT_SWIMMING;
				iMoveFlag = 1;
			}
		}
		else if(iMoveFlag == 1){
			/********** LEGS **********/
			thighLeft_RRJ = thighLeft_RRJ - 1.0f;
			thighRight_RRJ = thighRight_RRJ + 1.0f;
			if(thighRight_RRJ > MAX_THIGH_MOVEMENT_SWIMMING){
				thighRight_RRJ = MAX_THIGH_MOVEMENT_SWIMMING;
				thighLeft_RRJ = -MAX_THIGH_MOVEMENT_SWIMMING;
				iMoveFlag = 0;
			}
		}

	}
	else if(iWhichMovement == RRJ_SQUARTING){

		if(iMoveFlag == 0){

			thighLeft_RRJ = thighLeft_RRJ + 1.0f;
			thighRight_RRJ = thighRight_RRJ + 1.0f;
			shinBoneLeft_RRJ = shinBoneLeft_RRJ - 1.0f;
			shinBoneRight_RRJ = shinBoneRight_RRJ - 1.0f;

			chest_RRJ = chest_RRJ - 0.25f;
			hip_RRJ = hip_RRJ - 0.25f;

			shoulderLeft_RRJ = shoulderLeft_RRJ + 0.25f;
			shoulderRight_RRJ = shoulderRight_RRJ + 0.25f;

			body_YTranslation = body_YTranslation - 0.01f;
			//body_ZTranslation = body_ZTranslation + 0.01f;

			if(thighLeft_RRJ >= MAX_THIGH_SHINBONE_MOVEMENT_SQUARTING)
				iMoveFlag = 1;

		}
		else{
			thighLeft_RRJ = thighLeft_RRJ - 1.0f;
			thighRight_RRJ = thighRight_RRJ - 1.0f;
			shinBoneLeft_RRJ = shinBoneLeft_RRJ + 1.0f;
			shinBoneRight_RRJ = shinBoneRight_RRJ + 1.0f;

			chest_RRJ = chest_RRJ + 0.25f;
			hip_RRJ = hip_RRJ + 0.25f;

			shoulderLeft_RRJ = shoulderLeft_RRJ - 0.25f;
			shoulderRight_RRJ = shoulderRight_RRJ - 0.25f;

			body_YTranslation = body_YTranslation + 0.01f;
			//body_ZTranslation = body_ZTranslation - 0.01f;

			if(thighLeft_RRJ <= 0.0f){
				iMoveFlag = 0;
				body_YRot_Angle = body_YRot_Angle + 90.0f;
			}
		}
	}
	else if(iWhichMovement == RRJ_SKIPING){

		elbowLeft_RRJ = elbowLeft_RRJ - 4.0f;
		elbowRight_RRJ = elbowRight_RRJ - 4.0f;
		if(elbowLeft_RRJ < -360.0f)
			elbowLeft_RRJ = 0.0f;
		if(elbowRight_RRJ < -360.0f)
			elbowRight_RRJ = 0.0f;

		if(elbowLeft_RRJ <= -330.0f){
			//printf("0: %f\n", elbowLeft_RRJ);
			body_YTranslation = body_YTranslation + 0.05f;

			shinBoneLeft_RRJ = shinBoneLeft_RRJ - 4.0f;
			shinBoneRight_RRJ = shinBoneRight_RRJ - 4.0f;
			if(elbowLeft_RRJ <= -360.0f)
				iMoveFlag = 1;

			body_YRot_Angle = body_YRot_Angle + 1.0f;

		}
		else if(elbowLeft_RRJ <= -20.0f && iMoveFlag == 1 && elbowLeft_RRJ >= -50.0f){
			//printf("1: %f\n", elbowLeft_RRJ);
			body_YTranslation = body_YTranslation - 0.05f;
			
			shinBoneLeft_RRJ = shinBoneLeft_RRJ + 4.0f;
			shinBoneRight_RRJ = shinBoneRight_RRJ + 4.0f;

			if(elbowLeft_RRJ <= -45.0f){
				iMoveFlag = 0;
			}

			
			body_YRot_Angle = body_YRot_Angle + 1.0f;
		}
	}


}








void BodyWalking(float angle){


	/********** ARM MOVEMENT **********/
	if(iMoveFlag == 0){

		//Left Arm Forward

		//Shoulder
		shoulderLeft_RRJ = shoulderLeft_RRJ + 1.0f;
		shoulderRight_RRJ = shoulderRight_RRJ - 1.0f;
		if(shoulderLeft_RRJ >= MAX_SHOULDER_MOVEMENT_WALKING){
			iMoveFlag = 1;
			shoulderLeft_RRJ = MAX_SHOULDER_MOVEMENT_WALKING;
			shoulderRight_RRJ = -MAX_SHOULDER_MOVEMENT_WALKING;
		}


		//Left Elbow
		if(shoulderLeft_RRJ >= -MAX_SHOULDER_MOVEMENT_WALKING)
			elbowLeft_RRJ = elbowLeft_RRJ + 0.5f;

		//Right Elbow
		if(shoulderRight_RRJ >= -MAX_SHOULDER_MOVEMENT_WALKING && elbowRight_RRJ > 0.0f)
			elbowRight_RRJ = elbowRight_RRJ - 0.5f;






		//Right Thigh Forward
		thighRight_RRJ = thighRight_RRJ + 0.80f;
		thighLeft_RRJ = thighLeft_RRJ - 0.80f;



		if(thighLeft_RRJ <= 0.0f)
			shinBoneLeft_RRJ = shinBoneLeft_RRJ - 0.5f;


		if(thighRight_RRJ <= 0.0f)
			shinBoneRight_RRJ = shinBoneRight_RRJ - 0.5f;


		if(thighRight_RRJ >= 0.0f && shinBoneRight_RRJ < 0.0f)
			shinBoneRight_RRJ = shinBoneRight_RRJ + 1.0f;

		
	}
	else if(iMoveFlag == 1){

		//Right Arm Forward
		shoulderLeft_RRJ = shoulderLeft_RRJ - 1.0f;
		shoulderRight_RRJ = shoulderRight_RRJ + 1.0f;
		if(shoulderRight_RRJ >= MAX_SHOULDER_MOVEMENT_WALKING){
			iMoveFlag = 0;
			shoulderRight_RRJ = MAX_SHOULDER_MOVEMENT_WALKING;
			shoulderLeft_RRJ = -MAX_SHOULDER_MOVEMENT_WALKING;
		}


		//Right Elbow
		if(shoulderRight_RRJ >= -MAX_SHOULDER_MOVEMENT_WALKING)
			elbowRight_RRJ = elbowRight_RRJ + 0.5f;

		if(shoulderLeft_RRJ >= -MAX_SHOULDER_MOVEMENT_WALKING && elbowLeft_RRJ > 0.0f)
			elbowLeft_RRJ = elbowLeft_RRJ - 0.5f;






		//Left Thigh Forward
		thighLeft_RRJ = thighLeft_RRJ + 0.80f;
		thighRight_RRJ = thighRight_RRJ - 0.80f;



		if(thighRight_RRJ <= 0.0f)
			shinBoneRight_RRJ = shinBoneRight_RRJ - 0.5f;

		if(thighLeft_RRJ <= 0.0f)
			shinBoneLeft_RRJ = shinBoneLeft_RRJ - 0.5f;


		if(thighLeft_RRJ >= 0.0f && shinBoneLeft_RRJ < 0.0f)
			shinBoneLeft_RRJ = shinBoneLeft_RRJ + 1.0f;
	}


	body_XTranslation = 0.01f * cos(degToRad(90.0f + angle)) + body_XTranslation;
	body_ZTranslation = -(0.01f * sin(degToRad(90.0f + angle)) - body_ZTranslation);

	
}



void BodyRunning(float angle){


	/********** ARM MOVEMENT **********/
	if(iMoveFlag == 0){

		//Left Arm Forward

		//Shoulder
		shoulderLeft_RRJ = shoulderLeft_RRJ + 2.5f;
		shoulderRight_RRJ = shoulderRight_RRJ - 2.5f;
		if(shoulderLeft_RRJ >= MAX_SHOULDER_MOVEMENT_RUNNING){
			iMoveFlag = 1;
			//printf("In 0: %f\n", shoulderLeft_RRJ);
			shoulderLeft_RRJ = MAX_SHOULDER_MOVEMENT_RUNNING;
			shoulderRight_RRJ = -MAX_SHOULDER_MOVEMENT_RUNNING;
		}


		//Left Elbow
		if(shoulderLeft_RRJ >= -MAX_SHOULDER_MOVEMENT_RUNNING)
			elbowLeft_RRJ = elbowLeft_RRJ + 1.5f;

		//Right Elbow
		if(shoulderRight_RRJ >= -MAX_SHOULDER_MOVEMENT_RUNNING && elbowRight_RRJ > 0.0f)
			elbowRight_RRJ = elbowRight_RRJ - 1.5f;



		//Right Thigh Forward
		thighRight_RRJ = thighRight_RRJ + 2.5f;
		thighLeft_RRJ = thighLeft_RRJ - 2.5f;



		if(thighRight_RRJ <= 0.0f)
			shinBoneRight_RRJ = shinBoneRight_RRJ - 5.0;

		if(thighRight_RRJ >= 0.0f && shinBoneRight_RRJ < 0.0f)
			shinBoneRight_RRJ = shinBoneRight_RRJ + 4.0f;

		if(thighLeft_RRJ <= 0.0f && shinBoneLeft_RRJ < 0.0f)
			shinBoneLeft_RRJ = shinBoneLeft_RRJ + 1.0f;


		
	}
	else if(iMoveFlag == 1){

		//Right Arm Forward
		shoulderLeft_RRJ = shoulderLeft_RRJ - 2.5f;
		shoulderRight_RRJ = shoulderRight_RRJ + 2.5f;
		if(shoulderRight_RRJ >= MAX_SHOULDER_MOVEMENT_RUNNING){
			iMoveFlag = 0;
			//printf("In 1: %f\n", shoulderRight_RRJ);
			shoulderLeft_RRJ = -MAX_SHOULDER_MOVEMENT_RUNNING;
			shoulderRight_RRJ = MAX_SHOULDER_MOVEMENT_RUNNING;
		}


		//Right Elbow
		if(shoulderRight_RRJ >= -MAX_SHOULDER_MOVEMENT_RUNNING)
			elbowRight_RRJ = elbowRight_RRJ + 1.5f;

		if(shoulderLeft_RRJ >= -MAX_SHOULDER_MOVEMENT_RUNNING && elbowLeft_RRJ > 0.0f)
			elbowLeft_RRJ = elbowLeft_RRJ - 1.5f;






		//Left Thigh Forward
		thighLeft_RRJ = thighLeft_RRJ + 2.5f;
		thighRight_RRJ = thighRight_RRJ - 2.5f;


		if(thighLeft_RRJ <= 0.0f)
			shinBoneLeft_RRJ = shinBoneLeft_RRJ - 5.0f;


		if(thighLeft_RRJ >= 0.0f && shinBoneLeft_RRJ < 0.0f)
			shinBoneLeft_RRJ = shinBoneLeft_RRJ + 4.0f;


		if(thighRight_RRJ <= 0.0f && shinBoneRight_RRJ < 0.0f)
			shinBoneRight_RRJ = shinBoneRight_RRJ + 1.0f;

	}

	body_XTranslation = 0.03f * cos(degToRad(90.0f + angle)) + body_XTranslation;
	body_ZTranslation = -(0.03f * sin(degToRad(90.0f + angle)) - body_ZTranslation);
	
	//printf("TL: %f/ TR: %f/ SBL: %f/ SBR: %f\n", thighLeft_RRJ, thighRight_RRJ, shinBoneLeft_RRJ, shinBoneRight_RRJ);
	//printf("SL:%f/ SR:%f/ EL:%f/ ER:%f\n", shoulderLeft_RRJ, shoulderRight_RRJ, elbowLeft_RRJ, elbowRight_RRJ);
}



void BodySwimming(void){


	/********** ARMS **********/
	shoulderLeft_RRJ = shoulderLeft_RRJ - 1.5f;
	shoulderRight_RRJ = shoulderRight_RRJ - 1.5f;
	
	if(shoulderLeft_RRJ < -360.0f)
		shoulderLeft_RRJ = 0.0f;

	if(shoulderRight_RRJ < -360.0f)
		shoulderRight_RRJ = 0.0f;


	if(iMoveFlag == 0){
		/********** LEGS **********/
		thighLeft_RRJ = thighLeft_RRJ + 1.0f;
		thighRight_RRJ = thighRight_RRJ - 1.0f;
		if(thighLeft_RRJ > MAX_THIGH_MOVEMENT_SWIMMING){
			thighLeft_RRJ = MAX_THIGH_MOVEMENT_SWIMMING;
			thighRight_RRJ = -MAX_THIGH_MOVEMENT_SWIMMING;
			iMoveFlag = 1;
		}
	}
	else if(iMoveFlag == 1){
		/********** LEGS **********/
		thighLeft_RRJ = thighLeft_RRJ - 1.0f;
		thighRight_RRJ = thighRight_RRJ + 1.0f;
		if(thighRight_RRJ > MAX_THIGH_MOVEMENT_SWIMMING){
			thighRight_RRJ = MAX_THIGH_MOVEMENT_SWIMMING;
			thighLeft_RRJ = -MAX_THIGH_MOVEMENT_SWIMMING;
			iMoveFlag = 0;
		}
	}

	body_XTranslation = 0.02f * cos(degToRad(90.0f + body_YRot_Angle)) + body_XTranslation;
	body_ZTranslation = -(0.02f * sin(degToRad(90.0f + body_YRot_Angle)) - body_ZTranslation);

}



void BodySquarting(void){

	if(iMoveFlag == 0){

		thighLeft_RRJ = thighLeft_RRJ + 1.0f;
		thighRight_RRJ = thighRight_RRJ + 1.0f;
		shinBoneLeft_RRJ = shinBoneLeft_RRJ - 1.0f;
		shinBoneRight_RRJ = shinBoneRight_RRJ - 1.0f;

		chest_RRJ = chest_RRJ - 0.25f;
		hip_RRJ = hip_RRJ - 0.25f;


		shoulderLeft_RRJ = shoulderLeft_RRJ + 0.25f;
		shoulderRight_RRJ = shoulderRight_RRJ + 0.25f;

		//printf("0: SL: %f      SR: %f\n", shoulderLeft_RRJ, shoulderRight_RRJ);


		body_YTranslation = body_YTranslation - 0.01f;
		
		if(thighLeft_RRJ >= MAX_THIGH_SHINBONE_MOVEMENT_SQUARTING)
			iMoveFlag = 1;

	}
	else if(iMoveFlag == 1){
		thighLeft_RRJ = thighLeft_RRJ - 1.0f;
		thighRight_RRJ = thighRight_RRJ - 1.0f;
		shinBoneLeft_RRJ = shinBoneLeft_RRJ + 1.0f;
		shinBoneRight_RRJ = shinBoneRight_RRJ + 1.0f;

		chest_RRJ = chest_RRJ + 0.25f;
		hip_RRJ = hip_RRJ + 0.25f;

		shoulderLeft_RRJ = shoulderLeft_RRJ - 0.25f;
		shoulderRight_RRJ = shoulderRight_RRJ - 0.25f;

		//printf("1: SL: %f      SR: %f\n", shoulderLeft_RRJ, shoulderRight_RRJ);

		body_YTranslation = body_YTranslation + 0.01f;
		//body_ZTranslation = body_ZTranslation - 0.01f;

		if(thighLeft_RRJ <= 0.0f){
			iMoveFlag = 0;
			body_YRot_Angle = body_YRot_Angle + 90.0f;
		}
	}
}



void BodySkipping(void){

	elbowLeft_RRJ = elbowLeft_RRJ - 4.0f;
	elbowRight_RRJ = elbowRight_RRJ - 4.0f;
	if(elbowLeft_RRJ < -360.0f)
		elbowLeft_RRJ = 0.0f;
	if(elbowRight_RRJ < -360.0f)
		elbowRight_RRJ = 0.0f;

	if(elbowLeft_RRJ <= -330.0f){
		//printf("0: %f\n", elbowLeft_RRJ);
		body_YTranslation = body_YTranslation + 0.05f;

		shinBoneLeft_RRJ = shinBoneLeft_RRJ - 4.0f;
		shinBoneRight_RRJ = shinBoneRight_RRJ - 4.0f;
		if(elbowLeft_RRJ <= -360.0f)
			iMoveFlag = 1;

		body_YRot_Angle = body_YRot_Angle + 2.0f;

	}
	else if(elbowLeft_RRJ <= -20.0f && iMoveFlag == 1 && elbowLeft_RRJ >= -50.0f){
		//printf("1: %f\n", elbowLeft_RRJ);
		body_YTranslation = body_YTranslation - 0.05f;
		
		shinBoneLeft_RRJ = shinBoneLeft_RRJ + 4.0f;
		shinBoneRight_RRJ = shinBoneRight_RRJ + 4.0f;

		if(elbowLeft_RRJ <= -45.0f){
			iMoveFlag = 0;
		}

		
		body_YRot_Angle = body_YRot_Angle + 2.0f;
	}

}

