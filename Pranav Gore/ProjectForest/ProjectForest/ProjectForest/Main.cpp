#include <windows.h>
#include <windowsx.h>

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "OpenGL32.lib")
// For Audio
#pragma comment(lib, "OpenAL32.lib")

#pragma comment(lib, "assimp-vc140-mt.lib")
#pragma comment(lib,"freetyped.lib")

#include <stdio.h>
#include <gl\glew.h>
#include <gl\wglew.h>
#include <gl\GL.h>
#include <vector>
#include <string>
#include <sstream>

// Header File For OpenGL
#include "Audio\OpenAL_Header.h"

void InitOpenALWithAllSources(void);
bool gbWalkingAudioFlag1 = false;
bool gbWalkingAudioFlag2 = false;
bool gbRunningAudioFlag1 = false;
bool gbRunningAudioFlag2 = false;


bool gbFinalWin = true;

// Audio End//#define GLM_FORCE_CUDA
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\quaternion.hpp>
#include <glm\gtc\random.hpp>
#include <glm\gtx\rotate_vector.hpp>
#include <glm\ext.hpp>
#include <glm\gtx\string_cast.hpp>
#include <glm\gtc\noise.hpp>

#include "Data.h"
#include "Camera/Camera.h"
#include "Camera/Player.h"
#include "LightsMaterial/LightsMaterial.h"
#include "Texture/Texture.h"
#include "ShaderLoad/LoadShaders.h"
#include "BasicShapes/BasicShapes.h"
#include "Skydome/Skydome.h"
#include "Grass/Grass.h"
#include "Terrain/Terrain.h"
#include "ModelLoad/Model.h"	//Model Loading
#include "Texture/Texture.h"
#include "Texture/Tga.h"
#include "Fonts/Fonts.h"
#include "Water/WindowSizeHandler.h"
#include "Water/Water.h"
#include "LetterTranslation.h"
//using namespace std;

bool gbEscapeKeyIsPressed = false;
bool gbIsFullscreen = false;
bool gbActiveWindow = false;

GLboolean gbIsFirstPlay = GL_FALSE;

void ToggleFullscreen(void);

GLfloat fExposure = 1.0f;
void FadeIn(GLfloat Delta);
void FadeOut(GLfloat Delta);
GLboolean gbIsFadeOutComplete = GL_FALSE;
GLboolean gbIsFadeInComplete = GL_TRUE;

GLboolean gbFirstFadeOut = GL_FALSE;
GLboolean gbFirstFadeIn = GL_FALSE;
GLboolean gbSecondFadeOut = GL_FALSE;

GLboolean gbFirstFadeInPreparationComplete = GL_FALSE;

GLfloat gfWaterHeight = WATER_HEIGHT;

void EndCreditScene(GLfloat Delta);

GLboolean gbIsSceneEnded = GL_FALSE;

GLboolean gbNameShown = GL_FALSE;

GLboolean gbIsTorchOn = GL_FALSE;

//BOOL gbDone = FALSE;

HWND ghWnd = NULL;
HDC ghDC = NULL;
HGLRC ghRC = NULL;

POINT ClickPoint, MovePoint, OldMovePoint;
GLboolean gbFirstMouse = GL_TRUE;

CCommon *pCCommon = NULL;
CCamera *pCCamera = NULL;
CPlayer *pCPlayer = NULL;
CSkyDome *pCSkyDome = NULL;
CGrass *pCGrass = NULL;
CTerrain *pCTerrain = NULL;
TGALoader *pTGA = NULL;
//CSphere *pCSkySphere = NULL;
Water *pWater = NULL;


CDirectionalLight *pSunLight = NULL;
CPointLight *pLamp = NULL;
CSpotLight *pTorch = NULL;

//CSphere *pSunSphere = NULL;

glm::vec3 CameraPosition;
glm::vec3 CameraFront;
glm::vec3 CameraUp;
glm::vec3 CameraLook;
glm::vec3 CameraRight;

GLboolean gbIsNight = GL_FALSE;
GLboolean gbIsDayNightCyclePaused = GL_FALSE;

DWORD dwStyle;

GLboolean gbIsPlaying = GL_FALSE;

WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
GLuint giWinHeight = 0;
GLuint giWinWidth = 0;
TCHAR str[255];

GLfloat gfLastX = giWinWidth / 2.0f;
GLfloat gfLastY = giWinHeight / 2.0f;

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

GLfloat gfFOVY = FOV_Y;

GLfloat gfAnglePyramid = 0.0f;
GLfloat gfAngleCube = 0.0f;

GLuint gShaderProgramObjectBasicShader;
GLuint gShaderProgramObjectBasicShaderTexture;

// HDR
GLuint gShaderObjectHDR;
GLuint gFrameBufferObjectHDR;
GLuint gRenderBufferObjectDepth;
GLuint gColorBufferTexture;

GLuint gVAOHDRQuad;
GLuint gVBOHDR;

void InitializeQuadHDR(void);
void RenderQuadHDR(void);

// sample texture for opacity map test
GLuint gVAOSampleQuad;
GLuint gVBOSampleQuad;

void InitializeSampleQuad(void);
void RenderSampleQuad(void);

GLuint DiffuseTexture;
GLuint OpacityTexture;

GLuint gShaderObjectSampleTexture;

// MSAA FBO
GLuint gShaderObjectMSAA;
GLuint gFBOMSAA;
GLuint gRBOMSAA;
GLuint gTextureMSAA;

// General
GLuint gVAOPyramid;
GLuint gVAOCube;
GLuint gVAOGround;
GLuint gVAOLineGrid;
GLuint gVAOGroundGrid;
GLuint gVAOAxes;
GLuint gVAOLightCube;

GLuint gVBOPosition;
GLuint gVBOColor;
GLuint gVBONormal;
GLuint gVBOTexture;



GLint NumGridVertices = 0;

GLuint gModelMatrixUniform, gViewMatrixUniform, gProjectionMatrixUniform, gViewPositionUniform, gMVPMatrixUniform;
glm::mat4 gPerspectiveProjectionMatrix;
glm::mat4 gPerspectiveProjectionMatrixFont;
glm::mat4 gOrthographicProjectionMatrix;

GLint Samples;


//Timing related
double gd_resolution;
unsigned __int64 gu64_base;

float gfDeltaTime = 0.0f;
float gfElapsedTime = 0.0f;
int giFrameCount = 0;
__int64 giPreviousTime = 0;
__int64 giCurrentTime = 0;
__int64 giCountsPerSecond;
float fSecondsPerCount;
int fps__;

float LastFrame = 0.0f;

RECT screen;

std::stringstream strLightProperty;

////////////////////////////////
////////////////////////////////
FILE *pFilePosition = NULL;
FILE *pFileNormal = NULL;
FILE *pFileTexCoord = NULL;
FILE *pFileIndices = NULL;
FILE *pFileGeneratedTriangle = NULL;
FILE *pFileTangent = NULL;
///////////////////////////////////////////////
// Sun related
GLuint BitmapTextureSun;
GLuint DepthTexture;
GLuint SunTextures[4];
GLuint gFBOSun;
GLuint gVBOSun;

GLuint gVAOScreenFilledQuad;
GLuint gVBOPositionScreenFilledQuad;
GLuint gVBOTexCoordsScreenFilledQuad;

GLuint gVertexShaderObjectSunDepthTest;
GLuint gVertexShaderObjectBlur;
GLuint gVertexShaderObjectBlurHorizontal;
GLuint gVertexShaderObjectBlurVertical;
GLuint gVertexShaderObjectSunRays;

GLuint gFragmentShaderObjectSunDepthTest;
GLuint gFragmentShaderObjectBlurHorizontal;
GLuint gFragmentShaderObjectBlurVertical;
GLuint gFragmentShaderObjectSunRays;

GLuint gShaderProgramObjectSunDepthTest;
GLuint gShaderProgramObjectBlurHorizontal;
GLuint gShaderProgramObjectBlurVertical;
GLuint gShaderProgramObjectSunRays;

// Variables for OpenAudio
bool gbIsShiftKeyPressed = false;
bool gbRun = false;

//Font shader obj
GLuint gShaderProgramObjectFont;
//Water Shader obj
GLuint gShaderProgramObjectWater;
//Sprites
GLuint gShaderProgramObjectSprites;


//Font
GLuint gVAOFonts;
GLuint gVBOFonts;
Fonts fonts;
float gFontTransition = 0.0f;
GLfloat gAlphaFont = 0.0f;
GLfloat gAlphaFontN = 1.0f;
UINT gisDisplayNames = 0;
UINT gisDisplayHelp = 0;
//Water
GLuint gVAOWater;
GLuint gVBOWaterPosition;
GLuint gVBOWaterIndices;



GLint giSunTextureWidth, giSunTextureHeight;

GLfloat ScreenFilledQuad[] = {
	0.0f, 0.0f,
	1.0f, 0.0f,
	1.0f, 1.0f,
	0.0f, 1.0f
};

GLfloat ScreenFilledQuadTexCoords[] = {
	0.0f, 0.0f,
	1.0f, 0.0f,
	1.0f, 1.0f,
	0.0f, 1.0f
};

GLuint gShaderObjectHemisphere;

///////////////////////////////////////////
//Model Related
//ModelLoad::Model *gpModel = NULL;
ModelLoad::Model *gpConifer = NULL;
ModelLoad::Model *gpBroadLeaf = NULL;
ModelLoad::Model *gpBroadLeaf2 = NULL;
ModelLoad::Model *gpBush = NULL;
FILE *gpFileModel = NULL;


//cameraPos.x: 47.675076 cameraPos.y : 35.303402 cameraPos.z : -108.227242
//cameraPos.x : 171.475708 cameraPos.y : 36.060661 cameraPos.z : -271.067841
//cameraPos.x : 151.684952 cameraPos.y : 33.861130 cameraPos.z : -221.195175
//cameraPos.x : 201.068909 cameraPos.y : 37.667992 cameraPos.z : -309.120239

glm::vec3 ConiferPosition = glm::vec3(40.0f, 0.0f, 40.0f);

glm::vec3 BroadleafPosition = glm::vec3(-40.0f, 0.0f, 40.0f);
glm::vec3 BroadleafPosition2 = glm::vec3(55.0f, 0.0f, -100.0f);
glm::vec3 BroadleafPosition3 = glm::vec3(170.0f, 0.0f, -13.0f);


glm::vec3 Broadleaf2Position = glm::vec3(-40.0f, 0.0f, -40.0f);
glm::vec3 BushPosition = glm::vec3(10.0f, 0.0f, 10.0f);

// Sujay--- AstroMediComp and Raster Letters
ModelLoad::Model *gpLetterA_obj = NULL;
ModelLoad::Model *gpLetterS_obj = NULL;
ModelLoad::Model *gpLetterT_obj = NULL;
ModelLoad::Model *gpLetterR_obj = NULL;
ModelLoad::Model *gpLetterO_obj = NULL;
ModelLoad::Model *gpLetterM_obj = NULL;
ModelLoad::Model *gpLetterE_obj = NULL;
ModelLoad::Model *gpLetterD_obj = NULL;
ModelLoad::Model *gpLetterI_obj = NULL;
ModelLoad::Model *gpLetterC_obj = NULL;
ModelLoad::Model *gpLetterO_2_obj = NULL;
ModelLoad::Model *gpLetterM_2_obj = NULL;
ModelLoad::Model *gpLetterP_obj = NULL;

ModelLoad::Model *gpLetterR_RG_obj = NULL;
ModelLoad::Model *gpLetterA_RG_obj = NULL;
ModelLoad::Model *gpLetterS_RG_obj = NULL;
ModelLoad::Model *gpLetterT_RG_obj = NULL;
ModelLoad::Model *gpLetterE_RG_obj = NULL;
ModelLoad::Model *gpLetterR_RG2_obj = NULL;
///////////////////////////////////////
// set the lighting data

void SetDayLight(GLuint ShaderObject);
void SetNightLight(GLint ShaderObject);


//main function
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int nCmdShow)
{
	MSG msg;
	bool bDone = false;

	void Initialize(HINSTANCE hInstance, int Samples);
	void InitializeTimer(void);
	void CalculateDelta(void);
	void InitializeOpenGL(void);
	void Uninitialize(void);
	void Display(void);
	void ToggleFullscreen(void);
	void Update(void);

	void CheckAsyncKeyDown(void);
	void CalculateFPS(void);

	pCCommon = new CCommon();

	if (fopen_s(&pCCommon->pLogFile, pCCommon->logfilepath, "w") != 0)
	{
		MessageBox(NULL, TEXT("Failed to create the log file. Exiting now."), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
	{
		fprintf_s(pCCommon->pLogFile, "Log file created successfully.\n");
	}

	// opening position file
	if (fopen_s(&pFilePosition, "Logs/01_Position.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Failed to create the position file. Exiting now."), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
	{
		fprintf_s(pCCommon->pLogFile, "position file created successfully.\n");
	}

	// opening normal file
	if (fopen_s(&pFileNormal, "Logs/02_Normal.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Failed to create the normal file. Exiting now."), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
	{
		fprintf_s(pCCommon->pLogFile, "normal file created successfully.\n");
	}

	// opening texcoord file
	if (fopen_s(&pFileTexCoord, "Logs/03_TexCoord.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Failed to create the texcoord file. Exiting now."), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
	{
		fprintf_s(pCCommon->pLogFile, "texcoord file created successfully.\n");
	}

	// opening indices
	if (fopen_s(&pFileIndices, "Logs/04_Indices.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Failed to create the indices file. Exiting now."), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
	{
		fprintf_s(pCCommon->pLogFile, "indices file created successfully.\n");
	}

	// opening tangent
	if (fopen_s(&pFileTangent, "Logs/05_Tangents.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Failed to create the tangents file. Exiting now."), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
	{
		fprintf_s(pCCommon->pLogFile, "tangents file created successfully.\n");
	}


	Initialize(hInstance, MSAA_SAMPLES);

	//display created window and update window
	ShowWindow(ghWnd, nCmdShow);
	SetForegroundWindow(ghWnd);
	SetFocus(ghWnd);
	ToggleFullscreen();

	InitializeTimer();

	fclose(pCCommon->pLogFile);
	InitializeOpenGL();

	// Game Loop
	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
				bDone = true;
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			CalculateDelta();
			CheckAsyncKeyDown();
			Update();
			Display();
			CalculateFPS();

			if (gbActiveWindow == true)
			{
				if (gbEscapeKeyIsPressed == true)
					bDone = true;
			}
		}
	}

	Uninitialize();
	return((int)msg.wParam);
}

void InitializeTimer(void)
{
	// Counts per second
	QueryPerformanceFrequency((LARGE_INTEGER *)&giCountsPerSecond);
	// seconds per count
	fSecondsPerCount = 1.0f / giCountsPerSecond;

	// previous time
	QueryPerformanceCounter((LARGE_INTEGER *)&giPreviousTime);
}

void CalculateDelta(void)
{
	// get current count
	QueryPerformanceCounter((LARGE_INTEGER *)&giCurrentTime);

	// Delta time
	gfDeltaTime = (giCurrentTime - giPreviousTime) * fSecondsPerCount;
	gfElapsedTime += gfDeltaTime;
}

void CalculateFPS(void)
{
	giFrameCount++;

	TCHAR str[255];

	// frames per second
	if (gfElapsedTime >= 1.0f)
	{
		fps__ = giFrameCount;

		swprintf_s(str, 255, TEXT("OpenGL Project FPS : %d"), fps__);
		SetWindowText(ghWnd, str);

		giFrameCount = 0;
		gfElapsedTime = 0.0f;
	}

	giPreviousTime = giCurrentTime;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	void Resize(int Width, int Height);
	void Uninitialize(void);

	// Sujay-- for picking astromedicomp letters
	void AMC_Game(void);

	void OnLButtonDown(INT, INT, DWORD);
	void OnLButtonUp(INT, INT, DWORD);
	void OnRButtonDown(INT, INT, DWORD);
	void OnRButtonUp(INT, INT, DWORD);
	void OnMouseMove(INT, INT, DWORD);
	void OnMButtonDown(INT, INT, DWORD);
	void OnMButtonUp(INT, INT, DWORD);
	void OnMouseWheelScroll(short zDelta);

	void OnUpArrowPress(void);
	void OnLeftArrowPress(void);
	void OnRightArrowPress(void);
	void OnDownArrowPress(void);
	void OnPageUpPress(void);
	void OnPageDownPress(void);

	void CheckAsyncKeyDown(void);


	static GLfloat floatval = 0.0f;
	static glm::vec3 vector;

	switch (iMsg)
	{
	case WM_CREATE:
		break;

	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			gbActiveWindow = true;
		else
			gbActiveWindow = false;
		break;

		//case WM_ERASEBKGND:
		//return(0);

	case WM_SIZE:
		giWinWidth = LOWORD(lParam);
		giWinHeight = HIWORD(lParam);
		Resize(giWinWidth, giWinHeight);
		//Resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_LBUTTONDOWN:
		OnLButtonDown(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		AMC_Game();
		break;

	case WM_LBUTTONUP:
		OnLButtonUp(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_RBUTTONDOWN:
		OnRButtonDown(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_RBUTTONUP:
		OnRButtonUp(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_MOUSEWHEEL:
		OnMouseWheelScroll(GET_WHEEL_DELTA_WPARAM(wParam));
		break;

	case WM_MOUSEMOVE:
		OnMouseMove(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_MBUTTONDOWN:
		OnMButtonDown(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_MBUTTONUP:
		OnMButtonUp(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_KEYDOWN:
		//CheckAsyncKeyDown();
		switch (wParam)
		{
		case VK_ESCAPE:
			gbEscapeKeyIsPressed = true;
			break;

		case VK_RETURN:
			if (gbIsFullscreen == false)
			{
				ToggleFullscreen();
				gbIsFullscreen = true;
			}
			else
			{
				ToggleFullscreen();
				gbIsFullscreen = false;
			}
			break;

		case 0x46:	//F key
			if (gbIsDayNightCyclePaused == GL_FALSE)
			{
				gbIsDayNightCyclePaused = GL_TRUE;
			}
			else
			{
				gbIsDayNightCyclePaused = GL_FALSE;
			}
			break;

			//case VK_UP:
			//	OnUpArrowPress();
			//	break;

			//case VK_DOWN:
			//	OnDownArrowPress();
			//	break;

			//case VK_LEFT:
			//	OnLeftArrowPress();
			//	break;

			//case VK_RIGHT:
			//	OnRightArrowPress();
			//	break;

			//	// PAGE UP key
			//case VK_PRIOR:
			//	OnPageUpPress();
			//	break;

			//	// PAGE DOWN KEY
			//case VK_NEXT:
			//	OnPageDownPress();
			//	break;

			// W key
		case 0x57:
			fopen_s(&pCCommon->pLogFile, "Logs/GeneralLog.txt", "a+");

			vector = pCPlayer->GetPlayerPosition();
			fprintf(pCCommon->pLogFile, "cameraPos.x: %f cameraPos.y: %f cameraPos.z: %f\n", vector.x, vector.y, vector.z);

			vector = pCPlayer->GetCameraLook();
			fprintf(pCCommon->pLogFile, "CameraLook: %s\n", glm::to_string(vector).c_str());

			vector = pCPlayer->GetCameraSide();
			fprintf(pCCommon->pLogFile, "CameraSide: %s\n", glm::to_string(vector).c_str());

			vector = pCPlayer->GetCameraUp();
			fprintf(pCCommon->pLogFile, "CameraUp: %s\n", glm::to_string(vector).c_str());

			vector = pCPlayer->GetCameraFront();
			fprintf(pCCommon->pLogFile, "CameraFront: %s\n", glm::to_string(vector).c_str());

			vector = pCPlayer->GetPlayerFront();
			fprintf(pCCommon->pLogFile, "PlayerFront: %s\n", glm::to_string(vector).c_str());

			floatval = pCSkyDome->GetSunCDirY();
			fprintf(pCCommon->pLogFile, "GetSunCDirY: %f\n", floatval);

			vector = pCSkyDome->GetSunWPos();
			fprintf(pCCommon->pLogFile, "GetSunWPos: %s\n", glm::to_string(vector).c_str());

			vector = pCSkyDome->GetSunCPos();
			fprintf(pCCommon->pLogFile, "GetSunCPos: %s\n", glm::to_string(vector).c_str());

			vector = pCSkyDome->GetSunCDir();
			fprintf(pCCommon->pLogFile, "GetSunCDir: %s\n", glm::to_string(vector).c_str());

			fclose(pCCommon->pLogFile);
			
			break;

			// S key
		case 0x53:
			if (gbUniversalLetterMoveFlag == FALSE)
			{
				//gbUniversalLetterMoveFlag = TRUE;
				gbLetterA_SetFlag = TRUE;
				gbLetterS_SetFlag = TRUE;
				gbLetterT_SetFlag = TRUE;
				gbLetterR_SetFlag = TRUE;
				gbLetterO_SetFlag = TRUE;
				gbLetterM_SetFlag = TRUE;
				gbLetterE_SetFlag = TRUE;
				gbLetterD_SetFlag = TRUE;
				gbLetterI_SetFlag = TRUE;
				gbLetterC_SetFlag = TRUE;
				gbLetterO_2_SetFlag = TRUE;
				gbLetterM_2_SetFlag = TRUE;
				gbLetterP_SetFlag = TRUE;
			}
			break;

			// D key
		case 0x41:
			break;

			// E key to end the scene
		case 0x45:
			if (gbIsSceneEnded == GL_FALSE)
			{
				StopAL_BackgroundSource();
				PlayAL_EndCredits();
				gbIsSceneEnded = GL_TRUE;
			}
			break;

			// A key
		case 0x44:
			break;

		case VK_NUMPAD8:
			break;

		case VK_NUMPAD2:
			break;

		case VK_NUMPAD4:
			break;

		case VK_NUMPAD6:
			break;

		case VK_NUMPAD7:
			break;

		case VK_NUMPAD1:
			break;

			// E key
		//case 0x45:
			//cam.Roll(-1.0);
			//break;

			// Q key
		case 0x51:
			//cam.Roll(1.0);
			break;
		case 0x30:
			if (gisDisplayNames == 0)
				gisDisplayNames = 1;
			else
				gisDisplayNames = 0;
			break;

		case 0x70: //F1
			if (gisDisplayHelp == 0)
				gisDisplayHelp = 1;
			else
				gisDisplayHelp = 0;
			break;

			// P key
		case 0x50:
			if (gbIsPlaying == GL_FALSE)
			{
				if (gbIsFirstPlay == GL_FALSE)
				{
					PlayAL_BackgroundSource();
					gbIsFirstPlay = GL_TRUE;
				}

				gbIsPlaying = GL_TRUE;

				if (gbIsFullscreen == false)
				{
					ToggleFullscreen();
					gbIsFullscreen = true;
				}

				GetWindowRect(ghWnd, &screen);
				SetCursorPos((int)(screen.right * 0.5), (int)(screen.bottom * 0.5));
				ShowCursor(FALSE);
			}
			else
			{
				gbIsPlaying = GL_FALSE;
				ShowCursor(TRUE);
			}
			break;

			// L key
		case 0x4C:
			if (gbIsTorchOn == GL_FALSE)
			{
				gbIsTorchOn = GL_TRUE;
				//PlayAL_Torch();
			}
			else
				gbIsTorchOn = GL_FALSE;
			break;

		case VK_ADD:
			break;

		case VK_SUBTRACT:
			break;

		case VK_CONTROL:
			if (TOGGLE_CROUCH == TRUE)
			{
				if (pCPlayer->GetIsCrouching() == GL_TRUE)
				{
					pCPlayer->SetIsCrouching(GL_FALSE);
				}
				else if (pCPlayer->GetIsCrouching() == GL_FALSE)
				{
					pCPlayer->SetIsCrouching(GL_TRUE);
				}
			}
			break;

		case VK_SHIFT:

			/// For Audio Pause and Start
			if (gbIsShiftKeyPressed == false)
			{
				gbRun = true;
				gbIsShiftKeyPressed = true;
			}
			else if (gbIsShiftKeyPressed == true)
			{
				gbRun = false;
				gbIsShiftKeyPressed = false;
			}
			// Audio End

			if (TOGGLE_SPRINT == TRUE)
			{
				if (pCPlayer->GetIsSprinting() == GL_TRUE)
				{
					pCPlayer->SetIsSprinting(GL_FALSE);
				}
				else if (pCPlayer->GetIsSprinting() == GL_FALSE)
				{
					pCPlayer->SetIsSprinting(GL_TRUE);
				}
			}
			break;

		default:
			break;
		}
		break;

	case WM_CLOSE:
		Uninitialize();
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	default:
		break;
	}

	return(DefWindowProc(hWnd, iMsg, wParam, lParam));
}

void CheckAsyncKeyDown(void)
{
	if (GetAsyncKeyState(VK_UP) & 0x8000)
	{
		if (gbIsPlaying == GL_FALSE)
			pCCamera->ProcessNavigationKeys(CameraMovement::CAMERA_FORWARD, gfDeltaTime);
		else if (gbIsPlaying == GL_TRUE)
			pCPlayer->ControlPlayerMovement(PlayerMovement::PLAYER_FORWARD, gfDeltaTime);
	}

	if (GetAsyncKeyState(VK_DOWN) & 0x8000)
	{
		if (gbIsPlaying == GL_FALSE)
			pCCamera->ProcessNavigationKeys(CameraMovement::CAMERA_BACKWARD, gfDeltaTime);
		else if (gbIsPlaying == GL_TRUE)
			pCPlayer->ControlPlayerMovement(PlayerMovement::PLAYER_BACKWARD, gfDeltaTime);
	}

	if (GetAsyncKeyState(VK_LEFT) & 0x8000)
	{
		if (gbIsPlaying == GL_FALSE)
			pCCamera->ProcessNavigationKeys(CameraMovement::CAMERA_LEFT, gfDeltaTime);
		else if (gbIsPlaying == GL_TRUE)
			pCPlayer->ControlPlayerMovement(PlayerMovement::PLAYER_LEFT, gfDeltaTime);
	}

	if (GetAsyncKeyState(VK_RIGHT) & 0x8000)
	{
		if (gbIsPlaying == GL_FALSE)
			pCCamera->ProcessNavigationKeys(CameraMovement::CAMERA_RIGHT, gfDeltaTime);
		else if (gbIsPlaying == GL_TRUE)
			pCPlayer->ControlPlayerMovement(PlayerMovement::PLAYER_RIGHT, gfDeltaTime);
	}

	if (GetAsyncKeyState(VK_PRIOR) & 0x8000)
	{
		pCCamera->ProcessNavigationKeys(CameraMovement::CAMERA_UP, gfDeltaTime);
	}

	if (GetAsyncKeyState(VK_NEXT) & 0x8000)
	{
		pCCamera->ProcessNavigationKeys(CameraMovement::CAMERA_DOWN, gfDeltaTime);
	}

	if (GetAsyncKeyState(VK_SPACE) & 0x8000)
	{
		if (gbIsPlaying == GL_FALSE)
		{
			// code
		}
		else if (gbIsPlaying == GL_TRUE && pCPlayer->GetIsTouchingGround() == GL_TRUE)
			pCPlayer->ControlPlayerMovement(PlayerMovement::PLAYER_JUMP, gfDeltaTime);
	}

	if (TOGGLE_CROUCH == FALSE)
	{
		if (GetAsyncKeyState(VK_CONTROL))
		{
			if (pCPlayer->GetIsCrouching() == GL_FALSE)
			{
				pCPlayer->SetIsCrouching(GL_TRUE);
			}
		}
		else
		{
			if (pCPlayer->GetIsCrouching() == GL_TRUE)
			{
				pCPlayer->SetIsCrouching(GL_FALSE);
			}
		}
	}

	if (TOGGLE_SPRINT == FALSE)
	{
		if (GetAsyncKeyState(VK_SHIFT))
		{
			if (pCPlayer->GetIsSprinting() == GL_FALSE)
			{
				pCPlayer->SetIsSprinting(GL_TRUE);
			}
		}
		else
		{
			if (pCPlayer->GetIsSprinting() == GL_TRUE)
			{
				pCPlayer->SetIsSprinting(GL_FALSE);
			}
		}
	}

	if (pCPlayer->GetIsSprinting() == GL_TRUE)
	{
		if (gbRun == true)
		{
			if ((GetAsyncKeyState(VK_UP) & 0X8000) || (GetAsyncKeyState(VK_DOWN) & 0X8000) || (GetAsyncKeyState(VK_LEFT) & 0X8000) || (GetAsyncKeyState(VK_RIGHT) & 0X8000))
			{
				gbRunningAudioFlag1 = true;
				if (gbRunningAudioFlag1 == true && gbRunningAudioFlag2 == true)
				{
					PlayAL_RunningSource();
					PauseAL_WalkingSource();
					gbRunningAudioFlag2 = false;
				}
			}
			else
			{
				gbRunningAudioFlag1 = false;
				gbRunningAudioFlag2 = true;
				PauseAL_RunningSource();
			}
		}
		else if (gbRun == false)
		{
			if ((GetAsyncKeyState(VK_UP) & 0X8000) || (GetAsyncKeyState(VK_DOWN) & 0X8000) || (GetAsyncKeyState(VK_LEFT) & 0X8000) || (GetAsyncKeyState(VK_RIGHT) & 0X8000))
			{
				gbWalkingAudioFlag1 = true;
				if (gbWalkingAudioFlag1 == true && gbWalkingAudioFlag2 == true)
				{
					PlayAL_WalkingSource();
					PauseAL_RunningSource();
					gbWalkingAudioFlag2 = false;
				}
			}
			else
			{
				gbWalkingAudioFlag1 = false;
				gbWalkingAudioFlag2 = true;
				PauseAL_WalkingSource();
			}
		}
	}
	else
	{
		if (GetAsyncKeyState(VK_SHIFT))
		{
			if ((GetAsyncKeyState(VK_UP) & 0X8000) || (GetAsyncKeyState(VK_DOWN) & 0X8000) || (GetAsyncKeyState(VK_LEFT) & 0X8000) || (GetAsyncKeyState(VK_RIGHT) & 0X8000))
			{
				gbRunningAudioFlag1 = true;
				if (gbRunningAudioFlag1 == true && gbRunningAudioFlag2 == true)
				{
					PlayAL_RunningSource();
					PauseAL_WalkingSource();
					gbRunningAudioFlag2 = false;
				}
			}
			else
			{
				gbRunningAudioFlag1 = false;
				gbRunningAudioFlag2 = true;
				PauseAL_RunningSource();
			}
		}
		else
		{
			if ((GetAsyncKeyState(VK_UP) & 0X8000) || (GetAsyncKeyState(VK_DOWN) & 0X8000) || (GetAsyncKeyState(VK_LEFT) & 0X8000) || (GetAsyncKeyState(VK_RIGHT) & 0X8000))
			{
				gbWalkingAudioFlag1 = true;
				if (gbWalkingAudioFlag1 == true && gbWalkingAudioFlag2 == true)
				{
					PlayAL_WalkingSource();
					PauseAL_RunningSource();
					gbWalkingAudioFlag2 = false;
				}
			}
			else
			{
				gbWalkingAudioFlag1 = false;
				gbWalkingAudioFlag2 = true;
				PauseAL_WalkingSource();
			}
		}
	}

	if (gbIsPlaying == GL_TRUE)
	{
		pCPlayer->ControlMouseInput();
		SetCursorPos((int)(screen.right * 0.5), (int)(screen.bottom * 0.5));
	}

	if (gbLetterGameOver == TRUE && gbFinalWin == true)
	{
		PlayAL_FinalWinSource();
		gbFinalWin = false;
	}
}

void ToggleFullscreen(void)
{
	HMONITOR hMonitor;
	MONITORINFO mi;
	BOOL bWindowPlacement;
	BOOL bMonitorInfo;

	if (gbIsFullscreen == FALSE)
	{
		dwStyle = GetWindowLong(ghWnd, GWL_STYLE);
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{

			mi.cbSize = { sizeof(MONITORINFO) };

			bWindowPlacement = GetWindowPlacement(ghWnd, &wpPrev);
			hMonitor = MonitorFromWindow(ghWnd, MONITOR_DEFAULTTOPRIMARY);
			bMonitorInfo = GetMonitorInfo(hMonitor, &mi);

			if (bWindowPlacement == TRUE && bMonitorInfo == TRUE)
			{
				SetWindowLong(ghWnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghWnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
	}
	else
	{
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghWnd, &wpPrev);
		SetWindowPos(ghWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
	}
}

void Initialize(HINSTANCE hInstance, int Samples)
{

	//variable declarations
	WNDCLASSEX wndclass;
	HWND hWnd;
	TCHAR szClassName[] = TEXT("RTROpenGLProject");
	TCHAR szAppName[] = TEXT("A Walk in the Fields");

	void Initialize(HINSTANCE hInstance, int Samples);

	//WNDCLASSEX initialization
	wndclass.cbClsExtra = 0;
	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.cbWndExtra = 0;
	wndclass.hbrBackground = (HBRUSH)GetStockObject(GRAY_BRUSH);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hInstance = hInstance;
	wndclass.lpfnWndProc = WndProc;
	wndclass.lpszClassName = szClassName;
	wndclass.lpszMenuName = NULL;
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;

	//register wndclass
	RegisterClassEx(&wndclass);

	int iWidth = GetSystemMetrics(SM_CXSCREEN);
	int iHeight = GetSystemMetrics(SM_CYSCREEN);

	int iXFinalPosition = (iWidth / 2) - (WIN_WIDTH / 2);
	int iYFinalPosition = (iHeight / 2) - (WIN_HEIGHT / 2);

	//create window
	hWnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, szAppName, WS_OVERLAPPEDWINDOW, iXFinalPosition, iYFinalPosition, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);
	ghWnd = hWnd;

	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

	ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32;
	pfd.iLayerType = PFD_MAIN_PLANE;

	ghDC = GetDC(ghWnd);

	iPixelFormatIndex = ChoosePixelFormat(ghDC, &pfd);
	if (iPixelFormatIndex == 0)
	{
		fprintf(pCCommon->pLogFile, "ChoosePixelFormat() Error : Pixel Format Index is 0\n");
		//fclose(Essential.gpFile);
		//Essential.gpFile = NULL;

		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
	}

	static int MSAAPixelFormat = 0;

	if (SetPixelFormat(ghDC, MSAAPixelFormat == 0 ? iPixelFormatIndex : MSAAPixelFormat, &pfd) == false)
	{
		fprintf(pCCommon->pLogFile, "SetPixelFormat() Error : Failed to set the pixel format\n");
		//fclose(Essential.gpFile);
		//Essential.gpFile = NULL;

		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
	}

	ghRC = wglCreateContext(ghDC);
	if (ghRC == NULL)
	{
		fprintf(pCCommon->pLogFile, "wglCreateContext() Error : Rendering context ghRC is NULL\n");
		//fclose(Essential.gpFile);
		//Essential.gpFile = NULL;

		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
	}

	if (wglMakeCurrent(ghDC, ghRC) == false)
	{
		fprintf(pCCommon->pLogFile, "wglMakeCurrent() Error : wglMakeCurrent() Failed to set rendering context as current context\n");
		//fclose(Essential.gpFile);
		//Essential.gpFile = NULL;

		wglDeleteContext(ghRC);
		ghRC = NULL;
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
	}

	GLenum glew_error = glewInit();
	if (glew_error != GLEW_OK)
	{
		wsprintf(str, TEXT("Error: %s\n"), glewGetErrorString(glew_error));
		MessageBox(ghWnd, str, TEXT("GLEW Error"), MB_OK);

		wglDeleteContext(ghRC);
		ghRC = NULL;
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
	}

	//MessageBox(ghWnd, TEXT("Before if condition"), TEXT("Message"), MB_OK | MB_TOPMOST);
	if (MSAAPixelFormat == 0 && Samples > 0)
	{
		//MessageBox(ghWnd, TEXT("Inside first if condition"), TEXT("Message"), MB_OK | MB_TOPMOST);
		if (WGLEW_ARB_pixel_format && GLEW_ARB_multisample)
		{
			//MessageBox(ghWnd, TEXT("Inside second if condition"), TEXT("Message"), MB_OK | MB_TOPMOST);
			while (Samples > 0)
			{
				//MessageBox(ghWnd, TEXT("Inside while condition"), TEXT("Message"), MB_OK | MB_TOPMOST);

				UINT NumFormats = 0;

				int PFAttribs[] =
				{
					WGL_DRAW_TO_WINDOW_ARB, GL_TRUE,
					WGL_SUPPORT_OPENGL_ARB, GL_TRUE,
					WGL_DOUBLE_BUFFER_ARB, GL_TRUE,
					WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB,
					WGL_COLOR_BITS_ARB, 32,
					WGL_DEPTH_BITS_ARB, 24,
					WGL_ACCELERATION_ARB, WGL_FULL_ACCELERATION_ARB,
					WGL_SAMPLE_BUFFERS_ARB, GL_TRUE,
					WGL_SAMPLES_ARB, Samples,
					0
				};

				if (wglChoosePixelFormatARB(ghDC, PFAttribs, NULL, 1, &MSAAPixelFormat, &NumFormats) == TRUE && NumFormats > 0)
				{
					//MessageBox(ghWnd, TEXT("Inside third if condition"), TEXT("Message"), MB_OK | MB_TOPMOST);
					break;
				}
				Samples--;
			}

			//MessageBox(ghWnd, TEXT("Out of while condition"), TEXT("Message"), MB_OK | MB_TOPMOST);

			//wglDeleteContext(ghRC);
			//DestroyWindow(hWnd);
			//UnregisterClass(wndclass.lpszClassName, hInstance);
			return(Initialize(hInstance, Samples));
		}
		else
		{
			//MessageBox(ghWnd, TEXT("Inside else condition"), TEXT("Message"), MB_OK | MB_TOPMOST);
			Samples = 0;
		}
	}

	if (!GLEW_ARB_texture_non_power_of_two)
	{
		fprintf(pCCommon->pLogFile, "GL_ARB_texture_non_power_of_two not supported!\n");
	}

	if (!GLEW_ARB_depth_texture)
	{
		fprintf(pCCommon->pLogFile, "GLEW_ARB_depth_texture not supported!\n");
	}

	if (!GLEW_EXT_framebuffer_object)
	{
		fprintf(pCCommon->pLogFile, "GLEW_EXT_framebuffer_object not supported!\n");
	}
}

void InitializeOpenGL(void)
{
	void Uninitialize(void);
	void Resize(int, int);
	void InitializeLightCube(void);
	void InitializeModels(void);

	pSunLight = new CDirectionalLight();
	pLamp = new CPointLight[MAX_POINT_LIGHTS];
	pTorch = new CSpotLight();
	pCCamera = new CCamera();
	pCTerrain = new CTerrain();
	pCPlayer = new CPlayer(glm::vec3(0.0f, pCTerrain->GetHeightAt(glm::vec3(0.0f, 0.0f, 0.0f)), 0.0f), pCTerrain);
	pCSkyDome = new CSkyDome();
	pCGrass = new CGrass();
	//pCSkySphere = new CSphere();
	pTGA = new TGALoader();

	ShaderInfo BasicShader[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/BasicShader.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/BasicShader.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	gShaderProgramObjectBasicShader = LoadShaders(BasicShader);

	ShaderInfo BasicShaderTexture[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/BasicShaderTexture.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/BasicShaderTexture.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	gShaderProgramObjectBasicShaderTexture = LoadShaders(BasicShaderTexture);

	ShaderInfo SampleTextureShader[] = 
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/SampleTexture.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/SampleTexture.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	gShaderObjectSampleTexture = LoadShaders(SampleTextureShader);

	ShaderInfo SkyShader[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/AtmosphericScatter.vert.glsl", 0 },
	{ GL_FRAGMENT_SHADER, "Resources/Shaders/AtmosphericScatter.frag.glsl", 0 },
	{ GL_NONE, NULL, 0 }
	};

	ShaderInfo TerrainDay[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/TerrainDayTimeSugat.vert.glsl", 0 },
	{ GL_FRAGMENT_SHADER, "Resources/Shaders/TerrainDayTimeSugat.frag.glsl", 0 },
	{ GL_NONE, NULL, 0 }
	};

	ShaderInfo TerrainNight[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/TerrainNightMultilights.vert.glsl", 0 },
	{ GL_FRAGMENT_SHADER, "Resources/Shaders/TerrainNightMultilights.frag.glsl", 0 },
	{ GL_NONE, NULL, 0 }
	};

	/*ShaderInfo TerrainNormalMap[] =
	{
	{ GL_VERTEX_SHADER, "Resources/Shaders/TerrainNightTimeSinglePointLightNormalMap.vert.glsl", 0 },
	{ GL_FRAGMENT_SHADER, "Resources/Shaders/TerrainNightTimeSinglePointLightNormalMap.frag.glsl", 0 },
	{ GL_NONE, NULL, 0 }
	};

	gShaderProgramObjectTerrainNightSinglePointNormalMap = LoadShaders(TerrainNormalMap);*/

	ShaderInfo GrassDayLight[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/GrassDayLight.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/GrassDayLight.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	ShaderInfo GrassNight[] = 
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/GrassDayNight.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/GrassDayNight.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	ShaderInfo SunDepthTest[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/SunDepthTest.vert.glsl", 0 },
	{ GL_FRAGMENT_SHADER, "Resources/Shaders/SunDepthTest.frag.glsl", 0 },
	{ GL_NONE, NULL, 0 }
	};

	gShaderProgramObjectSunDepthTest = LoadShaders(SunDepthTest);

	ShaderInfo BlurHorizontal[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/Blur.vert.glsl", 0 },
	{ GL_FRAGMENT_SHADER, "Resources/Shaders/BlurHorizontal.frag.glsl", 0 },
	{ GL_NONE, NULL, 0 }
	};

	gVertexShaderObjectBlurHorizontal = LoadShaders(BlurHorizontal);

	ShaderInfo BlurVertical[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/Blur.vert.glsl", 0 },
	{ GL_FRAGMENT_SHADER, "Resources/Shaders/BlurVertical.frag.glsl", 0 },
	{ GL_NONE, NULL, 0 }
	};

	gVertexShaderObjectBlurHorizontal = LoadShaders(BlurVertical);

	ShaderInfo SunRays[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/SunRays.vert.glsl", 0 },
	{ GL_FRAGMENT_SHADER, "Resources/Shaders/SunRays.frag.glsl", 0 },
	{ GL_NONE, NULL, 0 }
	};

	gShaderProgramObjectSunRays = LoadShaders(SunRays);

	ShaderInfo HDR[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/HDR.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/HDR.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	gShaderObjectHDR = LoadShaders(HDR);

	ShaderInfo HemisphereLight[] = 
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/HemisphereLight.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/HemisphereLight.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	gShaderObjectHemisphere = LoadShaders(HemisphereLight);

	ShaderInfo waterShaderInfo[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/WaterVertexShaderSugat.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/WaterFragmentShaderSugat.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	gShaderProgramObjectWater = LoadShaders(waterShaderInfo);

	ShaderInfo fontShaderInfo[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/FontsvertexShaderSugat.vert.glsl", 0 },
	{ GL_FRAGMENT_SHADER, "Resources/Shaders/FontsfragmentShaderSugat.frag.glsl", 0 },
	{ GL_NONE, NULL, 0 }
	};

	fonts.LoadFontTexture("Resources/Fonts/IMPRISHA.TTF");
	gShaderProgramObjectFont = LoadShaders(fontShaderInfo);

	// For font
	glGenVertexArrays(1, &gVAOFonts);
	glBindVertexArray(gVAOFonts);

	// For font position
	glGenBuffers(1, &gVBOFonts);
	glBindBuffer(GL_ARRAY_BUFFER, gVBOFonts);
	//The 2D quad requires 6 vertices of 4 floats each so we reserve 6 * 4 floats of memory. 
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 6 * 4, NULL, GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(OGL_ATTRIBUTE_VERTEX);
	glVertexAttribPointer(OGL_ATTRIBUTE_VERTEX, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	//water data
	float positions[] = {
		-1.0f, 0.0f, 1.0f,
		-1.0f, 0.0f, -1.0f,
		1.0f, 0.0f, -1.0f,
		1.0f, 0.0f, 1.0f,
	};
	//int indices[] = { 0, 1, 3, 3, 1, 2 };
	int indices[] = { 0, 3, 1, 1, 3, 2 };
	// For water vao vbo
	glGenVertexArrays(1, &gVAOWater);
	glBindVertexArray(gVAOWater);

	glGenBuffers(1, &gVBOWaterPosition);
	// TODO add vbo to list so it can be cleared later when the program closes
	glBindBuffer(GL_ARRAY_BUFFER, gVBOWaterPosition);
	glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), positions, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_VERTEX);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &gVBOWaterIndices);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVBOWaterIndices);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(int), indices, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	pWater = new Water(pCTerrain, pCPlayer);


	// initialize light values
	pSunLight->SetAmbient(glm::vec3(0.2f, 0.2f, 0.2f));
	pSunLight->SetDiffuse(glm::vec3(1.0f, 1.0f, 1.0f));
	pSunLight->SetSpecular(glm::vec3(1.0f, 1.0f, 1.0f));
	pSunLight->SetDirection(glm::vec3(0.0f, 1.0f, 0.0f));

	pLamp[0].SetAmbient(glm::vec3(0.980f * 0.1f, 0.980f * 0.1f, 0.824f * 0.1f));
	pLamp[0].SetDiffuse(glm::vec3(0.980f, 0.980f, 0.824f)); // Light Golden Yellow Rod
	pLamp[0].SetSpecular(glm::vec3(0.980f, 0.980f, 0.824f));
	pLamp[0].SetPosition(glm::vec3(15.0f, 35.0f, 15.0f));
	pLamp[0].SetAttenuation(1.0f, 0.07f, 0.017f); // 65 Units

	pLamp[1].SetAmbient(glm::vec3(0.980f * 0.1f, 0.980f * 0.1f, 0.824f * 0.1f));
	pLamp[1].SetDiffuse(glm::vec3(0.980f, 0.980f, 0.824f)); // Light Golden Yellow Rod
	pLamp[1].SetSpecular(glm::vec3(0.980f, 0.980f, 0.824f));
	pLamp[1].SetPosition(glm::vec3(70.0f, 35.0f, 50.0f));
	pLamp[1].SetAttenuation(1.0f, 0.07f, 0.017f); // 65 Units

	pTorch->SetAmbient(glm::vec3(0.980f * 0.1f, 0.980f * 0.1f, 0.824f * 0.1f));
	pTorch->SetDiffuse(glm::vec3(0.980f, 0.980f, 0.824f));
	pTorch->SetSpecular(glm::vec3(0.980f, 0.980f, 0.824f));
	pTorch->SetPosition(glm::vec3(0.0f, 0.0f, 0.0f));
	pTorch->SetDirection(glm::vec3(0.0f, 0.0f, 0.0f));

	pTorch->SetAttenuation(1.0f, 0.022f, 0.0019f);	// 200 Units
	//pTorch->SetAttenuation(1.0f, 0.22f, 0.20f);	// 20 Units
	pTorch->SetSpotCutOff(15.0f, 20.0f);

	// for debug light probes
	InitializeLightCube();
	InitializeQuadHDR();

	pCTerrain->LoadHeightMap("Resources/Raw/terrain0-16bbp-257x257.raw", 16, 257, 257);
	pCTerrain->InitializeTerrain(LoadShaders(TerrainDay), LoadShaders(TerrainNight));

	///////////////////////////////////////////
	// Grass initilization
	//pCGrass->InitGrass(pCPlayer, pCTerrain, LoadShaders(GrassBasic));
	pCGrass->InitGrass(pCPlayer, pCTerrain, LoadShaders(GrassDayLight), LoadShaders(GrassNight));
	///////////////////////////////////////////

	//////////////////////////////////////
	// Sky initialization

	pCSkyDome->InitializeSky(LoadShaders(SkyShader));
	//pCSkySphere->InitializeSphere(10.0f, 25, 30);
	///////////////////////////////////////////////////////////////////

	//////////////////////////////////////
	//Model Loading
	InitializeModels();

	// load sampletextures
	InitializeSampleQuad();
	//DiffuseTexture = pTGA->LoadTGATexture("Sample/PalmFrond_Mat_Color.tga");
	//DiffuseTexture = pTGA->LoadTGATexture("Sample/Diffuse.tga");
	DiffuseTexture = pTGA->LoadTGATexture("Sample/Diffuse2.tga");
	//OpacityTexture = pTGA->LoadTGATexture("Sample/PalmFrond_Mat_Opacity.tga");
	OpacityTexture = pTGA->LoadTGATexture("Sample/Opacity.tga");
	//////////////////////////////////////

	// initialize screen filled quad
	glGenVertexArrays(1, &gVAOScreenFilledQuad);
	glBindVertexArray(gVAOScreenFilledQuad);

	// For position
	glGenBuffers(1, &gVBOPositionScreenFilledQuad);
	glBindBuffer(GL_ARRAY_BUFFER, gVBOPositionScreenFilledQuad);
	glBufferData(GL_ARRAY_BUFFER, sizeof(ScreenFilledQuad), ScreenFilledQuad, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_VERTEX, 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(OGL_ATTRIBUTE_VERTEX);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// For Texture
	glGenBuffers(1, &gVBOTexCoordsScreenFilledQuad);
	glBindBuffer(GL_ARRAY_BUFFER, gVBOTexCoordsScreenFilledQuad);
	glBufferData(GL_ARRAY_BUFFER, sizeof(ScreenFilledQuadTexCoords), ScreenFilledQuadTexCoords, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	//////////////////////////////////////

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Sun rendering
	glUseProgram(gShaderProgramObjectSunDepthTest);
	glUniform1i(glGetUniformLocation(gShaderProgramObjectSunDepthTest, "SunTexture"), 0);
	glUniform1i(glGetUniformLocation(gShaderProgramObjectSunDepthTest, "DepthTexture"), 1);
	glUseProgram(0);

	glUseProgram(gShaderProgramObjectSunRays);
	glUniform1i(glGetUniformLocation(gShaderProgramObjectSunRays, "LowBlurredSunTexture"), 0);
	glUniform1i(glGetUniformLocation(gShaderProgramObjectSunRays, "HighBlurredSunTexture"), 1);
	glUniform1f(glGetUniformLocation(gShaderProgramObjectSunRays, "Intensity"), 2.25f);
	glUniform3fv(glGetUniformLocation(gShaderProgramObjectSunRays, "Distortion"), 1, glm::value_ptr(glm::vec3(0.94f, 0.97f, 1.0f)));
	glUseProgram(0);


	glGenFramebuffers(1, &gFBOSun);
	glGenTextures(1, &DepthTexture);
	glGenTextures(4, SunTextures);



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///////////////////////////////////////
	// Create MSAA FBO
	glGenFramebuffers(1, &gFBOMSAA);
	glBindFramebuffer(GL_FRAMEBUFFER, gFBOMSAA);

	// create a multisampled color attachment
	glGenTextures(1, &gTextureMSAA);
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, gTextureMSAA);
	glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, MSAA_SAMPLES, GL_RGB, giWinWidth, giWinHeight, GL_TRUE);
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, gTextureMSAA, 0);

	glGenRenderbuffers(1, &gRBOMSAA);
	glBindRenderbuffer(GL_RENDERBUFFER, gRBOMSAA);
	glRenderbufferStorageMultisample(GL_RENDERBUFFER, MSAA_SAMPLES, GL_DEPTH24_STENCIL8, giWinWidth, giWinHeight);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, gRBOMSAA);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		fopen_s(&pCCommon->pLogFile, "Logs/GeneralLog.txt", "a+");
		fprintf(pCCommon->pLogFile, "Framebuffer not complete msaa!\n");
		fclose(pCCommon->pLogFile);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	////////////////////////////////////////
	/// Creating HDR framebuffers

	// configure floating point frame buffer
	glGenFramebuffers(1, &gFrameBufferObjectHDR);
	glBindFramebuffer(GL_FRAMEBUFFER, gFrameBufferObjectHDR);

	glGenTextures(1, &gColorBufferTexture);
	glBindTexture(GL_TEXTURE_2D, gColorBufferTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, giWinWidth, giWinHeight, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// create depth buffer (renderbuffer)
	glGenRenderbuffers(1, &gRenderBufferObjectDepth);
	glBindRenderbuffer(GL_RENDERBUFFER, gRenderBufferObjectDepth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, giWinWidth, giWinHeight);
	//glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH32F_STENCIL8, giWinWidth, giWinHeight);

	// attach buffers
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gColorBufferTexture, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, gRenderBufferObjectDepth);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		fopen_s(&pCCommon->pLogFile, "Logs/GeneralLog.txt", "a+");
		fprintf(pCCommon->pLogFile, "Framebuffer not complete!\n");
		fclose(pCCommon->pLogFile);
	}
	


	//glClearColor(0.3f, 0.0f, 0.5f, 1.0f);


	gPerspectiveProjectionMatrix = glm::mat4(1.0);
	gOrthographicProjectionMatrix = glm::mat4(1.0);

	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);

	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	glEnable(GL_CULL_FACE);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_TEXTURE_2D);


	// *************************** OpenAL *************************************
	InitOpenALWithAllSources();

	
	PlayAL_AmbienceSource();
	// ************************************************************************

	Resize(giWinWidth, giWinHeight);
	//Resize(WIN_WIDTH, WIN_HEIGHT);
}

void Display(void)
{
	void RenderScene(void);

	RenderScene();

	SwapBuffers(ghDC);
}

void RenderScene(void)
{
	
	void DrawPyramid(void);
	void DrawCube(void);
	void DrawGrid(void);
	void DrawGround(void);
	void DrawLightCube(glm::vec3 LightDiffuse);
	void DrawTrees(glm::mat4 ViewMatrix, glm::mat4 ProjectionMatrix);
	void DrawLettersObj(glm::mat4 ViewMatrix, glm::vec4 clipPlane);
	glClearColor(0.529f, 0.808f, 0.922f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//gPerspectiveProjectionMatrix = glm::perspective(glm::radians(gfFOVY), (GLfloat)giWinWidth / (GLfloat)giWinHeight, 0.1f, 5000.0f);
	glm::mat4 ModelMatrix = glm::mat4(1.0f);
	glm::mat4 TranslationMatrix = glm::mat4(1.0f);
	glm::mat4 RotationMatrix = glm::mat4(1.0f);
	glm::mat4 ScaleMatrix = glm::mat4(1.0f);
	glm::mat4 ViewMatrix = glm::mat4(1.0f);
	glm::mat4 ModelViewProjectionMatrix = glm::mat4(1.0f);

	if (gbIsPlaying == GL_FALSE)
	{
		CameraPosition = pCCamera->GetCameraPosition();
		ViewMatrix = pCCamera->GetViewMatrix();
		CameraFront = pCCamera->GetCameraFront();
		CameraUp = pCCamera->GetCameraUp();
		CameraLook = pCCamera->GetCameraFront();
		CameraRight = pCCamera->GetCameraRight();
	}
	else if (gbIsPlaying == GL_TRUE)
	{
		CameraPosition = pCPlayer->GetPlayerPosition();
		ViewMatrix = pCPlayer->GetViewMatrix();
		CameraFront = pCPlayer->GetCameraFront();
		CameraUp = pCPlayer->GetCameraUp();
		CameraLook = pCPlayer->GetCameraLook();
		CameraRight = pCPlayer->GetCameraSide();
	}

	GLfloat AmbientFactor = pCSkyDome->GetAmbientIntensity();
	GLfloat DiffuseFactor = pCSkyDome->GetDiffuseIntensity();

	if (AmbientFactor == 0.062500f && DiffuseFactor == 0.000000f)
	{
		gbIsNight = GL_TRUE;
	}
	else
	{
		gbIsNight = GL_FALSE;
	}

	// calculate the ambient and diffuse light color
	pSunLight->SetAmbient(glm::vec3(pCSkyDome->GetAmbientIntensity()));
	pSunLight->SetDiffuse(glm::vec3(pCSkyDome->GetLightColor() * pCSkyDome->GetDiffuseIntensity()));
	pSunLight->SetSpecular(glm::vec3(pCSkyDome->GetLightColor() * pCSkyDome->GetDiffuseIntensity()));
	pSunLight->SetDirection(glm::vec3(-pCSkyDome->GetSunWPos()));

	// calculate spotlight position and direction
	if (gbIsPlaying == GL_TRUE)
	{
		pTorch->SetPosition(pCPlayer->GetPlayerPosition());
		pTorch->SetDirection(pCPlayer->GetCameraFront());
	}
	else if (gbIsPlaying == GL_FALSE)
	{
		pTorch->SetPosition(pCCamera->GetCameraPosition());
		pTorch->SetDirection(pCCamera->GetCameraFront());
	}

	//Enable clip distance
	glEnable(GL_CLIP_DISTANCE0);

	//Reflection Texture **********************************************************************
	pWater->bindReflectionBuffer();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (gbIsPlaying == GL_FALSE)
	{
		GLfloat distance = -2 * (pCCamera->GetCameraPosition().y);
		glm::vec3 tempPos = glm::vec3(pCCamera->GetCameraPosition().x, pCCamera->GetCameraPosition().y + distance, pCCamera->GetCameraPosition().z);
		pCCamera->SetCameraPosition(tempPos);
		pCCamera->InvertPitch();
	}
	else if (gbIsPlaying == GL_TRUE)
	{
		CPlayer *tempCamera = pCPlayer;
		GLfloat distance = -2 * (tempCamera->GetPlayerPosition().y);

		glm::vec3 tempPos = glm::vec3(tempCamera->GetPlayerPosition().x, tempCamera->GetPlayerPosition().y + distance, tempCamera->GetPlayerPosition().z);
		tempCamera->SetPlayerPosition(tempPos);
		tempCamera->InvertLookAngleX(gfDeltaTime);
	}

	

	/////////////////////////////////////////////
	// Render the skydome

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(CameraPosition.x, CameraPosition.y - pCSkyDome->GetInnerRadius(), CameraPosition.z));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;

	pCSkyDome->RenderSky(CameraPosition, ModelViewProjectionMatrix);


	////////////////////////////////////////////////////////

	///////////////////////////////////////////////////
	// Render the Terrain

	if (gbIsNight == GL_FALSE)
	{
		glUseProgram(pCTerrain->GetShaderObject(TERRAIN_DAY));

		// Set Up the Sun Light for day time
		SetDayLight(pCTerrain->GetShaderObject(TERRAIN_DAY));

		ModelMatrix = glm::mat4(1.0f);
		TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
		ModelMatrix = ModelMatrix * TranslationMatrix;
		pCTerrain->RenderTerrain(ModelMatrix, ViewMatrix, gPerspectiveProjectionMatrix, pWater->getReflectionClipPlane());

		glUseProgram(0);

		//glUseProgram(pCGrass->GetShaderObject(GRASS_DAY));

		//SetDayLight(pCGrass->GetShaderObject(GRASS_DAY));
		/////////////////////////////////////////////
		// render grass
		//pCGrass->RenderGrass(ModelMatrix, ViewMatrix, gPerspectiveProjectionMatrix);

		glUseProgram(0);
	}
	else if (gbIsNight == GL_TRUE)
	{
		ModelMatrix = glm::mat4(1.0f);

		glUseProgram(pCTerrain->GetShaderObject(TERRAIN_NIGHT));
		SetNightLight(pCTerrain->GetShaderObject(TERRAIN_NIGHT));
		pCTerrain->RenderTerrainNight(ModelMatrix, ViewMatrix, gPerspectiveProjectionMatrix, pWater->getReflectionClipPlane());

		glUseProgram(0);

		////////////////////////////////////////////
		// render light cubes

		ModelMatrix = glm::mat4(1.0f);

		glUseProgram(gShaderProgramObjectBasicShader);

		for (int i = 0; i < MAX_POINT_LIGHTS; i++)
		{
			TranslationMatrix = glm::translate(glm::mat4(1.0f), pLamp[i].GetPosition());
			ModelMatrix = ModelMatrix * TranslationMatrix;

			ScaleMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.2f, 0.2f, 0.2f));
			ModelMatrix = ModelMatrix * ScaleMatrix;

			ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;
			glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShader, "u_mvp_matrix"), 1, GL_FALSE, glm::value_ptr(ModelViewProjectionMatrix));

			DrawLightCube(pLamp[i].GetDiffuse());

			ModelMatrix = glm::mat4(1.0f);
			ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;
			glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShader, "u_mvp_matrix"), 1, GL_FALSE, glm::value_ptr(ModelViewProjectionMatrix));
		}
		glUseProgram(0);

		// render the grass

		//glUseProgram(pCGrass->GetShaderObject(GRASS_NIGHT));
		//SetNightLight(pCGrass->GetShaderObject(GRASS_NIGHT));
		//pCGrass->RenderGrassNight(ModelMatrix, ViewMatrix, gPerspectiveProjectionMatrix);

		glUseProgram(0);

	}

	/////////////////////////////////////////////
	// hemisphere light model

	//glUseProgram(gShaderObjectHemisphere);

	////glUniform4fv(glGetUniformLocation(gShaderObjectHemisphere, "u_light_position"), 1, glm::value_ptr(pCSkyDome->GetSunWPos()));
	//glUniform4fv(glGetUniformLocation(gShaderObjectHemisphere, "u_light_position"), 1, glm::value_ptr(glm::vec3(0.0f, 10.0f, -10.0f)));

	////glUniform3fv(glGetUniformLocation(gShaderObjectHemisphere, "skyColor"), 1, glm::value_ptr(pCSkyDome->GetSunColor()));
	//glUniform3fv(glGetUniformLocation(gShaderObjectHemisphere, "skyColor"), 1, glm::value_ptr(glm::vec3(1.0f, 0.0f, 0.0f)));
	//glUniform3fv(glGetUniformLocation(gShaderObjectHemisphere, "groundColor"), 1, glm::value_ptr(glm::vec3(0.0f, 1.0f, 0.0f)));

	//glUniformMatrix4fv(glGetUniformLocation(gShaderObjectHemisphere, "u_projection_matrix"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	//glUniformMatrix4fv(glGetUniformLocation(gShaderObjectHemisphere, "u_view_matrix"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));

	//ModelMatrix = glm::mat4(1.0f);
	//TranslationMatrix = glm::mat4(1.0f);

	//TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -10.0f));
	//ModelMatrix = ModelMatrix * TranslationMatrix;

	//glUniformMatrix4fv(glGetUniformLocation(gShaderObjectHemisphere, "u_model_matrix"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	//

	////pCSkySphere->DrawSphere();
	//

	//glUseProgram(0);

	
	/////////////////////////////////////////////
	//Model Loading

	// draw tree models
	// conifer
	DrawTrees(ViewMatrix, gPerspectiveProjectionMatrix);

	DrawLettersObj(ViewMatrix, glm::vec4(0, 10, 0, 100000));


	FrameBufferUtilities::unbindCurrentBuffer();
	if (gbIsPlaying == GL_FALSE)
	{
		GLfloat distance = -2 * (pCCamera->GetCameraPosition().y);
		glm::vec3 tempPos = glm::vec3(pCCamera->GetCameraPosition().x, pCCamera->GetCameraPosition().y + distance, pCCamera->GetCameraPosition().z);
		pCCamera->SetCameraPosition(tempPos);
		pCCamera->InvertPitch();
	}
	else if (gbIsPlaying == GL_TRUE)
	{
		CPlayer *tempCamera = pCPlayer;
		GLfloat distance = -2 * (tempCamera->GetPlayerPosition().y);

		glm::vec3 tempPos = glm::vec3(tempCamera->GetPlayerPosition().x, tempCamera->GetPlayerPosition().y + distance, tempCamera->GetPlayerPosition().z);
		tempCamera->SetPlayerPosition(tempPos);
		tempCamera->InvertLookAngleX(gfDeltaTime);
	}




	//Refraction 
	pWater->bindRefractionBuffer();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/////////////////////////////////////////////

	

	// Render the skydome

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(CameraPosition.x, CameraPosition.y - pCSkyDome->GetInnerRadius(), CameraPosition.z));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;

	pCSkyDome->RenderSky(CameraPosition, ModelViewProjectionMatrix);


	////////////////////////////////////////////////////////

	///////////////////////////////////////////////////
	// Render the Terrain

	if (gbIsNight == GL_FALSE)
	{
		glUseProgram(pCTerrain->GetShaderObject(TERRAIN_DAY));

		// Set Up the Sun Light for day time
		SetDayLight(pCTerrain->GetShaderObject(TERRAIN_DAY));

		ModelMatrix = glm::mat4(1.0f);
		TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
		ModelMatrix = ModelMatrix * TranslationMatrix;
		pCTerrain->RenderTerrain(ModelMatrix, ViewMatrix, gPerspectiveProjectionMatrix, pWater->getRefractionClipPlane());

		glUseProgram(0);

		//glUseProgram(pCGrass->GetShaderObject(GRASS_DAY));

		//SetDayLight(pCGrass->GetShaderObject(GRASS_DAY));
		/////////////////////////////////////////////
		// render grass
		//pCGrass->RenderGrass(ModelMatrix, ViewMatrix, gPerspectiveProjectionMatrix);

		glUseProgram(0);
	}
	else if (gbIsNight == GL_TRUE)
	{
		ModelMatrix = glm::mat4(1.0f);

		glUseProgram(pCTerrain->GetShaderObject(TERRAIN_NIGHT));
		SetNightLight(pCTerrain->GetShaderObject(TERRAIN_NIGHT));
		pCTerrain->RenderTerrainNight(ModelMatrix, ViewMatrix, gPerspectiveProjectionMatrix, pWater->getRefractionClipPlane());

		glUseProgram(0);

		////////////////////////////////////////////
		// render light cubes

		ModelMatrix = glm::mat4(1.0f);

		glUseProgram(gShaderProgramObjectBasicShader);

		for (int i = 0; i < MAX_POINT_LIGHTS; i++)
		{
			TranslationMatrix = glm::translate(glm::mat4(1.0f), pLamp[i].GetPosition());
			ModelMatrix = ModelMatrix * TranslationMatrix;

			ScaleMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.2f, 0.2f, 0.2f));
			ModelMatrix = ModelMatrix * ScaleMatrix;

			ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;
			glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShader, "u_mvp_matrix"), 1, GL_FALSE, glm::value_ptr(ModelViewProjectionMatrix));

			DrawLightCube(pLamp[i].GetDiffuse());

			ModelMatrix = glm::mat4(1.0f);
			ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;
			glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShader, "u_mvp_matrix"), 1, GL_FALSE, glm::value_ptr(ModelViewProjectionMatrix));
		}
		glUseProgram(0);

		// render the grass

		//glUseProgram(pCGrass->GetShaderObject(GRASS_NIGHT));
		//SetNightLight(pCGrass->GetShaderObject(GRASS_NIGHT));
		//pCGrass->RenderGrassNight(ModelMatrix, ViewMatrix, gPerspectiveProjectionMatrix);

		glUseProgram(0);

	}

	/////////////////////////////////////////////
	// hemisphere light model

	//glUseProgram(gShaderObjectHemisphere);

	////glUniform4fv(glGetUniformLocation(gShaderObjectHemisphere, "u_light_position"), 1, glm::value_ptr(pCSkyDome->GetSunWPos()));
	//glUniform4fv(glGetUniformLocation(gShaderObjectHemisphere, "u_light_position"), 1, glm::value_ptr(glm::vec3(0.0f, 10.0f, -10.0f)));

	////glUniform3fv(glGetUniformLocation(gShaderObjectHemisphere, "skyColor"), 1, glm::value_ptr(pCSkyDome->GetSunColor()));
	//glUniform3fv(glGetUniformLocation(gShaderObjectHemisphere, "skyColor"), 1, glm::value_ptr(glm::vec3(1.0f, 0.0f, 0.0f)));
	//glUniform3fv(glGetUniformLocation(gShaderObjectHemisphere, "groundColor"), 1, glm::value_ptr(glm::vec3(0.0f, 1.0f, 0.0f)));

	//glUniformMatrix4fv(glGetUniformLocation(gShaderObjectHemisphere, "u_projection_matrix"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	//glUniformMatrix4fv(glGetUniformLocation(gShaderObjectHemisphere, "u_view_matrix"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));

	//ModelMatrix = glm::mat4(1.0f);
	//TranslationMatrix = glm::mat4(1.0f);

	//TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -10.0f));
	//ModelMatrix = ModelMatrix * TranslationMatrix;

	//glUniformMatrix4fv(glGetUniformLocation(gShaderObjectHemisphere, "u_model_matrix"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	//

	////pCSkySphere->DrawSphere();
	//

	//glUseProgram(0);


	/////////////////////////////////////////////
	//Model Loading

	// draw tree models
	// conifer
	DrawTrees(ViewMatrix, gPerspectiveProjectionMatrix);
	DrawLettersObj(ViewMatrix, glm::vec4(0, 2, 0, 100000));


	glDisable(GL_CLIP_DISTANCE0);
	FrameBufferUtilities::unbindCurrentBuffer();


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Render scene into floating point framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, gFBOMSAA);
	glClearColor(0.529f, 0.808f, 0.922f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/////////////////////////////////////////////
	// Render the skydome

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(CameraPosition.x, CameraPosition.y - pCSkyDome->GetInnerRadius(), CameraPosition.z));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;

	pCSkyDome->RenderSky(CameraPosition, ModelViewProjectionMatrix);


	////////////////////////////////////////////////////////

	///////////////////////////////////////////////////
	// Render the Terrain

	if (gbIsNight == GL_FALSE)
	{
		glUseProgram(pCTerrain->GetShaderObject(TERRAIN_DAY));

		// Set Up the Sun Light for day time
		SetDayLight(pCTerrain->GetShaderObject(TERRAIN_DAY));

		ModelMatrix = glm::mat4(1.0f);
		TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
		ModelMatrix = ModelMatrix * TranslationMatrix;
		pCTerrain->RenderTerrain(ModelMatrix, ViewMatrix, gPerspectiveProjectionMatrix, glm::vec4(0, -1, 0, 100000));

		glUseProgram(0);

		glUseProgram(pCGrass->GetShaderObject(GRASS_DAY));

		SetDayLight(pCGrass->GetShaderObject(GRASS_DAY));
		/////////////////////////////////////////////
		// render grass
		pCGrass->RenderGrass(ModelMatrix, ViewMatrix, gPerspectiveProjectionMatrix);

		glUseProgram(0);
	}
	else if (gbIsNight == GL_TRUE)
	{
		ModelMatrix = glm::mat4(1.0f);

		glUseProgram(pCTerrain->GetShaderObject(TERRAIN_NIGHT));
		SetNightLight(pCTerrain->GetShaderObject(TERRAIN_NIGHT));
		pCTerrain->RenderTerrainNight(ModelMatrix, ViewMatrix, gPerspectiveProjectionMatrix, glm::vec4(0, -1, 0, 100000));

		glUseProgram(0);

		////////////////////////////////////////////
		// render light cubes

		ModelMatrix = glm::mat4(1.0f);

		glUseProgram(gShaderProgramObjectBasicShader);

		for (int i = 0; i < MAX_POINT_LIGHTS; i++)
		{
			TranslationMatrix = glm::translate(glm::mat4(1.0f), pLamp[i].GetPosition());
			ModelMatrix = ModelMatrix * TranslationMatrix;

			ScaleMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.2f, 0.2f, 0.2f));
			ModelMatrix = ModelMatrix * ScaleMatrix;

			ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;
			glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShader, "u_mvp_matrix"), 1, GL_FALSE, glm::value_ptr(ModelViewProjectionMatrix));

			DrawLightCube(pLamp[i].GetDiffuse());

			ModelMatrix = glm::mat4(1.0f);
			ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;
			glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShader, "u_mvp_matrix"), 1, GL_FALSE, glm::value_ptr(ModelViewProjectionMatrix));
		}
		glUseProgram(0);

		// render the grass

		glUseProgram(pCGrass->GetShaderObject(GRASS_NIGHT));
		SetNightLight(pCGrass->GetShaderObject(GRASS_NIGHT));
		pCGrass->RenderGrassNight(ModelMatrix, ViewMatrix, gPerspectiveProjectionMatrix);

		glUseProgram(0);

	}

	


	/////////////////////////////////////////////
	//Model Loading

	// draw tree models
	// conifer
	DrawTrees(ViewMatrix, gPerspectiveProjectionMatrix);
	DrawLettersObj(ViewMatrix, glm::vec4(0, 2, 0, 100000));

	
	// water render
	ModelMatrix = glm::mat4(1.0f);
	ModelMatrix = ModelMatrix * glm::translate(glm::mat4(1.0f), glm::vec3(1, gfWaterHeight, 1));
	pWater->RenderWater(gShaderProgramObjectWater, gVAOWater, gVBOWaterIndices, ModelMatrix, ViewMatrix, gPerspectiveProjectionMatrix, gfDeltaTime);
	//font
	/////////////////////////////////////
	//render fonts
	glDisable(GL_DEPTH_TEST);
	glUseProgram(gShaderProgramObjectFont);
	ModelMatrix = glm::mat4(1.0f);
	ViewMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -6.0f));

	ModelMatrix = ModelMatrix * TranslationMatrix;

	ModelViewProjectionMatrix = gPerspectiveProjectionMatrixFont * ViewMatrix * ModelMatrix;

	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectFont, "u_mvp_matrix"), 1, GL_FALSE, glm::value_ptr(ModelViewProjectionMatrix));
	//check_gl_error();
	glActiveTexture(GL_TEXTURE0);
	glUniform1i(glGetUniformLocation(gShaderProgramObjectFont, "texture0_sampler_u"), 0);
	//check_gl_error();
	//check_gl_error();
	//For Font render
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glBindVertexArray(gVAOFonts);
	fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, "(C)AstroMediComp", 1.0f, 1.0f, 1.0f, -5.9f, 3.0f, 0.009f);
	fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, "Raster Group 2017-2018", 1.0f, 1.0f, 1.0f, -5.9f, 2.7f, 0.009f);
	//std::stringstream ss;
	////ss << "FPS: " << fps__;
	//ss << "Height At : " << pCTerrain->GetHeightAt(pCPlayer->GetPlayerPosition());
	//fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, ss.str().c_str(), 1.0f, 1.0f, 1.0f, -5.9f, 2.4f, 0.009f);


	//help
	if (gisDisplayHelp == 1) {
		fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, "Use Mouse Move to turn Left, Right, look Up and Down.", 1.0f, 1.0f, 1.0f, -1.0f, 3.0f, 0.009f);
		fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, "Use Up, Down, Left, Right Arrow Keys to navigate through the scene.", 1.0f, 1.0f, 1.0f, -1.0f, 2.7f, 0.009f);
		fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, "Press Spacebar to Jump.", 1.0f, 1.0f, 1.0f, -1.0f, 2.4f, 0.009f);
		fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, "Press Control to crouch.", 1.0f, 1.0f, 1.0f, -1.0f, 2.1f, 0.009f);
		fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, "Press Shift to sprint.", 1.0f, 1.0f, 1.0f, -1.0f, 1.8f, 0.009f);
		fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, "Click on the object to select it.", 1.0f, 1.0f, 1.0f, -1.0f, 1.5f, 0.009f);
		fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, "Press L Toggle Torch in night.", 1.0f, 1.0f, 1.0f, -1.0f, 1.2f, 0.009f);
		fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, "Press F1 to hide/show help.", 1.0f, 1.0f, 1.0f, -1.0f, 0.9f, 0.009f);
	}
	else {
		fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, "Press F1 to hide/show help.", 1.0f, 1.0f, 1.0f, 3.0f, 3.0f, 0.009f);
	}

	if (gbIsPlaying == GL_FALSE)
	{
		fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, "Press P to Play.", 1.0f, 1.0f, 1.0f, -1.5f, 2.0f, 0.018f);
	} 

	if (gisDisplayNames == 1) 
	{
		/*std::stringstream sss;
		sss << "Alpha: " << gAlphaFont;
		fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, sss.str().c_str(), 1.0f, 1.0f, 1.0f, -5.9f, 0.3f, 0.009f);
		*/
		if (gFontTransition >= 0.0 && gFontTransition <= 1.5) {
			std::stringstream sPranav;
			sPranav << "Pranav Gore";
			fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, sPranav.str().c_str(), gAlphaFont, gAlphaFont, gAlphaFont, -5.0, 0.6f, 0.016f);

			std::stringstream ssugat1;
			ssugat1 << "Sugat Mankar";
			fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, ssugat1.str().c_str(), gAlphaFontN, gAlphaFontN, gAlphaFontN, -5.0f, 0.1f, 0.016f);

		}
		if (gFontTransition >= 1.5 && gFontTransition <= 3.0) {
			std::stringstream sAtul;
			sAtul << "Atul Linganwar";
			fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, sAtul.str().c_str(), gAlphaFont, gAlphaFont, gAlphaFont, -5.0, 0.6f, 0.016f);

			std::stringstream sPrabhakar;
			sPrabhakar << "Prabhakar Patil";
			fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, sPrabhakar.str().c_str(), gAlphaFontN, gAlphaFontN, gAlphaFontN, -5.0f, 0.1f, 0.016f);
		}
		if (gFontTransition >= 3.0 && gFontTransition <= 4.5) {
			std::stringstream sSujay;
			sSujay << "Sujay Ingale";
			fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, sSujay.str().c_str(), gAlphaFont, gAlphaFont, gAlphaFont, -5.0, 0.6f, 0.016f);

			std::stringstream sAshish;
			sAshish << "Ashish Khare";
			fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, sAshish.str().c_str(), gAlphaFontN, gAlphaFontN, gAlphaFontN, -5.0f, 0.1f, 0.016f);
		}
		if (gFontTransition >= 4.5 && gFontTransition <= 6.0) {
			std::stringstream sAshishD;
			sAshishD << "Aashish Deshpande";
			fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, sAshishD.str().c_str(), gAlphaFont, gAlphaFont, gAlphaFont, -5.0, 0.6f, 0.016f);

			std::stringstream sPratik;
			sPratik << "Pratik Dhanave";
			fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, sPratik.str().c_str(), gAlphaFontN, gAlphaFontN, gAlphaFontN, -5.0f, 0.1f, 0.016f);
		}
		if (gFontTransition >= 6.0 && gFontTransition <= 7.5) {
			std::stringstream sGaurav;
			sGaurav << "Gaurav Patil";
			fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, sGaurav.str().c_str(), gAlphaFont, gAlphaFont, gAlphaFont, -5.0, 0.6f, 0.016f);

			std::stringstream sAbhijeet;
			sAbhijeet << "Abhijeet Kanade";
			fonts.RenderText(gShaderProgramObjectFont, gVBOFonts, sAbhijeet.str().c_str(), gAlphaFontN, gAlphaFontN, gAlphaFontN, -5.0f, 0.1f, 0.016f);

			gisDisplayNames = 0;
			gbNameShown = GL_TRUE;
		}


		glBindTexture(GL_TEXTURE_2D, 0);

		//glBindVertexArray(0);
		glDisable(GL_BLEND);
		glUseProgram(0);
		glEnable(GL_DEPTH_TEST);
		////////////////////////////////////////////////
	}

	

	// render samplequad
	/*glUseProgram(gShaderObjectSampleTexture);
	pCCommon->_check_gl_error(pCCommon->pLogFile, pCCommon->logfilepath, __FILE__, __LINE__);
	glActiveTexture(GL_TEXTURE0);
	pCCommon->_check_gl_error(pCCommon->pLogFile, pCCommon->logfilepath, __FILE__, __LINE__);
	glBindTexture(GL_TEXTURE_2D, DiffuseTexture);
	pCCommon->_check_gl_error(pCCommon->pLogFile, pCCommon->logfilepath, __FILE__, __LINE__);
	glUniform1i(glGetUniformLocation(gShaderObjectSampleTexture, "Diffuse"), 0);
	pCCommon->_check_gl_error(pCCommon->pLogFile, pCCommon->logfilepath, __FILE__, __LINE__);

	glActiveTexture(GL_TEXTURE1);
	pCCommon->_check_gl_error(pCCommon->pLogFile, pCCommon->logfilepath, __FILE__, __LINE__);
	glBindTexture(GL_TEXTURE_2D, OpacityTexture);
	pCCommon->_check_gl_error(pCCommon->pLogFile, pCCommon->logfilepath, __FILE__, __LINE__);
	glUniform1i(glGetUniformLocation(gShaderObjectSampleTexture, "Opacity"), 1);
	pCCommon->_check_gl_error(pCCommon->pLogFile, pCCommon->logfilepath, __FILE__, __LINE__);

	ModelViewProjectionMatrix = glm::mat4(1.0f);
	ModelMatrix = glm::mat4(1.0f);
	ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gShaderObjectSampleTexture, "u_mvp_matrix"), 1, GL_FALSE, glm::value_ptr(ModelViewProjectionMatrix));
	pCCommon->_check_gl_error(pCCommon->pLogFile, pCCommon->logfilepath, __FILE__, __LINE__);

	RenderSampleQuad();
	glUseProgram(0);
	pCCommon->_check_gl_error(pCCommon->pLogFile, pCCommon->logfilepath, __FILE__, __LINE__);

	glEnable(GL_CULL_FACE);*/
	/////////////////////////////////////////////
	
	//glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// blit the multisample buffer to hdr
	glBindFramebuffer(GL_READ_FRAMEBUFFER, gFBOMSAA);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gFrameBufferObjectHDR);
	glBlitFramebuffer(0, 0, giWinWidth, giWinHeight, 0, 0, giWinWidth, giWinHeight, GL_COLOR_BUFFER_BIT, GL_NEAREST);
	glBlitFramebuffer(0, 0, giWinWidth, giWinHeight, 0, 0, giWinWidth, giWinHeight, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
	glBlitFramebuffer(0, 0, giWinWidth, giWinHeight, 0, 0, giWinWidth, giWinHeight, GL_STENCIL_BUFFER_BIT, GL_NEAREST);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	//////////////////////////////////////////////
	// Now render the floating point color buffer to 2D quad and tonemap HDR colors to default framebuffer's (clamped) color range

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(gShaderObjectHDR);
	glActiveTexture(GL_TEXTURE0);

	glBindTexture(GL_TEXTURE_2D, gColorBufferTexture);
	glUniform1i(glGetUniformLocation(gShaderObjectHDR, "HDRTexture"), 0);

	glUniform1f(glGetUniformLocation(gShaderObjectHDR, "exposure"), fExposure);
	RenderQuadHDR();
	glUseProgram(0);


	//////////////////////////////////////
	// render sun

	// copy the depth buffer to the texture
	//glBindTexture(GL_TEXTURE_2D, DepthTexture);
	//glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, giWinWidth, giWinHeight);
	//glBindTexture(GL_TEXTURE_2D, 0);

	//bool CalculateSunRays = false;
	//int Test = 0;
	//int Tests = 16;
	//float Angle = 0.0f;
	//float AngleInc = 360.0f / Tests;

	////glm::mat4 BiasMatrix = glm::mat4(glm::vec4(0.5f, 0.0f, 0.0f, 0.0f), glm::vec4(0.0f, 0.5f, 0.0f, 0.0f), glm::vec4(0.0f, 0.0f, 0.5f, 0.0f), glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));
	//glm::mat4x4 BiasMatrix = glm::mat4x4(0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.5f, 0.5f, 0.5f, 1.0f);
	//glm::mat4 ViewProjectionBiasMatrix = BiasMatrix * gPerspectiveProjectionMatrix * ViewMatrix;

	//glm::vec3 SunWPos = pCSkyDome->GetSunWPos();
	//GLfloat SunR = pCSkyDome->GetSunRadius();

	//while (Test < Tests && !CalculateSunRays)
	//{
	//	glm::vec3 SunWEdge = SunWPos + glm::rotate(CameraRight, Angle, CameraFront) * SunR;

	//	if (SunWEdge.y - CameraPosition.y > 0.0f)
	//	{
	//		glm::vec4 SunPosProj = ViewProjectionBiasMatrix * glm::vec4(SunWEdge, 1.0f);
	//		SunPosProj /= SunPosProj.w;

	//		CalculateSunRays |= (SunPosProj.x >= 0.0f && SunPosProj.x <= 1.0f && SunPosProj.y >= 0.0f && SunPosProj.y <= 1.0f && SunPosProj.z >= 0.0f && SunPosProj.z <= 1.0f);
	//	}

	//	Angle += AngleInc;
	//	Test++;
	//}

	//

	//// calculate sunrays
	//if (CalculateSunRays)
	//{
	//	glm::vec4 SunPosProj = ViewProjectionBiasMatrix * glm::vec4(SunWPos, 1.0f);
	//	SunPosProj /= SunPosProj.w;

	//	glViewport(0, 0, giSunTextureWidth, giSunTextureHeight);

	//	// render sun sphere
	//	glBindFramebuffer(GL_FRAMEBUFFER, gFBOSun);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, SunTextures[1], 0);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

	//	glClear(GL_COLOR_BUFFER_BIT);

	//	glUseProgram(gShaderProgramObjectBasicShader);
	//	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(SunWPos.x, SunWPos.y, SunWPos.z));
	//	ModelMatrix = ModelMatrix * TranslationMatrix;

	//	ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;
	//	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShader, "u_mvp_matrix"), 1, GL_FALSE, glm::value_ptr(ModelViewProjectionMatrix));

	//	glVertexAttrib3fv(OGL_ATTRIBUTE_COLOR, glm::value_ptr(pCSkyDome->GetSunColor()));
	//	pSunSphere->DrawSphere();
	//	glUseProgram(0);

	//	// test of sun sphere is behind the scene geometry
	//	glBindFramebuffer(GL_FRAMEBUFFER, gFBOSun);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, SunTextures[0], 0);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

	//	glUseProgram(gShaderProgramObjectSunDepthTest);

	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, SunTextures[1]);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectSunDepthTest, "SunTexture"), 0);

	//	glActiveTexture(GL_TEXTURE1);
	//	glBindTexture(GL_TEXTURE_2D, DepthTexture);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectSunDepthTest, "DepthTexture"), 1);

	//	glBindVertexArray(gVAOScreenFilledQuad);
	//	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	//	glBindVertexArray(0);
	//	glUseProgram(0);

	//	glActiveTexture(GL_TEXTURE1);
	//	glBindTexture(GL_TEXTURE_2D, 0);

	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, 0);

	//	// blur sun sphere horizontally (low)
	//	glBindFramebuffer(GL_FRAMEBUFFER, gFBOSun);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, SunTextures[3], 0);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

	//	glUseProgram(gShaderProgramObjectBlurHorizontal);
	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, SunTextures[0]);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectBlurHorizontal, "Texture"), 0);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectBlurHorizontal, "Width"), 1);
	//	glBindVertexArray(gVAOScreenFilledQuad);
	//	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	//	glBindVertexArray(0);
	//	glUseProgram(0);

	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, 0);

	//	// blur sun sphere vertically (low)
	//	glBindFramebuffer(GL_FRAMEBUFFER, gFBOSun);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, SunTextures[1], 0);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

	//	glUseProgram(gShaderProgramObjectBlurVertical);
	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, SunTextures[3]);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectBlurVertical, "Texture"), 0);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectBlurVertical, "Width"), 1);
	//	glBindVertexArray(gVAOScreenFilledQuad);
	//	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	//	glBindVertexArray(0);
	//	glUseProgram(0);

	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, 0);

	//	// blur sun sphere horizontally (high)
	//	glBindFramebuffer(GL_FRAMEBUFFER, gFBOSun);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, SunTextures[3], 0);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

	//	glUseProgram(gShaderProgramObjectBlurHorizontal);
	//	
	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, SunTextures[0]);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectBlurHorizontal, "Texture"), 0);
	//	
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectBlurHorizontal, "Width"), 10);
	//	glBindVertexArray(gVAOScreenFilledQuad);
	//	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	//	glBindVertexArray(0);
	//	glUseProgram(0);

	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, 0);

	//	// blur sun sphere vertically (high)
	//	glBindFramebuffer(GL_FRAMEBUFFER, gFBOSun);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, SunTextures[2], 0);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

	//	glUseProgram(gShaderProgramObjectBlurVertical);
	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, SunTextures[3]);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectBlurVertical, "Texture"), 0);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectBlurVertical, "Width"), 10);
	//	glBindVertexArray(gVAOScreenFilledQuad);
	//	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	//	glBindVertexArray(0);
	//	glUseProgram(0);

	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, 0);

	//	// blur sun sphere radially and calculate sunrays
	//	glBindFramebuffer(GL_FRAMEBUFFER, gFBOSun);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, SunTextures[3], 0);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

	//	glUseProgram(gShaderProgramObjectSunRays);
	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, SunTextures[1]);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectSunDepthTest, "LowBlurredSunTexture"), 0);

	//	glActiveTexture(GL_TEXTURE1);
	//	glBindTexture(GL_TEXTURE_2D, SunTextures[2]);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectSunDepthTest, "HighBlurredSunTexture"), 1);

	//	glUniform2fv(glGetUniformLocation(gShaderProgramObjectSunDepthTest, "SunPosProj"), 1, glm::value_ptr(glm::vec2(SunPosProj)));

	//	glBindVertexArray(gVAOScreenFilledQuad);
	//	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	//	glBindVertexArray(0);

	//	glUseProgram(0);

	//	glActiveTexture(GL_TEXTURE1);
	//	glBindTexture(GL_TEXTURE_2D, 0);

	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, 0);

	//	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//	glViewport(0, 0, giWinWidth, giWinHeight);
	//}

	//gOrthographicProjectionMatrix = glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f);
	//ModelMatrix = glm::mat4(1.0f);

	//if (CalculateSunRays)
	//{
	//	glUseProgram(gShaderProgramObjectBasicShaderTexture);
	//	ModelViewProjectionMatrix = gOrthographicProjectionMatrix * ViewMatrix * ModelMatrix;
	//	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShaderTexture, "u_mvp_matrix"), 1, GL_FALSE, glm::value_ptr(ModelViewProjectionMatrix));

	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, SunTextures[3]);
	//	glUniform1i(glGetUniformLocation(gShaderProgramObjectBasicShaderTexture, "SunTexture"), 0);

	//	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);
	//	glEnable(GL_BLEND);

	//	glBindVertexArray(gVAOScreenFilledQuad);
	//	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	//	glBindVertexArray(0);

	//	glDisable(GL_BLEND);
	//	glUseProgram(0);
	//	glBindTexture(GL_TEXTURE_2D, 0);
	//}
}

void Update(void)
{

/*
	gfAnglePyramid += 0.001f;
	if (gfAnglePyramid >= 360.0f)
		gfAnglePyramid = 0.0f;

	gfAngleCube += 0.1f;
	if (gfAngleCube >= 360.0f)
		gfAngleCube = 0.0f;*/

	gfLetterRotationAngle = (gfLetterRotationAngle+1.0f);
	if (gfLetterRotationAngle >= 360.0f)
	{
		gfLetterRotationAngle = 0.0f;
	}
	
	if (gfLetterRotationAngle <= 180.0f)
	{
		LetterA_Translation.y += gbLetterA_SetFlag ? 0.0 : LetterBounceFactor;
		LetterS_Translation.y += gbLetterS_SetFlag ? 0.0 : LetterBounceFactor;
		LetterT_Translation.y += gbLetterT_SetFlag ? 0.0 : LetterBounceFactor;
		LetterR_Translation.y += gbLetterR_SetFlag ? 0.0 : LetterBounceFactor;
		LetterO_Translation.y += gbLetterO_SetFlag ? 0.0 : LetterBounceFactor;
		LetterM_Translation.y += gbLetterM_SetFlag ? 0.0 : LetterBounceFactor;
		LetterE_Translation.y += gbLetterE_SetFlag ? 0.0 : LetterBounceFactor;
		LetterD_Translation.y += gbLetterD_SetFlag ? 0.0 : LetterBounceFactor;
		LetterI_Translation.y += gbLetterI_SetFlag ? 0.0 : LetterBounceFactor;
		LetterC_Translation.y += gbLetterC_SetFlag ? 0.0 : LetterBounceFactor;
		LetterO_2_Translation.y += gbLetterO_2_SetFlag ? 0.0 : LetterBounceFactor;
		LetterM_2_Translation.y += gbLetterM_2_SetFlag ? 0.0 : LetterBounceFactor;
		LetterP_Translation.y += gbLetterP_SetFlag ? 0.0 : LetterBounceFactor;
	}
	else
	{
		LetterA_Translation.y -= gbLetterA_SetFlag ? 0.0 : LetterBounceFactor;
		LetterS_Translation.y -= gbLetterS_SetFlag ? 0.0 : LetterBounceFactor;
		LetterT_Translation.y -= gbLetterT_SetFlag ? 0.0 : LetterBounceFactor;
		LetterR_Translation.y -= gbLetterR_SetFlag ? 0.0 : LetterBounceFactor;
		LetterO_Translation.y -= gbLetterO_SetFlag ? 0.0 : LetterBounceFactor;
		LetterM_Translation.y -= gbLetterM_SetFlag ? 0.0 : LetterBounceFactor;
		LetterE_Translation.y -= gbLetterE_SetFlag ? 0.0 : LetterBounceFactor;
		LetterD_Translation.y -= gbLetterD_SetFlag ? 0.0 : LetterBounceFactor;
		LetterI_Translation.y -= gbLetterI_SetFlag ? 0.0 : LetterBounceFactor;
		LetterC_Translation.y -= gbLetterC_SetFlag ? 0.0 : LetterBounceFactor;
		LetterO_2_Translation.y -= gbLetterO_2_SetFlag ? 0.0 : LetterBounceFactor;
		LetterM_2_Translation.y -= gbLetterM_2_SetFlag ? 0.0 : LetterBounceFactor;
		LetterP_Translation.y -= gbLetterP_SetFlag ? 0.0 : LetterBounceFactor;
	}
	// Raster Rotation
	if (gbAllLettersReachedFinalPosFlag == TRUE)
	{
		gfRasterRotateAngle = gfRasterRotateAngle + 0.2f;
		if (gfRasterRotateAngle >= 90.0f)
		{
			gfRasterRotateAngle = 90.0f;
			gbLetterGameOver = TRUE;		//Atul use this flag as event of game over
		}
	}
	// Simple
	//if (gbLetterD_SetFlag == TRUE)  //x = 0 to 135, z = 5 to 445
	//{
	//	if (LetterD_Translation.x < LetterD_Final_Translation.x)
	//		LetterD_Translation.x += 0.5f;

	//	
	//	if (LetterD_Translation.z < LetterD_Final_Translation.z)
	//		LetterD_Translation.z += 0.5f;

	//}

	static int interpolationPoints = 1000;

		//A Linear translation
	if (gbLetterA_SetFlag == TRUE || gbUniversalLetterMoveFlag == TRUE)
	{// calculating interpolation along line using line equation, y = mx + c, x is varied by ourself with xd, y will be calculated accordingly
		static float slope = (LetterA_Final_Translation.z - LetterA_Translation.z) / (LetterA_Final_Translation.x - LetterA_Translation.x);
		static float xd = fabs(LetterA_Final_Translation.x - LetterA_Translation.x) / interpolationPoints;
		static bool isNegativeSlope = ((LetterA_Final_Translation.z - LetterA_Translation.z) < 0.0f) || ((LetterA_Final_Translation.x - LetterA_Translation.x) < 0.0);
		if (isNegativeSlope == false)
		{
			if (LetterA_Translation.x <= LetterA_Final_Translation.x)
				LetterA_Translation.x += xd;
			else
				gbLetterA_ReachedFinalPosFlag = TRUE;
		}
		else
		{
			if (LetterA_Translation.x >= LetterA_Final_Translation.x)
				LetterA_Translation.x -= xd;
			else
				gbLetterA_ReachedFinalPosFlag = TRUE;
		}
		static float z = LetterA_Translation.z - (slope * LetterA_Translation.x);
		//y=mx+c
		//if (LetterA_Translation.z >= LetterA_Final_Translation.z)
		LetterA_Translation.z = slope * LetterA_Translation.x + z;
	}

	//S Linear translation
	if (gbLetterS_SetFlag == TRUE || gbUniversalLetterMoveFlag == TRUE)
	{// calculating interpolation along line using line equation, y = mx + c, x is varied by ourself with xd, y will be calculated accordingly
		static float slope = (LetterS_Final_Translation.z - LetterS_Translation.z) / (LetterS_Final_Translation.x - LetterS_Translation.x);
		static float xd = fabs(LetterS_Final_Translation.x - LetterS_Translation.x) / interpolationPoints;
		static bool isNegativeSlope = ((LetterS_Final_Translation.z - LetterS_Translation.z) < 0.0f) || ((LetterS_Final_Translation.x - LetterS_Translation.x) < 0.0);
		if (isNegativeSlope == false)
		{
			if (LetterS_Translation.x <= LetterS_Final_Translation.x)
				LetterS_Translation.x += xd;
			else
				gbLetterS_ReachedFinalPosFlag = TRUE;
		}
		else
		{
			if (LetterS_Translation.x >= LetterS_Final_Translation.x)
				LetterS_Translation.x -= xd;
			else
				gbLetterS_ReachedFinalPosFlag = TRUE;
		}
		static float z = LetterS_Translation.z - (slope * LetterS_Translation.x);
		//y=mx+c
		//if (LetterS_Translation.z >= LetterS_Final_Translation.z)
		LetterS_Translation.z = slope * LetterS_Translation.x + z;
	}

	//T Linear
	if (gbLetterT_SetFlag == TRUE || gbUniversalLetterMoveFlag == TRUE)
	{// calculating interpolation along line using line equation, y = mx + c, x is varied by ourself with xd, y will be calculated accordingly
		static float slope = (LetterT_Final_Translation.z - LetterT_Translation.z) / (LetterT_Final_Translation.x - LetterT_Translation.x);
		static float xd = fabs(LetterT_Final_Translation.x - LetterT_Translation.x) / interpolationPoints;
		static bool isNegativeSlope = ((LetterT_Final_Translation.z - LetterT_Translation.z) < 0.0f) || ((LetterT_Final_Translation.x - LetterT_Translation.x) < 0.0);
		if (isNegativeSlope == false)
		{
			if (LetterT_Translation.x <= LetterT_Final_Translation.x)
				LetterT_Translation.x += xd;
			else
				gbLetterT_ReachedFinalPosFlag = TRUE;
		}
		else
		{
			if (LetterT_Translation.x >= LetterT_Final_Translation.x)
				LetterT_Translation.x -= xd;
			else
				gbLetterT_ReachedFinalPosFlag = TRUE;
		}
		static float z = LetterT_Translation.z - (slope * LetterT_Translation.x);
		//y=mx+c
		//if (LetterT_Translation.z >= LetterT_Final_Translation.z)
		LetterT_Translation.z = slope * LetterT_Translation.x + z;
	}

	//R Linear
	if (gbLetterR_SetFlag == TRUE || gbUniversalLetterMoveFlag == TRUE)
	{// calculating interpolation along line using line equation, y = mx + c, x is varied by ourself with xd, y will be calculated accordingly
		static float slope = (LetterR_Final_Translation.z - LetterR_Translation.z) / (LetterR_Final_Translation.x - LetterR_Translation.x);
		static float xd = fabs(LetterR_Final_Translation.x - LetterR_Translation.x) / interpolationPoints;
		static bool isNegativeSlope = ((LetterR_Final_Translation.z - LetterR_Translation.z) < 0.0f) || ((LetterR_Final_Translation.x - LetterR_Translation.x) < 0.0);
		if (isNegativeSlope == false)
		{
			if (LetterR_Translation.x <= LetterR_Final_Translation.x)
				LetterR_Translation.x += xd;
			else
				gbLetterR_ReachedFinalPosFlag = TRUE;
		}
		else
		{
			if (LetterR_Translation.x >= LetterR_Final_Translation.x)
				LetterR_Translation.x -= xd;
			else
				gbLetterR_ReachedFinalPosFlag = TRUE;
		}
		static float z = LetterR_Translation.z - (slope * LetterR_Translation.x);
		//y=mx+c
		//if (LetterR_Translation.z >= LetterR_Final_Translation.z)
		LetterR_Translation.z = slope * LetterR_Translation.x + z;
	}

	//O Linear
	if (gbLetterO_SetFlag == TRUE || gbUniversalLetterMoveFlag == TRUE)
	{// calculating interpolation along line using line equation, y = mx + c, x is varied by ourself with xd, y will be calculated accordingly
		static float slope = (LetterO_Final_Translation.z - LetterO_Translation.z) / (LetterO_Final_Translation.x - LetterO_Translation.x);
		static float xd = fabs(LetterO_Final_Translation.x - LetterO_Translation.x) / interpolationPoints;
		static bool isNegativeSlope = ((LetterO_Final_Translation.z - LetterO_Translation.z) < 0.0f) || ((LetterO_Final_Translation.x - LetterO_Translation.x) < 0.0);
		if (isNegativeSlope == false)
		{
			if (LetterO_Translation.x <= LetterO_Final_Translation.x)
				LetterO_Translation.x += xd;
			else
				gbLetterO_ReachedFinalPosFlag = TRUE;
		}
		else
		{
			if (LetterO_Translation.x >= LetterO_Final_Translation.x)
				LetterO_Translation.x -= xd;
			else
				gbLetterO_ReachedFinalPosFlag = TRUE;
		}
		static float z = LetterO_Translation.z - (slope * LetterO_Translation.x);
		//y=mx+c
		//if (LetterO_Translation.z >= LetterO_Final_Translation.z)
		LetterO_Translation.z = slope * LetterO_Translation.x + z;
	}

	//M Linear, //y=mx+c
	if (gbLetterM_SetFlag == TRUE || gbUniversalLetterMoveFlag == TRUE)
	{// calculating interpolation along line using line equation, y = mx + c, x is varied by ourself with xd, y will be calculated accordingly
		static float slope = (LetterM_Final_Translation.z - LetterM_Translation.z) / (LetterM_Final_Translation.x - LetterM_Translation.x);
		static float xd = fabs(LetterM_Final_Translation.x - LetterM_Translation.x) / interpolationPoints;
		static bool isNegativeSlope = ((LetterM_Final_Translation.z - LetterM_Translation.z) < 0.0f) || ((LetterM_Final_Translation.x - LetterM_Translation.x) < 0.0);


		if (isNegativeSlope == false)
		{
			if (LetterM_Translation.x <= LetterM_Final_Translation.x)
				LetterM_Translation.x += xd;
			else
				gbLetterM_ReachedFinalPosFlag = TRUE;
		}
		else
		{
			if (LetterM_Translation.x >= LetterM_Final_Translation.x)
				LetterM_Translation.x -= xd;
			else
				gbLetterM_ReachedFinalPosFlag = TRUE;
		}

		static float z = LetterM_Translation.z - (slope * LetterM_Translation.x);
		//y=mx+c
		//if (LetterM_Translation.z >= LetterM_Final_Translation.z)
		LetterM_Translation.z = slope * LetterM_Translation.x + z;
	}

	//E Linear
	if (gbLetterE_SetFlag == TRUE || gbUniversalLetterMoveFlag == TRUE)
	{// calculating interpolation along line using line equation, y = mx + c, x is varied by ourself with xd, y will be calculated accordingly
		static float slope = (LetterE_Final_Translation.z - LetterE_Translation.z) / (LetterE_Final_Translation.x - LetterE_Translation.x);
		static float xd = fabs(LetterE_Final_Translation.x - LetterE_Translation.x) / interpolationPoints;
		static bool isNegativeSlope = ((LetterE_Final_Translation.z - LetterE_Translation.z) < 0.0f) || ((LetterE_Final_Translation.x - LetterE_Translation.x) < 0.0);
		if (isNegativeSlope == false)
		{
			if (LetterE_Translation.x <= LetterE_Final_Translation.x)
				LetterE_Translation.x += xd;
			else
				gbLetterE_ReachedFinalPosFlag = TRUE;
		}
		else
		{
			if (LetterE_Translation.x >= LetterE_Final_Translation.x)
				LetterE_Translation.x -= xd;
			else
				gbLetterE_ReachedFinalPosFlag = TRUE;
		}
		static float z = LetterE_Translation.z - (slope * LetterE_Translation.x);
		//y=mx+c
		//if (LetterE_Translation.z >= LetterE_Final_Translation.z)
		LetterE_Translation.z = slope * LetterE_Translation.x + z;
	}

	//D Linear translation
	if (gbLetterD_SetFlag == TRUE || gbUniversalLetterMoveFlag == TRUE)
	{// calculating interpolation along line using line equation, y = mx + c, x is varied by ourself with xd, y will be calculated accordingly
		static float slope = (LetterD_Final_Translation.z - LetterD_Translation.z) / (LetterD_Final_Translation.x - LetterD_Translation.x);
		static float xd = fabs(LetterD_Final_Translation.x - LetterD_Translation.x) / interpolationPoints;
		static bool isNegativeSlope = ((LetterD_Final_Translation.z - LetterD_Translation.z) < 0.0f) || ((LetterD_Final_Translation.x - LetterD_Translation.x) < 0.0);
		if (isNegativeSlope == false)
		{
			if (LetterD_Translation.x <= LetterD_Final_Translation.x)
				LetterD_Translation.x += xd;
			else
				gbLetterD_ReachedFinalPosFlag = TRUE;
		}
		else
		{
			if (LetterD_Translation.x >= LetterD_Final_Translation.x)
				LetterD_Translation.x -= xd;
			else
				gbLetterD_ReachedFinalPosFlag = TRUE;
		}
		static float z = LetterD_Translation.z - (slope * LetterD_Translation.x);
		//y=mx+c
		//if (LetterD_Translation.z >= LetterD_Final_Translation.z)
		LetterD_Translation.z = slope * LetterD_Translation.x + z;
	}

	//I Linear translation
	if (gbLetterI_SetFlag == TRUE || gbUniversalLetterMoveFlag == TRUE)
	{// calculating interpolation along line using line equation, y = mx + c, x is varied by ourself with xd, y will be calculated accordingly
		static float slope = (LetterI_Final_Translation.z - LetterI_Translation.z) / (LetterI_Final_Translation.x - LetterI_Translation.x);
		static float xd = fabs(LetterI_Final_Translation.x - LetterI_Translation.x) / interpolationPoints;
		static bool isNegativeSlope = ((LetterI_Final_Translation.z - LetterI_Translation.z) < 0.0f) || ((LetterI_Final_Translation.x - LetterI_Translation.x) < 0.0);
		if (isNegativeSlope == false)
		{
			if (LetterI_Translation.x <= LetterI_Final_Translation.x)
				LetterI_Translation.x += xd;
			else
				gbLetterI_ReachedFinalPosFlag = TRUE;
		}
		else
		{
			if (LetterI_Translation.x >= LetterI_Final_Translation.x)
				LetterI_Translation.x -= xd;
			else
				gbLetterI_ReachedFinalPosFlag = TRUE;
		}
		static float z = LetterI_Translation.z - (slope * LetterI_Translation.x);
		//y=mx+c
		//if (LetterI_Translation.z >= LetterI_Final_Translation.z)
		LetterI_Translation.z = slope * LetterI_Translation.x + z;
	}

	//C Linear translation
	if (gbLetterC_SetFlag == TRUE || gbUniversalLetterMoveFlag == TRUE)
	{// calculating interpolation along line using line equation, y = mx + c, x is varied by ourself with xd, y will be calculated accordingly
		static float slope = (LetterC_Final_Translation.z - LetterC_Translation.z) / (LetterC_Final_Translation.x - LetterC_Translation.x);
		static float xd = fabs(LetterC_Final_Translation.x - LetterC_Translation.x) / interpolationPoints;
		static bool isNegativeSlope = ((LetterC_Final_Translation.z - LetterC_Translation.z) < 0.0f) || ((LetterC_Final_Translation.x - LetterC_Translation.x) < 0.0);
		if (isNegativeSlope == false)
		{
			if (LetterC_Translation.x <= LetterC_Final_Translation.x)
				LetterC_Translation.x += xd;
			else
				gbLetterC_ReachedFinalPosFlag = TRUE;
		}
		else
		{
			if (LetterC_Translation.x >= LetterC_Final_Translation.x)
				LetterC_Translation.x -= xd;
			else
				gbLetterC_ReachedFinalPosFlag = TRUE;
		}
		static float z = LetterC_Translation.z - (slope * LetterC_Translation.x);
		//y=mx+c
		//if (LetterC_Translation.z >= LetterC_Final_Translation.z)
		LetterC_Translation.z = slope * LetterC_Translation.x + z;
	}

	//O 2 Linear translation
	if (gbLetterO_2_SetFlag == TRUE || gbUniversalLetterMoveFlag == TRUE)
	{// calculating interpolation along line using line equation, y = mx + c, x is varied by ourself with xd, y will be calculated accordingly
		static float slope = (LetterO_2_Final_Translation.z - LetterO_2_Translation.z) / (LetterO_2_Final_Translation.x - LetterO_2_Translation.x);
		static float xd = fabs(LetterO_2_Final_Translation.x - LetterO_2_Translation.x) / interpolationPoints;
		static bool isNegativeSlope = ((LetterO_2_Final_Translation.z - LetterO_2_Translation.z) < 0.0f) || ((LetterO_2_Final_Translation.x - LetterO_2_Translation.x) < 0.0);
		if (isNegativeSlope == false)
		{
			if (LetterO_2_Translation.x <= LetterO_2_Final_Translation.x)
				LetterO_2_Translation.x += xd;
			else
				gbLetterO_2_ReachedFinalPosFlag = TRUE;
		}
		else
		{
			if (LetterO_2_Translation.x >= LetterO_2_Final_Translation.x)
				LetterO_2_Translation.x -= xd;
			else
				gbLetterO_2_ReachedFinalPosFlag = TRUE;
		}
		static float z = LetterO_2_Translation.z - (slope * LetterO_2_Translation.x);
		//y=mx+c
		//if (LetterO_2_Translation.z >= LetterO_2_Final_Translation.z)
		LetterO_2_Translation.z = slope * LetterO_2_Translation.x + z;
	}

	//M 2 Linear translation
	if (gbLetterM_2_SetFlag == TRUE || gbUniversalLetterMoveFlag == TRUE)
	{
		static float slope = (LetterM_2_Final_Translation.z - LetterM_2_Translation.z) / (LetterM_2_Final_Translation.x - LetterM_2_Translation.x);
		static float xd = fabs(LetterM_2_Final_Translation.x - LetterM_2_Translation.x) / interpolationPoints;
		static bool isNegativeSlope = ((LetterM_2_Final_Translation.z - LetterM_2_Translation.z) < 0.0f) || ((LetterM_2_Final_Translation.x - LetterM_2_Translation.x) < 0.0);


		if (isNegativeSlope == false)
		{
			if (LetterM_2_Translation.x <= LetterM_2_Final_Translation.x)
				LetterM_2_Translation.x += xd;
			else
				gbLetterM_2_ReachedFinalPosFlag = TRUE;
		}
		else
		{
			if (LetterM_2_Translation.x >= LetterM_2_Final_Translation.x)
				LetterM_2_Translation.x -= xd;
			else
				gbLetterM_2_ReachedFinalPosFlag = TRUE;
		}

		static float z = LetterM_2_Translation.z - (slope * LetterM_2_Translation.x);
		//y=mx+c
		//if (LetterM_2_Translation.z >= LetterM_2_Final_Translation.z)
			LetterM_2_Translation.z = slope * LetterM_2_Translation.x + z;
	}

	//P Linear translation
	if (gbLetterP_SetFlag == TRUE || gbUniversalLetterMoveFlag == TRUE)
	{// calculating interpolation along line using line equation, y = mx + c, x is varied by ourself with xd, y will be calculated accordingly
		static float slope = (LetterP_Final_Translation.z - LetterP_Translation.z) / (LetterP_Final_Translation.x - LetterP_Translation.x);
		static float xd = fabs(LetterP_Final_Translation.x - LetterP_Translation.x) / interpolationPoints;
		static bool isNegativeSlope = ((LetterP_Final_Translation.z - LetterP_Translation.z) < 0.0f) || ((LetterP_Final_Translation.x - LetterP_Translation.x) < 0.0);
		if (isNegativeSlope == false)
		{
			if (LetterP_Translation.x <= LetterP_Final_Translation.x)
				LetterP_Translation.x += xd;
			else
				gbLetterP_ReachedFinalPosFlag = TRUE;
		}
		else
		{
			if (LetterP_Translation.x >= LetterP_Final_Translation.x)
				LetterP_Translation.x -= xd;
			else
				gbLetterP_ReachedFinalPosFlag = TRUE;
		}
		static float z = LetterP_Translation.z - (slope * LetterP_Translation.x);
		//y=mx+c
		//if (LetterP_Translation.z >= LetterP_Final_Translation.z)
		LetterP_Translation.z = slope * LetterP_Translation.x + z;
	}

	if (gbLetterA_ReachedFinalPosFlag && gbLetterS_ReachedFinalPosFlag && gbLetterT_ReachedFinalPosFlag &&
		gbLetterR_ReachedFinalPosFlag && gbLetterO_ReachedFinalPosFlag && gbLetterM_ReachedFinalPosFlag &&
		gbLetterE_ReachedFinalPosFlag && gbLetterD_ReachedFinalPosFlag && gbLetterI_ReachedFinalPosFlag &&
		gbLetterC_ReachedFinalPosFlag && gbLetterO_2_ReachedFinalPosFlag && gbLetterM_2_ReachedFinalPosFlag &&
		gbLetterP_ReachedFinalPosFlag)
	{
		gbAllLettersReachedFinalPosFlag = TRUE;		
		gbUniversalLetterMoveFlag = TRUE;
	}



	pCPlayer->PlayerUpdate(gfDeltaTime);

	pCGrass->ControlGrassWaving(gfDeltaTime);

	if (gbIsDayNightCyclePaused == GL_FALSE)
		pCSkyDome->UpdateSky(gfDeltaTime);

	if (gbIsSceneEnded == GL_TRUE)
	{
		EndCreditScene(gfDeltaTime);
	}

	if (gisDisplayNames == 1) 
	{
		gFontTransition += 0.01;
		if (gFontTransition > 7.5)
			gFontTransition = 0.0f;

		gAlphaFont += 0.01f;
		if (gAlphaFont >= 1.0) {
			gAlphaFont = 0.0f;
		}
		gAlphaFontN -= 0.01f;
		if (gAlphaFontN <= 0.0) {
			gAlphaFontN = 1.0f;
		}
	}
}

void UpdateLetters(void)
{

}


void FadeIn(GLfloat Delta)
{
	if (gbIsFadeInComplete != GL_TRUE)
	{
		fExposure += 0.01f;
		if (fExposure >= 1.0f)
		{
			fExposure = 1.0f;
			gbIsFadeInComplete = GL_TRUE;
			gbIsFadeOutComplete = GL_FALSE;
		}
	}
}

void FadeOut(GLfloat Delta)
{
	if (gbIsFadeOutComplete != GL_TRUE)
	{
		fExposure -= 0.01f;
		if (fExposure <= 0.0f)
		{
			fExposure = 0.0f;
			gbIsFadeOutComplete = GL_TRUE;
			gbIsFadeInComplete = GL_FALSE;
		}
	}

	fprintf_s(pCCommon->pLogFile, "fExposure: %f", fExposure);
}

void EndCreditScene(GLfloat Delta)
{
	if (gbFirstFadeOut == GL_FALSE)
	{
		FadeOut(Delta);
	}
		

	if (fExposure <= 0.0f && gbFirstFadeOut == GL_FALSE)
		gbFirstFadeOut = GL_TRUE;

	if (gbFirstFadeInPreparationComplete == GL_FALSE && gbFirstFadeOut == GL_TRUE)
	{
		gbIsDayNightCyclePaused = GL_TRUE;

		gfWaterHeight = 25.0f;

		pCPlayer->SetPlayerPosition(glm::vec3(0.0f));
		pCPlayer->SetCameraLook(glm::vec3(0.0f));
		pCPlayer->SetCameraSide(glm::vec3(0.0f));
		pCPlayer->SetCameraUp(glm::vec3(0.0f));
		pCPlayer->SetCameraFront(glm::vec3(0.0f));
		pCPlayer->SetPlayerFront(glm::vec3(0.0f));

		pCSkyDome->SetSunCDirY(0.0f);
		pCSkyDome->SetSunWPos(glm::vec3(0.0f));
		pCSkyDome->SetSunCPos(glm::vec3(0.0f));
		pCSkyDome->SetSunCDir(glm::vec3(0.0f));

		
		// hard coded recorded values
		pCPlayer->SetPlayerPosition(glm::vec3(-198.260040f, 27.100000f, 144.541687f));
		pCPlayer->SetCameraLook(glm::vec3(-197.311157f, 27.085913f, 144.226395f));
		pCPlayer->SetCameraSide(glm::vec3(0.315325f, 0.000000f, 0.948984f));
		pCPlayer->SetCameraUp(glm::vec3(0.013369f, 0.999901f, -0.004442f));
		pCPlayer->SetCameraFront(glm::vec3(0.948890f, -0.014088f, -0.315294f));
		//pCPlayer->SetPlayerFront(glm::vec3(0.948984f, 0.000000f, -0.315325f));

		pCSkyDome->SetSunCDirY(0.027667f);
		pCSkyDome->SetSunWPos(glm::vec3(298.667419f, 41.265259f, 125.923248f));
		pCSkyDome->SetSunCPos(glm::vec3(511.762482f, 12.961061f, -8.551699f));
		pCSkyDome->SetSunCDir(glm::vec3(0.999451f, 0.027667f, -0.018255f));

		gbFirstFadeInPreparationComplete = GL_TRUE;
	}

	if (gbFirstFadeIn == GL_FALSE)
		FadeIn(Delta);
	
	if (fExposure >= 1.0f)
		gbFirstFadeIn = GL_TRUE;

	if (gbFirstFadeIn == GL_TRUE && gbNameShown == GL_FALSE)
	{
		gbIsDayNightCyclePaused = GL_FALSE;
		pCSkyDome->TransitionSpeed = 0.003f;
		gisDisplayNames = 1;
			//gbFirstFadeIn = GL_FALSE;
	}

	if (gisDisplayNames == 0 && gbSecondFadeOut == GL_FALSE)
	{
		FadeOut(Delta);
	}

	if (fExposure <= 0 && gbSecondFadeOut == GL_FALSE)
	{
		gbSecondFadeOut = GL_TRUE;
	}

}


void Resize(int Width, int Height)
{
	glm::mat4 SetInfiniteProjectionMatrix(GLfloat Left, GLfloat Right, GLfloat Bottom, GLfloat Top, GLfloat Near, GLfloat Far);

	GLfloat fLeft = 0.0f;
	GLfloat fRight = 0.0f;
	GLfloat fBottom = 0.0f;
	GLfloat fTop = 0.0f;
	GLfloat fNear = FRUSTUM_NEAR;
	GLfloat fFar = FRUSTUM_FAR;

	WindowSizeHandler::setFrameBufferSize((GLsizei)Width, (GLsizei)Height);

	if (Height == 0)
		Height = 1;

	if (Width == 0)
		Width = 1;

	//if (Width > Height) 
	//{
	//	glViewport(0, (Height - Width) / 2, (GLsizei)Width, (GLsizei)Width);
	//}
	//else 
	//{
	//	glViewport((Width - Height) / 2, 0, (GLsizei)Height, (GLsizei)Height);
	//}

	glViewport(0, 0, (GLsizei)Width, (GLsizei)Height);

	fTop = fNear * (GLfloat)tan(gfFOVY / 360.0 * glm::pi<float>());
	fRight = fTop * ((float)Width / (float)Height);

	fBottom = -fTop;
	fLeft = -fTop * ((float)Width / (float)Height);

	gPerspectiveProjectionMatrix = SetInfiniteProjectionMatrix(fLeft, fRight, fBottom, fTop, fNear, fFar);

	GLfloat aspectRatio = (GLfloat)Width / (GLfloat)Height;

	gPerspectiveProjectionMatrixFont = glm::perspective(45.0f, aspectRatio, 0.1f, 1000.0f);

	/*glBindTexture(GL_TEXTURE_2D, DepthTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, Width, Height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	giSunTextureWidth = Width / 2;
	giSunTextureHeight = Height / 2;

	for (int i = 0; i < 4; i++)
	{
	glBindTexture(GL_TEXTURE_2D, SunTextures[i]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, giSunTextureWidth, giSunTextureHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	}

	glUseProgram(gShaderProgramObjectBlurHorizontal);
	glUniform1f(glGetUniformLocation(gShaderProgramObjectBlurHorizontal, "odw"), 1.0f / (float)giSunTextureWidth);
	glUseProgram(gShaderProgramObjectBlurVertical);
	glUniform1f(glGetUniformLocation(gShaderProgramObjectBlurVertical, "odh"), 1.0f / (float)giSunTextureHeight);
	glUseProgram(0);*/

	//////////////////////////////////////////////////////
	// resize msaa framebuffer

	glBindFramebuffer(GL_FRAMEBUFFER, gFBOMSAA);

	// create a multisampled color attachment
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, gTextureMSAA);
	glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, MSAA_SAMPLES, GL_RGB, giWinWidth, giWinHeight, GL_TRUE);
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, gTextureMSAA, 0);

	glBindRenderbuffer(GL_RENDERBUFFER, gRBOMSAA);
	glRenderbufferStorageMultisample(GL_RENDERBUFFER, MSAA_SAMPLES, GL_DEPTH24_STENCIL8, giWinWidth, giWinHeight);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, gRBOMSAA);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	////////////////////////////////////////////////////
	// resize hdr framebuffer
	glBindTexture(GL_TEXTURE_2D, gColorBufferTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, giWinWidth, giWinHeight, 0, GL_RGBA, GL_FLOAT, NULL);

	// create depth buffer (renderbuffer)
	glBindRenderbuffer(GL_RENDERBUFFER, gRenderBufferObjectDepth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, giWinWidth, giWinHeight);

	// attach buffers
	glBindFramebuffer(GL_FRAMEBUFFER, gFrameBufferObjectHDR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gColorBufferTexture, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, gRenderBufferObjectDepth);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

glm::mat4 SetInfiniteProjectionMatrix(GLfloat Left, GLfloat Right, GLfloat Bottom, GLfloat Top, GLfloat Near, GLfloat Far)
{
	glm::mat4 result(glm::mat4(1.0f));

	if ((Right == Left) || (Top == Bottom) || (Near == Far) || (Near < 0.0) || (Far < 0.0))
		return result;

	result[0][0] = (2.0f * Near) / (Right - Left);
	result[1][1] = (2.0f * Near) / (Top - Bottom);

	result[2][0] = (Right + Left) / (Right - Left);
	result[2][1] = (Top + Bottom) / (Top - Bottom);
	//result[2][2] = -(Far + Near) / (Far - Near);
	result[2][2] = -1.0f;
	result[2][3] = -1.0f;

	//result[3][2] = -(2.0f * Far * Near) / (Far - Near);
	result[3][2] = -(2.0f * Near);
	result[3][3] = 0.0f;

	return result;
}

void Uninitialize(void)
{
	pWater->cleanUp();
	pCTerrain->DeleteTerrain();

	if (gbIsFullscreen == true)
	{
		dwStyle = GetWindowLong(ghWnd, GWL_STYLE);
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghWnd, &wpPrev);
		SetWindowPos(ghWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
	}

	if (gVAOPyramid)
	{
		glDeleteVertexArrays(1, &gVAOPyramid);
		gVAOPyramid = 0;
	}

	if (gVAOCube)
	{
		glDeleteVertexArrays(1, &gVAOCube);
		gVAOCube = 0;
	}

	if (gVBOPosition)
	{
		glDeleteVertexArrays(1, &gVBOPosition);
		gVBOPosition = 0;
	}

	if (gVBOColor)
	{
		glDeleteVertexArrays(1, &gVBOColor);
		gVBOColor = 0;
	}

	//delete[] pCCamera;
	//delete[] pCPlayer;
	//delete[] pCTerrain;
	//delete[] pCSkyDome;
	//delete[] pCGrass;

	glUseProgram(0);

	wglMakeCurrent(NULL, NULL);

	wglDeleteContext(ghRC);
	ghRC = NULL;

	ReleaseDC(ghWnd, ghDC);
	ghDC = NULL;



	if (pCCommon->pLogFile)
	{
		fopen_s(&pCCommon->pLogFile, "Logs/GeneralLog.txt", "a+");
		fprintf(pCCommon->pLogFile, "Log File Is Successfully Closed.\n");
		fclose(pCCommon->pLogFile);
		pCCommon->pLogFile = NULL;
	}

	DestroyWindow(ghWnd);
}

void InitializeModels(void)
{
	//MessageBox(ghWnd, TEXT("before loading conifer model"), TEXT("Msg"), MB_OK);

	gpConifer = new ModelLoad::Model("Resources/Models/ConiferOBJ/Conifer.obj");
	if (gpConifer == nullptr)
	{
		fopen_s(&pCCommon->pLogFile, "Logs/GeneralLog.txt", "a+");
		fprintf_s(pCCommon->pLogFile, "ERROR: ModelLoad::Shader() Failed. gpConifer=NULL\n");
		fclose(pCCommon->pLogFile);
		Uninitialize();
		exit(0);
	}
	
		// For AstroMediComp
	//LetterA
	gpLetterA_obj = new ModelLoad::Model((GLchar *)"Resources/Models/LettersObj/LetterA.obj");	//worked
	gpLetterA_obj->InitShaders("Resources/Shaders/ModelLoadingLetters.vs","Resources/Shaders/ModelLoadingLetters.frag");
	//gpLetterA_obj->InitShaders("Resources/Shaders/BasicShader.vert.glsl", "Resources/Shaders/BasicShader.frag.glsl");

	////LetterS
	gpLetterS_obj = new ModelLoad::Model((GLchar *)"Resources/Models/LettersObj/LetterS.obj");	//worked
	gpLetterS_obj->InitShaders("Resources/Shaders/ModelLoadingLetters.vs", "Resources/Shaders/ModelLoadingLetters.frag");

	//LetterT
	gpLetterT_obj = new ModelLoad::Model((GLchar *)"Resources/Models/LettersObj/LetterT.obj");	//worked
	gpLetterT_obj->InitShaders("Resources/Shaders/ModelLoadingLetters.vs", "Resources/Shaders/ModelLoadingLetters.frag");

	//LetterR
	gpLetterR_obj = new ModelLoad::Model((GLchar *)"Resources/Models/LettersObj/LetterR.obj");	//worked
	gpLetterR_obj->InitShaders("Resources/Shaders/ModelLoadingLetters.vs", "Resources/Shaders/ModelLoadingLetters.frag");

	//LetterO
	gpLetterO_obj = new ModelLoad::Model((GLchar *)"Resources/Models/LettersObj/LetterO.obj");	//worked
	gpLetterO_obj->InitShaders("Resources/Shaders/ModelLoadingLetters.vs", "Resources/Shaders/ModelLoadingLetters.frag");

	//LetterM
	gpLetterM_obj = new ModelLoad::Model((GLchar *)"Resources/Models/LettersObj/LetterM.obj");	//worked
	gpLetterM_obj->InitShaders("Resources/Shaders/ModelLoadingLetters.vs", "Resources/Shaders/ModelLoadingLetters.frag");

	//LetterE
	gpLetterE_obj = new ModelLoad::Model((GLchar *)"Resources/Models/LettersObj/LetterE.obj");	//worked
	gpLetterE_obj->InitShaders("Resources/Shaders/ModelLoadingLetters.vs", "Resources/Shaders/ModelLoadingLetters.frag");

	//LetterD
	gpLetterD_obj = new ModelLoad::Model((GLchar *)"Resources/Models/LettersObj/LetterD.obj");	//worked
	gpLetterD_obj->InitShaders("Resources/Shaders/ModelLoadingLetters.vs", "Resources/Shaders/ModelLoadingLetters.frag");

	//LetterI
	gpLetterI_obj = new ModelLoad::Model((GLchar *)"Resources/Models/LettersObj/LetterI.obj");	//worked
	gpLetterI_obj->InitShaders("Resources/Shaders/ModelLoadingLetters.vs", "Resources/Shaders/ModelLoadingLetters.frag");

	//LetterC
	gpLetterC_obj = new ModelLoad::Model((GLchar *)"Resources/Models/LettersObj/LetterC.obj");	//worked
	gpLetterC_obj->InitShaders("Resources/Shaders/ModelLoadingLetters.vs", "Resources/Shaders/ModelLoadingLetters.frag");
	
	//LetterO_2
	gpLetterO_2_obj = new ModelLoad::Model((GLchar *)"Resources/Models/LettersObj/LetterO.obj");	//worked
	gpLetterO_2_obj->InitShaders("Resources/Shaders/ModelLoadingLetters.vs", "Resources/Shaders/ModelLoadingLetters.frag");

	//LetterM_2
	gpLetterM_2_obj = new ModelLoad::Model((GLchar *)"Resources/Models/LettersObj/LetterM.obj");	//worked
	gpLetterM_2_obj->InitShaders("Resources/Shaders/ModelLoadingLetters.vs", "Resources/Shaders/ModelLoadingLetters.frag");

	//LetterP
	gpLetterP_obj = new ModelLoad::Model((GLchar *)"Resources/Models/LettersObj/LetterP.obj");	//worked
	gpLetterP_obj->InitShaders("Resources/Shaders/ModelLoadingLetters.vs", "Resources/Shaders/ModelLoadingLetters.frag");

	// For Raster
	//LetterR
	gpLetterR_RG_obj = new ModelLoad::Model((GLchar *)"Resources/Models/LettersObj/LetterR.obj");	//worked
	gpLetterR_RG_obj->InitShaders("Resources/Shaders/ModelLoadingLetters.vs", "Resources/Shaders/ModelLoadingLetters.frag");

	//LetterA
	gpLetterA_RG_obj = new ModelLoad::Model((GLchar *)"Resources/Models/LettersObj/LetterA.obj");	//worked
	gpLetterA_RG_obj->InitShaders("Resources/Shaders/ModelLoadingLetters.vs", "Resources/Shaders/ModelLoadingLetters.frag");

	//LetterS
	gpLetterS_RG_obj = new ModelLoad::Model((GLchar *)"Resources/Models/LettersObj/LetterS.obj");	//worked
	gpLetterS_RG_obj->InitShaders("Resources/Shaders/ModelLoadingLetters.vs", "Resources/Shaders/ModelLoadingLetters.frag");

	//LetterT
	gpLetterT_RG_obj = new ModelLoad::Model((GLchar *)"Resources/Models/LettersObj/LetterT.obj");	//worked
	gpLetterT_RG_obj->InitShaders("Resources/Shaders/ModelLoadingLetters.vs", "Resources/Shaders/ModelLoadingLetters.frag");

	//LetterE
	gpLetterE_RG_obj = new ModelLoad::Model((GLchar *)"Resources/Models/LettersObj/LetterE.obj");	//worked
	gpLetterE_RG_obj->InitShaders("Resources/Shaders/ModelLoadingLetters.vs", "Resources/Shaders/ModelLoadingLetters.frag");

	//LetterR
	gpLetterR_RG2_obj = new ModelLoad::Model((GLchar *)"Resources/Models/LettersObj/LetterR.obj");	//worked
	gpLetterR_RG2_obj->InitShaders("Resources/Shaders/ModelLoadingLetters.vs", "Resources/Shaders/ModelLoadingLetters.frag");

	//MessageBox(ghWnd, TEXT("after loading conifer model"), TEXT("Msg"), MB_OK);

	gpBroadLeaf = new ModelLoad::Model("Resources/Models/BroadLeafOBJ/Broadleaf_Hero_Field.obj");
	if (gpBroadLeaf == nullptr)
	{
		fopen_s(&pCCommon->pLogFile, "Logs/GeneralLog.txt", "a+");
		fprintf_s(pCCommon->pLogFile, "ERROR: ModelLoad::Shader() Failed. gpBroadLeaf=NULL\n");
		fclose(pCCommon->pLogFile);
		Uninitialize();
		exit(0);
	}

	//MessageBox(ghWnd, TEXT("after loading broadleaf model"), TEXT("Msg"), MB_OK);

	gpBroadLeaf2 = new ModelLoad::Model("Resources/Models/BroadLeafOBJ2/Broadleaf_Desktop_Field.obj");
	if (gpBroadLeaf2 == nullptr)
	{
		fopen_s(&pCCommon->pLogFile, "Logs/GeneralLog.txt", "a+");
		fprintf_s(pCCommon->pLogFile, "ERROR: ModelLoad::Shader() Failed. gpBroadLeaf2 =NULL\n");
		fclose(pCCommon->pLogFile);
		Uninitialize();
		exit(0);
	}

	//MessageBox(ghWnd, TEXT("after loading broadleaf 2 model"), TEXT("Msg"), MB_OK);

	gpBush = new ModelLoad::Model("Resources/Models/BushOBJ/Bush_Desktop.obj");
	if (gpBush == nullptr)
	{
		fopen_s(&pCCommon->pLogFile, "Logs/GeneralLog.txt", "a+");
		fprintf_s(pCCommon->pLogFile, "ERROR: ModelLoad::Shader() Failed. gpBush=NULL\n");
		fclose(pCCommon->pLogFile);
		Uninitialize();
		exit(0);
	}

	//MessageBox(ghWnd, TEXT("after loading Bush model"), TEXT("Msg"), MB_OK);
}

void DrawTrees(glm::mat4 ViewMatrix, glm::mat4 ProjectionMatrix)
{
	glm::mat4 ModelMatrix = glm::mat4(1.0f);
	glm::mat4 TranslationMatrix = glm::mat4(1.0f);
	glm::mat4 RotationMatrix = glm::mat4(1.0f);
	glm::mat4 ScaleMatrix = glm::mat4(1.0f);

	glDisable(GL_CULL_FACE);
	if (gbIsNight == GL_FALSE)
	{
		gpConifer->UseProgram();

		GLint id = -1;

		id = glGetUniformLocation(gpConifer->GetProgram(), "projection");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
		id = glGetUniformLocation(gpConifer->GetProgram(), "view");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ViewMatrix));

		SetDayLight(gpConifer->GetProgram());

		ConiferPosition.y = pCTerrain->GetHeightAt(ConiferPosition);
		TranslationMatrix = glm::translate(glm::mat4(1.0f), ConiferPosition);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		id = glGetUniformLocation(gpConifer->GetProgram(), "model");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ModelMatrix));

		gpConifer->Draw(gfDeltaTime);


		// draw broadleaf
		gpBroadLeaf->UseProgram();

		ModelMatrix = glm::mat4(1.0f);
		TranslationMatrix = glm::mat4(1.0f);
		RotationMatrix = glm::mat4(1.0f);

		id = -1;

		id = glGetUniformLocation(gpBroadLeaf->GetProgram(), "projection");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
		id = glGetUniformLocation(gpBroadLeaf->GetProgram(), "view");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ViewMatrix));

		SetDayLight(gpBroadLeaf->GetProgram());

		BroadleafPosition.y = pCTerrain->GetHeightAt(BroadleafPosition);
		TranslationMatrix = glm::translate(glm::mat4(1.0f), BroadleafPosition);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		id = glGetUniformLocation(gpBroadLeaf->GetProgram(), "model");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ModelMatrix));

		gpBroadLeaf->Draw(gfDeltaTime);

		//---------------------------------------------------------------------------------------
		ModelMatrix = glm::mat4(1.0f);
		TranslationMatrix = glm::mat4(1.0f);

		id = -1;

		id = glGetUniformLocation(gpBroadLeaf->GetProgram(), "projection");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
		id = glGetUniformLocation(gpBroadLeaf->GetProgram(), "view");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ViewMatrix));

		SetDayLight(gpBroadLeaf->GetProgram());

		BroadleafPosition2.y = pCTerrain->GetHeightAt(BroadleafPosition2);
		TranslationMatrix = glm::translate(glm::mat4(1.0f), BroadleafPosition2);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		id = glGetUniformLocation(gpBroadLeaf->GetProgram(), "model");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ModelMatrix));

		gpBroadLeaf->Draw(gfDeltaTime);

		//-----------------------------------------------------------------------------------
		/*ModelMatrix = glm::mat4(1.0f);
		TranslationMatrix = glm::mat4(1.0f);

		id = -1;

		id = glGetUniformLocation(gpBroadLeaf->GetProgram(), "projection");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
		id = glGetUniformLocation(gpBroadLeaf->GetProgram(), "view");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ViewMatrix));

		SetDayLight(gpBroadLeaf->GetProgram());

		BroadleafPosition3.y = pCTerrain->GetHeightAt(BroadleafPosition3);
		TranslationMatrix = glm::translate(glm::mat4(1.0f), BroadleafPosition3);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		id = glGetUniformLocation(gpBroadLeaf->GetProgram(), "model");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ModelMatrix));

		gpBroadLeaf->Draw(gfDeltaTime);*/

		//----------------------------------------------------------------------------------------
		ModelMatrix = glm::mat4(1.0f);
		TranslationMatrix = glm::mat4(1.0f);

		// draw broadleaf 2
		gpBroadLeaf2->UseProgram();

		ModelMatrix = glm::mat4(1.0f);
		TranslationMatrix = glm::mat4(1.0f);
		RotationMatrix = glm::mat4(1.0f);
		ScaleMatrix = glm::mat4(1.0f);
		id = -1;

		id = glGetUniformLocation(gpBroadLeaf2->GetProgram(), "projection");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
		id = glGetUniformLocation(gpBroadLeaf2->GetProgram(), "view");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ViewMatrix));

		Broadleaf2Position.y = pCTerrain->GetHeightAt(Broadleaf2Position);
		TranslationMatrix = glm::translate(glm::mat4(1.0f), Broadleaf2Position);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		id = glGetUniformLocation(gpBroadLeaf2->GetProgram(), "model");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ModelMatrix));

		SetDayLight(gpBroadLeaf2->GetProgram());

		gpBroadLeaf2->Draw(gfDeltaTime);

		// bush model
		gpBush->UseProgram();

		ModelMatrix = glm::mat4(1.0f);
		TranslationMatrix = glm::mat4(1.0f);
		RotationMatrix = glm::mat4(1.0f);
		ScaleMatrix = glm::mat4(1.0f);

		id = -1;

		id = glGetUniformLocation(gpBush->GetProgram(), "projection");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
		id = glGetUniformLocation(gpBush->GetProgram(), "view");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ViewMatrix));

		BushPosition.y = pCTerrain->GetHeightAt(BushPosition);
		TranslationMatrix = glm::translate(glm::mat4(1.0f), BushPosition);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		id = glGetUniformLocation(gpBush->GetProgram(), "model");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ModelMatrix));

		SetDayLight(gpBush->GetProgram());
		gpBush->Draw(gfDeltaTime);
	}
	else
	{
		gpConifer->UseProgramNight();

		ModelMatrix = glm::mat4(1.0f);
		TranslationMatrix = glm::mat4(1.0f);
		RotationMatrix = glm::mat4(1.0f);
		ScaleMatrix = glm::mat4(1.0f);
		GLint id = -1;

		id = glGetUniformLocation(gpConifer->GetProgramNight(), "projection");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
		id = glGetUniformLocation(gpConifer->GetProgramNight(), "view");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ViewMatrix));

		ConiferPosition.y = pCTerrain->GetHeightAt(ConiferPosition);
		TranslationMatrix = glm::translate(glm::mat4(1.0f), ConiferPosition);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		id = glGetUniformLocation(gpConifer->GetProgramNight(), "model");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ModelMatrix));

		SetNightLight(gpConifer->GetProgramNight());
		gpConifer->Draw(gfDeltaTime);
		///////////////////////////////////////////////////////////////////////////////////////

		// draw broadleaf
		gpBroadLeaf->UseProgramNight();

		ModelMatrix = glm::mat4(1.0f);
		TranslationMatrix = glm::mat4(1.0f);
		RotationMatrix = glm::mat4(1.0f);

		id = -1;

		id = glGetUniformLocation(gpBroadLeaf->GetProgramNight(), "projection");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
		id = glGetUniformLocation(gpBroadLeaf->GetProgramNight(), "view");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ViewMatrix));

		BroadleafPosition.y = pCTerrain->GetHeightAt(BroadleafPosition);
		TranslationMatrix = glm::translate(glm::mat4(1.0f), BroadleafPosition);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		id = glGetUniformLocation(gpBroadLeaf->GetProgramNight(), "model");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ModelMatrix));

		SetNightLight(gpBroadLeaf->GetProgramNight());

		gpBroadLeaf->Draw(gfDeltaTime);

		//--------------------------------------------------------------------------------------

		ModelMatrix = glm::mat4(1.0f);
		TranslationMatrix = glm::mat4(1.0f);
		RotationMatrix = glm::mat4(1.0f);

		id = -1;

		id = glGetUniformLocation(gpBroadLeaf->GetProgramNight(), "projection");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
		id = glGetUniformLocation(gpBroadLeaf->GetProgramNight(), "view");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ViewMatrix));

		BroadleafPosition2.y = pCTerrain->GetHeightAt(BroadleafPosition2);
		TranslationMatrix = glm::translate(glm::mat4(1.0f), BroadleafPosition2);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		id = glGetUniformLocation(gpBroadLeaf->GetProgramNight(), "model");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ModelMatrix));

		SetNightLight(gpBroadLeaf->GetProgramNight());

		gpBroadLeaf->Draw(gfDeltaTime);

		//---------------------------------------------------------------------------------------------

		/*ModelMatrix = glm::mat4(1.0f);
		TranslationMatrix = glm::mat4(1.0f);
		RotationMatrix = glm::mat4(1.0f);

		id = -1;

		id = glGetUniformLocation(gpBroadLeaf->GetProgramNight(), "projection");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
		id = glGetUniformLocation(gpBroadLeaf->GetProgramNight(), "view");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ViewMatrix));

		BroadleafPosition3.y = pCTerrain->GetHeightAt(BroadleafPosition3);
		TranslationMatrix = glm::translate(glm::mat4(1.0f), BroadleafPosition3);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		id = glGetUniformLocation(gpBroadLeaf->GetProgramNight(), "model");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ModelMatrix));

		SetNightLight(gpBroadLeaf->GetProgramNight());

		gpBroadLeaf->Draw(gfDeltaTime);*/
		//////////////////////////////////////////////////////////////////////////////////////////////

		///////////////////////////////////////////////////////////
		// draw broadleaf 2
		gpBroadLeaf2->UseProgramNight();

		ModelMatrix = glm::mat4(1.0f);
		TranslationMatrix = glm::mat4(1.0f);
		RotationMatrix = glm::mat4(1.0f);
		ScaleMatrix = glm::mat4(1.0f);
		id = -1;

		id = glGetUniformLocation(gpBroadLeaf2->GetProgramNight(), "projection");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
		id = glGetUniformLocation(gpBroadLeaf2->GetProgramNight(), "view");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ViewMatrix));

		Broadleaf2Position.y = pCTerrain->GetHeightAt(Broadleaf2Position);
		TranslationMatrix = glm::translate(glm::mat4(1.0f), Broadleaf2Position);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		id = glGetUniformLocation(gpBroadLeaf2->GetProgramNight(), "model");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ModelMatrix));

		SetNightLight(gpBroadLeaf2->GetProgramNight());
		gpBroadLeaf2->Draw(gfDeltaTime);

		///////////////////////////////////////////////////////////////////

		// bush model
		gpBush->UseProgramNight();

		ModelMatrix = glm::mat4(1.0f);
		TranslationMatrix = glm::mat4(1.0f);
		RotationMatrix = glm::mat4(1.0f);
		ScaleMatrix = glm::mat4(1.0f);

		id = -1;

		id = glGetUniformLocation(gpBush->GetProgramNight(), "projection");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
		id = glGetUniformLocation(gpBush->GetProgramNight(), "view");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ViewMatrix));

		BushPosition.y = pCTerrain->GetHeightAt(BushPosition);
		TranslationMatrix = glm::translate(glm::mat4(1.0f), BushPosition);
		ModelMatrix = ModelMatrix * TranslationMatrix;

		id = glGetUniformLocation(gpBush->GetProgramNight(), "model");
		glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(ModelMatrix));

		SetNightLight(gpBush->GetProgramNight());

		gpBush->Draw(gfDeltaTime);
	}
	glEnable(GL_CULL_FACE);
}

void InitializeLightCube(void)
{
	const GLfloat LightCubeVertices[] =
	{
		// Front Face
		1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,

		// Right Face
		1.0f, 1.0f, -1.0f,
		1.0f, 1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f,

		// Back Face
		-1.0f, 1.0f, -1.0f,
		1.0f, 1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,

		// Left Face
		-1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f,

		// Top Face
		1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,

		// Bottom Face
		1.0f, -1.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,
		-1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f
	};

	// For Quad
	glGenVertexArrays(1, &gVAOLightCube);
	glBindVertexArray(gVAOLightCube);

	// For Quad position
	glGenBuffers(1, &gVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, gVBOPosition);
	glBufferData(GL_ARRAY_BUFFER, sizeof(LightCubeVertices), LightCubeVertices, GL_STATIC_DRAW);

	glVertexAttribPointer(OGL_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(OGL_ATTRIBUTE_VERTEX);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void DrawLightCube(glm::vec3 LightDiffuse)
{
	glBindVertexArray(gVAOLightCube);

	glVertexAttrib3fv(OGL_ATTRIBUTE_COLOR, glm::value_ptr(LightDiffuse));

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 20, 4);
	glBindVertexArray(0);
}

void InitializeQuadHDR(void)
{
	float quadVertices[] = {
		// positions        // texture Coords
		-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
		1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
		1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
	};
	// setup plane VAO
	glGenVertexArrays(1, &gVAOHDRQuad);
	glBindVertexArray(gVAOHDRQuad);

	glGenBuffers(1, &gVBOHDR);
	glBindBuffer(GL_ARRAY_BUFFER, gVBOHDR);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_VERTEX);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);
}

void RenderQuadHDR(void)
{
	glBindVertexArray(gVAOHDRQuad);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}

void InitializeSampleQuad(void)
{
	//float quadVertices[] = {
	//	// positions        // texture Coords
	//	-0.5f,  0.5f, 0.0f, 0.0f, 1.0f,
	//	-0.5f, -0.5f, 0.0f, 0.0f, 0.0f,
	//	0.5f,  0.5f, 0.0f, 1.0f, 1.0f,
	//	0.5f, -0.5f, 0.0f, 1.0f, 0.0f,
	//};

	float quadVertices[] = {
		// positions        // texture Coords
		-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
		1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
		1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
	};

	// setup plane VAO
	glGenVertexArrays(1, &gVAOSampleQuad);
	glBindVertexArray(gVAOSampleQuad);

	glGenBuffers(1, &gVBOSampleQuad);
	glBindBuffer(GL_ARRAY_BUFFER, gVBOSampleQuad);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_VERTEX);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);
}

void RenderSampleQuad(void)
{
	glBindVertexArray(gVAOSampleQuad);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}

void OnLButtonDown(INT LeftClickX, INT LeftClickY, DWORD LeftClickFlags)
{
	RECT Clip;
	//RECT PrevClip;

	SetCapture(ghWnd);
	//GetClipCursor(&PrevClip);
	GetWindowRect(ghWnd, &Clip);
	ClipCursor(&Clip);

	GetCursorPos(&ClickPoint);
	//ShowCursor(FALSE);
}

void OnLButtonUp(INT LeftUpX, INT LeftUpY, DWORD LeftUpFlags)
{
	ClipCursor(NULL);
	ReleaseCapture();
	SetCursorPos(ClickPoint.x, ClickPoint.y);
	ShowCursor(TRUE);
}

void OnRButtonDown(INT RightClickX, INT RightClickY, DWORD RightClickFlags)
{
	RECT Clip;
	//RECT PrevClip;

	SetCapture(ghWnd);
	//GetClipCursor(&PrevClip);
	GetWindowRect(ghWnd, &Clip);
	ClipCursor(&Clip);

	GetCursorPos(&ClickPoint);
	ShowCursor(FALSE);
}

void OnRButtonUp(INT RightUpX, INT RightUpY, DWORD RightUpFlags)
{
	ClipCursor(NULL);
	ReleaseCapture();
	SetCursorPos(ClickPoint.x, ClickPoint.y);
	ShowCursor(TRUE);
}

void OnMButtonDown(INT MiddleDownX, INT MiddleDownY, DWORD MiddleDownFlags)
{
	RECT Clip;
	//RECT PrevClip;

	SetCapture(ghWnd);
	//GetClipCursor(&PrevClip);
	GetWindowRect(ghWnd, &Clip);
	ClipCursor(&Clip);

	GetCursorPos(&ClickPoint);
	ShowCursor(FALSE);
}

void OnMButtonUp(INT MiddleUpX, INT MiddleUpY, DWORD MiddleUpFlags)
{
	ClipCursor(NULL);
	ReleaseCapture();
	SetCursorPos(ClickPoint.x, ClickPoint.y);
	ShowCursor(TRUE);
}

void OnMouseMove(INT MouseMoveX, INT MouseMoveY, DWORD Flags)
{
	//fprintf(stream, "+X Threshold : %f -X Threshold : %f +Y Threshold : %f -Y Threshold : %f\n", iPositiveXThreshold, iNegativeXThreshold, iPositiveYThreshold, iNegativeYThreshold);
	//fprintf(stream, "On Mouse Move X : %d Y : %d\n", MouseMoveX, MouseMoveY);
	//ShowCursor(FALSE);


	if (Flags & MK_LBUTTON)
	{
		// Code
	}

	if (gbIsPlaying == GL_FALSE)
	{
		if (gbFirstMouse)
		{
			gfLastX = (GLfloat)MouseMoveX;
			gfLastY = (GLfloat)MouseMoveY;
			gbFirstMouse = GL_FALSE;
		}

		GLfloat fXOffset = MouseMoveX - gfLastX;
		GLfloat fYOffset = gfLastY - MouseMoveY;

		gfLastX = (GLfloat)MouseMoveX;
		gfLastY = (GLfloat)MouseMoveY;

		if (Flags & MK_RBUTTON)
		{
			pCCamera->ProcessMouseMovement(fXOffset, fYOffset, GL_TRUE);
		}
	}
	else if (gbIsPlaying == GL_TRUE)
	{
		if (gbFirstMouse)
		{
			gfLastX = (GLfloat)MouseMoveX;
			gfLastY = (GLfloat)MouseMoveY;
			gbFirstMouse = GL_FALSE;
		}

		//Player->ControlMouseInput(MouseMoveX, MouseMoveY);
	}
	ShowCursor(TRUE);
}

void OnMouseWheelScroll(short zDelta)
{
	void Resize(int Width, int Height);

	if (zDelta > 0)
	{
		gfFOVY -= ZOOM_FACTOR;
		if (gfFOVY <= 0.1f)
			gfFOVY = 0.1f;
	}
	else
	{
		gfFOVY += ZOOM_FACTOR;
		if (gfFOVY >= 45.0f)
			gfFOVY = 45.0f;
	}

	Resize(giWinWidth, giWinHeight);
}


void ScreenPositionToWorldRay(int MouseX, int MouseY, int ScreenWidth, int ScreenHeight, glm::mat4 ViewMatrix, glm::mat4 ProjectionMatrix, glm::vec3 &OutRayOrigin, glm::vec3 &OutRayDirection)
{
	// the ray start and end position in NDC
	glm::vec4 RayStartNDC(((float)MouseX / (float)ScreenWidth - 0.5f) * 2.0f, ((float)MouseY / (float)ScreenWidth - 0.5f) * 2.0f, -1.0f, 1.0f);
	glm::vec4 RayEndNDC(((float)MouseX / (float)ScreenWidth - 0.5f) * 2.0f, ((float)MouseY / (float)ScreenWidth - 0.5f) * 2.0f, 0.0f, 1.0f);

	// the projection matrix goes from camera space to NDC
	// hence to go from NDC to cmaera space find inverse of projection matrix
	glm::mat4 InverseProjectionMatrix = glm::inverse(ProjectionMatrix);

	// the view matrix goes from world space to camera space
	// hence to go from camera space to world space find inverse of view matrix
	glm::mat4 InverseViewMatrix = glm::inverse(ViewMatrix);

	glm::vec4 RayStartCamera = InverseProjectionMatrix * RayStartNDC;
	RayStartCamera /= RayStartCamera.w;

	glm::vec4 RayStartWorld = InverseViewMatrix * RayStartCamera;
	RayStartWorld /= RayStartWorld.w;

	glm::vec4 RayEndCamera = InverseProjectionMatrix * RayEndNDC;
	RayEndCamera /= RayEndCamera.w;

	glm::vec4 RayEndWorld = InverseViewMatrix * RayEndCamera;
	RayEndWorld /= RayEndWorld.w;

	// ray direction in world
	glm::vec3 RayDirWorld(RayEndWorld - RayStartWorld);
	RayDirWorld = glm::normalize(RayDirWorld);

	OutRayOrigin = glm::vec3(RayStartWorld);
	OutRayDirection = glm::normalize(RayDirWorld);
}

// parameters
// ray origin, in world space
// ray direction (not target position), in world space
// minimum x, y, z coordinates of the mesh (axis aligned bounding box) when not transformed
// maximum x, y, z coordinates. Often AABBMin * -1 if your mesh is centered (but this is not always a case)
// model matrix for transformation applied to the mesh so that bounding box will be transformed accordingly
// output: distance between ray origin and the intersection with the bounding box
bool TestRayOBBIntersection(glm::vec3 RayOrigin, glm::vec3 RayDirection, glm::vec3 AABBMin, glm::vec3 AABBMax, glm::mat4 ModelMatrix, float &IntersectionDistance)
{
	float tMin = 0.0f;
	float tMax = 100000.0f;

	// oriented bounding box position in world space
	glm::vec3 OBBPositionWorldSpace(ModelMatrix[3].x, ModelMatrix[3].y, ModelMatrix[3].z);

	glm::vec3 Delta = OBBPositionWorldSpace - RayOrigin;

	// test intersection with the 2 planes perpendicular to the OBB's X axis
	{
		glm::vec3 XAxis(ModelMatrix[0].x, ModelMatrix[0].y, ModelMatrix[0].z);
		float e = glm::dot(XAxis, Delta);
		float f = glm::dot(RayDirection, XAxis);

		if (fabs(f) > 0.001f)
		{
			float t1 = (e + AABBMin.x) / f; // intersection with the left plane
			float t2 = (e + AABBMax.x) / f; // intersection with the right plane
											// t1 and t2 now contain distance between ray origin and rat-plane intersections

											// t1 to represent the nearest intersection,
											// if it's not the case, invert t1 and t2
			if (t1 > t2)
			{
				// swap t1 and t2
				float w = t1;
				t1 = t2;
				t2 = w;
			}

			// tMax is the nearest "far" intersection (amongst the x, y and z planes pair)
			if (t2 < tMax)
				tMax = t2;

			// tMin is the farthest "near" intersection (amongst the x, y and z planes pair)
			if (t1 > tMin)
				tMin = t1;

			// if "far" is closer than "near", then there is nointersection
			if (tMax < tMin)
				return false;
		}
		else
		{
			// if the ray is almost parallel to the planes, so they don't have any intersection
			if (-e + AABBMin.x > 0.0f || -e + AABBMax.x < 0.0f)
				return false;
		}
	}

	// test intersection with the 2 planes perpendicular to the OBB's Y axis
	{
		glm::vec3 YAxis(ModelMatrix[0].x, ModelMatrix[0].y, ModelMatrix[0].z);
		float e = glm::dot(YAxis, Delta);
		float f = glm::dot(RayDirection, YAxis);

		if (fabs(f) > 0.001f)
		{
			float t1 = (e + AABBMin.y) / f;
			float t2 = (e + AABBMax.y) / f;

			if (t1 > t2)
			{
				float w = t1;
				t1 = t2;
				t2 = w;
			}

			if (t2 < tMax)
				tMax = t2;

			if (t1 > tMin)
				tMin = t1;

			if (tMin > tMax)
				return false;
		}
		else
		{
			if (-e + AABBMin.y > 0.0f || -e + AABBMax.y < 0.0f)
				return false;
		}
	}

	// test intersection with the 2 planes perpendicular to the OBB's Z axis
	{
		glm::vec3 ZAxis(ModelMatrix[2].x, ModelMatrix[0].y, ModelMatrix[0].z);
		float e = glm::dot(ZAxis, Delta);
		float f = glm::dot(RayDirection, ZAxis);

		if (fabs(f) > 0.001f)
		{
			float t1 = (e + AABBMin.z) / f;
			float t2 = (e + AABBMax.z) / f;

			if (t1 > t2)
			{
				float w = t1;
				t1 = t2;
				t2 = w;
			}

			if (t2 < tMax)
				tMax = t2;

			if (t1 > tMin)
				tMin = t1;

			if (tMin > tMax)
				return false;
		}
		else
		{
			if (-e + AABBMin.z > 0.0f || -e + AABBMax.z < 0.0f)
				return false;
		}
	}

	IntersectionDistance = tMin;
	return true;
}


void SetDayLight(GLuint ShaderObject)
{
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.ambient"), 1, glm::value_ptr(pSunLight->GetAmbient()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.diffuse"), 1, glm::value_ptr(pSunLight->GetDiffuse()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.specular"), 1, glm::value_ptr(pSunLight->GetSpecular()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.direction"), 1, glm::value_ptr(pSunLight->GetDirection()));

	glUniform3fv(glGetUniformLocation(ShaderObject, "u_view_position"), 1, glm::value_ptr(CameraPosition));
}

void SetNightLight(GLint ShaderObject)
{
	glUniform1i(glGetUniformLocation(ShaderObject, "num_lights"), MAX_POINT_LIGHTS);
	if (gbIsTorchOn == GL_TRUE)
		glUniform1i(glGetUniformLocation(ShaderObject, "torch"), 1);
	else
		glUniform1i(glGetUniformLocation(ShaderObject, "torch"), 0);

	// send all point lights data
	for (int i = 0; i < MAX_POINT_LIGHTS; i++)
	{
		// light ads position
		strLightProperty.str("");
		strLightProperty << "PointLight[" << i << "].ambient";
		glUniform3fv(glGetUniformLocation(ShaderObject, strLightProperty.str().c_str()), 1, glm::value_ptr(pLamp[i].GetAmbient()));

		strLightProperty.str("");
		strLightProperty << "PointLight[" << i << "].diffuse";
		glUniform3fv(glGetUniformLocation(ShaderObject, strLightProperty.str().c_str()), 1, glm::value_ptr(pLamp[i].GetDiffuse()));

		strLightProperty.str("");
		strLightProperty << "PointLight[" << i << "].specular";
		glUniform3fv(glGetUniformLocation(ShaderObject, strLightProperty.str().c_str()), 1, glm::value_ptr(pLamp[i].GetSpecular()));

		strLightProperty.str("");
		strLightProperty << "PointLight[" << i << "].position";
		glUniform3fv(glGetUniformLocation(ShaderObject, strLightProperty.str().c_str()), 1, glm::value_ptr(pLamp[i].GetPosition()));

		// attenuation
		strLightProperty.str("");
		strLightProperty << "PointLight[" << i << "].constant_attenuation";
		glUniform1f(glGetUniformLocation(ShaderObject, strLightProperty.str().c_str()), pLamp[i].GetConstantAttenuation());

		strLightProperty.str("");
		strLightProperty << "PointLight[" << i << "].linear_attenuation";
		glUniform1f(glGetUniformLocation(ShaderObject, strLightProperty.str().c_str()), pLamp[1].GetLinearAttenuation());

		strLightProperty.str("");
		strLightProperty << "PointLight[" << i << "].quadratic_attenuation";
		glUniform1f(glGetUniformLocation(ShaderObject, strLightProperty.str().c_str()), pLamp[i].GetQuadraticAttenuation());
	}

	// send spotlight data
	glUniform3fv(glGetUniformLocation(ShaderObject, "Torch.ambient"), 1, glm::value_ptr(pTorch->GetAmbient()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "Torch.diffuse"), 1, glm::value_ptr(pTorch->GetDiffuse()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "Torch.specular"), 1, glm::value_ptr(pTorch->GetSpecular()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "Torch.position"), 1, glm::value_ptr(pTorch->GetPosition()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "Torch.direction"), 1, glm::value_ptr(pTorch->GetDirection()));

	glUniform1f(glGetUniformLocation(ShaderObject, "Torch.constant_attenuation"), pTorch->GetConstantAttenuation());
	glUniform1f(glGetUniformLocation(ShaderObject, "Torch.linear_attenuation"), pTorch->GetLinearAttenuation());
	glUniform1f(glGetUniformLocation(ShaderObject, "Torch.quadratic_attenuation"), pTorch->GetQuadraticAttenuation());

	glUniform1f(glGetUniformLocation(ShaderObject, "Torch.inner_cutoff"), glm::cos(glm::radians(pTorch->GetInnerSpotCutOff())));
	glUniform1f(glGetUniformLocation(ShaderObject, "Torch.outer_cutoff"), glm::cos(glm::radians(pTorch->GetOuterSpotCutOff())));

	glUniform3fv(glGetUniformLocation(ShaderObject, "u_view_position"), 1, glm::value_ptr(CameraPosition));
}


void InitOpenALWithAllSources(void)
{
	//Code
	if (!InitializeAL())
	{
		Uninitialize();
		ExitProcess(EXIT_FAILURE);
	}

	InitializeAL_Listener();

	if (!InitializeAL_WalkingSource())
	{
		Uninitialize();
		ExitProcess(EXIT_FAILURE);
	}

	if (!InitializeAL_RunningSource())
	{
		Uninitialize();
		ExitProcess(EXIT_FAILURE);
	}
	if (!InitializeAL_BackgroundSource())
	{
		Uninitialize();
		ExitProcess(EXIT_FAILURE);
	}
	if (!InitializeAL_AmbienceSource())
	{
		Uninitialize();
		ExitProcess(EXIT_FAILURE);
	}

	if (!InitializeAL_ObjectPushed())
	{
		Uninitialize();
		ExitProcess(EXIT_FAILURE);
	}
	
	if (!InitializeAL_FinalWinSource())
	{
		Uninitialize();
		ExitProcess(EXIT_FAILURE);
	}

	if (!InitializeAL_EndCredits())
	{
		Uninitialize();
		ExitProcess(EXIT_FAILURE);
	}
}

// Sujay and Prabhakar -- AMC Game
void AMC_Game(void)
{
	glm::vec3 currentCameraPosition = pCCamera->GetCameraPosition();
	glm::vec3 currentPlayerPosition = pCPlayer->GetPlayerPosition();

	if (gbUniversalLetterMoveFlag == FALSE)
	{
		// LetterA
		if (abs((currentCameraPosition.x - LetterA_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentCameraPosition.z - LetterA_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE || abs((currentPlayerPosition.x - LetterA_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentPlayerPosition.z - LetterA_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE)
		{
			//LetterA_Translation = LetterA_Final_Translation;
			gbLetterA_SetFlag = TRUE;
			PlayAL_ObjectPushed();
		}

		// LetterS
		if (abs((currentCameraPosition.x - LetterS_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentCameraPosition.z - LetterS_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE || abs((currentPlayerPosition.x - LetterS_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentPlayerPosition.z - LetterS_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE)
		{
		//	LetterS_Translation = LetterS_Final_Translation;
			gbLetterS_SetFlag = TRUE;
			PlayAL_ObjectPushed();
		}

		// LetterT
		if (abs((currentCameraPosition.x - LetterT_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentCameraPosition.z - LetterT_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE || abs((currentPlayerPosition.x - LetterT_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentPlayerPosition.z - LetterT_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE)
		{
		//	LetterT_Translation = LetterT_Final_Translation;
			gbLetterT_SetFlag = TRUE;
			PlayAL_ObjectPushed();
		}

		// LetterR
		if (abs((currentCameraPosition.x - LetterR_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentCameraPosition.z - LetterR_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE || abs((currentPlayerPosition.x - LetterR_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentPlayerPosition.z - LetterR_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE)
		{
		//	LetterR_Translation = LetterR_Final_Translation;
			gbLetterR_SetFlag = TRUE;
			PlayAL_ObjectPushed();
		}

		// LetterO
		if (abs((currentCameraPosition.x - LetterO_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentCameraPosition.z - LetterO_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE || abs((currentPlayerPosition.x - LetterO_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentPlayerPosition.z - LetterO_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE)
		{
		//	LetterO_Translation = LetterO_Final_Translation;
			gbLetterO_SetFlag = TRUE;
			PlayAL_ObjectPushed();
		}

		// LetterM
		if (abs((currentCameraPosition.x - LetterM_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentCameraPosition.z - LetterM_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE || abs((currentPlayerPosition.x - LetterM_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentPlayerPosition.z - LetterM_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE)
		{
		//	LetterM_Translation = LetterM_Final_Translation;
			gbLetterM_SetFlag = TRUE;
			PlayAL_ObjectPushed();
		}

		// LetterE
		if (abs((currentCameraPosition.x - LetterE_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentCameraPosition.z - LetterE_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE || abs((currentPlayerPosition.x - LetterE_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentPlayerPosition.z - LetterE_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE)
		{
		//	LetterE_Translation = LetterE_Final_Translation;
			gbLetterE_SetFlag = TRUE;
			PlayAL_ObjectPushed();
		}

		// LetterD
		if (abs((currentCameraPosition.x - LetterD_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentCameraPosition.z - LetterD_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE || abs((currentPlayerPosition.x - LetterD_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentPlayerPosition.z - LetterD_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE)
		{
		//	LetterD_Translation = LetterD_Final_Translation;
			gbLetterD_SetFlag = TRUE;
			PlayAL_ObjectPushed();
		}

		// LetterI
		if (abs((currentCameraPosition.x - LetterI_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentCameraPosition.z - LetterI_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE || abs((currentPlayerPosition.x - LetterI_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentPlayerPosition.z - LetterI_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE)
		{
		//	LetterI_Translation = LetterI_Final_Translation;
			gbLetterI_SetFlag = TRUE;
			PlayAL_ObjectPushed();
		}

		// LetterC
		if (abs((currentCameraPosition.x - LetterC_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentCameraPosition.z - LetterC_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE || abs((currentPlayerPosition.x - LetterC_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentPlayerPosition.z - LetterC_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE)
		{
		//	LetterC_Translation = LetterC_Final_Translation;
			gbLetterC_SetFlag = TRUE;
			PlayAL_ObjectPushed();
		}

		// LetterO_2
		if (abs((currentCameraPosition.x - LetterO_2_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentCameraPosition.z - LetterO_2_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE || abs((currentPlayerPosition.x - LetterO_2_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentPlayerPosition.z - LetterO_2_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE)
		{
		//	LetterO_2_Translation = LetterO_2_Final_Translation;
			gbLetterO_2_SetFlag = TRUE;
			PlayAL_ObjectPushed();
		}

		// LetterM_2
		if (abs((currentCameraPosition.x - LetterM_2_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentCameraPosition.z - LetterM_2_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE || abs((currentPlayerPosition.x - LetterM_2_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentPlayerPosition.z - LetterM_2_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE)
		{
		//	LetterM_2_Translation = LetterM_2_Final_Translation;
			gbLetterM_2_SetFlag = TRUE;
			PlayAL_ObjectPushed();
		}

		// LetterP
		if (abs((currentCameraPosition.x - LetterP_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentCameraPosition.z - LetterP_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE || abs((currentPlayerPosition.x - LetterP_Translation.x)) <= LETTER_PICKING_MIN_DISTANCE && abs((currentPlayerPosition.z - LetterP_Translation.z)) <= LETTER_PICKING_MIN_DISTANCE)
		{
		//	LetterP_Translation = LetterP_Final_Translation;
			gbLetterP_SetFlag = TRUE;
			PlayAL_ObjectPushed();
		}

		if (gbLetterA_SetFlag == TRUE && gbLetterS_SetFlag == TRUE && gbLetterT_SetFlag == TRUE && gbLetterR_SetFlag == TRUE && gbLetterO_SetFlag == TRUE &&
			gbLetterM_SetFlag == TRUE && gbLetterE_SetFlag == TRUE && gbLetterD_SetFlag == TRUE && gbLetterI_SetFlag == TRUE &&
			gbLetterC_SetFlag == TRUE && gbLetterO_2_SetFlag == TRUE && gbLetterM_2_SetFlag == TRUE && gbLetterP_SetFlag == TRUE)
		{
			gbUniversalLetterMoveFlag = TRUE;
		}
	}
	else
	{
	/*	LetterA_Translation = LetterA_Final_Translation;
		LetterS_Translation = LetterS_Final_Translation;
		LetterT_Translation = LetterT_Final_Translation;
		LetterR_Translation = LetterR_Final_Translation;
		LetterO_Translation = LetterO_Final_Translation;
		LetterM_Translation = LetterM_Final_Translation;
		LetterE_Translation = LetterE_Final_Translation;
		LetterD_Translation = LetterD_Final_Translation;
		LetterI_Translation = LetterI_Final_Translation;
		LetterC_Translation = LetterC_Final_Translation;
		LetterO_2_Translation = LetterO_2_Final_Translation;
		LetterM_2_Translation = LetterM_2_Final_Translation;
		LetterP_Translation = LetterP_Final_Translation;*/

		gbLetterA_SetFlag = TRUE;
		gbLetterS_SetFlag = TRUE;
		gbLetterT_SetFlag = TRUE;
		gbLetterR_SetFlag = TRUE;
		gbLetterO_SetFlag = TRUE;
		gbLetterM_SetFlag = TRUE;
		gbLetterE_SetFlag = TRUE;
		gbLetterD_SetFlag = TRUE;
		gbLetterI_SetFlag = TRUE;
		gbLetterC_SetFlag = TRUE;
		gbLetterO_2_SetFlag = TRUE;
		gbLetterM_2_SetFlag = TRUE;
		gbLetterP_SetFlag = TRUE;

	}
}

void DrawLettersObj(glm::mat4 ViewMatrix, glm::vec4 clipPlane)
{
	// For AstroMediComp
	//Load Model A

	gpLetterA_obj->UseProgram();

	glm::mat4 ModelMatrix = glm::mat4(1.0f);
	glm::mat4 TranslationMatrix = glm::mat4(1.0f);
	glm::mat4 RotationMatrix = glm::mat4(1.0f);
	glm::mat4 ScaleMatrix = glm::mat4(1.0f);

	glUniformMatrix4fv(glGetUniformLocation(gpLetterA_obj->GetProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gpLetterA_obj->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniform4fv(glGetUniformLocation(gpLetterA_obj->GetProgram(), "u_model_color"), 1, glm::value_ptr(BlueColor));
	
	glUniform4f(glGetUniformLocation(gpLetterA_obj->GetProgram(), "clipPlane"), clipPlane.x, clipPlane.y, clipPlane.z, clipPlane.w);

	GLfloat heightAtLocation = pCTerrain->GetHeightAt(LetterA_Translation);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterA_Translation.x, LetterA_Translation.y + heightAtLocation, LetterA_Translation.z));
	if (gbLetterA_SetFlag == FALSE)
	{
		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(gfLetterRotationAngle), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	else
	{
		//	GLfloat heightAtLocation = pCTerrain->GetHeightAt(LetterA_Final_Translation);
		//	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterA_Final_Translation.x, LetterA_Final_Translation.y, LetterA_Final_Translation.z));
		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	ModelMatrix = ModelMatrix * TranslationMatrix;
	ModelMatrix = ModelMatrix * RotationMatrix;
	ScaleMatrix = glm::scale(glm::mat4(1.0f), LetterScaleConstant);
	ModelMatrix = ModelMatrix * ScaleMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gpLetterA_obj->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	gpLetterA_obj->Draw(gfDeltaTime * 100);

	//Load Model S
	gpLetterS_obj->UseProgram();

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::mat4(1.0f);
	RotationMatrix = glm::mat4(1.0f);
	ScaleMatrix = glm::mat4(1.0f);

	glUniformMatrix4fv(glGetUniformLocation(gpLetterS_obj->GetProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gpLetterS_obj->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniform4fv(glGetUniformLocation(gpLetterS_obj->GetProgram(), "u_model_color"), 1, glm::value_ptr(BlueColor));
	heightAtLocation = pCTerrain->GetHeightAt(LetterS_Translation);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterS_Translation.x, LetterS_Translation.y + heightAtLocation, LetterS_Translation.z));
	if (gbLetterS_SetFlag == FALSE)
	{
		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(gfLetterRotationAngle), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	else
	{
		//	GLfloat heightAtLocation = pCTerrain->GetHeightAt(LetterS_Final_Translation);
		//	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterS_Final_Translation.x, LetterS_Final_Translation.y, LetterS_Final_Translation.z));

		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	ModelMatrix = ModelMatrix * TranslationMatrix;
	ModelMatrix = ModelMatrix * RotationMatrix;
	ScaleMatrix = glm::scale(glm::mat4(1.0f), LetterScaleConstant);
	ModelMatrix = ModelMatrix * ScaleMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gpLetterS_obj->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	gpLetterS_obj->Draw(gfDeltaTime * 100);

	//Load Model T
	gpLetterT_obj->UseProgram();

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::mat4(1.0f);
	RotationMatrix = glm::mat4(1.0f);
	ScaleMatrix = glm::mat4(1.0f);

	glUniformMatrix4fv(glGetUniformLocation(gpLetterT_obj->GetProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gpLetterT_obj->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniform4fv(glGetUniformLocation(gpLetterT_obj->GetProgram(), "u_model_color"), 1, glm::value_ptr(BlueColor));
	heightAtLocation = pCTerrain->GetHeightAt(LetterT_Translation);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterT_Translation.x, LetterT_Translation.y + heightAtLocation, LetterT_Translation.z));
	if (gbLetterT_SetFlag == FALSE)
	{
		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(gfLetterRotationAngle), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	else
	{
		//	heightAtLocation = pCTerrain->GetHeightAt(LetterT_Final_Translation);
		//	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterT_Final_Translation.x, LetterT_Final_Translation.y, LetterT_Final_Translation.z));

		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	ModelMatrix = ModelMatrix * TranslationMatrix;
	ModelMatrix = ModelMatrix * RotationMatrix;
	ScaleMatrix = glm::scale(glm::mat4(1.0f), LetterScaleConstant);
	ModelMatrix = ModelMatrix * ScaleMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gpLetterT_obj->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	gpLetterT_obj->Draw(gfDeltaTime * 100);

	//Load Model R
	gpLetterR_obj->UseProgram();

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::mat4(1.0f);
	RotationMatrix = glm::mat4(1.0f);
	ScaleMatrix = glm::mat4(1.0f);

	glUniformMatrix4fv(glGetUniformLocation(gpLetterR_obj->GetProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gpLetterR_obj->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniform4fv(glGetUniformLocation(gpLetterR_obj->GetProgram(), "u_model_color"), 1, glm::value_ptr(BlueColor));
	heightAtLocation = pCTerrain->GetHeightAt(LetterR_Translation);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterR_Translation.x, LetterR_Translation.y + heightAtLocation, LetterR_Translation.z));
	if (gbLetterR_SetFlag == FALSE)
	{
		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(gfLetterRotationAngle), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	else
	{
		//	heightAtLocation = pCTerrain->GetHeightAt(LetterR_Final_Translation);
		//	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterR_Final_Translation.x, LetterR_Final_Translation.y, LetterR_Final_Translation.z));

		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	ModelMatrix = ModelMatrix * TranslationMatrix;
	ModelMatrix = ModelMatrix * RotationMatrix;
	ScaleMatrix = glm::scale(glm::mat4(1.0f), LetterScaleConstant);
	ModelMatrix = ModelMatrix * ScaleMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gpLetterR_obj->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	gpLetterR_obj->Draw(gfDeltaTime * 100);

	//Load Model O
	gpLetterO_obj->UseProgram();

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::mat4(1.0f);
	RotationMatrix = glm::mat4(1.0f);
	ScaleMatrix = glm::mat4(1.0f);

	glUniformMatrix4fv(glGetUniformLocation(gpLetterO_obj->GetProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gpLetterO_obj->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniform4fv(glGetUniformLocation(gpLetterO_obj->GetProgram(), "u_model_color"), 1, glm::value_ptr(BlueColor));
	heightAtLocation = pCTerrain->GetHeightAt(LetterO_Translation);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterO_Translation.x, LetterO_Translation.y + heightAtLocation, LetterO_Translation.z));
	if (gbLetterO_SetFlag == FALSE)
	{
		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(gfLetterRotationAngle), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	else
	{
		//	heightAtLocation = pCTerrain->GetHeightAt(LetterO_Final_Translation);
		//	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterO_Final_Translation.x, LetterO_Final_Translation.y, LetterO_Final_Translation.z));

		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	ModelMatrix = ModelMatrix * TranslationMatrix;
	ModelMatrix = ModelMatrix * RotationMatrix;
	ScaleMatrix = glm::scale(glm::mat4(1.0f), LetterScaleConstant);
	ModelMatrix = ModelMatrix * ScaleMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gpLetterO_obj->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	gpLetterO_obj->Draw(gfDeltaTime * 100);

	//Load Model M
	gpLetterM_obj->UseProgram();

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::mat4(1.0f);
	RotationMatrix = glm::mat4(1.0f);
	ScaleMatrix = glm::mat4(1.0f);

	glUniformMatrix4fv(glGetUniformLocation(gpLetterM_obj->GetProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gpLetterM_obj->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniform4fv(glGetUniformLocation(gpLetterM_obj->GetProgram(), "u_model_color"), 1, glm::value_ptr(RedColor));
	heightAtLocation = pCTerrain->GetHeightAt(LetterM_Translation);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterM_Translation.x, LetterM_Translation.y + heightAtLocation, LetterM_Translation.z));
	if (gbLetterM_SetFlag == FALSE)
	{
		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(gfLetterRotationAngle), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	else
	{
		//	heightAtLocation = pCTerrain->GetHeightAt(LetterM_Final_Translation);
		//	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterM_Final_Translation.x, LetterM_Final_Translation.y, LetterM_Final_Translation.z));

		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	ModelMatrix = ModelMatrix * TranslationMatrix;
	ModelMatrix = ModelMatrix * RotationMatrix;
	ScaleMatrix = glm::scale(glm::mat4(1.0f), LetterScaleConstant);
	ModelMatrix = ModelMatrix * ScaleMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gpLetterM_obj->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	gpLetterM_obj->Draw(gfDeltaTime * 100);

	//Load Model E
	gpLetterE_obj->UseProgram();

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::mat4(1.0f);
	RotationMatrix = glm::mat4(1.0f);
	ScaleMatrix = glm::mat4(1.0f);

	glUniformMatrix4fv(glGetUniformLocation(gpLetterE_obj->GetProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gpLetterE_obj->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniform4fv(glGetUniformLocation(gpLetterE_obj->GetProgram(), "u_model_color"), 1, glm::value_ptr(RedColor));
	heightAtLocation = pCTerrain->GetHeightAt(LetterE_Translation);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterE_Translation.x, LetterE_Translation.y + heightAtLocation, LetterE_Translation.z));
	if (gbLetterE_SetFlag == FALSE)
	{
		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(gfLetterRotationAngle), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	else
	{
		//	heightAtLocation = pCTerrain->GetHeightAt(LetterE_Final_Translation);
		//	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterE_Final_Translation.x, LetterE_Final_Translation.y, LetterE_Final_Translation.z));

		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	ModelMatrix = ModelMatrix * TranslationMatrix;
	ModelMatrix = ModelMatrix * RotationMatrix;
	ScaleMatrix = glm::scale(glm::mat4(1.0f), LetterScaleConstant);
	ModelMatrix = ModelMatrix * ScaleMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gpLetterE_obj->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	gpLetterE_obj->Draw(gfDeltaTime * 100);

	//Load Model D
	gpLetterD_obj->UseProgram();

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::mat4(1.0f);
	RotationMatrix = glm::mat4(1.0f);
	ScaleMatrix = glm::mat4(1.0f);

	glUniformMatrix4fv(glGetUniformLocation(gpLetterD_obj->GetProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gpLetterD_obj->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniform4fv(glGetUniformLocation(gpLetterD_obj->GetProgram(), "u_model_color"), 1, glm::value_ptr(RedColor));
	heightAtLocation = pCTerrain->GetHeightAt(LetterD_Translation);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterD_Translation.x, LetterD_Translation.y + heightAtLocation, LetterD_Translation.z));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	if (gbLetterD_SetFlag == FALSE)
	{

		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(gfLetterRotationAngle), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	else
	{
		//	heightAtLocation = pCTerrain->GetHeightAt(LetterD_Final_Translation);
		//	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterD_Final_Translation.x, LetterD_Final_Translation.y, LetterD_Final_Translation.z));

		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	}

	ModelMatrix = ModelMatrix * RotationMatrix;
	ScaleMatrix = glm::scale(glm::mat4(1.0f), LetterScaleConstant);
	ModelMatrix = ModelMatrix * ScaleMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gpLetterD_obj->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	gpLetterD_obj->Draw(gfDeltaTime * 100);

	//Load Model I
	gpLetterI_obj->UseProgram();

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::mat4(1.0f);
	RotationMatrix = glm::mat4(1.0f);
	ScaleMatrix = glm::mat4(1.0f);

	glUniformMatrix4fv(glGetUniformLocation(gpLetterI_obj->GetProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gpLetterI_obj->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniform4fv(glGetUniformLocation(gpLetterI_obj->GetProgram(), "u_model_color"), 1, glm::value_ptr(RedColor));
	heightAtLocation = pCTerrain->GetHeightAt(LetterI_Translation);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterI_Translation.x, LetterI_Translation.y + heightAtLocation, LetterI_Translation.z));
	if (gbLetterI_SetFlag == FALSE)
	{
		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(gfLetterRotationAngle), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	else
	{
		//	heightAtLocation = pCTerrain->GetHeightAt(LetterI_Final_Translation);
		//	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterI_Final_Translation.x, LetterI_Final_Translation.y, LetterI_Final_Translation.z));

		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	ModelMatrix = ModelMatrix * TranslationMatrix;
	ModelMatrix = ModelMatrix * RotationMatrix;
	ScaleMatrix = glm::scale(glm::mat4(1.0f), LetterScaleConstant);
	ModelMatrix = ModelMatrix * ScaleMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gpLetterI_obj->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	gpLetterI_obj->Draw(gfDeltaTime * 100);

	//Load Model C
	gpLetterC_obj->UseProgram();

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::mat4(1.0f);
	RotationMatrix = glm::mat4(1.0f);
	ScaleMatrix = glm::mat4(1.0f);

	glUniformMatrix4fv(glGetUniformLocation(gpLetterC_obj->GetProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gpLetterC_obj->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniform4fv(glGetUniformLocation(gpLetterC_obj->GetProgram(), "u_model_color"), 1, glm::value_ptr(GreenColor));
	heightAtLocation = pCTerrain->GetHeightAt(LetterC_Translation);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterC_Translation.x, LetterC_Translation.y + heightAtLocation, LetterC_Translation.z));
	if (gbLetterC_SetFlag == FALSE)
	{
		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(gfLetterRotationAngle), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	else
	{
		//	heightAtLocation = pCTerrain->GetHeightAt(LetterC_Final_Translation);
		//	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterC_Final_Translation.x, LetterC_Final_Translation.y, LetterC_Final_Translation.z));

		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	ModelMatrix = ModelMatrix * TranslationMatrix;
	ModelMatrix = ModelMatrix * RotationMatrix;
	ScaleMatrix = glm::scale(glm::mat4(1.0f), LetterScaleConstant);
	ModelMatrix = ModelMatrix * ScaleMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gpLetterC_obj->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	gpLetterC_obj->Draw(gfDeltaTime * 100);

	//Load Model O -2
	gpLetterO_2_obj->UseProgram();

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::mat4(1.0f);
	RotationMatrix = glm::mat4(1.0f);
	ScaleMatrix = glm::mat4(1.0f);

	glUniformMatrix4fv(glGetUniformLocation(gpLetterO_2_obj->GetProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gpLetterO_2_obj->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniform4fv(glGetUniformLocation(gpLetterO_2_obj->GetProgram(), "u_model_color"), 1, glm::value_ptr(GreenColor));
	heightAtLocation = pCTerrain->GetHeightAt(LetterO_2_Translation);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterO_2_Translation.x, LetterO_2_Translation.y + heightAtLocation, LetterO_2_Translation.z));
	if (gbLetterO_2_SetFlag == FALSE)
	{
		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(gfLetterRotationAngle), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	else
	{
		//	heightAtLocation = pCTerrain->GetHeightAt(LetterO_2_Final_Translation);
		//	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterO_2_Final_Translation.x, LetterO_2_Final_Translation.y, LetterO_2_Final_Translation.z));

		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	ModelMatrix = ModelMatrix * TranslationMatrix;
	ModelMatrix = ModelMatrix * RotationMatrix;
	ScaleMatrix = glm::scale(glm::mat4(1.0f), LetterScaleConstant);
	ModelMatrix = ModelMatrix * ScaleMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gpLetterO_2_obj->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	gpLetterO_2_obj->Draw(gfDeltaTime * 100);

	//Load Model M -2
	gpLetterM_2_obj->UseProgram();

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::mat4(1.0f);
	RotationMatrix = glm::mat4(1.0f);
	ScaleMatrix = glm::mat4(1.0f);

	glUniformMatrix4fv(glGetUniformLocation(gpLetterM_2_obj->GetProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gpLetterM_2_obj->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniform4fv(glGetUniformLocation(gpLetterM_2_obj->GetProgram(), "u_model_color"), 1, glm::value_ptr(GreenColor));
	heightAtLocation = pCTerrain->GetHeightAt(LetterM_2_Translation);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterM_2_Translation.x, LetterM_2_Translation.y + heightAtLocation, LetterM_2_Translation.z));
	if (gbLetterM_2_SetFlag == FALSE)
	{
		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(gfLetterRotationAngle), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	else
	{
		//	GLfloat heightAtLocation = pCTerrain->GetHeightAt(LetterM_2_Final_Translation);
		//	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterM_2_Final_Translation.x, LetterM_2_Final_Translation.y, LetterM_2_Final_Translation.z));

		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	ModelMatrix = ModelMatrix * TranslationMatrix;
	ModelMatrix = ModelMatrix * RotationMatrix;
	ScaleMatrix = glm::scale(glm::mat4(1.0f), LetterScaleConstant);
	ModelMatrix = ModelMatrix * ScaleMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gpLetterM_2_obj->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	gpLetterM_2_obj->Draw(gfDeltaTime * 100);

	//Load Model P
	gpLetterP_obj->UseProgram();

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::mat4(1.0f);
	RotationMatrix = glm::mat4(1.0f);
	ScaleMatrix = glm::mat4(1.0f);

	glUniformMatrix4fv(glGetUniformLocation(gpLetterP_obj->GetProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gpLetterP_obj->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniform4fv(glGetUniformLocation(gpLetterP_obj->GetProgram(), "u_model_color"), 1, glm::value_ptr(GreenColor));
	heightAtLocation = pCTerrain->GetHeightAt(LetterP_Translation);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterP_Translation.x, LetterP_Translation.y + heightAtLocation, LetterP_Translation.z));
	if (gbLetterP_SetFlag == FALSE)
	{
		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(gfLetterRotationAngle), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	else
	{
		//	GLfloat heightAtLocation = pCTerrain->GetHeightAt(LetterP_Final_Translation);
		//	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterP_Final_Translation.x, LetterP_Final_Translation.y, LetterP_Final_Translation.z));

		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	ModelMatrix = ModelMatrix * TranslationMatrix;
	ModelMatrix = ModelMatrix * RotationMatrix;
	ScaleMatrix = glm::scale(glm::mat4(1.0f), LetterScaleConstant);
	ModelMatrix = ModelMatrix * ScaleMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gpLetterP_obj->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	gpLetterP_obj->Draw(gfDeltaTime * 100);

	// For Raster
	//Load Model R
	gpLetterR_RG_obj->UseProgram();

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::mat4(1.0f);
	RotationMatrix = glm::mat4(1.0f);
	ScaleMatrix = glm::mat4(1.0f);

	glUniformMatrix4fv(glGetUniformLocation(gpLetterR_RG_obj->GetProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gpLetterR_RG_obj->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniform4fv(glGetUniformLocation(gpLetterA_obj->GetProgram(), "u_model_color"), 1, glm::value_ptr(RedColor));
	//	heightAtLocation = pCTerrain->GetHeightAt(LetterR_RG_Translation);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterR_RG_Translation.x, LetterR_RG_Translation.y, LetterR_RG_Translation.z));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	if (gbUniversalLetterMoveFlag == TRUE)
	{
		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(gfRasterRotateAngle), glm::vec3(1.0f, 0.0f, 0.0f));
		ModelMatrix = ModelMatrix * RotationMatrix;
		TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 17.0f, 0.0f));
		ModelMatrix = ModelMatrix * TranslationMatrix;
	}
	ScaleMatrix = glm::scale(glm::mat4(1.0f), LetterScaleConstant);
	ModelMatrix = ModelMatrix * ScaleMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gpLetterR_RG_obj->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	gpLetterR_RG_obj->Draw(gfDeltaTime * 100);

	//Load Model A
	gpLetterA_RG_obj->UseProgram();

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::mat4(1.0f);
	RotationMatrix = glm::mat4(1.0f);
	ScaleMatrix = glm::mat4(1.0f);

	glUniformMatrix4fv(glGetUniformLocation(gpLetterA_RG_obj->GetProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gpLetterA_RG_obj->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniform4fv(glGetUniformLocation(gpLetterA_obj->GetProgram(), "u_model_color"), 1, glm::value_ptr(GreenColor));
	//	heightAtLocation = pCTerrain->GetHeightAt(LetterA_RG_Translation);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterA_RG_Translation.x, LetterA_RG_Translation.y, LetterA_RG_Translation.z));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	if (gbUniversalLetterMoveFlag == TRUE)
	{
		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(gfRasterRotateAngle), glm::vec3(1.0f, 0.0f, 0.0f));
		ModelMatrix = ModelMatrix * RotationMatrix;
		TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 17.0f, 0.0f));
		ModelMatrix = ModelMatrix * TranslationMatrix;
	}
	ScaleMatrix = glm::scale(glm::mat4(1.0f), LetterScaleConstant);
	ModelMatrix = ModelMatrix * ScaleMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gpLetterA_RG_obj->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	gpLetterA_RG_obj->Draw(gfDeltaTime * 100);

	//Load Model S
	gpLetterS_RG_obj->UseProgram();

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::mat4(1.0f);
	RotationMatrix = glm::mat4(1.0f);
	ScaleMatrix = glm::mat4(1.0f);

	glUniformMatrix4fv(glGetUniformLocation(gpLetterS_RG_obj->GetProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gpLetterS_RG_obj->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniform4fv(glGetUniformLocation(gpLetterA_obj->GetProgram(), "u_model_color"), 1, glm::value_ptr(BlueColor));
	//	heightAtLocation = pCTerrain->GetHeightAt(LetterS_RG_Translation);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterS_RG_Translation.x, LetterS_RG_Translation.y, LetterS_RG_Translation.z));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	if (gbUniversalLetterMoveFlag == TRUE)
	{
		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(gfRasterRotateAngle), glm::vec3(1.0f, 0.0f, 0.0f));
		ModelMatrix = ModelMatrix * RotationMatrix;
		TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 17.0f, 0.0f));
		ModelMatrix = ModelMatrix * TranslationMatrix;
	}
	ScaleMatrix = glm::scale(glm::mat4(1.0f), LetterScaleConstant);
	ModelMatrix = ModelMatrix * ScaleMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gpLetterS_RG_obj->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	gpLetterS_RG_obj->Draw(gfDeltaTime * 100);

	//Load Model T
	gpLetterT_RG_obj->UseProgram();

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::mat4(1.0f);
	RotationMatrix = glm::mat4(1.0f);
	ScaleMatrix = glm::mat4(1.0f);

	glUniformMatrix4fv(glGetUniformLocation(gpLetterT_RG_obj->GetProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gpLetterT_RG_obj->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniform4fv(glGetUniformLocation(gpLetterA_obj->GetProgram(), "u_model_color"), 1, glm::value_ptr(CyanColor));
	//	heightAtLocation = pCTerrain->GetHeightAt(LetterT_RG_Translation);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterT_RG_Translation.x, LetterT_RG_Translation.y, LetterT_RG_Translation.z));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	if (gbUniversalLetterMoveFlag == TRUE)
	{
		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(gfRasterRotateAngle), glm::vec3(1.0f, 0.0f, 0.0f));
		ModelMatrix = ModelMatrix * RotationMatrix;
		TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 17.0f, 0.0f));
		ModelMatrix = ModelMatrix * TranslationMatrix;
	}
	ScaleMatrix = glm::scale(glm::mat4(1.0f), LetterScaleConstant);
	ModelMatrix = ModelMatrix * ScaleMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gpLetterT_RG_obj->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	gpLetterT_RG_obj->Draw(gfDeltaTime * 100);

	//Load Model E
	gpLetterE_RG_obj->UseProgram();

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::mat4(1.0f);
	RotationMatrix = glm::mat4(1.0f);
	ScaleMatrix = glm::mat4(1.0f);

	glUniformMatrix4fv(glGetUniformLocation(gpLetterE_RG_obj->GetProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gpLetterE_RG_obj->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniform4fv(glGetUniformLocation(gpLetterA_obj->GetProgram(), "u_model_color"), 1, glm::value_ptr(MagentaColor));
	//	heightAtLocation = pCTerrain->GetHeightAt(LetterE_RG_Translation);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterE_RG_Translation.x, LetterE_RG_Translation.y, LetterE_RG_Translation.z));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	if (gbUniversalLetterMoveFlag == TRUE)
	{
		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(gfRasterRotateAngle), glm::vec3(1.0f, 0.0f, 0.0f));
		ModelMatrix = ModelMatrix * RotationMatrix;
		TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 17.0f, 0.0f));
		ModelMatrix = ModelMatrix * TranslationMatrix;
	}
	ScaleMatrix = glm::scale(glm::mat4(1.0f), LetterScaleConstant);
	ModelMatrix = ModelMatrix * ScaleMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gpLetterE_RG_obj->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	gpLetterE_RG_obj->Draw(gfDeltaTime * 100);

	//Load Model R_2
	gpLetterR_RG2_obj->UseProgram();

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::mat4(1.0f);
	RotationMatrix = glm::mat4(1.0f);
	ScaleMatrix = glm::mat4(1.0f);

	glUniformMatrix4fv(glGetUniformLocation(gpLetterR_RG2_obj->GetProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gpLetterR_RG2_obj->GetProgram(), "view"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniform4fv(glGetUniformLocation(gpLetterA_obj->GetProgram(), "u_model_color"), 1, glm::value_ptr(YellowColor));
	//	heightAtLocation = pCTerrain->GetHeightAt(LetterR_RG2_Translation);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(LetterR_RG2_Translation.x, LetterR_RG2_Translation.y, LetterR_RG2_Translation.z));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	if (gbUniversalLetterMoveFlag == TRUE)
	{
		RotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(gfRasterRotateAngle), glm::vec3(1.0f, 0.0f, 0.0f));
		ModelMatrix = ModelMatrix * RotationMatrix;
		TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 17.0f, 0.0f));
		ModelMatrix = ModelMatrix * TranslationMatrix;
	}
	ScaleMatrix = glm::scale(glm::mat4(1.0f), LetterScaleConstant);
	ModelMatrix = ModelMatrix * ScaleMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gpLetterR_RG2_obj->GetProgram(), "model"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	gpLetterR_RG2_obj->Draw(gfDeltaTime * 100);
	/////////////////////////
}