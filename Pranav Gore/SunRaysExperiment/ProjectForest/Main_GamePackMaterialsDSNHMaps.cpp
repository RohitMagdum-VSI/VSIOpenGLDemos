#include <windows.h>
#include <windowsx.h>

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "OpenGL32.lib")

#include <stdio.h>
#include <gl\glew.h>
#include <gl\wglew.h>
#include <gl\GL.h>
#include <vector>
#include <string>
#include <sstream>


//#define GLM_FORCE_CUDA
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

#include "Skydome/Skydome.h"

#include "Terrain/Terrain.h"

#include "Texture/Texture.h"
#include "Texture/Tga.h"

#include "BasicShapes/BasicShapes.h"


bool gbEscapeKeyIsPressed = false;
bool gbIsFullscreen = false;
bool gbActiveWindow = false;

GLboolean gbIsFirstPlay = GL_FALSE;

void ToggleFullscreen(void);


//BOOL gbDone = FALSE;

HWND ghWnd = NULL;
HDC ghDC = NULL;
HGLRC ghRC = NULL;

POINT ClickPoint, MovePoint, OldMovePoint;
GLboolean gbFirstMouse = GL_TRUE;

CCommon *pCCommon = NULL; // Cleaned in Uninitialize()
CCamera *pCCamera = NULL; // no heap memory allocation still cleaned up
CPlayer *pCPlayer = NULL; // cleaned up
CSkyDome *pCSkyDome = NULL; // cleaned up

CTerrain *pCTerrain = NULL; // cleaned up
TGALoader *pTGA = NULL; // cleaned up

CDirectionalLight *pSunLight = NULL; // cleaned up

////////////////////////////////////////////////////////
// basic shapes
CRing *pCRing[] = { NULL, NULL };
CSphere *pCSphere[] = { NULL, NULL };
CTorus *pCTorus[] = { NULL, NULL };
CCylinder *pCCylinder[] = { NULL, NULL };
CCone *pCCone[] = { NULL, NULL };

// texture IDs for basic shapes
GLuint BrickDiffuse, BrickSpecular, BrickNormal, BrickNormalInv;
GLuint BrickDiffuseGranite, BrickDiffuseGrey, BrickDiffuseOldRed, BrickSpecular2, BrickNormal2, BrickNormalInv2;
GLuint GraniteDiffuse, GraniteSpecular, GraniteNormal, GraniteNormalInv;
GLuint PavingDiffuse, PavingSpecular, PavingNormal, PavingNormalInv;
GLuint PlasterDiffuseGreen, PlasterDiffuseWhite, PlasterDiffuseYellow, PlasterSpecular, PlasterNormal, PlasterNormalInv;
GLuint WoodPlatesDiffuse, WoodPlatesSpecular, WoodNormal, WoodNormalInv;

void DrawAllBasicShapesInOneRowWithNormalMap(GLuint DiffuseTexture, GLuint SpecularTexture, GLuint NormalTexture, glm::mat4 ModelMatrix, glm::mat4 TranslationMatrix, glm::mat4 RotationMatrix, glm::mat4 ViewMatrix);
void SetDayLightNormalMap(GLuint ShaderObject);

/////////////////////////////////////////////////////////

glm::vec3 Positions[10] = {
	glm::vec3(-5.0f, 0.0f, -5.0f),
	glm::vec3(5.0f, 0.0f, -5.0f),

	glm::vec3(-15.0f, 0.0f, -20.0f),
	glm::vec3(15.0f, 0.0f, -20.0f),

	glm::vec3(-15.0f, 15.0f, -15.0f),
	glm::vec3(15.0f, 15.0f, -15.0f),

	glm::vec3(-15.0f, -15.0f, -15.0f),
	glm::vec3(15.0f, -15.0f, -15.0f),

	glm::vec3(-15.0f, 15.0f, -20.0f),
	glm::vec3(15.0f, 15.0f, -20.0f),
};

//GLboolean IsMyShape = GL_TRUE;
GLboolean IsMyShape = GL_FALSE;
GLuint gTextureRingID;
GLuint gShaderProgramBasicShapes;
GLuint gShaderProgramObjectBasicShapesNormalMap;

glm::vec3 CameraPosition;
glm::vec3 CameraFront;
glm::vec3 CameraUp;
glm::vec3 CameraLook;
glm::vec3 CameraRight;

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

GLuint gShaderProgramObjectBasicShader; // cleaned up
GLuint gShaderProgramObjectBasicShaderTexture; // cleaned up


///////////////////////////////////////////////////////////////
// HDR - > all cleaned up
GLuint gShaderObjectHDR;
GLuint gFrameBufferObjectHDR;
GLuint gRenderBufferObjectDepth;
GLuint gColorBufferTexture;

GLuint gVAOHDRQuad;
GLuint gVBOHDR;

GLfloat fExposure = 1.0f;

void InitializeQuadHDR(void);
void RenderQuadHDR(void);

/////////////////////////////////////////////////////////

// sample texture for opacity map test -> all cleaned up
GLuint gVAOSampleQuad;
GLuint gVBOSampleQuad;

void InitializeSampleQuad(void);
void RenderSampleQuad(void);

GLuint DiffuseTexture;
GLuint OpacityTexture;

GLuint gShaderObjectSampleTexture;

/////////////////////////////////////////////////////

// MSAA FBO -> all cleaned up
GLuint gShaderObjectMSAA;
GLuint gFBOMSAA;
GLuint gRBOMSAA;
GLuint gTextureMSAA;

///////////////////////////////////////////////////////

// General -> all cleaned up
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

///////////////////////////////
// Aashish Sky Scattering
GLuint gShaderProgramObjectScatteringAashish;

GLuint gPI;
GLuint gJSteps;
GLuint gISteps;

glm::vec3 gSunPosition;
GLfloat Theta = 0.0f;

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

// Variables for OpenAudio
bool gbIsShiftKeyPressed = false;
bool gbRun = false;


///////////////////////////////////////
// set the lighting data

void SetDayLight(GLuint ShaderObject);


//////////////////////////////////////
// Volumetric Light
GLuint gShaderProgramObjectVolumeLight;

GLuint ModelViewMatrixUniform;
GLuint LightPositionUniform;
GLuint OcclusionTextureUniform;
GLuint SceneRenderUniform;

GLuint OcclusionFBO;
GLuint DepthFBO;

GLuint OcclusionMapTexture;
GLuint SceneRenderTexture;

/////////////////////////////////////////
// for Deffered rendering
GLuint GeometryBuffer;
GLuint PositionBufferTexture, NormalBufferTexture, AlbedoSpecularBufferTexture;
GLuint BufferAttchments[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
GLuint RBODepth;
GLuint ShaderProgramObjectDefferedRendering;
GLuint ShaderProgramObjectGBuffer;

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
	//ToggleFullscreen();

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

		swprintf_s(str, 255, TEXT("A Walk in the Fields FPS : %d"), fps__);
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
			break;

			// D key
		case 0x41:
			break;

			// E key to end the scene
		case 0x45:
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

			break;

		case 0x70: //F1

			break;

			// P key
		case 0x50:
			if (gbIsPlaying == GL_FALSE)
			{
				if (gbIsFirstPlay == GL_FALSE)
				{
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

			break;

		case VK_ADD:
			pCSkyDome->SetSunCPos(glm::rotate(pCSkyDome->GetSunCPos(), + 0.1f, pCSkyDome->GetSunRotVec()));
			break;

		case VK_SUBTRACT:
			pCSkyDome->SetSunCPos(glm::rotate(pCSkyDome->GetSunCPos(), - 0.1f, pCSkyDome->GetSunRotVec()));
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

	if (gbIsPlaying == GL_TRUE)
	{
		pCPlayer->ControlMouseInput();
		SetCursorPos((int)(screen.right * 0.5), (int)(screen.bottom * 0.5));
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

		if (WGLEW_EXT_swap_control)
		{
			wglSwapIntervalEXT(0);
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

	if (WGLEW_EXT_swap_control)
	{
		wglSwapIntervalEXT(0);
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

	pCCamera = new CCamera();
	pCTerrain = new CTerrain();
	pCPlayer = new CPlayer(glm::vec3(0.0f, pCTerrain->GetHeightAt(glm::vec3(0.0f, 0.0f, 0.0f)), 0.0f), pCTerrain);
	pCSkyDome = new CSkyDome();

	pTGA = new TGALoader();

	pCRing[0] = new CRing(2.5f, 5.0f, 100);
	pCRing[1] = new CRing(2.5f, 5.0f, 100);

	pCSphere[0] = new CSphere(5.0f, 100, 100);
	pCSphere[1] = new CSphere(5.0f, 100, 100);

	pCTorus[0] = new CTorus(2.5f, 5.0f, 100, 100);
	pCTorus[1] = new CTorus(2.5f, 5.0f, 100, 100);

	pCCylinder[0] = new CCylinder(5.0f, 7.0f, 100, GL_TRUE, GL_TRUE);
	pCCylinder[1] = new CCylinder(5.0f, 7.0f, 100, GL_TRUE, GL_TRUE);

	pCCone[0] = new CCone(5.0f, 7.0f, 100, GL_TRUE);
	pCCone[1] = new CCone(5.0f, 7.0f, 100, GL_TRUE);


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

	//ShaderInfo SampleTextureShader[] =
	//{
	//	{ GL_VERTEX_SHADER, "Resources/Shaders/SampleTexture.vert.glsl", 0 },
	//{ GL_FRAGMENT_SHADER, "Resources/Shaders/SampleTexture.frag.glsl", 0 },
	//{ GL_NONE, NULL, 0 }
	//};

	//gShaderObjectSampleTexture = LoadShaders(SampleTextureShader);

	//ShaderInfo SkyShader[] =
	//{
	//	{ GL_VERTEX_SHADER, "Resources/Shaders/AtmosphericScatterringGPUGems2.vert.glsl", 0 },
	//{ GL_FRAGMENT_SHADER, "Resources/Shaders/AtmosphericScatterringGPUGems2.frag.glsl", 0 },
	//{ GL_NONE, NULL, 0 }
	//};

	ShaderInfo SkyShader[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/AtmosphericScatterringPerFragment.vert.glsl", 0 },
	{ GL_FRAGMENT_SHADER, "Resources/Shaders/AtmosphericScatterringPerFragment.frag.glsl", 0 },
	{ GL_NONE, NULL, 0 }
	};

	ShaderInfo TerrainDay[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/TerrainIQFog1.vert.glsl", 0 },
	{ GL_FRAGMENT_SHADER, "Resources/Shaders/TerrainIQFog1.frag.glsl", 0 },
	{ GL_NONE, NULL, 0 }
	};

	ShaderInfo BasicShapes[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/BasicShapesIQFog1.vert.glsl", 0 },
	{ GL_FRAGMENT_SHADER, "Resources/Shaders/BasicShapesIQFog1.frag.glsl", 0 },
	{ GL_NONE, NULL, 0 }
	};

	gShaderProgramBasicShapes = LoadShaders(BasicShapes);

	ShaderInfo BasicShapesNM[] = 
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/BasicShapesIQFog1NM.vert.glsl", 0 },
	{ GL_FRAGMENT_SHADER, "Resources/Shaders/BasicShapesIQFog1NM.frag.glsl", 0 },
	{ GL_NONE, NULL, 0 }
	};

	gShaderProgramObjectBasicShapesNormalMap = LoadShaders(BasicShapesNM);

	ShaderInfo HDR[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/HDR.vert.glsl", 0 },
	{ GL_FRAGMENT_SHADER, "Resources/Shaders/HDR.frag.glsl", 0 },
	{ GL_NONE, NULL, 0 }
	};

	gShaderObjectHDR = LoadShaders(HDR);


	ShaderInfo SkyScatterAashish[] =
	{
				{ GL_VERTEX_SHADER, "Resources/Shaders/AtmosphericScatterringPerFragment2.vert.glsl", 0 },
	{ GL_FRAGMENT_SHADER, "Resources/Shaders/AtmosphericScatterringPerFragment2.frag.glsl", 0 },
	{ GL_NONE, NULL, 0 }
	};

	gShaderProgramObjectScatteringAashish = LoadShaders(SkyScatterAashish);

	// initialize light values
	pSunLight->SetAmbient(glm::vec3(0.2f, 0.2f, 0.2f));
	pSunLight->SetDiffuse(glm::vec3(1.0f, 1.0f, 1.0f));
	pSunLight->SetSpecular(glm::vec3(1.0f, 1.0f, 1.0f));
	pSunLight->SetDirection(glm::vec3(0.0f, 1.0f, 0.0f));



	// for debug light probes
	InitializeLightCube();
	InitializeQuadHDR();

	pCTerrain->LoadHeightMap("Resources/Raw/terrain0-16bbp-257x257.raw", 16, 257, 257);
	pCTerrain->InitializeTerrain(LoadShaders(TerrainDay), LoadShaders(0));



	//////////////////////////////////////
	// Sky initialization

	pCSkyDome->InitializeSky(LoadShaders(SkyShader));

	///////////////////////////////////////////////////////////////////
	// Basic shapes initialization
	gTextureRingID = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/BricksDiffuse.tga");


	BrickDiffuse = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/bricks_02_dif.tga");
	BrickSpecular = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/bricks_02_spec.tga");
	BrickNormal = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/bricks_02_nm.tga");
	BrickNormalInv = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/bricks_02_nm_inv.tga");

	BrickDiffuseGranite = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/bricks_03_dif_granite.tga");
	BrickDiffuseGrey = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/bricks_03_dif_grey.tga");
	BrickDiffuseOldRed = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/bricks_03_dif_old_red.tga");
	BrickSpecular2 = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/Bricks_03_spec.tga");
	BrickNormal2 = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/Bricks_03_nm.tga");
	BrickNormalInv2 = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/Bricks_03_nm_inv.tga");

	GraniteDiffuse = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/granite_01_dif.tga");
	GraniteSpecular = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/granite_01_spec.tga");
	GraniteNormal = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/granite_01_nm.tga");
	GraniteNormalInv = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/granite_01_nm_inv.tga");

	PavingDiffuse = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/paving_02_dif.tga");
	PavingSpecular = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/paving_02_spec.tga");
	PavingNormal = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/paving_02_nm.tga");
	PavingNormalInv = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/paving_02_nm_inv.tga");

	PlasterDiffuseGreen = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/plaster_02_dif_green.tga");
	PlasterDiffuseWhite = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/plaster_02_dif_white.tga");
	PlasterDiffuseYellow = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/plaster_02_dif_white.tga");
	PlasterSpecular = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/plaster_02_spec.tga");
	PlasterNormal = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/plaster_02_nm.tga");
	PlasterNormalInv = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/plaster_02_nm_inv.tga");

	WoodPlatesDiffuse = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/wood_plates_05_dif.tga");
	WoodPlatesSpecular = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/wood_plates_05_spec.tga");
	WoodNormal = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/wood_plates_05_nm.tga");
	WoodNormalInv = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/wood_plates_05_nm_inv.tga");


	glUseProgram(gShaderProgramBasicShapes);
	glUniform1i(glGetUniformLocation(gShaderProgramBasicShapes, "material.DiffuseTexture"), 0);
	glUniform1i(glGetUniformLocation(gShaderProgramBasicShapes, "material.SpecularTexture"), 1);
	glUseProgram(0);

	glUseProgram(gShaderProgramObjectBasicShapesNormalMap);
	glUniform1i(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "material.DiffuseTexture"), 0);
	glUniform1i(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "material.SpecularTexture"), 1);
	glUniform1i(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "material.NormalTexture"), 2);
	glUseProgram(0);

	//////////////////////////////////////////////////////////////////


	//glEnable(GL_FRAMEBUFFER_SRGB);

	///////////////////////////////////////
	// Create MSAA FBO
	glGenFramebuffers(1, &gFBOMSAA);
	glBindFramebuffer(GL_FRAMEBUFFER, gFBOMSAA);

	// create a multisampled color attachment
	glGenTextures(1, &gTextureMSAA);
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, gTextureMSAA);
	glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, MSAA_SAMPLES, GL_RGB8, giWinWidth, giWinHeight, GL_TRUE);
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

	// configure floating point frame buffer for HDR rendering
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
	//glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, giWinWidth, giWinHeight);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH32F_STENCIL8, giWinWidth, giWinHeight);

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


	// Create the ramebuffer for the occlusion map
	glGenFramebuffers(1, &OcclusionFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, OcclusionFBO);

	// create a texture for occlusion map
	glGenTextures(1, &OcclusionMapTexture);
	glBindTexture(GL_TEXTURE_2D, OcclusionMapTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, giWinWidth, giWinHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, OcclusionMapTexture, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

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

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//gPerspectiveProjectionMatrix = glm::perspective(glm::radians(gfFOVY), (GLfloat)giWinWidth / (GLfloat)giWinHeight, 0.1f, 5000.0f);
	glm::mat4 ModelMatrix = glm::mat4(1.0f);
	glm::mat4 ModelViewMatrix = glm::mat4(1.0f);
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

	// calculate the ambient and diffuse light color
	pSunLight->SetAmbient(glm::vec3(pCSkyDome->GetAmbientIntensity()));
	pSunLight->SetDiffuse(glm::vec3(pCSkyDome->GetLightColor() * pCSkyDome->GetDiffuseIntensity()));
	pSunLight->SetSpecular(glm::vec3(pCSkyDome->GetLightColor() * pCSkyDome->GetDiffuseIntensity()));
	pSunLight->SetDirection(glm::vec3(-pCSkyDome->GetSunWPos()));


	glBindFramebuffer(GL_FRAMEBUFFER, OcclusionFBO);
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(pCTerrain->GetShaderObject(TERRAIN_DAY));

	// Set Up the Sun Light for day time
	SetDayLight(pCTerrain->GetShaderObject(TERRAIN_DAY));

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	pCTerrain->RenderTerrain(ModelMatrix, ViewMatrix, gPerspectiveProjectionMatrix, glm::vec4(0, -1, 0, 100000));

	glUseProgram(0);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);


	// Render scene into msaa framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, gFBOMSAA);
	//glBindFramebuffer(GL_FRAMEBUFFER, gFrameBufferObjectHDR);

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


	glUseProgram(pCTerrain->GetShaderObject(TERRAIN_DAY));

	// Set Up the Sun Light for day time
	SetDayLight(pCTerrain->GetShaderObject(TERRAIN_DAY));

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	//pCTerrain->RenderTerrain(ModelMatrix, ViewMatrix, gPerspectiveProjectionMatrix, glm::vec4(0, -1, 0, 100000));

	glUseProgram(0);

	///////////////////////////////////////////////////////////////////////////////////////////////

	// render basic shapes
	
	//DrawAllBasicShapesInOneRow(BrickDiffuse, BrickSpecular, ModelMatrix, TranslationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRow(BrickDiffuseGranite, BrickSpecular2, ModelMatrix, TranslationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRow(BrickDiffuseGrey, BrickSpecular2, ModelMatrix, TranslationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRow(BrickDiffuseOldRed, BrickSpecular2, ModelMatrix, TranslationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRow(GraniteDiffuse, GraniteSpecular, ModelMatrix, TranslationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRow(PavingDiffuse, PavingSpecular, ModelMatrix, TranslationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRow(PlasterDiffuseGreen, PlasterSpecular, ModelMatrix, TranslationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRow(PlasterDiffuseWhite, PlasterSpecular, ModelMatrix, TranslationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRow(PlasterDiffuseYellow, PlasterSpecular, ModelMatrix, TranslationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRow(WoodPlatesDiffuse, WoodPlatesSpecular, ModelMatrix, TranslationMatrix, ViewMatrix);


	//DrawAllBasicShapesInOneRowWithNormalMap(BrickDiffuse, BrickSpecular, BrickNormalInv, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRowWithNormalMap(BrickDiffuseGranite, BrickSpecular2, BrickNormalInv2, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRowWithNormalMap(BrickDiffuseGrey, BrickSpecular2, BrickNormalInv2, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);
	DrawAllBasicShapesInOneRowWithNormalMap(BrickDiffuseOldRed, BrickSpecular2, BrickNormalInv2, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRowWithNormalMap(GraniteDiffuse, GraniteSpecular, GraniteNormalInv, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRowWithNormalMap(PavingDiffuse, PavingSpecular, PavingNormalInv, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRowWithNormalMap(PlasterDiffuseGreen, PlasterSpecular, PlasterNormalInv, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRowWithNormalMap(PlasterDiffuseWhite, PlasterSpecular, PlasterNormalInv, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRowWithNormalMap(PlasterDiffuseYellow, PlasterSpecular, PlasterNormalInv, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRowWithNormalMap(WoodPlatesDiffuse, WoodPlatesSpecular, WoodNormalInv, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);

	//////////////////////////////////////////////////////////
	// SkySphere Rendering
	/*Theta += 0.0125f;
	gSunPosition.x = 0.0f;
	gSunPosition.y = glm::cos(Theta) * 0.3f + 0.2f;
	gSunPosition.z = -0.5f;

	glUseProgram(gShaderProgramObjectScatteringAashish);

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 20.0f));
	ModelMatrix = ModelMatrix * TranslationMatrix;

	ModelViewProjectionMatrix = glm::mat4(1.0f);
	ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;

	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectScatteringAashish, "u_mvp_matrix"), 1, GL_FALSE, glm::value_ptr(ModelViewProjectionMatrix));

	glUniform1f(glGetUniformLocation(gShaderProgramObjectScatteringAashish, "PI"), 3.141592);
	glUniform1i(glGetUniformLocation(gShaderProgramObjectScatteringAashish, "iSteps"), 16);
	glUniform1i(glGetUniformLocation(gShaderProgramObjectScatteringAashish, "jSteps"), 8);

	glUniform3fv(glGetUniformLocation(gShaderProgramObjectScatteringAashish, "uSunPos"), 1, glm::value_ptr(gSunPosition));

	glDisable(GL_CULL_FACE);
	pCSkySphere->DrawSphere();
	glEnable(GL_CULL_FACE);

	glUseProgram(0);*/

	// blit the multisample buffer to hdr
	glBindFramebuffer(GL_READ_FRAMEBUFFER, gFBOMSAA);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gFrameBufferObjectHDR);
	glBlitFramebuffer(0, 0, giWinWidth, giWinHeight, 0, 0, giWinWidth, giWinHeight, GL_COLOR_BUFFER_BIT, GL_NEAREST);
	//glBlitFramebuffer(0, 0, giWinWidth, giWinHeight, 0, 0, giWinWidth, giWinHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR);
	glBlitFramebuffer(0, 0, giWinWidth, giWinHeight, 0, 0, giWinWidth, giWinHeight, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
	glBlitFramebuffer(0, 0, giWinWidth, giWinHeight, 0, 0, giWinWidth, giWinHeight, GL_STENCIL_BUFFER_BIT, GL_NEAREST);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
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

}


void DrawAllBasicShapesInOneRowWithNormalMap(GLuint DiffuseTexture, GLuint SpecularTexture, GLuint NormalTexture, glm::mat4 ModelMatrix, glm::mat4 TranslationMatrix, glm::mat4 RotationMatrix, glm::mat4 ViewMatrix)
{
	glUseProgram(gShaderProgramObjectBasicShapesNormalMap);

	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "u_view_matrix"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "u_projection_matrix"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));

	SetDayLightNormalMap(gShaderProgramObjectBasicShapesNormalMap);

	//bind texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, DiffuseTexture);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, SpecularTexture);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, NormalTexture);

	glUniform1f(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "material.Shininess"), 128.0f);


	// Ring 
	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -10.0f));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	RotationMatrix = glm::rotate(glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "u_model_matrix"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	pCRing[1]->DrawRing();

	// Sphere
	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(15.0f, 0.0f, -10.0f));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	RotationMatrix = glm::rotate(glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "u_model_matrix"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	pCSphere[1]->DrawSphere();

	// Torus
	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(30.0f, 0.0f, -10.0f));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "u_model_matrix"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	pCTorus[0]->DrawTorus();

	// Cylinder
	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(-15.0f, 0.0f, -10.0f));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	RotationMatrix = glm::rotate(glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "u_model_matrix"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	pCCylinder[0]->DrawCylinder();

	// Cone
	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(-30.0f, 0.0f, -10.0f));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	RotationMatrix = glm::rotate(glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "u_model_matrix"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	pCCone[1]->DrawCone();

	glUseProgram(0);
}

void Update(void)
{

	pCPlayer->PlayerUpdate(gfDeltaTime);

	if (gbIsDayNightCyclePaused == GL_FALSE)
		pCSkyDome->UpdateSky(gfDeltaTime);
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

	pCTerrain->DeleteTerrain();

	if (gbIsFullscreen == true)
	{
		dwStyle = GetWindowLong(ghWnd, GWL_STYLE);
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghWnd, &wpPrev);
		SetWindowPos(ghWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
	}


	

	if (gFBOMSAA)
	{
		glDeleteFramebuffers(1, &gFBOMSAA);
		gFBOMSAA = 0;
	}

	if (gRBOMSAA)
	{
		glDeleteRenderbuffers(1, &gRBOMSAA);
		gRBOMSAA = 0;
	}

	if (gTextureMSAA)
	{
		glDeleteTextures(1, &gTextureMSAA);
		gTextureMSAA = 0;
	}

	if (DiffuseTexture)
	{
		glDeleteTextures(1, &DiffuseTexture);
		DiffuseTexture = 0;
	}

	if (OpacityTexture)
	{
		glDeleteTextures(1, &OpacityTexture);
		OpacityTexture = 0;
	}

	if (gFrameBufferObjectHDR)
	{
		glDeleteFramebuffers(1, &gFrameBufferObjectHDR);
		gFrameBufferObjectHDR = 0;
	}

	if (gRenderBufferObjectDepth)
	{
		glDeleteRenderbuffers(1, &gRenderBufferObjectDepth);
		gRenderBufferObjectDepth = 0;
	}

	if (gColorBufferTexture)
	{
		glDeleteTextures(1, &gColorBufferTexture);
		gColorBufferTexture = 0;
	}

	if (gVBOColor)
	{
		glDeleteBuffers(1, &gVBOColor);
		gVBOColor = 0;
	}


	if (gVBOHDR)
	{
		glDeleteBuffers(1, &gVBOHDR);
		gVBOHDR = 0;
	}

	if (gVBONormal)
	{
		glDeleteBuffers(1, &gVBONormal);
		gVBONormal = 0;
	}

	if (gVBOPosition)
	{
		glDeleteBuffers(1, &gVBOPosition);
		gVBOPosition;
	}

	if (gVBOTexture)
	{
		glDeleteBuffers(1, &gVBOTexture);
		gVBOTexture = 0;
	}

	if (gVAOHDRQuad)
	{
		glDeleteVertexArrays(1, &gVAOHDRQuad);
		gVAOHDRQuad = 0;
	}

	if (gVAOLightCube)
	{
		glDeleteVertexArrays(1, &gVAOLightCube);
		gVAOLightCube = 0;
	}

	if (gVAOSampleQuad)
	{
		glDeleteVertexArrays(1, &gVAOSampleQuad);
		gVAOSampleQuad = 0;
	}

	if (gShaderObjectHDR)
	{
		glDeleteProgram(gShaderObjectHDR);
		gShaderObjectHDR = 0;
	}

	if (gShaderObjectMSAA)
	{
		glDeleteProgram(gShaderObjectMSAA);
		gShaderObjectMSAA = 0;
	}

	if (gShaderObjectSampleTexture)
	{
		glDeleteProgram(gShaderObjectSampleTexture);
		gShaderObjectSampleTexture = 0;
	}

	if (gShaderProgramObjectBasicShader)
	{
		glDeleteProgram(gShaderProgramObjectBasicShader);
		gShaderProgramObjectBasicShader = 0;
	}

	if (gShaderProgramObjectBasicShaderTexture)
	{
		glDeleteProgram(gShaderProgramObjectBasicShaderTexture);
		gShaderProgramObjectBasicShaderTexture = 0;
	}

	// basic shapes cleanup
	delete pCRing[0];
	delete pCRing[1];

	delete pCSphere[0];
	delete pCSphere[1];

	delete pCTorus[0];
	delete pCTorus[1];

	delete pCCylinder[0];
	delete pCCylinder[1];

	delete pCCone[0];
	delete pCCone[1];
	
	glDeleteTextures(1, &BrickDiffuse);
	glDeleteTextures(1, &BrickDiffuseGranite);
	glDeleteTextures(1, &BrickDiffuseGrey);
	glDeleteTextures(1, &BrickDiffuseOldRed);
	glDeleteTextures(1, &BrickNormal);
	glDeleteTextures(1, &BrickNormalInv);
	glDeleteTextures(1, &BrickNormal2);
	glDeleteTextures(1, &BrickNormalInv2);
	glDeleteTextures(1, &BrickSpecular);
	glDeleteTextures(1, &BrickSpecular2);

	glDeleteTextures(1, &GraniteDiffuse);
	glDeleteTextures(1, &GraniteSpecular);
	glDeleteTextures(1, &GraniteNormal);
	glDeleteTextures(1, &GraniteNormalInv);

	glDeleteTextures(1, &PlasterDiffuseGreen);
	glDeleteTextures(1, &PlasterDiffuseWhite);
	glDeleteTextures(1, &PlasterDiffuseYellow);
	glDeleteTextures(1, &PlasterNormal);
	glDeleteTextures(1, &PlasterNormalInv);
	glDeleteTextures(1, &PlasterSpecular);

	glDeleteTextures(1, &PavingDiffuse);
	glDeleteTextures(1, &PavingSpecular);
	glDeleteTextures(1, &PavingNormal);
	glDeleteTextures(1, &PavingNormalInv);

	glDeleteTextures(1, &WoodPlatesDiffuse);
	glDeleteTextures(1, &WoodPlatesSpecular);
	glDeleteTextures(1, &WoodNormal);
	glDeleteTextures(1, &WoodNormalInv);

	glDeleteProgram(gShaderProgramBasicShapes);
	glDeleteProgram(gShaderProgramObjectBasicShapesNormalMap);


	glUseProgram(0);

	if (pTGA != NULL)
	{
		delete pTGA;
		pTGA = NULL;
	}

	if (pCTerrain != NULL)
	{
		delete pCTerrain;
		pCTerrain = NULL;
	}

	if (pCSkyDome != NULL)
	{
		delete pCSkyDome;
		pCSkyDome = NULL;
	}



	if (pSunLight != NULL)
	{
		delete pSunLight;
		pSunLight = NULL;
	}

	if (pCCamera != NULL)
	{
		delete pCCamera;
		pCCamera = NULL;
	}

	if (pCPlayer != NULL)
	{
		delete pCPlayer;
		pCPlayer = NULL;
	}

	if (pCCommon)
	{
		delete pCCommon;
		pCCommon = NULL;
	}


	wglMakeCurrent(NULL, NULL);

	wglDeleteContext(ghRC);
	ghRC = NULL;

	ReleaseDC(ghWnd, ghDC);
	ghDC = NULL;

	DestroyWindow(ghWnd);
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


void SetDayLight(GLuint ShaderObject)
{
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.ambient"), 1, glm::value_ptr(pSunLight->GetAmbient()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.diffuse"), 1, glm::value_ptr(pSunLight->GetDiffuse()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.specular"), 1, glm::value_ptr(pSunLight->GetSpecular()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.direction"), 1, glm::value_ptr(pSunLight->GetDirection()));

	glUniform3fv(glGetUniformLocation(ShaderObject, "u_view_position"), 1, glm::value_ptr(CameraPosition));
}

void SetDayLightNormalMap(GLuint ShaderObject)
{
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.ambient"), 1, glm::value_ptr(pSunLight->GetAmbient()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.diffuse"), 1, glm::value_ptr(pSunLight->GetDiffuse()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.specular"), 1, glm::value_ptr(pSunLight->GetSpecular()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.direction"), 1, glm::value_ptr(pSunLight->GetDirection()));

	glUniform3fv(glGetUniformLocation(ShaderObject, "u_light_direction"), 1, glm::value_ptr(pSunLight->GetDirection()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "u_view_position"), 1, glm::value_ptr(CameraPosition));
}

