#include "Player.h"

// player class
CPlayer::CPlayer(glm::vec3 Position, CTerrain *pCTerrain)
{
	if (fopen_s(&pFileCamera, "Logs/LogCamera.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Log File Can Not Be Created\nExitting ..."), TEXT("Error"), MB_OK | MB_TOPMOST | MB_ICONSTOP);
		exit(0);
	}
	else
	{
		fprintf(pFileCamera, "Log File Is Successfully Opened.\n");
	}

	this->pCTerrain = pCTerrain;

	fPlayerHeight = PLAYER_HEIGHT_NORMAL;
	fMaxLookPitch = glm::half_pi<float>() - 0.2f;
	this->PlayerPosition = Position;

	fJumpTimer = 0.0f;
	fGravity = 0.0f;
	bIsTouchingGround = GL_FALSE;
	bIsMoving = GL_FALSE;
	bIsCrouching = GL_FALSE;
	bIsSprinting = GL_FALSE;

	fTargetLookAngleX = 0.0f;
	fTargetLookAngleY = 0.0f;

	fLookAngleX = 0.0f;
	fLookAngleY = 0.0f;

	//ComputeWalkingvectors();

	fJumpAccelerationTime = 0.10f;
	fJumpStrength = 100.0f;
	fGravityStrength = 9.81f;
	fMoveSpeed = PLAYER_MOVE_SPEED_NORMAL;

	MousePoint.x = 0;
	MousePoint.y = 0;

	fOldMouseX = 0.0;
	fOldMouseY = 0.0;
}

CPlayer::~CPlayer()
{
	if (pCTerrain != NULL)
	{
		delete pCTerrain;
	}

	if (pFileCamera)
	{
		fprintf(pFileCamera, "Log File Is Successfully Closed.\n");
		fclose(pFileCamera);
		pFileCamera = NULL;
	}
}

void CPlayer::PlayerUpdate(GLfloat fDelta)
{
	fGroundHeight = pCTerrain->GetHeightAt(PlayerPosition);

	if (fJumpTimer > 0.0f)
	{
		fGravity += (fJumpStrength * (fJumpTimer / fJumpAccelerationTime)) * fDelta;
	}
	fJumpTimer -= fDelta;

	// add gravity
	fGravity -= fGravityStrength * fDelta;
	PlayerPosition.y += fGravity * fDelta;

	if (PlayerPosition.y < fGroundHeight + fPlayerHeight)
	{
		PlayerPosition.y = fGroundHeight + fPlayerHeight;
		fGravity = 0.0f;

		bIsTouchingGround = GL_TRUE;
	}
	else
	{
		bIsTouchingGround = GL_FALSE;
	}

	// crouch
	if (bIsCrouching == GL_TRUE)
	{
		fPlayerHeight -= 0.05f;
		if (fPlayerHeight <= PLAYER_HEIGHT_CROUCHED)
		{
			fPlayerHeight = PLAYER_HEIGHT_CROUCHED;
		}
	}
	else if (bIsCrouching == GL_FALSE)
	{
		fPlayerHeight += 0.05f;
		if (fPlayerHeight >= PLAYER_HEIGHT_NORMAL)
		{
			fPlayerHeight = PLAYER_HEIGHT_NORMAL;
		}
	}

	// sprint
	if (bIsSprinting == GL_TRUE)
	{
		fMoveSpeed += 2.0f;
		if (fMoveSpeed >= PLAYER_MOVE_SPEED_FAST)
		{
			fMoveSpeed = PLAYER_MOVE_SPEED_FAST;
		}
	}
	else if (bIsSprinting == GL_FALSE)
	{
		fMoveSpeed -= 2.0f;
		if (fMoveSpeed <= PLAYER_MOVE_SPEED_NORMAL)
		{
			fMoveSpeed = PLAYER_MOVE_SPEED_NORMAL;
		}
	}

	ComputeWalkingvectors();
	ComputeCameraOrientation();
}

void CPlayer::ComputeWalkingvectors(void)
{
	PlayerUp = glm::vec3(0.0f, 1.0f, 0.0f);
	PlayerForward = glm::rotate(glm::vec3(0.0f, 0.0f, -1.0f), fLookAngleY, PlayerUp);
	PlayerSide = glm::normalize(glm::cross(PlayerForward, PlayerUp));
}

void CPlayer::ControlMouseInput(GLfloat MouseX, GLfloat MouseY)
{
	SetCursorPos(683, 384);
	GetCursorPos(&MousePoint);

	fTargetLookAngleX -= (MousePoint.y - fOldMouseY) * 0.01f;
	fTargetLookAngleY -= (MousePoint.x - fOldMouseX) * 0.01f;

	if (fTargetLookAngleX > fMaxLookPitch)
		fTargetLookAngleX = fMaxLookPitch;

	if (fTargetLookAngleX < -fMaxLookPitch)
		fTargetLookAngleX = -fMaxLookPitch;

	fOldMouseX = (GLfloat)MousePoint.x;
	fOldMouseY = (GLfloat)MousePoint.y;
}

void CPlayer::ControlMouseInput(void)
{
	//fprintf_s(pFileCamera, "MousePoint.x: %d\tMousePoint.y: %d\n", MousePoint.x, MousePoint.y);

	GetCursorPos(&MousePoint);

	fTargetLookAngleX -= (MousePoint.y - fOldMouseY) * 0.01f;
	fTargetLookAngleY -= (MousePoint.x - fOldMouseX) * 0.01f;

	if (fTargetLookAngleX > fMaxLookPitch)
		fTargetLookAngleX = fMaxLookPitch;

	if (fTargetLookAngleX < -fMaxLookPitch)
		fTargetLookAngleX = -fMaxLookPitch;

	fOldMouseX = (GLfloat)GetSystemMetrics(SM_CXSCREEN) / 2.0f;
	fOldMouseY = (GLfloat)GetSystemMetrics(SM_CYSCREEN) / 2.0f;

	//fprintf_s(pFileCamera, "MousePoint.x: %d\tMousePoint.y: %d\n", MousePoint.x, MousePoint.y);
}

void CPlayer::ComputeCameraOrientation(void)
{
	fLookAngleX = fLookAngleX + (fTargetLookAngleX - fLookAngleX) * 0.8f;
	fLookAngleY = fLookAngleY + (fTargetLookAngleY - fLookAngleY) * 0.8f;

	glm::mat4 CameraMatrix = glm::mat4(1.0f);
	CameraMatrix = glm::rotate(CameraMatrix, fLookAngleY, glm::vec3(0.0f, 1.0f, 0.0f));
	CameraMatrix = glm::rotate(CameraMatrix, fLookAngleX, glm::vec3(1.0f, 0.0f, 0.0f));

	CameraForward = -glm::vec3(CameraMatrix[2]);
	CameraSide = glm::vec3(CameraMatrix[0]);
	CameraUp = glm::vec3(CameraMatrix[1]);
	CameraLook = PlayerPosition + CameraForward;
}

void CPlayer::ControlPlayerMovement(PlayerMovement Direction, GLfloat fDelta)
{
	bIsMoving = GL_FALSE;
	TargetVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

	if (Direction == PLAYER_FORWARD)
	{
		TargetVelocity += glm::vec3(0.0f, 0.0f, fMoveSpeed);
		PlayerPosition = PlayerPosition + (PlayerForward * TargetVelocity.z * fDelta) + (PlayerSide * TargetVelocity.x * fDelta);
		bIsMoving = GL_TRUE;
	}

	if (Direction == PLAYER_BACKWARD)
	{
		TargetVelocity += glm::vec3(0.0f, 0.0f, -fMoveSpeed);
		PlayerPosition = PlayerPosition + (PlayerForward * TargetVelocity.z * fDelta) + (PlayerSide * TargetVelocity.x * fDelta);
		bIsMoving = GL_TRUE;
	}

	if (Direction == PLAYER_LEFT)
	{
		TargetVelocity += glm::vec3(-fMoveSpeed, 0.0f, 0.0f);
		PlayerPosition = PlayerPosition + (PlayerForward * TargetVelocity.z * fDelta) + (PlayerSide * TargetVelocity.x * fDelta);
		bIsMoving = GL_TRUE;
	}

	if (Direction == PLAYER_RIGHT)
	{
		TargetVelocity += glm::vec3(fMoveSpeed, 0.0f, 0.0f);
		PlayerPosition = PlayerPosition + (PlayerForward * TargetVelocity.z * fDelta) + (PlayerSide * TargetVelocity.x * fDelta);
		bIsMoving = GL_TRUE;
	}

	if (Direction == PLAYER_JUMP)
	{
		fJumpTimer = fJumpAccelerationTime;
	}
}

glm::vec3 CPlayer::GetPlayerPosition(void)
{
	return(PlayerPosition);
}

void CPlayer::SetPlayerPosition(glm::vec3 Position)
{
	this->PlayerPosition = Position;
}

void CPlayer::SetCameraLook(glm::vec3 Look)
{
	this->CameraLook = Look;
}

void CPlayer::SetCameraSide(glm::vec3 Side)
{
	this->CameraSide = Side;
}

void CPlayer::SetCameraUp(glm::vec3 Up)
{
	this->CameraUp = Up;
}

void CPlayer::SetCameraFront(glm::vec3 Front)
{
	this->CameraForward = Front;
}

void CPlayer::SetPlayerFront(glm::vec3 PlayerFront)
{
	this->PlayerForward = PlayerFront;
}

glm::vec3 CPlayer::GetCameraLook(void)
{
	return(CameraLook);
	//return(PlayerForward);
}

glm::vec3 CPlayer::GetCameraSide(void)
{
	return(CameraSide);
}

glm::vec3 CPlayer::GetCameraUp(void)
{
	return(CameraUp);
}

GLboolean CPlayer::GetIsMoving(void)
{
	return(bIsMoving);
}

GLboolean CPlayer::GetIsCrouching(void)
{
	return(bIsCrouching);
}

GLboolean CPlayer::GetIsTouchingGround(void)
{
	return(bIsTouchingGround);
}

void CPlayer::SetIsCrouching(GLboolean Value)
{
	this->bIsCrouching = Value;
}

GLboolean CPlayer::GetIsSprinting(void)
{
	return(bIsSprinting);
}

void CPlayer::SetIsSprinting(GLboolean Value)
{
	this->bIsSprinting = Value;
}

glm::vec3 CPlayer::GetCameraFront(void)
{
	return(CameraForward);
}

glm::vec3 CPlayer::GetPlayerFront(void)
{
	return(PlayerForward);
}

glm::mat4 CPlayer::GetViewMatrix(void)
{
	return(glm::lookAt(GetPlayerPosition(), GetCameraLook(), glm::vec3(0.0f, 1.0f, 0.0f)));
}

void CPlayer::InvertLookAngleX(GLfloat fDelta)
{
	fLookAngleX = -fLookAngleX;
	//PlayerUpdate(fDelta);
	ComputeCameraOrientation();
}
