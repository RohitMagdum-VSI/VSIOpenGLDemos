#include "Camera.h"

//using namespace glm;

void CCamera::UpdateCameraVectors(void)
{
	// calculate the new front vector
	GLfloat FrontVectorX = cos(glm::radians(CameraYaw)) * cos(glm::radians(CameraPitch));
	GLfloat FrontVectorY = sin(glm::radians(CameraPitch));
	GLfloat FrontVectorZ = sin(glm::radians(CameraYaw)) * cos(glm::radians(CameraPitch));
	glm::vec3 FrontVector(FrontVectorX, FrontVectorY, FrontVectorZ);
	CameraFront = glm::normalize(FrontVector);

	// Also re-calculate the right and up vector
	CameraRight = glm::normalize(glm::cross(CameraFront, WorldUp));
	CameraUp = glm::normalize(glm::cross(CameraRight, CameraFront));
}

glm::mat4 CCamera::GetViewMatrix(void)
{
	return(glm::lookAt(CameraPosition, CameraPosition + CameraFront, CameraUp));
}

glm::vec3 CCamera::GetCameraPosition(void)
{
	return(CameraPosition);
}

glm::vec3 CCamera::GetCameraFront(void)
{
	return(CameraFront);
}

//void CCamera::Pitch(GLfloat Angle)
//{
//
//}
//
//void CCamera::Yaw(GLfloat Angle)
//{
//
//}
//
//void CCamera::Roll(GLfloat Angle)
//{
//
//}

void CCamera::ProcessNavigationKeys(CameraMovement Direction, GLfloat DeltaTime)
{
	GLfloat Velocity = MovementSpeed * DeltaTime;

	if (Direction == CAMERA_FORWARD)
	{
		CameraPosition += CameraFront * Velocity;
	}
		
	if (Direction == CAMERA_BACKWARD)
	{
		CameraPosition -= CameraFront * Velocity;
	}
		
	if (Direction == CAMERA_LEFT)
	{
		CameraPosition -= CameraRight * Velocity;
	}
		
	if (Direction == CAMERA_RIGHT)
	{
		CameraPosition += CameraRight * Velocity;
	}
		
	if (Direction == CAMERA_UP)
	{
		CameraPosition += CameraUp * Velocity;
	}
		
	if (Direction == CAMERA_DOWN)
	{
		CameraPosition -= CameraUp * Velocity;
	}
		
}

void CCamera::MoveForward(GLfloat Velocity, GLfloat DeltaTime)
{

	CameraPosition += CameraFront * Velocity;
}

void CCamera::MoveBackward(GLfloat Velocity, GLfloat DeltaTime)
{
	CameraPosition -= CameraFront * Velocity;
}

void CCamera::MoveUpward(GLfloat Velocity, GLfloat DeltaTime)
{
	CameraPosition += CameraUp * Velocity;
}

void CCamera::MoveDownward(GLfloat Velocity, GLfloat DeltaTime)
{
	CameraPosition -= CameraUp * Velocity;
}

void CCamera::StrafeRight(GLfloat Velocity, GLfloat DeltaTime)
{
	CameraPosition += CameraRight * Velocity;
}

void CCamera::StrafeLeft(GLfloat Velocity, GLfloat DeltaTime)
{
	CameraPosition -= CameraRight * Velocity;
}

void CCamera::ProcessMouseMovement(GLfloat XOffset, GLfloat YOffset, GLboolean ConstrainPitch)
{
	XOffset *= MouseSensitivity;
	YOffset *= MouseSensitivity;

	CameraYaw += XOffset;
	CameraPitch += YOffset;

	if (ConstrainPitch)
	{
		if (CameraPitch > 89.0f)
			CameraPitch = 89.0f;
		if (CameraPitch < -89.0f)
			CameraPitch = -89.0f;
	}

	UpdateCameraVectors();
}

// player class
CPlayer::CPlayer(glm::vec3 Position)
{
	if (fopen_s(&pFileCamera, "LogCamera.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Log File Can Not Be Created\nExitting ..."), TEXT("Error"), MB_OK | MB_TOPMOST | MB_ICONSTOP);
		exit(0);
	}
	else
	{
		fprintf(pFileCamera, "Log File Is Successfully Opened.\n");
	}


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
}

CPlayer::~CPlayer()
{
	if (pFileCamera)
	{
		fprintf(pFileCamera, "Log File Is Successfully Closed.\n");
		fclose(pFileCamera);
		pFileCamera = NULL;
	}
}

void CPlayer::PlayerUpdate(GLfloat fDelta)
{
	if (fJumpTimer > 0.0f)
	{
		fGravity += (fJumpStrength * (fJumpTimer / fJumpAccelerationTime)) * fDelta;
	}
	fJumpTimer -= fDelta;

	// add gravity
	fGravity -= fGravityStrength * fDelta;
	PlayerPosition.y += fGravity * fDelta;

	if (PlayerPosition.y < GROUND_HEIGHT + fPlayerHeight)
	{
		PlayerPosition.y = GROUND_HEIGHT + fPlayerHeight;
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
	
	fTargetLookAngleX -= (float)((MousePoint.y - dOldMouseY) * 0.01f);
	fTargetLookAngleY -= (float)((MousePoint.x - dOldMouseX) * 0.01f);

	if (fTargetLookAngleX > fMaxLookPitch)
		fTargetLookAngleX = fMaxLookPitch;

	if (fTargetLookAngleX < -fMaxLookPitch)
		fTargetLookAngleX = -fMaxLookPitch;

	dOldMouseX = MousePoint.x;
	dOldMouseY = MousePoint.y;
}

void CPlayer::ControlMouseInput(void)
{
	
	GetCursorPos(&MousePoint);

	fTargetLookAngleX -= (float)(MousePoint.y - dOldMouseY) * 0.01f;
	fTargetLookAngleY -= (float)(MousePoint.x - dOldMouseX) * 0.01f;

	if (fTargetLookAngleX > fMaxLookPitch)
		fTargetLookAngleX = fMaxLookPitch;

	if (fTargetLookAngleX < -fMaxLookPitch)
		fTargetLookAngleX = -fMaxLookPitch;

	dOldMouseX = GetSystemMetrics(SM_CXSCREEN) / 2;
	dOldMouseY = GetSystemMetrics(SM_CYSCREEN) / 2;;
}

void CPlayer::ComputeCameraOrientation(void)
{
	fLookAngleX = fLookAngleX + (fTargetLookAngleX - fLookAngleX) * 0.8f;
	fLookAngleY = fLookAngleY + (fTargetLookAngleY - fLookAngleY) * 0.8f;

	fprintf_s(pFileCamera, "fLookAngleX: %f\tfLookAngleY: %f\n", fLookAngleX, fLookAngleY);

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

glm::vec3 CPlayer::GetCameraLook(void)
{
	return(CameraLook);
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

glm::mat4 CPlayer::GetViewMatrix(void)
{
	return(glm::lookAt(GetPlayerPosition(), GetCameraLook(), glm::vec3(0.0f, 1.0f, 0.0f)));
}

