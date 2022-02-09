#pragma once
#include <windows.h>

#include <math.h>
#include <gl\glew.h>
#include <gl\GL.h>
#include <vector>

#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm\gtx\rotate_vector.hpp>

#define GROUND_HEIGHT	0.0f

#define PLAYER_HEIGHT_NORMAL	2.0f
#define PLAYER_HEIGHT_CROUCHED	1.0f

#define PLAYER_MOVE_SPEED_NORMAL	3.0f
#define PLAYER_MOVE_SPEED_FAST		10.0f

#define TOGGLE_CROUCH	FALSE
#define TOGGLE_SPRINT	FALSE

enum CameraMovement
{
	CAMERA_FORWARD = 0,
	CAMERA_BACKWARD,
	CAMERA_LEFT,
	CAMERA_RIGHT,
	CAMERA_UP,
	CAMERA_DOWN
};

enum PlayerMovement
{
	PLAYER_FORWARD = 0,
	PLAYER_BACKWARD,
	PLAYER_LEFT,
	PLAYER_RIGHT,
	PLAYER_JUMP,
};

// default camera values
const GLfloat YAW = -90.0f;
const GLfloat PITCH = 0.0f;
const GLfloat ROLL = 0.0f;
const GLfloat SPEED = 100.0f;
const GLfloat SENSITIVITY = 0.1f;
const GLfloat ZOOM = 45.0f;

class CCamera
{
private:
	// camera attributes
	glm::vec3 CameraPosition;
	glm::vec3 CameraFront;
	glm::vec3 CameraUp;
	glm::vec3 CameraRight;
	glm::vec3 WorldUp;

	// Euler angles
	GLfloat CameraYaw;
	GLfloat CameraPitch;
	GLfloat CameraRoll;

	// Camera Options
	GLfloat MovementSpeed;
	GLfloat MouseSensitivity;
	GLfloat Zoom;

	// calculate the front vector from camera's euler angles
	void UpdateCameraVectors(void);

public:
	// constructor with vector values
	CCamera(glm::vec3 Pos, glm::vec3 Up, GLfloat Yaw = YAW, GLfloat Pitch = PITCH, GLfloat Roll = ROLL) : CameraFront(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
	{
		CameraPosition = Pos;
		WorldUp = Up;
		CameraYaw = Yaw;
		CameraPitch = Pitch;
		CameraRoll = Roll;
		UpdateCameraVectors();
	}

	// constructor with scalar values
	CCamera(GLfloat PosX, GLfloat PosY, GLfloat PosZ, GLfloat UpX, GLfloat UpY, GLfloat UpZ, GLfloat Yaw, GLfloat Pitch, GLfloat Roll) : CameraFront(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
	{
		CameraPosition = glm::vec3(PosX, PosY, PosZ);
		WorldUp = glm::vec3(UpX, UpY, UpZ);
		CameraYaw = Yaw;
		CameraPitch = Pitch;
		CameraRoll = Roll;
		UpdateCameraVectors();
	}

	// returns the view matrix calculated using Euler angles and the lookAt
	glm::mat4 GetViewMatrix(void);
	glm::vec3 GetCameraPosition(void);
	glm::vec3 GetCameraFront(void);

	//void Pitch(GLfloat Angle);
	//void Yaw(GLfloat Angle);
	//void Roll(GLfloat Angle);

	void ProcessMouseMovement(GLfloat XOffset, GLfloat YOffset, GLboolean ContrainPitch);

	void MoveForward(GLfloat Velocity, GLfloat DeltaTime);
	void MoveBackward(GLfloat Velocity, GLfloat DeltaTime);
	void MoveUpward(GLfloat Velocity, GLfloat DeltaTime);
	void MoveDownward(GLfloat Velocity, GLfloat DeltaTime);
	void StrafeRight(GLfloat Velocity, GLfloat DeltaTime);
	void StrafeLeft(GLfloat Velocity, GLfloat DeltaTime);

	void ProcessNavigationKeys(CameraMovement Direction, GLfloat DeltaTime);
};

class CPlayer
{
private:
	FILE * pFileCamera = NULL;

	GLfloat fMaxLookPitch;

	POINT MousePoint;

	// motion and orientation control
	glm::vec3 PlayerPosition;
	glm::vec3 PlayerForward;
	glm::vec3 PlayerSide;
	glm::vec3 PlayerUp;

	glm::vec3 CameraForward;
	glm::vec3 CameraSide;
	glm::vec3 CameraUp;
	glm::vec3 CameraLook;

	GLfloat fJumpTimer;
	GLfloat fGravity;
	GLboolean bIsTouchingGround;
	GLboolean bIsMoving;
	GLboolean bIsCrouching;
	GLboolean bIsSprinting;

	GLdouble dOldMouseX;
	GLdouble dOldMouseY;

	GLfloat fTargetLookAngleX;
	GLfloat fTargetLookAngleY;

	float fLookAngleX;
	float fLookAngleY;

	GLfloat fJumpAccelerationTime;
	GLfloat fJumpStrength;
	GLfloat fGravityStrength;
	GLfloat fMoveSpeed;

	glm::vec3 TargetVelocity;

public:
	CPlayer(glm::vec3 PlayerPosition);
	~CPlayer();
	GLfloat fPlayerHeight;

	//void ControlMouseInput(void);
	void ControlMouseInput(GLfloat XOffset, GLfloat YOffset);
	void ComputeWalkingvectors(void);
	void ControlMouseInput(void);
	void ControlPlayerMovement(PlayerMovement Direction, GLfloat DeltaTime);
	void PlayerUpdate(GLfloat fDelta);

	void ComputeCameraOrientation(void);
	glm::vec3 GetPlayerPosition(void);
	void SetPlayerPosition(glm::vec3 Position);

	glm::vec3 GetCameraLook(void);
	glm::vec3 GetCameraSide(void);
	glm::vec3 GetCameraUp(void);
	glm::vec3 GetCameraFront(void);

	GLboolean GetIsMoving(void);
	GLboolean GetIsTouchingGround(void);
	GLboolean GetIsCrouching(void);
	GLboolean GetIsSprinting(void);

	void SetIsCrouching(GLboolean Value);
	void SetIsSprinting(GLboolean Value);

	glm::mat4 GetViewMatrix(void);
};
