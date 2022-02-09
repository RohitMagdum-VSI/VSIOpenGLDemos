#pragma once

#include <Windows.h>
#include <stdio.h>

#include <GL/glew.h>
#include <GL/GL.h>

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtx\rotate_vector.hpp>

#include "../Terrain/Terrain.h"

#define GROUND_HEIGHT	0.0f

#define PLAYER_HEIGHT_NORMAL	2.0f
#define PLAYER_HEIGHT_CROUCHED	1.0f

#define PLAYER_MOVE_SPEED_NORMAL	5.0f
#define PLAYER_MOVE_SPEED_FAST		10.0f

#define TOGGLE_CROUCH	TRUE
#define TOGGLE_SPRINT	TRUE

#define WATER_HEIGHT 20.0f

enum PlayerMovement
{
	PLAYER_FORWARD = 0,
	PLAYER_BACKWARD,
	PLAYER_LEFT,
	PLAYER_RIGHT,
	PLAYER_JUMP
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

	GLfloat fOldMouseX;
	GLfloat fOldMouseY;

	GLfloat fTargetLookAngleX;
	GLfloat fTargetLookAngleY;

	float fLookAngleX;
	float fLookAngleY;

	GLfloat fJumpAccelerationTime;
	GLfloat fJumpStrength;
	GLfloat fGravityStrength;
	GLfloat fMoveSpeed;

	glm::vec3 TargetVelocity;

	CTerrain *pCTerrain = NULL;

	GLfloat fPlayerHeight;
	GLfloat fGroundHeight;

public:
	CPlayer(glm::vec3 PlayerPosition, CTerrain *pCTerrain);
	~CPlayer();


	//void ControlMouseInput(void);
	void ControlMouseInput(GLfloat XOffset, GLfloat YOffset);
	void ComputeWalkingvectors(void);
	void ControlMouseInput(void);
	void ControlPlayerMovement(PlayerMovement Direction, GLfloat DeltaTime);
	void PlayerUpdate(GLfloat fDelta);

	void ComputeCameraOrientation(void);
	
	void SetCameraLook(glm::vec3 Look);
	void SetCameraSide(glm::vec3 Side);
	void SetCameraUp(glm::vec3 Up);
	void SetCameraFront(glm::vec3 Front);
	void SetPlayerFront(glm::vec3 PlayerFront);
	void SetPlayerPosition(glm::vec3 Position);



	glm::vec3 GetCameraLook(void);
	glm::vec3 GetCameraSide(void);
	glm::vec3 GetCameraUp(void);
	glm::vec3 GetCameraFront(void);
	glm::vec3 GetPlayerFront(void);
	glm::vec3 GetPlayerPosition(void);

	GLboolean GetIsMoving(void);
	GLboolean GetIsTouchingGround(void);
	GLboolean GetIsCrouching(void);
	GLboolean GetIsSprinting(void);

	void SetIsCrouching(GLboolean Value);
	void SetIsSprinting(GLboolean Value);

	glm::mat4 GetViewMatrix(void);

	void InvertLookAngleX(GLfloat fDelta);
};

