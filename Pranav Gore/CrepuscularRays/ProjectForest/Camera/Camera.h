#pragma once

#include <Windows.h>
#include <string>

#include <math.h>
#include <gl\glew.h>
#include <gl\GL.h>
#include <vector>

//#define GLM_FORCE_CUDA
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtx\rotate_vector.hpp>

enum CameraMovement
{
	CAMERA_FORWARD = 0,
	CAMERA_BACKWARD,
	CAMERA_LEFT,
	CAMERA_RIGHT,
	CAMERA_UP,
	CAMERA_DOWN
};

// default camera values
const GLfloat YAW = -90.0f;
const GLfloat PITCH = 0.0f;
const GLfloat ROLL = 0.0f;
const GLfloat SPEED = 20.0f;
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
	CCamera(glm::vec3 Pos = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 Up = glm::vec3(0.0f, 1.0f, 0.0f), GLfloat Yaw = YAW, GLfloat Pitch = PITCH, GLfloat Roll = ROLL) : CameraFront(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
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
	glm::vec3 GetCameraUp(void);
	glm::vec3 GetCameraRight(void);

	void SetCameraPosition(glm::vec3 position);
	void InvertPitch();
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



