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
	return(this->CameraPosition);
}

glm::vec3 CCamera::GetCameraFront(void)
{
	return(this->CameraFront);
}

glm::vec3 CCamera::GetCameraUp(void)
{
	return(this->CameraUp);
}

glm::vec3 CCamera::GetCameraRight(void)
{
	return(this->CameraRight);
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


void CCamera::SetCameraPosition(glm::vec3 position)
{
	this->CameraPosition = position;
}

void CCamera::InvertPitch()
{
	this->CameraPitch = -CameraPitch;
	UpdateCameraVectors();
}
