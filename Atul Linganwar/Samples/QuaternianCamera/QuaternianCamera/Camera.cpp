#include <Windows.h>
#include <windowsx.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "Main.h"
#include "Camera.h"

Camera::Camera(vmath::vec3 pos, vmath::vec3 forward, vmath::vec3 up)
{
	vPos = pos;
	vForward = forward;
	vUp = up;

	fYaw = 0.0f;
	fPitch = 0.0f;
	fRoll = 0.0f;

	bIsRButtonDown = false;
	bFirstMouseCoordDone = false;
	LastMouseCoordinates.fLastMouseX = 0.0f;
	LastMouseCoordinates.fLastMouseY = 0.0f;

	vUp = vmath::normalize(vUp);
	vForward = vmath::normalize(vForward);

	vCameraRight = vmath::cross(vForward, vUp);
	vCameraRight = vmath::normalize(vCameraRight);

	fRotationSpeed = 2.0f;
}

Camera::Camera()
{
	vPos = vmath::vec3(0.0f, 0.0f, 0.0f);
	vForward = vmath::vec3(0.0f, 0.0f, -1.0f);
	vUp = vmath::vec3(0.0f, 1.0f, 0.0f);
	vCameraRight = vmath::vec3(1.0f, 0.0f, 0.0f);

	fRotationSpeed = 0.5;

	bIsRButtonDown = false;
	bFirstMouseCoordDone = false;
	LastMouseCoordinates.fLastMouseX = 0.0f;
	LastMouseCoordinates.fLastMouseY = 0.0f;

	fYaw = 0.0f;
	fPitch = 0.0f;
	fRoll = 0.0f;
}

void Camera::OnKeyPress(WPARAM wParam)
{
	float speed = 2.0f;

	switch (wParam)
	{
	case VK_UP:
		RotateX(speed);
		break;

	case VK_DOWN:
		RotateX(-speed);
		break;

	case VK_RIGHT:
		RotateY(speed);
		break;

	case VK_LEFT:
		RotateY(-speed);
		break;

	case VK_OEM_PERIOD:
		RotateZ(speed);
		break;

	case VK_OEM_COMMA:
		RotateZ(-speed);
		break;

	case 0x57: //W
		CameraMove(MOVE_FORWARD, speed);
		break;

	case 0x53: //S
		CameraMove(MOVE_BACKWARD, speed);
		break;

	case 0x41: //A
		CameraMove(MOVE_LEFT, speed);
		break;

	case 0x44: //D
		CameraMove(MOVE_RIGHT, speed);
		break;

	case VK_SPACE:
		CameraMove(MOVE_UP, speed);
		break;

	case VK_SHIFT:
		CameraMove(MOVE_DOWN, speed);
		break;

	default:
		break;
	};
}

void Camera::OnMouseMove(LPARAM lParam, WPARAM wParam)
{
	DWORD dwFlags = (DWORD)wParam;
	int iMouseX = GET_X_LPARAM(lParam);
	int iMouseY = GET_Y_LPARAM(lParam);

	if (MK_LBUTTON & dwFlags)
	{
		// Do right mouse button click handling.
	}

	if (MK_RBUTTON & dwFlags && bIsRButtonDown)
	{
		if (!bFirstMouseCoordDone)
		{
			LastMouseCoordinates.fLastMouseX = iMouseX;
			LastMouseCoordinates.fLastMouseY = iMouseY;
			bFirstMouseCoordDone = true;
		}

		GLfloat fDeltaX = iMouseX - LastMouseCoordinates.fLastMouseX;
		GLfloat fDeltaY = LastMouseCoordinates.fLastMouseY - iMouseY;

		LastMouseCoordinates.fLastMouseX = iMouseX;
		LastMouseCoordinates.fLastMouseY = iMouseY;

		RotateY(-fDeltaX);
		RotateX(fDeltaY);
	}
}

vmath::mat4 Camera::GetCameraViewMatrix()
{
	return vmath::lookat(vPos, vPos + vForward, vUp);
}

void Camera::RotateX(GLfloat fAmount)
{
	vmath::quaternion quatRotationX;

	fYaw = fRotationSpeed * fAmount;

	quatRotationX[0] = (float)sin(to_radians(fYaw / 2));
	quatRotationX[1] = 0.0f;
	quatRotationX[2] = 0.0f;
	quatRotationX[3] = (float)cos(to_radians(fYaw / 2));

	UpdateVectors(quatRotationX);
}

void Camera::RotateY(GLfloat fAmount)
{
	vmath::quaternion quatRotationY;

	fPitch = fRotationSpeed * fAmount;

	quatRotationY[0] = 0.0f;
	quatRotationY[1] = (float)sin(to_radians(fPitch / 2));
	quatRotationY[2] = 0.0f;
	quatRotationY[3] = (float)cos(to_radians(fPitch / 2));

	UpdateVectors(quatRotationY);
}

void Camera::RotateZ(GLfloat fAmount)
{
	vmath::quaternion quatRotationZ;

	fRoll = fRotationSpeed * fAmount;

	quatRotationZ[0] = 0.0f;
	quatRotationZ[1] = 0.0f;
	quatRotationZ[2] = (float)sin(to_radians(fRoll / 2));
	quatRotationZ[3] = (float)cos(to_radians(fRoll / 2));

	UpdateVectorsZ(quatRotationZ);
}

void Camera::CameraMove(MOVE_DIRECTION direction, float fAmount)
{
	switch (direction)
	{
	case MOVE_FORWARD:
		vPos = vPos + vForward * fAmount;
		break;

	case MOVE_BACKWARD:
		vPos = vPos - vForward * fAmount;
		break;

	case MOVE_RIGHT:
		vPos = vPos + vCameraRight * fAmount;
		break;

	case MOVE_LEFT:
		vPos = vPos - vCameraRight * fAmount;
		break;

	case MOVE_UP:
		vPos = vPos + vUp * fAmount;
		break;

	case MOVE_DOWN:
		vPos = vPos - vUp * fAmount;
		break;
	}
}

void Camera::UpdateVectors(vmath::quaternion q)
{
	vmath::quaternion forward = q * vForward;

	vForward = vmath::vec3(forward[0], forward[1], forward[2]);
	vForward = vmath::normalize(vForward);

	vCameraRight = vmath::cross(vForward, vUp);
	vCameraRight = vmath::normalize(vCameraRight);

	vUp = vmath::cross(vCameraRight, vForward);
	vUp = vmath::normalize(vUp);
}

void Camera::UpdateVectorsZ(vmath::quaternion q)
{
	vmath::quaternion forward = q * vCameraRight;

	vCameraRight = vmath::vec3(forward[0], forward[1], forward[2]);
	vCameraRight = vmath::normalize(vCameraRight);

	vUp = vmath::cross(vCameraRight, vForward);
	vUp = vmath::normalize(vUp);
}