#pragma once

#include <math.h>

class Camera
{
private:

	vmath::vec3 vPos;
	vmath::vec3 vForward;
	vmath::vec3 vUp;

	const vmath::vec3 yAxis = vmath::vec3(0.0f, 1.0f, 0.0f);

	vmath::vec3  RotateCamera(float fAngle, vmath::vec3 axis)
	{
		float fSinThetaBy2 = (float)sin(to_radians(fAngle / 2));
		float fCosThetaBy2 = (float)cos(to_radians(fAngle / 2));

		// First convert axis of rotation into quaternian
		float x = axis[0] * fSinThetaBy2;
		float y = axis[1] * fSinThetaBy2;
		float z = axis[2] * fSinThetaBy2;
		float w = fCosThetaBy2;

		vmath::quaternion quat = vmath::quaternion(x, y, z, w);

		// Simple multiplication MEANS the rotation in the world of complex numbers.
		vmath::quaternion retQuat = (quat * vForward);

		return vmath::vec3(retQuat[0], retQuat[1], retQuat[2]);
	}

public:

	Camera(vmath::vec3 pos, vmath::vec3 forward, vmath::vec3 up)
	{
		vPos = pos;
		vForward = forward;
		vUp = up;

		vUp = vmath::normalize(vUp);
		vForward = vmath::normalize(vForward);
	}

	Camera()
	{
		vPos = vmath::vec3(0.0f, 0.0f, 3.0f);
		vForward = vmath::vec3(0.0f, 0.0f, 1.0f);
		vUp = vmath::vec3(0.0f, 1.0f, 0.0f);
	}

	vmath::vec3 GetCameraPos() { return vPos; }
	vmath::vec3 GetCameraForward() { return vForward; }
	vmath::vec3 GetCameraUp() { return vUp; }

	void SetCameraPos(vmath::vec3 pos) { vPos = pos; }
	void SetCameraForward(vmath::vec3 forward) { vForward = forward; }
	void SetCameraUp(vmath::vec3 up) { vUp = up; }

	void CameraMove(vmath::vec3 direction, float fAmount)
	{
		vPos = vPos + (direction * fAmount);
	}

	vmath::vec3 GetCameraLeft()
	{
		vmath::vec3 vLeft = vmath::cross(vUp, vForward);;
		vLeft = vmath::normalize(vLeft);
		return vLeft;
	}

	vmath::vec3 GetCameraRight()
	{
		vmath::vec3 vRight = vmath::cross(vForward, vUp);;
		vRight = vmath::normalize(vRight);
		return vRight;
	}

	void RotateX(float fAngle)
	{
		vmath::vec3 vHorizontal = vmath::cross(yAxis, vForward);
		vHorizontal = vmath::normalize(vHorizontal);

		// Rotate vForward vector along horizontal axis by fAngle.
		vForward = RotateCamera(fAngle, vHorizontal);
		vForward = vmath::normalize(vForward);

		vUp = vmath::cross(vForward, vHorizontal);
		vUp = vmath::normalize(vUp);
	}

	void RotateY(float fAngle)
	{
		vmath::vec3 vHorizontal = vmath::cross(yAxis, vForward);
		vHorizontal = vmath::normalize(vHorizontal);

		// Rotate vForward vector along horizontal axis by fAngle.
		vForward = RotateCamera(fAngle, yAxis);
		vForward = vmath::normalize(vForward);

		vUp = vmath::cross(vForward, vHorizontal);
		vUp = vmath::normalize(vUp);
	}
};