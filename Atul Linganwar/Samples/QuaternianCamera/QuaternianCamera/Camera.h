#pragma once

typedef enum _MOVE_DIRECTION
{
	MOVE_FORWARD = 1,
	MOVE_BACKWARD,
	MOVE_RIGHT,
	MOVE_LEFT,
	MOVE_UP,
	MOVE_DOWN

}MOVE_DIRECTION;

typedef struct _LAST_MOUSE_COORDINATES
{
	GLfloat fLastMouseX;
	GLfloat fLastMouseY;

}LAST_MOUSE_COORDINATES;

class Camera
{
private:

	vmath::vec3 vPos;
	vmath::vec3 vForward;
	vmath::vec3 vUp;
	vmath::vec3 vCameraRight;

	GLfloat fYaw;
	GLfloat fPitch;
	GLfloat fRoll;

	GLfloat fRotationSpeed;

	LAST_MOUSE_COORDINATES LastMouseCoordinates;

	void RotateX(GLfloat fAmount);
	void RotateY(GLfloat fAmount);
	void RotateZ(GLfloat fAmount);

	void CameraMove(MOVE_DIRECTION direction, float fAmount);

	void UpdateVectors(vmath::quaternion q);
	void UpdateVectorsZ(vmath::quaternion q);

public:
	
	bool bIsRButtonDown;
	bool bFirstMouseCoordDone;

	Camera();
	Camera(vmath::vec3 pos, vmath::vec3 forward, vmath::vec3 up);

	vmath::vec3 GetCameraPos() { return vPos; }
	vmath::vec3 GetCameraForward() { return vForward; }
	vmath::vec3 GetCameraUp() { return vUp; }

	void SetCameraPos(vmath::vec3 pos) { vPos = pos; }
	void SetCameraForward(vmath::vec3 forward) { vForward = forward; }
	void SetCameraUp(vmath::vec3 up) { vUp = up; }

	void OnKeyPress(WPARAM wParam);
	void OnMouseMove(LPARAM lParam, WPARAM wParam);

	vmath::mat4 GetCameraViewMatrix();
};