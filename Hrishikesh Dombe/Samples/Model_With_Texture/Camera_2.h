#pragma once

#include <math.h>
#include <C:/glm/glm.hpp>
#include <C:/glm/gtc/matrix_transform.hpp>
#include <C:/glm/gtc/type_ptr.hpp>

namespace FRAG_Camera2
{
	enum Camera_Movement {
		FORWARD,
		BACKWARD,
		LEFT,
		RIGHT
	};

	const float YAW = -90.0f;
	const float PITCH = 0.0f;
	const float SPEED = 2.5f;
	const float SENSITIVITY = 0.1f;
	const float ZOOM = 45.0f;

	class Camera
	{
	public:
		glm::vec3 Position;
		glm::vec3 Front;
		glm::vec3 Up;
		glm::vec3 Right;
		glm::vec3 WorldUp;

		float Yaw;
		float Pitch;
		float Zoom;
		float MovementSpeed;
		float MouseSensitivity;

		Camera() : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
		{
			Position = glm::vec3(0.0f, 0.0f, 0.0f);
			WorldUp = glm::vec3(0.0f, 1.0f, 0.0f);
			Yaw = YAW;
			Pitch = PITCH;
			updateCameraVectors();
		}

		Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch) : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
		{
			Position = position;
			WorldUp = up;
			Yaw = yaw;
			Pitch = pitch;
			updateCameraVectors();
		}

		void ProcessKeyboard(Camera_Movement direction, float deltaTime)
		{
			float velocity = 0.7f * (deltaTime + 1.0f);

			if (direction == FORWARD)
				Position += Front * velocity;
			if (direction == BACKWARD)
				Position -= Front * velocity;
			if (direction == RIGHT)
				Position += Right * velocity;
			if (direction == LEFT)
				Position -= Right * velocity;
		}

		glm::vec3 GetCameraPostion()
		{
			return this->Position;
		}

		glm::vec3 GetTargetVector()
		{
			return (this->Position + this->Front);
		}

		glm::mat4 GetViewMatrix()
		{
			return glm::lookAt(Position, Position + Front, Up);
		}

		void ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch = false)
		{
			xoffset *= MouseSensitivity;
			yoffset *= MouseSensitivity;

			Yaw += xoffset;
			Pitch += yoffset;

			if (constrainPitch)
			{
				if (Pitch > 179.0f)
					Pitch = 179.0f;
				if (Pitch < -179.0f)
					Pitch = -179.0f;
			}
			updateCameraVectors();
		}

	private:
		void updateCameraVectors()
		{
			glm::vec3 front;

			front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
			front.y = sin(glm::radians(Pitch));
			front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));

			Front = glm::normalize(front);
			Right = glm::normalize(glm::cross(Front, WorldUp));
			Up = glm::normalize(glm::cross(Right, Front));
		}
	};

}


