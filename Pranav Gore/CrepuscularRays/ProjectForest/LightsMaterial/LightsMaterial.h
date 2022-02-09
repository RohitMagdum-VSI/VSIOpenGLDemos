#pragma once

#include <gl/glew.h>
#include <gl\GL.h>
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>

#define MAX_POINT_LIGHTS	1

class CDirectionalLight
{

private:
	glm::vec3 Ambient;
	glm::vec3 Diffuse;
	glm::vec3 Specular;
	glm::vec3 Direction;

public:
	CDirectionalLight();
	~CDirectionalLight();

	void SetAmbient(glm::vec3 La);
	void SetDiffuse(glm::vec3 Ld);
	void SetSpecular(glm::vec3 Ls);
	void SetDirection(glm::vec3 Dir);

	glm::vec3 GetAmbient(void);
	glm::vec3 GetDiffuse(void);
	glm::vec3 GetSpecular(void);
	glm::vec3 GetDirection(void);
};

class CPointLight
{
private:
	glm::vec3 Ambient;
	glm::vec3 Diffuse;
	glm::vec3 Specular;
	glm::vec3 Position;

	GLfloat fConstantAttenuation;
	GLfloat fLinearAttenuation;
	GLfloat fQuadraticAttenuation;

public:
	CPointLight();
	~CPointLight();

	void SetAmbient(glm::vec3 La);
	void SetDiffuse(glm::vec3 Ld);
	void SetSpecular(glm::vec3 Ls);
	void SetPosition(glm::vec3 Pos);
	void SetAttenuation(GLfloat Constant, GLfloat Linear, GLfloat Quadratic);

	glm::vec3 GetAmbient(void);
	glm::vec3 GetDiffuse(void);
	glm::vec3 GetSpecular(void);
	glm::vec3 GetPosition(void);
	GLfloat GetConstantAttenuation(void);
	GLfloat GetLinearAttenuation(void);
	GLfloat GetQuadraticAttenuation(void);
};

class CSpotLight
{
private:
	glm::vec3 Ambient;
	glm::vec3 Diffuse;
	glm::vec3 Specular;
	glm::vec3 Position;
	glm::vec3 Direction;

	GLfloat fConstantAttenuation;
	GLfloat fLinearAttenuation;
	GLfloat fQuadraticAttenuation;

	GLfloat SpotInnerCutOff;
	GLfloat SpotOuterCutOff;

	GLboolean bIsFlashLight;

public:
	CSpotLight();
	~CSpotLight();

	void SetAmbient(glm::vec3 La);
	void SetDiffuse(glm::vec3 Ld);
	void SetSpecular(glm::vec3 Ls);
	void SetPosition(glm::vec3 Pos);
	void SetDirection(glm::vec3 Dir);
	void SetAttenuation(GLfloat Constant, GLfloat Linear, GLfloat Quadratic);
	void SetSpotCutOff(GLfloat InnerCutOff, GLfloat OuterCutOff);
	void SetIsFlashLight(GLboolean Val);

	glm::vec3 GetAmbient(void);
	glm::vec3 GetDiffuse(void);
	glm::vec3 GetSpecular(void);
	glm::vec3 GetPosition(void);
	glm::vec3 GetDirection(void);
	GLfloat GetConstantAttenuation(void);
	GLfloat GetLinearAttenuation(void);
	GLfloat GetQuadraticAttenuation(void);
	GLfloat GetInnerSpotCutOff(void);
	GLfloat GetOuterSpotCutOff(void);
	GLboolean GetIsFlashLight(void);
};

class CMaterial
{
private:
	glm::vec3 Ambient;
	glm::vec3 Diffuse;
	glm::vec3 Specular;
	GLfloat Shininess;

public:
	CMaterial();
	~CMaterial();

	void SetAmbient(glm::vec3 Ka);
	void SetDiffuse(glm::vec3 Kd);
	void SetSpecular(glm::vec3 Ks);
	void SetShininess(GLfloat Intensity);

	glm::vec3 GetAmbient(void);
	glm::vec3 GetDiffuse(void);
	glm::vec3 GetSpecular(void);
	GLfloat GetShininess(void);
};
