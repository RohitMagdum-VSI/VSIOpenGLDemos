#include "LightsMaterial.h"

////////////////////////////////////////////////////
///////// Directional Light ///////////////////////

CDirectionalLight::CDirectionalLight()
{
	Ambient = glm::vec3(0.0f);
	Diffuse = glm::vec3(0.0f);
	Specular = glm::vec3(0.0f);
	Direction = glm::vec3(0.0f);
}

CDirectionalLight::~CDirectionalLight()
{
	// code
}

void CDirectionalLight::SetAmbient(glm::vec3 La)
{
	this->Ambient = La;
}

void CDirectionalLight::SetDiffuse(glm::vec3 Ld)
{
	this->Diffuse = Ld;
}

void CDirectionalLight::SetSpecular(glm::vec3 Ls)
{
	this->Specular = Ls;
}

void CDirectionalLight::SetDirection(glm::vec3 Dir)
{
	this->Direction = Dir;
}

glm::vec3 CDirectionalLight::GetAmbient(void)
{
	return(this->Ambient);
}

glm::vec3 CDirectionalLight::GetDiffuse(void)
{
	return(this->Diffuse);
}

glm::vec3 CDirectionalLight::GetSpecular(void)
{
	return(this->Specular);
}

glm::vec3 CDirectionalLight::GetDirection(void)
{
	return(this->Direction);
}

/////////////////////////////////////////////////////

//////////////////////////////////////////////////////
///////////////// Point Light ////////////////////////

CPointLight::CPointLight()
{
	Ambient = glm::vec3(0.0f);
	Diffuse = glm::vec3(0.0f);
	Specular = glm::vec3(0.0f);
	Position = glm::vec3(0.0f);

	fConstantAttenuation = 0.0f;
	fLinearAttenuation = 0.0f;
	fQuadraticAttenuation = 0.0f;
}

CPointLight::~CPointLight()
{
	// code
}

void CPointLight::SetAmbient(glm::vec3 La)
{
	this->Ambient = La;
}

void CPointLight::SetDiffuse(glm::vec3 Ld)
{
	this->Diffuse = Ld;
}

void CPointLight::SetSpecular(glm::vec3 Ls)
{
	this->Specular = Ls;
}

void CPointLight::SetPosition(glm::vec3 Pos)
{
	this->Position = Pos;
}

void CPointLight::SetAttenuation(GLfloat Constant, GLfloat Linear, GLfloat Quadratic)
{
	this->fConstantAttenuation = Constant;
	this->fLinearAttenuation = Linear;
	this->fQuadraticAttenuation = Quadratic;
}

glm::vec3 CPointLight::GetAmbient(void)
{
	return(this->Ambient);
}

glm::vec3 CPointLight::GetDiffuse(void)
{
	return(this->Diffuse);
}

glm::vec3 CPointLight::GetSpecular(void)
{
	return(this->Specular);
}

glm::vec3 CPointLight::GetPosition(void)
{
	return(this->Position);
}



GLfloat CPointLight::GetConstantAttenuation(void)
{
	return(this->fConstantAttenuation);
}

GLfloat CPointLight::GetLinearAttenuation(void)
{
	return(this->fLinearAttenuation);
}

GLfloat CPointLight::GetQuadraticAttenuation(void)
{
	return(this->fQuadraticAttenuation);
}

///////////////////////////////////////////////////

///////////////////////////////////////////////////
////////////////// Spot Light /////////////////////

CSpotLight::CSpotLight()
{
	Ambient = glm::vec3(0.0f);
	Diffuse = glm::vec3(0.0f);
	Specular = glm::vec3(0.0f);
	Position = glm::vec3(0.0f);
	Direction = glm::vec3(0.0f);

	fConstantAttenuation = 0.0f;
	fLinearAttenuation = 0.0f;
	fQuadraticAttenuation = 0.0f;

	SpotInnerCutOff = 0.0f;
	SpotOuterCutOff = 0.0f;

	bIsFlashLight = GL_FALSE;
}

CSpotLight::~CSpotLight()
{

}

void CSpotLight::SetAmbient(glm::vec3 La)
{
	this->Ambient = La;
}

void CSpotLight::SetDiffuse(glm::vec3 Ld)
{
	this->Diffuse = Ld;
}

void CSpotLight::SetSpecular(glm::vec3 Ls)
{
	this->Specular = Ls;
}

void CSpotLight::SetPosition(glm::vec3 Pos)
{
	this->Position = Pos;
}

void CSpotLight::SetDirection(glm::vec3 Dir)
{
	this->Direction = Dir;
}

void CSpotLight::SetAttenuation(GLfloat Constant, GLfloat Linear, GLfloat Quadratic)
{
	this->fConstantAttenuation = Constant;
	this->fLinearAttenuation = Linear;
	this->fQuadraticAttenuation = Quadratic;
}

void CSpotLight::SetSpotCutOff(GLfloat InnerCutOff, GLfloat OuterCutOff)
{
	this->SpotInnerCutOff = InnerCutOff;
	this->SpotOuterCutOff = OuterCutOff;
}

void CSpotLight::SetIsFlashLight(GLboolean Val)
{
	this->bIsFlashLight = Val;
}

glm::vec3 CSpotLight::GetAmbient(void)
{
	return(this->Ambient);
}

glm::vec3 CSpotLight::GetDiffuse(void)
{
	return(this->Diffuse);
}

glm::vec3 CSpotLight::GetSpecular(void)
{
	return(this->Specular);
}

glm::vec3 CSpotLight::GetPosition(void)
{
	return(this->Position);
}

glm::vec3 CSpotLight::GetDirection(void)
{
	return(this->Direction);
}

GLfloat CSpotLight::GetConstantAttenuation(void)
{
	return(this->fConstantAttenuation);
}

GLfloat CSpotLight::GetLinearAttenuation(void)
{
	return(this->fLinearAttenuation);
}

GLfloat CSpotLight::GetQuadraticAttenuation(void)
{
	return(this->fQuadraticAttenuation);
}

GLfloat CSpotLight::GetInnerSpotCutOff(void)
{
	return(this->SpotInnerCutOff);
}

GLfloat CSpotLight::GetOuterSpotCutOff(void)
{
	return(this->SpotOuterCutOff);
}

GLboolean CSpotLight::GetIsFlashLight(void)
{
	return(this->bIsFlashLight);
}

//////////////////////////////////////////////////

//////////////////////////////////////////////////
////////////// Material //////////////////////////

CMaterial::CMaterial()
{
	this->Ambient = glm::vec3(0.0f);
	this->Diffuse = glm::vec3(0.0f);
	this->Specular = glm::vec3(0.0f);
	this->Shininess = 0.0f;
}

CMaterial::~CMaterial()
{

}

void CMaterial::SetAmbient(glm::vec3 Ka)
{
	this->Ambient = Ka;
}

void CMaterial::SetDiffuse(glm::vec3 Kd)
{
	this->Diffuse = Kd;
}

void CMaterial::SetSpecular(glm::vec3 Ks)
{
	this->Specular = Ks;
}

void CMaterial::SetShininess(GLfloat Intensity)
{
	this->Shininess = Intensity;
}

glm::vec3 CMaterial::GetAmbient(void)
{
	return(this->Ambient);
}

glm::vec3 CMaterial::GetDiffuse(void)
{
	return(this->Diffuse);
}

glm::vec3 CMaterial::GetSpecular(void)
{
	return(this->Specular);
}

GLfloat CMaterial::GetShininess(void)
{
	return(this->Shininess);
}

