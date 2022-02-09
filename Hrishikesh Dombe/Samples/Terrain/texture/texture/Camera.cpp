#include<Windows.h>
#include "Camera.h"

Camera::Camera():
    mPosition(0.0f,0.0f,0.0f),
    mRight(1.0f,0.0f,0.0f),
    mUp(0.0f,1.0f,0.0f),
    mLook(0.0f,0.0f,1.0f)
{
    SetLens(0.25f*XM_PI,1.0f,1.0f,1000.0f);
}

Camera::~Camera()
{

}

XMVECTOR Camera::GetPositionXM()
{
    return XMLoadFloat3(&mPosition);
}

XMFLOAT3 Camera::GetPosition()
{
    return mPosition;
}

void Camera::SetPosition(float x, float y, float z)
{
    mPosition = XMFLOAT3(x,y,z);
}

void Camera::SetPosition(const XMFLOAT3& v)
{
    mPosition = v;
}

XMVECTOR Camera::GetRightXM()
{
    return XMLoadFloat3(&mRight);
}

XMFLOAT3 Camera::GetRight()
{
    return mRight;
}

XMVECTOR Camera::GetUpXM()
{
    return XMLoadFloat3(&mUp);
}

XMFLOAT3 Camera::GetUp()
{
    return mUp;
}

XMVECTOR Camera::GetLookXM()
{
    return XMLoadFloat3(&mLook);
}

XMFLOAT3 Camera::GetLook()
{
    return mLook;
}

float Camera::GetNearZ()
{
    return mNearZ;
}

float Camera::GetFarZ()
{
    return mFarZ;
}

float Camera::GetAspect()
{
    return mAspect;
}

float Camera::GetFovY()
{
    return mFovY;
}

float Camera::GetFovX()
{
    float halfWidth = 0.5f*GetNearWindowWidth();
    return 2.0f*atan(halfWidth/mNearZ);
}

float Camera::GetNearWindowWidth()
{
    return mAspect * mNearWindowHeight;
}

float Camera::GetNearWindowHeight()
{
    return mNearWindowHeight;
}

float Camera::GetFarWindowWidth()
{
    return mAspect * mFarWindowHeight;
}

float Camera::GetFarWindowHeight()
{
    return mFarWindowHeight;
}

void Camera::SetLens(float fovY, float aspect, float znear, float zfar)
{
    mFovY = fovY;
    mAspect = aspect;
    mNearZ = znear;
    mFarZ = zfar;

    mNearWindowHeight = 2.0f * mNearZ * tanf(0.5f * mFovY);
    mFarWindowHeight = 2.0f * mFarZ * tanf(0.5f * mFovY);

    XMMATRIX ProjectionMatrix = XMMatrixPerspectiveFovLH(mFovY,mAspect,mNearZ,mFarZ);
    XMStoreFloat4x4(&mProjectionMatrix,ProjectionMatrix);
}

void Camera::LookAt(FXMVECTOR position, FXMVECTOR target, FXMVECTOR worldUp)
{
    XMVECTOR L = XMVector3Normalize(XMVectorSubtract(target,position));
    XMVECTOR R = XMVector3Normalize(XMVector3Cross(worldUp,L));
    XMVECTOR U = XMVector3Cross(L,R);

    XMStoreFloat3(&mPosition,position);
    XMStoreFloat3(&mLook,L);
    XMStoreFloat3(&mRight,R);
    XMStoreFloat3(&mUp,U);
}

void Camera::LookAt(const XMFLOAT3& position, const XMFLOAT3& target, const XMFLOAT3& up)
{
    XMVECTOR P = XMLoadFloat3(&position);
    XMVECTOR T = XMLoadFloat3(&target);
    XMVECTOR U = XMLoadFloat3(&up);

    LookAt(P,T,U);
}

XMMATRIX Camera::View()
{
    return XMLoadFloat4x4(&mViewMatrix);
}

XMMATRIX Camera::Projection()
{
    return XMLoadFloat4x4(&mProjectionMatrix);
}

XMMATRIX Camera::ViewProjection()
{
    return XMMatrixMultiply(View(),Projection());
}

void Camera::Strafe(float d)
{
    //mPosition += d*mRight
    XMVECTOR s = XMVectorReplicate(d);
    XMVECTOR r = XMLoadFloat3(&mRight);
    XMVECTOR p = XMLoadFloat3(&mPosition);
    XMStoreFloat3(&mPosition, XMVectorMultiplyAdd(s,r,p));
}

void Camera::Walk(float d)
{
    //mPosition += d*mLook
    XMVECTOR s = XMVectorReplicate(d);
    XMVECTOR l = XMLoadFloat3(&mLook);
    XMVECTOR p = XMLoadFloat3(&mPosition);
    XMStoreFloat3(&mPosition, XMVectorMultiplyAdd(s,l,p));
}

void Camera::Pitch(float angle)
{
    //Rotate up and look vector about the right vector
    XMMATRIX R = XMMatrixRotationAxis(XMLoadFloat3(&mRight),angle);

    XMStoreFloat3(&mUp, XMVector3TransformNormal(XMLoadFloat3(&mUp),R));
    XMStoreFloat3(&mLook, XMVector3TransformNormal(XMLoadFloat3(&mLook),R));
}

void Camera::RotateY(float angle)
{
    //Rotate the basis vector about world y-axis

    XMMATRIX R = XMMatrixRotationY(angle);

    XMStoreFloat3(&mRight, XMVector3TransformNormal(XMLoadFloat3(&mRight), R));
    XMStoreFloat3(&mUp, XMVector3TransformNormal(XMLoadFloat3(&mUp), R));
    XMStoreFloat3(&mLook, XMVector3TransformNormal(XMLoadFloat3(&mLook), R));
}

void Camera::UpdateViewMatrix()
{
    XMVECTOR R = XMLoadFloat3(&mRight);
    XMVECTOR U = XMLoadFloat3(&mUp);
    XMVECTOR L = XMLoadFloat3(&mLook);
    XMVECTOR P = XMLoadFloat3(&mPosition);

    L = XMVector3Normalize(L);
    U = XMVector3Normalize(XMVector3Cross(L,R));

    R = XMVector3Cross(U, L);

    float x = -XMVectorGetX(XMVector3Dot(P, R));
    float y = -XMVectorGetY(XMVector3Dot(P, U));
    float z = -XMVectorGetZ(XMVector3Dot(P, L));

    XMStoreFloat3(&mRight, R);
    XMStoreFloat3(&mUp, U);
    XMStoreFloat3(&mLook, L);

    mViewMatrix(0,0) = mRight.x;
    mViewMatrix(1,0) = mRight.y;
    mViewMatrix(2,0) = mRight.z;
    mViewMatrix(3,0) = x;

    mViewMatrix(0,1) = mUp.x;
    mViewMatrix(1,1) = mUp.y;
    mViewMatrix(2,1) = mUp.z;
    mViewMatrix(3,1) = y;

    mViewMatrix(0,2) = mLook.x;
    mViewMatrix(1,2) = mLook.y;
    mViewMatrix(2,2) = mLook.z;
    mViewMatrix(3,2) = z;

    mViewMatrix(0,3) = 0.0f;
    mViewMatrix(1,3) = 0.0f;
    mViewMatrix(2,3) = 0.0f;
    mViewMatrix(3,3) = 1.0f;
}

void Camera::UpdateCameraPosition(float deltaTime)
{
    if(GetAsyncKeyState(0x41))//A
		Strafe(-10.0f*deltaTime);

	if(GetAsyncKeyState(0x44))//D
		Strafe(10.0f*deltaTime);

	if(GetAsyncKeyState(0x53))//S
		Walk(-10.0f*deltaTime);

	if(GetAsyncKeyState(0x57))//W
		Walk(10.0f*deltaTime);
}

