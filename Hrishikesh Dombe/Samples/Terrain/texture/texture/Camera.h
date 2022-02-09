#include<DirectXMath.h>

using namespace DirectX;

class Camera
{
public:
    Camera();
    ~Camera();

    //Getter, Setter for camera Position
    XMVECTOR GetPositionXM();
    XMFLOAT3 GetPosition();
    void SetPosition(float x, float y, float z);
    void SetPosition(const XMFLOAT3& v);

    //Getter for Camera basis vectors
    XMVECTOR GetRightXM();
    XMFLOAT3 GetRight();
    XMVECTOR GetUpXM();
    XMFLOAT3 GetUp();
    XMVECTOR GetLookXM();
    XMFLOAT3 GetLook();

    //Getter for frustum properties
    float GetNearZ();
    float GetFarZ();
    float GetAspect();
    float GetFovY();
    float GetFovX();

    //Getter for near and far plane dimensions
    float GetNearWindowWidth();
    float GetNearWindowHeight();
    float GetFarWindowWidth();
    float GetFarWindowHeight();

    //Set Frustum
    void SetLens(float fovY, float aspect, float znear, float zfar);

    //Define Camera via LookAt Parameters
    void LookAt(FXMVECTOR position, FXMVECTOR target, FXMVECTOR worldup);
    void LookAt(const XMFLOAT3& position, const XMFLOAT3& target, const XMFLOAT3& worldup);

    //Getter for View and Projection Matrix
    XMMATRIX View();
    XMMATRIX Projection();
    XMMATRIX ViewProjection();

    //Strafe and Walk with distance d
    void Strafe(float distance);
    void Walk(float distance);

    //Rotate Camera
    void Pitch(float angle);
    void RotateY(float angle);

    // After modifying camera position/orientation, call to rebuild the view matrix
    void UpdateViewMatrix();

    void UpdateCameraPosition(float deltaTime);

private:
    //Camera Co-ordinate System, with coordinate relative to world space.
    XMFLOAT3 mPosition;
    XMFLOAT3 mRight;
    XMFLOAT3 mUp;
    XMFLOAT3 mLook;

    //Frustum Properties
    float mNearZ;
    float mFarZ;
    float mAspect;
    float mFovY;
    float mNearWindowHeight;
    float mFarWindowHeight;

    XMFLOAT4X4 mViewMatrix;
    XMFLOAT4X4 mProjectionMatrix;
};
