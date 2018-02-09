#include <Windows.h>
#include <gl\glew.h>
#include <VSIUtilPlus.h>
#include <VSIUtilModelObject.h>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>

#define DEPTH_TEXTURE_SIZE 4096
using namespace VSIUtil;

class VSIDemoShadowMapping :public VSIUtilPlus
{
public:
    VSIUtilModelObject m_Cylinder;
    VSIUtilModelObject m_Sphere;
    VSIUtilModelObject m_Cube;
    VSIUtilModelObject m_Teapot;

    GLuint m_cubeVAO;
    GLuint m_cylinderVAO;
    GLuint m_sphereVAO;
    GLuint m_teapotVAO;

    GLuint m_progLight;
    GLuint m_progView;
    GLuint m_progLightDepth;

    GLfloat m_xTrans, m_yTrans, m_zTrans;

    GLuint m_lightMVPLocation;
    GLuint m_projMatrixLocation, m_mvMatrixLocation, m_shadowMatrixLocation, m_fullShadingLocation;

    GLuint m_depthFBO;
    GLuint m_depthTex;
    GLuint m_depthDebugTex;

    glm::vec3 m_lightPosition;
    glm::vec3 m_viewPosition;

    glm::mat4 m_lightProjMatrix;
    glm::mat4 m_lightViewMatrix;
    glm::mat4 m_cameraProjMatrix;
    glm::mat4 m_cameraViewMatrix;

    glm::mat4 mLightVPMatrix;
    glm::mat4 mShadowSBPVMatrix;

    GLuint mode;

    VSIDemoShadowMapping()
    {

    }

    ~VSIDemoShadowMapping()
    {

    }

    void WindowInit()
    {
        WCHAR tempName[128] = TEXT("VSI Shadow Mapping.");
        wcscpy_s(mszAppName, wcslen(tempName) + 1, tempName);
    }

    void loadShaders()
    {
        m_progLight = VSIUtilLoadShaders("VSIShadowLight.vs.glsl", "VSIShadowLight.fs.glsl");

        m_progView = VSIUtilLoadShaders("VSIShadowCamera.vs.glsl", "VSIShadowCamera.fs.glsl");
     
        m_progLightDepth = VSIUtilLoadShaders("VSIShadowLightView.vs.glsl", "VSIShadowLightView.fs.glsl");
        m_lightMVPLocation = glGetUniformLocation(m_progLight, "mvp");
        m_projMatrixLocation = glGetUniformLocation(m_progView, "proj_matrix");
        m_mvMatrixLocation = glGetUniformLocation(m_progView, "mv_matrix");
        m_shadowMatrixLocation = glGetUniformLocation(m_progView, "shadow_matrix");
        m_fullShadingLocation = glGetUniformLocation(m_progView, "full_shading");
    }

    void VSIUtilSceneInit()
    {
        GLuint temp = 0;
        loadShaders();
        
        glGenVertexArrays(1, &m_teapotVAO);
        glBindVertexArray(m_teapotVAO);
        m_CurrentModel = &m_Teapot;
        VSIUtilLoadMesh("Teapot.obj", VERTEX_AND_NORMAL);
        VSIUtilGenAndBindBuffer(temp, 0, VERTEX);
        VSIUtilGenAndBindBuffer(temp, 1, NORMALS);
        glBindVertexArray(0);

        glGenVertexArrays(1, &m_sphereVAO);
        glBindVertexArray(m_sphereVAO);
        m_CurrentModel = &m_Sphere;
        VSIUtilLoadMesh("sphere.obj", VERTEX_NORMAL_AND_TEXTURE);
        VSIUtilGenAndBindBuffer(temp, 0, VERTEX);
        VSIUtilGenAndBindBuffer(temp, 1, NORMALS);
        glBindVertexArray(0);


        glGenVertexArrays(1, &m_cubeVAO);
        glBindVertexArray(m_cubeVAO);
        m_CurrentModel = &m_Cube;
        VSIUtilLoadMesh("cube.obj", VERTEX_NORMAL_AND_TEXTURE);
        VSIUtilGenAndBindBuffer(temp, 0, VERTEX);
        VSIUtilGenAndBindBuffer(temp, 1, NORMALS);
        glBindVertexArray(0);


        glGenVertexArrays(1, &m_cylinderVAO);
        glBindVertexArray(m_cylinderVAO);
        m_CurrentModel = &m_Cylinder;
        VSIUtilLoadMesh("cylinder.obj", VERTEX_NORMAL_AND_TEXTURE);
        VSIUtilGenAndBindBuffer(temp, 0, VERTEX);
        VSIUtilGenAndBindBuffer(temp, 1, NORMALS);
        glBindVertexArray(0);


        glGenFramebuffers(1, &m_depthFBO);
        glBindFramebuffer(GL_FRAMEBUFFER, m_depthFBO);

        glGenTextures(1, &m_depthTex);
        glBindTexture(GL_TEXTURE_2D, m_depthTex);
        glTexStorage2D(GL_TEXTURE_2D, 11, GL_DEPTH_COMPONENT32F, DEPTH_TEXTURE_SIZE, DEPTH_TEXTURE_SIZE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, m_depthTex, 0);

        glGenTextures(1, &m_depthDebugTex);
        glBindTexture(GL_TEXTURE_2D, m_depthDebugTex);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, DEPTH_TEXTURE_SIZE, DEPTH_TEXTURE_SIZE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_depthDebugTex, 0);

        glBindTexture(GL_TEXTURE_2D, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glEnable(GL_DEPTH_TEST);

        
    }

    void VSIUtilRender()
    {
        static const GLfloat zeros[] = {0.0f, 0.0f, 0.0f, 0.0f};

        static GLfloat f = 0.0f;
        f += 0.0001f;

        glm::vec3 m_lightPosition = glm::vec3(20.0f, 20.0f, 20.0f);
        glm::vec3 m_viewPosition = glm::vec3(0.0f, 0.0f, 40.0f);

        m_lightProjMatrix = glm::frustum(-1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1000.0f);
        m_lightViewMatrix = glm::lookAt(m_lightPosition,
            glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

        m_cameraProjMatrix = glm::perspective(50.0f, 
            (float)mWidth/(float)mHeight,
            1.0f,
            2000.0f);


        m_cameraViewMatrix = glm::lookAt(glm::vec3(15.0f * sinf(f), 0.0f, 15.0f* cosf(f)),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0, 1.0f, 0.0f));

        m_Cube.m_modelMatrix = glm::rotate(glm::mat4(1.0f), f * 14.5f, glm::vec3(0.0f, 1.0f, 0.0f)) *
                               glm::rotate(glm::mat4(1.0f), 20.0f,     glm::vec3(1.0f, 0.0f, 0.0f)) *
                               glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, m_zTrans)) *
                               glm::scale(glm::mat4(1.0f), glm::vec3(m_xTrans, m_xTrans, m_xTrans));


        m_Cylinder.m_modelMatrix = m_cameraViewMatrix *
            glm::rotate(glm::mat4(1.0f), f * 3.7f, glm::vec3(0.0f, 1.0f, 0.0f)) *
            glm::translate(glm::mat4(1.0f), glm::vec3(sinf(f * 0.25f) * 3.0f, cosf(f * 0.25f) * 2.0f, -10.0f)) *
            glm::rotate(glm::mat4(1.0f), f * 99.0f, glm::vec3(0.0f, 0.0f, 1.0f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(m_yTrans, m_yTrans, m_yTrans));
            
        m_Sphere.m_modelMatrix = m_cameraViewMatrix * 
            glm::rotate(glm::mat4(1.0f), f * 3.7f, glm::vec3(0.0f, 1.0f, 0.0f)) *
            glm::translate(glm::mat4(1.0f), glm::vec3(sinf(f*0.37f) * 12.0f, cosf(f * 0.37f) * 12.0f, 0.0f)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(m_xTrans, m_xTrans, m_xTrans));

        m_Teapot.m_modelMatrix = m_cameraViewMatrix *
            glm::rotate(glm::mat4(1.0f), f * 50.0f, glm::vec3(1.0f, 0.0f, 0.0f))*
            glm::rotate(glm::mat4(1.0f), f * 130.0f, glm::vec3(0.0f, 1.0f, 0.0f)) *
            glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -4.0f, 0.0f));

        glEnable(GL_DEPTH_TEST);

        renderScene(f, true);

        renderScene(f, false);
    }

    void renderScene(GLfloat f, bool fromLight)
    {
        static const GLfloat ones[] = { 1.0f };
        static const GLfloat zero[] = { 0.0f };
        static const GLfloat gray[] = { 0.1f, 0.1f, 0.1f, 0.0f };

        static const glm::mat4 scaleBiasMatrix = glm::mat4(glm::vec4(0.5f, 0.0f, 0.0f, 0.0f),
                                                           glm::vec4(0.0f, 0.5f, 0.0f, 0.0f),
                                                           glm::vec4(0.0f, 0.0f, 0.5f, 0.0f),
                                                           glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));

        mLightVPMatrix = m_lightProjMatrix * m_lightViewMatrix;
        mShadowSBPVMatrix = scaleBiasMatrix * m_lightProjMatrix * m_lightViewMatrix;

        if (fromLight)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, m_depthFBO);
            glViewport(0, 0, DEPTH_TEXTURE_SIZE, DEPTH_TEXTURE_SIZE);
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(4.0f, 4.0f);
            glUseProgram(m_progLight);
            static const GLenum buffs[] = { GL_COLOR_ATTACHMENT0 };
            glDrawBuffers(1, buffs);
            glClearBufferfv(GL_COLOR, 0, zero);
        }
        else
        {
            glViewport(0, 0, mWidth, mHeight);
            glClearBufferfv(GL_COLOR, 0, gray);
            glUseProgram(m_progView);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, m_depthTex);
            glUniformMatrix4fv(m_projMatrixLocation, 1, GL_FALSE, glm::value_ptr(m_cameraProjMatrix));
            glDrawBuffer(GL_BACK);
        }

        glClearBufferfv(GL_DEPTH, 0, ones);
        
        glBindVertexArray(m_cubeVAO);
        renderObject(m_Cube, fromLight);
        
        glBindVertexArray(m_cylinderVAO);
        renderObject(m_Cylinder, fromLight);
        
        //glBindVertexArray(m_sphereVAO);
        //renderObject(m_Sphere, fromLight);
        
        glBindVertexArray(m_teapotVAO);
        renderObject(m_Teapot, fromLight);

        if (fromLight)
        {
            glDisable(GL_POLYGON_OFFSET_FILL);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
        else
        {
            glBindTexture(GL_TEXTURE_2D, 0);
        }
    }

    void renderObject(VSIUtilModelObject object, bool fromLight)
    {
        glm::mat4& model_matrix = object.m_modelMatrix;
        if (fromLight)
        {
            glUniformMatrix4fv(m_lightMVPLocation, 1, GL_FALSE, glm::value_ptr(mLightVPMatrix * object.m_modelMatrix));
        }
        else
        {
            glm::mat4 shadowMatrix = mShadowSBPVMatrix * model_matrix;
            glUniformMatrix4fv(m_shadowMatrixLocation, 1, GL_FALSE, glm::value_ptr(shadowMatrix));
            glUniformMatrix4fv(m_mvMatrixLocation, 1, GL_FALSE, glm::value_ptr(m_cameraViewMatrix * object.m_modelMatrix));
        }

        glDrawArrays(GL_TRIANGLES, 0, object.m_vertices.size());
    }

    void VSIUtilMessageHandler(UINT iMsg, WPARAM wParam, LPARAM lParam)
    {
        switch (iMsg)
        {
        case WM_CHAR:
            switch (wParam)
            {
            case 'q':
                m_xTrans += 0.5f;
                break;
            case 'w':
                m_yTrans += 0.5f;
                break;
            case 'e':
                m_zTrans += 0.5f;
                break;
            case 'a':
                m_xTrans -= 0.5f;
                break;
            case 's':
                m_yTrans -= 0.5f;
                break;
            case 'd':
                m_zTrans -= 0.5f;
                break;
            }
        }
    }
};

VSI_MAIN(VSIDemoShadowMapping)