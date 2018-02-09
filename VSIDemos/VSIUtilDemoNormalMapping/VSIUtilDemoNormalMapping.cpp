#include <stdio.h>
#include <gl\glew.h>
#include <VSIUtilPlus.h>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\matrix_transform.hpp>

using namespace VSIUtil;

class VSIDemoNormalMapping : public VSIUtilPlus
{
public:

    GLuint m_progObj;
    GLuint m_projLocation;
    GLuint m_mvLocation;
    GLuint m_vao;
    GLuint m_BB8Buffer[4];
    GLuint m_rimColorLocation;
    GLuint m_rimPowerLocation;
    GLuint m_lightPositionLocation;
    GLfloat m_xTrans, m_yTrans, m_zTrans;
    bool m_rimEnable;
    GLuint m_texture[2];

    glm::vec3 m_lightPosition;
    glm::vec3 m_rimColor;
    GLfloat m_rimPower;

    std::vector<glm::vec3> m_BB8Vertices;
    std::vector<glm::vec3> m_BB8Normals;
    std::vector<glm::vec2> m_BB8TextureCoords;
    std::vector<glm::vec3> m_BB8Tangents;

    VSIDemoNormalMapping()
    {
        m_rimColor = glm::vec3(0.3f, 0.3f, 0.3f);
        m_rimPower = 2.5f;
        m_rimEnable = true;
        m_lightPosition = glm::vec3(100.0, 100.0, 100.0);
        m_zTrans = -10.0f;
        m_yTrans = 0.0f;
    }

    ~VSIDemoNormalMapping()
    {

    }

    void WindowInit()
    {
        WCHAR tempName[128] = TEXT("VSI Normal Mapping.");
        wcscpy_s(mszAppName, wcslen(tempName) + 1, tempName);
    }

    std::vector<glm::vec3>& VSIUtilGetVertices()
    {
        return m_BB8Vertices;
    }

    std::vector<glm::vec3>& VSIUtilGetNormals()
    {
        return m_BB8Normals;
    }

    std::vector<glm::vec2>& VSIUtilGetTexcoords()
    {
        return m_BB8TextureCoords;
    }

    std::vector<glm::vec3>& VSIUtilGetTangents()
    {
        return m_BB8Tangents;
    }

    void VSIUtilSceneInit()
    {
        VSIUtilLoadMesh("cylinder.obj", VERTEX_NORMAL_AND_TEXTURE);

        m_progObj = VSIUtilLoadShaders("VSINormalMapping.vs.glsl", "VSINormalMapping.fs.glsl");
        
        m_projLocation = glGetUniformLocation(m_progObj, "projMatrix");
        m_mvLocation = glGetUniformLocation(m_progObj, "mvMatrix");
        m_lightPositionLocation = glGetUniformLocation(m_progObj, "lightPosition");

        glGenVertexArrays(1, &m_vao);
        glBindVertexArray(m_vao);

        VSIUtilGenAndBindBuffer(m_BB8Buffer[VERTEX], 0, VERTEX);
        VSIUtilGenAndBindBuffer(m_BB8Buffer[NORMALS], 1, NORMALS);
        VSIUtilGenAndBindBuffer(m_BB8Buffer[TEXTURECOORDS], 2, TEXTURECOORDS);
        VSIUtilGenAndBindBuffer(m_BB8Buffer[TANGENTS], 3, TANGENTS);

        glActiveTexture(GL_TEXTURE0); 
        VSIUtilLoadTexture("metal.jpg", &m_texture[0]);
        
        glActiveTexture(GL_TEXTURE1); 
        VSIUtilLoadTexture("normal.bmp", &m_texture[1]);
        glBindVertexArray(0);
    }

    void VSIUtilRender()
    {
        static const GLfloat black[] = { 0.0f, 0.0f, 0.0f, 1.0f };
        static const GLfloat one = 1.0f;

        glBindVertexArray(m_vao);
        glUseProgram(m_progObj);

        glClearBufferfv(GL_COLOR, 0, black);
        glClearBufferfv(GL_DEPTH, 0, &one);

        glUseProgram(m_progObj);

        static GLfloat f = 0.0f;
        f += 0.00001f;

        glm::mat4x4 translate = glm::translate(glm::mat4(1.0f), glm::vec3(m_xTrans, m_yTrans, m_zTrans));
        glm::mat4x4 y_rotate =
            glm::rotate(translate, f * 45.0f, glm::vec3(0.0, 1.0f, 0.0));
        glm::mat4x4 z_rotate =
            glm::rotate(y_rotate, 15.7f, glm::vec3(0.0, 0.0f, 1.0));

        glm::mat4 mvMatrix = z_rotate;

        glUniformMatrix4fv(m_mvLocation, 1, GL_FALSE, glm::value_ptr(mvMatrix));
        glUniformMatrix4fv(m_projLocation, 1, GL_FALSE, glm::value_ptr(mProjMatrix));
        glUniform3fv(m_lightPositionLocation, 1, glm::value_ptr(m_lightPosition));

        glDrawArrays(GL_TRIANGLES, 0, m_BB8Vertices.size());
    }

    void VSIUtilMessageHandler(UINT iMsg, WPARAM wParam, LPARAM lParam)
    {
        switch (iMsg)
        {
        case WM_CHAR:
            switch (wParam)
            {
            case 'p':
                m_lightPosition[0] += 5.0f;
                break;
            case 'P':
                m_lightPosition[0] -= 5.0f;
                break;
            case 'o':
                m_lightPosition[1] += 5.0f;
                break;
            case 'O':
                m_lightPosition[1] -= 5.0f;
                break;
            case 'i':
                m_lightPosition[2] += 5.0f;
                break;
            case 'I':
                m_lightPosition[2] -= 5.0f;
                break;
            case 'q':
                m_xTrans += 1.1f;
                break;
            case 'w':
                m_yTrans += 1.1f;
                break;
            case 'e':
                m_zTrans += 1.1f;
                break;
            case 'a':
                m_xTrans -= 1.1f;
                break;
            case 's':
                m_yTrans -= 5.1f;
                break;
            case 'd':
                m_zTrans -= 5.1f;
                break;
            }
        }
    }

};

VSI_MAIN(VSIDemoNormalMapping)