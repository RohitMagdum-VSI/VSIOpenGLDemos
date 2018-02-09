#include <stdio.h>
#include <gl\glew.h>
#include <VSIUtilPlus.h>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\matrix_transform.hpp>

using namespace VSIUtil;

class VSIDemoRimLighting : public VSIUtilPlus
{
public:
    GLuint m_progObj;
    GLuint m_projLocation;
    GLuint m_mvLocation;
    GLuint m_vao;
    GLuint m_BB8Buffer[3];
    GLuint m_rimColorLocation;
    GLuint m_rimPowerLocation;
    GLuint m_lightPositionLocation;
    bool m_rimEnable;

    glm::vec3 m_lightPosition;
    glm::vec3 m_rimColor;
    GLfloat m_rimPower;

    std::vector<glm::vec3> m_BB8Vertices;
    std::vector<glm::vec3> m_BB8Normals;
    std::vector<glm::vec2> m_BB8TextureCoords;

    VSIDemoRimLighting()
    {
        m_rimColor = glm::vec3(0.3f, 0.3f, 0.3f);
        m_rimPower = 2.5f;
        m_rimEnable = true;
        m_lightPosition = glm::vec3(100.0, 100.0, 100.0);
    }

    ~VSIDemoRimLighting()
    {

    }

    void WindowInit()
    {
        WCHAR tempName[128] = TEXT("VSI Rim Lighting.");
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
        return std::vector<glm::vec3>();
    }

    void VSIUtilSceneInit()
    {
        VSIUtilLoadMesh("bb8_triangulated.obj", VERTEX_NORMAL_AND_TEXTURE);

        m_progObj = VSIUtilLoadShaders("VSIRimLighting.vs.glsl", "VSIRimLighting.fs.glsl");

        m_projLocation     = glGetUniformLocation(m_progObj, "projMatrix");
        m_mvLocation       = glGetUniformLocation(m_progObj, "mvMatrix");
        m_rimColorLocation = glGetUniformLocation(m_progObj, "rim_color");
        m_rimPowerLocation = glGetUniformLocation(m_progObj, "rim_power");
        m_lightPositionLocation = glGetUniformLocation(m_progObj, "lightPosition");

        glGenVertexArrays(1, &m_vao);
        glBindVertexArray(m_vao);

        VSIUtilGenAndBindBuffer(m_BB8Buffer[VERTEX], 0, VERTEX);
        VSIUtilGenAndBindBuffer(m_BB8Buffer[NORMALS], 1, NORMALS);
        VSIUtilGenAndBindBuffer(m_BB8Buffer[TEXTURECOORDS], 2, TEXTURECOORDS);

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

        static GLfloat f = 0.0f;
        f += 0.0001f;
        
        glm::mat4x4 translate = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 100.0, -800.0f));
  //      glm::mat4x4 y_rotate =
  //          glm::rotate(translate, f * 45.0f, glm::vec3(0.0, 1.0f, 0.0));
        glm::mat4x4 z_rotate = 
            glm::rotate(translate, 15.7f, glm::vec3(0.0, 0.0f, 1.0));

        glm::mat4 mvMatrix = z_rotate;

        glUniformMatrix4fv(m_mvLocation, 1, GL_FALSE, glm::value_ptr(mvMatrix));
        glUniformMatrix4fv(m_projLocation, 1, GL_FALSE, glm::value_ptr(mProjMatrix));

        glUniform3fv(m_rimColorLocation, 1, m_rimEnable ? glm::value_ptr(m_rimColor) : glm::value_ptr(glm::vec3(0.0f)));
        glUniform1f(m_rimPowerLocation, m_rimPower);
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
            case 'Q':
                m_rimColor[0] += 0.1f;
                break;
            case 'W':
                m_rimColor[1] += 0.1f;
                break;
            case 'E':
                m_rimColor[2] += 0.1f;
                break;
            case 'R':
                m_rimPower *= 1.5f;
                break;
            case 'A':
                m_rimColor[0] -= 0.1f;
                break;
            case 'S':
                m_rimColor[1] -= 0.1f;
                break;
            case 'D':
                m_rimColor[2] -= 0.1f;
                break;
            case 'G':
                m_rimPower /= 1.5f;
                break;
            case 'Z':
                m_rimEnable = !m_rimEnable;
                if (m_rimEnable)
                {
                    MessageBoxA(NULL, "Enabling", "INFO", MB_OK);
                }
                else
                {
                    MessageBoxA(NULL, "Disabling", "INFO", MB_OK);
                }
            }
        }
    }

};

VSI_MAIN(VSIDemoRimLighting)