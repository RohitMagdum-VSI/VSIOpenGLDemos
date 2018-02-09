#include <Windows.h>
#include <stdio.h>
#include <gl\glew.h>
#include <VSIUtilPlus.h>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <string>

using namespace VSIUtil;

class VSIDemoCubeMapping :public VSIUtilPlus
{
public:

    GLuint m_vao;
    GLuint m_progSkyboxObj;
    GLuint m_progTeapotObj;
    std::vector<glm::vec3> m_vertices;
    std::vector<glm::vec3> m_normals;
    std::vector<glm::vec2> m_textures;
    std::vector<glm::vec3> m_tangents;
    std::vector<std::string> cubeMapTexture;
    GLfloat m_xTrans, m_yTrans, m_zTrans;

    GLuint texCubeMap;
    GLuint m_mvSkyboxLocation;

    GLuint m_projTeapotLocation;
    GLuint m_mvTeapotLocation;
    GLuint m_teapotTexture;
    GLuint m_teapotBuffer[2];

    VSIDemoCubeMapping()
    {
        m_zTrans = -20.0f;
    }

    ~VSIDemoCubeMapping()
    {

    }

    void WindowInit()
    {
        WCHAR tempName[128] = TEXT("VSI Cube Mapping");
        wcscpy_s(mszAppName, wcslen(tempName) + 1, tempName);
    }

    std::vector<glm::vec3>& VSIUtilGetVertices()
    {
        return m_vertices;
    }

    std::vector<glm::vec3>& VSIUtilGetNormals()
    {
        return m_normals;
    }

    std::vector<glm::vec2>& VSIUtilGetTexcoords()
    {
        return m_textures;
    }

    std::vector<glm::vec3>& VSIUtilGetTangents()
    {
        return m_tangents;
    }

    void VSIUtilSceneInit()
    {
        cubeMapTexture.push_back("posx.jpg");
        cubeMapTexture.push_back("negx.jpg");
        cubeMapTexture.push_back("posy.jpg");
        cubeMapTexture.push_back("negy.jpg");
        cubeMapTexture.push_back("posz.jpg");
        cubeMapTexture.push_back("negz.jpg");

        VSIUtilLoadMesh("Teapot.obj", VERTEX_AND_NORMAL);

        m_progSkyboxObj = VSIUtilLoadShaders("VSICubeMapping.vs.glsl", "VSICubeMapping.fs.glsl");
        m_progTeapotObj = VSIUtilLoadShaders("VSICubeMapTeapot.vs.glsl", "VSICubeMapTeapot.fs.glsl");

        m_mvSkyboxLocation = glGetUniformLocation(m_progSkyboxObj, "mvMatrix");

        m_projTeapotLocation = glGetUniformLocation(m_progTeapotObj, "proj_matrix");
        m_mvTeapotLocation = glGetUniformLocation(m_progTeapotObj, "mv_matrix");

        glGenVertexArrays(1, &m_vao);
        glBindVertexArray(m_vao);
        
        VSIUtilGenAndBindBuffer(m_teapotBuffer[0], 0, VERTEX);
        VSIUtilGenAndBindBuffer(m_teapotBuffer[1], 1, NORMALS);

        VSIUtilLoadCubeMapTextures(cubeMapTexture, &texCubeMap);
        glActiveTexture(GL_TEXTURE0);
        glDepthFunc(GL_LEQUAL);

    }

    void VSIUtilRender()
    {
        static const GLfloat black[] = { 0.0f, 0.0f, 0.0f, 1.0f };
        static const GLfloat one = 1.0f;

        glClearBufferfv(GL_COLOR, 0, black);
        glClearBufferfv(GL_DEPTH, 0, &one);

        static GLfloat f = 0.0f;
        f += 0.00001f;

/***************** SKYBOX CUBE MAP*********************/
        glBindVertexArray(m_vao);

        glm::mat4 viewMatrix = glm::lookAt(glm::vec3(15.0f * sinf(f), 0.0f, 15.0f* cosf(f)),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0, 1.0f, 0.0f));

        glm::mat4 mvMatrix = viewMatrix *
            glm::rotate(glm::mat4(1.0f), f * 50.0f, glm::vec3(1.0f, 0.0f, 0.0f))*
            glm::rotate(glm::mat4(1.0f), f * 130.0f, glm::vec3(0.0f, 1.0f, 0.0f)) *
            glm::translate(glm::mat4(1.0f), glm::vec3(f * 4.0f, -4.0f, 0.0f));

        glUseProgram(m_progSkyboxObj);

        glUniformMatrix4fv(m_mvSkyboxLocation, 1, GL_FALSE, glm::value_ptr(mvMatrix));

        glDisable(GL_DEPTH_TEST);

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glUseProgram(0);
/******************************************************/

/*******************TEAPOT CODE************************/
        glUseProgram(m_progTeapotObj);


        glm::mat4 mvTeapotmatrix = glm::translate(glm::mat4(1.0f), glm::vec3(m_xTrans, m_yTrans, m_zTrans)) *
            glm::rotate(glm::mat4(1.0f), f * 45.0f, glm::vec3(1.0, 0.0f, 0.0)) *
            glm::rotate(glm::mat4(1.0f), 15.7f, glm::vec3(0.0, 0.0f, 1.0));

        glUniformMatrix4fv(m_mvTeapotLocation, 1, GL_FALSE, glm::value_ptr(mvTeapotmatrix));
        glUniformMatrix4fv(m_projTeapotLocation, 1, GL_FALSE, glm::value_ptr(mProjMatrix));
        glEnable(GL_DEPTH_TEST);

        glDrawArrays(GL_TRIANGLES, 0, m_vertices.size());
    }

    void VSIUtilMessageHandler(UINT iMsg, WPARAM wParam, LPARAM lParam)
    {
        switch (iMsg)
        {
        case WM_CHAR:
            switch (wParam)
            {
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

VSI_MAIN(VSIDemoCubeMapping)