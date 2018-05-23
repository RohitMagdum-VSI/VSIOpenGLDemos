#include <stdio.h>
#include <gl\glew.h>
#include <VSIUtilPlus.h>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\matrix_transform.hpp>

using namespace VSIUtil;

class VSICameraRotation :public VSIUtilPlus
{
public:
    GLuint proj_location;
    GLuint mv_location;
    GLuint progObj;
    GLuint cube_buffer[3];
    GLuint vao;
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec2> textureCoords;
    std::vector<glm::vec3> normals;
    GLuint m_texture;

    VSICameraRotation()
    {
    }

    ~VSICameraRotation()
    {
    }

    void WindowInit()
    {
        WCHAR tempName[128] = TEXT("VSI Camera Rotation For Sneha's Demo");
        wcscpy_s(mszAppName, wcslen(tempName) + 1, TEXT("VSI Camera Rotation For Sneha's Demo"));
    }

    std::vector<glm::vec3>& VSIUtilGetVertices()
    {
        return vertices;
    }

    std::vector<glm::vec3>& VSIUtilGetNormals()
    {
        return normals;
    }

    std::vector<glm::vec2>& VSIUtilGetTexcoords()
    {
        return textureCoords;
    }

    std::vector<glm::vec3>& VSIUtilGetTangents()
    {
        return std::vector<glm::vec3>();
    }

    void VSIUtilMessageHandler(UINT iMsg, WPARAM wParam, LPARAM lParam)
    {

    }

    void VSIUtilSceneInit()
    {
        VSIUtilLoadMesh("Teapot.obj", VERTEX_AND_NORMAL);

        progObj = VSIUtilLoadShaders("VSICameraRotation.vs.glsl", "VSICameraRotation.fs.glsl");

        proj_location = glGetUniformLocation(progObj, "proj_matrix");
        mv_location = glGetUniformLocation(progObj, "mv_matrix");

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        VSIUtilGenAndBindBuffer(cube_buffer[0], 0, VERTEX);

        //       VSIUtilGenAndBindBuffer(cube_buffer[1], 1, TEXTURECOORDS);

        VSIUtilGenAndBindBuffer(cube_buffer[2], 2, NORMALS);

      //  VSIUtilLoadTexture("earth_orig.jpg", &m_texture);
        glBindVertexArray(0);
    }

    void VSIUtilRender()
    {
        static const GLfloat green[] = { 0.0f, 0.0f, 0.0f, 1.0f };
        static const GLfloat one = 1.0f;

        glBindVertexArray(vao);
        glUseProgram(progObj);

        glClearBufferfv(GL_COLOR, 0, green);
        glClearBufferfv(GL_DEPTH, 0, &one);

        static GLfloat f = 0.0f;
        f += 0.001f;

        glm::mat4 viewMatrix = glm::lookAt(glm::vec3(10.0f * sinf(f), 1.0, 10.0f* (cosf(f) + cosf(f))),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0, 1.0f, 0.0f));

        glm::mat4 mvMatrix = viewMatrix;// *glm::rotate(glm::mat4(1.0f), f * 50.0f, glm::vec3(1.0f, 0.0f, 0.0f))*
            //glm::rotate(glm::mat4(1.0f), f * 130.0f, glm::vec3(0.0f, 1.0f, 0.0f)) *
            //glm::translate(glm::mat4(1.0f), glm::vec3(f * 4.0f, -4.0f, 0.0f));

        glUniformMatrix4fv(mv_location, 1, GL_FALSE, glm::value_ptr(mvMatrix));
        glUniformMatrix4fv(proj_location, 1, GL_FALSE, glm::value_ptr(mProjMatrix));

        glDrawArrays(GL_TRIANGLES, 0, vertices.size());
    }

};

VSI_MAIN(VSICameraRotation)