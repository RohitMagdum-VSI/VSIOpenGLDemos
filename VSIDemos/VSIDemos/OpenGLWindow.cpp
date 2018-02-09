#include <stdio.h>
#include <gl\glew.h>
#include <VSIUtilPlus.h>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\matrix_transform.hpp>

using namespace VSIUtil;

class VSIDemo:public VSIUtilPlus
{
public:
    GLuint time_location;
    GLuint proj_location;
    GLuint mv_location;
    GLuint progObj;
    GLuint cube_buffer[3];
    GLuint vao;
    GLfloat m_angle;
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texCoords;
    std::vector<glm::vec3> tangents;
    GLuint m_texture;
    
    VSIDemo()
    {
        m_angle = 0.0f;
    }

    ~VSIDemo()
    {
    }

    void WindowInit()
    {
        WCHAR tempName[128] = TEXT("VSI Demo Window");
        wcscpy_s(mszAppName,wcslen(tempName) + 1, TEXT("VSI Demo Window"));
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
        return texCoords;
    }


    std::vector<glm::vec3>& VSIUtilGetTangents()
    {
        return tangents;
    }

    void VSIUtilMessageHandler(UINT iMsg, WPARAM wParam, LPARAM lParam)
    {
        switch (iMsg)
        {
        case WM_CHAR:
            switch (wParam)
            {
            case 'w':
                m_angle += 1.0f;
                break;
            case 's':
                m_angle -= 1.0f;
            }
        }
    }

    void VSIUtilSceneInit()
    {
        VSIUtilLoadMesh("sphere_zup.obj", VERTEX_NORMAL_AND_TEXTURE);
        
        progObj = VSIUtilLoadShaders("VSIDemo.vs.glsl", "VSIDemo.fs.glsl");
    
        proj_location = glGetUniformLocation(progObj, "proj_matrix");
        mv_location = glGetUniformLocation(progObj, "mv_matrix");
    
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
    
        VSIUtilGenAndBindBuffer(cube_buffer[0], 0, VERTEX);

        VSIUtilGenAndBindBuffer(cube_buffer[1], 1, TEXTURECOORDS);
        
        VSIUtilGenAndBindBuffer(cube_buffer[2], 2, NORMALS);
    
        VSIUtilLoadTexture("earth_orig.jpg", &m_texture);
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
        f += 0.0001f;
    
        glm::mat4 mv_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.0, -20.0f)) *
            glm::rotate(glm::mat4(1.0f), f * 45.0f, glm::vec3(0.0, 1.0f, 0.0)) *
            glm::rotate(glm::mat4(1.0f), 5.0f, glm::vec3(0.0, 0.0f, 1.0));
        glUniformMatrix4fv(mv_location, 1, GL_FALSE, glm::value_ptr(mv_matrix));
        glUniformMatrix4fv(proj_location, 1, GL_FALSE, glm::value_ptr(mProjMatrix));

        glDrawArrays(GL_TRIANGLES, 0, vertices.size());
    }

};

VSI_MAIN(VSIDemo)