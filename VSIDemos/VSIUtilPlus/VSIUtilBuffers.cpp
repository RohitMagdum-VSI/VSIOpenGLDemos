#include <gl\glew.h>
#include <VSIUtilPlus.h>
#include <stdio.h>
namespace VSIUtil
{
    void VSIUtilPlus::VSIUtilGenAndBindBuffer(GLuint buffer, 
        GLuint attribLocation, 
        BufferType bufferType)
    {

        std::vector<glm::vec3>& vertices = VSIUtilGetVertices();
        std::vector<glm::vec3>& normals = VSIUtilGetNormals();
        std::vector<glm::vec2>& textureCoords = VSIUtilGetTexcoords();
        std::vector<glm::vec3>& tangents = VSIUtilGetTangents();

        glGenBuffers(1, &buffer);
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        switch (bufferType)
        {
        case VERTEX:
            glBufferData(GL_ARRAY_BUFFER, (vertices.size() * sizeof(glm::vec3)), &(vertices[0].x), GL_STATIC_DRAW);
            break;
        case NORMALS:
            glBufferData(GL_ARRAY_BUFFER, (normals.size() * sizeof(glm::vec3)), &(normals[0].x), GL_STATIC_DRAW);
            break;
        case TEXTURECOORDS:
            glBufferData(GL_ARRAY_BUFFER, (textureCoords.size() * sizeof(glm::vec2)), &(textureCoords[0].x), GL_STATIC_DRAW);
            break;
        case TANGENTS:
            glBufferData(GL_ARRAY_BUFFER, tangents.size() * sizeof(glm::vec3), &(tangents[0].x), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(attribLocation);
       // glBindBuffer(GL_ARRAY_BUFFER, buffer);
        glVertexAttribPointer(attribLocation, (bufferType == 2) ? 2 : 3, GL_FLOAT, GL_FALSE, 0, NULL);
    }
}