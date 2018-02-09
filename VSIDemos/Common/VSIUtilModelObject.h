#pragma once

#include <Windows.h>
#include <glm\mat4x4.hpp>
#include <vector>
#include <gl\GL.h>
#include <gl\GLU.h>

namespace VSIUtil
{
    class VSIUtilModelObject
    {
    public:
        std::vector<glm::vec3> m_vertices;
        std::vector<glm::vec3> m_normals;
        std::vector<glm::vec2> m_texCoords;
        std::vector<glm::vec3> m_tangents;

        glm::mat4 m_modelMatrix;
    };
}