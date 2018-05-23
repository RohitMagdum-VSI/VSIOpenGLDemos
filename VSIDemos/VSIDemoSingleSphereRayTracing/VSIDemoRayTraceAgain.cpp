#include <gl\glew.h>
#include <VSIUtilPlus.h>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\matrix_transform.hpp>

using namespace VSIUtil;

class VSIDemoSingleSphereRayTracing : public VSIUtilPlus
{
public:

    struct ray
    {
        glm::vec3 origin;
        float f;
        glm::vec3 direction;
    };

    GLuint ray_buffer;
    std::vector<glm::vec3> m_vertices;
    std::vector<glm::vec3> m_normals;
    std::vector<glm::vec2> m_textures;
    std::vector<glm::vec3> m_tangents;
    GLuint progObj;
    GLuint progObj_2;
    GLuint proj_location;
    GLuint proj_location_2;
    GLuint mv_location;
    GLuint mv_location_2;
    GLuint vao[2];
    GLuint buffers[3];
    GLuint m_texture;
    GLuint line_buffer;
    GLuint fbo;
    GLuint color_texture_fbo;

    void WindowInit()
    {
        WCHAR tempName[128] = TEXT("VSI Single Sphere Ray Tracing");
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
        
        progObj_2 = VSIUtilLoadShaders("VSIDemoSimple.vs.glsl", "VSIDemoSimple.fs.glsl");
        progObj = VSIUtilLoadShaders("VSIDemoRayTraceAgain.vs.glsl", "VSIDemoRayTraceAgain.fs.glsl");
        proj_location_2 = glGetUniformLocation(progObj_2, "proj_matrix");
        mv_location_2 = glGetUniformLocation(progObj_2, "mv_matrix");

        glShadeModel(GL_SMOOTH);
        /* Set up depth buffer*/
        glClearDepth(1.0f);
        // enable depth testing
        glEnable(GL_DEPTH_TEST);
        // depth test to do
        glDepthFunc(GL_LEQUAL);
        // set really nice perspective calculations.
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
        // we will always cull back faces for better performance
        glEnable(GL_CULL_FACE);

        glGenBuffers(1, &ray_buffer);
        glBindBuffer(GL_UNIFORM_BUFFER, ray_buffer);
        glBufferData(GL_UNIFORM_BUFFER, 4000 * sizeof(ray), NULL, GL_DYNAMIC_DRAW);
        
        VSIUtilLoadTexture("earth_orig.jpg", &m_texture);

        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
       
        glGenTextures(1, &color_texture_fbo);
        glBindTexture(GL_TEXTURE_2D, color_texture_fbo);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB16F, 1024, 1024);
       
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color_texture_fbo, 0);
       
        static const GLenum draw_buffer[] = { GL_COLOR_ATTACHMENT0 };
        glDrawBuffers(1, draw_buffer);
    }

    void VSIUtilMessageHandler(UINT iMsg, WPARAM wParam, LPARAM lParam)
    {
        
    }

    void VSIUtilRender()
    {
        static const GLfloat green[] = { 0.0f, 0.0f, 0.0f, 1.0f };
        static const GLfloat one = 1.0f;


        glBindBufferBase(GL_UNIFORM_BUFFER, 0, ray_buffer);
        ray * r = (ray*)glMapBufferRange(GL_UNIFORM_BUFFER, 0, 4000 * sizeof(ray), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

        int i;

        static float k = 0.0f;
        for (i = 0; i < 4000; i++)
        {
            r[i].origin = glm::vec3(0.0, -2.0 + k, -4.0);
            r[i].f = 1.0f;
            r[i].direction = glm::vec3(0.0, -2.0 + k, 4.0f);
            k += 0.01;
        }
        
        glUnmapBuffer(GL_UNIFORM_BUFFER);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

//        glViewport(0, 0, 1024, 1024);
        glClearBufferfv(GL_COLOR, 0, green);
        glClearBufferfv(GL_DEPTH, 0, &one);

        glUseProgram(progObj_2);
        //  glPointSize(10.0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_texture);

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

  //      glViewport(0, 0, 1024, 1024);
        glClearBufferfv(GL_COLOR, 0, green);
        glClearBufferfv(GL_DEPTH, 0, &one);
        
        glUseProgram(0);
        glUseProgram(progObj);
        glActiveTexture(GL_TEXTURE0);

        glBindTexture(GL_TEXTURE_2D, color_texture_fbo);
  
//        glPointSize(10);
        
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    
        glBindTexture(GL_TEXTURE_2D, 0);
    }
};

VSI_MAIN(VSIDemoSingleSphereRayTracing)