#include <gl\glew.h>
#include <VSIUtilPlus.h>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\matrix_transform.hpp>

#define MAX_RECURSION_DEPTH 5
#define MAX_FB_WIDTH 2048
#define MAX_FB_HEIGHT 1096

enum
{
    MAX_SCENE_WIDTH = 2048,
    MAX_SCENE_HEIGHT = 2048
};
using namespace VSIUtil;

class VSIDemoSiliconAtom : public VSIUtilPlus
{
    struct
    {
        GLint   ray_origin;
        GLint   ray_lookat;
        GLint   aspect;
    }uniforms;

    struct sphere
    {
        glm::vec3   center;
        float       radius;
        glm::vec4   color;
    };

    struct light
    {
        glm::vec3   position;
        unsigned int    d;
    };

    struct uniforms_block
    {
        glm::mat4   mv_matrix;
        glm::mat4   view_matrix;
        glm::mat4   proj_matrix;
    };

    enum DEBUG_MODE
    {
        DEBUG_NONE,
        DEBUG_REFLECTED,
        DEBUG_REFRACTED,
        DEBUG_REFLECTED_COLOR,
        DEBUG_REFRACTED_COLOR
    };

    int         max_depth;
    int         debug_depth;
    DEBUG_MODE  debug_mode;
    bool        paused;

public:
    GLuint m_progObjPrepareTracing;
    GLuint m_progObjTraceProgram;
    GLuint m_progObjBlitProgram;

    GLuint  vao;
    GLuint  tex_composite;
    GLuint  ray_fbo[MAX_RECURSION_DEPTH];
    GLuint  tex_position[MAX_RECURSION_DEPTH];
    GLuint  tex_reflected[MAX_RECURSION_DEPTH];
    GLuint  tex_reflection_intensity[MAX_RECURSION_DEPTH];
    GLuint  tex_refracted[MAX_RECURSION_DEPTH];
    GLuint  tex_refraction_intensity[MAX_RECURSION_DEPTH];

    GLuint  uniforms_buffer;
    GLuint  sphere_buffer;
    GLuint  light_buffer;

    /* Bloom variables*/
    GLuint render_fbo;
    GLuint tex_scene;
    GLuint tex_brightpass;
    GLuint tex_depth;

    GLuint filter_fbo[2];
    GLuint tex_filter[2];

    GLuint tex_lut;

    GLuint m_progObjBloomScene;
    GLuint m_progObjBloomFilter;
    GLuint m_progObjBloomResolve;

    GLuint mResolve_Exposure;
    /******************/

    VSIDemoSiliconAtom()
        : max_depth(2),
        debug_depth(0),
        debug_mode(DEBUG_NONE),
        paused(false)
    {

    }

    ~VSIDemoSiliconAtom()
    {

    }

    void WindowInit()
    {
        WCHAR tempName[128] = TEXT("VSI Demo Silicon Atom.");
        wcscpy_s(mszAppName, wcslen(tempName) + 1, tempName);
    }

    std::vector<glm::vec3>& VSIUtilGetVertices()
    {
        return std::vector<glm::vec3>();
    }

    std::vector<glm::vec3>& VSIUtilGetNormals()
    {
        return std::vector<glm::vec3>();
    }

    std::vector<glm::vec2>& VSIUtilGetTexcoords()
    {
        return std::vector<glm::vec2>();
    }

    std::vector<glm::vec3>& VSIUtilGetTangets()
    {
        return std::vector<glm::vec3>();
    }

    void VSIUtilSceneInit()
    {
        LoadShaders();
        LoadBloomShaders();

        glGenBuffers(1, &uniforms_buffer);
        glBindBuffer(GL_UNIFORM_BUFFER, uniforms_buffer);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(uniforms_block), NULL, GL_DYNAMIC_DRAW);

        glGenBuffers(1, &sphere_buffer);
        glBindBuffer(GL_UNIFORM_BUFFER, sphere_buffer);
        glBufferData(GL_UNIFORM_BUFFER, 128 * sizeof(sphere), NULL, GL_DYNAMIC_DRAW);

        glGenBuffers(1, &light_buffer);
        glBindBuffer(GL_UNIFORM_BUFFER, light_buffer);
        glBufferData(GL_UNIFORM_BUFFER, 128 * sizeof(sphere), NULL, GL_DYNAMIC_DRAW);

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenFramebuffers(MAX_RECURSION_DEPTH, ray_fbo);
        glGenTextures(1, &tex_composite);
        glGenTextures(MAX_RECURSION_DEPTH, tex_position);
        glGenTextures(MAX_RECURSION_DEPTH, tex_reflected);
        glGenTextures(MAX_RECURSION_DEPTH, tex_refracted);
        glGenTextures(MAX_RECURSION_DEPTH, tex_reflection_intensity);
        glGenTextures(MAX_RECURSION_DEPTH, tex_refraction_intensity);

        glBindTexture(GL_TEXTURE_2D, tex_composite);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB16F, MAX_FB_WIDTH, MAX_FB_HEIGHT);

        for (int i = 0; i < MAX_RECURSION_DEPTH; i++)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, ray_fbo[i]);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex_composite, 0);

            glBindTexture(GL_TEXTURE_2D, tex_position[i]);
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB32F, MAX_FB_WIDTH, MAX_FB_HEIGHT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, tex_position[i], 0);

            glBindTexture(GL_TEXTURE_2D, tex_reflected[i]);
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB16F, MAX_FB_WIDTH, MAX_FB_HEIGHT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, tex_reflected[i], 0);

            glBindTexture(GL_TEXTURE_2D, tex_refracted[i]);
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB16F, MAX_FB_WIDTH, MAX_FB_HEIGHT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, tex_refracted[i], 0);

            glBindTexture(GL_TEXTURE_2D, tex_reflection_intensity[i]);
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB16F, MAX_FB_WIDTH, MAX_FB_HEIGHT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, tex_reflection_intensity[i], 0);

            glBindTexture(GL_TEXTURE_2D, tex_refraction_intensity[i]);
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB16F, MAX_FB_WIDTH, MAX_FB_HEIGHT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, tex_refraction_intensity[i], 0);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);

        /*Bloom code */
        //static const GLenum bloombuffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
        static const GLenum bloombuffers[] = { GL_COLOR_ATTACHMENT0 };

        static const GLfloat exposureLUT[20] = {11.0f, 6.0f, 3.2f, 2.8f, 2.2f, 1.90f, 1.80f, 1.80f, 1.70f, 1.70f,  1.60f, 1.60f, 1.50f, 1.50f, 1.40f, 1.40f, 1.30f, 1.20f, 1.10f, 1.00f };

        glGenFramebuffers(1, &render_fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, render_fbo);
        
        //glGenTextures(1, &tex_scene);
        //glBindTexture(GL_TEXTURE_2D, tex_scene);
        //glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, MAX_SCENE_WIDTH, MAX_SCENE_HEIGHT);
        //glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex_scene, 0);
        //
        glGenTextures(1, &tex_brightpass);
        glBindTexture(GL_TEXTURE_2D, tex_brightpass);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, MAX_SCENE_WIDTH, MAX_SCENE_HEIGHT);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex_brightpass, 0);
        
        //glGenTextures(1, &tex_depth);
        //glBindTexture(GL_TEXTURE_2D, tex_depth);
        //glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT32F, MAX_SCENE_WIDTH, MAX_SCENE_HEIGHT);
        //glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, tex_depth, 0);
        glDrawBuffers(1, bloombuffers);

        glGenFramebuffers(2, &filter_fbo[0]);
        glGenTextures(2, &tex_filter[0]);
        for (int i = 0; i < 2; i++)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, filter_fbo[i]);
            glBindTexture(GL_TEXTURE_2D, tex_filter[i]);
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, i ? MAX_SCENE_WIDTH : MAX_SCENE_HEIGHT, i ? MAX_SCENE_HEIGHT : MAX_SCENE_WIDTH);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex_filter[i], 0);
            glDrawBuffers(1, bloombuffers);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glGenTextures(1, &tex_lut);
        glBindTexture(GL_TEXTURE_1D, tex_lut);
        glTexStorage1D(GL_TEXTURE_1D, 1, GL_R32F, 20);
        glTexSubImage1D(GL_TEXTURE_1D, 0, 0, 20, GL_RED, GL_FLOAT, exposureLUT);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

        /***************/
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);

}

    void VSIUtilRender()
    {
        float aspect;
        static bool flag = true;
        if (flag)
        {
            Sleep(1000);
            flag = false;
        }
        static const GLfloat zeros[] = { 0.0f, 0.0f, 0.0f, 0.0f };
        static const GLfloat gray[] = { 0.1f, 0.1f, 0.1f, 0.0f };
        static const GLfloat onesp[] = { 1.0f };
        static double last_time = 0.0;
        static double total_time = 0.0;
        static double c = 0.0f;


        glClearBufferfv(GL_COLOR, 0, zeros);
        glClearBufferfv(GL_DEPTH, 0, onesp);

        total_time += c - last_time;
        last_time = c;

        c += 0.05f;
        float f = (float)total_time;

        glm::vec3 view_position = glm::vec3(0.0, 0.0, -26.0f);
        glm::vec3 lookat_position = glm::vec3(0.0, 0.0, 0.0);
        glm::mat4 view_matrix = glm::lookAt(view_position,
            lookat_position,
            glm::vec3(0.0, 1.0, 0.0));

        glBindBufferBase(GL_UNIFORM_BUFFER, 0, uniforms_buffer);
        uniforms_block * block = (uniforms_block *)glMapBufferRange(GL_UNIFORM_BUFFER,
            0,
            sizeof(uniforms_block),
            GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

        glm::mat4 model_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(7.0, 7.0, 7.0));

        block->mv_matrix = view_matrix * model_matrix;
        block->view_matrix = view_matrix;
        block->proj_matrix = glm::perspective(50.0f,
            (float)mWidth / (float)mHeight,
            0.1f,
            1000.0f);

        glUnmapBuffer(GL_UNIFORM_BUFFER);

        glBindBufferBase(GL_UNIFORM_BUFFER, 1, sphere_buffer);
        sphere * s = (sphere *)glMapBufferRange(GL_UNIFORM_BUFFER, 0, 128 * sizeof(sphere), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

        int i;
        static float theta = 0;
        static float alpha = 10;
        static float beta = 20;
        static float gamma = 30;

        /* Code for nucleus sphere */
        {
            i = 1;
            float fi = (float)i / 128.0f;
            s[i].center = glm::vec3(0.0, 0.0, -7.0);
            s[i].radius = fi * 2.3f + 3.0f;
            float r = 0.0f;
            float g = 0.0f;
            float b = 0.0f;
            r = (r - floorf(r)) * 0.8f + 0.2f;
            g = (g - floorf(g)) * 0.8f + 0.2f;
            b = (b - floorf(b)) * 0.8f + 0.2f;
            s[i].color = glm::vec4(r, g, b, 1.0f);
        }
        /* Code for electron one*/
        {
            i = 0;
            float fi = (float)i / 128.0f;
            s[i].center = glm::vec3(cos(theta) * 4.5, sin(theta) * 4.5, -7.0);
            s[i].radius = fi * 2.3f + 1.0f;
            float r = fi * 61.0f;
            float g = r + 0.25f;
            float b = g + 0.25f;
            //r = (r - floorf(r)) * 0.8f + 0.2f;
            //g = (g - floorf(g)) * 0.8f + 0.2f;
            //b = (b - floorf(b)) * 0.8f + 0.2f;
            s[i].color = glm::vec4(r, g, b, 1.0f);
        }
        /* Code for electron two */
        {
            i = 2;
            float fi = (float)i / 128.0f;
            s[i].center = glm::vec3(0.0, sin(alpha) * 4.5, (cos(alpha) * 4.5) - 7.0);
            s[i].radius = fi * 2.3f + 1.0f;
            float r = fi * 1.0f;
            float g = r + 0.25f;
            float b = g + 0.25f;
            //r = (r - floorf(r)) * 0.8f + 0.2f;
            //g = (g - floorf(g)) * 0.8f + 0.2f;
            //b = (b - floorf(b)) * 0.8f + 0.2f;
            s[i].color = glm::vec4(r, g, b, 1.0f);
        }
        /* code for electron three*/
        {
            i = 3;
            float fi = (float)i / 128.0f;
            s[i].center = glm::vec3(sin(beta) * 4.5, 0.0, (cos(beta) * 4.5) - 7.0);
            s[i].radius = fi * 2.3f + 1.0f;
            float r = fi * 1.0f;
            float g = r + 0.25f;
            float b = g + 0.25f;
            //r = (r - floorf(r)) * 0.8f + 0.2f;
            //g = (g - floorf(g)) * 0.8f + 0.2f;
            //b = (b - floorf(b)) * 0.8f + 0.2f;
            s[i].color = glm::vec4(r, g, b, 1.0f);
        }
        /* code for electron four*/
        {
            i = 4;
            float fi = (float)i / 128.0f;
            s[i].center = glm::vec3(sin(gamma) * 4.5, (cos(gamma) * 4.5), ((sin(gamma) * 4.5) + (cos(gamma) * 4.5)) - 7.0);
            s[i].radius = fi * 2.3f + 1.0f;
            float r = fi * 1.0f;
            float g = r + 0.25f;
            float b = g + 0.25f;
            //r = (r - floorf(r)) * 0.8f + 0.2f;
            //g = (g - floorf(g)) * 0.8f + 0.2f;
            //b = (b - floorf(b)) * 0.8f + 0.2f;
            s[i].color = glm::vec4(r, g, b, 1.0f);
        }



        theta += 0.01f;
        if (theta >= 360)
            theta = 0.0;

        alpha += 0.01f;
        if (alpha >= 360)
            alpha = 0.0;


        beta += 0.01f;
        if (beta >= 360)
            beta = 0.0;


        gamma += 0.01f;
        if (gamma >= 360)
            gamma = 0.0;

        glUnmapBuffer(GL_UNIFORM_BUFFER);

        glBindBufferBase(GL_UNIFORM_BUFFER, 3, light_buffer);
        light * l = (light*)glMapBufferRange(GL_UNIFORM_BUFFER, 0, 128 * sizeof(light), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

        f *= 1.0f;

        //for (i = 0; i < 2; i++)
        {
        //    float fi = 3.33f - (float)i;
            //l[i].position = glm::vec3(sinf(fi * 2.0f - f) * 15.75f,
            //    cosf(fi * 5.0f - f) * 5.75f,
            //    (sinf(fi * 3.0f - f) * cosf(fi * 2.5f - f)) * 19.4f);
            //l[0].position = glm::vec3(0.0, sin(alpha+3) * 4.5, (cos(alpha+3) * 4.5) - 7.0);
            //l[1].position = glm::vec3(cos(theta+5) * 4.5, sin(theta+5) * 4.5, -7.0);
            l[0].position = glm::vec3(-5.0, 0.0, 5.0);
            //l[1].position = glm::vec3(10.0, 0.0, 0.0);
        }

        glUnmapBuffer(GL_UNIFORM_BUFFER);

        glBindVertexArray(vao);
        //glViewport(0, 0, mWidth, mHeight);
        //aspect = (float)mWidth / (float)mHeight;
        //mProjMatrix = glm::perspective(45.0f,
        //    aspect,
        //    0.1f,
        //    1000.0f);

        glUseProgram(m_progObjPrepareTracing);
        glUniformMatrix4fv(uniforms.ray_lookat, 1, GL_FALSE, glm::value_ptr(view_matrix));
        glUniform3fv(uniforms.ray_origin, 1, glm::value_ptr(view_position));
        glUniform1f(uniforms.aspect, (float)mHeight / (float)mWidth);
        glBindFramebuffer(GL_FRAMEBUFFER, ray_fbo[0]);
        static const GLenum draw_buffers[] =
        {
            GL_COLOR_ATTACHMENT0,
            GL_COLOR_ATTACHMENT1,
            GL_COLOR_ATTACHMENT2,
            GL_COLOR_ATTACHMENT3,
            GL_COLOR_ATTACHMENT4,
            GL_COLOR_ATTACHMENT5
        };
        glDrawBuffers(6, draw_buffers);

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glUseProgram(m_progObjTraceProgram);
        recurse(0);

        /* Bloom code*/
        glUseProgram(m_progObjBloomScene);

        glBindFramebuffer(GL_FRAMEBUFFER, render_fbo);
        //glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDrawBuffer(GL_BACK);
       
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_composite);
        
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        
        //return;
        glDisable(GL_DEPTH_TEST);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glUseProgram(m_progObjBloomFilter);
        
        glBindFramebuffer(GL_FRAMEBUFFER, filter_fbo[0]);
        glBindTexture(GL_TEXTURE_2D, tex_brightpass);
       // glViewport(0, 0, mHeight, mWidth);
       // aspect = (float)mWidth / (float)mHeight;
       // mProjMatrix = glm::perspective(45.0f,
       //     aspect,
       //     0.1f,
       //     1000.0f);

        
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, filter_fbo[1]);
        glBindTexture(GL_TEXTURE_2D, tex_filter[0]);
        //glViewport(0, 0, mWidth, mHeight);
        //aspect = (float)mWidth / (float)mHeight;
        //mProjMatrix = glm::perspective(45.0f,
        //    aspect,
        //    0.1f,
        //    1000.0f);

        
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        
        glUseProgram(m_progObjBloomResolve);
        
        static GLfloat e = 0.0;
        glUniform1f(mResolve_Exposure, e);
        e += 0.001;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_filter[1]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, tex_composite);
        
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        
        return;
        /***********BLOOM CODE***********/
        glUseProgram(m_progObjBlitProgram);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDrawBuffer(GL_BACK);

        glActiveTexture(GL_TEXTURE0);

        switch (debug_mode)
        {
        case DEBUG_NONE:
            glBindTexture(GL_TEXTURE_2D, tex_brightpass);
            break;
        }

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }

    void recurse(int depth)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, ray_fbo[depth + 1]);

        static const GLenum draw_buffers[] =
        {
            GL_COLOR_ATTACHMENT0,
            GL_COLOR_ATTACHMENT1,
            GL_COLOR_ATTACHMENT2,
            GL_COLOR_ATTACHMENT3,
            GL_COLOR_ATTACHMENT4,
            GL_COLOR_ATTACHMENT5
        };

        glDrawBuffers(6, draw_buffers);

        glEnablei(GL_BLEND, 0);
        glBlendFunci(0, GL_ONE, GL_ONE);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_position[depth]);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, tex_reflected[depth]);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, tex_reflection_intensity[depth]);

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        if (depth != (max_depth - 1))
        {
            recurse(depth + 1);
        }

        glDisablei(GL_BLEND, 0);
    }

    void VSIUtilMessageHandler(UINT iMsg, WPARAM wParam, LPARAM lParam)
    {
    }

    void LoadBloomShaders()
    {
        m_progObjBloomScene = VSIUtilLoadShaders("VSIDemoSiliconAtomBloomScene.vs.glsl", "VSIDemoSiliconAtomBloomScene.fs.glsl");
        m_progObjBloomFilter = VSIUtilLoadShaders("VSIDemoSiliconAtomBloomFilter.vs.glsl", "VSIDemoSiliconAtomBloomFilter.fs.glsl");

        m_progObjBloomResolve = VSIUtilLoadShaders("VSIDemoSiliconAtomResolve.vs.glsl", "VSIDemoSiliconAtomResolve.fs.glsl");
        mResolve_Exposure = glGetUniformLocation(m_progObjBloomResolve, "exposure");
    }

    void LoadShaders()
    {
        m_progObjPrepareTracing = VSIUtilLoadShaders("VSIDemoSiliconAtomRayTracingPrepare.vs.glsl", "VSIDemoSiliconAtomRayTracingPrepare.fs.glsl");

        uniforms.ray_origin = glGetUniformLocation(m_progObjPrepareTracing, "ray_origin");
        uniforms.ray_lookat = glGetUniformLocation(m_progObjPrepareTracing, "ray_lookat");
        uniforms.aspect = glGetUniformLocation(m_progObjPrepareTracing, "aspect");

        m_progObjTraceProgram = VSIUtilLoadShaders("VSIDemoSiliconAtomTraceProgram.vs.glsl", "VSIDemoSiliconAtomTraceProgram.fs.glsl");

        m_progObjBlitProgram = VSIUtilLoadShaders("VSIDemoSiliconAtomBlitProgram.vs.glsl", "VSIDemoSiliconAtomBlitProgram.fs.glsl");


    }
};

VSI_MAIN(VSIDemoSiliconAtom);