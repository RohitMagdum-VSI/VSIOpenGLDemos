#pragma once

#include <Windows.h>
#include <glm\mat4x4.hpp>
#include <vector>
#include <gl\GL.h>
#include <gl\GLU.h>
#include <VSIUtilModelObject.h>

#ifdef VSIUTILPLUS_EXPORTS
#define VSIUTILPLUS_API __declspec(dllexport)
#else
#define VSIUTILPLUS_API __declspec(dllimport)
#endif

#define BUFFER_OFFSET(i) ((void*)(i))

#define MAX_NAME_LENGTH 2048

#define DEFAULT_WIDTH 1024
#define DEFAULT_HEIGHT 780
#define MAX_TEXTURES 100

namespace VSIUtil
{
    class VSIUtilPlus
    {
    public:
        HWND mhWnd;
        HINSTANCE mhInstance;
        HDC mhDC;
        HGLRC mhRC;
        bool mbDone;
        bool mbIsActiveWindow;
        bool mbIsEscapeKeyPressed;
        bool mbFullscreen;
        DWORD mdwStyle;
        WINDOWPLACEMENT mwpPrev;
        unsigned long mWidth;
        unsigned long mHeight;
        /*********Model related Variables *********/
        VSIUtilModelObject *m_CurrentModel;
        /******************************************/

        /* Windowing related apis*/
        void Initialize();
        void Uninitialize();
        void ToggleFullscreen();
        /************************/

        /** Shaders related Apis */
        GLubyte* loadShaderFromFile(char*);
        unsigned int getFileLength(std::ifstream&);
        int compileShader(GLubyte* source, GLenum shaderType, GLuint *shader);
        int linkShader(GLuint);
        /************************/
        
    public:
        glm::mat4 mProjMatrix;

        enum BufferType
        {
            VERTEX = 0,
            NORMALS,
            TEXTURECOORDS,
            TANGENTS
        };

        enum MeshType
        {
            UNKNOWN,
            ONLY_VERTEX,
            VERTEX_AND_NORMAL,
            VERTEX_AND_TEXTURE,
            VERTEX_NORMAL_AND_TEXTURE
        };

        /************************************/
        wchar_t mszAppName[512];

        VSIUTILPLUS_API WPARAM VSIUtilRun(HINSTANCE hInstance,
            HINSTANCE hPrevInstance,
            LPSTR lpCmdLine,
            int nCmdShow);

        static LRESULT CALLBACK mWndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam);
        VSIUTILPLUS_API virtual void VSIUtilResize(unsigned long, unsigned long);
        VSIUTILPLUS_API virtual void WindowInit();
        /************************************/

        /****Constructors and Destructors****/
        VSIUTILPLUS_API VSIUtilPlus();
        VSIUTILPLUS_API ~VSIUtilPlus();
        /************************************/


        VSIUTILPLUS_API virtual void VSIUtilRender();
        VSIUTILPLUS_API virtual void VSIUtilUpdate();
        VSIUTILPLUS_API virtual void VSIUtilSceneInit();
        VSIUTILPLUS_API virtual void VSIUtilSceneDeinit();

        /*************Mesh Loading apis. *****************/
        VSIUTILPLUS_API void VSIUtilLoadMesh(char*, MeshType);
        VSIUTILPLUS_API void VSIUtilCalculateTangents();
        /*************************************************/

        /*************Shader related apis.****************/
        VSIUTILPLUS_API int VSIUtilLoadShaders(char*, char*);
        VSIUTILPLUS_API int VSIUtilLoadShaders(char*, char*, char*, char*);
        /*************************************************/

        /*************Texture related apis.***************/
        VSIUTILPLUS_API int VSIUtilLoadTexture(char*, GLuint *texture);
        /*************************************************/

        /*************Cube mapping related apis. *********/
        VSIUTILPLUS_API void VSIUtilLoadCubeMapTextures(std::vector<std::string>, GLuint*);

        /*************Buffer related apis.****************/
        VSIUTILPLUS_API void VSIUtilGenAndBindBuffer(GLuint buffer, 
                                                             GLuint attribLocation, 
                                                             BufferType);
        /*************************************************/

        
        /************* Interfaces implement by Derived class ****************/
        VSIUTILPLUS_API virtual std::vector<glm::vec3>& VSIUtilGetVertices();
        VSIUTILPLUS_API virtual std::vector<glm::vec3>& VSIUtilGetNormals();
        VSIUTILPLUS_API virtual std::vector<glm::vec2>& VSIUtilGetTexcoords();
        VSIUTILPLUS_API virtual std::vector<glm::vec3>& VSIUtilGetTangents();
        VSIUTILPLUS_API virtual void VSIUtilMessageHandler(UINT, WPARAM, LPARAM) = 0;
        /*******************************************************************/
    };
}

#if defined WIN32
#define VSI_MAIN(ClassName)             \
int WINAPI WinMain(HINSTANCE hInstance, \
    HINSTANCE hPrevInstance,            \
    LPSTR lpCmdLine,                    \
    int nCmdShow)                       \
{                                       \
    ClassName *util = new ClassName();  \
    util->WindowInit();                 \
    WPARAM retVal = util->VSIUtilRun(   \
    hInstance,                          \
    hPrevInstance,                      \
    lpCmdLine,                          \
    nCmdShow);                          \
    delete(util);                       \
    util = NULL;                        \
    return (int)retVal;                 \
}
#endif