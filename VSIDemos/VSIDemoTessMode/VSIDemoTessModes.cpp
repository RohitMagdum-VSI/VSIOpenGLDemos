#include <gl\glew.h>
#include <VSIUtilPlus.h>

using namespace VSIUtil;

class VSIDemoTessModes:public VSIUtilPlus
{
public:
    GLuint m_VAO;
    GLuint m_progObj;

    VSIDemoTessModes()
    {

    }

    ~VSIDemoTessModes()
    {

    }

    void WindowInit()
    {
        WCHAR tempName[128] = TEXT("VSI Tessellation");
        wcscpy_s(mszAppName, wcslen(tempName) + 1, tempName);
    }

    void VSIUtilSceneInit()
    {
        m_progObj = VSIUtilLoadShaders("VSIDemoTess.vs.glsl", "VSIDemoTess.fs.glsl",
            "VSIDemoTess.Control.glsl",
            "VSIDemoTess.Evaluation.glsl");

        glGenVertexArrays(1, &m_VAO);
        glBindVertexArray(m_VAO);

        glPatchParameteri(GL_PATCH_VERTICES, 4);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }
    
    void VSIUtilRender()
    {
        const GLfloat black = 0.0f;// , 0.0f, 0.0f, 1.0f    };
        static const GLfloat one = 1.0f;
        glClearBufferfv(GL_COLOR, 0, &black);
        glClearBufferfv(GL_DEPTH, 0, &one);

        glUseProgram(m_progObj);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDrawArrays(GL_PATCHES, 0, 4);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    void VSIUtilMessageHandler(UINT iMsg, WPARAM wParam, LPARAM lParam)
    {

    }
};

VSI_MAIN(VSIDemoTessModes)