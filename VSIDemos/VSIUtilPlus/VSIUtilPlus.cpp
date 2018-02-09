// VSIUtilPlus.cpp : Defines the exported functions for the DLL application.
//

#include <Windows.h>
#include <stdio.h>
#include <gl\glew.h>
#include <VSIUtilPlus.h>
#include <glm\vec4.hpp>
#include <glm\vec3.hpp>
#include <glm\mat2x4.hpp>
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <fstream>


namespace VSIUtil
{
    VSIUtilPlus::VSIUtilPlus()
    {
        mbDone = false;
        mbIsActiveWindow = true;
        mbIsEscapeKeyPressed = false;
        mbFullscreen = false;
        mwpPrev = { sizeof(WINDOWPLACEMENT) };
        m_CurrentModel = NULL;
    }
    VSIUtilPlus::~VSIUtilPlus()
    {
        if (mszAppName[0] != L'\0')
        {
            UnregisterClass(mszAppName, (HINSTANCE)GetModuleHandle(NULL));
            memset(mszAppName, 0, sizeof(mszAppName));
        }
    }
    void VSIUtilPlus::WindowInit()
    {
        wcscpy_s(mszAppName, wcslen(TEXT("Default OpenGL Window")) + 1, TEXT("Default OpenGL Window"));
    }

    WPARAM VSIUtilPlus::VSIUtilRun(HINSTANCE hInstance,
        HINSTANCE hPrevInstance,
        LPSTR lpCmdLine,
        int nCmdShow)
    {
        WNDCLASSEX wc;
        MSG msg;
        BOOL bRetVal;

        mhInstance = hInstance;

        wc.cbSize = sizeof(WNDCLASSEX);
        wc.cbClsExtra = 0;
        wc.cbWndExtra = 0;
        wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
        wc.lpszClassName = TEXT("VSIUtil");
        wc.lpszMenuName = NULL;
        wc.hCursor = LoadCursor(NULL, IDC_ARROW);
        wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
        wc.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
        wc.lpfnWndProc = mWndProc;
        wc.hInstance = hInstance;
        wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;

        if (!RegisterClassEx(&wc))
        {
            char str[1024];
            sprintf_s(str, "Error = %d\n", GetLastError());
            MessageBoxA(NULL, str, "ERROR", MB_OK | MB_ICONEXCLAMATION);
            exit(0);
        }

        if (!(mhWnd = CreateWindowEx(WS_EX_APPWINDOW,
            TEXT("VSIUtil"),
            mszAppName,
            WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
            0,
            0,
            DEFAULT_WIDTH,
            DEFAULT_HEIGHT,
            NULL,
            NULL,
            mhInstance,
            NULL
        )))
        {
            MessageBoxA(NULL, "Window Creation Error.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
            exit(0);
        }

        (void)SetWindowLongPtr(mhWnd, GWLP_USERDATA, (LONG_PTR)this);

        Initialize();

        bRetVal = ShowWindow(mhWnd, SW_SHOW);
        bRetVal = SetForegroundWindow(mhWnd);
        if (bRetVal == 0)
        {
            MessageBoxA(NULL, "Failed to get to foreground.", "Error", MB_OK);
        }
        
        HWND hwnd = SetFocus(mhWnd);
        if (hwnd == NULL)
        {
            MessageBoxA(NULL, "Failed to set focus.", "Error", MB_OK);
        }

        VSIUtilSceneInit();
        while (this->mbDone == false)
        {
            
            if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
            {
                if (msg.message == WM_QUIT)
                    this->mbDone = true;
                else
                {
                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }
            }
            else
            {
                if (this->mbIsActiveWindow == true)
                {
                
                    if (this->mbIsEscapeKeyPressed == true)
                    {
                        this->mbDone = true;
                    }
                    VSIUtilRender();
                    VSIUtilUpdate();
                    SwapBuffers(mhDC);
                }
            }
        }

        VSIUtilSceneDeinit();
        Uninitialize();

        return msg.wParam;
    }

    void VSIUtilPlus::Initialize()
    {
        int iPixelFormat;
        static PIXELFORMATDESCRIPTOR pfd;

        ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

        pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
        pfd.nVersion = 1;
        pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
        pfd.iPixelType = PFD_TYPE_RGBA;
        pfd.cColorBits = 32;
        pfd.cRedBits = 8;
        pfd.cGreenBits = 8;
        pfd.cBlueBits = 8;
        pfd.cAlphaBits = 8;
        pfd.cDepthBits = 32;

        if (!(mhDC = GetDC(mhWnd)))
        {
            MessageBoxA(NULL, "Can not create a GL Device Context", "ERROR", MB_OK | MB_ICONEXCLAMATION);
            exit(0);
        }

        if (!(iPixelFormat = ChoosePixelFormat(mhDC, &pfd)))
        {
            MessageBoxA(NULL, "Can not find suitable pixel format", "ERROR", MB_OK | MB_ICONEXCLAMATION);
            exit(0);
        }

        if (!SetPixelFormat(mhDC, iPixelFormat, &pfd))
        {
            MessageBoxA(NULL, "Can not set the pixelformat", "ERROR", MB_OK | MB_ICONEXCLAMATION);
            exit(0);
        }

        if (!(mhRC = wglCreateContext(mhDC)))
        {
            MessageBoxA(NULL, "Can not create Rendring context", "ERROR", MB_OK | MB_ICONEXCLAMATION);
            exit(0);
        }

        if (!wglMakeCurrent(mhDC, mhRC))
        {
            MessageBoxA(NULL, "Can not activate the GL Rendering context", "ERROR", MB_OK | MB_ICONEXCLAMATION);
            exit(0);
        }

        GLenum err = glewInit();
        if (err != GLEW_OK)
        {
            MessageBoxA(NULL, "glewInit failed.", "ERROR", MB_OK);
            exit(0);
        }

        glEnable(GL_DEPTH_TEST);

        VSIUtilResize(DEFAULT_WIDTH, DEFAULT_HEIGHT);
    }

    void VSIUtilPlus::Uninitialize()
    {
        if (mbFullscreen == true)
        {
            mdwStyle = GetWindowLong(mhWnd, GWL_STYLE);
            SetWindowLong(mhWnd, GWL_STYLE, mdwStyle | WS_OVERLAPPEDWINDOW);
            SetWindowPlacement(mhWnd, &mwpPrev);
            SetWindowPos(mhWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);

            ShowCursor(TRUE);
        }

        wglMakeCurrent(NULL, NULL);

        if (mhRC)
        {
            wglDeleteContext(mhRC);
            mhRC = NULL;
        }

        if (mhDC)
        {
            ReleaseDC(mhWnd, mhDC);
            mhDC = NULL;
        }
    }

    LRESULT CALLBACK VSIUtilPlus::mWndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
    {
        
        VSIUtilPlus *pThis = reinterpret_cast<VSIUtilPlus*> (GetWindowLongPtr(hWnd, GWLP_USERDATA));

        if (pThis == nullptr)
        {
            goto RETURN;
        }

        switch (iMsg)
        {
        case WM_ACTIVATE:
        {
            if (HIWORD(wParam) == 0)
            {	pThis->mbIsActiveWindow = true;
            }
            else
                pThis->mbIsActiveWindow = false;
        }
            break;
        case WM_KEYDOWN:
            switch (wParam)
            {
            case VK_ESCAPE:
            {
                pThis->mbIsEscapeKeyPressed = true;
            }
                break;
            case 0x46: //for 'f' or 'F'
                if (pThis->mbFullscreen == false)
                {
                    pThis->ToggleFullscreen();
                    pThis->mbFullscreen = true;
                }
                else
                {
                    pThis->ToggleFullscreen();
                    pThis->mbFullscreen = false;
                }
                break;
            default:
                break;
            }
            break;
        case WM_SIZE:
            pThis->VSIUtilResize(LOWORD(lParam), HIWORD(lParam));
            break;
        case WM_LBUTTONDOWN:
            break;
        case WM_DESTROY:
            PostQuitMessage(0);
            break;
        default:
            break;
        }
        //if (pThis != nullptr)
        //{
        //	pThis->MessageHandler(iMsg, wParam, lParam);
        //}

        pThis->VSIUtilMessageHandler(iMsg, wParam, lParam);
        RETURN:
        return DefWindowProc(hWnd, iMsg, wParam, lParam);
    }

    void VSIUtilPlus::VSIUtilRender()
    {

    }

    void VSIUtilPlus::VSIUtilUpdate()
    {

    }

    void VSIUtilPlus::VSIUtilResize(unsigned long width, unsigned long height)
    {
        mWidth = width;
        mHeight = height;

        glViewport(0, 0, width, height);
        float aspect = (float)width / (float)height;
        mProjMatrix = glm::perspective(50.0f,
            aspect,
            1.0f,
            1000.0f);
        
    }

    void VSIUtilPlus::VSIUtilSceneInit()
    {

    }

    void VSIUtilPlus::VSIUtilSceneDeinit()
    {

    }

    void VSIUtilPlus::ToggleFullscreen()
    {
        MONITORINFO mi;

        //code
        if (mbFullscreen == false)
        {
            mdwStyle = GetWindowLong(mhWnd, GWL_STYLE);
            if (mdwStyle & WS_OVERLAPPEDWINDOW)
            {
                mi = { sizeof(MONITORINFO) };
                if (GetWindowPlacement(mhWnd, &mwpPrev) && GetMonitorInfo(MonitorFromWindow(mhWnd, MONITORINFOF_PRIMARY), &mi))
                {
                    SetWindowLong(mhWnd, GWL_STYLE, mdwStyle & ~WS_OVERLAPPEDWINDOW);
                    SetWindowPos(mhWnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);
                }
            }
            ShowCursor(FALSE);
        }
        else
        {
            //code
            SetWindowLong(mhWnd, GWL_STYLE, mdwStyle | WS_OVERLAPPEDWINDOW);
            SetWindowPlacement(mhWnd, &mwpPrev);
            SetWindowPos(mhWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
            ShowCursor(TRUE);
        }

    }

    std::vector<glm::vec3>& VSIUtilPlus::VSIUtilGetVertices()
    {
        return m_CurrentModel->m_vertices;
    }

    std::vector<glm::vec3>& VSIUtilPlus::VSIUtilGetNormals()
    {
        return m_CurrentModel->m_normals;
    }

    std::vector<glm::vec2>& VSIUtilPlus::VSIUtilGetTexcoords()
    {
        return m_CurrentModel->m_texCoords;
    }

    std::vector<glm::vec3>& VSIUtilPlus::VSIUtilGetTangents()
    {
        return m_CurrentModel->m_tangents;
    }
}

