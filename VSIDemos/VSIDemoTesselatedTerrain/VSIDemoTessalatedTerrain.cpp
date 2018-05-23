#include <Windows.h>
#include <gl\glew.h>
#include <VSIUtilPlus.h>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>

unsigned char identifier[] =
{
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A
};

using namespace VSIUtil;

class VSIDemoTessalatedTerrain :public VSIUtilPlus
{
public:
    GLuint m_progObj;
    GLuint m_VAO;
    GLuint m_displacementTex;
    GLuint m_terrainGrassTex;

    GLuint m_mvMatrixLocation;
    GLuint m_projMatrixLocation;
    GLuint m_mvpMatrixLocation;
    GLuint m_enableFogLocation;

    bool enable_fog;
    bool wireframe;

    GLfloat m_xTrans, m_yTrans, m_zTrans;

    VSIDemoTessalatedTerrain()
    {
        wireframe = false;
        enable_fog = false;
    }

    ~VSIDemoTessalatedTerrain()
    {

    }

    void WindowInit()
    {
        WCHAR tempName[128] = TEXT("VSI Tessellated Terrain");
        wcscpy_s(mszAppName, wcslen(tempName) + 1, tempName);
    }

    struct header
    {
        unsigned char       identifier[12];
        unsigned int        endianness;
        unsigned int        gltype;
        unsigned int        gltypesize;
        unsigned int        glformat;
        unsigned int        glinternalformat;
        unsigned int        glbaseinternalformat;
        unsigned int        pixelwidth;
        unsigned int        pixelheight;
        unsigned int        pixeldepth;
        unsigned int        arrayelements;
        unsigned int        faces;
        unsigned int        miplevels;
        unsigned int        keypairbytes;
    };


    

    const unsigned int swap32(const unsigned int u32)
    {
        union
        {
            unsigned int u32;
            unsigned char u8[4];
        } a, b;

        a.u32 = u32;
        b.u8[0] = a.u8[3];
        b.u8[1] = a.u8[2];
        b.u8[2] = a.u8[1];
        b.u8[3] = a.u8[0];

        return b.u32;
    }

    const unsigned short swap16(const unsigned short u16)
    {
        union
        {
            unsigned short u16;
            unsigned char u8[2];
        } a, b;

        a.u16 = u16;
        b.u8[0] = a.u8[1];
        b.u8[1] = a.u8[0];

        return b.u16;
    }

    unsigned int calculate_stride(const header& h, unsigned int width, unsigned int pad = 4)
    {
        unsigned int channels = 0;

        switch (h.glbaseinternalformat)
        {
        case GL_RED:    channels = 1;
            break;
        case GL_RG:     channels = 2;
            break;
        case GL_BGR:
        case GL_RGB:    channels = 3;
            break;
        case GL_BGRA:
        case GL_RGBA:   channels = 4;
            break;
        }

        unsigned int stride = h.gltypesize * channels * width;

        stride = (stride + (pad - 1)) & ~(pad - 1);

        return stride;
    }

    unsigned int calculate_face_size(const header& h)
    {
        unsigned int stride = calculate_stride(h, h.pixelwidth);

        return stride * h.pixelheight;
    }

    unsigned int load(const char * filename, unsigned int tex)
    {
        FILE * fp;
        GLuint temp = 0;
        GLuint retval = 0;
        header h;
        size_t data_start, data_end;
        unsigned char * data;
        GLenum target = GL_NONE;

        int err = fopen_s(&fp, filename, "rb");

        if (!fp)
        {
            err = GetLastError();
            return 0;
        }

        if (fread(&h, sizeof(h), 1, fp) != 1)
            goto fail_read;

        if (memcmp(h.identifier, identifier, sizeof(identifier)) != 0)
            goto fail_header;

        if (h.endianness == 0x04030201)
        {
            // No swap needed
        }
        else if (h.endianness == 0x01020304)
        {
            // Swap needed
            h.endianness = swap32(h.endianness);
            h.gltype = swap32(h.gltype);
            h.gltypesize = swap32(h.gltypesize);
            h.glformat = swap32(h.glformat);
            h.glinternalformat = swap32(h.glinternalformat);
            h.glbaseinternalformat = swap32(h.glbaseinternalformat);
            h.pixelwidth = swap32(h.pixelwidth);
            h.pixelheight = swap32(h.pixelheight);
            h.pixeldepth = swap32(h.pixeldepth);
            h.arrayelements = swap32(h.arrayelements);
            h.faces = swap32(h.faces);
            h.miplevels = swap32(h.miplevels);
            h.keypairbytes = swap32(h.keypairbytes);
        }
        else
        {
            goto fail_header;
        }

        // Guess target (texture type)
        if (h.pixelheight == 0)
        {
            if (h.arrayelements == 0)
            {
                target = GL_TEXTURE_1D;
            }
            else
            {
                target = GL_TEXTURE_1D_ARRAY;
            }
        }
        else if (h.pixeldepth == 0)
        {
            if (h.arrayelements == 0)
            {
                if (h.faces == 0)
                {
                    target = GL_TEXTURE_2D;
                }
                else
                {
                    target = GL_TEXTURE_CUBE_MAP;
                }
            }
            else
            {
                if (h.faces == 0)
                {
                    target = GL_TEXTURE_2D_ARRAY;
                }
                else
                {
                    target = GL_TEXTURE_CUBE_MAP_ARRAY;
                }
            }
        }
        else
        {
            target = GL_TEXTURE_3D;
        }

        // Check for insanity...
        if (target == GL_NONE ||                                    // Couldn't figure out target
            (h.pixelwidth == 0) ||                                  // Texture has no width???
            (h.pixelheight == 0 && h.pixeldepth != 0))              // Texture has depth but no height???
        {
            goto fail_header;
        }

        temp = tex;
        if (tex == 0)
        {
            glGenTextures(1, &tex);
        }

        glBindTexture(target, tex);

        data_start = ftell(fp) + h.keypairbytes;
        fseek(fp, 0, SEEK_END);
        data_end = ftell(fp);
        fseek(fp, data_start, SEEK_SET);

        data = new unsigned char[data_end - data_start];
        memset(data, 0, data_end - data_start);

        fread(data, 1, data_end - data_start, fp);

        if (h.miplevels == 0)
        {
            h.miplevels = 1;
        }

        switch (target)
        {
        case GL_TEXTURE_1D:
            glTexStorage1D(GL_TEXTURE_1D, h.miplevels, h.glinternalformat, h.pixelwidth);
            glTexSubImage1D(GL_TEXTURE_1D, 0, 0, h.pixelwidth, h.glformat, h.glinternalformat, data);
            break;
        case GL_TEXTURE_2D:
            // glTexImage2D(GL_TEXTURE_2D, 0, h.glinternalformat, h.pixelwidth, h.pixelheight, 0, h.glformat, h.gltype, data);
            if (h.gltype == GL_NONE)
            {
                glCompressedTexImage2D(GL_TEXTURE_2D, 0, h.glinternalformat, h.pixelwidth, h.pixelheight, 0, 420 * 380 / 2, data);
            }
            else
            {
                glTexStorage2D(GL_TEXTURE_2D, h.miplevels, h.glinternalformat, h.pixelwidth, h.pixelheight);
                {
                    unsigned char * ptr = data;
                    unsigned int height = h.pixelheight;
                    unsigned int width = h.pixelwidth;
                    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                    for (unsigned int i = 0; i < h.miplevels; i++)
                    {
                        glTexSubImage2D(GL_TEXTURE_2D, i, 0, 0, width, height, h.glformat, h.gltype, ptr);
                        ptr += height * calculate_stride(h, width, 1);
                        height >>= 1;
                        width >>= 1;
                        if (!height)
                            height = 1;
                        if (!width)
                            width = 1;
                    }
                }
            }
            break;
        case GL_TEXTURE_3D:
            glTexStorage3D(GL_TEXTURE_3D, h.miplevels, h.glinternalformat, h.pixelwidth, h.pixelheight, h.pixeldepth);
            glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, h.pixelwidth, h.pixelheight, h.pixeldepth, h.glformat, h.gltype, data);
            break;
        case GL_TEXTURE_1D_ARRAY:
            glTexStorage2D(GL_TEXTURE_1D_ARRAY, h.miplevels, h.glinternalformat, h.pixelwidth, h.arrayelements);
            glTexSubImage2D(GL_TEXTURE_1D_ARRAY, 0, 0, 0, h.pixelwidth, h.arrayelements, h.glformat, h.gltype, data);
            break;
        case GL_TEXTURE_2D_ARRAY:
        {
            glTexStorage3D(GL_TEXTURE_2D_ARRAY, h.miplevels, h.glinternalformat, h.pixelwidth, h.pixelheight, h.arrayelements);
            glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, h.pixelwidth, h.pixelheight, h.arrayelements, h.glformat, h.gltype, data);
            break;
        }
        case GL_TEXTURE_CUBE_MAP:
            glTexStorage2D(GL_TEXTURE_CUBE_MAP, h.miplevels, h.glinternalformat, h.pixelwidth, h.pixelheight);
            // glTexSubImage3D(GL_TEXTURE_CUBE_MAP, 0, 0, 0, 0, h.pixelwidth, h.pixelheight, h.faces, h.glformat, h.gltype, data);
            {
                unsigned int face_size = calculate_face_size(h);
                for (unsigned int i = 0; i < h.faces; i++)
                {
                    glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, 0, 0, h.pixelwidth, h.pixelheight, h.glformat, h.gltype, data + face_size * i);
                }
            }
            break;
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY, h.miplevels, h.glinternalformat, h.pixelwidth, h.pixelheight, h.arrayelements);
            glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, 0, 0, 0, 0, h.pixelwidth, h.pixelheight, h.faces * h.arrayelements, h.glformat, h.gltype, data);
            break;
        default:                                               // Should never happen
            goto fail_target;
        }

        if (h.miplevels == 1)
        {
            glGenerateMipmap(target);
        }

        retval = tex;

    fail_target:
        delete[] data;

    fail_header:;
    fail_read:;
        fclose(fp);

        return retval;
    }

    void VSIUtilSceneInit()
    {
        m_progObj = VSIUtilLoadShaders(
            "VSIDemoTessalatedTerrain.vs.glsl", "VSIDemoTessalatedTerrain.fs.glsl",
            "VSIDemoTessalatedTerrain.control.glsl", "VSIDemoTessalatedTerrain.evaluation.glsl");

        m_mvMatrixLocation = glGetUniformLocation(m_progObj, "mv_matrix");
        m_projMatrixLocation = glGetUniformLocation(m_progObj, "proj_matrix");
        m_mvpMatrixLocation = glGetUniformLocation(m_progObj, "mvp_matrix");
        m_enableFogLocation = glGetUniformLocation(m_progObj, "enable_fog");

        glGenVertexArrays(1, &m_VAO);
        glBindVertexArray(m_VAO);

        glPatchParameteri(GL_PATCH_VERTICES, 4);

        glEnable(GL_CULL_FACE);

//        load("terragen1.ktx", m_displacementTex);
        VSIUtilLoadTexture("displacement.jpg", &m_displacementTex);
        glActiveTexture(GL_TEXTURE1);
        VSIUtilLoadTexture("terraingrass.jpg", &m_terrainGrassTex);
    }

    void VSIUtilRender()
    {
        static const GLfloat black[] = { 0.0f, 0.0f, 0.0f, 1.0f };
        static const GLfloat one = 1.0f;
        
        static GLfloat f = 0.0f;
        f += 0.001f;

        float t = (float)f * 0.03f + m_xTrans;
        float r = sinf(t * 5.37f) * 15.0f + m_zTrans;
        float h = cosf(t * 4.79f) * 2.0f - m_yTrans;

        glClearBufferfv(GL_COLOR, 0, black);
        glClearBufferfv(GL_DEPTH, 0, &one);

        glm::mat4 mv_matrix = glm::lookAt(glm::vec3(sinf(t) * r, h, cosf(t) * r), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        float aspect = (float)mWidth / (float)mHeight;
        mProjMatrix = glm::perspective(50.0f,
            aspect,
            1.0f,
            1000.0f);
        glUseProgram(m_progObj);

        glUniformMatrix4fv(m_mvMatrixLocation, 1, GL_FALSE, glm::value_ptr(mv_matrix));
        glUniformMatrix4fv(m_projMatrixLocation, 1, GL_FALSE, glm::value_ptr(mProjMatrix));
        glUniformMatrix4fv(m_mvpMatrixLocation, 1, GL_FALSE, glm::value_ptr(mProjMatrix * mv_matrix));
        glUniform1i(m_enableFogLocation, enable_fog ? 1 : 0);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);

        if (wireframe)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        else
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        glDrawArraysInstanced(GL_PATCHES, 0, 4, 64 * 64);
    }
    void VSIUtilMessageHandler(UINT iMsg, WPARAM wParam, LPARAM lParam)
    {
        switch (iMsg)
        {
        case WM_CHAR:
            switch (wParam)
            {
            case 'q':
                m_xTrans += 1.5f;
                break;      
            case 'w':       
                m_yTrans += 1.5f;
                break;      
            case 'e':       
                m_zTrans += 1.5f;
                break;      
            case 'a':       
                m_xTrans -= 1.5f;
                break;      
            case 's':       
                m_yTrans -= 1.5f;
                break;      
            case 'd':       
                m_zTrans -= 1.5f;
                break;
            case 'x':
                enable_fog = !enable_fog;
                break;
            case 'z':
                wireframe = !wireframe;
                break;
            }
        }
    }
};

VSI_MAIN(VSIDemoTessalatedTerrain)