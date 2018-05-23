#include <stdio.h>
#include <math.h>
#include <gl\glew.h>
#include <VSIUtilPlus.h>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include "Perlin.h"

static int p[B + B + 2];
static double g3[B + B + 2][3];
static double g2[B + B + 2][2];
static double g1[B + B + 2];
static int start = 1;

using namespace VSIUtil;

class VSIDemoNoise :public VSIUtilPlus
{
public:
    GLuint mProgObj;
    GLuint mVao;
    
    GLuint mOffsetLocation;
    GLuint mProjMatLocation;
    GLuint mMVLocation;
    
    GLuint buffer[3];
    int  mNoise3DTexSize;
    GLuint mNoise3DTexName;
    GLubyte *mNoise3DTexPtr;
    std::vector<glm::vec3> mVertices;
    std::vector<glm::vec2> mTextureCoords;
    std::vector<glm::vec3> mNormals;
    GLuint mTexture;

    VSIDemoNoise()
    {
        mNoise3DTexSize = 128;
        mNoise3DTexName = 0;
        mNoise3DTexPtr = NULL;
    }

    ~VSIDemoNoise()
    {
    }

    void WindowInit()
    {
        WCHAR tempName[128] = TEXT("VSI Noise Demo");
        wcscpy_s(mszAppName, wcslen(tempName) + 1, TEXT("VSI Noise Demo"));
    }

    std::vector<glm::vec3>& VSIUtilGetVertices()
    {
        return mVertices;
    }

    std::vector<glm::vec3>& VSIUtilGetNormals()
    {
        return mNormals;
    }

    std::vector<glm::vec2>& VSIUtilGetTexcoords()
    {
        return mTextureCoords;
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
        VSIUtilLoadMesh("Sphere.obj", VERTEX_NORMAL_AND_TEXTURE);

        mProgObj = VSIUtilLoadShaders("VSIDemoNoise.vs.glsl", "VSIDemoNoise.fs.glsl");

        mProjMatLocation = glGetUniformLocation(mProgObj, "ProjMatrix");
        mMVLocation = glGetUniformLocation(mProgObj, "MVMatrix");
        mOffsetLocation = glGetUniformLocation(mProgObj, "offset");

        glGenVertexArrays(1, &mVao);
        glBindVertexArray(mVao);

        VSIUtilGenAndBindBuffer(buffer[VERTEX], 0, VERTEX);

        VSIUtilGenAndBindBuffer(buffer[TEXTURECOORDS], 1, TEXTURECOORDS);

        VSIUtilGenAndBindBuffer(buffer[NORMALS], 2, NORMALS);
        //VSIUtilLoadTexture("earth_orig.jpg", &mTexture);
        Make3DNoiseTexture();
        LoadTexture();

        glBindVertexArray(0);
    }

    void LoadTexture()
    {
        glGenTextures(1, &mNoise3DTexName);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, mNoise3DTexName);
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, mNoise3DTexSize,
            mNoise3DTexSize, mNoise3DTexSize, 0, GL_RGBA,
            GL_UNSIGNED_BYTE, mNoise3DTexPtr);
    }
    void VSIUtilRender()
    {
        static GLfloat offset = 0.0f;
        static const GLfloat green[] = { 0.0f, 0.0f, 0.0f, 1.0f };
        static const GLfloat one = 1.0f;

        glBindVertexArray(mVao);

        glUseProgram(mProgObj);

        glClearBufferfv(GL_COLOR, 0, green);
        glClearBufferfv(GL_DEPTH, 0, &one);

        static GLfloat f = 0.0f;
        f += 0.001f;

        glm::mat4 viewMatrix = glm::lookAt(glm::vec3(0.0, 0.0, -2.0f),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0, 1.0f, 0.0f));

        glm::mat4 mvMatrix = viewMatrix * glm::rotate(glm::mat4(1.0f),  90.0f, glm::vec3(0.0f, 1.0f, 0.0f));// *
                                        //glm::rotate(glm::mat4(1.0f), f * 130.0f, glm::vec3(0.0f, 1.0f, 0.0f)) *
                                        //glm::translate(glm::mat4(1.0f), glm::vec3(f * 4.0f, -4.0f, 0.0f));

        glUniformMatrix4fv(mMVLocation, 1, GL_FALSE, glm::value_ptr(mvMatrix));
        glUniformMatrix4fv(mProjMatLocation, 1, GL_FALSE, glm::value_ptr(mProjMatrix));
        glUniform1f(mOffsetLocation, offset);
        offset += 0.00001f;
        glDrawArrays(GL_TRIANGLES, 0, mVertices.size());
    }

    void Make3DNoiseTexture(void)
    {
        int f, i, j, k, inc;
        int startFrequency = 40;
        int numOctaves = 4;
        double ni[3];
        double inci, incj, inck;
        int frequency = startFrequency;
        GLubyte *ptr;
        double amplitude = 0.6;


        if ((mNoise3DTexPtr = (GLubyte*)malloc(mNoise3DTexSize *
            mNoise3DTexSize *
            mNoise3DTexSize * 4)) == NULL)
        {
            return;
        }

        for (f = 0, inc = 0; f < numOctaves;
            ++f, frequency *= 2, ++inc, amplitude *= 0.5)
        {
            ptr = mNoise3DTexPtr;
            ni[0] = ni[1] = ni[2] = 0;

            inci = 1.0 / (mNoise3DTexSize / frequency);

            for (i = 0; i < mNoise3DTexSize; ++i, ni[0] += inci)
            {
                incj = 1.0 / (mNoise3DTexSize / frequency);
                for (j = 0; j < mNoise3DTexSize; ++j, ni[1] += incj)
                {
                    inck = 1.0 / (mNoise3DTexSize / frequency);
                    for (k = 0; k < mNoise3DTexSize; ++k, ni[2] += inck, ptr += 4)
                    {
                        *(ptr + inc) = (GLubyte)(((noise3(ni) + 1.0) * amplitude) * 128.0);
                    }
                }
            }
        }
    }

    void normalize2(double v[2])
    {
        double s;

        s = sqrt(v[0] * v[0] + v[1] * v[1]);
        v[0] = v[0] / s;
        v[1] = v[1] / s;
    }

    void normalize3(double v[3])
    {
        double s;

        s = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        v[0] = v[0] / s;
        v[1] = v[1] / s;
        v[2] = v[2] / s;
    }

    void init(void)
    {
        int i, j, k;

        for (i = 0; i < B; i++) {
            p[i] = i;
            g1[i] = (double)((rand() % (B + B)) - B) / B;

            for (j = 0; j < 2; j++)
                g2[i][j] = (double)((rand() % (B + B)) - B) / B;
            normalize2(g2[i]);

            for (j = 0; j < 3; j++)
                g3[i][j] = (double)((rand() % (B + B)) - B) / B;
            normalize3(g3[i]);
        }

        while (--i) {
            k = p[i];
            p[i] = p[j = rand() % B];
            p[j] = k;
        }

        for (i = 0; i < B + 2; i++) {
            p[B + i] = p[i];
            g1[B + i] = g1[i];
            for (j = 0; j < 2; j++)
                g2[B + i][j] = g2[i][j];
            for (j = 0; j < 3; j++)
                g3[B + i][j] = g3[i][j];
        }
    }
    
    double noise3(double vec[3])
    {
        int bx0, bx1, by0, by1, bz0, bz1, b00, b10, b01, b11;
        double rx0, rx1, ry0, ry1, rz0, rz1, *q, sy, sz, a, b, c, d, t, u, v;
        int i, j;

        if (start) {
            start = 0;
            init();
        }

        setup(0, bx0, bx1, rx0, rx1);
        setup(1, by0, by1, ry0, ry1);
        setup(2, bz0, bz1, rz0, rz1);

        i = p[bx0];
        j = p[bx1];

        b00 = p[i + by0];
        b10 = p[j + by0];
        b01 = p[i + by1];
        b11 = p[j + by1];

        t = s_curve(rx0);
        sy = s_curve(ry0);
        sz = s_curve(rz0);

        q = g3[b00 + bz0]; u = at3(rx0, ry0, rz0);
        q = g3[b10 + bz0]; v = at3(rx1, ry0, rz0);
        a = lerp(t, u, v);

        q = g3[b01 + bz0]; u = at3(rx0, ry1, rz0);
        q = g3[b11 + bz0]; v = at3(rx1, ry1, rz0);
        b = lerp(t, u, v);

        c = lerp(sy, a, b);

        q = g3[b00 + bz1]; u = at3(rx0, ry0, rz1);
        q = g3[b10 + bz1]; v = at3(rx1, ry0, rz1);
        a = lerp(t, u, v);

        q = g3[b01 + bz1]; u = at3(rx0, ry1, rz1);
        q = g3[b11 + bz1]; v = at3(rx1, ry1, rz1);
        b = lerp(t, u, v);

        d = lerp(sy, a, b);

        return lerp(sz, c, d);
    }
};

VSI_MAIN(VSIDemoNoise)
