#include <stdio.h>
#include <SOIL.h>
#include <gl\glew.h>
#include <gl\GL.h>
#include <gl\GLU.h>
#include <VSIUtilPlus.h>

namespace VSIUtil
{
    int VSIUtilPlus::VSIUtilLoadTexture(char* imageFile, GLuint* texture)
    {
        GLvoid* image;
        int iRetVal = 0;
        int width, height;
        /* load an image file directly as a new OpenGL texture */

        image = SOIL_load_image
        (
            imageFile,
            &width,
            &height,
            0,
            SOIL_LOAD_RGBA
        );
        if (image == NULL)
        {
            const char* ch = SOIL_last_result();
            MessageBoxA(NULL, ch, "ERROR", MB_OK);
            return -1;
        }
        glGenTextures(1, texture);
        glBindTexture(GL_TEXTURE_2D, *texture);
        glTexStorage2D(GL_TEXTURE_2D, 8, GL_RGB8, width, height);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, image);
        glGenerateMipmap(GL_TEXTURE_2D);

        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        return 0;
    }

    void VSIUtilPlus::VSIUtilLoadCubeMapTextures(std::vector<std::string> textureFiles, GLuint *texture)
    {
        GLvoid *image;
        int width, height;

        glGenTextures(1, texture);
        glBindTexture(GL_TEXTURE_CUBE_MAP, *texture);

        for (int i = 0; i < 6; i++)
        {
            image = SOIL_load_image(
                    textureFiles[i].c_str(),
                    &width,
                    &height,
                    0,
                    SOIL_LOAD_RGBA
            );
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
        }
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    }
}