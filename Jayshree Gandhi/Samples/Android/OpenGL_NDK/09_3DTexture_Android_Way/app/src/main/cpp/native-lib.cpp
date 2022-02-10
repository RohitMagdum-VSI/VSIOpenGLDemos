#include <jni.h>
#include <string>

//OpenGL
#include <GLES3/gl32.h>

//GLM
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#include <android/log.h>

//JNI information and error logs
#define LOG_TAG "JGG: glOpenGLES32PerspectiveNative"
#define LOGINFO(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGERROR(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

//global variables
enum
{
    JGG_ATTRIBUTE_VERTEX = 0,
    JGG_ATTRIBUTE_COLOR,
    JGG_ATTRIBUTE_NORMAL,
    JGG_ATTRIBUTE_TEXCOORD0
};

GLuint gShaderProgramObject;

GLuint gVao_pyramid;
GLuint gVbo_texture_pyramid;
GLuint gVbo_position_pyramid;

GLuint gVao_cube;
GLuint gVbo_texture_cube;
GLuint gVbo_position_cube;

GLint gMUniform;
GLint gVUniform;
GLint gPUniform;

GLuint gTextureSamplerUniform;

GLuint gTexture_Kundali;
int kundaliHeight;
int kundaliWidth;
int *kundaliPixels;

GLuint gTexture_Stone;
int stoneHeight;
int stoneWidth;
int *stonePixels;

GLfloat gAngleRotation;

glm::mat4 gPerspectiveProjectionMatrix;

void update();

GLuint loadAndCompileShader(GLenum shaderType, const char *sourceCode)
{
    GLuint shaderID = glCreateShader(shaderType);

    if (shaderID)
    {
        glShaderSource(shaderID, 1, &sourceCode, NULL);
        glCompileShader(shaderID);

        GLint iInfoLogLength = 0;
        GLint iShaderCompiledStatus = 0;
        char *szInfoLog = NULL;

        glGetShaderiv(shaderID, GL_COMPILE_STATUS, &iShaderCompiledStatus);
        if (iShaderCompiledStatus == GL_FALSE)
        {
            glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &iInfoLogLength);
            if (iInfoLogLength > 0)
            {
                szInfoLog = (char *)malloc(iInfoLogLength);
                if (szInfoLog != NULL)
                {
                    GLsizei written;
                    glGetShaderInfoLog(shaderID, iInfoLogLength, &written, szInfoLog);
                    LOGERROR("Could not compile shader %d:\n%s\n", shaderType, szInfoLog);
                    free(szInfoLog);
                }
                glDeleteShader(shaderID);
                shaderID = 0;
            }
        }
    }

    return shaderID;
}

GLuint createProgramObjectAndLinkShader(GLuint vertexShaderID, GLuint fragmentShaderID)
{
    if (!vertexShaderID || !fragmentShaderID)
        return 0;

    GLuint program = glCreateProgram();

    if (program)
    {
        glAttachShader(program, vertexShaderID);
        glAttachShader(program, fragmentShaderID);

        glBindAttribLocation(program, JGG_ATTRIBUTE_VERTEX, "vPosition");
        glBindAttribLocation(program, JGG_ATTRIBUTE_TEXCOORD0, "vTexcoord");

        glLinkProgram(program);

        GLint iInfoLogLength = 0;
        GLint iShaderProgramLinkStatus = 0;
        char *szInfoLog = NULL;

        glGetProgramiv(program, GL_LINK_STATUS, &iShaderProgramLinkStatus);

        if (iShaderProgramLinkStatus == GL_FALSE)
        {
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &iInfoLogLength);
            if (iInfoLogLength > 0)
            {
                szInfoLog = (char *)malloc(iInfoLogLength);
                if (szInfoLog != NULL)
                {
                    GLsizei written;
                    glGetProgramInfoLog(program, iInfoLogLength, &written, szInfoLog);
                    LOGERROR("Could not link program :\n%s\n", szInfoLog);
                    free(szInfoLog);
                }
                glDeleteProgram(program);
                program = 0;
            }
        }
    }

    return program;
}

GLint createProgramObject(const char *vertexSource, const char *fragementSource)
{
    GLuint vsID = loadAndCompileShader(GL_VERTEX_SHADER, vertexSource);
    GLuint fsID = loadAndCompileShader(GL_FRAGMENT_SHADER, fragementSource);

    return createProgramObjectAndLinkShader(vsID, fsID);
}

GLint loadTexture(int texWidth, int texHeight, int *texPixels)
{
    GLuint texture_id;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        texWidth,
        texHeight,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        texPixels);

    glGenerateMipmap(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);

    return texture_id;
}

static void printGLString(const char *name, GLenum s)
{
    LOGINFO("GL %s = %s\n", name, (const char *)glGetString(s));
}

void printOpenGLESInfo()
{
    printGLString("Version", GL_VERSION);
    printGLString("Vendor", GL_VENDOR);
    printGLString("Renderer", GL_RENDERER);
    printGLString("Extensions", GL_EXTENSIONS);
    printGLString("Shading Language", GL_SHADING_LANGUAGE_VERSION);
}

extern "C" JNIEXPORT void JNICALL
Java_com_jayshree_ndkoglthreedtextureandroid_GLESNativeLib_loadBitMap(JNIEnv *env, jclass type, jintArray pixels_, jint width, jint height, jstring texture_)
{
    jint *pixels = env->GetIntArrayElements(pixels_, NULL);
    const char *texture = env->GetStringUTFChars(texture_, 0);

    if (strcmp(texture, "stone") == 0)
    {
        stoneWidth = width;
        stoneHeight = height;
        stonePixels = pixels;
    }
    else if (strcmp(texture, "kundali") == 0)
    {
        kundaliWidth = width;
        kundaliHeight = height;
        kundaliPixels = pixels;
    }

    env->ReleaseIntArrayElements(pixels_, pixels, 0);
    env->ReleaseStringUTFChars(texture_, texture);
}

extern "C" JNIEXPORT void JNICALL
Java_com_jayshree_ndkoglthreedtextureandroid_GLESNativeLib_resize(JNIEnv *env, jclass type, jint width, jint height)
{
    GLfloat aspectRatio;

    if (height == 0)
        height = 1;

    glViewport(0, 0, (GLsizei)width, (GLsizei)height);

    aspectRatio = (GLfloat)width / (GLfloat)height;

    gPerspectiveProjectionMatrix = glm::perspective(45.0f, aspectRatio, 0.01f, 100.0f);
}

extern "C" JNIEXPORT void JNICALL
Java_com_jayshree_ndkoglthreedtextureandroid_GLESNativeLib_init(JNIEnv *env, jclass type)
{
    printOpenGLESInfo();

    const GLchar *vertexShaderSource =
        "#version 320 es"
        "\n"
        "in vec4 vPosition;"
        "in vec2 vTexcoord;"
        "uniform mat4 u_m_matrix;"
        "uniform mat4 u_v_matrix;"
        "uniform mat4 u_p_matrix;"
        "out vec2 out_texcoord;"
        "void main(void)"
        "{"
        "   gl_Position = u_p_matrix * u_v_matrix * u_m_matrix * vPosition;"
        "   out_texcoord = vTexcoord;"
        "}";

    const GLchar *fragmentShaderSource =
        "#version 320 es"
        "\n"
        "precision highp float;"
        "in vec2 out_texcoord;"
        "uniform sampler2D u_texture_sampler;"
        "out vec4 FragColor;"
        "void main(void)"
        "{"
        "   vec4 tex = texture(u_texture_sampler, out_texcoord);"
        "   FragColor = vec4(tex.b, tex.g, tex.r, tex.a);"
        "}";

    gShaderProgramObject = createProgramObject(vertexShaderSource, fragmentShaderSource);

    gMUniform = glGetUniformLocation(gShaderProgramObject, "u_m_matrix");
    gVUniform = glGetUniformLocation(gShaderProgramObject, "u_v_matrix");
    gPUniform = glGetUniformLocation(gShaderProgramObject, "u_p_matrix");
    gTextureSamplerUniform = glGetUniformLocation(gShaderProgramObject, "u_texture_sampler");

    const GLfloat pyramidVertices[] = {
        0.0f, 1.0f, 0.0f,   //apex
        -1.0f, -1.0f, 1.0f, //left bottom
        1.0f, -1.0f, 1.0f,  //right bottom

        0.0f, 1.0f, 0.0f,   //apex
        1.0f, -1.0f, 1.0f,  //left bottom
        1.0f, -1.0f, -1.0f, //right bottom

        0.0f, 1.0f, 0.0f,    //apex
        1.0f, -1.0f, -1.0f,  //left bottom
        -1.0f, -1.0f, -1.0f, //right bottom

        0.0f, 1.0f, 0.0f,    //apex
        -1.0f, -1.0f, -1.0f, //left bottom
        -1.0f, -1.0f, 1.0f   //right bottom
    };

    const GLfloat pyramidTexcoords[] = {
        0.5f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f,

        0.5f, 1.0f,
        1.0f, 0.0f,
        0.0f, 0.0f,

        0.5f, 1.0f,
        1.0f, 0.0f,
        0.0f, 0.0f,

        0.5f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f};

    const GLfloat cubeVertices[] = {
        1.0f, 1.0f, -1.0f,
        -1.0f, 1.0f, -1.0f,
        -1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,

        1.0f, -1.0f, 1.0f,
        -1.0f, -1.0f, 1.0f,
        -1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,

        1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 1.0f,
        1.0f, -1.0f, 1.0f,

        1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f, 1.0f, -1.0f,
        1.0f, 1.0f, -1.0f,

        -1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f, 1.0f,

        1.0f, 1.0f, -1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, -1.0f, 1.0f,
        1.0f, -1.0f, -1.0f};

    const GLfloat cubeTexcoords[] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f,

        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f,

        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f,

        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f,

        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f,

        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f};

    glGenVertexArrays(1, &gVao_pyramid);
    glBindVertexArray(gVao_pyramid);

    glGenBuffers(1, &gVbo_position_pyramid);
    glBindBuffer(GL_ARRAY_BUFFER, gVbo_position_pyramid);
    glBufferData(GL_ARRAY_BUFFER, sizeof(pyramidVertices), pyramidVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(JGG_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(JGG_ATTRIBUTE_VERTEX);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &gVbo_texture_pyramid);
    glBindBuffer(GL_ARRAY_BUFFER, gVbo_texture_pyramid);
    glBufferData(GL_ARRAY_BUFFER, sizeof(pyramidTexcoords), pyramidTexcoords, GL_STATIC_DRAW);
    glVertexAttribPointer(JGG_ATTRIBUTE_TEXCOORD0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(JGG_ATTRIBUTE_TEXCOORD0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);

    glGenVertexArrays(1, &gVao_cube);
    glBindVertexArray(gVao_cube);

    glGenBuffers(1, &gVbo_position_cube);
    glBindBuffer(GL_ARRAY_BUFFER, gVbo_position_cube);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(JGG_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(JGG_ATTRIBUTE_VERTEX);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &gVbo_texture_cube);
    glBindBuffer(GL_ARRAY_BUFFER, gVbo_texture_cube);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeTexcoords), cubeTexcoords, GL_STATIC_DRAW);
    glVertexAttribPointer(JGG_ATTRIBUTE_TEXCOORD0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(JGG_ATTRIBUTE_TEXCOORD0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);

    gTexture_Kundali = loadTexture(kundaliWidth, kundaliHeight, kundaliPixels);
    gTexture_Stone = loadTexture(stoneWidth, stoneHeight, stonePixels);

    glEnable(GL_TEXTURE_2D);

    glClearDepthf(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glClearColor(0.1f, 0.2f, 0.3f, 0.0f);

    glm::mat4 gPerspectiveProjectionMatrix = glm::mat4(1.0);
}

extern "C" JNIEXPORT void JNICALL
Java_com_jayshree_ndkoglthreedtextureandroid_GLESNativeLib_display(JNIEnv *env, jclass type)
{
    //code
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(gShaderProgramObject);

    //pyramid
    glm::mat4 modelMatrix = glm::mat4(1.0);
    glm::mat4 viewMatrix = glm::mat4(1.0);
    glm::mat4 translationMatrix = glm::mat4(1.0);
    glm::mat4 rotationMatrix = glm::mat4(1.0);

    translationMatrix = glm::translate(glm::mat4(1.0), glm::vec3(-1.5f, 0.0f, -5.0f));
    rotationMatrix = glm::rotate(glm::mat4(1.0), glm::radians(gAngleRotation), glm::vec3(0.0f, 1.0f, 0.0f));
    modelMatrix *= translationMatrix;
    modelMatrix *= rotationMatrix;

    glUniformMatrix4fv(gMUniform, 1, GL_FALSE, glm::value_ptr(modelMatrix));
    glUniformMatrix4fv(gVUniform, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(gPUniform, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gTexture_Stone);
    glUniform1i(gTextureSamplerUniform, 0);

    glBindVertexArray(gVao_pyramid);
    glDrawArrays(GL_TRIANGLES, 0, 12);
    glBindVertexArray(0);

    glBindTexture(GL_TEXTURE_2D, 0);

    //cube
    modelMatrix = glm::mat4(1.0);
    viewMatrix = glm::mat4(1.0);
    translationMatrix = glm::mat4(1.0);
    rotationMatrix = glm::mat4(1.0);
    glm::mat4 scaleMatrix = glm::mat4(1.0);

    translationMatrix = glm::translate(glm::mat4(1.0), glm::vec3(1.5f, 0.0f, -5.0f));
    rotationMatrix = glm::rotate(glm::mat4(1.0), glm::radians(gAngleRotation), glm::vec3(1.0f, 1.0f, 1.0f));
    scaleMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.75f));

    modelMatrix *= translationMatrix;
    modelMatrix *= scaleMatrix;
    modelMatrix *= rotationMatrix;

    glUniformMatrix4fv(gMUniform, 1, GL_FALSE, glm::value_ptr(modelMatrix));
    glUniformMatrix4fv(gVUniform, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(gPUniform, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gTexture_Kundali);
    glUniform1i(gTextureSamplerUniform, 0);

    glBindVertexArray(gVao_cube);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 20, 4);
    glBindVertexArray(0);

    glBindTexture(GL_TEXTURE_2D, 0);

    glUseProgram(0);

    update();
}

void update()
{
    gAngleRotation = gAngleRotation + 0.5f;
    if (gAngleRotation >= 360.0f)
    {
        gAngleRotation = 0.0f;
    }
}
