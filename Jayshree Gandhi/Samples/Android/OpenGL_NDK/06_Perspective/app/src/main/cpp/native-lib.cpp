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

GLuint gVao;

GLuint gVbo_color;
GLuint gVbo_position;

GLint gMUniform;
GLint gVUniform;
GLint gPUniform;

glm::mat4 gPerspectiveProjectionMatrix;

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
        glBindAttribLocation(program, JGG_ATTRIBUTE_COLOR, "vColor");

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
Java_com_jayshree_ndkoglperspective_GLESNativeLib_resize(JNIEnv *env, jclass type, jint width, jint height)
{
    GLfloat aspectRatio;

    if (height == 0)
        height = 1;

    glViewport(0, 0, (GLsizei)width, (GLsizei)height);

    aspectRatio = (GLfloat)width / (GLfloat)height;

    gPerspectiveProjectionMatrix = glm::perspective(45.0f, aspectRatio, 0.01f, 100.0f);
}

extern "C" JNIEXPORT void JNICALL
Java_com_jayshree_ndkoglperspective_GLESNativeLib_init(JNIEnv *env, jclass type)
{
    printOpenGLESInfo();

    const GLchar *vertexShaderSource =
        "#version 320 es"
        "\n"
        "in vec4 vPosition;"
        "in vec4 vColor;"
        "uniform mat4 u_m_matrix;"
        "uniform mat4 u_v_matrix;"
        "uniform mat4 u_p_matrix;"
        "out vec4 out_color;"
        "void main(void)"
        "{"
        "   gl_Position = u_p_matrix * u_v_matrix * u_m_matrix * vPosition;"
        "   out_color = vColor;"
        "}";

    const GLchar *fragmentShaderSource =
        "#version 320 es"
        "\n"
        "precision highp float;"
        "in vec4 out_color;"
        "out vec4 FragColor;"
        "void main(void)"
        "{"
        "   FragColor = out_color;"
        "}";

    gShaderProgramObject = createProgramObject(vertexShaderSource, fragmentShaderSource);

    gMUniform = glGetUniformLocation(gShaderProgramObject, "u_m_matrix");
    gVUniform = glGetUniformLocation(gShaderProgramObject, "u_v_matrix");
    gPUniform = glGetUniformLocation(gShaderProgramObject, "u_p_matrix");

    const GLfloat triangleVertices[] = {
        0.0f, 1.0f, 0.0f,   //apex
        -1.0f, -1.0f, 0.0f, //left bottom
        1.0f, -1.0f, 0.0f   //right bottom
    };

    const GLfloat triangleColor[] = {
        1.0f, 0.0f, 0.0f, //apex
        0.0f, 1.0f, 0.0f, //left bottom
        0.0f, 0.0f, 1.0f  //right bottom
    };

    glGenVertexArrays(1, &gVao);
    glBindVertexArray(gVao);

    glGenBuffers(1, &gVbo_position);
    glBindBuffer(GL_ARRAY_BUFFER, gVbo_position);
    glBufferData(GL_ARRAY_BUFFER, sizeof(triangleVertices), triangleVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(JGG_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(JGG_ATTRIBUTE_VERTEX);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &gVbo_color);
    glBindBuffer(GL_ARRAY_BUFFER, gVbo_color);
    glBufferData(GL_ARRAY_BUFFER, sizeof(triangleColor), triangleColor, GL_STATIC_DRAW);
    glVertexAttribPointer(JGG_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(JGG_ATTRIBUTE_COLOR);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);

    glClearDepthf(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glClearColor(0.1f, 0.2f, 0.3f, 0.0f);

    glm::mat4 gPerspectiveProjectionMatrix = glm::mat4(1.0);
}

extern "C" JNIEXPORT void JNICALL
Java_com_jayshree_ndkoglperspective_GLESNativeLib_display(JNIEnv *env, jclass type)
{
    //code
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(gShaderProgramObject);

    glm::mat4 modelMatrix = glm::mat4(1.0);
    glm::mat4 viewMatrix = glm::mat4(1.0);
    glm::mat4 translationMatrix = glm::mat4(1.0f);

    translationMatrix = glm::translate(glm::mat4(1.0), glm::vec3(0.0f, 0.0f, -3.0f));

    modelMatrix *= translationMatrix;

    glUniformMatrix4fv(gMUniform, 1, GL_FALSE, glm::value_ptr(modelMatrix));
    glUniformMatrix4fv(gVUniform, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(gPUniform, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));

    glBindVertexArray(gVao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    glUseProgram(0);
}
