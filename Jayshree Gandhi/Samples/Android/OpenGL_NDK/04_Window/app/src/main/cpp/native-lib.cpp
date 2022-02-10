#include <jni.h>
#include <string>

#include <GLES3/gl32.h>

extern "C" JNIEXPORT void JNICALL
Java_com_jayshree_ndkoglwindow_GLESNativeLib_resize(JNIEnv *env, jclass type, jint width, jint height)
{
    //code
    if (height == 0)
        height = 1;

    glViewport(0, 0, (GLsizei)width, (GLsizei)height);
}

extern "C" JNIEXPORT void JNICALL
Java_com_jayshree_ndkoglwindow_GLESNativeLib_init(JNIEnv *env, jclass type)
{
    //code
    glClearColor(0.1f, 0.2f, 0.3f, 0.0f);
}

extern "C" JNIEXPORT void JNICALL
Java_com_jayshree_ndkoglwindow_GLESNativeLib_display(JNIEnv *env, jclass type)
{
    //code
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
