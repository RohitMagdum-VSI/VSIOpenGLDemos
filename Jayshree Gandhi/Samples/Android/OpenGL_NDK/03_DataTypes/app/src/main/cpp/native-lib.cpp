#include <jni.h>
#include <string>

extern "C" JNIEXPORT jstring JNICALL
Java_com_jayshree_ndkdatatypes_MainActivity_stringFromJNI(JNIEnv *env, jobject /* this */)
{
    std::string hello = "Data Types";
    return env->NewStringUTF(hello.c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_jayshree_ndkdatatypes_MainActivity_intFromJNI(JNIEnv *env, jobject instance)
{
    return 5222;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_jayshree_ndkdatatypes_MainActivity_longFromJNI(JNIEnv *env, jobject instance)
{
    return 500000;
}

extern "C" JNIEXPORT jbyte JNICALL
Java_com_jayshree_ndkdatatypes_MainActivity_byteFromJNI(JNIEnv *env, jobject instance)
{
    return 127;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_jayshree_ndkdatatypes_MainActivity_sumOfTwoInt(JNIEnv *env, jobject instance, jint i, jint j)
{
    return (i + j);
}
