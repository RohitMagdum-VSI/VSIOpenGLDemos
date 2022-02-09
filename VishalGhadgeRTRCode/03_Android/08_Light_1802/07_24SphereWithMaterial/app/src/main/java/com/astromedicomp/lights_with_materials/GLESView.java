package com.astromedicomp.lights_with_materials;

import android.content.Context;
import android.opengl.GLSurfaceView;//	for open gl surface view all related.
import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;
import android.opengl.GLES30;	//	Change this version as per requirement.
import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnGestureListener;
import android.view.GestureDetector.OnDoubleTapListener;
import java.nio.ShortBuffer;

//	For VBO
import java.nio.ByteBuffer;	//	nio - non-blocking I/O
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import android.opengl.Matrix;

public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer,OnGestureListener,OnDoubleTapListener{

	private GestureDetector gestureDetector;
	private final Context context;

    int g_iCurrentWidth;
    int g_iCurrentHeight;

    private int shaderObjectVertex;
	private int shaderObjectFragment;
	private int shaderObjectProgram;

	//+	For sphere
    private int[] vao_sphere = new int[1];
    private int[] vbo_sphere_position = new int[1];
    private int[] vbo_sphere_normal = new int[1];
    private int[] vbo_sphere_element = new int[1];
	
	private int numVertices;
	private int numElements;
	//-	For sphere

	//+ Uniforms
	private int g_iModelMat4Uniform;
	private int g_iViewMat4Uniform;
	private int g_iProjectionMat4Uniform;
	private int g_iRotationMat4Uniform;

	private int g_iLaVec3Uniform;
	private int g_iLdVec3Uniform;
	private int g_iLsVec3Uniform;
	private int g_iLightPositionVec4Uniform;

	private int g_iKaVec3Uniform;
	private int g_iKdVec3Uniform;
	private int g_iKsVec3Uniform;
	private int g_iMaterialShininessUniform;

	private int g_iDoubleTapUniform;
	//- Uniforms

	//	Light
	private float g_farrLightAmbient[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	private float g_farrLightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	private float g_farrLightSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	private float g_farrLightPosition[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Give position runtime

	///////////////////////////////////////////////////////////////////////////
	//+Material

    //
    //	Materail 00
    //
    private float g_arrMaterial00Ambient[] = { 0.0215f, 0.1745f, 0.0215f, 1.0f };
    private float g_arrMaterial00Diffuse[] = { 0.07568f, 0.61424f, 0.07568f, 1.0f };
    private float g_arrMaterial00Specular[] = { 0.633f, 0.727811f, 0.633f, 1.0f };
    private float g_Material00Shininess = 0.6f * 128.0f;

    //
    //	Materail 10
    //
    private float g_arrMaterial10Ambient[] = { 0.135f, 0.2225f, 0.1575f, 1.0f };
    private float g_arrMaterial10Diffuse[] = { 0.54f, 0.89f, 0.63f, 1.0f };
    private float g_arrMaterial10Specular[] = { 0.316228f, 0.316228f, 0.316228f, 1.0f };
    private float g_Material10Shininess = 0.1f * 128.0f;

    //
    //	Materail 20
    //
    private float g_arrMaterial20Ambient[] = { 0.05375f, 0.05f, 0.06625f, 1.0f };
    private float g_arrMaterial20Diffuse[] = { 0.18275f, 0.17f, 0.22525f, 1.0f };
    private float g_arrMaterial20Specular[] = { 0.332741f, 0.328634f, 0.346435f, 1.0f };
    private float g_Material20Shininess = 0.3f * 128.0f;

    //
    //	Materail 30
    //
    private float g_arrMaterial30Ambient[] = { 0.25f, 0.20725f, 0.20725f, 1.0f };
    private float g_arrMaterial30Diffuse[] = { 1.0f, 0.829f, 0.829f, 1.0f };
    private float g_arrMaterial30Specular[] = { 0.296648f, 0.296648f, 0.296648f, 1.0f };
    private float g_Material30Shininess = 0.088f * 128.0f;

    //
    //	Materail 40
    //
    private float g_arrMaterial40Ambient[] = { 0.1745f, 0.01175f, 0.01175f, 1.0f };
    private float g_arrMaterial40Diffuse[] = { 0.61424f, 0.04136f, 0.04136f, 1.0f };
    private float g_arrMaterial40Specular[] = { 0.727811f, 0.626959f, 0.626959f, 1.0f };
    private float g_Material40Shininess = 0.6f * 128.0f;

    //
    //	Materail 50
    //
    private float g_arrMaterial50Ambient[] = { 0.1f, 0.18725f, 0.1745f, 1.0f };
    private float g_arrMaterial50Diffuse[] = { 0.396f, 0.74151f, 0.69102f, 1.0f };
    private float g_arrMaterial50Specular[] = { 0.297254f, 0.30829f, 0.306678f, 1.0f };
    private float g_Material50Shininess = 0.1f * 128.0f;

    //
    //	Materail 01
    //
    private float g_arrMaterial01Ambient[] = { 0.329412f, 0.223529f, 0.027451f, 1.0f };
    private float g_arrMaterial01Diffuse[] = { 0.780392f, 0.568627f, 0.113725f, 1.0f };
    private float g_arrMaterial01Specular[] = { 0.992157f, 0.941176f, 0.807843f, 1.0f };
    private float g_Material01Shininess = 0.21794872f * 128.0f;

    //
    //	Materail 11
    //
    private float g_arrMaterial11Ambient[] = { 0.2125f, 0.1275f, 0.054f, 1.0f };
    private float g_arrMaterial11Diffuse[] = { 0.714f, 0.4284f, 0.18144f, 1.0f };
    private float g_arrMaterial11Specular[] = { 0.393548f, 0.271906f, 0.166721f, 1.0f };
    private float g_Material11Shininess = 0.2f * 128.0f;

    //
    //	Materail 21
    //
    private float g_arrMaterial21Ambient[] = { 0.25f, 0.25f, 0.25f, 1.0f };
    private float g_arrMaterial21Diffuse[] = { 0.4f, 0.4f, 0.4f, 1.0f };
    private float g_arrMaterial21Specular[] = { 0.774597f, 0.774597f, 0.774597f, 1.0f };
    private float g_Material21Shininess = 0.6f * 128.0f;

    //
    //	Materail 31
    //
    private float g_arrMaterial31Ambient[] = { 0.19125f, 0.0735f, 0.0225f, 1.0f };
    private float g_arrMaterial31Diffuse[] = { 0.7038f, 0.27048f, 0.0828f, 1.0f };
    private float g_arrMaterial31Specular[] = { 0.256777f, 0.137622f, 0.296648f, 1.0f };
    private float g_Material31Shininess = 0.1f * 128.0f;

    //
    //	Materail 41
    //
    private float g_arrMaterial41Ambient[] = { 0.24725f, 0.1995f, 0.0745f, 1.0f };
    private float g_arrMaterial41Diffuse[] = { 0.75164f, 0.60648f, 0.22648f, 1.0f };
    private float g_arrMaterial41Specular[] = { 0.628281f, 0.555802f, 0.366065f, 1.0f };
    private float g_Material41Shininess = 0.4f * 128.0f;

    //
    //	Materail 51
    //
    private float g_arrMaterial51Ambient[] = { 0.19225f, 0.19225f, 0.19225f, 1.0f };
    private float g_arrMaterial51Diffuse[] = { 0.50754f, 0.50754f, 0.50754f, 1.0f };
    private float g_arrMaterial51Specular[] = { 0.508273f, 0.508273f, 0.508273f, 1.0f };
    private float g_Material51Shininess = 0.4f * 128.0f;

    //
    //	Materail 02
    //
    private float g_arrMaterial02Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    private float g_arrMaterial02Diffuse[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    private float g_arrMaterial02Specular[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    private float g_Material02Shininess = 0.25f * 128.0f;

    //
    //	Materail 12
    //
    private float g_arrMaterial12Ambient[] = { 0.0f, 0.1f, 0.06f, 1.0f };
    private float g_arrMaterial12Diffuse[] = { 0.0f, 0.50980392f, 0.50980392f, 1.0f };
    private float g_arrMaterial12Specular[] = { 0.50980392f, 0.50980392f, 0.50980392f, 1.0f };
    private float g_Material12Shininess = 0.25f * 128.0f;

    //
    //	Materail 22
    //
    private float g_arrMaterial22Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    private float g_arrMaterial22Diffuse[] = { 0.1f, 0.35f, 0.1f, 1.0f };
    private float g_arrMaterial22Specular[] = { 0.45f, 0.45f, 0.45f, 1.0f };
    private float g_Material22Shininess = 0.25f * 128.0f;

    //
    //	Materail 32
    //
    private float g_arrMaterial32Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    private float g_arrMaterial32Diffuse[] = { 0.5f, 0.0f, 0.0f, 1.0f };
    private float g_arrMaterial32Specular[] = { 0.7f, 0.6f, 0.6f, 1.0f };
    private float g_Material32Shininess = 0.25f * 128.0f;

    //
    //	Materail 42
    //
    private float g_arrMaterial42Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    private float g_arrMaterial42Diffuse[] = { 0.55f, 0.55f, 0.55f, 1.0f };
    private float g_arrMaterial42Specular[] = { 0.70f, 0.70f, 0.70f, 1.0f };
    private float g_Material42Shininess = 0.25f * 128.0f;

    //
    //	Materail 52
    //
    private float g_arrMaterial52Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    private float g_arrMaterial52Diffuse[] = { 0.5f, 0.5f, 0.0f, 1.0f };
    private float g_arrMaterial52Specular[] = { 0.60f, 0.60f, 0.50f, 1.0f };
    private float g_Material52Shininess = 0.25f * 128.0f;

    //
    //	Materail 03
    //
    private float g_arrMaterial03Ambient[] = { 0.02f, 0.02f, 0.02f, 1.0f };
    private float g_arrMaterial03Diffuse[] = { 0.01f, 0.01f, 0.01f, 1.0f };
    private float g_arrMaterial03Specular[] = { 0.4f, 0.4f, 0.4f, 1.0f };
    private float g_Material03Shininess = 0.078125f * 128.0f;

    //
    //	Materail 13
    //
    private float g_arrMaterial13Ambient[] = { 0.0f, 0.05f, 0.05f, 1.0f };
    private float g_arrMaterial13Diffuse[] = { 0.4f, 0.5f, 0.5f, 1.0f };
    private float g_arrMaterial13Specular[] = { 0.04f, 0.7f, 0.7f, 1.0f };
    private float g_Material13Shininess = 0.078125f * 128.0f;

    //
    //	Materail 23
    //
    private float g_arrMaterial23Ambient[] = { 0.0f, 0.05f, 0.0f, 1.0f };
    private float g_arrMaterial23Diffuse[] = { 0.4f, 0.5f, 0.4f, 1.0f };
    private float g_arrMaterial23Specular[] = { 0.04f, 0.7f, 0.04f, 1.0f };
    private float g_Material23Shininess = 0.078125f * 128.0f;

    //
    //	Materail 33
    //
    private float g_arrMaterial33Ambient[] = { 0.05f, 0.0f, 0.0f, 1.0f };
    private float g_arrMaterial33Diffuse[] = { 0.5f, 0.4f, 0.4f, 1.0f };
    private float g_arrMaterial33Specular[] = { 0.7f, 0.04f, 0.04f, 1.0f };
    private float g_Material33Shininess = 0.078125f * 128.0f;

    //
    //	Materail 43
    //
    private float g_arrMaterial43Ambient[] = { 0.05f, 0.05f, 0.05f, 1.0f };
    private float g_arrMaterial43Diffuse[] = { 0.5f, 0.5f, 0.5f, 1.0f };
    private float g_arrMaterial43Specular[] = { 0.7f, 0.7f, 0.7f, 1.0f };
    private float g_Material43Shininess = 0.78125f * 128.0f;

    //
    //	Materail 53
    //
    private float g_arrMaterial53Ambient[] = { 0.05f, 0.05f, 0.0f, 1.0f };
    private float g_arrMaterial53Diffuse[] = { 0.5f, 0.5f, 0.4f, 1.0f };
    private float g_arrMaterial53Specular[] = { 0.7f, 0.7f, 0.04f, 1.0f };
    private float g_Material53Shininess = 0.078125f * 128.0f;

    //-Material
    ///////////////////////////////////////////////////////////////////////////

	private float perspectiveProjectionMatrix[] = new float[16];	//	4*4 Matrix.

    private float g_fAngleLight = 0.0f;
	private int g_iDoubleTap;	// For Light

    char chAnimationAxis = 'x';
    private float g_fRotateX = 1.0f;
    private float g_fRotateY = 0.0f;
    private float g_fRotateZ = 0.0f;
	
    public GLESView(Context drawingContext)
	{
		super(drawingContext);
		context = drawingContext;
		
		//
		//	Accordingly set EGLContext to supported version of OpenGL-ES
		//
		setEGLContextClientVersion(3);
		
		//
		//	Set renderer for drawing on the GLSurfaceView.
		//
		setRenderer(this);	//	Because of this OnSurfaceCreated() get called.
		
		//
		//	Render the view only when there is change in the drawing data.
		//
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
		
		
		gestureDetector = new GestureDetector(context,this, null, false);
		gestureDetector.setOnDoubleTapListener(this);	//	this means handler i.e) who is going to handle.
	}
	
	//---------------------------------------------------------------------------
	//+	Overriden methods of GLSurfaceView.Renderer
	
	//
	//	Init code
	//
	@Override
	public void onSurfaceCreated(GL10 gl, EGLConfig config)
	{
		//	OpenGL-ES version check.
		String version = gl.glGetString(GL10.GL_VERSION);
		System.out.println("RTR: "+version);
		//	Get GLSL version.
		String glslVersion = gl.glGetString(GLES30.GL_SHADING_LANGUAGE_VERSION);
		System.out.println("RTR: "+glslVersion);
		initialize(gl);
	}
	
	//
	//	Change size code.
	//
	@Override
	public void onSurfaceChanged(GL10 unused, int width, int height)
	{
	    g_iCurrentHeight = height;
	    g_iCurrentWidth = width;
		resize(width, height);
	}
	
	//
	//	Rendering code.
	//
	@Override
	public void onDrawFrame(GL10 unused)
	{
		draw();
	}
	//-	Overriden methods of GLSurfaceView.Renderer
	//---------------------------------------------------------------------------
	
	//
	//	Handling 'onTouchEvent' is the most IMPORTANT.
	//	Because it triggers all gesture and tap events.
	//
	@Override
	public boolean onTouchEvent(MotionEvent event)
	{
		int eventAction = event.getAction();
		if (!gestureDetector.onTouchEvent(event))
		{
			super.onTouchEvent(event);
		}
		return(true);
	}
	
	//---------------------------------------------------------------------------
	//+	Abstract method from OnDoubleTapListener
	
	//
	//	onDoubleTap
	//
	@Override
	public boolean onDoubleTap(MotionEvent e)
	{
		System.out.println("RTR:Double Tap");

		g_iDoubleTap++;
		if (g_iDoubleTap > 1)
		{
			g_iDoubleTap = 0;
		}
		return(true);
	}
	
	//
	//	onDoubleTapEvent
	//
	@Override
	public boolean onDoubleTapEvent(MotionEvent e)
	{
		//	Do not write any code here bacause already written 'onDoubleTap'
		return true;
	}
	
	//
	//	onSingleTapConfirmed
	//
	@Override
	public boolean onSingleTapConfirmed(MotionEvent e)
	{
		System.out.println("RTR:Single Tap");

		chAnimationAxis += 1;
		if (chAnimationAxis > 'z')
        {
            chAnimationAxis = 'x';
        }

		return true;
	}
	
	//
	//	onDown
	//
	@Override
	public boolean onDown(MotionEvent e)
	{
		//	Do not write any code here bacause already written 'onSingleTapConfirmed'
		return true;
	}
	//-	Abstract method from OnDoubleTapListener
	//---------------------------------------------------------------------------
	
	//---------------------------------------------------------------------------
	//+	abstract method from OnGestureListener so must be implemented
	
	//
	//	onFling
	//
	@Override
	public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY)
	{
		System.out.println("RTR:Flingg");
		return true;
	}
	
	//
	//	onLongPress
	//
	@Override
	public void onLongPress(MotionEvent e)
	{
		System.out.println("RTR:Long press");
	}
	
	//
	//	onScroll
	//
	@Override
	public boolean onScroll(MotionEvent e1,MotionEvent e2, float distanceX, float distanceY)
	{
		System.out.println("RTR:Scroll");
		uninitialize();
		System.exit(0);
		return true;
	}
	
	//
	//	onShowPress
	//
	@Override
	public void onShowPress(MotionEvent e)
	{
		
	}
	
	//
	//	onSingleTapUp
	//
	@Override
	public boolean onSingleTapUp(MotionEvent e)
	{
		return true;
	}
	//-	abstract method from OnGestureListener so must be implemented
	//---------------------------------------------------------------------------
	
	//---------------------------------------------------------------------------
	//+	OpenGL methods
	
	private void initialize(GL10 gl)
	{
		//---------------------------------------------------------------------------
		//+	Vertex shader 
		shaderObjectVertex = GLES30.glCreateShader(GLES30.GL_VERTEX_SHADER);
		
		//	Vertex shader source code.
		final String vertexShaderSourceCode = String.format
		(
		"#version 300 es"+
		"\n"+
		"in vec4 vPosition;"+
		"in vec3 vNormal;"+
		"uniform mat4 u_model_matrix;"+
		"uniform mat4 u_view_matrix;"+
		"uniform mat4 u_projection_matrix;"+
		"uniform mat4 u_rotation_matrix;"+
		"uniform vec4 u_light_position;"+
		"uniform mediump int u_double_tap;"+
		"out vec3 transformed_normals;"+
		"out vec3 light_direction;"+
		"out vec3 viewer_vector;"+
		"void main(void)"+
		"{"+
			"if (1 == u_double_tap)"+
			"{"+
				"vec4 eyeCoordinates = u_view_matrix * u_model_matrix * vPosition;"											+
				"transformed_normals = normalize(mat3(u_view_matrix * u_model_matrix) * vNormal);"											+
				"vec4 rotated_light_position = u_rotation_matrix * u_light_position;"											+
				"light_direction = normalize(vec3(rotated_light_position) - eyeCoordinates.xyz);"											+
				"viewer_vector = normalize(-eyeCoordinates.xyz);"											+
			"}"+
			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;"+
		"}"
		);

		//	Provide source code to shader
		GLES30.glShaderSource(shaderObjectVertex,vertexShaderSourceCode);

		//	Compile shader and check for errors.
		GLES30.glCompileShader(shaderObjectVertex);

		int[] iShaderCompiledStatus = new int[1];
		int[] iInfoLogLength = new int[1];
		String szInfoLog = null;

		GLES30.glGetShaderiv(shaderObjectVertex,GLES30.GL_COMPILE_STATUS, iShaderCompiledStatus, 0);
		if (GLES30.GL_FALSE == iShaderCompiledStatus[0])
		{
			GLES30.glGetShaderiv(shaderObjectVertex, GLES30.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if (iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES30.glGetShaderInfoLog(shaderObjectVertex);
				System.out.println("RTR: Vertex shader compilation status Log: "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}
		//-	Vertex shader
		//---------------------------------------------------------------------------

		//---------------------------------------------------------------------------
		//+	Fragment shader
		shaderObjectFragment = GLES30.glCreateShader(GLES30.GL_FRAGMENT_SHADER);

		//	Vertex shader source code.
		final String fragmentShaderSourceCode = String.format
		(
		"#version 300 es"+
		"\n"+
		"precision highp float;"+
		"in vec3 out_phong_ads_color;"				+
		"in vec3 transformed_normals;"			+
		"in vec3 light_direction;"			+
		"in vec3 viewer_vector;"			+
		"uniform vec3 u_La;	"				+
		"uniform vec3 u_Ld;	"				+
		"uniform vec3 u_Ls;	"				+
		"uniform vec3 u_Ka;"					+
		"uniform vec3 u_Kd;"					+
		"uniform vec3 u_Ks;"					+
		"uniform float u_material_shininess;"		+
		"uniform mediump int u_double_tap;"			+
		"out vec4 vFragColor;"						+
		"void main(void)"+
		"{"+
			"vec3 phong_ads_color;"+
			"if (1 == u_double_tap)"+
			"{"+
				"vec3 normalized_transformed_normals = normalize(transformed_normals);"											+
				"vec3 normalized_light_direction = normalize(light_direction);"											+
				"vec3 normalized_viewer_vector = normalize(viewer_vector);"											+
				"float tn_dot_ld = max(dot(normalized_transformed_normals, normalized_light_direction), 0.0);"											+
				"vec3 ambient = u_La * u_Ka;"											+
				"vec3 diffuse = u_Ld * u_Kd * tn_dot_ld;"											+
				"vec3 reflection_vector = reflect(-normalized_light_direction, normalized_transformed_normals);"											+
				"vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector, normalized_viewer_vector), 0.0), u_material_shininess);"											+
				"phong_ads_color = ambient + diffuse + specular;"											+
			"}"+
			"else"+
			"{"+
				"phong_ads_color = vec3(1.0,1.0,1.0);"+
			"}"+
			"vFragColor = vec4(phong_ads_color, 1.0);"+
		"}"
		);

		//	Provide source code to shader
		GLES30.glShaderSource(shaderObjectFragment,fragmentShaderSourceCode);
		
		//	Compile shader and check for errors.
		GLES30.glCompileShader(shaderObjectFragment);
		
		iShaderCompiledStatus[0] = 0;
		iInfoLogLength[0] = 0;
		szInfoLog = null;
		
		GLES30.glGetShaderiv(shaderObjectFragment,GLES30.GL_COMPILE_STATUS, iShaderCompiledStatus, 0);
		if (GLES30.GL_FALSE == iShaderCompiledStatus[0])
		{
			GLES30.glGetShaderiv(shaderObjectFragment, GLES30.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if (iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES30.glGetShaderInfoLog(shaderObjectFragment);
				System.out.println("RTR: Fragment shader compilation status Log: "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}
		//-	Fragment shader 
		//---------------------------------------------------------------------------

		//---------------------------------------------------------------------------
		//+	Shader program
		shaderObjectProgram = GLES30.glCreateProgram();
		
		GLES30.glAttachShader(shaderObjectProgram,shaderObjectVertex);
		
		GLES30.glAttachShader(shaderObjectProgram,shaderObjectFragment);
		
		//	pre-link binding of shader program object with vertex shader attributes.
		GLES30.glBindAttribLocation(shaderObjectProgram, GLESMacros.RTR_ATTRIBUTE_POSITION, "vPosition");
		GLES30.glBindAttribLocation(shaderObjectProgram, GLESMacros.RTR_ATTRIBUTE_NORMAL, "vNormal");
		
		//	Link the 2 shaders together to shader program
		GLES30.glLinkProgram(shaderObjectProgram);
		int[] iLinkStatus = new int[1];
		iInfoLogLength[0] = 0;
		szInfoLog = null;
		GLES30.glGetProgramiv(shaderObjectProgram, GLES30.GL_LINK_STATUS, iLinkStatus, 0);
		if (GLES30.GL_FALSE == iLinkStatus[0])
		{
			GLES30.glGetProgramiv(shaderObjectProgram, GLES30.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if (iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES30.glGetProgramInfoLog(shaderObjectProgram);
				System.out.println("RTR: Shader program Link log "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}
		//-	Shader program 
		//---------------------------------------------------------------------------
		
		//	Get uniform location's.
		g_iModelMat4Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_model_matrix");
		g_iViewMat4Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_view_matrix");
		g_iProjectionMat4Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_projection_matrix");
		g_iDoubleTapUniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_double_tap");

		g_iRotationMat4Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_rotation_matrix");

		//	Light
		g_iLaVec3Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_La");
		g_iLdVec3Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_Ld");
		g_iLsVec3Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_Ls");
		g_iLightPositionVec4Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_light_position");

		g_iKaVec3Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_Ka");
		g_iKdVec3Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_Kd");
		g_iKsVec3Uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_Ks");
		g_iMaterialShininessUniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_material_shininess");
		
		//+ Sphere Data
		Sphere sphere=new Sphere();
        float sphere_vertices[]=new float[1146];
        float sphere_normals[]=new float[1146];
        float sphere_textures[]=new float[764];
        short sphere_elements[]=new short[2280];
        sphere.getSphereVertexData(sphere_vertices, sphere_normals, sphere_textures, sphere_elements);
        numVertices = sphere.getNumberOfSphereVertices();
        numElements = sphere.getNumberOfSphereElements();
		//- Sphere Data
		
		ByteBuffer byBuffer;
		FloatBuffer fVerticesBuffer;

		///////////////////////////////////////////////////////////////////
		//+	VAO Sphere
		
		GLES30.glGenVertexArrays(1, vao_sphere, 0);
		GLES30.glBindVertexArray(vao_sphere[0]);
		
		///////////////////////////////////////////////////////////////////
		//+	VBO Position
		GLES30.glGenBuffers(1, vbo_sphere_position, 0);
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo_sphere_position[0]);
		
		//	Convert array to byte data.
		byBuffer = ByteBuffer.allocateDirect(sphere_vertices.length * 4);
		byBuffer.order(ByteOrder.nativeOrder());
		fVerticesBuffer = byBuffer.asFloatBuffer();
		fVerticesBuffer.put(sphere_vertices);
		fVerticesBuffer.position(0);
		
		GLES30.glBufferData(
							GLES30.GL_ARRAY_BUFFER,
							sphere_vertices.length * 4,
							fVerticesBuffer,
							GLES30.GL_STATIC_DRAW
							);
							
		GLES30.glVertexAttribPointer(GLESMacros.RTR_ATTRIBUTE_POSITION, 3, GLES30.GL_FLOAT, false, 0, 0);
		
		GLES30.glEnableVertexAttribArray(GLESMacros.RTR_ATTRIBUTE_POSITION);
		
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);
		//-	VBO Position
		///////////////////////////////////////////////////////////////////
		
		///////////////////////////////////////////////////////////////////
		//+	VBO Normal
		GLES30.glGenBuffers(1, vbo_sphere_normal, 0);
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo_sphere_normal[0]);
		
		//	Convert array to byte data.
		byBuffer = ByteBuffer.allocateDirect(sphere_normals.length * 4);
		byBuffer.order(ByteOrder.nativeOrder());
		fVerticesBuffer = byBuffer.asFloatBuffer();
		fVerticesBuffer.put(sphere_normals);
		fVerticesBuffer.position(0);
		
		GLES30.glBufferData(
							GLES30.GL_ARRAY_BUFFER,
							sphere_normals.length * 4,
							fVerticesBuffer,
							GLES30.GL_STATIC_DRAW
							);
							
		GLES30.glVertexAttribPointer(GLESMacros.RTR_ATTRIBUTE_NORMAL, 3, GLES30.GL_FLOAT, false, 0, 0);
		
		GLES30.glEnableVertexAttribArray(GLESMacros.RTR_ATTRIBUTE_NORMAL);
		
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);
		//-	VBO Normal
		///////////////////////////////////////////////////////////////////
		
		///////////////////////////////////////////////////////////////////
		//+	VBO Element
		GLES30.glGenBuffers(1, vbo_sphere_element, 0);
		GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
		
		//	Convert array to byte data.
		ShortBuffer shVerticesBuffer;
		
		byBuffer = ByteBuffer.allocateDirect(sphere_elements.length * 2);
		byBuffer.order(ByteOrder.nativeOrder());
		shVerticesBuffer = byBuffer.asShortBuffer();
		shVerticesBuffer.put(sphere_elements);
		shVerticesBuffer.position(0);
		
		GLES30.glBufferData(
							GLES30.GL_ELEMENT_ARRAY_BUFFER,
							sphere_elements.length * 2,
							shVerticesBuffer,
							GLES30.GL_STATIC_DRAW
							);
							
		GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, 0);
		//-	VBO Element
		///////////////////////////////////////////////////////////////////
		
		GLES30.glBindVertexArray(0);
		
		//-	VAO Triangle
		///////////////////////////////////////////////////////////////////
		
		//	Enable depth testing.
		GLES30.glEnable(GLES30.GL_DEPTH_TEST);
		//	dept test to do
		GLES30.glDepthFunc(GLES30.GL_LEQUAL);
		//	We will always cull backfaces for better performance
		GLES30.glEnable(GLES30.GL_CULL_FACE);
		
		//	Set the background frame color.
		GLES30.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		
		//	Set Projection matrix to identity.
		Matrix.setIdentityM(perspectiveProjectionMatrix, 0);
	}
	
	private void resize(int width, int height)
	{
		if (height == 0)
			height = 1;

		//	Adjust the viewport based on geometry changes, 
		//	such as screen rotation.
		GLES30.glViewport(0, 0, width, height);
		Matrix.perspectiveM(perspectiveProjectionMatrix, 0, 45.0f, (float)(width/6)/(float)(height/4), 0.1f, 100.0f);
	}
	
	public void draw()
	{
		//	Draw background color.
		GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT | GLES30.GL_DEPTH_BUFFER_BIT);
		
		//	Use shader program.
		GLES30.glUseProgram(shaderObjectProgram);
		
		if (1 == g_iDoubleTap)
		{
			GLES30.glUniform1i(g_iDoubleTapUniform, 1);

			GLES30.glUniform3fv(g_iLaVec3Uniform, 1, g_farrLightAmbient, 0);
			GLES30.glUniform3fv(g_iLdVec3Uniform, 1, g_farrLightDiffuse, 0);
			GLES30.glUniform3fv(g_iLsVec3Uniform, 1, g_farrLightSpecular, 0);
		}
		else
		{
			GLES30.glUniform1i(g_iDoubleTapUniform, 0);
		}

        float fHeightMulti = 0.03f;
        float fWidthMulti = 0.03f;

        float fWidthAdjust = 0.15f;
        float fHeightAdjust = 0.25f;
        int iViewPortWidth= g_iCurrentWidth / 6;
        int iViewPortHeight= g_iCurrentHeight / 4;

        //
        //	First column,
        //
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
		Sphere00();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere10();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere20();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere30();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere40();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere50();

        //
        //	Second column,
        //
        fHeightMulti = fHeightMulti + fHeightAdjust;
        fWidthMulti = 0.03f;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere01();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere11();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere21();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere31();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere41();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere51();

        //
        //	Third column,
        //
        fHeightMulti = fHeightMulti + fHeightAdjust;
        fWidthMulti = 0.03f;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere02();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere12();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere22();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere32();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere42();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere52();

        //
        //	Fourth column,
        //
        fHeightMulti = fHeightMulti + fHeightAdjust;
        fWidthMulti = 0.03f;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere03();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere13();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere23();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere33();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere43();

        fWidthMulti = fWidthMulti + fWidthAdjust;
        GLES30.glViewport((int)(g_iCurrentWidth * fWidthMulti), (int)(g_iCurrentHeight * fHeightMulti), iViewPortWidth, iViewPortHeight);
        Sphere53();

        //	un-use program.
		GLES30.glUseProgram(0);

		update();

		//	render/flush
		requestRender();	//	Same a SwapBuffers()
	}

	void Sphere00()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial00Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial00Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial00Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material00Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere10()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial10Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial10Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial10Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material10Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere20()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial20Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial20Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial20Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material20Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere30()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial30Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial30Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial30Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material30Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere40()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial40Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial40Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial40Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material40Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere50()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial50Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial50Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial50Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material50Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    //
    //  Second column
    //

    void Sphere01()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial01Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial01Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial01Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material01Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere11()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial11Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial11Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial11Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material11Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere21()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial21Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial21Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial21Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material21Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere31()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial31Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial31Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial31Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material31Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere41()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial41Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial41Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial41Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material41Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere51()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial51Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial51Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial51Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material51Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    //
    //  Third column
    //
    void Sphere02()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial02Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial02Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial02Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material02Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere12()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial12Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial12Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial12Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material12Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere22()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial22Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial22Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial22Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material22Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere32()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial32Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial32Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial32Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material32Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere42()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial42Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial42Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial42Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material42Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere52()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial52Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial52Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial52Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material52Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    //
    //  Fourth column
    //
    void Sphere03()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial03Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial03Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial03Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material03Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere13()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial13Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial13Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial13Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material13Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere23()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial23Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial23Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial23Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material23Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere33()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial33Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial33Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial33Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material33Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere43()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial43Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial43Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial43Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material43Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void Sphere53()
    {
        //	OpenGL-ES drawing.
        float modelMatrix[] = new float[16];
        float viewMatrix[] = new float[16];
        float rotationMatrix[] = new float[16];

        Matrix.setIdentityM(modelMatrix, 0);
        Matrix.setIdentityM(viewMatrix, 0);
        Matrix.setIdentityM(rotationMatrix, 0);

        Matrix.translateM(modelMatrix, 0, 0.0f,0.0f,-2.0f);

        GLES30.glUniformMatrix4fv(g_iModelMat4Uniform, 1, false, modelMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iViewMat4Uniform, 1, false, viewMatrix, 0);
        GLES30.glUniformMatrix4fv(g_iProjectionMat4Uniform, 1, false, perspectiveProjectionMatrix, 0);

        if (chAnimationAxis == 'x') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 1.0f, 0.0f, 0.0f);    //  X-axis rotation
            g_farrLightPosition[1] = g_fAngleLight;
            g_farrLightPosition[0] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else if (chAnimationAxis == 'y') {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 1.0f, 0.0f);    //  Y-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }
        else
        {
            Matrix.rotateM(rotationMatrix, 0, g_fAngleLight, 0.0f, 0.0f, 1.0f);    //  Z-axis rotation
            g_farrLightPosition[0] = g_fAngleLight;
            g_farrLightPosition[1] = g_farrLightPosition[2] = 0.0f;
            GLES30.glUniform4fv(g_iLightPositionVec4Uniform, 1, g_farrLightPosition, 0);
            GLES30.glUniformMatrix4fv(g_iRotationMat4Uniform, 1, false, rotationMatrix, 0);
        }

        GLES30.glUniform3fv(g_iKaVec3Uniform, 1, g_arrMaterial53Ambient, 0);
        GLES30.glUniform3fv(g_iKdVec3Uniform, 1, g_arrMaterial53Diffuse, 0);
        GLES30.glUniform3fv(g_iKsVec3Uniform, 1, g_arrMaterial53Specular, 0);
        GLES30.glUniform1f(g_iMaterialShininessUniform, g_Material53Shininess);

        //	Bind VAO
        GLES30.glBindVertexArray(vao_sphere[0]);

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo_sphere_element[0]);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, numElements, GLES30.GL_UNSIGNED_SHORT, 0);

        // unbind VAO
        GLES30.glBindVertexArray(0);
    }

    void update()
    {
        float fSpeed = 1.0f;

        g_fAngleLight = g_fAngleLight + fSpeed;
        if (g_fAngleLight >= 360)
        {
            g_fAngleLight = 0.0f;
        }
    }
	
	void uninitialize()
	{
		System.out.println("RTR:Uninitialize-->");
		if (vao_sphere[0] != 0)
		{
			GLES30.glDeleteVertexArrays(1, vao_sphere, 0);
			vao_sphere[0] = 0;
		}
		
		if (vbo_sphere_position[0] != 0)
		{
			GLES30.glDeleteBuffers(1, vbo_sphere_position, 0);
			vbo_sphere_position[0] = 0;
		}
		
		if (vbo_sphere_normal[0] != 0)
		{
			GLES30.glDeleteBuffers(1, vbo_sphere_normal, 0);
			vbo_sphere_normal[0] = 0;
		}
		
		if (vbo_sphere_element[0] != 0)
		{
			GLES30.glDeleteBuffers(1, vbo_sphere_element, 0);
			vbo_sphere_element[0] = 0;
		}
		
		if (shaderObjectProgram != 0)
		{
			if (shaderObjectVertex != 0)
			{
				GLES30.glDetachShader(shaderObjectProgram, shaderObjectVertex);
				GLES30.glDeleteShader(shaderObjectVertex);
				shaderObjectVertex = 0;
			}
			if (shaderObjectFragment != 0)
			{
				GLES30.glDetachShader(shaderObjectProgram, shaderObjectFragment);
				GLES30.glDeleteShader(shaderObjectFragment);
				shaderObjectFragment = 0;
			}
		}
		
		if (shaderObjectProgram != 0)
		{
			GLES30.glDeleteProgram(shaderObjectProgram);
			shaderObjectProgram = 0;
		}
		System.out.println("RTR:Uninitialize<--");
	}
	//-	OpenGL methods
	//---------------------------------------------------------------------------
}
