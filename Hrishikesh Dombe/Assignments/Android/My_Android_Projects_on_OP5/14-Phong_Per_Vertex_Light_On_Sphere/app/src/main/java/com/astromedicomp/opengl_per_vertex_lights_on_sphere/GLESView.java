package com.astromedicomp.opengl_per_vertex_lights_on_sphere;

import android.content.Context; //For drawing context related
import android.opengl.GLSurfaceView; //For OpenGL Surface View and all related
import javax.microedition.khronos.opengles.GL10; //For OpenGLES 1.0 needed as param type GL10
import javax.microedition.khronos.egl.EGLConfig; //For EGLConfig needed as param type EGLConfig
import android.opengl.GLES32; // For OpenGLES 3.2
import android.view.Gravity;
import android.view.MotionEvent; // For "MotionEvent"
import android.view.GestureDetector; // For GestureDetector
import android.view.GestureDetector.OnGestureListener; // OnGestureListener
import android.view.GestureDetector.OnDoubleTapListener; // OnDoubleTapListener

//For vbo
import java.nio.ByteBuffer;//nio For non-blocking I/O
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

//For Matrix math
import android.opengl.Matrix;

//A View for OpenGLES3 graphics which also receives touch events
public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnGestureListener, OnDoubleTapListener
{
	//Class variables or class fields
	private final Context context;

	private GestureDetector gestureDetector;

	//Shader Object
	private int vertexShaderObject;
	private int fragmentShaderObject;

	//Program Object
	private int shaderProgramObject;

	//As there are no pointers in java, when we want to use any variable as out parameter
	//we use array with only one member.
	//Vertex Array Object
	private int[] vao_sphere = new int[1];
	private int[] vbo_position = new int[1];
	private int[] vbo_normal = new int[1];
	private int[] vbo_elements = new int[1];

	private float angleSphere;

	private int modelMatrixUniform, viewMatrixUniform, projectionMatrixUniform;
	private int laUniform,ldUniform,lsUnifrom,lightPositionUniform;
	private int kaUniform,kdUniform,ksUniform,materialShininessUniform;

	private float light_ambient[] = {0.0f,0.0f,0.0f,1.0f};
	private float light_diffuse[] = {1.0f,1.0f,1.0f,1.0f};
	private float light_specular[] = {1.0f,1.0f,1.0f,1.0f};
	private float light_position[] = {100.0f,100.0f,100.0f,1.0f};

	private float material_ambient[] = {0.0f,0.0f,0.0f,1.0f};
	private float material_diffuse[] = {1.0f,1.0f,1.0f,1.0f};
	private float material_specular[] = {1.0f,1.0f,1.0f,1.0f};
	private float material_shininess = 50.0f;

	private int doubleTapUniform;

	//4 x 4 matrix
	private float perspectiveProjectionMatrix[] = new float[16];

	private int singleTap,doubleTap;

	Sphere sphere = new Sphere();
	float sphere_vertices[] = new float[1146];
	float sphere_normals[] = new float[1146];
	float sphere_textures[] = new float[764];
	short sphere_elements[] = new short[2280];

	int iNumVertices,iNumElements;

	public GLESView(Context drawingContext)
	{
		super(drawingContext);

		context = drawingContext;

		//Accordingly set EGLContext to current supported version of OpenGL-ES
		setEGLContextClientVersion(3);

		//Set Renderer for drawing on GLSurfaceView
		setRenderer(this);

		//Render the view only when there is a change in drawing data
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);

		//this means 'handle' i.e who is going to handle 
		gestureDetector = new GestureDetector(context,this,null,false);
	
		//this means 'handle' i.e who is going to handle
		gestureDetector.setOnDoubleTapListener(this);
	}

	//overriden method of GLSurfaceView.Renderer(Init Code)
	@Override
	public void onSurfaceCreated(GL10 gl,EGLConfig config)
	{
		//OpenGL-ES version check
		String version = gl.glGetString(GL10.GL_VERSION);
		System.out.println("HAD: OpenGL-ES Version"+version);//"+" for concatination

		String glslVersion=gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
		System.out.println("HAD: GLSL Version"+glslVersion);

		initialize(gl);
	}

	//overriden method of GLSurfaceView.Renderer(Change Size Code)
	@Override
	public void onSurfaceChanged(GL10 unused,int width,int height)
	{
		resize(width,height);
	}

	//overriden method of GLSurfaceView.Renderer(Rendering Code)
	@Override
	public void onDrawFrame(GL10 unused)
	{
		draw();
	}

	//Handling 'onTouchEvent' Is The Most IMPORTANT, Because It Triggers All Gesture And Events
	@Override
	public boolean onTouchEvent(MotionEvent e)
	{
		//code
		int eventaction=e.getAction();
		if(!gestureDetector.onTouchEvent(e))
			super.onTouchEvent(e);
		return(true);
	}

	//abstract method from OnDoubleTapListener so must be implemented
	@Override
	public boolean onDoubleTap(MotionEvent e)
	{
		doubleTap++;
		if(doubleTap>1)
			doubleTap=0;
		return(true);
	}

	//abstract method from OnDoubleTapListener so must be implemented
	@Override
	public boolean onDoubleTapEvent(MotionEvent e)
	{
		//Do not write any code here because already written 'onDOubleTap'
		return(true);
	}

	//abstract method from OnDoubleTapListener so must be implemented
	@Override
	public boolean onSingleTapConfirmed(MotionEvent e)
	{
		singleTap++;
		if(singleTap>1)
			singleTap=0;
		return(true);
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public boolean onDown(MotionEvent e)
	{
		//Do not write any code here because already written 'onSingleTapConfirmed'
		return(true);
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public boolean onFling(MotionEvent e1,MotionEvent e2,float velocityX,float velocityY)
	{
		return(true);
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public void onLongPress(MotionEvent e)
	{
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public boolean onScroll(MotionEvent e1,MotionEvent e2,float distanceX,float distanceY)
	{
		uninitialize();
		System.exit(0);
		return(true);
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public void onShowPress(MotionEvent e)
	{

	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public boolean onSingleTapUp(MotionEvent e)
	{
		return(true);
	}

	private void initialize(GL10 gl)
	{
		//Vertex Shader
		//Create Vertex Shader

		vertexShaderObject=GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		//vertex shader source code
		final String vertexShaderSourceCode = String.format
		(
			"#version 320 es"+
			"\n"+
			"in vec4 vPosition;"+
			"in vec3 vNormal;"+
			"uniform mat4 u_model_matrix;"+
			"uniform mat4 u_view_matrix;"+
			"uniform mat4 u_projection_matrix;"+
			"uniform int u_double_tap;"+
			"uniform vec3 u_La;"+
			"uniform vec3 u_Ld;"+
			"uniform vec3 u_Ls;"+
			"uniform vec4 u_light_position;"+
			"uniform vec3 u_Ka;"+
			"uniform vec3 u_Kd;"+
			"uniform vec3 u_Ks;"+
			"uniform float u_material_shininess;"+
			"out vec3 phong_ads_color;"+
			"void main(void)"+
			"{"+
			"if(u_double_tap == 1)"+
			"{"+
			"vec4 eye_coordinates = u_view_matrix * u_model_matrix * vPosition;"+
			"vec3 transformed_normals = normalize(mat3(u_view_matrix * u_model_matrix) * vNormal);"+
			"vec3 light_direction = normalize(vec3(u_light_position) - eye_coordinates.xyz);"+
			"float tn_dot_ld = max(dot(transformed_normals, light_direction),0.0);"+
			"vec3 ambient = u_La * u_Ka;"+
			"vec3 diffuse = u_Ld * u_Kd * tn_dot_ld;"+
			"vec3 reflection_vector = reflect(-light_direction, transformed_normals);"+
			"vec3 viewer_vector = normalize(-eye_coordinates.xyz);"+
			"vec3 specular = u_Ls * u_Ks * pow(max(dot(reflection_vector,viewer_vector),0.0),u_material_shininess);"+
			"phong_ads_color = ambient + diffuse + specular;"+
			"}"+
			"else"+
			"{"+
			"phong_ads_color = vec3(1.0,1.0,1.0);"+
			"}"+
			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;"+
			"}"
		);

		//Provide Source Code to Shader
		GLES32.glShaderSource(vertexShaderObject,vertexShaderSourceCode);

		//Compile Shader & Check errors
		GLES32.glCompileShader(vertexShaderObject);
		int[] iShaderCompiledStatus = new int[1];
		int[] iInfoLogLength = new int[1];
		String szInfoLog = null; 
		GLES32.glGetShaderiv(vertexShaderObject,GLES32.GL_COMPILE_STATUS,iShaderCompiledStatus,0);
		if(iShaderCompiledStatus[0] == GLES32.GL_FALSE)
		{
			GLES32.glGetShaderiv(vertexShaderObject,GLES32.GL_INFO_LOG_LENGTH,iInfoLogLength,0);
			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES32.glGetShaderInfoLog(vertexShaderObject);
				System.out.println("HAD: Vertex Shader Compilation Log = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//Fragment Shader
		//Create Shader
		fragmentShaderObject = GLES32.glCreateShader(GLES32.GL_FRAGMENT_SHADER);

		//Fragment Shader Source Code
		final String fragmentShaderSourceCode = String.format
		(
			"#version 320 es"+
			"\n"+
			"precision highp float;"+
			"in vec3 phong_ads_color;"+
			"out vec4 FragColor;"+
			"void main(void)"+
			"{"+
			"FragColor=vec4(phong_ads_color,1.0);"+
			"}"
		);

		GLES32.glShaderSource(fragmentShaderObject,fragmentShaderSourceCode);

		//Compile Shader and check for errors
		GLES32.glCompileShader(fragmentShaderObject);
		iShaderCompiledStatus[0] = 0;//reinitialize
		iInfoLogLength[0] = 0;//reinitialize
		szInfoLog=null;//reinitialize

		GLES32.glGetShaderiv(fragmentShaderObject,GLES32.GL_COMPILE_STATUS,iShaderCompiledStatus,0);
		if(iShaderCompiledStatus[0] == GLES32.GL_FALSE)
		{
			GLES32.glGetShaderiv(fragmentShaderObject,GLES32.GL_INFO_LOG_LENGTH,iInfoLogLength,0);
			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES32.glGetShaderInfoLog(fragmentShaderObject);
				System.out.println("HAD: Fragment Shader Compilation Log = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//Create Shader Program
		shaderProgramObject = GLES32.glCreateProgram();

		//Attach Vertex Shader to Program Object
		GLES32.glAttachShader(shaderProgramObject,vertexShaderObject);

		//Attach Fragment Shader to Program Object
		GLES32.glAttachShader(shaderProgramObject,fragmentShaderObject);

		//Pre-link binding of shader program object with vertex shader attributes
		GLES32.glBindAttribLocation(shaderProgramObject,GLESMacros.HAD_ATTRIBUTE_VERTEX,"vPosition");
		GLES32.glBindAttribLocation(shaderProgramObject,GLESMacros.HAD_ATTRIBUTE_NORMAL,"vNormal");

		//Link 2 shaders together to shader program object
		GLES32.glLinkProgram(shaderProgramObject);
		int[] iShaderProgramLinkStatus = new int[1];
		iInfoLogLength[0] = 0;//reinitialize
		szInfoLog = null;//reinitialize

		GLES32.glGetProgramiv(shaderProgramObject,GLES32.GL_LINK_STATUS,iShaderProgramLinkStatus,0);
		if(iShaderProgramLinkStatus[0] == GLES32.GL_FALSE)
		{
			GLES32.glGetProgramiv(shaderProgramObject,GLES32.GL_INFO_LOG_LENGTH,iInfoLogLength,0);
			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES32.glGetProgramInfoLog(shaderProgramObject);
				System.out.println("HAD: Shader Program Link Log = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//Get MVP uniform location
		modelMatrixUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_model_matrix");
		viewMatrixUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_view_matrix");
		projectionMatrixUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_projection_matrix");
		doubleTapUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_double_tap");
		laUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_La");
		ldUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_Ld");
		lsUnifrom = GLES32.glGetUniformLocation(shaderProgramObject,"u_Ls");
		kaUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_Ka");
		kdUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_Kd");
		ksUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_Ks");
		lightPositionUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_light_position");
		materialShininessUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_material_shininess");

		//Vertices, Color,Shader Attribs, Vbo, Vao initializations
		sphere.getSphereVertexData(sphere_vertices,sphere_normals,sphere_textures,sphere_elements);
		iNumVertices = sphere.getNumberOfSphereVertices();
		iNumElements = sphere.getNumberOfSphereElements();

		/*****************Square*****************/
		GLES32.glGenVertexArrays(1,vao_sphere,0);
		GLES32.glBindVertexArray(vao_sphere[0]);

		/****************Sphere Position**************/
		GLES32.glGenBuffers(1,vbo_position,0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,vbo_position[0]);

		ByteBuffer byteBuffer=ByteBuffer.allocateDirect(sphere_vertices.length*4);
		byteBuffer.order(ByteOrder.nativeOrder());
		FloatBuffer verticesBuffer=byteBuffer.asFloatBuffer();
		verticesBuffer.put(sphere_vertices);
		verticesBuffer.position(0);

		GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,sphere_vertices.length*4,verticesBuffer,GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(GLESMacros.HAD_ATTRIBUTE_VERTEX,3,GLES32.GL_FLOAT,false,0,0);

		GLES32.glEnableVertexAttribArray(GLESMacros.HAD_ATTRIBUTE_VERTEX);

		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,0);

		/****************Sphere Color**************/
		GLES32.glGenBuffers(1,vbo_normal,0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,vbo_normal[0]);

		byteBuffer=ByteBuffer.allocateDirect(sphere_normals.length*4);
		byteBuffer.order(ByteOrder.nativeOrder());
		FloatBuffer nomralBuffer=byteBuffer.asFloatBuffer();
		nomralBuffer.put(sphere_normals);
		nomralBuffer.position(0);

		GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,sphere_normals.length*4,nomralBuffer,GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(GLESMacros.HAD_ATTRIBUTE_NORMAL,3,GLES32.GL_FLOAT,false,0,0);

		GLES32.glEnableVertexAttribArray(GLESMacros.HAD_ATTRIBUTE_NORMAL);

		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,0);

		/****************Sphere Elements************/
		GLES32.glGenBuffers(1,vbo_elements,0);
		GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER,vbo_elements[0]);

		byteBuffer=ByteBuffer.allocateDirect(sphere_elements.length*2);
		byteBuffer.order(ByteOrder.nativeOrder());
		ShortBuffer elementBuffer=byteBuffer.asShortBuffer();
		elementBuffer.put(sphere_elements);
		elementBuffer.position(0);

		GLES32.glBufferData(GLES32.GL_ELEMENT_ARRAY_BUFFER,sphere_elements.length*2,elementBuffer,GLES32.GL_STATIC_DRAW);

		GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER,0);

		GLES32.glBindVertexArray(0);

		//Enable DepthTest
		GLES32.glEnable(GLES32.GL_DEPTH_TEST);

		//Specify Depth test to be done
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);

		//We will always cull the back faces for better performance
		//GLES32.glEnable(GLES32.GL_CULL_FACE);

		//Set the background frame color
		GLES32.glClearColor(0.0f,0.0f,0.0f,1.0f);

		singleTap=0;
		doubleTap=0;

		//Set ProjectionMatrix to identity matrix
		Matrix.setIdentityM(perspectiveProjectionMatrix,0);
	}

	private void resize(int width,int height)
	{
		//Adjust the viewport based on geometry changes such as screen rotation
		GLES32.glViewport(0,0,width,height);

		//Perspective Projection
		Matrix.perspectiveM(perspectiveProjectionMatrix,0,45.0f,(float)width/(float)height,0.1f,100.0f);
	}

	public void draw()
	{
		//Draw background color
		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT|GLES32.GL_DEPTH_BUFFER_BIT);

		//Use Shader Program
		GLES32.glUseProgram(shaderProgramObject);

		if(doubleTap == 1)
		{
			GLES32.glUniform1i(doubleTapUniform,1);

			GLES32.glUniform3fv(laUniform,1,light_ambient,0);
			GLES32.glUniform3fv(ldUniform,1,light_diffuse,0);
			GLES32.glUniform3fv(lsUnifrom,1,light_specular,0);
			GLES32.glUniform4fv(lightPositionUniform,1,light_position,0);

			GLES32.glUniform3fv(kaUniform,1,material_ambient,0);
			GLES32.glUniform3fv(kdUniform,1,material_diffuse,0);
			GLES32.glUniform3fv(ksUniform,1,material_specular,0);
			GLES32.glUniform1f(materialShininessUniform,material_shininess);
		}
		else
		{
			GLES32.glUniform1i(doubleTapUniform,0);
		}

		float modelMatrix[] = new float[16];
		float viewMatrix[] = new float[16];
		float rotationMatrix[] = new float[16];

		//set ModelView and ModelViewProjection matrices to identity
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(viewMatrix,0);
		Matrix.setIdentityM(rotationMatrix,0);

		Matrix.translateM(modelMatrix,0,0.0f,0.0f,-2.0f);

		Matrix.rotateM(rotationMatrix,0,angleSphere,0.0f,1.0f,0.0f);

		Matrix.multiplyMM(modelMatrix,0,modelMatrix,0,rotationMatrix,0);

		GLES32.glUniformMatrix4fv(modelMatrixUniform,1,false,modelMatrix,0);

		GLES32.glUniformMatrix4fv(viewMatrixUniform,1,false,viewMatrix,0);

		GLES32.glUniformMatrix4fv(projectionMatrixUniform,1,false,perspectiveProjectionMatrix,0);

		GLES32.glBindVertexArray(vao_sphere[0]);

		GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER,vbo_elements[0]);
		GLES32.glDrawElements(GLES32.GL_TRIANGLES,iNumElements,GLES32.GL_UNSIGNED_SHORT,0);
		GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER,0);

		GLES32.glBindVertexArray(0);

		//un-use shader program
		GLES32.glUseProgram(0);

		update();
		//Like SwapBuffers() in Windows
		requestRender();
	}

	private void update()
	{
		if(singleTap==1)
		{
			angleSphere=angleSphere+1.0f;
			if(angleSphere>=360.0f)
				angleSphere=angleSphere-360.0f;
		}
	}

	public void uninitialize()
	{
		if(vao_sphere[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1,vao_sphere,0);
			vao_sphere[0]=0;
		}

		if(vbo_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1,vbo_position,0);
			vbo_position[0]=0;
		}

		if(vbo_normal[0] != 0)
		{
			GLES32.glDeleteBuffers(1,vbo_normal,0);
			vbo_normal[0]=0;
		}

		if(vbo_elements[0] != 0)
		{
			GLES32.glDeleteBuffers(1,vbo_elements,0);
			vbo_elements[0]=0;
		}

		if(shaderProgramObject != 0)
		{
			if(vertexShaderObject != 0)
			{
				GLES32.glDetachShader(shaderProgramObject,vertexShaderObject);
				GLES32.glDeleteShader(vertexShaderObject);
				vertexShaderObject=0;
			}

			if(fragmentShaderObject != 0)
			{
				GLES32.glDetachShader(shaderProgramObject,fragmentShaderObject);
				GLES32.glDeleteShader(fragmentShaderObject);
				fragmentShaderObject=0;
			}
		}

		if(shaderProgramObject != 0)
		{
			GLES32.glDeleteProgram(shaderProgramObject);
			shaderProgramObject=0;
		}
	}
}