package com.astromedicomp.opengl_window;

//Default supplied package by android SDK
import android.app.Activity;
import android.os.Bundle;

//Later added package
import android.view.Window;
import android.view.WindowManager;
import android.content.pm.ActivityInfo;

public class MainActivity extends Activity {

	private GLESView glesView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

	//This is done to remove ActionBar
	requestWindowFeature(Window.FEATURE_NO_TITLE);

	//This is done to make Window Fullscreen
	getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,WindowManager.LayoutParams.FLAG_FULLSCREEN);

     super.onCreate(savedInstanceState);
     //setContentView(R.layout.activity_main);
	
	//Force activity window orientation to landscape
	MainActivity.this.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
	
	glesView = new GLESView(this);

	//Set view as content view of the Activity
	setContentView(glesView);
    }

	@Override
	protected void onPause()
	{
		super.onPause();
	}

	@Override
	protected void onResume()
	{
		super.onResume();
	}
}
