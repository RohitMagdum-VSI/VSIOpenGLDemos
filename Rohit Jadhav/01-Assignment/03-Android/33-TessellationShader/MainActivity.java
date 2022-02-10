package com.rohit_r_jadhav.tessillation_shader;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;

import android.view.Window;
import android.view.WindowManager;
import android.content.pm.ActivityInfo;
import android.view.View;

public class MainActivity extends AppCompatActivity{

	private GLESView glesView;

	@Override
	protected void onCreate(Bundle savedInstanceState){
		super.onCreate(savedInstanceState);

		//For Disableing Titile Bar
		this.supportRequestWindowFeature(Window.FEATURE_NO_TITLE);

		this.getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_HIDE_NAVIGATION | 
												View.SYSTEM_UI_FLAG_IMMERSIVE);

		//For FullScreen
		this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

		//For Forced Landscape
		MainActivity.this.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

		glesView = new GLESView(this);

		setContentView(glesView);
	}

	@Override
	public void onPause(){
		super.onPause();
	}

	@Override
	public void onResume(){
		super.onResume();
	}
}


