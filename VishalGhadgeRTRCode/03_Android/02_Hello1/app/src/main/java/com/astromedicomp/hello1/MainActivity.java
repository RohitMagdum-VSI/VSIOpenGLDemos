package com.astromedicomp.hello1;

import android.app.Activity;
import android.os.Bundle;
import android.view.Window;
import android.widget.TextView;
import android.view.WindowManager;	//	for "WindowManager"
import android.content.pm.ActivityInfo;
import android.view.Gravity;
import android.graphics.Color;

public class MainActivity extends Activity {

	//private MyView myView;
	
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
		
		//
		//	Fullscreen.
		//
		this.requestWindowFeature(Window.FEATURE_NO_TITLE);
		this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,WindowManager.LayoutParams.FLAG_FULLSCREEN);
		
		//
		//	Make landscape.
		//
		MainActivity.this.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
		
		//
		//	Write HelloWorld using MyView.
		//
		TextView myTextView = new TextView(this);
		myTextView.setText("Hello World 1");
		myTextView.setTextSize(60);
		myTextView.setTextColor(Color.rgb(0,100,0));
		myTextView.setGravity(Gravity.CENTER);
		setContentView(myTextView);
		
        //setContentView(R.layout.activity_main);
    }
}
