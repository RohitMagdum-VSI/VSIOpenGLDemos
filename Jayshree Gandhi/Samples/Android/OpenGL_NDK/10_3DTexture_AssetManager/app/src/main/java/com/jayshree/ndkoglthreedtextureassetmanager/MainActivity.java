package com.jayshree.ndkoglthreedtextureassetmanager;

//by default
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;

//by user
//for Window
import android.view.Window;

//for WindowManager
import android.view.WindowManager;

//for ActivityInfo where pm => package manager
import android.content.pm.ActivityInfo;

//for Color
import android.graphics.Color;

//for View
import android.view.View;

public class MainActivity extends AppCompatActivity {

    private GLESView glesView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        //get rid of title bar
        this.supportRequestWindowFeature(Window.FEATURE_NO_TITLE);

        //get rid of navigation bar
        this.getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_HIDE_NAVIGATION | View.SYSTEM_UI_FLAG_IMMERSIVE);

        //make fullscreen
        this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,WindowManager.LayoutParams.FLAG_FULLSCREEN);

        //forced landscape orientation
        this.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        //set background color
        //this.getWindow().getDecorView().setBackgroundColor(Color.BLACK);

        //define your own view
        glesView = new GLESView(this);

        //set this view
        setContentView(glesView);

        //asset manager way
        GLESNativeLib.readAssetsNative(getAssets());
    }

    @Override
    protected void onPause(){
        super.onPause();
    }

    @Override
    protected void onResume(){
        super.onResume();
    }
}
