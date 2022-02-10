package com.jayshree.ndkdatatypes;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Example of a call to a native method
        TextView tv = findViewById(R.id.sample_text);
        tv.setText("String: "+ stringFromJNI() + "\nInt: " + intFromJNI() + "\nLong: " + longFromJNI() + "\nByte: " + byteFromJNI() + "\nSum of 2 int(10,20): " + sumOfTwoInt(10,20));
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();

    public native int intFromJNI();

    public native long longFromJNI();

    public native byte byteFromJNI();

    public native int sumOfTwoInt(int i, int j);
}
