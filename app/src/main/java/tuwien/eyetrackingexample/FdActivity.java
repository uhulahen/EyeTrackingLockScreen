package tuwien.eyetrackingexample;

import android.app.Activity;
import android.app.KeyguardManager;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.os.Vibrator;
import android.telephony.PhoneStateListener;
import android.telephony.TelephonyManager;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.SeekBar;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import utils.LockscreenService;
import utils.LockscreenUtils;

public class FdActivity extends Activity implements CvCameraViewListener2, LockscreenUtils.OnLockStatusChangedListener {

    private static final String TAG = "OCVSample::Activity";
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    public static final int JAVA_DETECTOR = 0;
    private static final int TM_SQDIFF = 0;
    private static final int TM_SQDIFF_NORMED = 1;
    private static final int TM_CCOEFF = 2;
    private static final int TM_CCOEFF_NORMED = 3;
    private static final int TM_CCORR = 4;
    private static final int TM_CCORR_NORMED = 5;

    // Member variables
    private LockscreenUtils mLockscreenUtils;

    private int learn_frames = 0;
    private int learn_framescounter = 25;
    private Mat teplateR;
    private Mat teplateL;
    int method = 0;

    private MenuItem mItemFace50;
    private MenuItem mItemFace40;
    private MenuItem mItemFace30;
    private MenuItem mItemFace20;
    private MenuItem mItemType;

    private Mat mRgba;
    private Mat mGray;
    // matrix for zooming
    private Mat mZoomWindow;
    private Mat mZoomWindow2;

    private File mCascadeFile;
    private CascadeClassifier mJavaDetector;
    private CascadeClassifier mJavaDetectorEye;


    private int mDetectorType = JAVA_DETECTOR;
    private String[] mDetectorName;

    private float mRelativeFaceSize = 0.2f;
    private int mAbsoluteFaceSize = 0;

    private CameraBridgeViewBase mOpenCvCameraView;

    private SeekBar mMethodSeekbar;
    private TextView mValue;
    private TextView mHelpText;
    private TextView numberLeft,numberRight,bufferSize,intervalSize;
    private TextView blinkText, closedText, closedhintText;
    //private Button buttonOne,buttonTwo,buttonThree,buttonFour,buttonFive,buttonSix;
    private SeekBar seekBarLeft,seekBarRight,seekBarBuffer,seekBarInterval;

    private FrameLayout lytOverlay, lytSettings, focusOverlay,closedOverlay,openOverlay;

    private boolean zoomWindow = false,
            progressFlagLeft = true,
            progressFlagRight = true,
            closedOverlayFlag = true,
            lrFlag = false;

    double xCenter = -1;
    double yCenter = -1;
    double eyeCenterxOld = 0;
    double eyeCenterx = 0;
    double eyeCenterxRightOld = 0;
    double eyeCenterxRight = 0;

    int buffSize = 50;
    private int maxBuff=15;

    private Date lastTimestamp;
    long expiremilis = 2000l;

    List<List<Double>> xyBufferLeft = new ArrayList<List<Double>>();
    List<List<Double>> xyBufferRight = new ArrayList<List<Double>>();
    List<String> gestureList = new ArrayList<String>();

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");


                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(
                                R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir,
                                "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        // --------------------------------- load left eye
                        // classificator -----------------------------------
                        InputStream iser = getResources().openRawResource(
                                R.raw.haarcascade_lefteye_2splits);
                        File cascadeDirER = getDir("cascadeER",
                                Context.MODE_PRIVATE);
                        File cascadeFileER = new File(cascadeDirER,
                                "haarcascade_eye_right.xml");
                        FileOutputStream oser = new FileOutputStream(cascadeFileER);

                        byte[] bufferER = new byte[4096];
                        int bytesReadER;
                        while ((bytesReadER = iser.read(bufferER)) != -1) {
                            oser.write(bufferER, 0, bytesReadER);
                        }
                        iser.close();
                        oser.close();

                        mJavaDetector = new CascadeClassifier(
                                mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from "
                                    + mCascadeFile.getAbsolutePath());

                        mJavaDetectorEye = new CascadeClassifier(
                                cascadeFileER.getAbsolutePath());
                        if (mJavaDetectorEye.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetectorEye = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from "
                                    + mCascadeFile.getAbsolutePath());



                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }
                    mOpenCvCameraView.setCameraIndex(1);
                    mOpenCvCameraView.enableFpsMeter();
                    mOpenCvCameraView.enableView();

                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };
    private void init() {
        mLockscreenUtils = new LockscreenUtils();
        lastTimestamp = new Date(System.currentTimeMillis()+expiremilis);
        Log.i(TAG, "lastTimestamp (init): "
                + lastTimestamp);
    }
    // Handle events of calls and unlock screen if necessary
    private class StateListener extends PhoneStateListener {
        @Override
        public void onCallStateChanged(int state, String incomingNumber) {

            super.onCallStateChanged(state, incomingNumber);
            switch (state) {
                case TelephonyManager.CALL_STATE_RINGING:
                    unlockHomeButton();
                    break;
                case TelephonyManager.CALL_STATE_OFFHOOK:
                    break;
                case TelephonyManager.CALL_STATE_IDLE:
                    break;
            }
        }
    };

    @Override
    public void onBackPressed() {
        return;
    }

    // Lock home button
    public void lockHomeButton() {
        mLockscreenUtils.lock(FdActivity.this);
    }

    // Unlock home button and wait for its callback
    public void unlockHomeButton() {
        mLockscreenUtils.unlock();
    }

    // Simply unlock device when home button is successfully unlocked
    @Override
    public void onLockStatusChanged(boolean isLocked) {
        if (!isLocked) {
            unlockDevice();
        }
    }

    @Override
    protected void onStop() {
        super.onStop();
        unlockHomeButton();
    }

    @SuppressWarnings("deprecation")
    private void disableKeyguard() {
        KeyguardManager mKM = (KeyguardManager) getSystemService(KEYGUARD_SERVICE);
        KeyguardManager.KeyguardLock mKL = mKM.newKeyguardLock("IN");
        mKL.disableKeyguard();
    }

    @SuppressWarnings("deprecation")
    private void enableKeyguard() {
        KeyguardManager mKM = (KeyguardManager) getSystemService(KEYGUARD_SERVICE);
        KeyguardManager.KeyguardLock mKL = mKM.newKeyguardLock("IN");
        mKL.reenableKeyguard();
    }

    //Simply unlock device by finishing the activity
    private void unlockDevice()
    {
        finish();
    }

    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        init();

        // unlock screen in case of app get killed by system
        if (getIntent() != null && getIntent().hasExtra("kill")
                && getIntent().getExtras().getInt("kill") == 1) {
            enableKeyguard();
            unlockHomeButton();
        } else {

            try {
                // disable keyguard
                disableKeyguard();

                // lock home button
                lockHomeButton();

                // start service for observing intents
                startService(new Intent(this, LockscreenService.class));

                // listen the events get fired during the call
                StateListener phoneStateListener = new StateListener();
                TelephonyManager telephonyManager = (TelephonyManager) getSystemService(TELEPHONY_SERVICE);
                telephonyManager.listen(phoneStateListener,
                        PhoneStateListener.LISTEN_CALL_STATE);

            } catch (Exception e) {
            }

        }

        lytOverlay = (FrameLayout) findViewById(R.id.Overlay);
        focusOverlay = (FrameLayout) findViewById(R.id.focusOverlay);
        closedOverlay = (FrameLayout) findViewById(R.id.closedOverlay);
        openOverlay = (FrameLayout) findViewById(R.id.openOverlay);
        lytSettings = (FrameLayout) findViewById(R.id.Settings);

        blinkText = (TextView) findViewById(R.id.blinkText);
        closedText = (TextView) findViewById(R.id.closedText);
        closedhintText = (TextView) findViewById(R.id.closedhintText);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);

        mMethodSeekbar = (SeekBar) findViewById(R.id.methodSeekBar);
        mValue = (TextView) findViewById(R.id.method);
        mHelpText = (TextView) findViewById(R.id.helpText);
        numberLeft = (TextView) findViewById(R.id.numberLeft);
        numberRight = (TextView) findViewById(R.id.numberRight);
        intervalSize = (TextView) findViewById(R.id.intervalSize);
        bufferSize = (TextView) findViewById(R.id.bufferSize);

        seekBarLeft = (SeekBar) findViewById(R.id.seekBarLeft);
        seekBarRight = (SeekBar) findViewById(R.id.seekBarRight);
        seekBarBuffer = (SeekBar) findViewById(R.id.seekBarBuffer);
        seekBarInterval = (SeekBar) findViewById(R.id.seekBarInterval);

        /*seekBarLeft.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                int progressValue = seekBar.getProgress();
                if(progressValue<100)numberLeft.setText(String.valueOf((progressValue) / 10));
                if((seekBarLeft.getProgress() / 10) == 7 && (seekBarRight.getProgress() / 10) == 5){
                    Log.i(TAG, "Lockscreen finish");
                    unlockHomeButton();
                }
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });*/

        /*seekBarRight.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                int progressValue = seekBar.getProgress();
                if(progressValue<100)numberRight.setText(String.valueOf((progressValue) / 10));
                if((seekBarLeft.getProgress() / 10) == 7 && (seekBarRight.getProgress() / 10) == 5){
                    Log.i(TAG, "Lockscreen finish");
                    unlockHomeButton();
                }
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });*/

        seekBarBuffer.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                maxBuff = seekBar.getProgress();
                int progressValue = seekBar.getProgress();
                bufferSize.setText(String.valueOf(progressValue));
                Log.i(TAG, "new BufferSize (maxBuff): " + maxBuff);
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });
        seekBarInterval.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                expiremilis = seekBar.getProgress();
                int progressValue = seekBar.getProgress();
                intervalSize.setText(String.valueOf(progressValue));
                Log.i(TAG, "new Intervalsize (expiremilis): " + expiremilis);
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });

        mMethodSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                // TODO Auto-generated method stub

            }

            @Override
            public void onProgressChanged(SeekBar seekBar, int progress,
                                          boolean fromUser) {
                method = progress;
                switch (method) {
                    case 0:
                        mValue.setText("TM_SQDIFF");
                        break;
                    case 1:
                        mValue.setText("TM_SQDIFF_NORMED");
                        break;
                    case 2:
                        mValue.setText("TM_CCOEFF");
                        break;
                    case 3:
                        mValue.setText("TM_CCOEFF_NORMED");
                        break;
                    case 4:
                        mValue.setText("TM_CCORR");
                        break;
                    case 5:
                        mValue.setText("TM_CCORR_NORMED");
                        break;
                }


            }
        });
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        //Alex: Darf nicht released werden -> sonst Error beim nächsten Start (bei locked Screen)
        /*mGray.release();
        mRgba.release();
        mZoomWindow.release();
        mZoomWindow2.release();*/
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }

        if (mZoomWindow == null || mZoomWindow2 == null)
            CreateAuxiliaryMats();

        MatOfRect faces = new MatOfRect();

        if (mJavaDetector != null)
            mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2,
                    2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize),
                    new Size());

        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {
            Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
            xCenter = (facesArray[i].x + facesArray[i].width + facesArray[i].x) / 2;
            yCenter = (facesArray[i].y + facesArray[i].y + facesArray[i].height) / 2;
            Point center = new Point(xCenter, yCenter);

            Imgproc.circle(mRgba, center, 10, new Scalar(255, 0, 0, 255), 3);

            Imgproc.putText(mRgba, "[" + center.x + "," + center.y + "]",
                    new Point(center.x + 20, center.y + 20),
                    Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255,
                            255));

            Rect r = facesArray[i];
            // compute the eye area
            Rect eyearea = new Rect(r.x + r.width / 8,
                    (int) (r.y + (r.height / 4.5)), r.width - 2 * r.width / 8,
                    (int) (r.height / 3.0));
            // split it
            Rect eyearea_right = new Rect(r.x + r.width / 16,
                    (int) (r.y + (r.height / 4.5)),
                    (r.width - 2 * r.width / 16) / 2, (int) (r.height / 3.0));
            Rect eyearea_left = new Rect(r.x + r.width / 16
                    + (r.width - 2 * r.width / 16) / 2,
                    (int) (r.y + (r.height / 4.5)),
                    (r.width - 2 * r.width / 16) / 2, (int) (r.height / 3.0));
            // draw the area - mGray is working grayscale mat, if you want to
            // see area in rgb preview, change mGray to mRgba
            // Alex: Eye Area ist rot
            Imgproc.rectangle(mRgba, eyearea_left.tl(), eyearea_left.br(),
                    new Scalar(255, 0, 0, 255), 2);
            Imgproc.rectangle(mRgba, eyearea_right.tl(), eyearea_right.br(),
                    new Scalar(255, 0, 0, 255), 2);

            if (learn_frames < learn_framescounter) {
                teplateR = get_template(mJavaDetectorEye, eyearea_right, 24,eyearea_right.tl().x + eyearea_right.width,false); //TODO: rechtes Auge
                teplateL = get_template(mJavaDetectorEye, eyearea_left, 24,eyearea_left.tl().x,true);
                learn_frames++;
            } else {
                // Learning finished, use the new templates for template
                // matching
                match_eye(eyearea_right, teplateR, method, xyBufferRight,eyearea_right.tl().x + eyearea_right.width,false);
                match_eye(eyearea_left, teplateL, method, xyBufferLeft,eyearea_left.tl().x,true);
                showClosed();
            }

            // cut eye areas and put them to zoom windows
            if(zoomWindow) {
                Imgproc.resize(mRgba.submat(eyearea_left), mZoomWindow2,
                        mZoomWindow2.size());
                Imgproc.resize(mRgba.submat(eyearea_right), mZoomWindow,
                        mZoomWindow.size());
            }

        }
        Core.flip(mRgba, mRgba, 1); //Alex: flip mirrored image
        return mRgba;
    }

    private void addToBuffer(double x, double y, double eyeAreax,List<List<Double>> xyBuffer, boolean flag, double eyeCenterxEye) {
        int maxBuffx = 4;

        if(xyBuffer.size()>maxBuff && eyeCenterxEye != 0 && closedOverlayFlag==true) {
            xyBuffer.remove(0);
            int xBuff=0;
            int yBuff=0;
            for(int i=0;i<xyBuffer.size();i++){
                if(xyBuffer.get(i).get(0)> eyeCenterxEye){
                    xBuff--;
                }else if(xyBuffer.get(i).get(0)< eyeCenterxEye){
                    xBuff++;
                }
            }
            if(xBuff<maxBuffx && (lastTimestamp.before(new Date(System.currentTimeMillis())))){
                lastTimestamp = new Date(System.currentTimeMillis()+expiremilis);
                Log.i(TAG, "lastTimestamp (left): "
                        + lastTimestamp);
                setProgressValue(numberLeft);
                Log.i(TAG, "SeekBarLeft - xBuff: " + xBuff + " - yBuff: " + yBuff);
                vibrate(150);
                gestureList.add("left");
            }else if(xBuff>maxBuffx && (lastTimestamp.before(new Date(System.currentTimeMillis())))){
                lastTimestamp = new Date(System.currentTimeMillis()+expiremilis);
                Log.i(TAG, "lastTimestamp (right): "
                        + lastTimestamp);
                setProgressValue(numberRight);
                Log.i(TAG, "SeekBarRight - xBuff: " + xBuff + " - yBuff: " + yBuff);
                vibrate(50);
                gestureList.add("right");
            }
        }
        if(gestureList.size()>2) {
            if (gestureList.get(0) == "left" && gestureList.get(1) == "right" && gestureList.get(2) == "left") {
                //unlockDevice();
                closedOverlayFlag = false;
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (lytOverlay.getVisibility() == View.VISIBLE) {
                            focusOverlay.setVisibility(View.GONE);
                            closedOverlay.setVisibility(View.GONE);
                            openOverlay.setVisibility(View.VISIBLE);
                        }
                    }

                });
            }
        }
        List<Double> list = new ArrayList<Double>();
        xyBuffer.add(list);
        if(flag){                           //Alex: linkes Auge
            list.add(x - eyeAreax);
        }else{                              //Alex: rechtes Auge
            list.add(eyeAreax - x);
        }

    }

    private void setProgressValue(final TextView textNumber){
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                textNumber.setText(String.valueOf(Integer.parseInt((String) textNumber.getText()) + 1));
            }
        });
    }

    /*private void setSeekBarProgressLeft(SeekBar seekBarInProgress){
        int progressValue = seekBarInProgress.getProgress();
        if(progressFlagLeft) {
            seekBarInProgress.setProgress(progressValue + 1);
            if(progressValue==99)progressFlagLeft=false;
        }else{
            seekBarInProgress.setProgress(progressValue - 1);
            if(progressValue==1)progressFlagLeft=true;
        }
        seekBarInProgress.refreshDrawableState();
    }

    private void setSeekBarProgressRight(SeekBar seekBarInProgress){

        int progressValue = seekBarInProgress.getProgress();
        if(progressFlagRight) {
            seekBarInProgress.setProgress(progressValue + 1);
            if(progressValue==99)progressFlagRight=false;
        }else{
            seekBarInProgress.setProgress(progressValue - 1);
            if(progressValue==1)progressFlagRight=true;
        }
        seekBarInProgress.refreshDrawableState();
    }
    private void setButtonTransparency(Button button,float alphaMult){
        float currAlpha = button.getAlpha();
        float newAlpha = currAlpha * alphaMult;
        if(newAlpha>1)newAlpha=1;
        if(newAlpha<(float)0.1)newAlpha=(float)0.1;
        button.setAlpha(newAlpha);
    }*/
    private void setEyeCenter(double x,double eyeAreax,boolean flag) {
        //TODO: Learn best EyeCenter

        if(flag){
            eyeCenterxOld = eyeCenterx;
            eyeCenterx = x - eyeAreax;
            Log.i(TAG, "eyeCenterx (left): " + eyeCenterx);
            lrFlag = true;
        }else{
            eyeCenterxRightOld = eyeCenterxRight;
            eyeCenterxRight = eyeAreax - x;
            Log.i(TAG, "eyeCenterx (right): " + eyeCenterxRight);
            lrFlag = false;
        }
        if(lrFlag) {
            if (eyeCenterxOld != 0 && eyeCenterxRightOld != 0) {
                if (Math.abs(eyeCenterx - eyeCenterxRight) > Math.abs(eyeCenterxOld - eyeCenterxRightOld)) {
                    eyeCenterx = eyeCenterxOld;
                    eyeCenterxRight = eyeCenterxRightOld;
                    Log.i(TAG, "Not a new EyeCenter: " + eyeCenterx + ", " + eyeCenterxRight);
                } else {
                    Log.i(TAG, "New EyeCenter: " + eyeCenterx + ", " + eyeCenterxRight);
                }
            }
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        mItemType = menu.add(mDetectorName[mDetectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);
        else if (item == mItemType) {
            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[tmpDetectorType]);
        }
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void CreateAuxiliaryMats() {
        if (mGray.empty())
            return;

        int rows = mGray.rows();
        int cols = mGray.cols();

        if (mZoomWindow == null) {
            mZoomWindow = mRgba.submat(rows / 2 + rows / 10, rows, cols / 2
                    + cols / 10, cols);
            mZoomWindow2 = mRgba.submat(0, rows / 2 - rows / 10, cols / 2
                    + cols / 10, cols);
        }
    }

    private void match_eye(Rect area, Mat mTemplate, int type, List<List<Double>> xyBuffer,double eyeAreax,boolean flag) {
        Point matchLoc;
        Mat mROI = mGray.submat(area);
        int result_cols = mROI.cols() - mTemplate.cols() + 1;
        int result_rows = mROI.rows() - mTemplate.rows() + 1;
        // Check for bad template size
        if (mTemplate.cols() == 0 || mTemplate.rows() == 0) {
            return ;
        }
        Mat mResult = new Mat(result_cols, result_rows, CvType.CV_8U);

        switch (type) {
            case TM_SQDIFF:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_SQDIFF);
                break;
            case TM_SQDIFF_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult,
                        Imgproc.TM_SQDIFF_NORMED);
                break;
            case TM_CCOEFF:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCOEFF);
                break;
            case TM_CCOEFF_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult,
                        Imgproc.TM_CCOEFF_NORMED);
                break;
            case TM_CCORR:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCORR);
                break;
            case TM_CCORR_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult,
                        Imgproc.TM_CCORR_NORMED);
                break;
        }

        Core.MinMaxLocResult mmres = Core.minMaxLoc(mResult);
        // there is difference in matching methods - best match is max/min value
        if (type == TM_SQDIFF || type == TM_SQDIFF_NORMED) {
            matchLoc = mmres.minLoc;
        } else {
            matchLoc = mmres.maxLoc;
        }

        Point matchLoc_tx = new Point(matchLoc.x + area.x, matchLoc.y + area.y);
        Point matchLoc_ty = new Point(matchLoc.x + mTemplate.cols() + area.x,
                matchLoc.y + mTemplate.rows() + area.y);

        // Alex: Linkes und rechtes Auge wird mit gelbem Rahmen eingegrenzt
        Imgproc.rectangle(mRgba, matchLoc_tx, matchLoc_ty, new Scalar(255, 255, 0,
                255));
        Rect rec = new Rect(matchLoc_tx,matchLoc_ty);
        if(flag){
            addToBuffer(matchLoc_tx.x,matchLoc_tx.y, eyeAreax, xyBuffer, flag, eyeCenterx); //TODO: Alex: relative x,y Koordinaten innerhalb des roten Bereiches (Eye Area) berechnen und einfügen - aber nur für ein Auge (Funktion wird ja 2x aufgerufen)
            Log.i(TAG, "xyBuffer (left): " + xyBuffer);
        }else{
            //addToBuffer(matchLoc_tx.x + rec.width,matchLoc_tx.y, eyeAreax, xyBuffer, flag, eyeCenterxRight);
            //Log.i(TAG, "xyBuffer (right): " + xyBuffer);
        }
    }

    private Mat get_template(CascadeClassifier clasificator, Rect area, int size,double eyeAreax, boolean flag) {
        Mat template = new Mat();
        Mat mROI = mGray.submat(area);
        MatOfRect eyes = new MatOfRect();
        Point iris = new Point();
        Rect eye_template = new Rect();
        clasificator.detectMultiScale(mROI, eyes, 1.15, 2,
                Objdetect.CASCADE_FIND_BIGGEST_OBJECT
                        | Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30),
                new Size());

        Rect[] eyesArray = eyes.toArray();
        for (int i = 0; i < eyesArray.length;) {
            Rect e = eyesArray[i];
            e.x = area.x + e.x;
            e.y = area.y + e.y;
            Rect eye_only_rectangle = new Rect((int) e.tl().x,
                    (int) (e.tl().y + e.height * 0.4), (int) e.width,
                    (int) (e.height * 0.6));
            mROI = mGray.submat(eye_only_rectangle);
            Mat vyrez = mRgba.submat(eye_only_rectangle);


            Core.MinMaxLocResult mmG = Core.minMaxLoc(mROI);

            Imgproc.circle(vyrez, mmG.minLoc, 2, new Scalar(255, 255, 255, 255), 2);
            iris.x = mmG.minLoc.x + eye_only_rectangle.x;
            iris.y = mmG.minLoc.y + eye_only_rectangle.y;
            eye_template = new Rect((int) iris.x - size / 2, (int) iris.y
                    - size / 2, size, size);
            Imgproc.rectangle(mRgba, eye_template.tl(), eye_template.br(),
                    new Scalar(255, 0, 0, 255), 2);
            template = (mGray.submat(eye_template)).clone();
            if(flag){
                setEyeCenter(eye_template.tl().x,eyeAreax,flag);
            }else{
                setEyeCenter(eye_template.tl().x + eye_template.width,eyeAreax,flag);
            }
            return template;
        }
        return template;
    }

    public void onRecreateClick(View v)
    {
        learn_frames = 0;
        closedOverlayFlag = true;
        xyBufferLeft.clear();
        xyBufferRight.clear();
        gestureList.clear();
        if(lytOverlay.getVisibility() == View.VISIBLE){
            focusOverlay.setVisibility(View.VISIBLE);
            closedOverlay.setVisibility(View.GONE);
            openOverlay.setVisibility(View.GONE);
        }
    }
    public void onCloseClick(View v)
    {
        unlockDevice();
    }
    public void onHelpClick(View v)
    {
        if(mHelpText.getVisibility() == View.GONE) {
            mHelpText.setVisibility(View.VISIBLE);
        }else{
            mHelpText.setVisibility(View.GONE);
        }
    }
    public void toggleZoomWindow(View v)
    {
        if(zoomWindow){
            zoomWindow = false;
        }else {
            zoomWindow = true;
        }
    }
    public void showSettings(View v){
        lytSettings.setVisibility(View.VISIBLE);
    }
    public void hideSettings(View v){
        lytSettings.setVisibility(View.GONE);
    }
    public void hideOverlay(View v){
        hideSettings(v);
        closedOverlayFlag = true;
        lytOverlay.setVisibility(View.GONE);
    }
    public void showOverlay(View v){
        hideSettings(v);
        closedOverlayFlag = true;
        lytOverlay.setVisibility(View.VISIBLE);
    }
    public void showClosed(){
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                closedOverlay.setVisibility(View.VISIBLE);
                focusOverlay.setVisibility(View.GONE);
                /*if (closedOverlayFlag) {
                    blinkText.setVisibility(View.GONE);
                    closedText.setVisibility(View.VISIBLE);
                    closedhintText.setVisibility(View.VISIBLE);
                } else {
                    closedText.setVisibility(View.GONE);
                    closedhintText.setVisibility(View.GONE);
                    //blinkText.setVisibility(View.VISIBLE);
                }*/
            }
        });
    }
    private void vibrate(int duration) {
        Vibrator vb = (Vibrator) this.getSystemService(Context.VIBRATOR_SERVICE);
        vb.vibrate(duration);
    }
}