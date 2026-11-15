package xr.pathfinder.bridge

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import java.io.BufferedWriter
import java.io.OutputStreamWriter
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * Path Finder sensor bridge: streams the phone's raw IMU and ARCore's camera tracking to a laptop
 * over USB (adb reverse tcp:7777 tcp:7777), one JSON object per line:
 *   {"k":"g","t":ns,"v":[x,y,z]}  gyroscope, rad/s       (uncalibrated if the phone has it)
 *   {"k":"a","t":ns,"v":[x,y,z]}  accelerometer, m/s^2   (uncalibrated if the phone has it)
 *   {"k":"ar","t":ns,"s":"TRACKING|PAUSED|STOPPED","r":"NONE|INSUFFICIENT_LIGHT|...",
 *    "p":[x,y,z],"q":[x,y,z,w],"n":points}   ARCore pose of the Android sensor frame (world y-up)
 * The "Simulated IMU bias" button adds a constant bias before sending; such lines carry "sim":1.
 * Nothing is stored on the phone and nothing goes anywhere except the USB link.
 */
class MainActivity : Activity(), SensorEventListener, GLSurfaceView.Renderer {
    private lateinit var status: TextView
    private lateinit var gl: GLSurfaceView
    private var session: Session? = null
    private var texId = 0
    private val queue = LinkedBlockingQueue<String>(20000)
    @Volatile private var running = true
    @Volatile private var connected = false
    @Volatile private var simBias = false
    private var nGyro = 0L
    private var nAcc = 0L
    private var arState = "none"
    private var arReason = ""
    private var installRequested = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        gl = GLSurfaceView(this).apply {
            preserveEGLContextOnPause = true
            setEGLContextClientVersion(2)
            setRenderer(this@MainActivity)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        }
        status = TextView(this).apply {
            setTextColor(Color.WHITE); textSize = 18f; setPadding(32, 32, 32, 32)
        }
        val bias = Button(this).apply {
            text = "Simulated IMU bias: OFF"
            setOnClickListener {
                simBias = !simBias
                text = if (simBias) "Simulated IMU bias: ON" else "Simulated IMU bias: OFF"
            }
        }
        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(status)
            addView(bias, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        }
        setContentView(FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
            addView(gl, FrameLayout.LayoutParams(1, 1))         // GL only drives ARCore; nothing is drawn
            addView(panel, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT, Gravity.TOP))
        })
        Thread(::sender, "sender").start()
        startSensors()
        tick()
    }

    private fun startSensors() {
        val sm = getSystemService(SENSOR_SERVICE) as SensorManager
        val g = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE_UNCALIBRATED) ?: sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        val a = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER_UNCALIBRATED) ?: sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        sm.registerListener(this, g, SensorManager.SENSOR_DELAY_FASTEST)
        sm.registerListener(this, a, SensorManager.SENSOR_DELAY_FASTEST)
    }

    override fun onSensorChanged(e: SensorEvent) {
        val gyro = e.sensor.type == Sensor.TYPE_GYROSCOPE || e.sensor.type == Sensor.TYPE_GYROSCOPE_UNCALIBRATED
        var x = e.values[0]; var y = e.values[1]; var z = e.values[2]
        if (simBias) { if (gyro) { x += 0.05f; z -= 0.03f } else { x += 0.5f; y -= 0.3f } }
        if (gyro) nGyro++ else nAcc++
        val k = if (gyro) "g" else "a"
        queue.offer("{\"k\":\"$k\",\"t\":${e.timestamp},\"v\":[$x,$y,$z]${if (simBias) ",\"sim\":1" else ""}}")
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun sender() {
        while (running) {
            try {
                Socket().use { s ->
                    s.connect(InetSocketAddress("127.0.0.1", 7777), 1000)
                    s.tcpNoDelay = true
                    connected = true
                    val w = BufferedWriter(OutputStreamWriter(s.getOutputStream()))
                    while (running) {
                        val line = queue.poll(200, TimeUnit.MILLISECONDS) ?: continue
                        w.write(line); w.write("\n")
                        if (queue.isEmpty()) w.flush()
                    }
                }
            } catch (ex: Exception) {
                connected = false
                queue.clear()
                Thread.sleep(500)
            }
        }
    }

    private fun tick() {
        status.text = "Path Finder sensor bridge\n" +
            "Laptop link: ${if (connected) "connected" else "waiting (adb reverse tcp:7777 tcp:7777)"}\n" +
            "IMU samples sent: gyro $nGyro, accel $nAcc\n" +
            "ARCore: $arState $arReason\n" +
            "Cover the camera or shake the phone to break a sensor."
        status.postDelayed({ tick() }, 250)
    }

    override fun onResume() {
        super.onResume()
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 1); return
        }
        if (session == null) {
            try {
                when (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                    ArCoreApk.InstallStatus.INSTALL_REQUESTED -> { installRequested = true; return }
                    else -> {}
                }
                session = Session(this).also {
                    it.configure(Config(it).apply { updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE })
                }
            } catch (ex: Exception) {
                arState = "unavailable"; arReason = ex.javaClass.simpleName
            }
        }
        try { session?.resume() } catch (ex: Exception) { arState = "error"; arReason = ex.javaClass.simpleName }
        gl.onResume()
    }

    override fun onPause() {
        super.onPause()
        gl.onPause()
        session?.pause()
    }

    override fun onDestroy() {
        running = false
        session?.close()
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(rc: Int, p: Array<out String>, r: IntArray) {
        super.onRequestPermissionsResult(rc, p, r)
        recreate()
    }

    // ---- GL: only to give ARCore a camera texture --------------------------------------------
    override fun onSurfaceCreated(g: GL10?, c: EGLConfig?) {
        val t = IntArray(1)
        GLES20.glGenTextures(1, t, 0)
        texId = t[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, texId)
        session?.setCameraTextureName(texId)
    }

    override fun onSurfaceChanged(g: GL10?, w: Int, h: Int) {
        session?.setDisplayGeometry(windowManager.defaultDisplay.rotation, w.coerceAtLeast(1), h.coerceAtLeast(1))
    }

    override fun onDrawFrame(g: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        val s = session ?: return
        try {
            s.setCameraTextureName(texId)
            val f = s.update()
            val cam = f.camera
            arState = cam.trackingState.name
            arReason = if (cam.trackingState == TrackingState.TRACKING) "" else cam.trackingFailureReason.name
            val pose = f.androidSensorPose
            val pts = f.acquirePointCloud().use { it.points.remaining() / 4 }
            queue.offer("{\"k\":\"ar\",\"t\":${f.timestamp},\"s\":\"${cam.trackingState.name}\"," +
                "\"r\":\"${cam.trackingFailureReason.name}\",\"p\":[${pose.tx()},${pose.ty()},${pose.tz()}]," +
                "\"q\":[${pose.qx()},${pose.qy()},${pose.qz()},${pose.qw()}],\"n\":$pts}")
        } catch (ex: Exception) {
            arState = "error"; arReason = ex.javaClass.simpleName
        }
    }
}
