package com.sametersoyoglu.objectmlapp

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import com.sametersoyoglu.objectmlapp.ml.AutoModel1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class MainActivity : AppCompatActivity() {

    val paint= Paint()
    lateinit var imageView: ImageView
    lateinit var button: Button
    lateinit var bitmap: Bitmap
    lateinit var model: AutoModel1
    lateinit var labels: List<String>
    val imageProcessor = ImageProcessor.Builder().add(ResizeOp(300,300,ResizeOp.ResizeMethod.BILINEAR)).build()

    var colors = listOf<Int>(
        Color.BLUE,Color.GREEN,Color.RED,Color.CYAN,Color.GRAY,Color.BLACK,Color.DKGRAY,Color.MAGENTA,Color.YELLOW)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val intent = Intent()
        intent.setType("image/*")
        intent.setAction(Intent.ACTION_GET_CONTENT)

        labels = FileUtil.loadLabels(this,"labels.txt")
        model = AutoModel1.newInstance(this)

        imageView = findViewById(R.id.imageV)
        button = findViewById(R.id.btn)

        button.setOnClickListener {
            startActivityForResult(intent,101)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 101) {
            var url = data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver,url)
            get_predictions()

        }
    }
    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }

    fun get_predictions() {

        var image = TensorImage.fromBitmap(bitmap)
        image = imageProcessor.process(image)

        val outputs = model.process(image)
        val locations = outputs.locationsAsTensorBuffer.floatArray
        val classes = outputs.classesAsTensorBuffer.floatArray
        val scores = outputs.scoresAsTensorBuffer.floatArray
        val numberOfDetections = outputs.numberOfDetectionsAsTensorBuffer.floatArray

        var mutable = bitmap.copy(Bitmap.Config.ARGB_8888,true)
        val canvas = Canvas(mutable)

        val h = mutable.height
        val w = mutable.width

        paint.textSize = h/12f
        paint.strokeWidth = h/85f
        var x = 0
        scores.forEachIndexed{index, fl ->
            x = index
            x *= 4
            if (fl > 0.5) {
                paint.setColor(colors.get(index))
                paint.style = Paint.Style.STROKE
                canvas.drawRect(RectF(locations.get(x+1)*w,locations.get(x)*h,locations.get(x+3)*w,locations.get(x+2)*h),paint)
                paint.style = Paint.Style.FILL
                canvas.drawText(labels.get(classes.get(index).toInt())+" "+fl.toString(),locations.get(x+1)*w,locations.get(x)*h,paint)

            }
        }

        imageView.setImageBitmap(mutable)

    }


}