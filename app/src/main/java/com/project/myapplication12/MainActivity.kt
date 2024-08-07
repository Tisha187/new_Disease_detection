package com.project.myapplication12

import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.project.myapplication12.ml.Degree
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    lateinit var imageView: ImageView
    lateinit var selctbtn:Button
    lateinit var predictbtn: Button
    lateinit var resview: TextView
    lateinit var bitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        selctbtn = findViewById(R.id.btn1)
        imageView = findViewById(R.id.imageView)
        predictbtn = findViewById(R.id.btn2)
        resview = findViewById(R.id.textView)


         var labels = application.assets.open("labels.txt").bufferedReader().readLines()
        //Image Processor

        var imageProcessor = ImageProcessor.Builder()
            .add(NormalizeOp(0.0f,255.0f))
            .add(ResizeOp(256,256,ResizeOp.ResizeMethod.BILINEAR))
            .build()


        selctbtn.setOnClickListener {

            // get the image from phone storage
            val intent  = Intent()
            intent.setType("image/*")
            intent.setAction(Intent.ACTION_GET_CONTENT)

            startActivityForResult(intent,101)
        }

        predictbtn.setOnClickListener {

            var tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)

            tensorImage = imageProcessor.process(tensorImage)

            val model = Degree.newInstance(this)

// Creates inputs for reference.

            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(32, 256, 256, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(tensorImage.buffer)

// Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

            var maxIdx = 0
            outputFeature0.forEachIndexed { index, fl ->
               if( outputFeature0[maxIdx]<fl){
                   maxIdx = index
               }
            }

            resview.setText(labels[maxIdx])



// Releases model resources if no longer used.
            model.close()

        }


    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if(requestCode == 101){
            var uri = data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver,uri)
            imageView.setImageBitmap(bitmap)

        }
    }
    }
