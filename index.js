const express = require("express");
const multer = require("multer");
const { Storage } = require("@google-cloud/storage");
const tf = require("@tensorflow/tfjs-node");
const { Firestore } = require("@google-cloud/firestore");
const sharp = require("sharp");
const cors = require("cors");
const fs = require("fs");
const path = require("path");

const app = express();
const port = process.env.PORT || 8080;

app.use(cors());

// Tentukan path ke file kredensial (.key.json) untuk akun layanan
const keyFilePath = path.join(__dirname, "./submissionmlgctaufanali.json"); // Menggunakan path baru

// Konfigurasi Google Cloud Storage menggunakan kredensial service account
const storage = new Storage({ keyFilename: keyFilePath });
const bucketName = "submissionmlgctaufanali";  // Pastikan nama bucket benar
const modelBucket = storage.bucket(bucketName);

// Konfigurasi Firestore menggunakan kredensial service account
const firestore = new Firestore({ keyFilename: keyFilePath });
const predictionsCollection = firestore.collection("predictions");

// Konfigurasi Multer untuk upload
const upload = multer({
  limits: {
    fileSize: 5000000, // 5MB, bisa sesuaikan sesuai kebutuhan
  },
});

// Fungsi untuk memastikan unduhan model dan bobot
async function loadModelFromGCS() {
  try {
    const modelFolderPath = "./model";
    await fs.promises.mkdir(modelFolderPath, { recursive: true }); // Membuat folder jika belum ada

    // Mendapatkan file dari Google Cloud Storage
    const [files] = await modelBucket.getFiles({ prefix: "model/" });

    // Cari file model.json dan semua file .bin yang diperlukan
    const modelFile = files.find((file) => file.name.endsWith("model.json"));
    if (!modelFile) {
      throw new Error("Model file not found in bucket");
    }

    // Unduh file model.json
    const modelFilePath = `${modelFolderPath}/model.json`;
    await modelFile.download({ destination: modelFilePath });

    // Download semua file .bin yang terkait dengan model
    const binFiles = files.filter((file) => file.name.endsWith(".bin"));
    for (let binFile of binFiles) {
      const binFilePath = `${modelFolderPath}/${binFile.name.split("/").pop()}`;
      await binFile.download({ destination: binFilePath });
      console.log(`Downloaded ${binFile.name}`);
    }

    // Periksa jika file model.json dan .bin sudah tersedia
    await fs.promises.access(modelFilePath);
    for (let binFile of binFiles) {
      const binFilePath = `${modelFolderPath}/${binFile.name.split("/").pop()}`;
      await fs.promises.access(binFilePath);
    }

    console.log("Model and weight files are available");

    // Memuat model dari file yang telah diunduh
    const model = await tf.loadGraphModel(`file://${modelFilePath}`);
    return model;
  } catch (error) {
    console.error("Error loading model:", error);
    throw error;
  }
}

// Fungsi preprocessing gambar
async function preprocessImage(file) {
  try {
    // Resize dan normalisasi gambar
    const imageBuffer = await sharp(file.buffer)
      .resize(224, 224)
      .toFormat("jpeg")
      .toBuffer();

    // Konversi gambar ke tensor
    const tensor = tf.node
      .decodeImage(imageBuffer)
      .toFloat()
      .expandDims(0)
      .div(255.0);

    return tensor;
  } catch (error) {
    console.error("Error preprocessing image:", error);
    throw error;
  }
}

app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        status: "fail",
        message: "Tidak ada gambar yang diunggah",
      });
    }

    // Periksa apakah file yang diupload adalah gambar
    if (!req.file.mimetype.startsWith("image/")) {
      return res.status(400).json({
        status: "fail",
        message: "File yang diunggah bukan gambar",
      });
    }

    // Muat model dari GCS
    const model = await loadModelFromGCS();

    // Preprocess gambar
    const imageTensor = await preprocessImage(req.file);

    // Prediksi menggunakan model
    const prediction = model.predict(imageTensor);
    const predictionValue = prediction.dataSync()[0];

    // Tentukan hasil prediksi
    const result = predictionValue > 0.5 ? "Cancer" : "Non-cancer";
    const suggestion =
      result === "Cancer"
        ? "Segera periksa ke dokter!"
        : "Penyakit kanker tidak terdeteksi.";

    // Buat ID unik untuk setiap prediksi
    const id = generateUniqueId();

    // Simpan hasil prediksi ke Firestore
    await predictionsCollection.doc(id).set({
      id,
      result,
      suggestion,
      createdAt: new Date().toISOString(),
    });

    // Kirim response
    res.json({
      status: "success",
      message: "Model is predicted successfully",
      data: {
        id,
        result,
        suggestion,
        createdAt: new Date().toISOString(),
      },
    });
  } catch (error) {
    console.error("Prediction error:", error);
    res.status(400).json({
      status: "fail",
      message: "Terjadi kesalahan dalam melakukan prediksi",
    });
  }
});

// Fungsi untuk menghasilkan ID unik
function generateUniqueId() {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
    var r = (Math.random() * 16) | 0,
      v = c == "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// Start server
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});