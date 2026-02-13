#include "esp_camera.h"
#include "animal_detector_model.h"

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ================= CAMERA PINS (AI THINKER) =================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define TRIGGER_PIN 13
#define OUTPUT_PIN  12

// ================= TFLITE =================
constexpr int kTensorArenaSize = 50 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

void setup() {

  Serial.begin(115200);

  pinMode(TRIGGER_PIN, INPUT);
  pinMode(OUTPUT_PIN, OUTPUT);
  digitalWrite(OUTPUT_PIN, LOW);

  // ===== CAMERA CONFIG =====
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;

  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size   = FRAMESIZE_96X96;
  config.fb_count     = 1;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera init failed!");
    while (1);
  }

  // ===== TFLITE INIT =====
  const tflite::Model* model =
      tflite::GetModel(animal_detector_int8_tflite);

  static tflite::MicroErrorReporter micro_error_reporter;
  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena,
      kTensorArenaSize, &micro_error_reporter);

  interpreter = &static_interpreter;

  interpreter->AllocateTensors();

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("System Ready");
}

void loop() {

  if (digitalRead(TRIGGER_PIN) == HIGH) {

    Serial.println("Capturing...");

    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Capture failed");
      return;
    }

    // ===== PREPROCESS (RGB565 → Grayscale → INT8) =====
    for (int i = 0; i < 96 * 96; i++) {

      uint16_t pixel = ((uint16_t*)fb->buf)[i];

      uint8_t r = ((pixel >> 11) & 0x1F) << 3;
      uint8_t g = ((pixel >> 5) & 0x3F) << 2;
      uint8_t b = (pixel & 0x1F) << 3;

      uint8_t gray = (0.299 * r + 0.587 * g + 0.114 * b);

      input->data.int8[i] = gray - 128;
    }

    // ===== INFERENCE =====
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Inference failed");
      esp_camera_fb_return(fb);
      return;
    }

    // ===== OUTPUT =====
    int8_t raw = output->data.int8[0];

    float probability =
        (raw - output->params.zero_point) *
        output->params.scale;

    Serial.print("Probability: ");
    Serial.println(probability, 6);

    if (probability < 0.5) {

      Serial.println("Animal Detected");
      digitalWrite(OUTPUT_PIN, HIGH);

    } else {

      Serial.println("No Animal");
      digitalWrite(OUTPUT_PIN, LOW);
    }

    esp_camera_fb_return(fb);
    delay(2000);
  }
}
