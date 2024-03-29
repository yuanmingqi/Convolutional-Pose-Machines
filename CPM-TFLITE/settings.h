#ifndef TENSORFLOW_LITE_EXAMPLES_LABEL_IMAGE_LABEL_IMAGE_H_
#define TENSORFLOW_LITE_EXAMPLES_LABEL_IMAGE_LABEL_IMAGE_H_

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
    namespace cpm {

        struct Settings {
            bool verbose = false;
            bool accel = false;
            bool old_accel = false;
            bool input_floating = false;
            bool profiling = false;
            bool allow_fp16 = false;
            bool gl_backend = false;
            int loop_count = 1;
            float input_mean = 127.5f;
            float input_std = 127.5f;
            string model_name = "../saved_model/converted_model.tflite";
            tflite::FlatBufferModel* model;
            string input_layer_type = "uint8_t";
            int number_of_threads = 4;
            int max_profiling_buffer_entries = 1024;
            int number_of_warmup_runs = 2;
        };

    }  // namespace label_image
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_LABEL_IMAGE_LABEL_IMAGE_H_