#include <sys/time.h>   // NOLINT(build/include_order)


#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "settings.h"
#include "utils.h"

#include "opencv2/opencv.hpp"

using namespace std;

#define LOG(x) std::cerr

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;

TfLiteDelegatePtr CreateGPUDelegate(tflite::cpm::Settings* s) {
#if defined(__ANDROID__)
    TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
  gpu_opts.inference_preference =
      TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  gpu_opts.is_precision_loss_allowed = s->allow_fp16 ? 1 : 0;
  return evaluation::CreateGPUDelegate(s->model, &gpu_opts);
#else
    return tflite::evaluation::CreateGPUDelegate(s->model);
#endif
}

TfLiteDelegatePtrMap GetDelegates(tflite::cpm::Settings* s) {
    TfLiteDelegatePtrMap delegates;
    if (s->gl_backend) {
        auto delegate = CreateGPUDelegate(s);
        if (!delegate) {
            LOG(INFO) << "GPU acceleration is unsupported on this platform.";
        } else {
            delegates.emplace("GPU", std::move(delegate));
        }
    }

    if (s->accel) {
        auto delegate = tflite::evaluation::CreateNNAPIDelegate();
        if (!delegate) {
            LOG(INFO) << "NNAPI acceleration is unsupported on this platform.";
        } else {
            delegates.emplace("NNAPI", tflite::evaluation::CreateNNAPIDelegate());
        }
    }
    return delegates;
}


void RunInference(tflite::cpm::Settings* s){
    // load model file and build the interpreter.
    if (!s->model_name.c_str()) {
        LOG(ERROR) << "no model file name\n";
        exit(-1);
    }

    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());
    std::cout << s->model_name << std::endl;
    if (!model) {
        LOG(FATAL) << "\nFailed to mmap model " << s->model_name << "\n";
        exit(-1);
    }

    // invoke error reporter
    if (s->verbose){
        model->error_reporter();
    }
    LOG(INFO) << "resolved reporter\n";

    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        LOG(FATAL) << "Failed to construct interpreter\n";
        exit(-1);
    }

    // allocate all of the tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOG(FATAL) << "Failed to allocate tensors!";
    }

    // print the tensors which interpreter has got
    if (s->verbose) PrintInterpreterState(interpreter.get());

    // invoke the NNAPI and set the computing precision
    interpreter->UseNNAPI(s->old_accel);
    interpreter->SetAllowFp16PrecisionForFp32(s->allow_fp16);

    // If you have acceleration device, then load the graph into computing device
    auto delegates_ = GetDelegates(s);
    for (const auto& delegate : delegates_) {
        if (interpreter->ModifyGraphWithDelegate(delegate.second.get()) !=
            kTfLiteOk) {
            LOG(FATAL) << "Failed to apply " << delegate.first << " delegate.";
        } else {
            LOG(INFO) << "Applied " << delegate.first << " delegate.";
        }
    }

    // set the number of threads
    if (s->number_of_threads != -1) {
        interpreter->SetNumThreads(s->number_of_threads);
    }

    // fetch the inputs' tensor and outputs' tensor
    const std::vector<int> inputs = interpreter->inputs();
    const std::vector<int> outputs = interpreter->outputs();

    // print the number of inputs and outputs.
    LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
    LOG(INFO) << "number of outputs: " << outputs.size() << "\n";

    // construct the input vector
    int input = interpreter->inputs()[0];
    TfLiteIntArray* dims = interpreter->tensor(input)->dims;
    int wanted_height = dims->data[1];
    int wanted_width = dims->data[2];
    int wanted_channels = dims->data[3];

    // fill input tensor with image data
    std::string imgpath = "/home/ai/Downloads/00010100000006E8-20190906-073144-359.jpg";
    cv::Mat img = cv::imread(imgpath);
    int cols = wanted_width;
    int rows = wanted_height;
    cv::Size dsize = cv::Size(cols, rows);
    cv::Mat img_rs;
    cv::resize(img, img_rs, dsize);
    cv::cvtColor(img_rs, img_rs, cv::COLOR_BGR2RGB);

    auto img_inputs = interpreter->typed_tensor<float>(input);
    for (int i = 0; i < img_rs.cols * img_rs.rows * 3; i++){
        img_inputs[i] =  img_rs.data[i]/ 255.0;
    }

    // invoke the model to inference
    struct timeval start_time, stop_time;
    gettimeofday(&start_time, nullptr);

    if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke tflite!\n";
    }

    gettimeofday(&stop_time, nullptr);
    LOG(INFO) << "invoked \n";
    LOG(INFO) << "average time: "
              << (get_us(stop_time) - get_us(start_time)) / (s->loop_count * 1000)
              << " ms \n";

    // get the dims of outputs
    // the index of line 175 depends on the which output you want.
    // output = interpreter->outputs()[i];
    int output = interpreter->outputs()[5];
    TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
    auto output_size = output_dims->data[output_dims->size - 1];

    cout << "Output dims" << endl;
    for (int i=0;i<4;i++){
        cout<<output_dims->data[i]<<endl;
    }

    // get outputs
    auto* score_map = interpreter->typed_output_tensor<float>(5);

    auto* feature_map = new float[46*46];

    cv::Mat ori_img = cv::imread("/home/ai/Downloads/00010100000006E8-20190906-073144-359.jpg");

    for (int j = 0; j < 10;j++){
        int index = 0;
        for (int i=j;i<46*46*10;i+=10){
            feature_map[index] = (float)score_map[i];
            index++;
        }
        cv::Mat feature_map_mat(46,46,CV_32FC1, feature_map);
        double countMinVal = 0;
        double countMaxVal = 0;
        cv::Point minPoint;
        cv::Point maxPoint;
        cv::minMaxLoc(feature_map_mat, &countMinVal, &countMaxVal, &minPoint, &maxPoint);

        cout << " -- " << maxPoint.x  * 1280/ 46  << " - " << minPoint.y  * 720 / 46 << endl;

        auto px = (float)maxPoint.x;
        auto py = (float)maxPoint.y;
        float x = px / 46 * 1280;
        float y = py / 46 * 720 ;
        int xx = (int)x;
        int yy = (int)y;

        cv::circle(ori_img, cv::Point(xx,yy), 20, cv::Scalar(255 - j* 20,100 + j*10,0), -1);
    }

    cv::imshow("", ori_img);
    cv::waitKey(0);


}

int main() {
    tflite::cpm::Settings s;
    RunInference(&s);
}
