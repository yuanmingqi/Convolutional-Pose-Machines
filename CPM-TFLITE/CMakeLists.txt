cmake_minimum_required(VERSION 3.15)
project(cpm-tflite)

set(CMAKE_CXX_STANDARD 14)

include_directories(/home/ai/CLionProjects/tensorflow-master/)
include_directories(/home/ai/CLionProjects/tensorflow-master/tensorflow/lite/tools/make/downloads/flatbuffers/include)
include_directories(/home/ai/CLionProjects/tensorflow-master/tensorflow/lite/tools/make/downloads/absl)

include_directories(/usr/local/include/opencv/)
include_directories(/usr/local/include/opencv2/)
include_directories(/usr/local/include)

link_libraries(/usr/lib /usr/local/lib)

set(OPENCV_LIBS
        opencv_calib3d;
        opencv_core;
        opencv_features2d;
        opencv_flann;
        opencv_highgui;
        opencv_imgcodecs;
        opencv_imgproc;
        opencv_ml;
        opencv_objdetect;
        opencv_photo;
        opencv_shape;
        opencv_stitching;
        opencv_superres;
        opencv_videoio;
        opencv_video;
        opencv_videostab;)

add_executable(cpm-tflite main.cpp utils.cc )

target_link_libraries(cpm-tflite /home/ai/CLionProjects/tensorflow-master/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a -lpthread -ldl -lrt ${OPENCV_LIBS})
