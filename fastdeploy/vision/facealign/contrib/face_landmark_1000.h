// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {

namespace vision {

namespace facealign {
/*! @brief FaceLandmark1000 model object used when to load a FaceLandmark1000 model exported by FaceLandmark1000.
 */
class FASTDEPLOY_DECL FaceLandmark1000 : public FastDeployModel {
 public:
  /** \brief  Set path of model file and the configuration of runtime.
   *
   * \param[in] model_file Path of model file, e.g ./face_landmarks_1000.onnx
   * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams, if the model format is ONNX, this parameter will be ignored
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in "valid_cpu_backends"
   * \param[in] model_format Model format of the loaded model, default is ONNX format
   */
  FaceLandmark1000(const std::string& model_file,
                   const std::string& params_file = "",
                   const RuntimeOption& custom_option = RuntimeOption(),
                   const ModelFormat& model_format = ModelFormat::ONNX);

  std::string ModelName() const { return "FaceLandmark1000"; }
  /** \brief Predict the face detection result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
   * \param[in] result The output face detection result will be writen to this structure
   * \return true if the prediction successed, otherwise false
   */
  virtual bool Predict(cv::Mat* im, FaceAlignmentResult* result);

  /** \brief Get the input size of image
   *
   * \return Vector of int values, default {128,128}
   */
  std::vector<int> GetSize() { return size_; }
  /** \brief Set the input size of image
   *
   * \param[in] size Vector of int values which represents {width, height} of image
   */
  void SetSize(const std::vector<int>& size) { size_ = size; }

 private:
  bool Initialize();

  bool Preprocess(Mat* mat, FDTensor* outputs,
                  std::map<std::string, std::array<int, 2>>* im_info);

  bool Postprocess(FDTensor& infer_result, FaceAlignmentResult* result,
                   const std::map<std::string, std::array<int, 2>>& im_info);
  // tuple of (width, height), default (128, 128)
  std::vector<int> size_;
};

}  // namespace facealign
}  // namespace vision
}  // namespace fastdeploy
