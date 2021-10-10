#include <iostream>
#include <tensorflow/c/c_api.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/public/session.h>

#include "MLPModel.h"
#include "MnistDataset.h"
using namespace std;
using namespace tensorflow;

const size_t batch_size = 64;

int main() {
  cout << "Hello from TensorFlow C library version " << TF_Version() << endl;
  Session *session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    cout << status.ToString() << "\n";
    return 1;
  }
  cout << "Session successfully created.\n";
  MnistDataset dataset("..");

  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  const float learning_rate = 0.0001f;
  MLP mlp(scope, learning_rate);
  mlp.initialize();
  vector<MnistImage> imagesBatch;
  vector<float> oneHotLabelsBatch;
  float loss;

  for (int epoch = 0; epoch < 300; epoch++) {
    dataset.shuffle();
    int correct_preds = 0;
    vector<float> losses;
    for (int i = 0;; i++) {
      int64_t dims[] = {batch_size, 28 * 28};
      int64_t labelDims[] = {batch_size, 10};
      dataset.getTrainMinibatch(batch_size, i, imagesBatch, oneHotLabelsBatch);
      if (imagesBatch.size() == 0)
        break;
      Tensor imagesTensor(DT_UINT8, TensorShape({(int)imagesBatch.size(), 28 * 28}));
      Tensor labelsTensor(DT_FLOAT, TensorShape({(int)imagesBatch.size(), 10}));
      if (imagesBatch.size() * sizeof(MnistImage) !=
          imagesTensor.dim_size(0) * imagesTensor.dim_size(1)) {
        printf("Data size: %d, tensor size: %d\n",
               imagesBatch.size() * sizeof(MnistImage),
               imagesTensor.dim_size(0) * imagesTensor.dim_size(1));
      }
      if (oneHotLabelsBatch.size() !=
          labelsTensor.dim_size(0) * labelsTensor.dim_size(1)) {
        printf("Labels size: %d, tensor size: %d\n", oneHotLabelsBatch.size(),
               labelsTensor.dim_size(0) * labelsTensor.dim_size(1));
      }


      memcpy(imagesTensor.data(), imagesBatch.data(),
             imagesBatch.size() * sizeof(MnistImage));
      memcpy(labelsTensor.data(), oneHotLabelsBatch.data(),
             oneHotLabelsBatch.size() * sizeof(float));

      mlp.trainStep(imagesTensor, labelsTensor, loss, correct_preds);
      losses.push_back(loss);
      // printf("%f\n", loss);
    }
    float mean_loss = 0;
    for (int i = 0; i < losses.size(); i++) {
      mean_loss += losses[i];
    }
    mean_loss /= losses.size();
    printf("Epoch: %d, loss: %f, accuracy: %f\n", epoch, mean_loss, (float)correct_preds/dataset.trainSize());
  }

  return 0;
}
