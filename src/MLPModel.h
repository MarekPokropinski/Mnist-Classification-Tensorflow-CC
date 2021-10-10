#ifndef MLPMODEL
#define MLPMODEL
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/cc/framework/ops.h>
// #include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
// #include <tensorflow/core/framework/graph.pb.h>
// #include <tensorflow/core/framework/tensor.h>
// #include <tensorflow/core/graph/default_device.h>
// #include <tensorflow/core/graph/graph_def_builder.h>
// #include <tensorflow/core/lib/core/errors.h>
// #include <tensorflow/core/lib/core/stringpiece.h>
// #include <tensorflow/core/lib/core/threadpool.h>
// #include <tensorflow/core/lib/io/path.h>
// #include <tensorflow/core/lib/strings/str_util.h>
// #include <tensorflow/core/lib/strings/stringprintf.h>
// #include <tensorflow/core/platform/env.h>
// #include <tensorflow/core/platform/init_main.h>
// #include <tensorflow/core/platform/logging.h>
// #include <tensorflow/core/platform/types.h>
#include <tensorflow/cc/client/client_session.h>
// #include <tensorflow/core/util/command_line_flags.h>

#include <tensorflow/cc/framework/scope.h>

#include <unordered_map>
#include <memory>

using namespace std;
using namespace tensorflow;

class MLP {
  Scope scope;
  unordered_map<string, Output> vars_map;
  unordered_map<string, TensorShape> shapes_map;
  unordered_map<string, Output> inits_map;
  Output input_placeholder;
  Output label_placeholder;
  Output output;
  Output loss_tensor;
  Output prediction_tensor;

  vector<Output> vars;
  vector<Output> optimizer_inits;

  vector<Operation> applyOptim;

  unique_ptr<ClientSession> session;

  Status build();
  Status build_loss();
  

  unsigned int input_size;
  unsigned int hidden_size;
  unsigned int output_size;

  float learning_rate;

  Input add_dense(string name, Scope scope, int input_size, int output_size, Input input);

public:
  MLP(Scope &scope, float learning_rate);
  Status initialize();
  Status trainStep(Tensor& x, Tensor& y, float& loss, int& correct);

};
#endif