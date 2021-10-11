#include "MLPModel.h"

#include <cmath>
#include <tensorflow/cc/framework/gradients.h>
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/errors.h>
#include <tensorflow/core/lib/core/stringpiece.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/io/path.h>
#include <tensorflow/core/lib/strings/str_util.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/util/command_line_flags.h>

MLP::MLP(Scope &scope, float learning_rate)
    : scope(scope.NewSubScope("mlp")), learning_rate(learning_rate) {
  input_size = 28 * 28;
  hidden_size = 256;
  output_size = 10;
  TF_CHECK_OK(build());
  TF_CHECK_OK(build_loss());
}

Input MLP::add_dense(string name, Scope scope, int input_size, int output_size,
                     Input input) {
  vars_map[name + "_W"] =
      ops::Variable(scope.WithOpName("W"), {input_size, output_size}, DT_FLOAT);

  vars_map[name + "_b"] =
      ops::Variable(scope.WithOpName("b"), {output_size}, DT_FLOAT);
  auto uniform_negone_to_one = ops::Subtract(
      scope,
      ops::Mul(scope,
               ops::RandomUniform(scope, {input_size, output_size}, DT_FLOAT),
               Input::Initializer(2.f)),
      Input::Initializer(1.f));
  auto xavier_vals =
      ops::Mul(scope, uniform_negone_to_one,
               Input::Initializer(sqrt(6.f) / sqrtf(input_size + output_size)));
  inits_map[name + "_W"] = ops::Assign(scope.WithOpName("W_init"),
                                       vars_map[name + "_W"], xavier_vals);
  inits_map[name + "_b"] =
      ops::Assign(scope.WithOpName("b_init"), vars_map[name + "_b"],
                  ops::ZerosLike(scope, vars_map[name + "_b"]));
  shapes_map[name + "_W"] = {input_size, output_size};
  shapes_map[name + "_b"] = {output_size};

  auto mm =
      ops::MatMul(scope.WithOpName("matmul"), input, vars_map[name + "_W"]);

  ops::EnsureShape(scope, mm, {-1, output_size});
  auto added_bias =
      ops::Add(scope.WithOpName("add_bias"), mm, vars_map[name + "_b"]);
  return added_bias;
}

Status MLP::build() {
  input_placeholder = ops::Placeholder(scope.WithOpName("input"), DT_UINT8);
  auto float_vals = ops::Cast(scope, input_placeholder, DT_FLOAT);
  auto scaled_vals = ops::Div(scope.WithOpName("scaled_input"), float_vals,
                              ops::Const(scope, 255.0f));
  auto dense1 = add_dense("dense1", scope.NewSubScope("dense1"), input_size,
                          hidden_size, scaled_vals);
  auto relu1 = ops::Relu(scope.WithOpName("relu1"), dense1);
  ops::EnsureShape(scope, relu1, {-1, hidden_size});
  auto dense2 = add_dense("dense2", scope.NewSubScope("dense2"), hidden_size,
                          output_size, relu1);
  output = ops::Softmax(scope.WithOpName("softmax"), dense2);
  prediction_tensor = ops::ArgMax(scope.WithOpName("prediction"), dense2, 1);
  return scope.status();
}

Status MLP::build_loss() {
  label_placeholder =
      ops::Placeholder(scope.WithOpName("label_placeholder"), DT_FLOAT);
  Scope loss_scope = scope.NewSubScope("loss");
  loss_tensor = ops::Mean(
      loss_scope.WithOpName("Loss"),
      ops::SquaredDifference(loss_scope, output, label_placeholder), {0, 1});
  TF_CHECK_OK(loss_scope.status());

  vector<TensorShape> shapes;

  for (auto &key_val : vars_map) {
    vars.push_back(key_val.second);
    shapes.push_back(shapes_map[key_val.first]);
  }

  vector<Output> grads;

  TF_CHECK_OK(AddSymbolicGradients(scope, {loss_tensor}, vars, &grads));

  for (size_t i = 0; i < vars.size(); i++) {
    auto m_var = ops::Variable(scope, shapes[i], DT_FLOAT);
    auto v_var = ops::Variable(scope, shapes[i], DT_FLOAT);

    optimizer_inits.push_back(
        ops::Assign(scope, m_var, Input::Initializer(0.0f, shapes[i])));
    optimizer_inits.push_back(
        ops::Assign(scope, v_var, Input::Initializer(0.0f, shapes[i])));
    auto adam =
        ops::ApplyAdam(scope, vars[i], m_var, v_var, .0f, .0f, learning_rate,
                       0.9f, 0.999f, 0.0000001f, {grads[i]});
    // auto adam = ops::ApplyGradientDescent(scope, vars[i], learning_rate,
    // {grads[i]});
    applyOptim.push_back(adam.operation);
  }
  return scope.status();
}

Status MLP::initialize() {
  if (!scope.ok())
    return scope.status();
  vector<Output> ops_to_run;
  for (pair<string, Output> key_val : inits_map)
    ops_to_run.push_back(key_val.second);
  for (Output optim_init : optimizer_inits) {
    ops_to_run.push_back(optim_init);
  }
  session = unique_ptr<ClientSession>(new ClientSession(scope));
  TF_CHECK_OK(session->Run(ops_to_run, nullptr));
  return Status::OK();
}
Status MLP::trainStep(Tensor &x, Tensor &y, float &loss, int &correct) {
  if (!scope.ok())
    return scope.status();
  vector<Tensor> out_tensors;

  ClientSession::FeedType inputs(
      {{input_placeholder, x}, {label_placeholder, y}});

  TF_CHECK_OK(session->Run(inputs, {loss_tensor, output, prediction_tensor},
                           applyOptim, &out_tensors));
  loss = out_tensors[0].scalar<float>()(0);

  auto pred_mat = out_tensors[2].vec<int64_t>();
  auto y_mat = y.matrix<float>();
  for (int i = 0; i < pred_mat.dimension(0); i++) {
    if (y_mat(i, pred_mat(i)) == 1.0f) {
      correct += 1;
    }
  }

  return Status::OK();
}
Status MLP::validationStep(Tensor &x, Tensor &y, float &loss, int &correct) {
  if (!scope.ok())
    return scope.status();
  vector<Tensor> out_tensors;

  ClientSession::FeedType inputs(
      {{input_placeholder, x}, {label_placeholder, y}});

  TF_CHECK_OK(session->Run(inputs, {loss_tensor, output, prediction_tensor}, &out_tensors));
  loss = out_tensors[0].scalar<float>()(0);

  auto pred_mat = out_tensors[2].vec<int64_t>();
  auto y_mat = y.matrix<float>();
  for (int i = 0; i < pred_mat.dimension(0); i++) {
    if (y_mat(i, pred_mat(i)) == 1.0f) {
      correct += 1;
    }
  }

  return Status::OK();
}