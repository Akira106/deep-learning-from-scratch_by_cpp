#ifndef LAYER_RELU_H_
#define LAYER_RELU_H_
#include <Eigen/Core>
#include "layer.h"

using namespace Eigen;

class Layer_Relu:public Layer{
  public:
    MatrixXd forward(const MatrixXd &X) override;
    MatrixXd backward(const MatrixXd &dout) override;
  private:
    Matrix<bool,Dynamic,Dynamic> mask;

};

#endif

