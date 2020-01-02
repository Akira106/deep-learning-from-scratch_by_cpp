#ifndef LAST_LAYER_SOFTMAXWITHLOSS_H_
#define LAST_LAYER_SOFTMAXWITHLOSS_H_
#include <Eigen/Core>
#include "last_layer.h"

using namespace Eigen;
using namespace std;

class Last_Layer_Softmaxwithloss:public Last_Layer{

  public:
    double forward(const MatrixXd &X,const MatrixXd &t) override;
    MatrixXd backward(const MatrixXd &t) override;

  private:
    MatrixXd softmax(const MatrixXd &X);
    double cross_entropy_error(const MatrixXd &t);

};

#endif


