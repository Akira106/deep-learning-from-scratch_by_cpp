#include "layer_relu.h"
#include "myfunc.h"

using namespace Eigen;

MatrixXd Layer_Relu::forward(const MatrixXd &X){

    this->mask = X.array()<=0.0;
    MatrixXd out = X;
    myfunc::ChangeValueFromBool(out,this->mask,0.0);

    return out;
}

MatrixXd Layer_Relu::backward(const MatrixXd &dout){

    MatrixXd dx=dout;
    myfunc::ChangeValueFromBool(dx,this->mask,0.0);
    return dx;

}
