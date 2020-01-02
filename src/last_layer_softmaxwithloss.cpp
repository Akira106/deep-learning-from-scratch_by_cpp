#include "last_layer_softmaxwithloss.h"
#include "myfunc.h"

#include <iostream>
using namespace Eigen;
using namespace std;


double Last_Layer_Softmaxwithloss::forward(const MatrixXd &X, const MatrixXd &t){
    this->y = this->softmax(X);
    this->loss = this->cross_entropy_error(t);
    return this->loss;
}

MatrixXd Last_Layer_Softmaxwithloss::backward(const MatrixXd &t){
    MatrixXd dx = (this->y - t)/this->y.rows();
    return dx;
}


MatrixXd Last_Layer_Softmaxwithloss::softmax(const MatrixXd &X){
    MatrixXd Xshift = X.colwise()-X.rowwise().maxCoeff(); //オーバーフロー対策
    MatrixXd expXshift = Xshift.array().exp();
    VectorXd sum_expshift_inv = expXshift.rowwise().sum().array().inverse();
    return (expXshift.transpose()*sum_expshift_inv.asDiagonal()).transpose();
}

double Last_Layer_Softmaxwithloss::cross_entropy_error(const MatrixXd &t){
    return -1*(t.array()*(this->y.array().log().array())).sum()/this->y.rows();
}
