#include <stdexcept>
#include "layer_affine.h"
#include "myfunc.h"

using namespace Eigen;
using namespace std;

Layer_Affine::Layer_Affine(double learning_rate){
    this->check_params();
}

Layer_Affine::Layer_Affine(const map<string,MatrixXd> &mat_map,const map<string,VectorXd> &vec_map,double learning_rate):Layer(mat_map,vec_map,learning_rate){
    this->check_params();
}


MatrixXd Layer_Affine::forward(const MatrixXd &X){
    this->X = X;

    MatrixXd out = (this->X*this->mat_map["W"]).rowwise()+this->vec_map["b"].transpose();

    return out;
}

MatrixXd Layer_Affine::backward(const MatrixXd &dout){

    MatrixXd dx = dout*this->mat_map["W"].transpose();
    this->dmat_map["W"]=this->X.transpose()*dout;
    this->dvec_map["b"]=dout.colwise().sum();

    return dx;
}

void Layer_Affine::check_params(void){
    auto itr_map=this->mat_map.find("W");
    if(itr_map==this->mat_map.end()){
        myfunc::error(FLERR,"'W'parameter is not set");
    }

    auto itr_vec=this->vec_map.find("b");
    if(itr_vec==this->vec_map.end()){
        myfunc::error(FLERR,"'b'parameter is not set");
    }
}
