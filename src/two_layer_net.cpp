#include "two_layer_net.h"
#include "myfunc.h"
#include <random>

using namespace Eigen;
using namespace std;

Two_layer_net::Two_layer_net(int input_size, int hidden_size,int output_size,double learning_rate, double weight_init_std,int seed){
    this->input_size=input_size;
    this->hidden_size=hidden_size;
    this->output_size=output_size;
    this->weight_init_std=weight_init_std;
    this->loss_value=0.0;

    this->W1=MatrixXd(this->input_size,this->hidden_size);
    this->W2=MatrixXd(this->hidden_size,this->output_size);
    //乱数による初期化
    default_random_engine engine(seed);
    normal_distribution<> dist(0.0,this->weight_init_std);
    for(int i=0;i<this->input_size;i++){
        for(int j=0;j<this->hidden_size;j++){
            W1.coeffRef(i,j)=dist(engine);
        }
    }
    for(int i=0;i<this->hidden_size;i++){
        for(int j=0;j<this->output_size;j++){
            W2.coeffRef(i,j)=dist(engine);
        }
    }

    this->b1=VectorXd::Zero(this->hidden_size);
    this->b2=VectorXd::Zero(this->output_size);
    W1_map["W"] = W1;
    W2_map["W"] = W2;
    b1_map["b"] = b1;
    b2_map["b"] = b2;
    layers.push_back(unique_ptr<Layer>(new Layer_Affine(W1_map,b1_map,learning_rate)));
    layers.push_back(unique_ptr<Layer>(new Layer_Relu()));
    layers.push_back(unique_ptr<Layer>(new Layer_Affine(W2_map,b2_map,learning_rate)));
    last_layer = unique_ptr<Last_Layer>(new Last_Layer_Softmaxwithloss());
}

Two_layer_net::~Two_layer_net(void){
;
}

MatrixXd Two_layer_net::predict(MatrixXd X){
    for(int i=0;i<this->layers.size();i++){
        X=this->layers[i]->forward(X);
    }
    return X;
}

double Two_layer_net::loss(const MatrixXd &X, const MatrixXd &t){
    MatrixXd y = this->predict(X);
    return this->last_layer->forward(y,t);
}

void Two_layer_net::gradient(const MatrixXd &X, const MatrixXd &t){
    this->loss_value = this->loss(X,t);
    MatrixXd dout = this->last_layer->backward(t);
    for(int i=this->layers.size()-1;i>=0;i--){
        dout = this->layers[i]->backward(dout);
    }
}

void Two_layer_net::update(void){
    for(int i=0;i<this->layers.size();i++){
        this->layers[i]->update_SGD();
    }
}

double Two_layer_net::accuracy(const MatrixXd &X, const MatrixXd &t){

    MatrixXd y = this->predict(X);
    VectorXi argmax_y = myfunc::argmax<double>(y);
    int nsize=X.rows();
    double ncorrect=0;
    for(int i=0;i<nsize;i++){
        if(t.coeffRef(i,argmax_y.coeffRef(i))==1.0){
            ncorrect+=1;
        }
    }
    return ncorrect/nsize;

}
