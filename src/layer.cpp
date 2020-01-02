#include <iostream>
#include "layer.h"

using namespace Eigen;
using namespace std;

Layer::Layer(double learning_rate){
    this->learning_rate=learning_rate;
}

Layer::Layer(const map<string,MatrixXd> &mat_map,const map<string,VectorXd> &vec_map,double learning_rate){
    this->learning_rate=learning_rate;
    this->mat_map = mat_map;
    this->vec_map = vec_map;

    int nrow,ncol;
    for(auto itr=this->mat_map.begin();itr!=this->mat_map.end();++itr){
        nrow=itr->second.rows();
        ncol=itr->second.cols();
        this->dmat_map[itr->first] = MatrixXd::Zero(nrow,ncol);
    }
    for(auto itr=this->vec_map.begin();itr!=this->vec_map.end();++itr){
        nrow=itr->second.size();
        this->dvec_map[itr->first] = VectorXd::Zero(nrow);
    }

}

void Layer::update_SGD(void){

    int nrow,ncol;
    for(auto itr=this->mat_map.begin();itr!=this->mat_map.end();++itr){
        this->mat_map[itr->first] -= this->learning_rate*this->dmat_map[itr->first];
    }
    for(auto itr=this->vec_map.begin();itr!=this->vec_map.end();++itr){
        this->vec_map[itr->first] -= this->learning_rate*this->dvec_map[itr->first];
    }

}

