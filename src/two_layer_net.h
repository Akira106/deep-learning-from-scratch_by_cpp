#ifndef TWO_LAYER_NET_H_
#define TWO_LAYER_NET_H_

#include <iostream>
#include <Eigen/Core>
#include <map>
#include <string>
#include <vector>
#include "layer.h"
#include "layer_affine.h"
#include "layer_relu.h"
#include "last_layer.h"
#include "last_layer_softmaxwithloss.h"

using namespace Eigen;
using namespace std;

class Two_layer_net{

  public:
    Two_layer_net(int input_size, int hidden_size,int output_size,double learning_rate=0.001, double weight_init_std=0.01, int seed=12345);
    ~Two_layer_net();

    MatrixXd predict(MatrixXd X);
    double loss(const MatrixXd &X, const MatrixXd &t);
    void gradient(const MatrixXd &X, const MatrixXd &t);
    void update(void);
    double accuracy(const MatrixXd &X, const MatrixXd &t);

    double loss_value;

  private:
    MatrixXd W1;
    MatrixXd W2;
    VectorXd b1;
    VectorXd b2;

    map<string, MatrixXd> W1_map;
    map<string, MatrixXd> W2_map;
    map<string, VectorXd> b1_map;
    map<string, VectorXd> b2_map;

    vector<unique_ptr<Layer> > layers;
    unique_ptr<Last_Layer> last_layer;
    int input_size;
    int hidden_size;
    int output_size;
    double weight_init_std;
};
#endif
