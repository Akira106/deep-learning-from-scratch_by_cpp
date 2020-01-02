#ifndef LAYER_H_
#define LAYER_H_
#include <Eigen/Core>
#include <map>
#include <string>
using namespace Eigen;
using namespace std;
class Layer{
  public:
    Layer(double learning_rate=0.001);
    Layer(const map<string,MatrixXd> &mat_map,const map<string,VectorXd> &vec_map,double learning_rate=0.001);
    virtual ~Layer() = default;

    virtual MatrixXd forward(const MatrixXd &X) = 0;
    virtual MatrixXd backward(const MatrixXd &dout) = 0;
    void update_SGD(void);

  protected:
    map<string,MatrixXd>mat_map;
    map<string,VectorXd>vec_map;
    map<string,MatrixXd>dmat_map;
    map<string,VectorXd>dvec_map;
    double learning_rate;
  
};

#endif
