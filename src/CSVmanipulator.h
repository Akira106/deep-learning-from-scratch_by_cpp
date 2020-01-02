#ifndef CSVMANIPUALTOR_H_
#define CSVMANIPUALTOR_H_

/* csvファイルを扱うクラス */
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Core>
#include "myfunc.h"

using namespace std;
using namespace Eigen;


template <class dtype>
class CSVmanipulator{

  public:
    CSVmanipulator(const string &filename,char delimiter=',');
    CSVmanipulator(const vector<vector<dtype> > &values);
    void read(const string &filename,char delimiter=',');
    void write(const string &filename,char delimiter=',');
    vector<vector<dtype> > get_values(void);
    Matrix<dtype,Dynamic,Dynamic> get_values_eigen(void);

  private:
    vector<vector<dtype> > values;
    Matrix<dtype,Dynamic,Dynamic> values_eigen;
    void setMatrix(void);
};

template<class dtype>CSVmanipulator<dtype>::CSVmanipulator(const string &filename,char delimiter){

    this->read(filename,delimiter);
}

template<class dtype>CSVmanipulator<dtype>::CSVmanipulator(const vector<vector<dtype> > &values){

    this->values = values;
    this->setMatrix();
}

template<class dtype>void CSVmanipulator<dtype>::read(const string &filename,char delimiter){

    ifstream ifs(filename);
    if(ifs.fail()){
        myfunc::error(FLERR,filename+" doesn't exist");
    }
    string line;

    vector<dtype> inner;
    vector<vector<dtype> > values;

    while (getline(ifs, line)) {

        vector<string> strvec = myfunc::split(line, delimiter);
        inner.resize(0);
        for (int i=0; i<strvec.size();i++){
            inner.push_back(myfunc::str2num<dtype>(strvec[i]));
        }
        values.push_back(inner);
    }

    this->values = values;
    this->setMatrix();
}

template<class dtype>void CSVmanipulator<dtype>::write(const string &filename,char delimiter){

    ofstream ofs(filename);
    string line;
    for(int i=0;i<this->values.size();i++){
        line=myfunc::num2str<dtype>(this->values[i][0]); /*0行目だけ別実装(末尾に,がつくのを避けるため)*/
        for(int j=1;j<this->values[i].size();j++){
            line+=delimiter;
            line+=myfunc::num2str<dtype>(this->values[i][j]);
        }
        line+="\n";
        ofs<<line;
    }
    ofs.close();
}

template<class dtype> vector<vector<dtype> > CSVmanipulator<dtype>::get_values(void){
    return this->values;
}

template<class dtype> Matrix<dtype,Dynamic,Dynamic> CSVmanipulator<dtype>::get_values_eigen(void){
    return this->values_eigen;
}

/*vectorの値をMatrixに変換するだけ
  二次元vectorのままだと変換できないので、一次元を一度経由する*/
template<class dtype> void CSVmanipulator<dtype>::setMatrix(void){

    int nrow=this->values.size();
    int ncol=this->values[0].size();
    vector<dtype> values_vector(nrow*ncol);
    for(int i=0;i<nrow;i++){
        for(int j=0;j<ncol;j++){
            values_vector[i*ncol+j]=this->values[i][j];
        }
    }
    Matrix<dtype,Dynamic,Dynamic,RowMajor>values_eigen_rowmajor = Map<Matrix<dtype,Dynamic,Dynamic,RowMajor> >(values_vector.data(),nrow,ncol);
    this->values_eigen = values_eigen_rowmajor;
}
#endif
