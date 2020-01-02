#include <iostream>
#include <Eigen/Core>
#include "two_layer_net.h"
#include "myfunc.h"
#include "CSVmanipulator.h"

using namespace std;
using namespace Eigen;

int main(int argc, char** argv){

    if(argc<5){
        myfunc::error(FLERR,"USAGE:./main x_train.csv t_train.csv x_test.csv t_test.csv");
    }

    cout <<"Reading training data ..."<<endl;
    CSVmanipulator<double> X_train_manip(argv[1]);
    CSVmanipulator<double> t_train_manip(argv[2]);
    CSVmanipulator<double> X_test_manip(argv[3]);
    CSVmanipulator<double> t_test_manip(argv[4]);
    cout <<"done! "<<endl;

    MatrixXd X_train = X_train_manip.get_values_eigen();
    MatrixXd t_train = t_train_manip.get_values_eigen();
    MatrixXd X_test = X_test_manip.get_values_eigen();
    MatrixXd t_test = t_test_manip.get_values_eigen();

    int input_size = X_train.cols();
    int hidden_size = 50;
    int output_size = t_train.cols();
    double learning_rate = 0.1;
    double weight_init_std = 0.01;
    int seed = 12345;

    Two_layer_net network(input_size,hidden_size,output_size,learning_rate,weight_init_std,seed);

    int sample_size = X_train.rows();
    int batch_size = sample_size/10;
    int iters_num = 100;
    int plot_freq = iters_num/10;

    MatrixXd X_batch, t_batch;
    vector<vector<int> > batch_mask;
    double loss,train_acc,test_acc;
    vector<vector<double> > log_list;

    cout <<"training ..."<<endl;
    cout <<"loss, train_accuracy, test_accuracy"<<endl;
    for(int i=0;i<iters_num;i++){
        batch_mask = myfunc::random_choice(sample_size,batch_size);
        for(int j=0;j<batch_mask.size();j++){
            X_batch = X_train(batch_mask[j],all);
            t_batch = t_train(batch_mask[j],all);

            network.gradient(X_batch, t_batch);
            network.update();
        }
        loss = network.loss_value;
        train_acc = network.accuracy(X_train, t_train);
        test_acc = network.accuracy(X_test, t_test);
        log_list.push_back(vector<double>{loss,train_acc,test_acc});
        if(i%plot_freq==0){
            cout <<loss<<", "<<train_acc<<", "<<test_acc<<endl;
        }
    }

    CSVmanipulator<double> log_manip(log_list);
    log_manip.write("train.log");
    return 0;
}
