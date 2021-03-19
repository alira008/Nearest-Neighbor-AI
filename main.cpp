#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <vector>
#include <unordered_set>
#include <limits>
#include <cmath>
#include <ctime>

using namespace std;

void print_menu(int (&choices)[2]);
void get_data(int choice, vector<vector<double> > &arr, int (&data_info)[2]);
void forward_selection_search(const vector<vector<double> > &data);
void backward_elimination_search(const vector<vector<double> > &data);
vector<double> relevant_features_data(const vector<double> &data, const unordered_set<unsigned> &set_features);
double euclidean_distance(const vector<vector<double> > a, const vector<vector<double> > b);
double leave_one_out_cross_validation(const vector<vector<double> > &data, const unordered_set<unsigned> &current_features, unsigned k, string alg="fss");
void print_set(const unordered_set<unsigned> &set);
void output_set_to_file(const unordered_set<unsigned> &set);

int main() {
	//	Timer variables
	clock_t start, end;
	double total_time = 0.0;
	//	Will store our user's choices from the menu
	//	choices[0] will be our data file choice
	//	choices[1] will be our algorithm choice
	int choices[2];
	vector<vector<double> > data;
	//	data_info[0] will hold number of features in data data_info[1] will hold number of instances
	int data_info[2];

	//	Print our menu and get the user's choices for data file and algorithm choice
	print_menu(choices);

	//	Get data from our data file
	get_data(choices[0], data, data_info);
	cout << "Number of features: " << data_info[0] << "\n" << "Number of instances: " << data_info[1] << "\n" << endl;

	//	Start timer
	start = clock();
	//	Call our search algorithm
	if(choices[1] == 1) {
		forward_selection_search(data);
	}
	else {
		backward_elimination_search(data);
	}
	end = clock();
	total_time = double(end-start)/(double)(CLOCKS_PER_SEC);
	cout << "Search took " << total_time << " seconds to find best combination of features." << endl;

}

void print_menu(int (&choices)[2]) {
	string algs[2] = { "Forward Selection Search", "Backward Elimination Search" };
	string alg_name;
	int file_choice = 0, alg_choice = 0;
	int ret[2];

	//	Ask user which test data we would like to try
	cout << "Which test data file would you like to try.\n"
			"\t1. Small test data file\n"
			"\t2. Large test data file\n"
			<< endl;
	cin >> file_choice;

	//	Check to see if we were given the right input
	if(file_choice != 1 && file_choice != 2){
		cout << "Your choice can only be \"1\" or \"2\"...exiting";
		exit(0);
	}

	//	Ask user which test data we would like to try
	cout << "Which search algorithm would you like to use?\n"
			"\t1. Forward Selection\n"
			"\t2. Backward Elimination\n"
			<< endl;
	cin >> alg_choice;

	//	Check to see if we were given the right input
	if(alg_choice != 1 && alg_choice != 2){
		cout << "Your choice can only be \"1\" or \"2\"...exiting";
		exit(0);
	}

	choices[0] = file_choice;
	choices[1] = alg_choice;

	//	Output filename to file
	alg_name = algs[alg_choice-1];
	ofstream accuracy_file("./large_file_accuracy_cpp.txt", ios_base::app);
	accuracy_file << alg_name << endl;
	accuracy_file.close();

}

void get_data(int choice, vector<vector<double> > &arr, int (&data_info)[2]) {
	string files[2] = { "CS170_SMALLtestdata__20.txt", "CS170_largetestdata__69.txt" };
	string filename = files[choice-1];
	ifstream data_file(filename);
	stringstream ss;
	string line;
	string str_num;
	double num = 0;

	//	Check if we opened file
	if(data_file.is_open()) {
		//	First look at file line by line
		while(getline(data_file, line)) {
			//	vector to hold our columns
			vector<double> cols;
			//	Parse the numbers from each line
			istringstream scientific_string(line);
			while(scientific_string >> num) {
				//	Place our numbers in our vector of columns
				cols.push_back(num);
			}
			//	Place our vector of columns into our vector of vectors
			arr.push_back(cols);
		}	

		//	close data file
		data_file.close();
	}
	else {
		cout << "Error opening file...exiting" << endl;
		exit(0);
	}
	
	//	Number of features
	data_info[0] = arr[0].size() - 1;
	//	Number of instances
	data_info[1] = arr.size();

	//	Output filename to file
	ofstream accuracy_file("./large_file_accuracy_cpp.txt", ios_base::app);
	accuracy_file << filename << endl;
	accuracy_file.close();
}

void forward_selection_search(const vector<vector<double> > &data) {
	//	Size of Second dimension in the 2D array
	unsigned size = data[0].size();
	//	This set will have our current set of features
	unordered_set<unsigned> current_features;
	double best_accuracy_overall = 0.0;
	//	This set will have our best set of features
	unordered_set<unsigned> best_features_overall;
	//	Feature we would like to add to this level
	unsigned feature_to_add;
	//	This will track our best acurracy so far
	double best_accuracy;
	//	This will track our acurracy for each cross validation
	double accuracy;
	//	Iterator for unordered set
	unordered_set<unsigned>::const_iterator iter;

	//	Skip first column because the data is just the class label.	
	for(unsigned i = 1; i < size; i++) {

		feature_to_add = 0;
		best_accuracy = 0.0;

		cout << "On level " << i << " of the search tree" << endl;
		for(unsigned k = 1; k < size; k++) {
			//	Check if feature has not been added yet. Check if we should add this one
			iter = current_features.find(k);
			//	If iter is end, that means feature is not in the set
			if(iter == current_features.end()) {
				accuracy = leave_one_out_cross_validation(data, current_features, k);
				cout << "\tConsidering adding feature " << k << " with " << fixed << setprecision(3) << (accuracy*100) << "% accuracy." << endl;

				if(accuracy > best_accuracy) {
					best_accuracy = accuracy;
					feature_to_add = k;
				}
			}
		}

		current_features.insert(feature_to_add);
		cout << "\tAdded feature " << feature_to_add << endl;

		if(best_accuracy > best_accuracy_overall) {
			best_accuracy_overall = best_accuracy;
			best_features_overall = current_features;
		}
		else{
			cout << "**** Warning accuracy has decreased! Continuing search in case of local maxima. ****" << endl;
		}

		cout << "On level " << i << " , the best feature subset was ";
		print_set(current_features);
		cout << " . Accuracy was " << fixed << setprecision(3) << (best_accuracy*100) << "%\n" << endl;

		//	Output accuracy to file
		ofstream accuracy_file("./large_file_accuracy_cpp.txt", ios_base::app);
		accuracy_file << fixed << setprecision(3) << (best_accuracy*100) << "\n";
		accuracy_file.close();
	}

	cout << "Finished search! The best feature subset is ";
	print_set(best_features_overall);
	cout << " , which had an accuracy of " << fixed << setprecision(3) << (best_accuracy_overall*100) << "%" << endl;

	//	Output accuracy to file
	ofstream accuracy_file("./large_file_accuracy_cpp.txt", ios_base::app);
	accuracy_file << "Best set: ";
	accuracy_file.close();
	output_set_to_file(best_features_overall);
	accuracy_file.open("./large_file_accuracy_cpp.txt", ios_base::app);
	accuracy_file << " which had an accuracy of " << fixed << setprecision(3) << (best_accuracy_overall*100) << endl;
	accuracy_file.close();
}

void backward_elimination_search(const vector<vector<double> > &data) {
	//	Size of Second dimension in the 2D array
	unsigned size = data[0].size();
	//	This set will have our current set of features
	unordered_set<unsigned> current_features;
	double best_accuracy_overall = 0.0;
	//	This set will have our best set of features
	unordered_set<unsigned> best_features_overall;
	//	Feature we would like to add to this level
	unsigned feature_to_remove;
	//	This will track our best acurracy so far
	double best_accuracy;
	//	This will track our acurracy for each cross validation
	double accuracy;
	//	Iterator for unordered set
	unordered_set<unsigned>::const_iterator iter;

	//	Initiate current_features and best_features_overall sets to have all features
	for(unsigned i = 1; i < size; i++) {
		current_features.insert(i);
		best_features_overall.insert(i);
	}

	//	Skip first column because the data is just the class label.	
	for(unsigned i = 1; i < size; i++) {

		feature_to_remove = 0;
		best_accuracy = 0.0;

		cout << "On level " << i << " of the search tree" << endl;
		for(unsigned k = 1; k < size; k++) {
			//	Check if feature has been removed already
			iter = current_features.find(k);
			//	If iter is not end, that means feature is in the set
			if(iter != current_features.end()) {
				accuracy = leave_one_out_cross_validation(data, current_features, k, "bes");
				cout << "\tConsidering removing feature " << k << " with " << fixed << setprecision(3) << (accuracy*100) << "% accuracy." << endl;

				if(accuracy > best_accuracy) {
					best_accuracy = accuracy;
					feature_to_remove = k;
				}
			}
		}

		current_features.erase(feature_to_remove);
		cout << "\tRemoved feature " << feature_to_remove << endl;

		if(best_accuracy > best_accuracy_overall) {
			best_accuracy_overall = best_accuracy;
			best_features_overall = current_features;
		}
		else{
			cout << "**** Warning accuracy has decreased! Continuing search in case of local maxima. ****" << endl;
		}

		cout << "On level " << i << " , the best feature subset was ";
		print_set(current_features);
		cout << " . Accuracy was " << fixed << setprecision(3) << (best_accuracy*100) << "%\n" << endl;

		//	Output accuracy to file
		ofstream accuracy_file("./large_file_accuracy_cpp.txt", ios_base::app);
		accuracy_file << fixed << setprecision(3) << (best_accuracy*100) << "\n";
		accuracy_file.close();
	}

	cout << "Finished search! The best feature subset is ";
	print_set(best_features_overall);
	cout << " , which had an accuracy of " << fixed << setprecision(3) << (best_accuracy_overall*100) << "%" << endl;

	//	Output accuracy to file
	ofstream accuracy_file("./large_file_accuracy_cpp.txt", ios_base::app);
	accuracy_file << "Best set: ";
	accuracy_file.close();
	output_set_to_file(best_features_overall);
	accuracy_file.open("./large_file_accuracy_cpp.txt", ios_base::app);
	accuracy_file << " which had an accuracy of " << fixed << setprecision(3) << (best_accuracy_overall*100) << endl;
	accuracy_file.close();
}

vector<double> relevant_features_data(const vector<double> &data, const unordered_set<unsigned> &set_features) {
	vector<double> rel_features;
	unsigned feature;

	for(unordered_set<unsigned>::iterator iter = set_features.begin(); iter != set_features.end(); iter++) {
		feature = *iter;
		rel_features.push_back(data[feature]);
	}

	return rel_features;

}

double euclidean_distance(const vector<double> a, const vector<double> b) {
	double sum = 0.0;
	unsigned size = 0;
	double diff = 0.0;

	if(a.size() == b.size()) {
		size = a.size();
	}
	else {
		cout << "Error: Sizes of array are not the same...exiting" << endl;
		exit(0);
	}

	//	Calculate the difference and raise to power of 2
	for(unsigned i = 0; i < size; i++) {
		diff = a[i] - b[i];
		sum += diff*diff;
	}

	return sqrt(sum);

}

double leave_one_out_cross_validation(const vector<vector<double> > &data, const unordered_set<unsigned> &current_features, unsigned k, string alg) {
	//	Size of first dimension in the 2D array
	unsigned size = data.size();
	unsigned num_correctly_classified = 0;
	//	Copy our current_features and add the k feature we want to add into our test_features set
	unordered_set<unsigned> test_features = current_features;
	vector<double> object_to_classify;
	double label_object_to_classify = 0.0;
	//	Calculated distance
	double distance = 0.0;
	//	Nearest neighbor values
	vector<double> nn_object;
	double nn_dist = 0.0, nn_label = 0.0;
	unsigned nn_loc = 0;
	//	Accuracy we will be returning
	double accuracy = 0;

	if(alg == "fss"){
		test_features.insert(k);
	}
	else{
		test_features.erase(k);
	}

	for(unsigned i = 0; i < size; i++){
		object_to_classify = relevant_features_data(data[i], test_features);
		label_object_to_classify = data[i][0];

		//	Nearest neighbor distance and location will initially be infinity
		nn_dist = numeric_limits<double>::max();
		nn_loc = numeric_limits<double>::max();

		//	Try to find the nearest neighbor to our object to classify by checking every data point
		for(unsigned k = 0; k < size; k++) {
			//	We should skip over the data point that is our object to classify
			if(k != i) {
				nn_object = relevant_features_data(data[k], test_features);
				//	Euclidean distance
				distance = euclidean_distance(object_to_classify, nn_object);

				if(distance < nn_dist) {
					nn_dist = distance;
					nn_loc = k;
					nn_label = data[nn_loc][0];
				}
			}
		}

		//	If we classify objects correctly increment num_correctly_classified
		if(label_object_to_classify == nn_label) {
			num_correctly_classified++;
		}
	}

	accuracy = (double)num_correctly_classified/(double)size;

	return accuracy;

}

void print_set(const unordered_set<unsigned> &set) {
	unsigned feature = 0;

	cout << "{ ";
	for(unordered_set<unsigned>::iterator iter = set.begin(); iter != set.end(); iter++) {
		feature = *iter;
		cout << feature << " ";
	}
	cout << "}";

}

void output_set_to_file(const unordered_set<unsigned> &set) {
	unsigned feature = 0;
	ofstream file("./large_file_accuracy_cpp.txt", ios_base::app);

	file << "{ ";
	for(unordered_set<unsigned>::iterator iter = set.begin(); iter != set.end(); iter++) {
		feature = *iter;
		file << feature << " ";
	}
	file << "}";
	file.close();
}
