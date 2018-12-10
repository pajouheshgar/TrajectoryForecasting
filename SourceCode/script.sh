mkdir Dataset/SDD/cache -p
mkdir Dataset/Mnist/cache -p
mkdir Mnist_Saved_Models -p
mkdir SDD_Saved_Models -p
tar -xvzf Dataset/Mnist/sequences.tar.gz
tar -xvzf Dataset/SDD/annotations.tar.gz

python SDD_Preprocess.py
