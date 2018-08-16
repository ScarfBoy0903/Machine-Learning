wget 'https://www.dropbox.com/s/dqj7okcigm7ssy2/best_model.h5?dl=1'
mv best_model.h5\?dl\=1 best_model.h5
wget 'https://www.dropbox.com/s/jdzydgv0z1j6ses/w2v_matrix.npy?dl=1'
mv w2v_matrix.npy\?dl\=1 w2v_matrix.npy 
python test.py $1 $2 $3
