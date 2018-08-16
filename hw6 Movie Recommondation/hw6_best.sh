wget 'https://www.dropbox.com/s/vi3xc4u7iem2cf6/HW6_best_model.h5?dl=1'
mv HW6_best_model.h5\?dl\=1 HW6_best_model.h5
python best_predict.py $1 $2 $3 $4
