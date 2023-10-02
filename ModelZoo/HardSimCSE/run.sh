echo "====================== start nn_train ======================" && 
#python -u -B train.py $1 && 

echo "\n====================== start nn_eval ======================" &&
python -u -B eval.py $1
