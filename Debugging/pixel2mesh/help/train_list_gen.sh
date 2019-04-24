#!/bin/bash
# get all filename in specified path
# image_path=./kitti_test/

image_path=./Debugging/dataset/clips/0601/
# log_path=./prediction_info.txt
list_file=./pixel2mesh/utils/train_list_new.txt

# rm -f $log_path
# echo "" > $log_path

if test -s $list_file; then
    echo "Kitti list file exist"
else
    echo "Generaing kitti list..."
    files=$(ls $image_path)
    for filename in $files
    do
        echo $image_path$filename >> $list_file
    done
    echo "Done"
    
    # echo "prediction_info txt deleted"
fi


