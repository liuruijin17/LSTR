root=../results/LSTR_CULANE
data_dir=../../../CULane/
exp=500000
split=testing
detect_dir=${root}/${exp}/${split}
w_lane=30;
iou=0.5;  # Set iou to 0.3 or 0.5
im_w=1640
im_h=590
frame=1
list=${data_dir}list/test.txt
out=${root}/${exp}_iou${iou}.txt
./evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out
