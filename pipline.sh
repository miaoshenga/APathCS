#
##------数据分割处理-------
#example
#total_size=16
#chunk_size=10
#github
total_size=329968
chunk_size=100000
#
##-----减去余数，除整+1-----
num=$(((total_size - total_size%chunk_size)/chunk_size+1))
# 打印数据
echo $num

#------处理注释描述-------
python scripts/descr_extract.py --data_name 'github' --chunk_size $chunk_size

#------定义数据路径------
lang_path=/data/hugang/DeveCode/mydata/PathCS/github/java


for  ((i=0; i<=$num; i++))
do
    echo $i
    java -jar cli-0.3.jar pathContexts --lang java --project $lang_path/code_files/train_$((chunk_size*(i))) --output $lang_path/code_path/train_$((i))
done

java -jar cli-0.3.jar pathContexts --lang java --project $lang_path/code_files/test --output $lang_path/code_path/test

##------先前合并2个------
python scripts/share_vocab.py --data_name  'github' --train_path code_path/train_0/java --test_path code_path/train_1/java --out_path code_path/train/java --merge_vocab True

#----- 再合并后面------
for ((i=2; i<$num; i++))
do
   python scripts/share_vocab.py --data_name 'github' --train_path code_path/train/java --test_path code_path/train_$i/java --out_path code_path/train/java  --merge_vocab True
done

python scripts/share_vocab.py --data_name 'github'  --train_path code_path/train/java --test_path code_path/test/java --out_path code_path/test/java

#
python scripts/data_process.py  --data_name  'github'

#------模型框架训练------
#python main.py

