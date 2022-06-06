# Implicit Event Argument Extraction with Argument-argument Relational Knowledge

The source code fo the manuscript entitled " Implicit Event Argument Extraction with Argument-argument Relational Knowledge" (TKDE-2021-12-1652)



## Quick Start

```
python teacher.py --help
python AREA.py --help

# Train and evaluate the teacher model with oracle knowledge
python teacher.py --model bert-base-uncased \
		  --output_dir ./RAMS_teacher_uncased_fenci
		  --train_file ../proc/data/RAMS-event/processed-data/RAMS_out_fenci/train_data_fenci.json \
		  --dev_file ../proc/data/RAMS-event/processed-data/RAMS_out_fenci/test_data_fenci.json \
		  --test_file ../proc/data/RAMS-event/processed-data/RAMS_out_fenci/test_data_fenci.json \
		  --gold_file ../proc/data/RAMS-event/processed-data/RAMS_out_fenci/test_data_fenci.json \
		  --do_train \
		  --do_eval \
		  --add_if_trigger_embedding
									
# Distill the teacher to student model with CKD and RKD
python AREA.py --student_model bert-base-uncased \
	       --model1 ./RAMS_teacher_uncased_fenci \
               --train_file ../proc/data/RAMS-event/processed-data/RAMS_out_fenci/train_data_fenci.json \
               --dev_file ../proc/data/RAMS-event/processed-data/RAMS_out_fenci/test_data_fenci.json \
               --test_file ../proc/data/RAMS-event/processed-data/RAMS_out_fenci/test_data_fenci.json \
               --gold_file ../proc/data/RAMS-event/processed-data/RAMS_out_fenci/test_data_fenci.json \
               --do_train \
               --do_eval \
               --add_if_trigger_embedding \
               --distance_metric RBF

```

### Note
- We modify the source code of pytorch_pretrained==0.6.1 in `/code/pytorch_pretrained_bert_self/`.

