#!/bin/bash
# Quick fix script to update model names on Lightning.ai
# Run this in your part-2-code_old directory on Lightning.ai

echo "Fixing model identifier from google/t5-small-v1 to t5-small..."

# Fix train_t5.py
sed -i "s/'google\/t5-small-v1'/'t5-small'/g" train_t5.py
sed -i 's/"google\/t5-small-v1"/"t5-small"/g' train_t5.py

# Fix load_data.py
sed -i "s/'google\/t5-small-v1'/'t5-small'/g" load_data.py
sed -i 's/"google\/t5-small-v1"/"t5-small"/g' load_data.py

# Fix t5_utils.py
sed -i "s/'google\/t5-small-v1'/'t5-small'/g" t5_utils.py
sed -i 's/"google\/t5-small-v1"/"t5-small"/g' t5_utils.py

echo "Done! All model identifiers updated to 't5-small'"
echo "You can now run: python train_t5.py --finetune --learning_rate 3e-4 --weight_decay 0.01 --batch_size 16 --test_batch_size 32 --max_n_epochs 20 --patience_epochs 5 --scheduler_type linear --num_warmup_epochs 1 --experiment_name t5_ft_best"

