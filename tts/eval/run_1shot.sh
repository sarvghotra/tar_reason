if [ ! -f /tmp/ta_tok.pth ]; then
    cp /home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/ta_tok.pth /tmp/
fi

python tts/eval/1shot_gen.py \
    --model /home/mila/s/sarvjeet-singh.ghotra/scratch/git/tar_reason/output_dir/sft_slfreflect/checkpoint-240 \
    --out_dir tts/eval/images/1shot-sft_slfreflect_crit_240_geneval2/ \
    --prompts_file tts/eval/geneval2_prompts.jsonl \
    --generate_images \
    --ar_path /home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/ar_dtok_lp_256px.pth \
    --decoder_path /home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/vq_ds16_t2i.pt \
    --encoder_path /tmp/ta_tok.pth

# /home/mila/s/sarvjeet-singh.ghotra/scratch/git/tar_reason/output_dir/sft_slfreflect/checkpoint-240
# /home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/Tar-7B