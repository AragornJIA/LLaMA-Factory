python ../llama.cpp/convert_hf_to_gguf.py --outfile ../trained_llm_models/$1/$1.gguf --outtype auto ../trained_llm_models/$1

quantization_type=$2

if [ "$quantization_type" -eq 2 ]; then
  suffix="Q4_0"
elif [ "$quantization_type" -eq 7 ]; then
  suffix="Q8_0"
else
  suffix="Q5_0"
fi

../llama.cpp/llama-quantize ../trained_llm_models/$1/$1.gguf ../trained_llm_models/$1/$1-$suffix.gguf $2