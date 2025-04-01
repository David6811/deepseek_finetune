from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 加载已经微调好的模型
final_saved_path = "./final_saved_path"
model = AutoModelForCausalLM.from_pretrained(final_saved_path)
tokenizer = AutoTokenizer.from_pretrained(final_saved_path)

# 构建文本生成 pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "Tell me some singing skills?"

generated_texts = pipe(prompt, max_length=50, num_return_sequences=1)

print("开始回答:", generated_texts[0]["generated_text"])