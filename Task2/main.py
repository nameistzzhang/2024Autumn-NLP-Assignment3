from concurrent.futures import ThreadPoolExecutor, as_completed
from deepseek import DeepSeek
from gsm8k import loadDataset
from gsm8k import extractFinalAnswer
import tqdm

# 初始化 DeepSeek 实例和数据集
deepseek = DeepSeek()
dataset = loadDataset()

# 定义 prompts
naive_prompt = "Here is a math question for you, directly output the answer: \nQ: "
cot_prompt = "Here is a math question for you, let's think step by step: \nQ: "
icl_prompt = ("Q: A car travels 60 miles in 2 hours. What is the speed?\nA: To find the speed, divide the distance by time. "
             "The speed is 60/2 = 30 miles per hour.\n#### 30\nHere is a math question for you, imitate the problem-solving process above and think step by step: \nQ: ")
refine_prompt1 = "Here is a math question for you, let's think step by step: \nQ: "
refine_prompt2_1 = "I think your previous answer may have errors, the question and the previous answer are below: \nQ: "
refine_prompt2_2 = "Refine your previous answer and output the answer in a new line after the string '#### '"

def process_data(data):
    """处理单条数据并返回结果"""
    question = data["question"]
    gt_answer = data["answer"]
    gt_final_answer = extractFinalAnswer(gt_answer)

    # 调用 deepseek API 获取答案
    # deepseek_answer = deepseek(
    #     "you are a helpful math assistant",
    #     icl_prompt + question + "\noutput the answer in a new line after the string '#### '"
    # )
    # deepseek_final_answer = extractFinalAnswer(deepseek_answer)
    
    first_deepseek_answer = deepseek(
        "you are a helpful math assistant",
        refine_prompt1 + question + "\noutput the answer in a new line after the string '#### '"
    )
    deepseek_answer = deepseek(
        "you are a helpful math assistant",
        refine_prompt2_1 + question + "\n" + "A: " + first_deepseek_answer + "\n" + refine_prompt2_2
    )
    deepseek_final_answer = extractFinalAnswer(deepseek_answer)

    # 返回结果
    return gt_final_answer, deepseek_final_answer

# 多线程处理
correct_count = 0
total_count = 0
results = []

with ThreadPoolExecutor() as executor:
    # 提交任务
    future_to_data = {executor.submit(process_data, data): data for data in dataset}

    # 使用 tqdm 显示进度条
    for future in tqdm.tqdm(as_completed(future_to_data), total=len(dataset)):
        try:
            gt_final_answer, deepseek_final_answer = future.result()
            if deepseek_final_answer == gt_final_answer:
                correct_count += 1
            total_count += 1
        except Exception as e:
            print(f"Error processing data: {e}")

# 输出准确率
print(f"Accuracy: {correct_count / total_count * 100:.2f}%")