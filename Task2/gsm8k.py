from datasets import load_dataset

def loadDataset():
    dataset = load_dataset("gsm8k", "main")
    test_data = dataset["test"]
    return test_data

import re
def extractFinalAnswer(ans):
    # search for "#### " string the integral behind the string is the answer
    ans = re.search(r"#### (.+)", ans).group(1)
    return ans
