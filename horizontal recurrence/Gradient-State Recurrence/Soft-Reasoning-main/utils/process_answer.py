import re
def get_cot_answer(answer):
    cot_answer = answer.split('Here are some examples:')[0].split('Question')[0].split('Final Answer:')[0].split('Please let me')[0].strip()
    if "The answer is:" in cot_answer and "Answer:" not in cot_answer:
        cot_answer = cot_answer.replace("The answer is:", "Answer:")
    if "The answer is:" in cot_answer and "Answer:" in cot_answer:
        cot_answer = cot_answer.split('The answer is:')[0].strip()
    return cot_answer

def get_clean_answer(args,cot_answer, allow_answer=False):
    if "Answer:" in cot_answer:
        if args.dataset == "strategyqa":
            split_answer = cot_answer.split('Answer:')[1].strip()
            for remove_char in [',', '$', '%', 'g', '.']:
                split_answer = split_answer.replace(remove_char, '')
            split_answer = split_answer.strip()
            if "true" in split_answer.lower():
                return "true"
            elif "false" in split_answer.lower():
                return "false"
            if "yes" in split_answer.lower():
                return "true"
            elif "no" in split_answer.lower():
                return "false"
            return split_answer.lower()
        else:
            split_answer = cot_answer.split('Answer:')[1].strip()
            for remove_char in [',', '$', '%', 'g']:
                split_answer = split_answer.replace(remove_char, '')
            pattern = r'\d+/\d+|\d+\.\d+|\d+'
            numbers_in_string = re.findall(pattern, split_answer)
            if len(numbers_in_string) >0:
                return numbers_in_string[0]
    elif allow_answer and "answer" in cot_answer:
        if args.dataset == "strategyqa":
            split_answer = cot_answer.split('answer')[1].strip()
            for remove_char in [',', '$', '%', 'g', '.']:
                split_answer = split_answer.replace(remove_char, '')
            split_answer = split_answer.strip()
            if "true" in split_answer.lower():
                return "true"
            elif "false" in split_answer.lower():
                return "false"
            if "yes" in split_answer.lower():
                return "true"
            elif "no" in split_answer.lower():
                return "false"
            return split_answer.lower()
        else:
            split_answer = cot_answer.split('answer')[1].strip()
            for remove_char in [',', '$', '%', 'g']:
                split_answer = split_answer.replace(remove_char, '')
            pattern = r'\d+/\d+|\d+\.\d+|\d+'
            numbers_in_string = re.findall(pattern, split_answer)
            if len(numbers_in_string) >0:
                return numbers_in_string[0]

    return None

def get_number(string):
    if re.match(r'.*\..*\.\d$', string) and string.endswith('.0'):
        string = string[:-2]
    # Updated regex pattern to handle negative numbers
    pattern = r'-?\d+/\d+|-?\d+\.\d+|-?\d+'
    numbers_in_string = re.findall(pattern, string)
    if len(numbers_in_string) > 0:
        num = numbers_in_string[0]
        if '/' in num:
            numerator, denominator = map(float, num.split('/'))
            number_in_string = numerator / denominator
        else:
            number_in_string = float(num)
        return number_in_string
    return None


def evaluate_answer(args, string, number_to_compare, tolerance=0.01):
    if string is None:
        return False
    if "gsm8k" in args.dataset or "svamp" in args.dataset or "gsm_hard" in args.dataset or "aime" in args.dataset:
        if isinstance(number_to_compare, str):
            number_to_compare = number_to_compare.strip()
            number_to_compare = number_to_compare.replace(',', '')
            number_to_compare = get_number(number_to_compare)
    elif "strategyqa" in args.dataset:
        if string.lower() == number_to_compare.lower():
            return True
        else:
            return False
    numbers_in_string = get_number(string)
    if numbers_in_string is not None and number_to_compare is not None:
        if abs(numbers_in_string - number_to_compare) <= tolerance:
            return True
    return False
