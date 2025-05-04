# module4.py

def get_input_count():
    return int(input("How many numbers will you enter? "))

def collect_numbers(total):
    values = []
    for i in range(1, total + 1):
        num = int(input(f"Number {i}: "))
        values.append(num)
    return values

def get_search_target():
    return int(input("Enter the number you want to find: "))

def find_position(values, target):
    for idx, val in enumerate(values):
        if val == target:
            return idx + 1  # Convert to 1-based index
    return -1

def main():
    count = get_input_count()
    user_values = collect_numbers(count)
    target = get_search_target()
    result = find_position(user_values, target)
    print(result)

if __name__ == "__main__":
    main()
