# Run this program using:
# python module5_call.py

import logging
from module5_mod import NumberHandler

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def main():
    number_handler = NumberHandler()

    while True:
        try:
            count = int(input("Enter a positive integer (count): "))
            if count <= 0:
                raise ValueError
            break
        except ValueError:
            logger.warning("Please enter a valid positive integer.")

    logger.info(f"Enter {count} integers one by one:")
    for i in range(count):
        while True:
            try:
                input_value = int(input(f"Value {i + 1}: "))
                number_handler.insert_value(input_value)
                break
            except ValueError:
                logger.warning("Invalid input. Please enter an integer.")

    while True:
        try:
            target_value = int(input("Enter a value to search for: "))
            break
        except ValueError:
            logger.warning("Invalid input. Please enter an integer.")

    search_result = number_handler.search_value(target_value)
    logger.info(str(search_result))

if __name__ == "__main__":
    main()
