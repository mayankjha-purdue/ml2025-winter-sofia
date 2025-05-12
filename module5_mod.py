# module5_mod.py

import logging

logger = logging.getLogger(__name__)

class NumberHandler:
    def __init__(self):
        self.values = []
        logger.debug("Initialized NumberHandler with empty list.")

    def insert_value(self, input_value):
        self.values.append(input_value)
        logger.debug(f"Inserted value: {input_value}")

    def search_value(self, target_value):
        try:
            index = self.values.index(target_value) + 1
            logger.debug(f"Found {target_value} at position {index}")
            return index
        except ValueError:
            logger.debug(f"{target_value} not found in the list.")
            return -1
