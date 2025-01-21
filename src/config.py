import logging

class Config:
    @staticmethod
    def get_logging_level_by_module(module_name):
        # Default to INFO level logging
        return logging.INFO

# Create a singleton instance
config = Config() 