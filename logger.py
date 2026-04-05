import json
import logging
import sys
from datetime import datetime
from config import LOG_FILE

# Configure logging format
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)

# Create logger
logger = logging.getLogger("ResearchAgent")
logger.setLevel(logging.INFO)

# File Handler (Traceability)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console Handler (To replace print)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def log_tool_call(tool_name, request_data, response_data):
    """Logs the details of a tool call for traceability."""
    trace_entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool_name,
        "request": request_data,
        "response": response_data
    }
    # Log the full trace to the file at INFO level
    logger.info(f"TOOL_CALL_TRACE: {json.dumps(trace_entry, indent=2)}")
