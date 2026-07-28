import re

log_path = r"C:\Users\91704\.gemini\antigravity-ide\brain\32c27f6a-84a5-4674-97eb-1b845457a7c2\.system_generated\tasks\task-110.log"

try:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    retrains = re.findall(r"\[CONTINUAL\] Retrained", content)
    errors = re.findall(r"IndexError", content)
    tracebacks = re.findall(r"Traceback", content)
    
    print(f"Retrains found: {len(retrains)}")
    print(f"Errors found: {len(errors)}")
    print(f"Tracebacks found: {len(tracebacks)}")
    
except Exception as e:
    print(f"Error: {e}")
