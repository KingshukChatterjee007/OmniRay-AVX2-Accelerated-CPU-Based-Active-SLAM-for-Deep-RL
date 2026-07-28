import re

log_path = r"C:\Users\91704\.gemini\antigravity-ide\brain\32c27f6a-84a5-4674-97eb-1b845457a7c2\.system_generated\tasks\task-1167.log"

try:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    starts = re.findall(r"🚀 \[[a-zA-Z0-9_]+\] seed=", content)
    successes = re.findall(r"✅ \[[a-zA-Z0-9_]+\] seed=\d+ completed in", content)
    failures = re.findall(r"❌ \[[a-zA-Z0-9_]+\] seed=", content)
    
    print(f"Starts: {len(starts)}")
    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")
    if starts:
        print(f"Last start: {starts[-1]}")
    if successes:
        print(f"Last success: {successes[-1]}")
    
    # Print the last 20 lines
    lines = content.splitlines()
    print("\nLast 20 lines:")
    for line in lines[-20:]:
        print(line)
        
except Exception as e:
    print(f"Error: {e}")
