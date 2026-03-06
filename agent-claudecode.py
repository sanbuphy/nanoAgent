import os
import json
import subprocess
import sys
import glob as glob_module
from datetime import datetime
from pathlib import Path
from typing import Any
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)

MEMORY_FILE = "agent_memory.md"
RULES_DIR = ".agent/rules"
SKILLS_DIR = ".agent/skills"
MCP_CONFIG = ".agent/mcp.json"

current_plan = []
plan_mode = False

base_tools = [
    {"type": "function", "function": {"name": "read", "description": "Read file with line numbers", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write", "description": "Write content to file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Replace string in file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "glob", "description": "Find files by pattern", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search files for pattern", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "plan", "description": "Break down complex task into steps and execute sequentially", "parameters": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]}}}
]

def read(path, offset=None, limit=None):
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        start = offset if offset else 0
        end = start + limit if limit else len(lines)
        numbered = [f"{i+1:4d} {line}" for i, line in enumerate(lines[start:end], start)]
        return ''.join(numbered)
    except Exception as e:
        return f"Error: {str(e)}"

def write(path, content):
    try:
        with open(path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error: {str(e)}"

def edit(path, old_string, new_string):
    try:
        with open(path, 'r') as f:
            content = f.read()
        if content.count(old_string) != 1:
            return f"Error: old_string must appear exactly once"
        new_content = content.replace(old_string, new_string)
        with open(path, 'w') as f:
            f.write(new_content)
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error: {str(e)}"

def glob(pattern):
    try:
        files = glob_module.glob(pattern, recursive=True)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return '\n'.join(files) if files else "No files found"
    except Exception as e:
        return f"Error: {str(e)}"

def grep(pattern, path="."):
    try:
        result = subprocess.run(f"grep -r '{pattern}' {path}", shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout if result.stdout else "No matches found"
    except Exception as e:
        return f"Error: {str(e)}"

def bash(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error: {str(e)}"

def plan(task):
    global current_plan, plan_mode
    if plan_mode:
        return "Error: Cannot plan within a plan"
    print(f"[Plan] Breaking down: {task}")
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "Break task into 3-5 steps. Return JSON with 'steps' array."},
            {"role": "user", "content": task}
        ],
        response_format={"type": "json_object"}
    )
    try:
        plan_data = json.loads(response.choices[0].message.content)
        steps = plan_data.get("steps", [task])
        current_plan = steps
        print(f"[Plan] Created {len(steps)} steps")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
        return f"Plan created with {len(steps)} steps. Executing now..."
    except:
        return "Error: Failed to create plan"

available_functions = {"read": read, "write": write, "edit": edit, "glob": glob, "grep": grep, "bash": bash, "plan": plan}

def parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    if not raw_arguments:
        return {}
    try:
        parsed = json.loads(raw_arguments)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError as error:
        return {"_argument_error": f"Invalid JSON arguments: {error}"}

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return ""
    try:
        with open(MEMORY_FILE, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            return '\n'.join(lines[-50:]) if len(lines) > 50 else content
    except:
        return ""

def save_memory(task, result):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n## {timestamp}\n**Task:** {task}\n**Result:** {result}\n"
    try:
        with open(MEMORY_FILE, 'a') as f:
            f.write(entry)
    except:
        pass

def load_rules():
    rules = []
    if not os.path.exists(RULES_DIR):
        return ""
    try:
        for rule_file in Path(RULES_DIR).glob("*.md"):
            with open(rule_file, 'r') as f:
                rules.append(f"# {rule_file.stem}\n{f.read()}")
        return "\n\n".join(rules) if rules else ""
    except:
        return ""

def load_skills():
    skills = []
    if not os.path.exists(SKILLS_DIR):
        return []
    try:
        for skill_file in Path(SKILLS_DIR).glob("*.json"):
            with open(skill_file, 'r') as f:
                skills.append(json.load(f))
        return skills
    except:
        return []

def load_mcp_tools():
    if not os.path.exists(MCP_CONFIG):
        return []
    try:
        with open(MCP_CONFIG, 'r') as f:
            config = json.load(f)
            mcp_tools = []
            for server_name, server_config in config.get("mcpServers", {}).items():
                if server_config.get("disabled", False):
                    continue
                for tool in server_config.get("tools", []):
                    mcp_tools.append({"type": "function", "function": tool})
            return mcp_tools
    except:
        return []

def run_agent_step(messages, tools, max_iterations=5):
    global current_plan, plan_mode
    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            tools=tools
        )
        message = response.choices[0].message
        messages.append(message)
        if not message.tool_calls:
            return message.content, messages
        for tool_call in message.tool_calls:
            function_payload = getattr(tool_call, "function", None)
            if function_payload is None:
                continue
            function_name = str(getattr(function_payload, "name", ""))
            raw_arguments = str(getattr(function_payload, "arguments", ""))
            function_args = parse_tool_arguments(raw_arguments)
            print(f"[Tool] {function_name}({function_args})")
            function_impl = available_functions.get(function_name)
            if "_argument_error" in function_args:
                function_response = f"Error: {function_args['_argument_error']}"
            elif function_name == "plan" and function_impl is not None:
                plan_mode = True
                function_response = function_impl(**function_args)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": function_response})
                if current_plan:
                    results = []
                    for i, step in enumerate(current_plan, 1):
                        print(f"\n[Step {i}/{len(current_plan)}] {step}")
                        messages.append({"role": "user", "content": step})
                        result, messages = run_agent_step(messages, [t for t in tools if t["function"]["name"] != "plan"])
                        results.append(result)
                        print(f"\n{result}")
                    plan_mode = False
                    current_plan = []
                    return "\n".join(results), messages
            elif function_impl is not None:
                function_response = function_impl(**function_args)
            else:
                function_response = f"Error: Unknown tool '{function_name}'"
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": function_response})
    return "Max iterations reached", messages

def run_agent_claudecode(task, use_plan=False):
    global plan_mode, current_plan
    print("[Init] Loading ClaudeCode features...")
    memory = load_memory()
    rules = load_rules()
    skills = load_skills()
    mcp_tools = load_mcp_tools()
    all_tools = base_tools + mcp_tools
    context_parts = ["You are a helpful assistant that can interact with the system. Be concise."]
    if rules:
        context_parts.append(f"\n# Rules\n{rules}")
        print(f"[Rules] Loaded {len(rules.split('# '))-1} rule files")
    if skills:
        context_parts.append(f"\n# Skills\n" + "\n".join([f"- {s['name']}: {s.get('description', '')}" for s in skills]))
        print(f"[Skills] Loaded {len(skills)} skills")
    if mcp_tools:
        print(f"[MCP] Loaded {len(mcp_tools)} MCP tools")
    if memory:
        context_parts.append(f"\n# Previous Context\n{memory}")
    messages = [{"role": "system", "content": "\n".join(context_parts)}]
    if use_plan:
        plan_mode = True
        plan(task)
        results = []
        for i, step in enumerate(current_plan, 1):
            print(f"\n[Step {i}/{len(current_plan)}] {step}")
            messages.append({"role": "user", "content": step})
            result, messages = run_agent_step(messages, [t for t in all_tools if t["function"]["name"] != "plan"])
            results.append(result)
            print(f"\n{result}")
        plan_mode = False
        current_plan = []
        final_result = "\n".join(results)
    else:
        messages.append({"role": "user", "content": task})
        final_result, messages = run_agent_step(messages, all_tools)
        print(f"\n{final_result}")
    save_memory(task, final_result)
    return final_result

if __name__ == "__main__":
    use_plan = "--plan" in sys.argv
    if use_plan:
        sys.argv.remove("--plan")
    if len(sys.argv) < 2:
        print("Usage: python agent-claudecode.py [--plan] 'your task'")
        print("  --plan: Enable task planning")
        print("\nFeatures: Memory, Rules, Skills, MCP, Plan tool")
        sys.exit(1)
    task = " ".join(sys.argv[1:])
    run_agent_claudecode(task, use_plan=use_plan)




