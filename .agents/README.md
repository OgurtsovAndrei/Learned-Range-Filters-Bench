# AI Custom Prompts & Skills directory (.agents/)

Welcome! The `.agents/` directory is designed to house local, project-specific AI agent guidelines, task descriptions, custom prompters, and skill specifications. This keeps team workflows standardized and easily shareable.

---

## 1. Directory Structure

We recommend the following layout inside this folder:

```text
.agents/
├── README.md               # This handbook
├── prompts/                # Local prompter templates (e.g. system-prompt.txt)
└── skills/                 # Custom tool or automation definitions
    └── my-custom-skill/
        ├── SKILL.md        # Detailed instructions for the skill
        └── run-script.py   # Automation script used by the skill
```

---

## 2. Defining Custom Skills (SKILL.md)

A standard project-level AI skill is documented using a `SKILL.md` file. Here is the recommended template:

```markdown
---
name: name-of-your-skill
description: Brief description of what this skill accomplishes.
---

# Skill: Name of Your Skill

## Objective
A detailed breakdown of the task the agent should perform.

## Prerequisites
Dependencies (e.g. python packages, CGo builds) required to execute the skill.

## Detailed Steps
1. Step one of execution.
2. Step two of execution.

## Example Script / Command
```bash
python3 .agents/skills/my-custom-skill/run-script.py --args
\```
```

---

## 3. Best Practices for multi-agent cooperation
- **Check-in commits**: Custom tools or generators should perform atomic modifications and leave clean git working trees.
- **Verification checks**: Ensure that any automation run by an agent is wrapped with validation steps (e.g., executing compilation checks).
