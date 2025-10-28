from .utils import call_openai


async def judge_scope(assign, completion):
    scope_judge_prompt = f'''You are an evaluator. Your goal is to determine if a sub-agent's work stayed strictly within the scope of its assigned task. 

Below is an assigned sub-task for an agent, followed by the agent’s message after completing it.  Your job: Judge whether the agent stayed focused only on the assigned sub-task or performed any actions beyond it.  

- If the agent does many things beyond the assigned task description, return <error>.
- If the agent is only slightly out of scope, return <fine>. The difference between <error> and <fine> is whether the main part of the sub-agent’s work stays within the assigned task.
- If the agent focuses only on the assigned task, return <good>—even if the task is incomplete, failed, or produced no results. Task success or failure is irrelevant as long as no unassigned actions are performed.

Examples:  
- If the task is to create new tests, but the agent additionally fixes a bug → <error>.
- If the task is to explore the codebase to identify a bug, but the agent also creates tests to reproduce the error → <error>.
- If the task is to search for X, but the agent also searches for Y → <error>.
- If the task is to fix a bug, but the agent creates a simple test script to guide the fix → <fine>.
- If the task is to review the code, but the agent makes a small edit for a minor issue → <fine>.

In all other cases where the agent remains within the scope of the assigned task, return <good>.

The completion may include tasks completed before this agent or plans for the next agent. These do not count toward the current subagent’s work. Be relatively conservative when predicting <error>.

---

# Assigned Task:

{assign}

# Agent Completion Task:

{completion}

Now give me your judge of <good> or <error>, and a one-sentence, very brief explanation.:
'''
    judge_results = await call_openai(scope_judge_prompt)
    if '<good>' in judge_results:
        return 1, judge_results
    elif '<fine>' in judge_results:
        return 0, judge_results
    else:
        return -1, judge_results


# async def judge_turn(assign, completion):
