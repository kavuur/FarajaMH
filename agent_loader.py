import yaml
from crewai import Agent, Task
from langchain_openai import ChatOpenAI
from helper import get_openai_api_key

def load_llm():
    """
    Load the LLM with the OpenAI key.
    """
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment!")
    return ChatOpenAI(
        temperature=0.0,
        # model="gpt-4o",
        model="ft:gpt-4o-mini-2024-07-18:aphrc-ai-api:domain-tuned:CIUzXaVc",
        openai_api_key=api_key
    )

def load_agents_from_yaml(yaml_path, llm):
    """
    Returns a dictionary of agents keyed by their id.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    agent_dict = {}
    for a in data.get('agents', []):
        agent_obj = Agent(
            name=a['id'],
            role=a.get('role', ''),
            goal=a.get('description', ''),
            backstory='',
            tools=[],
            llm=llm,
            verbose=True
        )
        agent_dict[a['id']] = agent_obj
    return agent_dict

def load_tasks_from_yaml(yaml_path, agent_dict):
    """
    Returns a list of Task objects.
    For each YAML task with multiple agents, creates one Task per agentpip install --upgrade langchain
.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    tasks = []
    for t in data.get('tasks', []):
        expected_output = t.get('expected_output', 'Provide a clear and detailed answer.')

        agent_ids = t.get('agents', [])
        if not agent_ids:
            # No agents listed
            task_obj = Task(
                name=t['name'],
                description=t['description'],
                expected_output=expected_output,
                agent=None
            )
            tasks.append(task_obj)
        else:
            for agent_id in agent_ids:
                if agent_id in agent_dict:
                    task_obj = Task(
                        name=f"{t['name']} - {agent_id}",
                        description=t['description'],
                        expected_output=expected_output,
                        agent=agent_dict[agent_id]
                    )
                    tasks.append(task_obj)
                else:
                    print(f"Warning: Agent id '{agent_id}' not found in agents.yaml")
    return tasks

