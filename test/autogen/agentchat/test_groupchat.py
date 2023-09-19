from flaml import autogen
import json
import logging
import os
import shutil

def test_chat_manager():
    agent1 = autogen.ConversableAgent(
        "alice",
        max_consecutive_auto_reply=2,
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is alice sepaking.",
    )
    agent2 = autogen.ConversableAgent(
        "bob",
        max_consecutive_auto_reply=2,
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is bob speaking.",
    )
    groupchat = autogen.GroupChat(agents=[agent1, agent2], messages=[], max_round=2)
    group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=False)
    agent1.initiate_chat(group_chat_manager, message="hello")

    assert len(agent1.chat_messages[group_chat_manager]) == 2
    assert len(groupchat.messages) == 2

    group_chat_manager.reset()
    assert len(groupchat.messages) == 0
    agent1.reset()
    agent2.reset()
    agent2.initiate_chat(group_chat_manager, message="hello")
    assert len(groupchat.messages) == 2

def test_e2e():
    # model: chat, mode: roleplay
    test_group_chat('chat', 'roleplay')

    # model: gpt-4, mode: roleplay
    test_group_chat('gpt-4', 'roleplay')



    # model: chat, mode: naive
    test_group_chat('chat', 'naive')

    # model: chat, mode: role_play_original
    test_group_chat('chat', 'role_play_original')



    # model: gpt-4, mode: naive
    test_group_chat('gpt-4', 'naive')

    # model: gpt-4, mode: role_play_original
    test_group_chat('gpt-4', 'role_play_original')

    #two agent chat
    test_two_agent_chat('chat')
    test_two_agent_chat('gpt-4')

def test_group_chat(model, mode):
    work_dir = f".\\test\\autogen\\agentchat\\paper-{model}-{mode}"

    # clear work_dir

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.mkdir(work_dir)


    config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST.json",
    filter_dict={
        "model": [model],
    },
    )
    gpt4_config = {
    "seed": 42,  # change the seed for different trials
    "temperature": 0.1,
    "config_list": config_list_gpt4,
    "request_timeout": 360,
    }
    user_proxy = autogen.UserProxyAgent(
       name="Admin",
       system_message="you are Admin, a human user. You will reply [TERMINATE] if task get resolved.",
        max_consecutive_auto_reply=10,
       human_input_mode="ALWAYS",
    )
    engineer = autogen.AssistantAgent(
        name="Engineer",
        llm_config=gpt4_config,
        system_message='''Engineer, You write code to resolve given task. If code running fail, you rewrite code.
        Your reply should be in the form of:
        Part 1:
        ```sh 
        // shell script to install python package if needed
        ```
        Part 2:
        ```python
        // python code to resolve task
        ```''',
    )
    critic = autogen.AssistantAgent(
        name="Critic",
        system_message=f'''Critic, find the bug and ask engineer to fix it, you don't write code.''',
        llm_config=gpt4_config,
    )

    executor = autogen.AssistantAgent(
        name="Executor",
        system_message="Executor, you are python code executor, you run python code automatically. If no code is provided in previous message, you ask engineer to write code.",
        llm_config=False,
        default_auto_reply="no code provided, @engineer, please write code to resolve task.",
        code_execution_config={"last_n_messages": 3, "work_dir": work_dir },
    )
    groupchat = autogen.GroupChat(agents=[user_proxy, engineer, critic, executor], messages=[], max_round=50)
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config, mode=mode)

    # load tasks from tasks.txt
    with open(".\\test\\autogen\\agentchat\\tasks.txt", "r") as f:
        tasks = f.readlines()
    tasks = [task.strip() for task in tasks]

    def init_groupchat():
        groupchat.reset()
        manager.reset()
        for agent in groupchat.agents:
            agent.reset()

        # set inital message to groupchat
        groupchat.messages.append(
            {
                "name": user_proxy.name,
                "content": "Welcome to the group chat! Work together to resolve my task. I'll reply [TERMINATE] when converstion ended.",
                "role": "user",
            }
        )
        groupchat.messages.append(
            {
                "name": user_proxy.name,
                "content": f"critic, if code running fail, you ask engineer to rewrite code.",
                "role": "user",
            }
        )
        
        groupchat.messages.append(
            {
                "name": user_proxy.name,
                "content": f"engineer, you write python code step by step to resolve my task.",
                "role": "user",
            }
        )
        
        groupchat.messages.append(
            {
                "name": user_proxy.name,
                "content": f"executor, you run python code from enginner and report bug",
                "role": "user",
            }
        )

    # run the experiment
    for i, task in enumerate(tasks):
        init_groupchat()
        # split index from task
        i = task.split(":", 1)[0]
        task = task.split(":", 1)[1]
        logging.info(f"Task: {task}")
        print(f"Task {i}: {task}")
        prompt = f'''task: {task}'''
        try:
            user_proxy.initiate_chat(manager, clear_history=False, message=prompt)
            # save chat messages to output/chat_history_{model}_{i}.txt using utf-8 encoding
            with open(f"{work_dir}\\chat_history_{model}_{mode}_{i}.txt", "w", encoding='utf-8') as f:
                for message in groupchat.messages:
                    # write a seperator
                    f.write("-" * 20 + "\n")
                    f.write(f'''###
{message["name"]}
###''' + "\n")
                    f.write(message["content"] + "\n")
                    f.write("-" * 20 + "\n")
        except Exception as e:
            print("Something went wrong. Please try again.")

def test_two_agent_chat(model):
    config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST.json",
    filter_dict={
        "model": [model],
    },
    )
    gpt4_config = {
    "seed": 42,  # change the seed for different trials
    "temperature": 0,
    "config_list": config_list_gpt4,
    "request_timeout": 360,
    }

    work_dir = f".\\test\\autogen\\agentchat\\twoagent-{model}"

    # clear work_dir

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.mkdir(work_dir)



    with open(".\\test\\autogen\\agentchat\\tasks.txt", "r") as f:
        tasks = f.readlines()
    tasks = [task.strip() for task in tasks]

    # run the experiment
    for i, task in enumerate(tasks):
        # split index from task
        i = task.split(":", 1)[0]
        task = task.split(":", 1)[1]
        logging.info(f"Task: {task}")
        print(f"Task {i}: {task}")
        twoagent_chat(model, i, task,gpt4_config, work_dir)
    


def twoagent_chat(model, problem_id, problem, config_list, work_dir):    
    from flaml.autogen import AssistantAgent, UserProxyAgent
    autogen.ChatCompletion.start_logging()
    # config_list = autogen.config_list_from_models(key_file_path=KEY_LOC, model_list=["gpt-4"], exclude="aoai")
    # create an AssistantAgent instance named "assistant"
    assistant = AssistantAgent(
        name="assistant",
        llm_config=config_list,
    )
    # create a UserProxyAgent instance named "user"
    user = UserProxyAgent(
        name="user",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=10,
        code_execution_config={"last_n_messages": 3, "work_dir": work_dir},
    )

    user.initiate_chat(
    assistant,
    message=problem,
    )
    log = autogen.ChatCompletion.logged_history
    file_path = f"{work_dir}\\twoagent-{model}-" +str(problem_id) +".txt"
    with open(file_path, 'w', encoding='utf-8') as f:
        for message in log:
            messages = json.loads(message)
            for msg in messages:
                # write a seperator
                f.write("-" * 20 + "\n")
                f.write(f'''###
{msg["role"]}
###''' + "\n")
                f.write(msg["content"] + "\n")
                f.write("-" * 20 + "\n")

def test_plugin():
    # Give another Agent class ability to manage group chat
    agent1 = autogen.ConversableAgent(
        "alice",
        max_consecutive_auto_reply=2,
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is alice sepaking.",
    )
    agent2 = autogen.ConversableAgent(
        "bob",
        max_consecutive_auto_reply=2,
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is bob speaking.",
    )
    groupchat = autogen.GroupChat(agents=[agent1, agent2], messages=[], max_round=2)
    group_chat_manager = autogen.ConversableAgent(name="deputy_manager", llm_config=False)
    group_chat_manager.register_reply(
        autogen.Agent,
        reply_func=autogen.GroupChatManager.run_chat,
        config=groupchat,
        reset_config=autogen.GroupChat.reset,
    )
    agent1.initiate_chat(group_chat_manager, message="hello")

    assert len(agent1.chat_messages[group_chat_manager]) == 2
    assert len(groupchat.messages) == 2


if __name__ == "__main__":
    # test_broadcast()
    # test_chat_manager()
    test_plugin()
