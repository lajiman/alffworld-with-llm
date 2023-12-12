import json
import random
import os
from logging import Logger
import yaml
from utils.llm import llm
from utils.logger import get_logger
import alfworld
import alfworld.agents.environment


def process_ob(ob): # align the format as the prompt
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2 :]
    return ob


def act_chat(observation: str, task_description: str, task_type: str, logger: Logger):
    #load prompt
    prompt_file = "Task_2_ReAct/alfworld_act.json"
    with open(prompt_file, "r") as f:
        d = json.load(f)

    system_prompt = [
        {
            "role": "system",
            "content": "Interact with a household to solve a task."
            + " Only reply with 'Action:' followed by the action to do."
            + " Do not apologize."  # necessary for ChatGPT (more apologetic than GPT-3)
            + " Follow the format of the three examples below.",
        }
    ]
    description_prompt = [
        {"role": "user", "content": f"DESCRIPTION {observation}"}
    ]
    task_prompt = [{"role": "user", "content": f"TASK: {task_description}"+
                    " When you see what you need, remember to take it from the place! After you have found your object, You can try to put your object in/on a place even the place is not empty!"}]

    intext_prompt_0 = [{"role": "user", "content": "First exanple:\n"+d[f"{task_type}_0"]}]
    intext_prompt_1 = [{"role": "user", "content": "Second exanple:\n"+d[f"{task_type}_1"]}]
    intext_prompt_2 = [{"role": "user", "content": "Third exanple:\n"+d[f"{task_type}_2"]}]

    chat_prompts = (
        system_prompt
        + intext_prompt_0
        + intext_prompt_1
        + intext_prompt_2
        + description_prompt
        + task_prompt
    )

    for i in range(1, 15):
        try:
            action = llm(chat_prompts)
            action = action.split('\n')[0]
            action = action.replace(".","")
        except Exception as e:
            logger.error(f"Error {str(e)}")
            return 0, i
        
        action = action.replace("Action:","").strip().lower()
        if "put " in action and (" in " in action or " on " in action):
            action = action.replace(" in ", " in/on ").replace(" on ", " in/on ")
        if "I apologize" in action or "Have a great day!" in action:
            logger.info(f"{i} Action: {action}")
            return 0, i  
        
        observation, reward, done, info = env.step([action])
        observation, reward, done = (
            process_ob(observation[0]),
            info["won"][0],
            done[0],
        )
        logger.info(f"{i} Action: {action}")
        logger.info(f"{i} Observation: {observation}")
        chat_prompts.append({"role": "assistant", "content": "Action: "+action})
        chat_prompts.append({"role": "user", "content": "Observation: "+observation})
        if done:
            return reward, i

    return 0, i



def react_chat(observation: str, task_description: str, task_type: str, logger: Logger):
    #load prompt
    prompt_file = "Task_2_ReAct/alfworld_react.json"
    with open(prompt_file, "r") as f:
        d = json.load(f)

    system_prompt = [
        {
            "role": "system",
            "content": "Interact with a household to solve a task."
            + " You can reply with 'Action:' followed by the action to do,"
            + " or you can reply with 'Think:' followed by the idea."
            + " Do not apologize."  # necessary for ChatGPT (more apologetic than GPT-3)
            + " Refer to the format of the three examples below.",
        }
    ]
    description_prompt = [
        {"role": "user", "content": f"DESCRIPTION {observation}"}
    ]
    task_prompt = [{"role": "user", "content": f"TASK: {task_description}"
                    + " When you see what you need, remember to take it from the place!"
                    + " After you have found your object, you can try to put your object in/on a place even the place is NOT EMPTY!"
                    + " When I ask you to go to a place, please check those place in DESCRIPTION."}]

    intext_prompt_0 = [{"role": "user", "content": "First exanple:\n"+d[f"{task_type}_0"]}]
    intext_prompt_1 = [{"role": "user", "content": "Second exanple:\n"+d[f"{task_type}_1"]}]
    intext_prompt_2 = [{"role": "user", "content": "Third exanple:\n"+d[f"{task_type}_2"]}]

    chat_prompts = (
        system_prompt
        + intext_prompt_0
        + intext_prompt_1
        + intext_prompt_2
        + description_prompt
        + task_prompt
    )

    # print(chat_prompts)

    think_count = 0
    for i in range(1, 50):
        try:
            action = llm(chat_prompts)
            action = action.split('\n')[0]
            action = action.replace(".","")
        except Exception as e:
            logger.error(f"Error {str(e)}")
            return 0, i
                
        if "Action:" in action:
            # print("hello"+action)
            action = action.replace("Action:","").strip().lower()
            if "put " in action and (" in " in action or " on " in action):
                action = action.replace(" in ", " in/on ").replace(" on ", " in/on ")
            if "I apologize" in action or "Have a great day!" in action:
                logger.info(f"{i} Action: {action}")
                return 0, i-think_count
            
            observation, reward, done, info = env.step([action])
            observation, reward, done = (
                process_ob(observation[0]),
                info["won"][0],
                done[0],
            )
            logger.info(f"{i} Action: {action}")
            logger.info(f"{i} Observation: {observation}")
            chat_prompts.append({"role": "assistant", "content": "Action: "+action})
            chat_prompts.append({"role": "user", "content": "Observation: "+observation})

        elif "Think:" in action:
            if "I apologize" in action or "Have a great day!" in action:
                logger.info(f"{i} Action: {action}")
                return 0, i-think_count
            observation = "Observation: OK."
            done=False
            think_count += 1
            
            logger.info(f"{i} {action}")
            logger.info(f"{i} {observation}")
            chat_prompts.append({"role": "assistant", "content": action})
            chat_prompts.append({"role": "user", "content": observation})

        if done:
            return reward, i-think_count

    return 0, i-think_count


if __name__ == "__main__":
    os.environ['ALFWORLD_DATA'] = "/home/laji_man/hw/nlp/assignment4/alfworld/data"

    with open('Task_2_ReAct/base_config.yaml') as reader:
        config = yaml.safe_load(reader)

    react = False
    if react:
        run_name = "output/react"
    else:
        run_name = "output/act"

    prefixes = ["pick_and_place","pick_clean_then_place","pick_heat_then_place","pick_cool_then_place","look_at_obj","pick_two_obj"]

    # in this task, we will use the eval_out_of_distribution data split
    split = "eval_out_of_distribution"
    env = alfworld.agents.environment.AlfredTWEnv(config, train_eval=split)
    env = env.init_env(batch_size=1)

    NUM_GAMEFILES = len(env.gamefiles)
    logger = get_logger(f"{run_name}.log")
    cnts = [0] * 6
    rs = [0] * 6
    results = []

    for n in range(NUM_GAMEFILES):
        # Set seed for reproducibility
        random.seed(config["general"]["random_seed"])

        ob, info = env.reset()
        ob = "\n".join(ob[0].split("\n\n")[1:])
        scene_observation, task_description = ob.split("\n")
        name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])

        logger.info(name)
        logger.info(scene_observation)
        logger.info(task_description)

        for i, task_type in enumerate(prefixes):
            if name.startswith(task_type):
                # use ReAct baseline
                if react:
                    r, length = react_chat(
                        scene_observation, task_description, task_type, logger
                    )
                # use Act baseline
                else:
                    r, length = act_chat(
                        scene_observation, task_description, task_type, logger
                    )

                rs[i] += r
                cnts[i] += 1
                results.append(
                    {
                        "task": task_type,
                        "success": r,
                        "length": length,
                    }
                )
                out_log = f"# {n + 1} r: {r} rs: {rs} cnts: {cnts} sum(rs) / sum(cnts): {sum(rs) / sum(cnts)}"
                logger.info(out_log)
        logger.info("------------\n")

        # save results
        with open(f"{run_name}.json", "w") as f:
            json.dump(results, f)
        
    assert len(results) == NUM_GAMEFILES, f"{len(results)} != {NUM_GAMEFILES}"