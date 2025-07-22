import os
import json
import base64
import argparse
from mimetypes import guess_type
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

from openai import AzureOpenAI

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate videos using Azure vision-language models')
    parser.add_argument('--gpt_model', type=str, default="gpt-4o",
                      help='Model to use for evaluation (e.g., "gpt-4o", "gpt-o1")')
    parser.add_argument('--total_frames', type=int, default=8,
                      help='Number of frames to evaluate per video')
    parser.add_argument('--run_in_parallel', action='store_true',
                      help='Run evaluations in parallel (default: False)')
    return parser.parse_args()

run_in_parallel = False  # Set to True to run in parallel, False to run sequentially
debug_mode = False

# Parse command line arguments
args = parse_args()
gpt_model = args.gpt_model
total_frames = args.total_frames
run_in_parallel = args.run_in_parallel

frames_folder = "videos/sampled_frames"  # New path to sampled frames
prompts_json_path = "prompts-with-standard-and-index.json"  # New source of prompts

# Load all prompts from the single JSON file
with open(prompts_json_path, "r") as f:
    prompts_data = json.load(f)

# Build a map from prompt IDs to prompt data
prompt_id_to_data = {}
for section in prompts_data.values():
    if isinstance(section, dict):
        for sub_section in section.values():
            if isinstance(sub_section, list):
                # Each item in the list should have Prompt_index, Physics_index, and Detailed_index
                for prompt in sub_section:
                    if all(key in prompt for key in ['Prompt_index', 'Physics_index', 'Detailed_index']):
                        prompt_id_to_data[prompt['Prompt_index']] = prompt
                        prompt_id_to_data[prompt['Physics_index']] = prompt
                        prompt_id_to_data[prompt['Detailed_index']] = prompt
            elif isinstance(sub_section, dict):
                for prompt_list in sub_section.values():
                    if isinstance(prompt_list, list):
                        # Each item in the list should have Prompt_index, Physics_index, and Detailed_index
                        for prompt in prompt_list:
                            if all(key in prompt for key in ['Prompt_index', 'Physics_index', 'Detailed_index']):
                                prompt_id_to_data[prompt['Prompt_index']] = prompt
                                prompt_id_to_data[prompt['Physics_index']] = prompt
                                prompt_id_to_data[prompt['Detailed_index']] = prompt

print(f"Total prompts loaded: {len(prompt_id_to_data)}")

# Filter prompts to only include those whose folders exist
filtered_prompt_ids = []
for prompt_id in prompt_id_to_data.keys():
    folder_path = os.path.join(frames_folder, prompt_id)
    if os.path.exists(folder_path):
        filtered_prompt_ids.append(prompt_id)
    else:
        print(f"Missing folder for {prompt_id}: {folder_path}")

print(f"Prompts after filtering: {len(filtered_prompt_ids)}")
print("Prompts to process:")
for prompt_id in filtered_prompt_ids:
    print(f"- {prompt_id}")

is_two_steps_prompt = False
llm_prompt_type = "one_step"

if llm_prompt_type == "one_step":
    output_json_prefix = "one_step"
elif llm_prompt_type == "two_step_with_standard_first":
    output_json_prefix = "with_standard_first_step"
elif llm_prompt_type == "two_step_with_standard_last":
    output_json_prefix = "with_standard_last_step"
elif llm_prompt_type == "two_step_no_standard_first":
    output_json_prefix = "no_standard_first_step"
elif llm_prompt_type == "two_step_no_standard_last":
    output_json_prefix = "no_standard_last_step"
else:
    raise ValueError("Invalid llm_prompt_type")

if is_two_steps_prompt and llm_prompt_type == "one_step":
    raise ValueError("Invalid combination of is_two_steps_prompt and llm_prompt_type")

if gpt_model == "gpt-4o":
    api_base = "APIBASE"
    api_key= "APIKEY"
    deployment_name = 'DEPLOYMENTNAME'
    api_version = 'APIVERSION'
elif gpt_model == "gpt-o1":
    api_key="APIKEY"
    api_base = "APIBASE"
    deployment_name = 'DEPLOYMENTNAME'
    api_version = 'APIVERSION'
else:
    raise ValueError("Invalid model name")

client = AzureOpenAI(
    api_key=api_key,  
    api_version=api_version,
    base_url=f"{api_base}/openai/deployments/{deployment_name}"
)

def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    return f"data:{mime_type};base64,{base64_encoded_data}"

def call_gpt(prompt, image_paths=None):
    ms = [
        { "role": "user", 
         "content": [  
            { 
                "type": "text", 
                "text": prompt
            },
        ] 
        } 
    ]
    
    if image_paths:
        for image_path in image_paths:
            ms[0]['content'].append({
                "type": "image_url",
                "image_url": {
                    "url": local_image_to_data_url(image_path)
                }
            })

    if debug_mode:
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=ms
            )
            response_json = json.loads(response.json())
            return response_json['choices'][0]['message']['content']
        except Exception as e:
            print(f"Debug mode error: {str(e)}")
            return None
    else:
        attempt = 0
        max_retries = 3
        retry_delay = 2

        while attempt < max_retries:
            try:
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=ms
                )
                response_json = json.loads(response.json())
                content = response_json['choices'][0]['message']['content']
                if not content:
                    print(f"Warning: Empty response content received on attempt {attempt + 1}")
                return content
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                attempt += 1
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached, returning None")
                    return None

def get_gpt_description(object, event, physics_phenomenon_list, llm_prompt_type):
    if llm_prompt_type == "one_step":
        prompt = f"""Suppose you are an expert in judging and evaluating the quality of AI-generated videos. These are frames evenly sampled from a generated video from the begining to the end. This is a generated video from a video model rather than captured from real world, so the video could be low quality, such as fuzzy, inconsistency, especially not following real world physics. Compare the objects and quantities visually present in the video with the specified object(s): "{object}". Answer "Yes" if the object(s) could be found in the video, otherwise answer "No". Also, pleaes check if "{event}" is visually depicted in the video, and answer "Yes" or "No". Lastly, please check if video satisfies the standards in list: "{physics_phenomenon_list}", and answer "Yes" or "No" for each standard in the list.
        Return your evaluation in the following JSON format:
        "Objects": "Yes/No",
        "Event": "Yes/No",
        "Standard_1": "Yes/No",
        "Standard_2": "Yes/No",
        "...": "Yes/No"
        """
        return prompt
    elif llm_prompt_type == "two_step_with_standard_first":
        prompt = f"""Suppose you are an expert in judging and evaluating the quality of AI-generated videos. These are frames evenly sampled from a generated video from the begining to the end. This is a generated video from a video model rather than captured from real world, so the video could be low quality, such as fuzzy, inconsistency, especially not following real world physics. Please tell me what is in this video, including what happened and any physics phenomena you observe. Besides, I will also use your response to check if the video satisfies the following standards: {physics_phenomenon_list}, so please include information related to the the standards. Please be sure to include objects in the video, the main event, and any physics phenomena you observe."""
        return prompt
    elif llm_prompt_type == "two_step_with_standard_last":
        pass
    elif llm_prompt_type == "two_step_no_standard_first":
        prompt = f"""Suppose you are an expert in judging and evaluating the quality of AI-generated videos. These are frames evenly sampled from a generated video from the begining to the end. This is a generated video from a video model rather than captured from real world, so the video could be low quality, such as fuzzy, inconsistency, especially not following real world physics. Please tell me what is in this video, including what happened and any physics phenomena you observe. Please be sure to include objects in the video, the main event, and any physics phenomena you observe."""
        return prompt
    elif llm_prompt_type == "two_step_no_standard_last":
        pass
    else:
        raise ValueError("Invalid llm_prompt_type")

def process_single_prompt(prompt_id, prompt_data):
    # Join objects list into a string if needed
    objects = prompt_data['Basic_Standards']['Objects']
    if isinstance(objects, list):
        objects_str = ', '.join(objects)
    else:
        objects_str = str(objects)

    result_entry = {
        "data": prompt_data,
        "model_name": gpt_model,
        "is_two_steps_prompt": is_two_steps_prompt,
        "llm_prompt_type": llm_prompt_type,
        "total_frames": total_frames
    }

    llm_prompt = get_gpt_description(
        objects_str,
        prompt_data['Basic_Standards']['Event'],
        prompt_data['Key_Standards'],
        llm_prompt_type
    )

    # Look for frames in the prompt_id folder
    video_folder = os.path.join(frames_folder, prompt_id)
    if not os.path.exists(video_folder):
        print(f"Warning: Folder not found: {video_folder}")
        return None

    frame_files = sorted([f for f in os.listdir(video_folder) if f.startswith('frame_') and f.endswith('.png')])
    if len(frame_files) != total_frames:
        print(f"Warning: Expected {total_frames} frames in {video_folder}, found {len(frame_files)}")
        return None

    frame_paths = [os.path.join(video_folder, f) for f in frame_files]
    response = call_gpt(llm_prompt, frame_paths)
    if response is None:
        print(f"Warning: No response received for prompt {prompt_id}")
        result_entry["response"] = "Error: No response received"
    else:
        result_entry["response"] = response

    result_entry["llm_prompt"] = llm_prompt

    output_json_folder = os.path.join(
        "automatic_results", gpt_model, f"frame-{total_frames}",
        f"is_two_step_{is_two_steps_prompt}"
    )
    os.makedirs(output_json_folder, exist_ok=True)

    output_json_file = os.path.join(
        output_json_folder, f"{output_json_prefix}_automatic_result_{prompt_id}.json"
    )
    with open(output_json_file, "w") as outfile:
        json.dump(result_entry, outfile, indent=4)

    return output_json_file

def main():
    if not filtered_prompt_ids:
        print(f"No prompts found in {prompts_json_path}")
        return
    print(f"Found {len(filtered_prompt_ids)} prompts to process")
    print(f"Using model: {gpt_model}")
    print(f"Evaluating {total_frames} frames per video")
    
    if run_in_parallel:
        max_workers = min(100, len(filtered_prompt_ids))
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for prompt_id in filtered_prompt_ids:
                futures.append(executor.submit(process_single_prompt, prompt_id, prompt_id_to_data[prompt_id]))
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing prompts"):
                try:
                    result_path = future.result()
                    print(f"Saved result to: {result_path}")
                except Exception as ex:
                    print(f"Error processing a prompt: {ex}")
    else:
        for prompt_id in tqdm(filtered_prompt_ids, desc="Processing prompts"):
            try:
                result_path = process_single_prompt(prompt_id, prompt_id_to_data[prompt_id])
                print(f"Saved result to: {result_path}")
            except Exception as ex:
                print(f"Error processing a prompt: {ex}")

if __name__ == "__main__":
    main()